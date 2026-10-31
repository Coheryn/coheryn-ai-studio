//! Bounded, non-mapped reads: file truncation produces an error rather than SIGBUS.
use crate::{Descriptor, Error, ErrorCode, Result};
use sha2::{Digest, Sha256};
use std::{
    fs::{File, Metadata},
    io::{Read, Seek, SeekFrom},
    path::Path,
    sync::{Arc, Mutex},
    time::SystemTime,
};

/// Encode a SHA-256 output with the lowercase, zero-padded manifest representation.
pub fn digest_hex(digest: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in digest {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 15)]));
    }
    output
}

pub fn sha256_file(path: impl AsRef<Path>) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(digest_hex(hasher.finalize().into()))
}

pub struct Storage {
    source: Source,
    bytes: u64,
    modified: Option<SystemTime>,
}
enum Source {
    File(Mutex<File>),
    // Two checked temporal kernels are exposed in source [channel,time,y,x]
    // order. This is a byte view, not dequantization or a merged model copy.
    TemporalPatch {
        source: Arc<Storage>,
        offsets: [u64; 2],
        spatial: u64,
        channels: u64,
        element_bytes: u64,
    },
}
impl Storage {
    pub(crate) fn temporal_patch(
        source: Arc<Storage>,
        offsets: [u64; 2],
        rows: u64,
        spatial: u64,
        channels: u64,
        element_bytes: u64,
    ) -> Result<Self> {
        let half = rows
            .checked_mul(spatial)
            .and_then(|v| v.checked_mul(channels))
            .and_then(|v| v.checked_mul(element_bytes))
            .filter(|n| *n > 0)
            .ok_or_else(|| Error::invalid("patch view size overflow"))?;
        if !matches!(element_bytes, 2 | 4)
            || offsets.iter().any(|offset| {
                offset
                    .checked_add(half)
                    .is_none_or(|end| end > source.bytes)
            })
        {
            return Err(Error::invalid("patch view outside validated storage"));
        }
        Ok(Self {
            source: Source::TemporalPatch {
                source,
                offsets,
                spatial,
                channels,
                element_bytes,
            },
            bytes: half
                .checked_mul(2)
                .ok_or_else(|| Error::invalid("patch view overflow"))?,
            modified: None,
        })
    }
    /// Takes ownership of an already range-validated tensor file. The format loader
    /// validates its header through this same handle before creating tensor views.
    pub(crate) fn from_tensor_file(file: File) -> Result<Self> {
        let meta = file.metadata()?;
        if !meta.is_file() {
            return Err(Error::invalid("tensor storage must be a regular file"));
        }
        Ok(Self {
            source: Source::File(Mutex::new(file)),
            bytes: meta.len(),
            modified: meta.modified().ok(),
        })
    }
    pub fn open(path: &Path, descriptor: &Descriptor) -> Result<Self> {
        Self::open_with_limits(path, descriptor, crate::ParseLimits::default())
    }
    pub(crate) fn open_with_limits(
        path: &Path,
        descriptor: &Descriptor,
        limits: crate::ParseLimits,
    ) -> Result<Self> {
        let file = File::open(path)?;
        let meta = file.metadata()?;
        if meta.len() != descriptor.file_bytes {
            return Err(Error::new(
                ErrorCode::ModelChanged,
                "model changed since inspection",
            ));
        }
        let current = crate::gguf::parse(
            std::io::BufReader::new(file.try_clone()?),
            meta.len(),
            limits,
        )?;
        if current.header_sha256 != descriptor.header_sha256
            || file.metadata()?.modified().ok() != meta.modified().ok()
        {
            return Err(Error::new(
                ErrorCode::ModelChanged,
                "model header changed since inspection",
            ));
        }
        Ok(Self {
            source: Source::File(Mutex::new(file)),
            bytes: meta.len(),
            modified: meta.modified().ok(),
        })
    }
    pub fn unchanged(&self) -> Result<()> {
        let file = match &self.source {
            Source::TemporalPatch { source, .. } => return source.unchanged(),
            Source::File(file) => file,
        };
        let m: Metadata = file
            .lock()
            .map_err(|_| Error::new(ErrorCode::Internal, "weight storage lock poisoned"))?
            .metadata()?;
        if m.len() != self.bytes || m.modified().ok() != self.modified {
            return Err(Error::new(
                ErrorCode::ModelChanged,
                "model file changed during execution",
            ));
        }
        Ok(())
    }
    pub fn read(&self, offset: u64, output: &mut [u8]) -> Result<()> {
        if offset
            .checked_add(output.len() as u64)
            .is_none_or(|end| end > self.bytes)
        {
            return Err(Error::invalid("weight read outside validated file"));
        }
        let file = match &self.source {
            Source::TemporalPatch {
                source,
                offsets,
                spatial,
                channels,
                element_bytes,
            } => {
                let segment = spatial * element_bytes;
                let mut cursor = offset;
                let mut target = output;
                while !target.is_empty() {
                    let index = cursor / segment;
                    let within = cursor % segment;
                    let row = index / (channels * 2);
                    let channel = (index % (channels * 2)) / 2;
                    let time = (index % 2) as usize;
                    let physical = offsets[time] + (row * channels + channel) * segment + within;
                    let count = target.len().min((segment - within) as usize);
                    source.read(physical, &mut target[..count])?;
                    target = &mut target[count..];
                    cursor += count as u64;
                }
                return Ok(());
            }
            Source::File(file) => file,
        };
        let mut file = file
            .lock()
            .map_err(|_| Error::new(ErrorCode::Internal, "weight storage lock poisoned"))?;
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(output)
            .map_err(|e| Error::new(ErrorCode::ModelChanged, format!("weight read failed: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temporal_patch_view_preserves_channel_time_order_and_fragmented_ranges() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("patches");
        let physical: Vec<u16> = (0..12).chain(100..112).collect();
        std::fs::write(
            &path,
            physical
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let source = Arc::new(Storage::from_tensor_file(File::open(&path).unwrap()).unwrap());
        let view = Storage::temporal_patch(source.clone(), [0, 24], 2, 2, 3, 2).unwrap();
        let expected: Vec<u16> = vec![
            0, 1, 100, 101, 2, 3, 102, 103, 4, 5, 104, 105, 6, 7, 106, 107, 8, 9, 108, 109, 10, 11,
            110, 111,
        ];
        let expected: Vec<u8> = expected.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut actual = vec![0; 48];
        view.read(0, &mut actual).unwrap();
        assert_eq!(actual, expected);
        let mut fragment = vec![0; 29];
        view.read(3, &mut fragment).unwrap();
        assert_eq!(fragment, expected[3..32]);
        assert!(view.read(40, &mut fragment).is_err());
        assert!(Storage::temporal_patch(source, [0, 25], 2, 2, 3, 2).is_err());
        File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(8)
            .unwrap();
        assert_eq!(view.unchanged().unwrap_err().code, ErrorCode::ModelChanged);
    }

    #[test]
    fn standard_sha256_vectors_and_file_fingerprints_remain_stable() {
        assert_eq!(
            digest_hex(Sha256::digest(b"").into()),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        let expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert_eq!(digest_hex(Sha256::digest(b"abc").into()), expected);
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("abc");
        std::fs::write(&path, b"abc").unwrap();
        assert_eq!(sha256_file(path).unwrap(), expected);
    }
}

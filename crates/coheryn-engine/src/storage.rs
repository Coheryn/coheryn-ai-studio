//! Bounded, non-mapped reads: file truncation produces an error rather than SIGBUS.
use crate::{Descriptor, Error, ErrorCode, Result};
use sha2::{Digest, Sha256};
use std::{
    fs::{File, Metadata},
    io::Read,
    path::Path,
    sync::Arc,
    time::SystemTime,
};

/// The owned handle has no cursor-relative readers after validation. Positional
/// reads can therefore share it without a Coheryn seek/read lock. Windows updates
/// the cursor internally, but every read supplies its own complete byte offset.
struct WeightFile {
    #[cfg(any(unix, windows))]
    file: File,
    #[cfg(not(any(unix, windows)))]
    file: std::sync::Mutex<File>,
}

impl WeightFile {
    fn new(file: File) -> Self {
        #[cfg(not(any(unix, windows)))]
        let file = std::sync::Mutex::new(file);
        Self { file }
    }

    fn metadata(&self) -> Result<Metadata> {
        #[cfg(not(any(unix, windows)))]
        let file = self
            .file
            .lock()
            .map_err(|_| Error::new(ErrorCode::Internal, "weight storage lock poisoned"))?;
        #[cfg(any(unix, windows))]
        let file = &self.file;
        Ok(file.metadata()?)
    }

    fn read(&self, offset: u64, output: &mut [u8]) -> Result<()> {
        #[cfg(unix)]
        let result = {
            use std::os::unix::fs::FileExt;
            self.file.read_exact_at(output, offset)
        };
        #[cfg(windows)]
        let result = {
            use std::os::windows::fs::FileExt;
            complete_read(output, offset, |buffer, position| {
                self.file.seek_read(buffer, position)
            })
        };
        #[cfg(not(any(unix, windows)))]
        let result = {
            use std::io::{Seek, SeekFrom};
            let mut file = self
                .file
                .lock()
                .map_err(|_| Error::new(ErrorCode::Internal, "weight storage lock poisoned"))?;
            file.seek(SeekFrom::Start(offset))?;
            file.read_exact(output)
        };
        result.map_err(|error| {
            Error::new(ErrorCode::ModelChanged, format!("weight read failed: {error}"))
        })
    }
}

// Windows exposes short positional reads rather than read_exact_at. Retry only
// interruptions, advance by bytes actually read, and reject EOF/offset overflow.
#[cfg(any(windows, test))]
fn complete_read(
    mut output: &mut [u8],
    mut offset: u64,
    mut read: impl FnMut(&mut [u8], u64) -> std::io::Result<usize>,
) -> std::io::Result<()> {
    use std::io::ErrorKind;
    while !output.is_empty() {
        match read(output, offset) {
            Ok(0) => return Err(ErrorKind::UnexpectedEof.into()),
            Ok(count) => {
                if count > output.len() {
                    return Err(ErrorKind::InvalidData.into());
                }
                let Some(next_offset) = offset.checked_add(count as u64) else {
                    return Err(ErrorKind::InvalidInput.into());
                };
                offset = next_offset;
                output = &mut output[count..];
            }
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

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
    File(WeightFile),
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
            source: Source::File(WeightFile::new(file)),
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
            source: Source::File(WeightFile::new(file)),
            bytes: meta.len(),
            modified: meta.modified().ok(),
        })
    }
    pub fn unchanged(&self) -> Result<()> {
        let file = match &self.source {
            Source::TemporalPatch { source, .. } => return source.unchanged(),
            Source::File(file) => file,
        };
        let m = file.metadata()?;
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
        file.read(offset, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_file_reads_keep_requested_offsets_under_worker_interleaving() {
        use std::sync::Barrier;

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("interleaved-weights");
        let bytes: Vec<u8> = (0..65549)
            .map(|index: usize| ((index * 197 + (index >> 8)) % 251) as u8)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let source = Storage::from_tensor_file(File::open(&path).unwrap()).unwrap();
        let barrier = Barrier::new(4);
        std::thread::scope(|scope| {
            for lane in 0..4 {
                let source = &source;
                let bytes = &bytes;
                let barrier = &barrier;
                scope.spawn(move || {
                    let mut buffer = [0; 257];
                    barrier.wait();
                    for round in 0..512 {
                        let count = (round * 31 + lane * 7) % (buffer.len() + 1);
                        let offset = (round * 997 + lane * 16411) % (bytes.len() - count + 1);
                        source.read(offset as u64, &mut buffer[..count]).unwrap();
                        assert_eq!(&buffer[..count], &bytes[offset..offset + count]);
                        if round % 16 == 0 {
                            source.unchanged().unwrap();
                            std::thread::yield_now();
                        }
                    }
                });
            }
        });
        source.unchanged().unwrap();
    }

    #[test]
    fn file_boundaries_overflow_and_truncation_remain_recoverable() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("bounded-weights");
        std::fs::write(&path, b"0123456789").unwrap();
        let source = Storage::from_tensor_file(File::open(&path).unwrap()).unwrap();
        let mut tail = [0; 3];
        source.read(7, &mut tail).unwrap();
        assert_eq!(&tail, b"789");
        source.read(10, &mut []).unwrap();
        for offset in [8, 10, 11, u64::MAX] {
            let mut output = [0x5a; 3];
            assert_eq!(
                source.read(offset, &mut output).unwrap_err().code,
                ErrorCode::InvalidGguf
            );
            assert_eq!(output, [0x5a; 3]);
        }
        assert!(source.read(11, &mut []).is_err());
        File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(3)
            .unwrap();
        assert_eq!(
            source.read(7, &mut tail).unwrap_err().code,
            ErrorCode::ModelChanged
        );
        assert_eq!(source.unchanged().unwrap_err().code, ErrorCode::ModelChanged);
    }

    #[test]
    fn short_positional_reads_retry_only_interruptions_and_advance_exactly() {
        use std::io::{Error as IoError, ErrorKind};

        let mut output = [0; 5];
        let mut offsets = Vec::new();
        let mut calls = 0;
        complete_read(&mut output, 41, |buffer, offset| {
            offsets.push(offset);
            calls += 1;
            if matches!(calls, 1 | 3) {
                return Err(IoError::from(ErrorKind::Interrupted));
            }
            let count = buffer.len().min(2);
            for (index, byte) in buffer[..count].iter_mut().enumerate() {
                *byte = (offset + index as u64) as u8;
            }
            Ok(count)
        })
        .unwrap();
        assert_eq!(output, [41, 42, 43, 44, 45]);
        assert_eq!(offsets, [41, 41, 43, 43, 45]);
        complete_read(&mut [], u64::MAX, |_, _| panic!("empty read reached I/O")).unwrap();
        for (offset, count, kind) in [
            (0, 0, ErrorKind::UnexpectedEof),
            (u64::MAX, 1, ErrorKind::InvalidInput),
            (0, 2, ErrorKind::InvalidData),
        ] {
            assert_eq!(
                complete_read(&mut [0], offset, |_, _| Ok(count))
                    .unwrap_err()
                    .kind(),
                kind
            );
        }
        let mut calls = 0;
        assert_eq!(
            complete_read(&mut [0; 3], 0, |_, _| {
                calls += 1;
                if calls == 1 {
                    Ok(1)
                } else {
                    Err(IoError::from(ErrorKind::PermissionDenied))
                }
            })
            .unwrap_err()
            .kind(),
            ErrorKind::PermissionDenied
        );
        assert_eq!(calls, 2);
    }

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

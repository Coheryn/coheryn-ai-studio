//! Bounded, non-mapped reads: file truncation produces an error rather than SIGBUS.
use crate::{Descriptor, Error, ErrorCode, Result};
use sha2::{Digest, Sha256};
use std::{
    fs::{File, Metadata},
    io::{Read, Seek, SeekFrom},
    path::Path,
    sync::Mutex,
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
    file: Mutex<File>,
    bytes: u64,
    modified: Option<SystemTime>,
}
impl Storage {
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
            file: Mutex::new(file),
            bytes: meta.len(),
            modified: meta.modified().ok(),
        })
    }
    pub fn unchanged(&self) -> Result<()> {
        let m: Metadata = self
            .file
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
        let mut file = self
            .file
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

//! Bounded, optional GPU primitives. No external inference runtime or model code.
//!
//! This crate does not yet qualify a model/backend combination. CPU builds do
//! not load Metal. The private runtime owns all pointers and GPU synchronization.
use std::{fmt, sync::Arc};

#[cfg(all(feature = "metal", target_os = "macos"))]
#[allow(unsafe_code)]
mod runtime;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use runtime::{Device, Matrix, Workspace};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    Unsupported,
    InvalidInput,
    ResourceExhausted,
    Cancelled,
    Internal,
}

#[derive(Debug)]
pub struct Error {
    pub kind: ErrorKind,
    pub message: String,
}
impl Error {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;

/// Implementations retain the caller's shared CPU/GPU admission reservation.
/// The runtime drops the GPU allocation before releasing its lease.
pub trait MemoryLease: Send + Sync {}
impl<T: Send + Sync> MemoryLease for T {}
pub trait MemoryBudget: Send + Sync {
    fn reserve(&self, bytes: u64) -> Result<Box<dyn MemoryLease>>;
}
pub type SharedBudget = Arc<dyn MemoryBudget>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum WeightType {
    F32 = 0,
    F16 = 1,
    Bf16 = 2,
}
impl WeightType {
    fn bytes(self) -> usize {
        if self == Self::F32 { 4 } else { 2 }
    }
}

/// Explicit safety ceilings for the first dense kernel; not a support matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixShape {
    pub width: usize,
    pub rows: usize,
    pub dtype: WeightType,
}
impl MatrixShape {
    pub fn weight_bytes(self) -> Result<usize> {
        if !(1..=32768).contains(&self.width) || !(1..=1_048_576).contains(&self.rows) {
            return Err(Error::new(ErrorKind::InvalidInput, "GPU matrix shape exceeds kernel bounds"));
        }
        self.width.checked_mul(self.rows).and_then(|n| n.checked_mul(self.dtype.bytes()))
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "GPU weight size overflow"))
    }
    pub fn workspace_bytes(self, batch: usize) -> Result<usize> {
        self.weight_bytes()?;
        if !(1..=32).contains(&batch) {
            return Err(Error::new(ErrorKind::InvalidInput, "GPU batch must be between 1 and 32"));
        }
        self.width.checked_add(self.rows).and_then(|n| n.checked_mul(batch))
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "GPU workspace size overflow"))
    }
}

/// CPU-only builds report unavailability explicitly; they never pretend to run GPU work.
pub fn compiled() -> bool {
    cfg!(all(feature = "metal", target_os = "macos"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_bound_every_shader_index_and_allocation() {
        let shape = MatrixShape { width: 32768, rows: 1_048_576, dtype: WeightType::F16 };
        assert_eq!(shape.weight_bytes().unwrap(), 68_719_476_736);
        assert_eq!(shape.workspace_bytes(32).unwrap(), 138_412_032);
        for (width, rows, batch) in [(0, 1, 1), (1, 0, 1), (32769, 1, 1), (1, 1_048_577, 1), (1, 1, 0), (1, 1, 33), (usize::MAX, usize::MAX, 1)] {
            assert!(MatrixShape { width, rows, dtype: WeightType::F32 }.workspace_bytes(batch).is_err());
        }
    }
}

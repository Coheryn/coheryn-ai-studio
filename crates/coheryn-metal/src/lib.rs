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

/// Explicit owned-kernel choice. The cooperative implementation requires its
/// own qualification; the established default remains the reference kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelProfile {
    Reference,
    SimdCompensated,
}
impl KernelProfile {
    pub fn label(self) -> &'static str {
        match self {
            Self::Reference => "reference-f32-compensated-v1",
            Self::SimdCompensated => "simd-f32-compensated-v1",
        }
    }
}

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
        Self {
            kind,
            message: message.into(),
        }
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
    Q4Zero = 2,
    Q5Zero = 6,
    Q8Zero = 8,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Bf16 = 30,
}
impl WeightType {
    fn layout(self) -> (usize, usize) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 | Self::Bf16 => (1, 2),
            Self::Q4Zero => (32, 18),
            Self::Q5Zero => (32, 22),
            Self::Q8Zero => (32, 34),
            Self::Q4K => (256, 144),
            Self::Q5K => (256, 176),
            Self::Q6K => (256, 210),
        }
    }
}
impl TryFrom<u32> for WeightType {
    type Error = Error;
    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4Zero),
            6 => Ok(Self::Q5Zero),
            8 => Ok(Self::Q8Zero),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            30 => Ok(Self::Bf16),
            _ => Err(Error::new(
                ErrorKind::Unsupported,
                "no Metal kernel for this tensor encoding",
            )),
        }
    }
}

/// Explicit safety ceilings for reference matrix kernels; not a support matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixShape {
    pub width: usize,
    pub rows: usize,
    pub dtype: WeightType,
}

/// Per-request activation capacity, independent of each layer's weight encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkspaceLimits {
    pub width: usize,
    pub rows: usize,
    pub batch: usize,
}
impl WorkspaceLimits {
    pub fn bytes(self) -> Result<usize> {
        MatrixShape {
            width: self.width,
            rows: self.rows,
            dtype: WeightType::F32,
        }
        .workspace_bytes(self.batch)
    }
}
impl MatrixShape {
    pub fn row_bytes(self) -> Result<usize> {
        if !(1..=32768).contains(&self.width) || !(1..=1_048_576).contains(&self.rows) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "GPU matrix shape exceeds kernel bounds",
            ));
        }
        let (block, bytes) = self.dtype.layout();
        if !self.width.is_multiple_of(block) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "GPU row is not quantization-block aligned",
            ));
        }
        (self.width / block)
            .checked_mul(bytes)
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "GPU row size overflow"))
    }
    pub fn weight_bytes(self) -> Result<usize> {
        self.row_bytes()?
            .checked_mul(self.rows)
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "GPU weight size overflow"))
    }
    pub fn workspace_bytes(self, batch: usize) -> Result<usize> {
        self.weight_bytes()?;
        if !(1..=32).contains(&batch) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "GPU batch must be between 1 and 32",
            ));
        }
        self.width
            .checked_add(self.rows)
            .and_then(|n| n.checked_mul(batch))
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
        let shape = MatrixShape {
            width: 32768,
            rows: 1_048_576,
            dtype: WeightType::F16,
        };
        assert_eq!(shape.weight_bytes().unwrap(), 68_719_476_736);
        assert_eq!(shape.workspace_bytes(32).unwrap(), 138_412_032);
        for (width, rows, batch) in [
            (0, 1, 1),
            (1, 0, 1),
            (32769, 1, 1),
            (1, 1_048_577, 1),
            (1, 1, 0),
            (1, 1, 33),
            (usize::MAX, usize::MAX, 1),
        ] {
            assert!(
                MatrixShape {
                    width,
                    rows,
                    dtype: WeightType::F32
                }
                .workspace_bytes(batch)
                .is_err()
            );
        }
    }

    #[test]
    fn packed_layouts_reject_partial_blocks_and_unknown_encodings() {
        for (encoding, block, bytes) in [
            (0, 1, 4),
            (1, 1, 2),
            (30, 1, 2),
            (2, 32, 18),
            (6, 32, 22),
            (8, 32, 34),
            (12, 256, 144),
            (13, 256, 176),
            (14, 256, 210),
        ] {
            let dtype = WeightType::try_from(encoding).unwrap();
            let shape = MatrixShape {
                width: 256,
                rows: 3,
                dtype,
            };
            assert_eq!(shape.row_bytes().unwrap(), 256 / block * bytes);
            assert_eq!(shape.weight_bytes().unwrap(), 3 * 256 / block * bytes);
            if block > 1 {
                assert!(
                    MatrixShape {
                        width: 255,
                        ..shape
                    }
                    .weight_bytes()
                    .is_err()
                );
            }
        }
        assert_eq!(
            WeightType::try_from(999).unwrap_err().kind,
            ErrorKind::Unsupported
        );
    }
}

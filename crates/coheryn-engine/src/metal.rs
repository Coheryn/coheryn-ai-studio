//! Optional GPU matrix operations under the same engine admission budget.
//! Architecture dispatch remains CPU until its GPU execution profile qualifies.
use crate::{
    Cancellation, Error, ErrorCode, Result, gguf::Tensor, memory::Budget, storage::Storage,
};
use coheryn_metal::{
    Device, Matrix, MatrixShape, MemoryBudget, MemoryLease, WeightType, Workspace,
};
pub use coheryn_metal::WorkspaceLimits;
use std::sync::Arc;

fn engine_error(error: coheryn_metal::Error) -> Error {
    use coheryn_metal::ErrorKind;
    let code = match error.kind {
        ErrorKind::Unsupported => ErrorCode::UnsupportedFeature,
        ErrorKind::InvalidInput => ErrorCode::InvalidRequest,
        ErrorKind::ResourceExhausted => ErrorCode::ResourceExhausted,
        ErrorKind::Cancelled => ErrorCode::Cancelled,
        ErrorKind::Internal => ErrorCode::Internal,
    };
    Error::new(code, error.message)
}

struct SharedEngineBudget(Budget);
impl MemoryBudget for SharedEngineBudget {
    fn reserve(&self, bytes: u64) -> coheryn_metal::Result<Box<dyn MemoryLease>> {
        self.0
            .reserve(bytes)
            .map(|lease| Box::new(lease) as Box<dyn MemoryLease>)
            .map_err(|error| {
                coheryn_metal::Error::new(
                    coheryn_metal::ErrorKind::ResourceExhausted,
                    error.to_string(),
                )
            })
    }
}

pub struct MetalCompute(Arc<Device>);
impl MetalCompute {
    pub fn new(budget: Budget) -> Result<Self> {
        Device::new(Arc::new(SharedEngineBudget(budget)))
            .map(Self)
            .map_err(engine_error)
    }
    pub fn name(&self) -> String {
        self.0.name()
    }
    /// Bind a validated dense/packed matrix without a full CPU heap copy.
    /// GGUF and Safetensors adapters share this canonical Tensor/Storage view.
    pub fn upload(
        &self,
        storage: Arc<Storage>,
        data: u64,
        tensor: &Tensor,
        cancel: &Cancellation,
    ) -> Result<MetalMatrix> {
        cancel.check()?;
        storage.unchanged()?;
        if tensor.dimensions.len() != 2 {
            return Err(Error::invalid("GPU matrix must have two dimensions"));
        }
        let dtype = WeightType::try_from(tensor.dtype).map_err(engine_error)?;
        let shape = MatrixShape {
            width: usize::try_from(tensor.dimensions[0])
                .map_err(|_| Error::invalid("GPU width overflow"))?,
            rows: usize::try_from(tensor.dimensions[1])
                .map_err(|_| Error::invalid("GPU rows overflow"))?,
            dtype,
        };
        let bytes = shape.weight_bytes().map_err(engine_error)? as u64;
        let offset = data
            .checked_add(tensor.offset)
            .ok_or_else(|| Error::invalid("GPU tensor offset overflow"))?;
        if tensor.bytes != bytes || offset.checked_add(bytes).is_none() {
            return Err(Error::invalid("GPU tensor byte range mismatch"));
        }
        // Preserve original storage errors instead of classifying changed files
        // as numeric failures. The platform reader is bounded and synchronous.
        let mut read_error = None;
        let result = self.0.upload(
            shape,
            |position, chunk| {
                if let Err(error) = storage
                    .unchanged()
                    .and_then(|()| storage.read(offset + position, chunk))
                {
                    let message = error.to_string();
                    read_error = Some(error);
                    return Err(coheryn_metal::Error::new(
                        coheryn_metal::ErrorKind::InvalidInput,
                        message,
                    ));
                }
                Ok(())
            },
            || cancel.check().is_err(),
        );
        if let Some(error) = read_error {
            return Err(error);
        }
        let raw = result.map_err(engine_error)?;
        storage.unchanged()?;
        Ok(MetalMatrix { raw, storage })
    }
    pub fn workspace(&self, matrix: &MetalMatrix, batch: usize) -> Result<MetalWorkspace> {
        self.0
            .workspace(matrix.raw.shape(), batch)
            .map(MetalWorkspace)
            .map_err(engine_error)
    }
    pub fn workspace_with_limits(&self, limits: WorkspaceLimits) -> Result<MetalWorkspace> {
        self.0
            .workspace_with_limits(limits)
            .map(MetalWorkspace)
            .map_err(engine_error)
    }
}

pub struct MetalMatrix {
    raw: Matrix,
    storage: Arc<Storage>,
}
pub struct MetalWorkspace(Workspace);
impl MetalWorkspace {
    /// Caller supplies the architecture's qualified activation precision; the
    /// GPU primitive never silently rounds or quantizes these F32 operands.
    pub fn multiply(
        &mut self,
        matrix: &MetalMatrix,
        input: &[f32],
        output: &mut [f32],
        cancel: &Cancellation,
    ) -> Result<()> {
        matrix.storage.unchanged()?;
        self.0
            .multiply(&matrix.raw, input, output, || cancel.check().is_err())
            .map_err(engine_error)?;
        matrix.storage.unchanged()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gpu_leases_share_cpu_admission_and_return_it_on_drop() {
        let budget = Budget::new(1024);
        let cpu = budget.reserve(512).unwrap();
        let shared = SharedEngineBudget(budget.clone());
        let gpu = shared.reserve(512).unwrap();
        assert!(shared.reserve(1).is_err());
        drop(gpu);
        let reclaimed = budget.reserve(512).unwrap();
        drop(reclaimed);
        drop(cpu);
        assert!(budget.reserve(1024).is_ok());
    }
}

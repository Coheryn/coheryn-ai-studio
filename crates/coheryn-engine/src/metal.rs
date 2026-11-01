//! Optional GPU matrix operations under the same engine admission budget.
//! Public default execution remains CPU; explicit experimental bindings opt in.
use crate::{
    Cancellation, Error, ErrorCode, Result, gguf::Tensor, memory::Budget, storage::Storage,
};
pub use coheryn_metal::WorkspaceLimits;
use coheryn_metal::{
    Device, Matrix, MatrixShape, MemoryBudget, MemoryLease, WeightType, Workspace,
};
use std::sync::{Arc, Mutex};

mod resident;
pub(crate) use resident::{Execution, Resident};
pub(crate) type MatrixSource = (Arc<Storage>, u64, Tensor);

#[derive(Debug, serde::Serialize)]
pub struct ModelInfo {
    pub device: String,
    pub matrices: usize,
    pub encoded_weight_bytes: u64,
    pub driver_weight_allocation_bytes: u64,
    pub request_workspace_bytes: u64,
    pub completed_model_dispatches: u64,
    pub qualification: &'static str,
}

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

struct ReservedBudget(Mutex<crate::memory::Reservation>);
impl MemoryBudget for ReservedBudget {
    fn reserve(&self, bytes: u64) -> coheryn_metal::Result<Box<dyn MemoryLease>> {
        self.0
            .lock()
            .map_err(|_| {
                coheryn_metal::Error::new(
                    coheryn_metal::ErrorKind::ResourceExhausted,
                    "GPU admission lock poisoned",
                )
            })?
            .split(bytes)
            .map(|lease| Box::new(lease) as Box<dyn MemoryLease>)
            .map_err(|error| {
                coheryn_metal::Error::new(
                    coheryn_metal::ErrorKind::ResourceExhausted,
                    error.to_string(),
                )
            })
    }
}

pub struct MetalCompute {
    device: Arc<Device>,
    budget: Budget,
}
impl MetalCompute {
    pub fn new(budget: Budget) -> Result<Self> {
        Device::new(Arc::new(SharedEngineBudget(budget.clone())))
            .map(|device| Self { device, budget })
            .map_err(engine_error)
    }
    pub fn name(&self) -> String {
        self.device.name()
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
        self.upload_in(storage, data, tensor, cancel, None)
    }
    fn upload_in(
        &self,
        storage: Arc<Storage>,
        data: u64,
        tensor: &Tensor,
        cancel: &Cancellation,
        admitted: Option<crate::memory::Reservation>,
    ) -> Result<MetalMatrix> {
        cancel.check()?;
        storage.unchanged()?;
        let shape = matrix_shape(tensor)?;
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
        let read = |position, chunk: &mut [u8]| {
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
        };
        let result = if let Some(admitted) = admitted {
            let allocator = self.allocator(admitted, bytes)?;
            self.device
                .upload_with_budget(shape, read, || cancel.check().is_err(), &allocator)
        } else {
            self.device.upload(shape, read, || cancel.check().is_err())
        };
        if let Some(error) = read_error {
            return Err(error);
        }
        let raw = result.map_err(engine_error)?;
        storage.unchanged()?;
        Ok(MetalMatrix { raw, storage })
    }
    pub fn workspace(&self, matrix: &MetalMatrix, batch: usize) -> Result<MetalWorkspace> {
        self.device
            .workspace(matrix.raw.shape(), batch)
            .map(MetalWorkspace)
            .map_err(engine_error)
    }
    pub fn workspace_with_limits(&self, limits: WorkspaceLimits) -> Result<MetalWorkspace> {
        self.device
            .workspace_with_limits(limits)
            .map(MetalWorkspace)
            .map_err(engine_error)
    }
    fn allocator(
        &self,
        admitted: crate::memory::Reservation,
        bytes: u64,
    ) -> Result<coheryn_metal::SharedBudget> {
        if admitted.bytes() != bytes || !admitted.covers(&self.budget, bytes) {
            return Err(Error::resource(
                "GPU allocation needs an exact reservation from its model budget",
            ));
        }
        Ok(Arc::new(ReservedBudget(Mutex::new(admitted))))
    }
    fn workspace_admitted(
        &self,
        limits: WorkspaceLimits,
        admitted: crate::memory::Reservation,
    ) -> Result<MetalWorkspace> {
        let allocator = self.allocator(admitted, limits.bytes().map_err(engine_error)? as u64)?;
        self.device
            .workspace_with_budget(limits, &allocator)
            .map(MetalWorkspace)
            .map_err(engine_error)
    }
}

fn matrix_shape(tensor: &Tensor) -> Result<MatrixShape> {
    if tensor.dimensions.len() != 2 {
        return Err(Error::invalid("GPU matrix must have two dimensions"));
    }
    let shape = MatrixShape {
        width: usize::try_from(tensor.dimensions[0])
            .map_err(|_| Error::invalid("GPU width overflow"))?,
        rows: usize::try_from(tensor.dimensions[1])
            .map_err(|_| Error::invalid("GPU rows overflow"))?,
        dtype: WeightType::try_from(tensor.dtype).map_err(engine_error)?,
    };
    if shape.weight_bytes().map_err(engine_error)? as u64 != tensor.bytes {
        return Err(Error::invalid("GPU tensor byte count mismatch"));
    }
    Ok(shape)
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
        self.multiply_cooperative(matrix, input, output, cancel, &mut None)
    }
    pub(crate) fn multiply_cooperative(
        &mut self,
        matrix: &MetalMatrix,
        input: &[f32],
        output: &mut [f32],
        cancel: &Cancellation,
        cooperate: &mut Option<&mut crate::multimodal::WorkYield<'_>>,
    ) -> Result<()> {
        matrix.storage.unchanged()?;
        let mut callback_error = None;
        let result = self.0.multiply_cooperative(
            &matrix.raw,
            input,
            output,
            || cancel.check().is_err(),
            &mut || {
                if let Some(callback) = cooperate.as_deref_mut() {
                    if let Err(error) = callback() {
                        let message = error.to_string();
                        callback_error = Some(error);
                        return Err(coheryn_metal::Error::new(
                            coheryn_metal::ErrorKind::Internal,
                            message,
                        ));
                    }
                }
                Ok(())
            },
        );
        if let Some(error) = callback_error {
            return Err(error);
        }
        result.map_err(engine_error)?;
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
    #[test]
    fn prepaid_gpu_splits_do_not_double_charge_or_release_live_buffers() {
        let budget = Budget::new(512);
        let allocator = ReservedBudget(Mutex::new(budget.reserve(512).unwrap()));
        let input = allocator.reserve(100).unwrap();
        let output = allocator.reserve(400).unwrap();
        assert_eq!(budget.stats().reserved, 512);
        assert!(allocator.reserve(13).is_err());
        assert_eq!(budget.stats().reserved, 512);
        drop(allocator);
        assert_eq!(budget.stats().reserved, 500);
        drop(input);
        assert_eq!(budget.stats().reserved, 400);
        drop(output);
        assert_eq!(budget.stats().reserved, 0);
    }
}

//! Shared immutable GPU weights and private, prepaid request activation buffers.
use super::{
    MetalCompute, MetalMatrix, MetalWorkspace, WorkspaceLimits, engine_error, matrix_shape,
};
use crate::{
    Cancellation, Error, ErrorCode, Result,
    gguf::Tensor,
    memory::{Budget, Reservation},
    storage::Storage,
};
use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Key {
    storage: usize,
    offset: u64,
    bytes: u64,
    dtype: u32,
    width: u64,
    rows: u64,
}
impl Key {
    fn new(storage: &Storage, data: u64, tensor: &Tensor) -> Result<Self> {
        matrix_shape(tensor)?;
        let offset = data
            .checked_add(tensor.offset)
            .ok_or_else(|| Error::invalid("GPU source offset overflow"))?;
        offset
            .checked_add(tensor.bytes)
            .ok_or_else(|| Error::invalid("GPU source range overflow"))?;
        Ok(Self {
            storage: std::ptr::from_ref(storage) as usize,
            offset,
            bytes: tensor.bytes,
            dtype: tensor.dtype,
            width: tensor.dimensions[0],
            rows: tensor.dimensions[1],
        })
    }
}

pub(crate) struct Resident {
    matrices: HashMap<Key, MetalMatrix>,
    compute: MetalCompute,
    limits: WorkspaceLimits,
    weight_bytes: u64,
    dispatches: AtomicU64,
}
impl Resident {
    pub(crate) fn load(
        sources: Vec<super::MatrixSource>,
        budget: Budget,
        cancel: &Cancellation,
        cpu_request_minimum: u64,
        kernel: super::KernelProfile,
    ) -> Result<Arc<Self>> {
        if sources.is_empty() || sources.len() > 4096 {
            return Err(Error::request(
                "GPU graph must have 1..=4096 matrix bindings",
            ));
        }
        let mut unique = HashMap::new();
        unique
            .try_reserve(sources.len())
            .map_err(|_| Error::resource("GPU binding allocation failed"))?;
        let mut limits = WorkspaceLimits {
            width: 1,
            rows: 1,
            batch: 1,
        };
        for (storage, data, tensor) in sources {
            cancel.check()?;
            storage.unchanged()?;
            let shape = matrix_shape(&tensor)?;
            limits.width = limits.width.max(shape.width);
            limits.rows = limits.rows.max(shape.rows);
            let key = Key::new(&storage, data, &tensor)?;
            // Exact tied aliases retain one allocation and one source handle.
            unique.entry(key).or_insert((storage, data, tensor));
        }
        let weight_bytes = unique.keys().try_fold(0u64, |sum, key| {
            sum.checked_add(key.bytes)
                .ok_or_else(|| Error::resource("GPU resident byte sum overflow"))
        })?;
        let mut admission = budget.reserve(weight_bytes)?;
        // Reject an impossible minimum request before compiling or uploading.
        let request_minimum = cpu_request_minimum
            .checked_add(limits.bytes().map_err(engine_error)? as u64)
            .ok_or_else(|| Error::resource("GPU minimum request overflow"))?;
        drop(budget.reserve(request_minimum)?);
        let compute = MetalCompute::with_kernel_profile(budget, kernel)?;
        let mut matrices = HashMap::new();
        matrices
            .try_reserve(unique.len())
            .map_err(|_| Error::resource("GPU matrix handles allocation failed"))?;
        for (key, (storage, data, tensor)) in unique {
            cancel.check()?;
            let lease = admission.split(tensor.bytes)?;
            let matrix = compute.upload_in(storage, data, &tensor, cancel, Some(lease))?;
            matrices.insert(key, matrix);
        }
        cancel.check()?;
        Ok(Arc::new(Self {
            matrices,
            compute,
            limits,
            weight_bytes,
            dispatches: AtomicU64::new(0),
        }))
    }
    pub(crate) fn session_bytes(&self) -> Result<u64> {
        self.limits
            .bytes()
            .map(|bytes| bytes as u64)
            .map_err(engine_error)
    }
    pub(crate) fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }
    pub(crate) fn info(&self) -> Result<super::ModelInfo> {
        Ok(super::ModelInfo {
            device: self.compute.name(),
            kernel_profile: self.compute.kernel_profile().label(),
            matrices: self.matrices.len(),
            encoded_weight_bytes: self.weight_bytes,
            driver_weight_allocation_bytes: self
                .matrices
                .values()
                .map(|matrix| matrix.raw.allocated_bytes())
                .sum(),
            request_workspace_bytes: self.session_bytes()?,
            completed_model_dispatches: self.dispatches.load(Ordering::Relaxed),
            qualification: "experimental-text-matrix-profile; model/backend qualification pending",
        })
    }
    pub(crate) fn begin(self: &Arc<Self>, reservation: &mut Reservation) -> Result<Execution> {
        let bytes = self.session_bytes()?;
        if !reservation.covers(&self.compute.budget, bytes) {
            return Err(Error::resource(
                "GPU session admission is foreign or too small",
            ));
        }
        let lease = reservation.split(bytes)?;
        let workspace = self.compute.workspace_admitted(self.limits, lease)?;
        Ok(Execution {
            resident: Arc::clone(self),
            workspace,
        })
    }
}

pub(crate) struct Execution {
    // Drop private GPU buffers before the shared model ownership.
    workspace: MetalWorkspace,
    resident: Arc<Resident>,
}
impl Execution {
    pub(crate) fn capacity_bytes(&self) -> usize {
        // Limits were checked before allocation and cannot change afterwards.
        self.resident
            .limits
            .bytes()
            .expect("validated GPU workspace geometry")
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn multiply(
        &mut self,
        storage: &Storage,
        data: u64,
        tensor: &Tensor,
        input: &[f32],
        output: &mut [f32],
        cancel: &Cancellation,
        cooperate: &mut Option<&mut crate::multimodal::WorkYield<'_>>,
    ) -> Result<()> {
        let key = Key::new(storage, data, tensor)?;
        let matrix = self.resident.matrices.get(&key).ok_or_else(|| {
            Error::new(
                ErrorCode::UnsupportedFeature,
                format!("tensor {} is not bound to this GPU graph", tensor.name),
            )
        })?;
        let before = self.workspace.0.completed_dispatches();
        let result = self
            .workspace
            .multiply_cooperative(matrix, input, output, cancel, cooperate);
        let completed = self.workspace.0.completed_dispatches() - before;
        let _ = self
            .resident
            .dispatches
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
                Some(old.saturating_add(completed))
            });
        result
    }
}

//! Audited FFI boundary. Buffers are private, shared-storage and never exposed as
//! pointers. CPU access happens before commit or after waitUntilCompleted. No
//! fallible operations or callbacks occur between commit and retirement.
use crate::{
    Error, ErrorKind, KernelProfile, MatrixShape, MemoryLease, Result, SharedBudget,
    WorkspaceLimits,
};
use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResource, MTLResourceOptions, MTLSize,
};
use std::{ffi::c_void, ptr::NonNull, sync::Arc};

const UPLOAD_CHUNK: usize = 256 << 10;
const PRODUCTS_PER_DISPATCH: usize = 4 << 20;

fn resource(message: &str) -> Error {
    Error::new(ErrorKind::ResourceExhausted, message)
}
fn check(cancel: &impl Fn() -> bool) -> Result<()> {
    if cancel() {
        Err(Error::new(ErrorKind::Cancelled, "GPU operation cancelled"))
    } else {
        Ok(())
    }
}

// Field order matters: release the Metal resource before its admission lease.
struct Buffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    _lease: Box<dyn MemoryLease>,
}

// Also drain on Rust unwinding after submission: driver-retained resources alone
// do not retain the engine admission lease associated with each buffer.
struct InFlight(Retained<ProtocolObject<dyn MTLCommandBuffer>>);
impl Drop for InFlight {
    fn drop(&mut self) {
        if matches!(
            self.0.status(),
            MTLCommandBufferStatus::Committed | MTLCommandBufferStatus::Scheduled
        ) {
            self.0.waitUntilCompleted();
        }
    }
}

pub struct Device {
    raw: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    dense: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    kernel: KernelProfile,
    simd_width: usize,
    budget: SharedBudget,
}
impl Device {
    /// Compile only Coheryn's embedded kernels, never shader source from models.
    /// Admission covers buffer requests; compiler/driver overhead is separately
    /// observed host memory and cannot be enforced by this allocation interface.
    pub fn new(budget: SharedBudget) -> Result<Arc<Self>> {
        Self::with_kernel_profile(budget, KernelProfile::Reference)
    }

    pub fn with_kernel_profile(budget: SharedBudget, kernel: KernelProfile) -> Result<Arc<Self>> {
        autoreleasepool(|_| {
            let raw = MTLCreateSystemDefaultDevice().ok_or_else(|| {
                Error::new(ErrorKind::Unsupported, "no Metal device is available")
            })?;
            // Shared-storage CPU/GPU coherency is the only implemented policy.
            if !raw.hasUnifiedMemory() {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "this Metal backend requires unified memory",
                ));
            }
            let queue = raw
                .newCommandQueueWithMaxCommandBufferCount(1)
                .ok_or_else(|| resource("Metal command queue creation failed"))?;
            let options = MTLCompileOptions::new();
            // Keep macOS 11 compatibility; mathMode requires newer OS releases.
            #[allow(deprecated)]
            options.setFastMathEnabled(false);
            let source = match kernel {
                KernelProfile::Reference => NSString::from_str(include_str!("dense.metal")),
                KernelProfile::SimdCompensated => NSString::from_str(&format!(
                    "{}\n{}",
                    include_str!("dense.metal"),
                    include_str!("simd.metal")
                )),
            };
            let library = raw
                .newLibraryWithSource_options_error(&source, Some(&options))
                .map_err(|e| {
                    Error::new(
                        ErrorKind::Internal,
                        format!("Metal kernel compilation failed: {e}"),
                    )
                })?;
            let name = match kernel {
                KernelProfile::Reference => "dense_reference",
                KernelProfile::SimdCompensated => "dense_simd_compensated",
            };
            let function = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| {
                    Error::new(ErrorKind::Internal, "embedded Metal function is missing")
                })?;
            let dense = raw
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| {
                    Error::new(
                        ErrorKind::Internal,
                        format!("Metal pipeline creation failed: {e}"),
                    )
                })?;
            let simd_width = dense.threadExecutionWidth();
            if kernel == KernelProfile::SimdCompensated
                && (!simd_width.is_power_of_two()
                    || simd_width > 128
                    || simd_width > dense.maxTotalThreadsPerThreadgroup())
            {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "cooperative Metal kernel requires a complete power-of-two SIMD group",
                ));
            }
            Ok(Arc::new(Self {
                raw,
                queue,
                dense,
                kernel,
                simd_width,
                budget,
            }))
        })
    }

    pub fn name(&self) -> String {
        self.raw.name().to_string()
    }
    pub fn kernel_profile(&self) -> KernelProfile {
        self.kernel
    }

    fn allocate(&self, bytes: usize, budget: &SharedBudget) -> Result<Buffer> {
        if bytes == 0 || bytes > self.raw.maxBufferLength() {
            return Err(resource("GPU buffer exceeds device allocation limit"));
        }
        // Admission happens before asking Metal to allocate. Failure drops the
        // lease immediately; no temporarily unaccounted live GPU buffer exists.
        let lease = budget.reserve(bytes as u64)?;
        let raw = self
            .raw
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| resource("Metal buffer allocation failed"))?;
        if raw.length() < bytes {
            return Err(Error::new(
                ErrorKind::Internal,
                "Metal returned an undersized buffer",
            ));
        }
        Ok(Buffer { raw, _lease: lease })
    }

    /// Upload canonical little-endian dense or packed rows through a bounded reader.
    /// Reader errors/cancellation discard the incomplete private buffer. The
    /// caller remains responsible for validating immutable model-file identity.
    pub fn upload(
        self: &Arc<Self>,
        shape: MatrixShape,
        read: impl FnMut(u64, &mut [u8]) -> Result<()>,
        cancel: impl Fn() -> bool,
    ) -> Result<Matrix> {
        self.upload_with_budget(shape, read, cancel, &self.budget)
    }

    /// Use an already admitted allocator while preserving this device's identity.
    pub fn upload_with_budget(
        self: &Arc<Self>,
        shape: MatrixShape,
        mut read: impl FnMut(u64, &mut [u8]) -> Result<()>,
        cancel: impl Fn() -> bool,
        budget: &SharedBudget,
    ) -> Result<Matrix> {
        check(&cancel)?;
        let bytes = shape.weight_bytes()?;
        let buffer = self.allocate(bytes, budget)?;
        for offset in (0..bytes).step_by(UPLOAD_CHUNK) {
            check(&cancel)?;
            let length = (bytes - offset).min(UPLOAD_CHUNK);
            // SAFETY: shared buffer is at least `bytes` long, uniquely owned
            // and has never been submitted to a GPU. Checked offset/length
            // stay in bounds. Initialize bytes before creating Rust references.
            let slice = unsafe {
                let pointer = buffer.raw.contents().as_ptr().cast::<u8>().add(offset);
                pointer.write_bytes(0, length);
                std::slice::from_raw_parts_mut(pointer, length)
            };
            read(offset as u64, slice)?;
        }
        check(&cancel)?;
        Ok(Matrix {
            buffer,
            device: Arc::clone(self),
            shape,
        })
    }

    /// Per-request reusable input/output storage; no shared mutable GPU scratch.
    pub fn workspace(self: &Arc<Self>, shape: MatrixShape, batch: usize) -> Result<Workspace> {
        shape.workspace_bytes(batch)?;
        self.workspace_with_limits(WorkspaceLimits {
            width: shape.width,
            rows: shape.rows,
            batch,
        })
    }

    /// Reuse one admitted activation allocation across validated layer shapes.
    pub fn workspace_with_limits(self: &Arc<Self>, limits: WorkspaceLimits) -> Result<Workspace> {
        self.workspace_with_budget(limits, &self.budget)
    }

    pub fn workspace_with_budget(
        self: &Arc<Self>,
        limits: WorkspaceLimits,
        budget: &SharedBudget,
    ) -> Result<Workspace> {
        limits.bytes()?;
        let input = self.allocate(limits.width * limits.batch * 4, budget)?;
        let output = self.allocate(limits.rows * limits.batch * 4, budget)?;
        Ok(Workspace {
            input,
            output,
            device: Arc::clone(self),
            limits,
            dispatches: 0,
        })
    }
}

pub struct Matrix {
    buffer: Buffer,
    device: Arc<Device>,
    shape: MatrixShape,
}
// SAFETY: upload finishes all CPU writes before publishing a Matrix. Its buffer
// is private and immutable thereafter; every owned shader binds it read-only.
// Matrix methods expose neither mutable bytes nor raw handles. Retained Metal
// resources may be released on another thread, the device/queue are Send+Sync,
// and the admission lease itself requires Send+Sync. Concurrent dispatches
// therefore share only immutable weights, never activation/output storage.
unsafe impl Send for Matrix {}
// SAFETY: the same immutable-resource invariant permits concurrent shared reads.
unsafe impl Sync for Matrix {}
impl Matrix {
    pub fn shape(&self) -> MatrixShape {
        self.shape
    }
    /// Driver allocation size is observational, distinct from requested bytes.
    pub fn allocated_bytes(&self) -> u64 {
        self.buffer.raw.allocatedSize() as u64
    }
}

pub struct Workspace {
    input: Buffer,
    output: Buffer,
    device: Arc<Device>,
    limits: WorkspaceLimits,
    dispatches: u64,
}
// SAFETY: private buffers can move between workers only while exclusively
// owned. multiply requires &mut self and retires all commands before returning
// or unwinding; an idle Workspace has no GPU access in flight. It is not Sync.
unsafe impl Send for Workspace {}
impl Workspace {
    pub fn completed_dispatches(&self) -> u64 {
        self.dispatches
    }
    pub fn allocated_bytes(&self) -> u64 {
        self.input.raw.allocatedSize() as u64 + self.output.raw.allocatedSize() as u64
    }

    /// Synchronous, bounded-dispatch matrix execution. Cancellation discards all
    /// partial GPU output and leaves caller output unchanged. Previously
    /// committed commands always retire before buffers or reservations release.
    pub fn multiply(
        &mut self,
        matrix: &Matrix,
        input: &[f32],
        output: &mut [f32],
        cancel: impl Fn() -> bool,
    ) -> Result<()> {
        self.multiply_cooperative(matrix, input, output, cancel, &mut || Ok(()))
    }

    pub fn multiply_cooperative(
        &mut self,
        matrix: &Matrix,
        input: &[f32],
        output: &mut [f32],
        cancel: impl Fn() -> bool,
        cooperate: &mut impl FnMut() -> Result<()>,
    ) -> Result<()> {
        check(&cancel)?;
        let shape = matrix.shape;
        if !Arc::ptr_eq(&self.device, &matrix.device)
            || shape.width > self.limits.width
            || shape.rows > self.limits.rows
            || input.is_empty()
            || !input.len().is_multiple_of(shape.width)
            || input.len() / shape.width > self.limits.batch
            || output.len() != input.len() / shape.width * shape.rows
            || input.iter().any(|value| !value.is_finite())
        {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "GPU matrix/workspace/input mismatch",
            ));
        }
        let batch = input.len() / shape.width;
        // SAFETY: exclusively borrowed workspace, no pending GPU command,
        // shared buffer is large enough for the checked F32 input; source is a
        // distinct Rust slice and cannot alias the private Metal allocation.
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                self.input.raw.contents().as_ptr().cast::<f32>(),
                input.len(),
            );
        }
        // Bound both products and launched SIMD lanes, including narrow rows
        // where most lanes have no input columns but still join the reduction.
        let unit_width = match self.device.kernel {
            KernelProfile::Reference => shape.width,
            KernelProfile::SimdCompensated => shape.width.max(self.device.simd_width),
        };
        let rows = (PRODUCTS_PER_DISPATCH / (unit_width * batch)).max(1);
        for first in (0..shape.rows).step_by(rows) {
            check(&cancel)?;
            autoreleasepool(|_| {
                self.dispatch(matrix, batch, first, (shape.rows - first).min(rows))
            })?;
            self.dispatches = self
                .dispatches
                .checked_add(1)
                .ok_or_else(|| Error::new(ErrorKind::Internal, "GPU dispatch counter overflow"))?;
            check(&cancel)?;
            cooperate()?;
        }
        // SAFETY: every requested output element was written by successful
        // commands and waitUntilCompleted retired all GPU writes. Workspace
        // remains exclusively borrowed. Alignment is guaranteed by Metal.
        let values = unsafe {
            std::slice::from_raw_parts(
                self.output.raw.contents().as_ptr().cast::<f32>(),
                output.len(),
            )
        };
        if values.iter().any(|value| !value.is_finite()) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "GPU arithmetic produced non-finite output",
            ));
        }
        output.copy_from_slice(values);
        Ok(())
    }

    fn dispatch(&self, matrix: &Matrix, batch: usize, first: usize, rows: usize) -> Result<()> {
        let row_bytes = matrix.shape.row_bytes()?;
        let command = self
            .device
            .queue
            .commandBuffer()
            .ok_or_else(|| resource("Metal command buffer allocation failed"))?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| resource("Metal compute encoder allocation failed"))?;
        encoder.setComputePipelineState(&self.device.dense);
        let mut shape = [
            matrix.shape.width as u32,
            matrix.shape.rows as u32,
            batch as u32,
            matrix.shape.dtype as u32,
            first as u32,
            rows as u32,
            row_bytes as u32,
        ];
        // SAFETY: only the selected embedded first-party shader is dispatched.
        // Shape bounds were validated before allocation; buffers match its
        // declared scalar types and full index ranges. Bindings 0..=3 are fixed.
        // Input/weights are read-only, output is unique to this workspace;
        // parameter bytes are copied by Metal before this stack value expires.
        // All resources remain strongly retained until command completion.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&matrix.buffer.raw), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&self.input.raw), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&self.output.raw), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::from(&mut shape).cast::<c_void>(),
                std::mem::size_of_val(&shape),
                3,
            );
            let grid = MTLSize {
                width: rows,
                height: batch,
                depth: 1,
            };
            match self.device.kernel {
                KernelProfile::Reference => encoder.dispatchThreads_threadsPerThreadgroup(
                    grid,
                    MTLSize {
                        width: rows
                            .min(self.device.dense.maxTotalThreadsPerThreadgroup())
                            .min(32),
                        height: 1,
                        depth: 1,
                    },
                ),
                // One complete hardware SIMD group cooperates on each output.
                // The constructor checked its width against pipeline limits.
                KernelProfile::SimdCompensated => encoder
                    .dispatchThreadgroups_threadsPerThreadgroup(
                        grid,
                        MTLSize {
                            width: self.device.simd_width,
                            height: 1,
                            depth: 1,
                        },
                    ),
            }
        }
        encoder.endEncoding();
        // No callbacks/fallible work between commit and retirement. Driver
        // failure is checked only after it is safe to release resources.
        let flight = InFlight(command);
        flight.0.commit();
        flight.0.waitUntilCompleted();
        if flight.0.status() != MTLCommandBufferStatus::Completed {
            return Err(Error::new(
                ErrorKind::Internal,
                format!("Metal command failed: {:?}", flight.0.error()),
            ));
        }
        Ok(())
    }
}

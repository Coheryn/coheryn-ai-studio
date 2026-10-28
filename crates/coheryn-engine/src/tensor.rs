//! Shared bounded weight operations. A tile is read once for all input rows.
use crate::{Cancellation, Error, Result, cpu, gguf::Tensor, memory, quant, storage::Storage};
use rayon::prelude::*;

pub(crate) struct Workspace {
    pub(crate) io: Vec<u8>,
    pub(crate) floats: Vec<f32>,
    pub(crate) operand: Vec<f32>,
    pub(crate) quantized: quant::QuantizedInput,
    extra_quantized: Vec<quant::QuantizedInput>,
    tile_rows: usize,
    pool: Option<crate::parallel::CpuPool>,
}
impl Workspace {
    pub(crate) fn install<T: Send>(&self, work: impl FnOnce() -> T + Send) -> T {
        if let Some(pool) = &self.pool {
            pool.install(work)
        } else {
            work()
        }
    }
    pub(crate) fn new(
        width: usize,
        io_bytes: usize,
        threads: usize,
        tile_rows: usize,
    ) -> Result<Self> {
        if !(1..=32).contains(&tile_rows)
            || !(4096..=256 << 10).contains(&io_bytes)
            || !io_bytes.is_power_of_two()
            || width == 0
        {
            return Err(Error::request("invalid tensor workspace bounds"));
        }
        memory::worker_bytes(threads)?;
        let mut io = Vec::new();
        io.try_reserve_exact(io_bytes)
            .map_err(|_| Error::resource("I/O allocation failed"))?;
        io.resize(io_bytes, 0);
        let mut extra_quantized = Vec::new();
        extra_quantized
            .try_reserve_exact(tile_rows - 1)
            .map_err(|_| Error::resource("activation handles allocation failed"))?;
        for _ in 1..tile_rows {
            extra_quantized.push(quant::QuantizedInput::new(width)?);
        }
        Ok(Self {
            io,
            floats: memory::zeros(io_bytes / 2)?,
            operand: memory::zeros(memory::product(&[width, tile_rows])?)?,
            quantized: quant::QuantizedInput::new(width)?,
            extra_quantized,
            tile_rows,
            pool: if threads > 1 {
                Some(crate::parallel::CpuPool::new(threads)?)
            } else {
                None
            },
        })
    }
    fn activation(&self, row: usize) -> &quant::QuantizedInput {
        if row == 0 {
            &self.quantized
        } else {
            &self.extra_quantized[row - 1]
        }
    }
    pub(crate) fn capacity_bytes(&self) -> usize {
        self.io.capacity()
            + 4 * (self.floats.capacity() + self.operand.capacity())
            + self.quantized.capacity_bytes()
            + self
                .extra_quantized
                .iter()
                .map(|q| q.capacity_bytes())
                .sum::<usize>()
            + self.extra_quantized.capacity() * std::mem::size_of::<quant::QuantizedInput>()
    }
}

pub(crate) fn read_row(
    storage: &Storage,
    data: u64,
    tensor: &Tensor,
    row: usize,
    out: &mut [f32],
    work: &mut Workspace,
) -> Result<()> {
    if tensor.dimensions.first().copied() != Some(out.len() as u64) {
        return Err(Error::invalid("tensor row shape mismatch"));
    }
    let bytes = quant::row_bytes(tensor.dtype, out.len())?;
    let offset = row
        .checked_mul(bytes)
        .ok_or_else(|| Error::invalid("row offset overflow"))?;
    if offset
        .checked_add(bytes)
        .is_none_or(|end| end as u64 > tensor.bytes)
        || bytes > work.io.len()
    {
        return Err(Error::invalid("tensor row outside checked storage"));
    }
    storage.read(data + tensor.offset + offset as u64, &mut work.io[..bytes])?;
    cpu::decode(&work.io[..bytes], tensor.dtype, out)
}

/// Token-major inputs/outputs; GGUF matrices remain [input, output].
pub(crate) fn matmul(
    storage: &Storage,
    data: u64,
    tensor: &Tensor,
    input: &[f32],
    output: &mut [f32],
    work: &mut Workspace,
    cancel: &Cancellation,
) -> Result<()> {
    matmul_cooperative(
        storage, data, tensor, input, output, work, cancel, &mut None,
    )
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul_cooperative(
    storage: &Storage,
    data: u64,
    tensor: &Tensor,
    input: &[f32],
    output: &mut [f32],
    work: &mut Workspace,
    cancel: &Cancellation,
    cooperate: &mut Option<&mut crate::multimodal::WorkYield<'_>>,
) -> Result<()> {
    if tensor.dimensions.len() != 2 {
        return Err(Error::invalid("matrix needs two dimensions"));
    }
    let width = usize::try_from(tensor.dimensions[0])
        .map_err(|_| Error::invalid("matrix width overflow"))?;
    let rows = usize::try_from(tensor.dimensions[1])
        .map_err(|_| Error::invalid("matrix rows overflow"))?;
    if width == 0
        || rows == 0
        || input.is_empty()
        || !input.len().is_multiple_of(width)
        || output.len() != memory::product(&[input.len() / width, rows])?
    {
        return Err(Error::invalid("matrix input/output shape mismatch"));
    }
    let row_bytes = quant::row_bytes(tensor.dtype, width)?;
    let chunk_rows = (work.io.len() / row_bytes).min(if quant::is_packed(tensor.dtype) {
        usize::MAX
    } else {
        work.floats.len() / width
    });
    if chunk_rows == 0 || work.operand.len() < memory::product(&[width, work.tile_rows])? {
        return Err(Error::resource("matrix exceeds admitted tensor workspace"));
    }
    for (input, output) in input
        .chunks(work.tile_rows * width)
        .zip(output.chunks_mut(work.tile_rows * rows))
    {
        cancel.check()?;
        storage.unchanged()?;
        let n = input.len() / width;
        if tensor.dtype == 1 {
            cpu::round_f16(input, &mut work.operand[..input.len()])?;
        } else if quant::is_packed(tensor.dtype) {
            for (i, row) in input.chunks_exact(width).enumerate() {
                if i == 0 {
                    work.quantized.prepare(row, tensor.dtype)?;
                } else {
                    work.extra_quantized[i - 1].prepare(row, tensor.dtype)?;
                }
            }
        }
        for start in (0..rows).step_by(chunk_rows) {
            cancel.check()?;
            let count = chunk_rows.min(rows - start);
            let bytes = count * row_bytes;
            storage.read(
                data + tensor.offset + (start * row_bytes) as u64,
                &mut work.io[..bytes],
            )?;
            if !quant::is_packed(tensor.dtype) {
                cpu::decode(
                    &work.io[..bytes],
                    tensor.dtype,
                    &mut work.floats[..count * width],
                )?;
            }
            let compute_row = |index: usize, destination: &mut [f32]| -> Result<()> {
                for (group, outputs) in destination[start..start + count].chunks_mut(8).enumerate()
                {
                    cancel.check()?;
                    for (j, value) in outputs.iter_mut().enumerate() {
                        let row = group * 8 + j;
                        *value = if quant::is_packed(tensor.dtype) {
                            quant::dot(
                                &work.io[row * row_bytes..(row + 1) * row_bytes],
                                tensor.dtype,
                                work.activation(index),
                            )?
                        } else {
                            let source = if tensor.dtype == 1 {
                                &work.operand[index * width..(index + 1) * width]
                            } else {
                                &input[index * width..(index + 1) * width]
                            };
                            cpu::dot(&work.floats[row * width..(row + 1) * width], source)
                        };
                        if !value.is_finite() {
                            return Err(Error::invalid("non-finite matrix result"));
                        }
                    }
                }
                Ok(())
            };
            if let Some(pool) = &work.pool {
                if n > 1 {
                    pool.install(|| {
                        output
                            .par_chunks_mut(rows)
                            .enumerate()
                            .try_for_each(|(i, dst)| compute_row(i, dst))
                    })?;
                } else if count >= pool.current_num_threads() * 8 {
                    let source = if tensor.dtype == 1 {
                        &work.operand[..width]
                    } else {
                        input
                    };
                    pool.install(|| {
                        output[start..start + count]
                            .par_chunks_mut(8)
                            .enumerate()
                            .try_for_each(|(group, dst)| {
                                cancel.check()?;
                                for (j, value) in dst.iter_mut().enumerate() {
                                    let row = group * 8 + j;
                                    *value = if quant::is_packed(tensor.dtype) {
                                        quant::dot(
                                            &work.io[row * row_bytes..(row + 1) * row_bytes],
                                            tensor.dtype,
                                            &work.quantized,
                                        )?
                                    } else {
                                        cpu::dot(
                                            &work.floats[row * width..(row + 1) * width],
                                            source,
                                        )
                                    };
                                    if !value.is_finite() {
                                        return Err(Error::invalid("non-finite matrix result"));
                                    }
                                }
                                Ok(())
                            })
                    })?;
                } else {
                    compute_row(0, output)?;
                }
            } else {
                for (i, dst) in output.chunks_exact_mut(rows).enumerate() {
                    compute_row(i, dst)?;
                }
            }
            if let Some(yield_work) = cooperate.as_deref_mut() {
                yield_work()?;
            }
        }
    }
    Ok(())
}

/// Borrowed independent vectors. Only weight tiles are shared; activation rounding,
/// errors, cancellation and outputs remain request-local.
pub(crate) struct BatchVector<'a> {
    pub input: &'a [f32],
    pub output: &'a mut [f32],
    pub workspace: &'a mut Workspace,
    pub cancel: &'a Cancellation,
    pub error: &'a mut Option<Error>,
}
#[derive(Debug, Default)]
pub(crate) struct BatchIo {
    pub tiles: u64,
    pub bytes: u64,
}
pub(crate) fn matmul_batch(
    storage: &Storage,
    data: u64,
    tensor: &Tensor,
    vectors: &mut [BatchVector<'_>],
) -> Result<BatchIo> {
    if vectors.is_empty() || vectors.len() > 16 || tensor.dimensions.len() != 2 {
        return Err(Error::request("invalid native matrix batch"));
    }
    let width = usize::try_from(tensor.dimensions[0])
        .map_err(|_| Error::invalid("matrix width overflow"))?;
    let rows = usize::try_from(tensor.dimensions[1])
        .map_err(|_| Error::invalid("matrix rows overflow"))?;
    if width == 0 || rows == 0 {
        return Err(Error::invalid("empty matrix"));
    }
    let row_bytes = quant::row_bytes(tensor.dtype, width)?;
    for vector in vectors.iter_mut() {
        if vector.error.is_some() {
            continue;
        }
        let prepared = (|| {
            vector.cancel.check()?;
            if vector.input.len() != width
                || vector.output.len() != rows
                || vector.workspace.operand.len() < width
            {
                return Err(Error::invalid("batch matrix vector shape mismatch"));
            }
            if tensor.dtype == 1 {
                cpu::round_f16(vector.input, &mut vector.workspace.operand[..width])?;
            } else if quant::is_packed(tensor.dtype) {
                vector
                    .workspace
                    .quantized
                    .prepare(vector.input, tensor.dtype)?;
            }
            Ok(())
        })();
        if let Err(error) = prepared {
            *vector.error = Some(error);
        }
    }
    // Lend the first lane's already-admitted tile buffers to the group. Restored on
    // every Result path, including shared storage failure; no batch-sized heap copy.
    let mut io = std::mem::take(&mut vectors[0].workspace.io);
    let mut floats = std::mem::take(&mut vectors[0].workspace.floats);
    let result = (|| {
        let chunk_rows = (io.len() / row_bytes).min(if quant::is_packed(tensor.dtype) {
            usize::MAX
        } else {
            floats.len() / width
        });
        if chunk_rows == 0 {
            return Err(Error::resource("batch matrix exceeds admitted tile"));
        }
        let mut stats = BatchIo::default();
        for start in (0..rows).step_by(chunk_rows) {
            for vector in vectors.iter_mut() {
                if vector.error.is_none()
                    && let Err(error) = vector.cancel.check()
                {
                    *vector.error = Some(error);
                }
            }
            if vectors.iter().all(|v| v.error.is_some()) {
                break;
            }
            storage.unchanged()?;
            let count = chunk_rows.min(rows - start);
            let bytes = count * row_bytes;
            storage.read(
                data + tensor.offset + (start * row_bytes) as u64,
                &mut io[..bytes],
            )?;
            stats.tiles += 1;
            stats.bytes += bytes as u64;
            if !quant::is_packed(tensor.dtype) {
                cpu::decode(&io[..bytes], tensor.dtype, &mut floats[..count * width])?;
            }
            for vector in vectors.iter_mut().filter(|v| v.error.is_none()) {
                let work = &*vector.workspace;
                let source = if tensor.dtype == 1 {
                    &work.operand[..width]
                } else {
                    vector.input
                };
                let cancel = vector.cancel;
                let compute = |group: usize, dst: &mut [f32]| -> Result<()> {
                    cancel.check()?;
                    for (j, value) in dst.iter_mut().enumerate() {
                        let row = group * 8 + j;
                        *value = if quant::is_packed(tensor.dtype) {
                            quant::dot(
                                &io[row * row_bytes..(row + 1) * row_bytes],
                                tensor.dtype,
                                &work.quantized,
                            )?
                        } else {
                            cpu::dot(&floats[row * width..(row + 1) * width], source)
                        };
                        if !value.is_finite() {
                            return Err(Error::invalid("non-finite batch matrix result"));
                        }
                    }
                    Ok(())
                };
                let result = if let Some(pool) = &work.pool {
                    pool.install(|| {
                        vector.output[start..start + count]
                            .par_chunks_mut(8)
                            .enumerate()
                            .try_for_each(|(g, d)| compute(g, d))
                    })
                } else {
                    vector.output[start..start + count]
                        .chunks_mut(8)
                        .enumerate()
                        .try_for_each(|(g, d)| compute(g, d))
                };
                if let Err(error) = result {
                    *vector.error = Some(error);
                }
            }
        }
        Ok(stats)
    })();
    vectors[0].workspace.io = io;
    vectors[0].workspace.floats = floats;
    result
}

#[cfg(test)]
mod batch_tests {
    use super::*;
    use crate::inference::cache_model_fixture as fixture;
    #[test]
    fn shared_tiles_match_independent_float_and_packed_matvecs_and_isolate_cancelled_lane() {
        let cases: serde_json::Value =
            serde_json::from_str(include_str!("../../../fixtures/quantized/kernels.json")).unwrap();
        for dtype in [0, 1, 2, 6, 8, 12, 14] {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("weights.gguf");
            let mut bytes = fixture::packed_model(dtype);
            std::fs::write(&path, &bytes).unwrap();
            let descriptor = crate::inspect(&path, Default::default()).unwrap();
            let tensor = descriptor
                .tensors
                .iter()
                .find(|t| t.name == "blk.0.attn_q.weight")
                .unwrap();
            let width = tensor.dimensions[0] as usize;
            let rows = tensor.dimensions[1] as usize;
            let row_bytes = quant::row_bytes(dtype, width).unwrap();
            let start = (descriptor.data_offset + tensor.offset) as usize;
            for row in 0..rows {
                if dtype <= 1 {
                    for col in 0..width {
                        let value = ((row * 13 + col * 7) % 31) as f32 / 32. - 0.5;
                        let encoded = if dtype == 0 {
                            value.to_le_bytes().to_vec()
                        } else {
                            half::f16::from_f32(value).to_le_bytes().to_vec()
                        };
                        let offset = start + row * row_bytes + col * encoded.len();
                        bytes[offset..offset + encoded.len()].copy_from_slice(&encoded);
                    }
                } else {
                    let case = cases["cases"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|c| {
                            c["dtype"] == dtype && c["blocks"] == 1 && c["pattern"] == 1 + row % 3
                        })
                        .unwrap();
                    let block: Vec<u8> = case["bytes"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|b| b.as_u64().unwrap() as u8)
                        .collect();
                    for dst in bytes[start + row * row_bytes..start + (row + 1) * row_bytes]
                        .chunks_exact_mut(block.len())
                    {
                        dst.copy_from_slice(&block);
                    }
                }
            }
            std::fs::write(&path, bytes).unwrap();
            let storage = Storage::open(&path, &descriptor).unwrap();
            let inputs: Vec<Vec<f32>> = (0..3)
                .map(|lane| {
                    (0..width)
                        .map(|i| ((i * 7 + lane * 11) % 29) as f32 * 0.125 - 0.5)
                        .collect()
                })
                .collect();
            let mut expected = vec![vec![0.; rows]; 3];
            for (input, output) in inputs.iter().zip(&mut expected) {
                matmul(
                    &storage,
                    descriptor.data_offset,
                    tensor,
                    input,
                    output,
                    &mut Workspace::new(width, 4096, 1, 1).unwrap(),
                    &Cancellation::default(),
                )
                .unwrap();
            }
            let mut outputs = vec![vec![0.; rows]; 3];
            let mut work: Vec<_> = (0..3)
                .map(|_| Workspace::new(width, 4096, 1, 1).unwrap())
                .collect();
            let cancellations = [
                Cancellation::default(),
                Cancellation::default(),
                Cancellation::default(),
            ];
            let mut errors = [None, None, None];
            let mut vectors: Vec<_> = inputs
                .iter()
                .zip(outputs.iter_mut())
                .zip(work.iter_mut())
                .zip(cancellations.iter())
                .zip(errors.iter_mut())
                .map(
                    |((((input, output), workspace), cancel), error)| BatchVector {
                        input,
                        output,
                        workspace,
                        cancel,
                        error,
                    },
                )
                .collect();
            let stats =
                matmul_batch(&storage, descriptor.data_offset, tensor, &mut vectors).unwrap();
            assert_eq!(
                stats.bytes, tensor.bytes,
                "dtype {dtype}: one shared read per tile"
            );
            assert!(vectors.iter().all(|v| v.error.is_none()));
            drop(vectors);
            assert_eq!(outputs, expected, "dtype {dtype}");
            cancellations[0].cancel();
            let mut vectors: Vec<_> = inputs
                .iter()
                .zip(outputs.iter_mut())
                .zip(work.iter_mut())
                .zip(cancellations.iter())
                .zip(errors.iter_mut())
                .map(
                    |((((input, output), workspace), cancel), error)| BatchVector {
                        input,
                        output,
                        workspace,
                        cancel,
                        error,
                    },
                )
                .collect();
            matmul_batch(&storage, descriptor.data_offset, tensor, &mut vectors).unwrap();
            drop(vectors);
            assert_eq!(
                errors[0].as_ref().unwrap().code,
                crate::ErrorCode::Cancelled
            );
            assert!(errors[1..].iter().all(Option::is_none));
            assert_eq!(outputs[1..], expected[1..]);
            assert!(
                work.iter().all(|w| w.io.len() == 4096),
                "loaned tile restored after cancellation"
            );
        }
    }
}

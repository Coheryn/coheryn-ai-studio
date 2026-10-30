use crate::{Cancellation, Error, Result, cpu, memory, tensor::Workspace};
use rayon::prelude::*;

pub(crate) struct AttentionScratch {
    pub scores: Vec<f32>,
    pub accumulator: Vec<f64>,
}
impl AttentionScratch {
    pub fn new(rows: usize, head_dim: usize) -> Result<Self> {
        Ok(Self {
            scores: memory::zeros(rows)?,
            accumulator: memory::zeros_f64(head_dim)?,
        })
    }
    pub fn capacity_bytes(&self) -> usize {
        self.scores.capacity() * 4 + self.accumulator.capacity() * 8
    }
}
/// Non-causal spatial attention retains one score row per worker, never an N×N matrix.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    scratch: &mut [AttentionScratch],
    hidden: usize,
    heads: usize,
    work: &Workspace,
    cancel: &Cancellation,
) -> Result<()> {
    attention_cooperative(
        q, k, v, out, scratch, hidden, heads, work, cancel, &mut None,
    )
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_cooperative(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    scratch: &mut [AttentionScratch],
    hidden: usize,
    heads: usize,
    work: &Workspace,
    cancel: &Cancellation,
    cooperate: &mut Option<&mut crate::multimodal::WorkYield<'_>>,
) -> Result<()> {
    if hidden == 0
        || heads == 0
        || !hidden.is_multiple_of(heads)
        || q.is_empty()
        || !q.len().is_multiple_of(hidden)
        || q.len() != k.len()
        || q.len() != v.len()
        || q.len() != out.len()
        || scratch.is_empty()
    {
        return Err(Error::invalid("vision attention dimensions disagree"));
    }
    let rows = q.len() / hidden;
    let head_dim = hidden / heads;
    if scratch
        .iter()
        .any(|s| s.scores.len() < rows || s.accumulator.len() != head_dim)
    {
        return Err(Error::resource(
            "vision attention exceeds admitted row workspace",
        ));
    }
    let scale = 1. / (head_dim as f32).sqrt();
    for (tile, target_rows) in out.chunks_mut(32 * hidden).enumerate() {
        let group = (target_rows.len() / hidden).div_ceil(scratch.len());
        let compute = |(worker, (output, scratch)): (
            usize,
            (&mut [f32], &mut AttentionScratch),
        )|
         -> Result<()> {
            for (local, target) in output.chunks_exact_mut(hidden).enumerate() {
                cancel.check()?;
                let row = tile * 32 + worker * group + local;
                for head in 0..heads {
                    let offset = head * head_dim;
                    let query = &q[row * hidden + offset..row * hidden + offset + head_dim];
                    let scores = &mut scratch.scores[..rows];
                    let mut maximum = f32::NEG_INFINITY;
                    for (key, score) in scores.iter_mut().enumerate() {
                        if key.is_multiple_of(64) {
                            cancel.check()?;
                        }
                        *score = cpu::dot(
                            query,
                            &k[key * hidden + offset..key * hidden + offset + head_dim],
                        ) * scale;
                        if !score.is_finite() {
                            return Err(Error::invalid("non-finite vision attention score"));
                        }
                        maximum = maximum.max(*score);
                    }
                    let mut sum = 0f64;
                    for score in scores.iter_mut() {
                        *score = (*score - maximum).exp();
                        sum += f64::from(*score);
                    }
                    let inverse = (1. / sum) as f32;
                    for score in scores.iter_mut() {
                        *score *= inverse;
                    }
                    scratch.accumulator.fill(0.);
                    for (key, score) in scores.iter().enumerate() {
                        if key.is_multiple_of(64) {
                            cancel.check()?;
                        }
                        for (sum, value) in scratch
                            .accumulator
                            .iter_mut()
                            .zip(&v[key * hidden + offset..key * hidden + offset + head_dim])
                        {
                            *sum += f64::from(*score * *value);
                        }
                    }
                    for (dst, sum) in target[offset..offset + head_dim]
                        .iter_mut()
                        .zip(&scratch.accumulator)
                    {
                        *dst = *sum as f32;
                        if !dst.is_finite() {
                            return Err(Error::invalid("non-finite vision attention output"));
                        }
                    }
                }
            }
            Ok(())
        };
        if scratch.len() > 1 {
            work.install(|| {
                target_rows
                    .par_chunks_mut(group * hidden)
                    .zip(scratch.par_iter_mut())
                    .enumerate()
                    .try_for_each(compute)
            })
        } else {
            target_rows
                .chunks_mut(group * hidden)
                .zip(scratch.iter_mut())
                .enumerate()
                .try_for_each(compute)
        }?;
        if let Some(yield_work) = cooperate.as_deref_mut() {
            yield_work()?;
        }
    }
    Ok(())
}
// Preserve paired trigonometric evaluation across optimization boundaries. On the
// qualified CPU, splitting this call changes F32 values before F16 projections.
#[inline(never)]
fn spatial_trig(angle: f32) -> (f32, f32) {
    angle.sin_cos()
}
/// Grouped raster order: four adjacent patch rows form each spatial merge group.
pub(crate) fn rope_cache(
    grid: [usize; 3],
    head_dim: usize,
    cache: &mut [f32],
    cancel: &Cancellation,
) -> Result<()> {
    let rows = memory::product(&grid)?;
    if grid[0] != 1
        || !grid[1].is_multiple_of(2)
        || !grid[2].is_multiple_of(2)
        || grid[1] == 0
        || grid[2] == 0
        || head_dim == 0
        || !head_dim.is_multiple_of(4)
        || cache.len() != memory::product(&[rows, head_dim])?
    {
        return Err(Error::invalid(
            "vision spatial rotation dimensions disagree",
        ));
    }
    let quarter = head_dim / 4;
    let step = 10000f32.powf(-2. / (head_dim / 2) as f32);
    for row in 0..rows {
        cancel.check()?;
        let group = row / 4;
        let width = grid[2] / 2;
        let y = group / width * 2 + row % 4 / 2;
        let x = group % width * 2 + row % 2;
        for (axis, position) in [y, x].into_iter().enumerate() {
            let mut angle = position as f32;
            for i in 0..quarter {
                let base = row * head_dim + 2 * (axis * quarter + i);
                // Paired trig follows the scalar reference rounding on this CPU.
                let (sin, cos) = spatial_trig(angle);
                cache[base] = cos;
                cache[base + 1] = sin;
                angle *= step;
            }
        }
    }
    Ok(())
}
pub(crate) fn rotate(
    values: &mut [f32],
    hidden: usize,
    head_dim: usize,
    cache: &[f32],
    cancel: &Cancellation,
) -> Result<()> {
    if hidden == 0
        || head_dim == 0
        || !head_dim.is_multiple_of(4)
        || !hidden.is_multiple_of(head_dim)
        || !values.len().is_multiple_of(hidden)
        || cache.len() != memory::product(&[values.len() / hidden, head_dim])?
    {
        return Err(Error::invalid("vision spatial rotation buffer mismatch"));
    }
    for (row, values) in values.chunks_exact_mut(hidden).enumerate() {
        cancel.check()?;
        for head in values.chunks_exact_mut(head_dim) {
            for i in 0..head_dim / 2 {
                let a = head[i];
                let b = head[i + head_dim / 2];
                let base = row * head_dim + 2 * i;
                let cos = cache[base];
                let sin = cache[base + 1];
                head[i] = a * cos - b * sin;
                head[i + head_dim / 2] = a * sin + b * cos;
            }
        }
    }
    Ok(())
}
pub(crate) fn activation(values: &mut [f32], quick: bool, cancel: &Cancellation) -> Result<()> {
    for chunk in values.chunks_mut(4096) {
        cancel.check()?;
        for x in chunk {
            *x = if quick {
                *x * (1. / (1. + (-1.702 * *x).exp()))
            } else {
                0.5 * *x * (1. + (0.797_884_6 * *x * (1. + 0.044715 * *x * *x)).tanh())
            };
            if !x.is_finite() {
                return Err(Error::invalid("non-finite vision activation"));
            }
        }
    }
    Ok(())
}
/// Match the frozen scalar F16 lookup-GELU contract without retaining a table:
/// round input to binary16, evaluate tanh GELU, then round output to binary16.
/// The qualified profile saturates inputs outside [-10,10] before conversion.
pub(crate) fn merger_activation(values: &mut [f32], cancel: &Cancellation) -> Result<()> {
    for chunk in values.chunks_mut(4096) {
        cancel.check()?;
        for value in chunk {
            let x = *value;
            if !x.is_finite() {
                return Err(Error::invalid("non-finite merger activation"));
            }
            *value = if x <= -10. {
                0.
            } else if x >= 10. {
                x
            } else {
                let x = half::f16::from_f32(x).to_f32();
                let result = 0.5 * x * (1. + (0.797_884_6 * x * (1. + 0.044715 * x * x)).tanh());
                half::f16::from_f32(result).to_f32()
            };
        }
    }
    Ok(())
}
pub(crate) fn residual(values: &mut [f32], delta: &[f32], cancel: &Cancellation) -> Result<()> {
    if values.len() != delta.len() {
        return Err(Error::invalid("vision residual dimensions disagree"));
    }
    for (x, y) in values.chunks_mut(4096).zip(delta.chunks(4096)) {
        cancel.check()?;
        for (x, y) in x.iter_mut().zip(y) {
            *x += y;
            if !x.is_finite() {
                return Err(Error::invalid("non-finite vision residual"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tiled_attention_matches_independent_67_row_fixture_and_yields_after_joined_units() {
        let f: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../fixtures/vlm/cooperative-attention-v1.json"
        ))
        .unwrap();
        let vector = |key: &str| {
            f[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let q = vector("q");
        let k = vector("k");
        let v = vector("v");
        let expected = vector("expected");
        for threads in [1, 2] {
            let work = Workspace::new(4, 4096, threads, 7).unwrap();
            let mut scratch = (0..threads)
                .map(|_| AttentionScratch::new(67, 2).unwrap())
                .collect::<Vec<_>>();
            let mut out = vec![0.; q.len()];
            let mut units = 0;
            let mut checkpoint = || {
                units += 1;
                Ok(())
            };
            attention_cooperative(
                &q,
                &k,
                &v,
                &mut out,
                &mut scratch,
                4,
                2,
                &work,
                &Cancellation::default(),
                &mut Some(&mut checkpoint),
            )
            .unwrap();
            assert_eq!(units, 3);
            for (got, want) in out.iter().zip(&expected) {
                assert!((got - want).abs() < 2e-6, "{got} != {want}");
            }
            out.fill(-42.);
            let mut cancel = || {
                Err(Error::new(
                    crate::ErrorCode::Cancelled,
                    "stop after one complete query tile",
                ))
            };
            assert_eq!(
                attention_cooperative(
                    &q,
                    &k,
                    &v,
                    &mut out,
                    &mut scratch,
                    4,
                    2,
                    &work,
                    &Cancellation::default(),
                    &mut Some(&mut cancel)
                )
                .unwrap_err()
                .code,
                crate::ErrorCode::Cancelled
            );
            assert!(out[..128].iter().any(|v| *v != -42.));
            assert!(out[128..].iter().all(|v| *v == -42.));
        }
    }
    #[test]
    fn spatial_axes_follow_merge_order_and_scalar_trigonometry() {
        let cancel = Cancellation::default();
        let mut cache = vec![0.; 8 * 8];
        rope_cache([1, 2, 4], 8, &mut cache, &cancel).unwrap();
        assert_eq!(&cache[..8], &[1., 0., 1., 0., 1., 0., 1., 0.]);
        // Row 1 is (y=0,x=1), row 2 (y=1,x=0), row 4 (y=0,x=2).
        assert_eq!(cache[8 + 4].to_bits(), 0x3f0a5140);
        assert_eq!(cache[16].to_bits(), 0x3f0a5140);
        assert_eq!(cache[32 + 4].to_bits(), 0xbed51133);
        let mut input = vec![1.; 8 * 8];
        rotate(&mut input, 8, 8, &cache, &cancel).unwrap();
        assert_eq!(&input[..8], &[1.; 8]);
        assert!((input[8 + 2] - (1f32.cos() - 1f32.sin())).abs() < 1e-7);
        assert!(rope_cache([1, 3, 4], 8, &mut cache, &cancel).is_err());
        assert!(rotate(&mut input, 8, 3, &cache, &cancel).is_err());
        cancel.cancel();
        assert_eq!(
            rope_cache([1, 2, 4], 8, &mut cache, &cancel)
                .unwrap_err()
                .code,
            crate::ErrorCode::Cancelled
        );
    }
    #[test]
    fn spatial_attention_includes_future_rows_and_parallelism_preserves_results() {
        let q = vec![0.; 24];
        let k = q.clone();
        let v: Vec<f32> = (0..24).map(|n| n as f32).collect();
        let cancel = Cancellation::default();
        let mut single = vec![0.; 24];
        let mut parallel = vec![0.; 24];
        for (threads, out) in [(1, &mut single), (2, &mut parallel)] {
            let work = Workspace::new(8, 4096, threads, 1).unwrap();
            let mut scratch = (0..threads)
                .map(|_| AttentionScratch::new(3, 4).unwrap())
                .collect::<Vec<_>>();
            attention(&q, &k, &v, out, &mut scratch, 8, 2, &work, &cancel).unwrap();
            for row in out.as_chunks::<8>().0 {
                for (i, x) in row.iter().enumerate() {
                    assert!((*x - (8 + i) as f32).abs() < 2e-6);
                }
            }
            scratch[0].scores.clear();
            assert!(attention(&q, &k, &v, out, &mut scratch, 8, 2, &work, &cancel).is_err());
        }
        assert_eq!(single, parallel);
    }
    #[test]
    fn merger_precision_matches_f16_input_and_output_rounding() {
        let mut x = [-11., -2.0001, 0., 2.0001, 11.];
        merger_activation(&mut x, &Cancellation::default()).unwrap();
        assert_eq!(x[0], 0.);
        assert_eq!(x[2], 0.);
        assert_eq!(x[4], 11.);
        for i in [1, 3] {
            assert_eq!(x[i], half::f16::from_f32(x[i]).to_f32());
        }
        assert_eq!(x[3], 1.9550781);
        assert!(merger_activation(&mut [f32::INFINITY], &Cancellation::default()).is_err());
        let c = Cancellation::default();
        c.cancel();
        assert!(merger_activation(&mut [1.], &c).is_err());
    }
    #[test]
    fn activation_and_residual_reject_nonfinite_and_cancelled_work() {
        let cancel = Cancellation::default();
        let mut a = [-2., 0., 2.];
        activation(&mut a, true, &cancel).unwrap();
        for (x, y) in a.into_iter().zip([-0.06434137, 0., 1.9356586]) {
            assert!((x - y).abs() < 1e-6);
        }
        let mut a = [-2., 0., 2.];
        activation(&mut a, false, &cancel).unwrap();
        for (x, y) in a.into_iter().zip([-0.0454023, 0., 1.9545977]) {
            assert!((x - y).abs() < 1e-6);
        }
        assert!(activation(&mut [f32::NAN], true, &cancel).is_err());
        assert!(residual(&mut [f32::MAX], &[f32::MAX], &cancel).is_err());
        cancel.cancel();
        assert!(activation(&mut [1.; 8], true, &cancel).is_err());
    }
}

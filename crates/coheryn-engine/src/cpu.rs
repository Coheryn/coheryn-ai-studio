//! Coheryn-owned scalar kernels; GGML matrix order is [input, output].
use crate::{Error, Result};
use half::f16;

pub fn decode(bytes: &[u8], dtype: u32, output: &mut [f32]) -> Result<()> {
    if crate::quant::is_packed(dtype) {
        return crate::quant::decode(bytes, dtype, output);
    }
    let size = match dtype {
        0 => 4,
        1 | 30 => 2,
        _ => return Err(Error::invalid("reference decoder expects F32/F16/BF16")),
    };
    if Some(bytes.len()) != output.len().checked_mul(size) {
        return Err(Error::invalid("numeric buffer shape mismatch"));
    }
    for (dst, src) in output.iter_mut().zip(bytes.chunks_exact(size)) {
        *dst = if size == 4 {
            f32::from_le_bytes(src.try_into().expect("four bytes"))
        } else if dtype == 30 {
            f32::from_bits(u32::from(u16::from_le_bytes(src.try_into().expect("two bytes"))) << 16)
        } else {
            f16::from_bits(u16::from_le_bytes(src.try_into().expect("two bytes"))).to_f32()
        };
        if !dst.is_finite() {
            return Err(Error::invalid("non-finite weight"));
        }
    }
    Ok(())
}
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    // F32 products with a wide scalar reference reduction, rounded back to F32.
    // Optimized F32 accumulators must qualify against this independent baseline.
    a.iter().zip(b).map(|(x, y)| (*x * *y) as f64).sum::<f64>() as f32
}
/// FP16 matvec contract: binary16 operands, F32 products and output, wide reduction.
pub fn round_f16(input: &[f32], output: &mut [f32]) -> Result<()> {
    if input.len() != output.len() {
        return Err(Error::invalid("FP16 operand shape mismatch"));
    }
    for (src, dst) in input.iter().zip(output) {
        *dst = f16::from_f32(*src).to_f32();
        if !dst.is_finite() {
            return Err(Error::invalid("non-finite FP16 activation"));
        }
    }
    Ok(())
}
pub fn rms_norm(x: &[f32], weight: &[f32], epsilon: f32, out: &mut [f32]) {
    // Wide reduction avoids context-dependent roundoff before binary16 operands.
    let sum: f64 = x.iter().map(|v| (*v * *v) as f64).sum();
    let mean = (sum / x.len() as f64) as f32;
    let scale = (mean + epsilon).sqrt().recip();
    for ((y, x), w) in out.iter_mut().zip(x).zip(weight) {
        *y = *x * scale * *w;
    }
}
/// Qwen/NeoX pairs first and second halves within each head.
pub fn rope(x: &mut [f32], head_dim: usize, position: usize, base: f32) {
    let frequency_step = base.powf(-2.0 / head_dim as f32);
    for head in x.chunks_exact_mut(head_dim) {
        let mut angle = position as f32;
        for i in 0..head_dim / 2 {
            let sin = angle.sin();
            let cos = angle.cos();
            let a = head[i];
            let b = head[i + head_dim / 2];
            head[i] = a.mul_add(cos, -(b * sin));
            head[i + head_dim / 2] = a.mul_add(sin, b * cos);
            angle *= frequency_step;
        }
    }
}
pub fn swiglu(gate: &mut [f32], up: &[f32]) {
    for (g, u) in gate.iter_mut().zip(up) {
        *g = (*g / (1.0 + (-*g).exp())) * *u;
    }
}
/// Online softmax attention: O(head_dim) workspace, no attention score matrix.
pub fn attention(
    q: &[f32],
    keys: &[f32],
    values: &[f32],
    kv_heads: usize,
    head_dim: usize,
    output: &mut [f32],
    mut check: impl FnMut() -> Result<()>,
) -> Result<()> {
    let heads = q.len() / head_dim;
    let stride = kv_heads * head_dim;
    for h in 0..heads {
        check()?;
        let query = &q[h * head_dim..(h + 1) * head_dim];
        let target = &mut output[h * head_dim..(h + 1) * head_dim];
        target.fill(0.0);
        let kv = h / (heads / kv_heads);
        let mut maximum = f32::NEG_INFINITY;
        let mut denominator = 0.0;
        for (i, (k, v)) in keys
            .chunks_exact(stride)
            .zip(values.chunks_exact(stride))
            .enumerate()
        {
            if i % 64 == 0 {
                check()?;
            }
            let k = &k[kv * head_dim..(kv + 1) * head_dim];
            let v = &v[kv * head_dim..(kv + 1) * head_dim];
            let score = dot(query, k) / (head_dim as f32).sqrt();
            let next = maximum.max(score);
            let old = (maximum - next).exp();
            let new = (score - next).exp();
            denominator = denominator * old + new;
            for (o, v) in target.iter_mut().zip(v) {
                *o = *o * old + new * *v;
            }
            maximum = next;
        }
        for o in target {
            *o /= denominator;
        }
    }
    Ok(())
}

/// Linear-in-context workspace and stable two-pass softmax. No quadratic score matrix.
/// Wide value reduction matches the qualified non-flash scalar attention contract.
pub struct AttentionWorkspace<'a> {
    pub output: &'a mut [f32],
    pub scores: &'a mut [f32],
    pub accumulator: &'a mut [f64],
}
pub fn attention_rows(
    q: &[f32],
    keys: &[f32],
    values: &[f32],
    kv_heads: usize,
    head_dim: usize,
    work: AttentionWorkspace<'_>,
    mut check: impl FnMut() -> Result<()>,
) -> Result<()> {
    let stride = kv_heads
        .checked_mul(head_dim)
        .filter(|n| *n > 0)
        .ok_or_else(|| Error::invalid("invalid attention dimensions"))?;
    if q.is_empty()
        || !q.len().is_multiple_of(head_dim)
        || !(q.len() / head_dim).is_multiple_of(kv_heads)
        || keys.is_empty()
        || !keys.len().is_multiple_of(stride)
        || keys.len() != values.len()
        || work.output.len() != q.len()
        || work.scores.len() < keys.len() / stride
        || work.accumulator.len() < head_dim
    {
        return Err(Error::invalid("attention workspace/shape mismatch"));
    }
    let heads = q.len() / head_dim;
    let count = keys.len() / stride;
    let scale = 1. / (head_dim as f32).sqrt();
    let scores = &mut work.scores[..count];
    let acc = &mut work.accumulator[..head_dim];
    for h in 0..heads {
        check()?;
        let group = h / (heads / kv_heads);
        let query = &q[h * head_dim..(h + 1) * head_dim];
        let mut maximum = f32::NEG_INFINITY;
        for (index, (score, k)) in scores.iter_mut().zip(keys.chunks_exact(stride)).enumerate() {
            if index % 64 == 0 {
                check()?;
            }
            *score = dot(query, &k[group * head_dim..(group + 1) * head_dim]) * scale;
            if !score.is_finite() {
                return Err(Error::invalid("non-finite attention score"));
            }
            maximum = maximum.max(*score);
        }
        let mut denominator = 0f64;
        for score in scores.iter_mut() {
            *score = (*score - maximum).exp();
            denominator += f64::from(*score);
        }
        let inverse = (1. / denominator) as f32;
        for score in scores.iter_mut() {
            *score *= inverse;
        }
        acc.fill(0.);
        for (index, (score, row)) in scores.iter().zip(values.chunks_exact(stride)).enumerate() {
            if index % 64 == 0 {
                check()?;
            }
            for (sum, value) in acc
                .iter_mut()
                .zip(&row[group * head_dim..(group + 1) * head_dim])
            {
                *sum += f64::from(*score * *value);
            }
        }
        for (out, value) in work.output[h * head_dim..(h + 1) * head_dim]
            .iter_mut()
            .zip(acc.iter())
        {
            *out = *value as f32;
            if !out.is_finite() {
                return Err(Error::invalid("non-finite attention result"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn half_and_float_decoding() {
        let mut v = [0.; 2];
        decode(&[0, 0, 0, 0x3c], 1, &mut v).unwrap();
        assert_eq!(v, [0., 1.]);
        assert!(decode(&[0xff; 4], 0, &mut [0.]).is_err());
        let mut rounded = [0.; 2];
        round_f16(&[1.0001, -2.0001], &mut rounded).unwrap();
        assert_eq!(rounded, [1., -2.]);
    }
    #[test]
    fn reference_reduction_preserves_small_terms() {
        assert_eq!(dot(&[1e8, 1., -1e8], &[1., 1., 1.]), 1.);
    }
    #[test]
    fn norm_and_rope_reference() {
        let mut out = [0.; 2];
        rms_norm(&[3., 4.], &[1., 1.], 0., &mut out);
        assert!((out[0] - 0.84852815).abs() < 1e-6);
        let mut x = [1., 2., 3., 4.];
        rope(&mut x, 4, 0, 10000.);
        assert_eq!(x, [1., 2., 3., 4.]);
        rope(&mut x, 4, 1, 10000.);
        assert!((x[0] - (1f32.cos() - 3. * 1f32.sin())).abs() < 1e-6);
    }
    #[test]
    fn online_attention_matches_explicit_softmax() {
        let mut out = [0.; 2];
        attention(
            &[1., 0.],
            &[1., 0., 0., 1.],
            &[2., 4., 6., 8.],
            1,
            2,
            &mut out,
            || Ok(()),
        )
        .unwrap();
        let p = (1f32 / 2f32.sqrt()).exp();
        assert!((out[0] - (2. * p + 6.) / (p + 1.)).abs() < 1e-6);
        assert!((out[1] - (4. * p + 8.) / (p + 1.)).abs() < 1e-6);
    }
}

#[cfg(test)]
mod row_attention_tests {
    use super::*;
    #[test]
    fn row_attention_checks_shapes_cancellation_and_matches_double_reference() {
        let q = [1., -0.5, 0.25, 1., -1., 0.5, 0.5, -0.25];
        let keys: Vec<f32> = (0..129 * 4).map(|i| (i % 17) as f32 / 16. - 0.5).collect();
        let values: Vec<f32> = (0..129 * 4).map(|i| (i % 23) as f32 / 8. - 1.).collect();
        let mut scores = [0.; 129];
        let mut acc = [0f64; 2];
        let mut output = [0.; 8];
        attention_rows(
            &q,
            &keys,
            &values,
            2,
            2,
            AttentionWorkspace {
                output: &mut output,
                scores: &mut scores,
                accumulator: &mut acc,
            },
            || Ok(()),
        )
        .unwrap();
        for h in 0..4 {
            let kv = h / 2;
            let weights: Vec<f64> = (0..129)
                .map(|row| {
                    let score = (0..2)
                        .map(|j| f64::from(q[h * 2 + j]) * f64::from(keys[row * 4 + kv * 2 + j]))
                        .sum::<f64>()
                        / 2f64.sqrt();
                    score.exp()
                })
                .collect();
            let sum = weights.iter().sum::<f64>();
            for j in 0..2 {
                let expected = weights
                    .iter()
                    .enumerate()
                    .map(|(row, w)| w * f64::from(values[row * 4 + kv * 2 + j]))
                    .sum::<f64>()
                    / sum;
                assert!((f64::from(output[h * 2 + j]) - expected).abs() < 1e-6);
            }
        }
        assert!(
            attention_rows(
                &q,
                &keys,
                &values,
                0,
                2,
                AttentionWorkspace {
                    output: &mut output,
                    scores: &mut scores,
                    accumulator: &mut acc
                },
                || Ok(())
            )
            .is_err()
        );
        assert!(
            attention_rows(
                &q,
                &keys,
                &values,
                2,
                2,
                AttentionWorkspace {
                    output: &mut output,
                    scores: &mut scores[..1],
                    accumulator: &mut acc
                },
                || Ok(())
            )
            .is_err()
        );
        let error = attention_rows(
            &q,
            &keys,
            &values,
            2,
            2,
            AttentionWorkspace {
                output: &mut output,
                scores: &mut scores,
                accumulator: &mut acc,
            },
            || Err(Error::new(crate::ErrorCode::Cancelled, "test cancellation")),
        )
        .unwrap_err();
        assert_eq!(error.code, crate::ErrorCode::Cancelled);
    }
}

/// Affine LayerNorm shared by native encoder graphs.
pub fn layer_norm(x: &mut [f32], weight: &[f32], bias: &[f32], epsilon: f32) -> Result<()> {
    if x.is_empty()
        || x.len() != weight.len()
        || x.len() != bias.len()
        || !epsilon.is_finite()
        || epsilon <= 0.
    {
        return Err(Error::invalid("LayerNorm shape/epsilon mismatch"));
    }
    let mean = (x.iter().map(|v| f64::from(*v)).sum::<f64>() as f32) / x.len() as f32;
    let mut variance = 0f64;
    for value in x.iter_mut() {
        *value -= mean;
        variance += f64::from(*value * *value);
    }
    let scale = 1. / ((variance / x.len() as f64) as f32 + epsilon).sqrt();
    for ((value, weight), bias) in x.iter_mut().zip(weight).zip(bias) {
        *value *= scale;
        *value *= weight;
        *value += bias;
        if !value.is_finite() {
            return Err(Error::invalid("non-finite LayerNorm output"));
        }
    }
    Ok(())
}

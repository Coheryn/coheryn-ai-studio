//! Original native scalar operations for the hybrid recurrent decoder.
use crate::{Cancellation, Error, Result, cpu, memory};

fn finite(values: &[f32]) -> Result<()> {
    if values.iter().any(|v| !v.is_finite()) {
        Err(Error::request("non-finite recurrent operation"))
    } else {
        Ok(())
    }
}
pub(crate) fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}
pub(crate) fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}
pub(crate) fn softplus(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

pub(crate) fn norm(
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
    zero_centered: bool,
    out: &mut [f32],
) -> Result<()> {
    if input.is_empty()
        || input.len() != weight.len()
        || out.len() != input.len()
        || !epsilon.is_finite()
        || epsilon <= 0.0
    {
        return Err(Error::request("invalid recurrent norm shape/epsilon"));
    }
    finite(input)?;
    finite(weight)?;
    let mean = input.iter().map(|x| f64::from(*x * *x)).sum::<f64>() / input.len() as f64;
    if !mean.is_finite() || !(mean as f32).is_finite() {
        return Err(Error::request("normalization reduction overflow"));
    }
    let scale = (mean as f32 + epsilon).sqrt().recip();
    for ((y, x), w) in out.iter_mut().zip(input).zip(weight) {
        *y = (*x * scale) * (*w + f32::from(u8::from(zero_centered)));
    }
    finite(out)
}

/// Per-channel causal convolution; retained history contains pre-convolution inputs.
pub(crate) struct ConvState {
    history: Vec<f32>,
    channels: usize,
    kernel: usize,
    cursor: usize,
}
impl ConvState {
    pub(crate) fn copy_from(&mut self, source: &Self) -> Result<()> {
        if self.channels != source.channels || self.kernel != source.kernel {
            return Err(Error::request("convolution cache shape mismatch"));
        }
        self.history.copy_from_slice(&source.history);
        self.cursor = source.cursor;
        Ok(())
    }

    pub(crate) fn capacity_bytes(&self) -> usize {
        self.history.capacity() * std::mem::size_of::<f32>()
    }
    pub(crate) fn new(channels: usize, kernel: usize) -> Result<Self> {
        if channels == 0 || channels > 65536 || !(1..=32).contains(&kernel) {
            return Err(Error::request("invalid causal convolution dimensions"));
        }
        Ok(Self {
            history: memory::zeros(memory::product(&[channels, kernel])?)?,
            channels,
            kernel,
            cursor: 0,
        })
    }
    pub(crate) fn step(
        &mut self,
        input: &[f32],
        weights: &[f32],
        out: &mut [f32],
        cancel: &Cancellation,
    ) -> Result<()> {
        if input.len() != self.channels
            || out.len() != self.channels
            || weights.len() != self.history.len()
        {
            return Err(Error::request("causal convolution shape mismatch"));
        }
        cancel.check()?;
        finite(input)?;
        finite(weights)?;
        for (channel, y) in out.iter_mut().enumerate() {
            if channel % 128 == 0 {
                cancel.check()?;
            }
            let start = channel * self.kernel;
            self.history[start + self.cursor] = input[channel];
            let sum = (0..self.kernel)
                .map(|i| {
                    let slot = (self.cursor + 1 + i) % self.kernel;
                    f64::from(self.history[start + slot] * weights[start + i])
                })
                .sum::<f64>() as f32;
            *y = silu(sum);
        }
        self.cursor = (self.cursor + 1) % self.kernel;
        finite(out)
    }
}

/// Gated delta-rule state [value_head, key_dimension, value_dimension].
/// The update decays old memory, writes a gated correction, then reads with q.
pub(crate) struct DeltaState {
    memory: Vec<f32>,
    heads: usize,
    key: usize,
    value: usize,
}
impl DeltaState {
    pub(crate) fn copy_from(&mut self, source: &Self) -> Result<()> {
        if (self.heads, self.key, self.value) != (source.heads, source.key, source.value) {
            return Err(Error::request("recurrent cache shape mismatch"));
        }
        self.memory.copy_from_slice(&source.memory);
        Ok(())
    }

    pub(crate) fn capacity_bytes(&self) -> usize {
        self.memory.capacity() * std::mem::size_of::<f32>()
    }
    pub(crate) fn new(heads: usize, key: usize, value: usize) -> Result<Self> {
        if heads == 0 || heads > 256 || key == 0 || key > 512 || value == 0 || value > 512 {
            return Err(Error::request("invalid recurrent state dimensions"));
        }
        Ok(Self {
            memory: memory::zeros(memory::product(&[heads, key, value])?)?,
            heads,
            key,
            value,
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        log_decay: &[f32],
        beta: &[f32],
        out: &mut [f32],
        cancel: &Cancellation,
    ) -> Result<()> {
        if q.len() != self.heads * self.key
            || k.len() != q.len()
            || v.len() != self.heads * self.value
            || out.len() != v.len()
            || log_decay.len() != self.heads
            || beta.len() != self.heads
        {
            return Err(Error::request("delta-rule shape mismatch"));
        }
        cancel.check()?;
        for values in [q, k, v, log_decay, beta] {
            finite(values)?;
        }
        if log_decay.iter().any(|g| *g > 0.0) || beta.iter().any(|b| !(0.0..=1.0).contains(b)) {
            return Err(Error::request("invalid delta-rule decay/update gate"));
        }
        out.fill(0.0);
        for h in 0..self.heads {
            cancel.check()?;
            let q = &q[h * self.key..(h + 1) * self.key];
            let k = &k[h * self.key..(h + 1) * self.key];
            let q_norm = cpu::dot(q, q);
            let k_norm = cpu::dot(k, k);
            if !q_norm.is_finite() || !k_norm.is_finite() {
                return Err(Error::request("recurrent normalization overflow"));
            }
            let q_scale = (q_norm + 1e-6).sqrt().recip() / (self.key as f32).sqrt();
            let k_scale = (k_norm + 1e-6).sqrt().recip();
            let decay = log_decay[h].exp();
            let head = h * self.key * self.value;
            for j in 0..self.value {
                if j % 32 == 0 {
                    cancel.check()?;
                }
                let mut prediction = 0.0f64;
                for (i, key) in k.iter().enumerate() {
                    let cell = &mut self.memory[head + i * self.value + j];
                    *cell *= decay;
                    prediction += f64::from(*cell * (*key * k_scale));
                }
                let correction = (v[h * self.value + j] - prediction as f32) * beta[h];
                let mut read = 0.0f64;
                for (i, (query, key)) in q.iter().zip(k).enumerate() {
                    let cell = &mut self.memory[head + i * self.value + j];
                    *cell += (*key * k_scale) * correction;
                    read += f64::from(*cell * (*query * q_scale));
                }
                out[h * self.value + j] = read as f32;
            }
        }
        finite(out)?;
        finite(&self.memory)
    }
}

#[cfg(test)]
pub(crate) fn partial_rope(
    values: &mut [f32],
    head: usize,
    rotary: usize,
    position: usize,
    base: f32,
) -> Result<()> {
    partial_mrope(values, head, rotary, [position; 3], base)
}
pub(crate) fn partial_mrope(
    values: &mut [f32],
    head: usize,
    rotary: usize,
    positions: [usize; 3],
    base: f32,
) -> Result<()> {
    if head == 0
        || !values.len().is_multiple_of(head)
        || rotary == 0
        || rotary > head
        || !rotary.is_multiple_of(2)
        || !base.is_finite()
        || base <= 0.0
    {
        return Err(Error::request("invalid partial rotary dimensions"));
    }
    for row in values.chunks_exact_mut(head) {
        for i in 0..rotary / 2 {
            let axis = if positions[0] == positions[1] && positions[1] == positions[2] {
                0
            } else {
                if rotary != 64 {
                    return Err(Error::request("unqualified image rotary sections"));
                }
                if i % 3 == 1 && i < 33 {
                    1
                } else if i % 3 == 2 && i < 30 {
                    2
                } else {
                    0
                }
            };
            let angle = positions[axis] as f32 * base.powf(-((2 * i) as f32) / rotary as f32);
            let (sin, cos) = angle.sin_cos();
            let a = row[i];
            let b = row[i + rotary / 2];
            row[i] = a * cos - b * sin;
            row[i + rotary / 2] = b * cos + a * sin;
        }
    }
    finite(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_delta_vectors_preserve_state_across_tokens_and_cancel() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../fixtures/qwen35/operations-v1.json"
        ))
        .unwrap();
        let mut state = DeltaState::new(2, 3, 2).unwrap();
        let cancel = Cancellation::default();
        for step in fixture["steps"].as_array().unwrap() {
            let get = |key: &str| {
                step[key]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect::<Vec<_>>()
            };
            let mut out = [0.; 4];
            state
                .step(
                    &get("q"),
                    &get("k"),
                    &get("v"),
                    &get("log_decay"),
                    &get("beta"),
                    &mut out,
                    &cancel,
                )
                .unwrap();
            for (actual, expected) in out.iter().zip(get("output")) {
                assert!((*actual - expected).abs() < 2e-6, "{actual} != {expected}");
            }
            for (actual, expected) in state.memory.iter().zip(get("state")) {
                assert!((*actual - expected).abs() < 2e-6);
            }
        }
        let snapshot = state.memory.clone();
        cancel.cancel();
        assert!(
            state
                .step(
                    &[0.; 6],
                    &[0.; 6],
                    &[0.; 4],
                    &[-1.; 2],
                    &[0.5; 2],
                    &mut [0.; 4],
                    &cancel
                )
                .is_err()
        );
        assert_eq!(snapshot, state.memory);
    }
    #[test]
    fn convolution_keeps_raw_inputs_and_stable_gates_cover_extremes() {
        let mut state = ConvState::new(1, 3).unwrap();
        let mut out = [0.];
        let cancel = Cancellation::default();
        for (x, y) in [(1., 3.), (2., 8.), (-1., 2.), (4., 12.)] {
            state.step(&[x], &[1., 2., 3.], &mut out, &cancel).unwrap();
            assert!((out[0] - silu(y)).abs() < 1e-6);
        }
        assert_eq!(sigmoid(1000.), 1.);
        assert_eq!(sigmoid(-1000.), 0.);
        assert_eq!(softplus(1000.), 1000.);
        assert_eq!(softplus(-1000.), 0.);
        let mut out = [0.; 3];
        norm(&[1., 2., 3.], &[-1., 0., 1.], 1e-6, true, &mut out).unwrap();
        assert_eq!(out[0], 0.);
        assert!((out[1] - 2. / (14f32 / 3. + 1e-6).sqrt()).abs() < 1e-6);
        assert!(norm(&[], &[], 1e-6, true, &mut []).is_err());
        assert!(norm(&[f32::MAX], &[1.], 1e-6, false, &mut [0.]).is_err());
        let mut row = [1., 2., 3., 4., 5., 6., 7., 8.];
        partial_rope(&mut row, 8, 4, 0, 10000.).unwrap();
        assert_eq!(row, [1., 2., 3., 4., 5., 6., 7., 8.]);
        partial_rope(&mut row, 8, 4, 3, 10000.).unwrap();
        assert_eq!(&row[4..], &[5., 6., 7., 8.]);
        assert!(partial_rope(&mut row, 8, 3, 0, 10000.).is_err());
    }
}

#[cfg(test)]
mod image_rotary_tests {
    use super::*;
    #[test]
    fn interleaved_spatial_rotary_preserves_nonrotated_tail_and_text_equivalence() {
        let mut text: Vec<f32> = (0..256).map(|i| i as f32 / 256.).collect();
        let mut multi = text.clone();
        partial_rope(&mut text, 256, 64, 7, 1e7).unwrap();
        partial_mrope(&mut multi, 256, 64, [7; 3], 1e7).unwrap();
        assert_eq!(text, multi);
        let mut value = vec![0.; 256];
        value[..32].fill(1.);
        partial_mrope(&mut value, 256, 64, [1, 2, 3], 1e7).unwrap();
        for (j, position) in [(0, 1.), (1, 2.), (2, 3.), (30, 1.), (31, 2.)] {
            let angle = position * 1e7f32.powf(-((2 * j) as f32) / 64.);
            assert!((value[j] - angle.cos()).abs() < 1e-7);
            assert!((value[j + 32] - angle.sin()).abs() < 1e-7);
        }
        assert!(value[64..].iter().all(|v| *v == 0.));
    }
}

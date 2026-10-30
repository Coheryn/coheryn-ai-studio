//! Native packed GGUF CPU operations. No inference runtime or persistent dequantized weights.
//! Layouts follow the pinned GGUF/ggml block contract; independent development fixtures
//! freeze decoding, activation rounding and scalar accumulation before qualification.
use crate::{Error, ErrorCode, Result};
use half::f16;

pub fn supported(dtype: u32) -> bool {
    matches!(dtype, 0 | 1 | 2 | 6 | 8 | 12 | 14)
}
pub fn is_packed(dtype: u32) -> bool {
    matches!(dtype, 2 | 6 | 8 | 12 | 14)
}
pub fn row_bytes(dtype: u32, width: usize) -> Result<usize> {
    let (block, bytes) = match dtype {
        0 => (1, 4),
        1 | 30 => (1, 2),
        2 => (32, 18),
        6 => (32, 22),
        8 => (32, 34),
        12 => (256, 144),
        14 => (256, 210),
        _ => {
            return Err(Error::new(
                ErrorCode::UnsupportedGgufTensorType,
                format!("no native CPU kernel for dtype {dtype}"),
            ));
        }
    };
    if width == 0 || !width.is_multiple_of(block) {
        return Err(Error::invalid("numeric row is not block aligned"));
    }
    (width / block)
        .checked_mul(bytes)
        .ok_or_else(|| Error::resource("packed row byte overflow"))
}
fn half(bytes: &[u8]) -> Result<f32> {
    let value = f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32();
    if value.is_finite() {
        Ok(value)
    } else {
        Err(Error::invalid("non-finite quantization scale"))
    }
}
fn k4_coefficients(scales: &[u8], group: usize) -> (i32, i32) {
    if group < 4 {
        (
            i32::from(scales[group] & 63),
            i32::from(scales[group + 4] & 63),
        )
    } else {
        (
            i32::from((scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4)),
            i32::from((scales[group + 4] >> 4) | ((scales[group] >> 6) << 4)),
        )
    }
}
fn q6_value(block: &[u8], index: usize) -> i32 {
    let half = index / 128;
    let group = (index % 128) / 32;
    let lane = index % 32;
    let low = block[half * 64 + (group % 2) * 32 + lane];
    let low = if group < 2 { low & 15 } else { low >> 4 };
    let high = (block[128 + half * 32 + lane] >> (2 * group)) & 3;
    i32::from(low | (high << 4)) - 32
}
/// Decode only the requested bounded rows (embedding lookup or deep validation).
pub fn decode(bytes: &[u8], dtype: u32, out: &mut [f32]) -> Result<()> {
    if !is_packed(dtype) || row_bytes(dtype, out.len())? != bytes.len() {
        return Err(Error::invalid("packed decode shape mismatch"));
    }
    let (width, size) = if matches!(dtype, 2 | 6 | 8) {
        (
            32,
            match dtype {
                2 => 18,
                6 => 22,
                _ => 34,
            },
        )
    } else {
        (256, if dtype == 12 { 144 } else { 210 })
    };
    for (block, dst) in bytes.chunks_exact(size).zip(out.chunks_exact_mut(width)) {
        if dtype == 14 {
            let d = half(&block[208..])?;
            for (i, value) in dst.iter_mut().enumerate() {
                *value = (d * (block[192 + i / 16] as i8) as f32) * q6_value(block, i) as f32;
            }
        } else {
            let d = half(block)?;
            match dtype {
                8 => {
                    for (value, q) in dst.iter_mut().zip(&block[2..]) {
                        *value = d * (*q as i8) as f32;
                    }
                }
                2 => {
                    for j in 0..16 {
                        dst[j] = d * (i32::from(block[j + 2] & 15) - 8) as f32;
                        dst[j + 16] = d * (i32::from(block[j + 2] >> 4) - 8) as f32;
                    }
                }
                6 => {
                    let high = u32::from_le_bytes(block[2..6].try_into().expect("high bits"));
                    for j in 0..16 {
                        dst[j] = d
                            * ((u32::from(block[6 + j] & 15) | (((high >> j) & 1) << 4)) as i32
                                - 16) as f32;
                        dst[j + 16] = d
                            * ((u32::from(block[6 + j] >> 4) | (((high >> (j + 16)) & 1) << 4))
                                as i32
                                - 16) as f32;
                    }
                }
                12 => {
                    let min = half(&block[2..])?;
                    for group in 0..8 {
                        let (scale, minimum) = k4_coefficients(&block[4..16], group);
                        let ds = d * scale as f32;
                        let dm = min * minimum as f32;
                        for j in 0..32 {
                            let packed = block[16 + (group / 2) * 32 + j];
                            let q = if group % 2 == 0 {
                                packed & 15
                            } else {
                                packed >> 4
                            };
                            dst[group * 32 + j] = ds * f32::from(q) - dm;
                        }
                    }
                }
                _ => unreachable!("validated packed dtype"),
            }
        }
    }
    Ok(())
}

/// Reusable, canonical Q8_0/Q8_K activation blocks, prepared once per matrix input.
/// Both forms occupy O(input dimension) bytes, independent of matrix output rows.
pub struct QuantizedInput {
    packed: Vec<u8>,
    width: usize,
    kind: u32,
    used: usize,
}
impl QuantizedInput {
    pub fn new(max_width: usize) -> Result<Self> {
        let small = max_width
            .div_ceil(32)
            .checked_mul(34)
            .ok_or_else(|| Error::resource("activation size overflow"))?;
        let large = max_width
            .div_ceil(256)
            .checked_mul(292)
            .ok_or_else(|| Error::resource("activation size overflow"))?;
        let mut packed = Vec::new();
        packed
            .try_reserve_exact(small.max(large))
            .map_err(|_| Error::resource("activation buffer allocation failed"))?;
        packed.resize(small.max(large), 0);
        Ok(Self {
            packed,
            width: 0,
            kind: 0,
            used: 0,
        })
    }
    pub fn capacity_bytes(&self) -> usize {
        self.packed.capacity()
    }
    pub fn encoded(&self) -> &[u8] {
        &self.packed[..self.used]
    }
    pub fn prepare(&mut self, input: &[f32], weight_dtype: u32) -> Result<()> {
        self.width = 0;
        self.used = 0;
        let (width, size, kind) = match weight_dtype {
            2 | 6 | 8 => (32, 34, 8),
            12 | 14 => (256, 292, 15),
            _ => return Err(Error::invalid("activation requires packed weight dtype")),
        };
        if input.is_empty()
            || !input.len().is_multiple_of(width)
            || input.iter().any(|v| !v.is_finite())
        {
            return Err(Error::invalid("non-finite or unaligned activation"));
        }
        let used = (input.len() / width)
            .checked_mul(size)
            .filter(|n| *n <= self.packed.len())
            .ok_or_else(|| Error::resource("activation exceeds reserved buffer"))?;
        for (src, dst) in input
            .chunks_exact(width)
            .zip(self.packed[..used].chunks_exact_mut(size))
        {
            let mut maximum = 0f32;
            for &v in src {
                if v.abs() > maximum.abs() {
                    maximum = v;
                }
            }
            if kind == 8 {
                let d = maximum.abs() / 127.;
                let inverse = if d == 0. { 0. } else { 1. / d };
                let stored = f16::from_f32(d);
                if !stored.is_finite() || !inverse.is_finite() {
                    return Err(Error::invalid("Q8_0 activation scale overflow"));
                }
                dst[..2].copy_from_slice(&stored.to_bits().to_le_bytes());
                for (q, x) in dst[2..].iter_mut().zip(src) {
                    *q = ((*x * inverse).round() as i8) as u8;
                }
            } else {
                dst.fill(0);
                if maximum != 0. {
                    let inverse = -127. / maximum;
                    let d = 1. / inverse;
                    if !inverse.is_finite() || !d.is_finite() {
                        return Err(Error::invalid("Q8_K activation scale overflow"));
                    }
                    dst[..4].copy_from_slice(&d.to_le_bytes());
                    for (q, x) in dst[4..260].iter_mut().zip(src) {
                        *q = ((*x * inverse).round_ties_even().min(127.) as i8) as u8;
                    }
                    for group in 0..16 {
                        let sum = dst[4 + group * 16..4 + (group + 1) * 16]
                            .iter()
                            .map(|q| i16::from(*q as i8))
                            .sum::<i16>();
                        dst[260 + group * 2..262 + group * 2].copy_from_slice(&sum.to_le_bytes());
                    }
                }
            }
        }
        self.width = input.len();
        self.used = used;
        self.kind = kind;
        Ok(())
    }
}

/// Packed integer products. The eight scalar K-format lanes preserve the pinned
/// reference reduction order while avoiding full-row/model float materialization.
pub fn dot(row: &[u8], dtype: u32, input: &QuantizedInput) -> Result<f32> {
    let (width, size, kind) = match dtype {
        2 => (32, 18, 8),
        6 => (32, 22, 8),
        8 => (32, 34, 8),
        12 => (256, 144, 15),
        14 => (256, 210, 15),
        _ => return Err(Error::invalid("dot requires packed weights")),
    };
    if input.kind != kind || row_bytes(dtype, input.width)? != row.len() {
        return Err(Error::invalid("packed dot input shape mismatch"));
    }
    let mut sum = 0f32;
    let mut lanes = [0f32; 8];
    let activation_size = if kind == 8 { 34 } else { 292 };
    for (weight, activation) in row
        .chunks_exact(size)
        .zip(input.encoded().chunks_exact(activation_size))
    {
        if width == 32 {
            let dw = half(weight)?;
            let da = half(activation)?;
            let mut integer = 0i32;
            if dtype == 8 {
                for (w, a) in weight[2..].iter().zip(&activation[2..]) {
                    integer += i32::from(*w as i8) * i32::from(*a as i8);
                }
                sum = (integer as f32).mul_add(dw * da, sum);
            } else if dtype == 6 {
                let high = u32::from_le_bytes(weight[2..6].try_into().expect("high bits"));
                for i in 0..16 {
                    let low =
                        (u32::from(weight[6 + i] & 15) | (((high >> i) & 1) << 4)) as i32 - 16;
                    let upper = (u32::from(weight[6 + i] >> 4) | (((high >> (i + 16)) & 1) << 4))
                        as i32
                        - 16;
                    integer += low * i32::from(activation[2 + i] as i8)
                        + upper * i32::from(activation[18 + i] as i8);
                }
                sum = (dw * da).mul_add(integer as f32, sum);
            } else {
                for i in 0..16 {
                    integer +=
                        (i32::from(weight[i + 2] & 15) - 8) * i32::from(activation[2 + i] as i8);
                    integer +=
                        (i32::from(weight[i + 2] >> 4) - 8) * i32::from(activation[18 + i] as i8);
                }
                sum = (integer as f32 * dw).mul_add(da, sum);
            }
        } else {
            let da = f32::from_le_bytes(activation[..4].try_into().expect("scale bytes"));
            let mut integers = [0i32; 8];
            if dtype == 12 {
                let dw = half(weight)?;
                let dm = half(&weight[2..])?;
                let mut minimum_sum = 0i32;
                for group in 0..8 {
                    let (scale, minimum) = k4_coefficients(&weight[4..16], group);
                    let group_sum = i32::from(i16::from_le_bytes([
                        activation[260 + group * 4],
                        activation[261 + group * 4],
                    ])) + i32::from(i16::from_le_bytes([
                        activation[262 + group * 4],
                        activation[263 + group * 4],
                    ]));
                    minimum_sum += minimum * group_sum;
                    for i in 0..32 {
                        let packed = weight[16 + (group / 2) * 32 + i];
                        let q = if group % 2 == 0 {
                            packed & 15
                        } else {
                            packed >> 4
                        };
                        integers[i % 8] +=
                            scale * i32::from(q) * i32::from(activation[4 + group * 32 + i] as i8);
                    }
                }
                for i in 0..8 {
                    lanes[i] = (dw * da).mul_add(integers[i] as f32, lanes[i]);
                }
                sum = (-(dm * da)).mul_add(minimum_sum as f32, sum);
            } else {
                let dw = half(&weight[208..])?;
                for i in 0..256 {
                    integers[i % 8] += q6_value(weight, i)
                        * i32::from(weight[192 + i / 16] as i8)
                        * i32::from(activation[4 + i] as i8);
                }
                for i in 0..8 {
                    lanes[i] = (dw * da).mul_add(integers[i] as f32, lanes[i]);
                }
            }
        }
    }
    if width == 256 {
        for lane in lanes {
            sum += lane;
        }
    }
    if sum.is_finite() {
        Ok(sum)
    } else {
        Err(Error::invalid("non-finite packed matvec result"))
    }
}

//! Bounded native still-image preparation. This does not execute a vision model.
mod header;
mod resample;
use crate::{
    Cancellation, Error, ErrorCode, Result,
    memory::{self, Budget, Reservation},
};
use ::image::ImageDecoder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    io::Cursor,
    ops::{Deref, DerefMut},
};

pub const PROFILE: &str = "coheryn-image-preprocess-v1";
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ImageFormat {
    Png,
    Jpeg,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ImageInfo {
    pub format: ImageFormat,
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub orientation: u8,
    pub encoded_bytes: u64,
}
impl ImageInfo {
    pub fn oriented_dimensions(&self) -> (u32, u32) {
        if self.orientation >= 5 {
            (self.height, self.width)
        } else {
            (self.width, self.height)
        }
    }
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ImageLimits {
    pub max_encoded_bytes: u64,
    pub max_total_encoded_bytes: u64,
    pub max_edge: u32,
    pub max_source_pixels: u64,
    pub max_processed_pixels: u64,
    pub max_image_tokens: usize,
    pub max_total_image_tokens: usize,
    pub max_images: usize,
}
impl Default for ImageLimits {
    fn default() -> Self {
        Self {
            max_encoded_bytes: 16 << 20,
            max_total_encoded_bytes: 32 << 20,
            max_edge: 8192,
            max_source_pixels: 16 << 20,
            max_processed_pixels: 1 << 20,
            max_image_tokens: 1024,
            max_total_image_tokens: 4096,
            max_images: 4,
        }
    }
}
impl ImageLimits {
    pub fn validate(&self) -> Result<()> {
        if self.max_encoded_bytes == 0
            || self.max_encoded_bytes > 64 << 20
            || self.max_total_encoded_bytes == 0
            || self.max_total_encoded_bytes > 128 << 20
            || self.max_edge == 0
            || self.max_edge > 16384
            || self.max_source_pixels == 0
            || self.max_source_pixels > 64 << 20
            || self.max_processed_pixels == 0
            || self.max_processed_pixels > 16 << 20
            || self.max_image_tokens == 0
            || self.max_image_tokens > 16384
            || self.max_total_image_tokens == 0
            || self.max_total_image_tokens > 262144
            || self.max_images == 0
            || self.max_images > 16
        {
            return Err(Error::request("invalid finite image limits"));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Processor {
    #[serde(rename = "llava15-pad336-v1")]
    Llava15Pad336,
    #[serde(rename = "qwen2vl-still-v1")]
    Qwen2VlStill,
    #[serde(rename = "qwen35-still-v1")]
    Qwen35Still,
}
impl Processor {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "llava15-pad336-v1" => Ok(Self::Llava15Pad336),
            "qwen2vl-still-v1" => Ok(Self::Qwen2VlStill),
            "qwen35-still-v1" => Ok(Self::Qwen35Still),
            _ => Err(Error::new(
                ErrorCode::UnsupportedImageProcessor,
                "unknown qualified preprocessing profile",
            )),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Llava15Pad336 => "llava15-pad336-v1",
            Self::Qwen2VlStill => "qwen2vl-still-v1",
            Self::Qwen35Still => "qwen35-still-v1",
        }
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct ImagePlan {
    pub profile: Processor,
    pub info: ImageInfo,
    pub width: u32,
    pub height: u32,
    pub resized_width: u32,
    pub resized_height: u32,
    pub grid: [usize; 3],
    pub patch_rows: usize,
    pub patch_features: usize,
    pub image_feature_tokens: usize,
    pub decoder_stage_bytes: u64,
    pub resample_stage_bytes: u64,
    pub normalization_stage_bytes: u64,
    pub peak_reserved_bytes: u64,
    pub output_reserved_bytes: u64,
}
fn limit(message: &str) -> Error {
    Error::new(ErrorCode::ImageInputLimitExceeded, message)
}
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| Error::resource("image byte arithmetic overflow"))
}
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .ok_or_else(|| Error::resource("image byte arithmetic overflow"))
}
fn bytes(n: usize, size: usize) -> Result<u64> {
    Ok(memory::product(&[n, size])? as u64)
}
fn mean() -> [f32; 3] {
    [0.48145466, 0.4578275, 0.40821073]
}
fn deviation() -> [f32; 3] {
    [0.26862954, 0.261_302_6, 0.275_777_1]
}

/// Structural preflight includes complete container ranges and orientation, not pixel decoding.
pub fn inspect(encoded: &[u8], limits: ImageLimits, cancel: &Cancellation) -> Result<ImageInfo> {
    header::inspect(encoded, &limits, cancel)
}
/// Validate source geometry independently of any model's resize/patch profile.
pub fn validate_info(info: ImageInfo, limits: ImageLimits) -> Result<()> {
    limits.validate()?;
    if info.width == 0
        || info.height == 0
        || info.width > limits.max_edge
        || info.height > limits.max_edge
        || !matches!(info.channels, 1..=4)
        || !(1..=8).contains(&info.orientation)
        || info.encoded_bytes == 0
        || info.encoded_bytes > limits.max_encoded_bytes
        || mul(info.width as u64, info.height as u64)? > limits.max_source_pixels
    {
        return Err(limit("invalid or oversized image descriptor"));
    }
    Ok(())
}
pub fn plan(info: ImageInfo, profile: Processor, limits: ImageLimits) -> Result<ImagePlan> {
    limits.validate()?;
    validate_info(info, limits)?;
    let (sw, sh) = info.oriented_dimensions();
    let (width, height, rw, rh) = match profile {
        Processor::Llava15Pad336 => {
            let scale = (336.0f32 / sw as f32).min(336.0 / sh as f32);
            (
                336,
                336,
                ((sw as f32 * scale).ceil() as u32).clamp(1, 336),
                ((sh as f32 * scale).ceil() as u32).clamp(1, 336),
            )
        }
        Processor::Qwen2VlStill => {
            if sw.max(sh) > sw.min(sh) * 200 {
                return Err(limit("Qwen still-image aspect ratio exceeds 200"));
            }
            let aligned = |v: f32| -> u32 { (v / 28.).round_ties_even() as u32 * 28 };
            let mut w = aligned(sw as f32).max(28);
            let mut h = aligned(sh as f32).max(28);
            if u64::from(w) * u64::from(h) > 12845056 {
                let ratio = (sw as f32 * sh as f32 / 12845056.).sqrt();
                w = ((sw as f32 / ratio / 28.).floor() as u32 * 28).max(28);
                h = ((sh as f32 / ratio / 28.).floor() as u32 * 28).max(28);
            } else if u64::from(w) * u64::from(h) < 3136 {
                let ratio = (3136.0 / (sw as f32 * sh as f32)).sqrt();
                w = (sw as f32 * ratio / 28.).ceil() as u32 * 28;
                h = (sh as f32 * ratio / 28.).ceil() as u32 * 28;
            }
            (w, h, w, h)
        }
        Processor::Qwen35Still => {
            if sw.max(sh) > sw.min(sh) * 200 {
                return Err(limit("Qwen still-image aspect ratio exceeds 200"));
            }
            let (factor, minimum, maximum) = if profile == Processor::Qwen35Still {
                (32., 65536., 16777216.)
            } else {
                (28., 3136., 12845056.)
            };
            let mut w =
                ((sw as f64 / factor).round_ties_even() as u32 * factor as u32).max(factor as u32);
            let mut h =
                ((sh as f64 / factor).round_ties_even() as u32 * factor as u32).max(factor as u32);
            if u64::from(w) * u64::from(h) > maximum as u64 {
                let ratio = (sw as f64 * sh as f64 / maximum).sqrt();
                w = ((sw as f64 / ratio / factor).floor() as u32 * factor as u32)
                    .max(factor as u32);
                h = ((sh as f64 / ratio / factor).floor() as u32 * factor as u32)
                    .max(factor as u32);
            } else if u64::from(w) * u64::from(h) < minimum as u64 {
                let ratio = (minimum / (sw as f64 * sh as f64)).sqrt();
                w = (sw as f64 * ratio / factor).ceil() as u32 * factor as u32;
                h = (sh as f64 * ratio / factor).ceil() as u32 * factor as u32;
            }
            (w, h, w, h)
        }
    };
    let pixels = mul(width as u64, height as u64)?;
    if width == 0
        || height == 0
        || width > 32768
        || height > 32768
        || pixels > limits.max_processed_pixels
    {
        return Err(limit(
            "canonical processor output exceeds pixel limits; no automatic quality reduction",
        ));
    }
    let patch = if profile == Processor::Qwen35Still {
        16
    } else {
        14
    };
    let grid = [1, (height / patch) as usize, (width / patch) as usize];
    let patches = memory::product(&grid)?;
    let tokens = if profile != Processor::Llava15Pad336 {
        patches / 4
    } else {
        patches
    };
    if tokens > limits.max_image_tokens || tokens > limits.max_total_image_tokens {
        return Err(limit(
            "canonical image feature count exceeds configured limit",
        ));
    }
    let raw = mul(
        mul(info.width as u64, info.height as u64)?,
        info.channels as u64,
    )?;
    // Width/height and format bounds are checked independently of best-effort codec
    // allocation limits. The decoder allowance covers raw data, coefficients and metadata.
    let decoder = add(add(mul(raw, 5)?, mul(info.encoded_bytes, 3)?)?, 1 << 20)?;
    let normalized = mul(pixels, 12)?;
    let rgb = mul(pixels, 3)?;
    let axes = resample::axis_bytes(sw as usize, rw as usize)?
        .checked_add(resample::axis_bytes(sh as usize, rh as usize)?)
        .ok_or_else(|| Error::resource("filter reservation overflow"))?;
    let taps = resample::maximum_taps(sh as usize, rh as usize)?;
    let ring = add(
        mul(mul(rw as u64, 3)?, taps as u64)?,
        bytes(taps, std::mem::size_of::<usize>())?,
    )?;
    let resample = add(add(add(add(info.encoded_bytes, raw)?, rgb)?, axes)?, ring)?;
    let normalization = add(add(info.encoded_bytes, rgb)?, normalized)?;
    let peak = add(decoder.max(resample).max(normalization), 4096)?;
    Ok(ImagePlan {
        profile,
        info,
        width,
        height,
        resized_width: rw,
        resized_height: rh,
        grid,
        patch_rows: patches,
        patch_features: patch as usize
            * patch as usize
            * 3
            * if profile == Processor::Llava15Pad336 {
                1
            } else {
                2
            },
        image_feature_tokens: tokens,
        decoder_stage_bytes: decoder,
        resample_stage_bytes: resample,
        normalization_stage_bytes: normalization,
        peak_reserved_bytes: peak,
        output_reserved_bytes: add(normalized, 4096)?,
    })
}

struct Buffer<T> {
    values: Vec<T>,
    _observed: memory::BufferAccounting,
}
impl<T: Clone> Buffer<T> {
    fn filled(count: usize, value: T, budget: &Budget) -> Result<Self> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| Error::resource("image buffer allocation failed"))?;
        values.resize(count, value);
        let actual = bytes(values.capacity(), std::mem::size_of::<T>())?;
        let mut observed = budget.track_buffers(actual);
        observed.grow(actual)?;
        Ok(Self {
            values,
            _observed: observed,
        })
    }
}
impl<T> Deref for Buffer<T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        &self.values
    }
}
impl<T> DerefMut for Buffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.values
    }
}

pub struct PreparedImage {
    plan: ImagePlan,
    processed_rgb_sha256: String,
    normalized: Buffer<f32>,
    _reservation: Reservation,
}
impl std::fmt::Debug for PreparedImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedImage")
            .field("plan", &self.plan)
            .finish_non_exhaustive()
    }
}
impl PreparedImage {
    pub(crate) fn admitted_to(&self, budget: &Budget) -> bool {
        self._reservation
            .covers(budget, self.plan.output_reserved_bytes)
    }
    pub(crate) fn recycle(self, held: &mut Reservation) -> Result<()> {
        let Self {
            normalized,
            _reservation,
            ..
        } = self;
        drop(normalized);
        held.absorb(_reservation)
    }
    pub fn plan(&self) -> &ImagePlan {
        &self.plan
    }
    pub fn processed_rgb_sha256(&self) -> &str {
        &self.processed_rgb_sha256
    }
    pub fn normalized_hwc(&self) -> &[f32] {
        &self.normalized
    }
    /// Logical patch rows without copying RGB planes or repeating the still frame.
    pub fn patch_values(&self, row: usize) -> Result<impl Iterator<Item = f32> + '_> {
        if row >= self.plan.patch_rows {
            return Err(Error::request("image patch index outside plan"));
        }
        let merged = self.plan.profile != Processor::Llava15Pad336;
        let patch = if self.plan.profile == Processor::Qwen35Still {
            16
        } else {
            14
        };
        let (py, px) = if merged {
            let group = row / 4;
            let groups_w = self.plan.grid[2] / 2;
            (
                group / groups_w * 2 + row % 4 / 2,
                group % groups_w * 2 + row % 2,
            )
        } else {
            (row / self.plan.grid[2], row % self.plan.grid[2])
        };
        let time = if merged { 2 } else { 1 };
        let width = self.plan.width as usize;
        Ok((0..self.plan.patch_features).map(move |i| {
            let channel = i / (time * patch * patch);
            let y = i / patch % patch;
            let x = i % patch;
            self.normalized[((py * patch + y) * width + px * patch + x) * 3 + channel]
        }))
    }
}
struct Source<'a> {
    raw: &'a [u8],
    info: ImageInfo,
}
impl Source<'_> {
    fn pixel(&self, x: usize, y: usize, channel: usize) -> u8 {
        let w = self.info.width as usize;
        let h = self.info.height as usize;
        let (x, y) = match self.info.orientation {
            1 => (x, y),
            2 => (w - 1 - x, y),
            3 => (w - 1 - x, h - 1 - y),
            4 => (x, h - 1 - y),
            5 => (y, x),
            6 => (y, h - 1 - x),
            7 => (w - 1 - y, h - 1 - x),
            8 => (w - 1 - y, x),
            _ => unreachable!("validated orientation"),
        };
        self.raw[(y * w + x) * self.info.channels as usize
            + if self.info.channels <= 2 { 0 } else { channel }]
    }
}
fn codec_error(error: ::image::ImageError) -> Error {
    let code = match &error {
        ::image::ImageError::Limits(_) => ErrorCode::ResourceExhausted,
        ::image::ImageError::Unsupported(_) => ErrorCode::UnsupportedImageFormat,
        ::image::ImageError::IoError(io) if io.kind() == std::io::ErrorKind::OutOfMemory => {
            ErrorCode::ResourceExhausted
        }
        _ => ErrorCode::InvalidImage,
    };
    Error::new(code, format!("image codec: {error}"))
}
fn decode_pixels(
    encoded: &[u8],
    info: ImageInfo,
    allowance: u64,
    budget: &Budget,
    cancel: &Cancellation,
) -> Result<Buffer<u8>> {
    let mut reader = ::image::ImageReader::with_format(
        Cursor::new(encoded),
        match info.format {
            ImageFormat::Png => ::image::ImageFormat::Png,
            ImageFormat::Jpeg => ::image::ImageFormat::Jpeg,
        },
    );
    let mut limits = ::image::Limits::default();
    limits.max_image_width = Some(info.width);
    limits.max_image_height = Some(info.height);
    limits.max_alloc = Some(allowance);
    reader.limits(limits);
    let decoder = reader.into_decoder().map_err(codec_error)?;
    let length = memory::product(&[
        info.width as usize,
        info.height as usize,
        info.channels as usize,
    ])?;
    if decoder.dimensions() != (info.width, info.height) || decoder.total_bytes() != length as u64 {
        return Err(Error::new(
            ErrorCode::InvalidImage,
            "decoder disagrees with checked image descriptor",
        ));
    }
    let mut raw = Buffer::filled(length, 0u8, budget)?;
    // One bounded baseline-image decode is a codec unit. Cancellation is checked
    // before/after it; resources stay admitted until the decoder acknowledges return.
    decoder.read_image(&mut raw).map_err(codec_error)?;
    cancel.check()?;
    Ok(raw)
}

pub struct DecodedImage {
    info: ImageInfo,
    raw: Buffer<u8>,
    budget: Budget,
    _reservation: Reservation,
}
impl std::fmt::Debug for DecodedImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecodedImage")
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}
impl DecodedImage {
    pub fn info(&self) -> ImageInfo {
        self.info
    }
    pub fn pixels_rgb(&self) -> impl Iterator<Item = u8> + '_ {
        let (width, height) = self.info.oriented_dimensions();
        let source = Source {
            raw: &self.raw,
            info: self.info,
        };
        (0..width as usize * height as usize * 3)
            .map(move |i| source.pixel(i / 3 % width as usize, i / 3 / width as usize, i % 3))
    }
}
pub fn decode(
    encoded: &[u8],
    limits: ImageLimits,
    budget: &Budget,
    cancel: &Cancellation,
) -> Result<DecodedImage> {
    let info = inspect(encoded, limits, cancel)?;
    let raw_bytes = mul(
        mul(info.width as u64, info.height as u64)?,
        info.channels as u64,
    )?;
    let allowance = add(
        add(mul(raw_bytes, 5)?, mul(info.encoded_bytes, 3)?)?,
        1 << 20,
    )?;
    let mut held = budget.reserve(add(allowance, 4096)?)?;
    let raw = decode_pixels(encoded, info, allowance, budget, cancel)?;
    let reservation = held.split(add(bytes(raw.values.capacity(), 1)?, 4096)?)?;
    Ok(DecodedImage {
        info,
        raw,
        budget: budget.clone(),
        _reservation: reservation,
    })
}

fn finish_pixels(
    rgb: Buffer<u8>,
    plan: ImagePlan,
    budget: &Budget,
    held: &mut Reservation,
    cancel: &Cancellation,
) -> Result<PreparedImage> {
    let mut hash = Sha256::new();
    for chunk in rgb.chunks(4096) {
        cancel.check()?;
        hash.update(chunk);
    }
    let digest = crate::storage::digest_hex(hash.finalize().into());
    let mut normalized = Buffer::filled(rgb.len(), 0.0f32, budget)?;
    let mean = if plan.profile == Processor::Qwen35Still {
        [0.5; 3]
    } else {
        mean()
    };
    let deviation = if plan.profile == Processor::Qwen35Still {
        [0.5; 3]
    } else {
        deviation()
    };
    for (i, (pixel, value)) in rgb.iter().zip(normalized.iter_mut()).enumerate() {
        if i % 4096 == 0 {
            cancel.check()?;
        }
        *value = (*pixel as f32 / 255. - mean[i % 3]) / deviation[i % 3];
    }
    drop(rgb);
    cancel.check()?;
    let output_bytes = add(bytes(normalized.values.capacity(), 4)?, 4096)?;
    Ok(PreparedImage {
        plan,
        processed_rgb_sha256: digest,
        normalized,
        _reservation: held.split(output_bytes)?,
    })
}

/// Process already-decoded pixels without a second codec pass or RGB copy.
pub fn prepare_decoded(
    decoded: &DecodedImage,
    profile: Processor,
    limits: ImageLimits,
    cancel: &Cancellation,
) -> Result<PreparedImage> {
    cancel.check()?;
    let p = plan(decoded.info, profile, limits)?;
    let raw = bytes(decoded.raw.values.capacity(), 1)?;
    let resampling = p
        .resample_stage_bytes
        .checked_sub(decoded.info.encoded_bytes)
        .and_then(|n| n.checked_sub(raw))
        .ok_or_else(|| Error::resource("invalid decoded-image capacity"))?;
    let normalization = p
        .normalization_stage_bytes
        .checked_sub(decoded.info.encoded_bytes)
        .ok_or_else(|| Error::resource("invalid normalization capacity"))?;
    let mut held = decoded
        .budget
        .reserve(add(resampling.max(normalization), 4096)?)?;
    let rgb = resample::run(
        Source {
            raw: &decoded.raw,
            info: decoded.info,
        },
        &p,
        &decoded.budget,
        cancel,
    )?;
    finish_pixels(rgb, p, &decoded.budget, &mut held, cancel)
}

pub(crate) fn prepare_admitted(
    encoded: &[u8],
    plan: ImagePlan,
    budget: &Budget,
    held: &mut Reservation,
    cancel: &Cancellation,
) -> Result<PreparedImage> {
    cancel.check()?;
    if !held.covers(budget, plan.peak_reserved_bytes) {
        return Err(Error::resource("invalid image stage admission"));
    }
    let raw = decode_pixels(encoded, plan.info, plan.decoder_stage_bytes, budget, cancel)?;
    let rgb = resample::run(
        Source {
            raw: &raw,
            info: plan.info,
        },
        &plan,
        budget,
        cancel,
    )?;
    drop(raw);
    finish_pixels(rgb, plan, budget, held, cancel)
}
/// Admit the complete ordered image stage atomically, then process sequentially.
/// This is processor admission only: vision encoder/projector/LM costs remain separate.
pub fn prepare_many(
    inputs: &[&[u8]],
    profile: Processor,
    limits: ImageLimits,
    budget: &Budget,
    cancel: &Cancellation,
) -> Result<Vec<PreparedImage>> {
    limits.validate()?;
    cancel.check()?;
    if inputs.is_empty() || inputs.len() > limits.max_images {
        return Err(limit("image count exceeds configured limit"));
    }
    let mut plans = Vec::with_capacity(inputs.len());
    let encoded = inputs
        .iter()
        .try_fold(0, |sum, input| add(sum, input.len() as u64))?;
    if encoded > limits.max_total_encoded_bytes {
        return Err(limit("aggregate image byte limit exceeded"));
    }
    let mut retained = 0;
    let mut peak = 0;
    let mut tokens = 0usize;
    for input in inputs {
        let info = inspect(input, limits, cancel)?;
        let p = plan(info, profile, limits)?;
        tokens = tokens
            .checked_add(p.image_feature_tokens)
            .ok_or_else(|| limit("image feature count overflow"))?;
        if tokens > limits.max_total_image_tokens {
            return Err(limit("aggregate image feature limit exceeded"));
        }
        peak = peak.max(add(retained, p.peak_reserved_bytes - info.encoded_bytes)?);
        retained = add(retained, p.output_reserved_bytes)?;
        plans.push(p);
    }
    let mut held = budget.reserve(add(encoded, peak)?)?;
    let mut prepared = Vec::with_capacity(inputs.len());
    for (input, p) in inputs.iter().zip(plans) {
        prepared.push(prepare_admitted(input, p, budget, &mut held, cancel)?);
    }
    Ok(prepared)
}
pub fn prepare(
    encoded: &[u8],
    profile: Processor,
    limits: ImageLimits,
    budget: &Budget,
    cancel: &Cancellation,
) -> Result<PreparedImage> {
    Ok(prepare_many(&[encoded], profile, limits, budget, cancel)?.remove(0))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_eight_orientations_map_asymmetric_source_without_copying() {
        let raw = [1, 2, 3, 4, 5, 6];
        let expected: [&[u8]; 8] = [
            &[1, 2, 3, 4, 5, 6],
            &[2, 1, 4, 3, 6, 5],
            &[6, 5, 4, 3, 2, 1],
            &[5, 6, 3, 4, 1, 2],
            &[1, 3, 5, 2, 4, 6],
            &[5, 3, 1, 6, 4, 2],
            &[6, 4, 2, 5, 3, 1],
            &[2, 4, 6, 1, 3, 5],
        ];
        for orientation in 1..=8 {
            let info = ImageInfo {
                format: ImageFormat::Png,
                width: 2,
                height: 3,
                channels: 1,
                orientation,
                encoded_bytes: 1,
            };
            let source = Source { raw: &raw, info };
            let (width, height) = info.oriented_dimensions();
            let mut actual = Vec::new();
            for y in 0..height as usize {
                for x in 0..width as usize {
                    for channel in 0..3 {
                        assert_eq!(source.pixel(x, y, channel), source.pixel(x, y, 0));
                    }
                    actual.push(source.pixel(x, y, 0));
                }
            }
            assert_eq!(actual, expected[orientation as usize - 1]);
        }
    }
}

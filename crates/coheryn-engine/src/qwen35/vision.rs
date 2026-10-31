//! Native still-image encoder for the validated Qwen3.5 source component.
use super::*;
use crate::{
    cpu,
    memory::{self, BufferAccounting},
    multimodal::{
        image::{PreparedImage, Processor},
        vision::kernels as spatial,
    },
    safetensors::BoundTensor,
    tensor::{self, Workspace},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const PROFILE: &str = "qwen35-still-cpu-v1";
const METADATA: u64 = 8 << 20;
#[derive(Clone, Serialize, Deserialize)]
pub struct VisionConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub layers: usize,
    pub output: usize,
    pub positions: usize,
}
fn bad(message: &str) -> Error {
    Error::new(ErrorCode::UnsupportedImageProcessor, message)
}
fn integer(value: &serde_json::Value, name: &str, max: usize) -> Result<usize> {
    value[name]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .filter(|n| *n > 0 && *n <= max)
        .ok_or_else(|| bad("invalid vision dimension"))
}
impl VisionConfig {
    pub fn parse(root: &serde_json::Value, processor: &[u8], set: &TensorSet) -> Result<Self> {
        let v = &root["vision_config"];
        let config = Self {
            hidden: integer(v, "hidden_size", 2048)?,
            intermediate: integer(v, "intermediate_size", 8192)?,
            heads: integer(v, "num_heads", 32)?,
            layers: integer(v, "depth", 64)?,
            output: integer(v, "out_hidden_size", 8192)?,
            positions: integer(v, "num_position_embeddings", 16384)?,
        };
        let side = config.positions.isqrt();
        let activation = v["hidden_act"]
            .as_str()
            .ok_or_else(|| bad("missing vision activation"))?;
        // The source activation name is normalized to our native tanh-GELU profile.
        let expected = crate::storage::digest_hex(Sha256::digest(activation.as_bytes()).into());
        if root["text_config"]["rope_parameters"]["mrope_interleaved"] != true
            || root["text_config"]["rope_parameters"]["mrope_section"]
                != serde_json::json!([11, 11, 10])
            || !config.hidden.is_multiple_of(config.heads)
            || !(config.hidden / config.heads).is_multiple_of(4)
            || side * side != config.positions
            || v["patch_size"] != 16
            || v["temporal_patch_size"] != 2
            || v["spatial_merge_size"] != 2
            || v["in_channels"] != 3
            || v["deepstack_visual_indexes"]
                .as_array()
                .is_none_or(|a| !a.is_empty())
            || expected != ACTIVATION_HASH
            || root["text_config"]["hidden_size"].as_u64() != Some(config.output as u64)
            || root["image_token_id"] != 248056
        {
            return Err(bad("unsupported Qwen3.5 still-image component/profile"));
        }
        let p = crate::safetensors::configuration(processor)?;
        if p["patch_size"] != 16
            || p["temporal_patch_size"] != 2
            || p["merge_size"] != 2
            || p["size"]["shortest_edge"] != 65536
            || p["size"]["longest_edge"] != 16777216
            || p["image_mean"] != serde_json::json!([0.5, 0.5, 0.5])
            || p["image_std"] != serde_json::json!([0.5, 0.5, 0.5])
        {
            return Err(bad("unsupported image resize/normalization profile"));
        }
        let mut names = BTreeSet::new();
        let mut require = |name: String, shape: &[usize]| -> Result<()> {
            let t = set
                .tensor(&name)
                .ok_or_else(|| bad("missing required vision tensor"))?;
            if !matches!(t.dtype, 0 | 1 | 30)
                || t.dimensions.iter().rev().copied().collect::<Vec<_>>()
                    != shape.iter().map(|n| *n as u64).collect::<Vec<_>>()
            {
                return Err(bad("vision tensor shape/encoding mismatch"));
            }
            names.insert(name);
            Ok(())
        };
        let h = config.hidden;
        let f = config.intermediate;
        let o = config.output;
        require(
            "model.visual.patch_embed.proj.weight".into(),
            &[h, 3, 2, 16, 16],
        )?;
        require("model.visual.patch_embed.proj.bias".into(), &[h])?;
        require(
            "model.visual.pos_embed.weight".into(),
            &[config.positions, h],
        )?;
        for i in 0..config.layers {
            let prefix = format!("model.visual.blocks.{i}");
            for n in ["norm1", "norm2"] {
                for suffix in ["weight", "bias"] {
                    require(format!("{prefix}.{n}.{suffix}"), &[h])?;
                }
            }
            for (n, width, rows) in [
                ("attn.qkv", h, 3 * h),
                ("attn.proj", h, h),
                ("mlp.linear_fc1", h, f),
                ("mlp.linear_fc2", f, h),
            ] {
                require(format!("{prefix}.{n}.weight"), &[rows, width])?;
                require(format!("{prefix}.{n}.bias"), &[rows])?;
            }
        }
        for suffix in ["weight", "bias"] {
            require(format!("model.visual.merger.norm.{suffix}"), &[h])?;
        }
        for (n, width, rows) in [("linear_fc1", 4 * h, 4 * h), ("linear_fc2", 4 * h, o)] {
            require(format!("model.visual.merger.{n}.weight"), &[rows, width])?;
            require(format!("model.visual.merger.{n}.bias"), &[rows])?;
        }
        if set
            .names()
            .filter(|n| n.starts_with("model.visual."))
            .any(|n| !names.contains(n))
        {
            return Err(bad("unrecognized vision tensor"));
        }
        Ok(config)
    }
    pub fn estimate(&self, patches: usize, threads: usize) -> Result<u64> {
        if !(4..=4096).contains(&patches) || !patches.is_multiple_of(4) {
            return Err(Error::resource(
                "vision patch count must be 4..=4096 in merge groups",
            ));
        }
        let floats = memory::product(&[
            patches,
            20 * self.hidden + 2 * self.intermediate + self.output,
        ])? as u64;
        // Norm and bias vectors are request-owned, separate from tiled weights.
        // Charge them explicitly so deeper compatible configurations cannot consume
        // the fixed workspace/metadata allowance before their buffers are created.
        let parameters = memory::product(&[self.layers, 9 * self.hidden + self.intermediate, 4])?
            as u64
            + memory::product(&[7 * self.hidden + self.output, 4])? as u64;
        floats
            .checked_mul(4)
            .and_then(|n| n.checked_add(parameters))
            .and_then(|n| n.checked_add(METADATA))
            .and_then(|n| n.checked_add(memory::worker_bytes(threads).ok()? as u64))
            .ok_or_else(|| Error::resource("vision capacity overflow"))
    }
}
// Replaced from the pinned source activation identifier during source preparation.
const ACTIVATION_HASH: &str = "0da9aaf09927b6504c5844992570291ff221b3db243c4a4c609a74972c3625f4";
struct Linear {
    weight: BoundTensor,
    bias: Vec<f32>,
    width: usize,
    rows: usize,
}
impl Linear {
    fn new(set: &TensorSet, name: &str, width: usize, rows: usize) -> Result<Self> {
        Ok(Self {
            weight: set.bind(&format!("{name}.weight"))?,
            bias: set.read_vector(&format!("{name}.bias"))?,
            width,
            rows,
        })
    }
    fn apply(
        &self,
        set: &TensorSet,
        input: &[f32],
        out: &mut [f32],
        work: &mut Workspace,
        cancel: &Cancellation,
        cooperate: &mut Option<&mut crate::multimodal::WorkYield<'_>>,
    ) -> Result<()> {
        let (storage, t) = set.bound(&self.weight)?;
        let mut t = t.clone();
        t.dimensions = vec![self.width as u64, self.rows as u64];
        tensor::matmul_cooperative(storage, 0, &t, input, out, work, cancel, cooperate)
            .map_err(|error| set.source_error(error))?;
        for row in out.chunks_exact_mut(self.rows) {
            cancel.check()?;
            for (v, b) in row.iter_mut().zip(&self.bias) {
                *v += b;
            }
        }
        Ok(())
    }
}
struct Norm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}
impl Norm {
    fn new(set: &TensorSet, name: &str) -> Result<Self> {
        Ok(Self {
            weight: set.read_vector(&format!("{name}.weight"))?,
            bias: set.read_vector(&format!("{name}.bias"))?,
        })
    }
    fn apply(&self, input: &[f32], out: &mut [f32], cancel: &Cancellation) -> Result<()> {
        out.copy_from_slice(input);
        for row in out.chunks_exact_mut(self.weight.len()) {
            cancel.check()?;
            cpu::layer_norm(row, &self.weight, &self.bias, 1e-6)?;
        }
        Ok(())
    }
}
struct Block {
    n1: Norm,
    n2: Norm,
    qkv: Linear,
    out: Linear,
    up: Linear,
    down: Linear,
}
pub struct VisionResult {
    pub values: Vec<f32>,
    pub grid: [usize; 3],
    pub encoding_ms: f64,
    _buffers: BufferAccounting,
    _reservation: Reservation,
}
pub struct Checkpoint<'a> {
    pub name: &'a str,
    pub rows: usize,
    pub width: usize,
    pub values: &'a [f32],
}
pub type Observer<'a> = dyn FnMut(Checkpoint<'_>) -> Result<()> + 'a;
fn gelu_exact(values: &mut [f32], cancel: &Cancellation) -> Result<()> {
    for chunk in values.chunks_mut(4096) {
        cancel.check()?;
        for v in chunk {
            let x = f64::from(*v);
            let z = (x / std::f64::consts::SQRT_2).abs();
            let t = 1. / (1. + 0.3275911 * z);
            let poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736)
                * t
                + 0.254829592)
                * t;
            let erf = (1. - poly * (-z * z).exp()).copysign(x);
            *v = (0.5 * x * (1. + erf)) as f32;
            if !v.is_finite() {
                return Err(Error::invalid("nonfinite vision GELU"));
            }
        }
    }
    Ok(())
}
#[allow(clippy::too_many_arguments)]
pub(super) fn encode(
    model: &Model,
    image: &PreparedImage,
    config: &VisionConfig,
    threads: usize,
    cancel: &Cancellation,
    mut cooperate: Option<&mut crate::multimodal::WorkYield<'_>>,
    held: Option<Reservation>,
    mut observer: Option<&mut Observer<'_>>,
) -> Result<VisionResult> {
    if image.plan().profile != Processor::Qwen35Still {
        return Err(bad("vision encoder requires its qualified processor"));
    }
    let plan = image.plan();
    let n = plan.patch_rows;
    let h = config.hidden;
    let f = config.intermediate;
    let out_tokens = n / 4;
    let held = if let Some(held) = held {
        if !held.covers(&model.budget, config.estimate(n, threads)?) {
            return Err(Error::resource("invalid vision admission"));
        }
        held
    } else {
        model.budget.reserve(config.estimate(n, threads)?)?
    };
    let mut accounting = model.budget.track_buffers(held.bytes());
    let start = Instant::now();
    let mut work = Workspace::new((4 * h).max(f).max(1536), 256 << 10, threads, 32)?;
    work.preserve_f32_operands();
    let patch = Linear::new(&model.set, "model.visual.patch_embed.proj", 1536, h)?;
    let pos = model.set.bind("model.visual.pos_embed.weight")?;
    let mut blocks = Vec::new();
    for i in 0..config.layers {
        let p = format!("model.visual.blocks.{i}");
        blocks.push(Block {
            n1: Norm::new(&model.set, &format!("{p}.norm1"))?,
            n2: Norm::new(&model.set, &format!("{p}.norm2"))?,
            qkv: Linear::new(&model.set, &format!("{p}.attn.qkv"), h, 3 * h)?,
            out: Linear::new(&model.set, &format!("{p}.attn.proj"), h, h)?,
            up: Linear::new(&model.set, &format!("{p}.mlp.linear_fc1"), h, f)?,
            down: Linear::new(&model.set, &format!("{p}.mlp.linear_fc2"), f, h)?,
        });
    }
    let merger_norm = Norm::new(&model.set, "model.visual.merger.norm")?;
    let merger_a = Linear::new(&model.set, "model.visual.merger.linear_fc1", 4 * h, 4 * h)?;
    let merger_b = Linear::new(
        &model.set,
        "model.visual.merger.linear_fc2",
        4 * h,
        config.output,
    )?;
    let mut x = memory::zeros(n * h)?;
    let mut normalized = memory::zeros(n * h)?;
    let mut qkv = memory::zeros(n * 3 * h)?;
    let mut q = memory::zeros(n * h)?;
    let mut k = memory::zeros(n * h)?;
    let mut v = memory::zeros(n * h)?;
    let mut attended = memory::zeros(n * h)?;
    let mut projected = memory::zeros(n * h)?;
    let mut mlp = memory::zeros(n * f.max(h))?;
    let mut patches = memory::zeros(32 * 1536)?;
    let mut position_row = memory::zeros(h)?;
    let mut position_sum = memory::zeros_f64(h)?;
    let mut scratch: Vec<_> = (0..threads)
        .map(|_| spatial::AttentionScratch::new(n, h / config.heads))
        .collect::<Result<_>>()?;
    let mut merged = memory::zeros(n * h)?;
    let mut output = memory::zeros(out_tokens * config.output)?;
    // The conservative allocation reservation includes bounded parameter vectors and worker overhead.
    let arrays = [
        &x,
        &normalized,
        &qkv,
        &q,
        &k,
        &v,
        &attended,
        &projected,
        &mlp,
        &patches,
        &position_row,
        &merged,
        &output,
    ];
    let parameters = patch.bias.capacity()
        + merger_norm.weight.capacity()
        + merger_norm.bias.capacity()
        + merger_a.bias.capacity()
        + merger_b.bias.capacity()
        + blocks
            .iter()
            .map(|b| {
                b.n1.weight.capacity()
                    + b.n1.bias.capacity()
                    + b.n2.weight.capacity()
                    + b.n2.bias.capacity()
                    + b.qkv.bias.capacity()
                    + b.out.bias.capacity()
                    + b.up.bias.capacity()
                    + b.down.bias.capacity()
            })
            .sum::<usize>();
    accounting.grow(
        parameters as u64 * 4
            + arrays.iter().map(|a| a.capacity() as u64 * 4).sum::<u64>()
            + position_sum.capacity() as u64 * 8
            + work.capacity_bytes() as u64
            + scratch
                .iter()
                .map(|s| s.capacity_bytes() as u64)
                .sum::<u64>(),
    )?;
    for begin in (0..n).step_by(32) {
        cancel.check()?;
        let count = (n - begin).min(32);
        for row in 0..count {
            for (dst, value) in patches[row * 1536..(row + 1) * 1536]
                .iter_mut()
                .zip(image.patch_values(begin + row)?)
            {
                *dst = value;
            }
        }
        patch.apply(
            &model.set,
            &patches[..count * 1536],
            &mut x[begin * h..(begin + count) * h],
            &mut work,
            cancel,
            &mut cooperate,
        )?;
    }
    let side = config.positions.isqrt();
    let gh = plan.grid[1];
    let gw = plan.grid[2];
    let (storage, tensor) = model.set.bound(&pos)?;
    for row in 0..n {
        cancel.check()?;
        let group = row / 4;
        let y = group / (gw / 2) * 2 + row % 4 / 2;
        let col = group % (gw / 2) * 2 + row % 2;
        let fy = (y as f64 * (side - 1) as f64 / (gh - 1) as f64) as f32;
        let fx = (col as f64 * (side - 1) as f64 / (gw - 1) as f64) as f32;
        let y0 = fy as usize;
        let x0 = fx as usize;
        let dy = fy - y0 as f32;
        let dx = fx - x0 as f32;
        position_sum.fill(0.);
        for (yy, xx, w) in [
            (y0, x0, (1. - dy) * (1. - dx)),
            (y0, (x0 + 1).min(side - 1), (1. - dy) * dx),
            ((y0 + 1).min(side - 1), x0, dy * (1. - dx)),
            ((y0 + 1).min(side - 1), (x0 + 1).min(side - 1), dy * dx),
        ] {
            tensor::read_row(
                storage,
                0,
                tensor,
                yy * side + xx,
                &mut position_row,
                &mut work,
            )?;
            for (sum, value) in position_sum.iter_mut().zip(&position_row) {
                *sum += f64::from(*value * w);
            }
        }
        for (value, pos) in x[row * h..(row + 1) * h].iter_mut().zip(&position_sum) {
            *value += *pos as f32;
        }
    }
    let mut observe = |name: &str, values: &[f32], width: usize| -> Result<()> {
        if let Some(o) = observer.as_deref_mut() {
            o(Checkpoint {
                name,
                rows: values.len() / width,
                width,
                values,
            })?;
        }
        Ok(())
    };
    observe("embedding", &x, h)?;
    let head_dim = h / config.heads;
    let half = head_dim / 2;
    let quarter = head_dim / 4;
    for (index, block) in blocks.iter().enumerate() {
        cancel.check()?;
        block.n1.apply(&x, &mut normalized, cancel)?;
        block.qkv.apply(
            &model.set,
            &normalized,
            &mut qkv,
            &mut work,
            cancel,
            &mut cooperate,
        )?;
        for row in 0..n {
            q[row * h..(row + 1) * h].copy_from_slice(&qkv[row * 3 * h..row * 3 * h + h]);
            k[row * h..(row + 1) * h].copy_from_slice(&qkv[row * 3 * h + h..row * 3 * h + 2 * h]);
            v[row * h..(row + 1) * h].copy_from_slice(&qkv[row * 3 * h + 2 * h..(row + 1) * 3 * h]);
        }
        for row in 0..n {
            let group = row / 4;
            let y = group / (gw / 2) * 2 + row % 4 / 2;
            let col = group % (gw / 2) * 2 + row % 2;
            for values in [&mut q, &mut k] {
                for head in 0..config.heads {
                    let base = row * h + head * head_dim;
                    for j in 0..half {
                        let axis = if j < quarter { y } else { col };
                        let frequency =
                            10000f32.powf(-(((j % quarter) * 2) as f32) / (half as f32));
                        let (sin, cos) = (axis as f32 * frequency).sin_cos();
                        let a = values[base + j];
                        let b = values[base + half + j];
                        values[base + j] = a * cos - b * sin;
                        values[base + half + j] = a * sin + b * cos;
                    }
                }
            }
        }
        spatial::attention_cooperative(
            &q,
            &k,
            &v,
            &mut attended,
            &mut scratch,
            h,
            config.heads,
            &work,
            cancel,
            &mut cooperate,
        )?;
        block.out.apply(
            &model.set,
            &attended,
            &mut projected,
            &mut work,
            cancel,
            &mut cooperate,
        )?;
        spatial::residual(&mut x, &projected, cancel)?;
        block.n2.apply(&x, &mut normalized, cancel)?;
        block.up.apply(
            &model.set,
            &normalized,
            &mut mlp[..n * f],
            &mut work,
            cancel,
            &mut cooperate,
        )?;
        spatial::activation(&mut mlp[..n * f], false, cancel)?;
        block.down.apply(
            &model.set,
            &mlp[..n * f],
            &mut projected,
            &mut work,
            cancel,
            &mut cooperate,
        )?;
        spatial::residual(&mut x, &projected, cancel)?;
        observe(&format!("layer.{index}"), &x, h)?;
    }
    merger_norm.apply(&x, &mut normalized, cancel)?;
    merger_a.apply(
        &model.set,
        &normalized,
        &mut merged,
        &mut work,
        cancel,
        &mut cooperate,
    )?;
    gelu_exact(&mut merged, cancel)?;
    merger_b.apply(
        &model.set,
        &merged,
        &mut output,
        &mut work,
        cancel,
        &mut cooperate,
    )?;
    observe("merged", &output, config.output)?;
    let output_bytes = output.capacity() as u64 * 4;
    let mut held = held;
    let output_reservation = held.split(output_bytes + 4096)?;
    // Scratch/worker accounting is retired before the owned output is returned.
    drop(accounting);
    let mut output_accounting = model.budget.track_buffers(output_reservation.bytes());
    output_accounting.grow(output_bytes)?;
    Ok(VisionResult {
        values: output,
        grid: plan.grid,
        encoding_ms: start.elapsed().as_secs_f64() * 1000.,
        _buffers: output_accounting,
        _reservation: output_reservation,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_gelu_matches_independent_values() {
        let mut values = [-2., -1., 0., 1., 2.];
        gelu_exact(&mut values, &Default::default()).unwrap();
        for (a, b) in values
            .iter()
            .zip([-0.045500264, -0.15865526, 0., 0.8413447, 1.9544997])
        {
            assert!((*a - b).abs() < 2e-6);
        }
    }
}

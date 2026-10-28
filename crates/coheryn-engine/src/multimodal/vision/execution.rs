use super::{
    VisionConfig,
    kernels::{self, AttentionScratch},
};
use crate::multimodal::{
    image::{ImagePlan, PreparedImage, Processor},
    positions::ImagePositions,
};
use crate::{
    Cancellation, Descriptor, Error, ParseLimits, Result, cpu,
    gguf::{self, Tensor},
    memory::{self, Budget, BufferAccounting, MemoryStats, Reservation},
    storage::{Storage, digest_hex},
    tensor::{self, Workspace},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};

const IO: usize = 256 << 10;
pub const PROFILE: &str = "coheryn-qwen2vl-vision-v2";
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VisionOptions {
    pub threads: usize,
    pub tile_tokens: usize,
    pub max_patches: usize,
}
impl Default for VisionOptions {
    fn default() -> Self {
        Self {
            threads: 1,
            tile_tokens: 32,
            max_patches: 4096,
        }
    }
}
impl VisionOptions {
    pub(crate) fn validate(&self) -> Result<()> {
        memory::worker_bytes(self.threads)?;
        if !(1..=32).contains(&self.tile_tokens)
            || !(4..=4096).contains(&self.max_patches)
            || !self.max_patches.is_multiple_of(4)
        {
            return Err(Error::request(
                "vision limits require 1..=32 matrix rows and 4..=4096 patches in groups of four",
            ));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct VisionPlan {
    pub profile: &'static str,
    pub patch_rows: usize,
    pub output_tokens: usize,
    pub output_width: usize,
    pub grid: [usize; 3],
    pub threads: usize,
    pub tile_tokens: usize,
    pub workspace_bytes: u64,
    pub output_bytes: u64,
    pub peak_reserved_bytes: u64,
}
pub struct VisionCheckpoint<'a> {
    pub name: &'a str,
    pub rows: usize,
    pub width: usize,
    pub values: &'a [f32],
}
type Observer<'a> = dyn FnMut(VisionCheckpoint<'_>) -> Result<()> + 'a;
pub struct VisionResult {
    pub model_sha256: String,
    pub plan: VisionPlan,
    pub encoding_ms: f64,
    pub memory: MemoryStats,
    data: Vec<f32>,
    _buffers: BufferAccounting,
    _reservation: Reservation,
}
impl VisionResult {
    pub(crate) fn recycle(self, held: &mut Reservation) -> Result<()> {
        let Self {
            data,
            _buffers,
            _reservation,
            ..
        } = self;
        drop(data);
        drop(_buffers);
        held.absorb(_reservation)
    }
    pub fn embeddings(&self) -> &[f32] {
        &self.data
    }
}
impl std::fmt::Debug for VisionResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VisionResult")
            .field("plan", &self.plan)
            .finish_non_exhaustive()
    }
}
pub struct VisionModel {
    descriptor: Descriptor,
    config: VisionConfig,
    storage: Storage,
    sha256: String,
    budget: Budget,
    _resident: Reservation,
}
fn parse_limits() -> ParseLimits {
    ParseLimits {
        max_metadata_entries: 512,
        max_tensors: 2048,
        max_array_items: 1024,
        max_string_bytes: 8192,
        max_header_bytes: 1 << 20,
        max_metadata_allocation: 4 << 20,
        max_depth: 4,
    }
}
fn sum(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(0usize, |a, b| {
        a.checked_add(*b)
            .ok_or_else(|| Error::resource("vision reservation arithmetic overflow"))
    })
}
impl VisionModel {
    pub fn load(path: impl AsRef<Path>, budget: Budget, cancel: &Cancellation) -> Result<Self> {
        cancel.check()?;
        let path = path.as_ref();
        let stat = std::fs::metadata(path)?;
        if !stat.is_file() || stat.len() > 8 << 30 {
            return Err(Error::invalid(
                "vision component must be a regular GGUF file within 8 GiB",
            ));
        }
        let _loading = budget.reserve(8 << 20)?;
        let d = gguf::inspect(path, parse_limits())?;
        let c = VisionConfig::from_descriptor(&d)?;
        cancel.check()?;
        let metadata = d
            .data_offset
            .checked_mul(16)
            .and_then(|n| n.checked_add(d.tensors.len() as u64 * 512))
            .and_then(|n| n.checked_add(64 << 10))
            .ok_or_else(|| Error::resource("vision metadata reservation overflow"))?;
        let resident = budget.reserve(metadata)?;
        let storage = Storage::open_with_limits(path, &d, parse_limits())?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(IO)
            .map_err(|_| Error::resource("vision hash buffer allocation failed"))?;
        bytes.resize(IO, 0);
        let mut observed = budget.track_buffers(IO as u64);
        observed.grow(bytes.capacity() as u64)?;
        let mut hash = Sha256::new();
        let mut offset = 0;
        while offset < d.file_bytes {
            cancel.check()?;
            let n = ((d.file_bytes - offset).min(IO as u64)) as usize;
            storage.read(offset, &mut bytes[..n])?;
            hash.update(&bytes[..n]);
            offset += n as u64;
        }
        storage.unchanged()?;
        cancel.check()?;
        Ok(Self {
            descriptor: d,
            config: c,
            storage,
            sha256: digest_hex(hash.finalize().into()),
            budget,
            _resident: resident,
        })
    }
    pub fn resident_bytes(&self) -> u64 {
        self._resident.bytes()
    }
    pub fn config(&self) -> &VisionConfig {
        &self.config
    }
    pub fn model_sha256(&self) -> &str {
        &self.sha256
    }
    pub fn plan(&self, image: &PreparedImage, options: &VisionOptions) -> Result<VisionPlan> {
        options.validate()?;
        let p = image.plan();
        if p.profile != Processor::Qwen2VlStill
            || p.patch_features != 1176
            || !image.admitted_to(&self.budget)
        {
            return Err(Error::request(
                "vision input requires a Qwen still image admitted to this resource budget",
            ));
        }
        if image.normalized_hwc().len()
            != memory::product(&[p.width as usize, p.height as usize, 3])?
        {
            return Err(Error::invalid("prepared vision input dimensions disagree"));
        }
        self.plan_image(p, options)
    }
    pub(crate) fn plan_image(&self, p: &ImagePlan, options: &VisionOptions) -> Result<VisionPlan> {
        self.config.plan_image(p, options)
    }
    pub fn encode(
        &self,
        image: &PreparedImage,
        options: &VisionOptions,
        cancel: &Cancellation,
    ) -> Result<VisionResult> {
        self.run(image, options, cancel, None)
    }
    pub fn trace(
        &self,
        image: &PreparedImage,
        options: &VisionOptions,
        cancel: &Cancellation,
        observer: &mut Observer<'_>,
    ) -> Result<VisionResult> {
        self.run(image, options, cancel, Some(observer))
    }
    fn run(
        &self,
        image: &PreparedImage,
        options: &VisionOptions,
        cancel: &Cancellation,
        observer: Option<&mut Observer<'_>>,
    ) -> Result<VisionResult> {
        let mut held = self
            .budget
            .reserve(self.plan(image, options)?.peak_reserved_bytes)?;
        self.run_reserved(image, options, cancel, observer, &mut held, None)
    }
    pub(crate) fn encode_reserved(
        &self,
        image: &PreparedImage,
        options: &VisionOptions,
        cancel: &Cancellation,
        held: &mut Reservation,
    ) -> Result<VisionResult> {
        self.run_reserved(image, options, cancel, None, held, None)
    }
    pub(crate) fn encode_cooperative(
        &self,
        image: &PreparedImage,
        options: &VisionOptions,
        cancel: &Cancellation,
        held: &mut Reservation,
        cooperate: &mut crate::multimodal::WorkYield<'_>,
    ) -> Result<VisionResult> {
        self.run_reserved(image, options, cancel, None, held, Some(cooperate))
    }
    fn run_reserved(
        &self,
        image: &PreparedImage,
        options: &VisionOptions,
        cancel: &Cancellation,
        mut observer: Option<&mut Observer<'_>>,
        held: &mut Reservation,
        mut cooperate: Option<&mut crate::multimodal::WorkYield<'_>>,
    ) -> Result<VisionResult> {
        cancel.check()?;
        self.storage.unchanged()?;
        let plan = self.plan(image, options)?;
        if !held.covers(&self.budget, plan.peak_reserved_bytes) {
            return Err(Error::resource("invalid vision stage admission"));
        }
        let mut s = Session::new(&self.config, &plan, &self.budget)?;
        let mut output = memory::zeros(memory::product(&[plan.output_tokens, plan.output_width])?)?;
        let mut output_buffers = self.budget.track_buffers(plan.output_bytes);
        output_buffers.grow((output.capacity() * 4) as u64)?;
        let start = Instant::now();
        let n = plan.patch_rows;
        let h = self.config.hidden;
        for row in 0..n {
            cancel.check()?;
            for (i, value) in image.patch_values(row)?.enumerate() {
                if (i / 196) % 2 == 0 {
                    s.patches[row * 588 + (i / 392) * 196 + i % 196] = value;
                }
            }
        }
        for (index, name) in ["v.patch_embd.weight", "v.patch_embd.weight.1"]
            .into_iter()
            .enumerate()
        {
            let mut view = self.tensor(name)?.clone();
            view.dimensions = vec![588, h as u64];
            tensor::matmul_cooperative(
                &self.storage,
                self.descriptor.data_offset,
                &view,
                &s.patches,
                if index == 0 {
                    &mut s.states
                } else {
                    &mut s.delta
                },
                &mut s.work,
                cancel,
                &mut cooperate,
            )?;
        }
        kernels::residual(&mut s.states, &s.delta, cancel)?;
        emit(&mut observer, "patch_embeddings", n, h, &s.states)?;
        kernels::rope_cache(plan.grid, self.config.head_dim, &mut s.rope, cancel)?;
        for layer in 0..self.config.layers {
            cancel.check()?;
            self.storage.unchanged()?;
            s.norm.copy_from_slice(&s.states);
            self.norm(
                &format!("v.blk.{layer}.ln1"),
                &mut s.norm,
                &mut s.weight,
                &mut s.bias,
                &mut s.work,
                cancel,
                &mut cooperate,
            )?;
            self.affine(
                &format!("v.blk.{layer}.attn_q"),
                &s.norm,
                &mut s.q,
                &mut s.bias,
                &mut s.work,
                cancel,
                &mut cooperate,
            )?;
            self.affine(
                &format!("v.blk.{layer}.attn_k"),
                &s.norm,
                &mut s.k,
                &mut s.bias,
                &mut s.work,
                cancel,
                &mut cooperate,
            )?;
            self.affine(
                &format!("v.blk.{layer}.attn_v"),
                &s.norm,
                &mut s.v,
                &mut s.bias,
                &mut s.work,
                cancel,
                &mut cooperate,
            )?;
            kernels::rotate(&mut s.q, h, self.config.head_dim, &s.rope, cancel)?;
            kernels::rotate(&mut s.k, h, self.config.head_dim, &s.rope, cancel)?;
            kernels::attention_cooperative(
                &s.q,
                &s.k,
                &s.v,
                &mut s.norm,
                &mut s.attention,
                h,
                self.config.heads,
                &s.work,
                cancel,
                &mut cooperate,
            )?;
            self.affine(
                &format!("v.blk.{layer}.attn_out"),
                &s.norm,
                &mut s.delta,
                &mut s.bias,
                &mut s.work,
                cancel,
                &mut cooperate,
            )?;
            kernels::residual(&mut s.states, &s.delta, cancel)?;
            s.norm.copy_from_slice(&s.states);
            self.norm(
                &format!("v.blk.{layer}.ln2"),
                &mut s.norm,
                &mut s.weight,
                &mut s.bias,
                &mut s.work,
                cancel,
                &mut cooperate,
            )?;
            // Legacy GGUF up/down names are opposite to their actual matrix shapes.
            self.affine(
                &format!("v.blk.{layer}.ffn_down"),
                &s.norm,
                &mut s.ff,
                &mut s.bias,
                &mut s.work,
                cancel,
                &mut cooperate,
            )?;
            kernels::activation(&mut s.ff, true, cancel)?;
            self.affine(
                &format!("v.blk.{layer}.ffn_up"),
                &s.ff,
                &mut s.delta,
                &mut s.bias,
                &mut s.work,
                cancel,
                &mut cooperate,
            )?;
            kernels::residual(&mut s.states, &s.delta, cancel)?;
            emit(
                &mut observer,
                &format!("layer_out-{layer}"),
                n,
                h,
                &s.states,
            )?;
        }
        self.norm(
            "v.post_ln",
            &mut s.states,
            &mut s.weight,
            &mut s.bias,
            &mut s.work,
            cancel,
            &mut cooperate,
        )?;
        emit(&mut observer, "post_norm", n, h, &s.states)?;
        // Four already-adjacent hidden rows are the merger input; no shuffle copy.
        self.affine(
            "mm.0",
            &s.states,
            &mut s.norm,
            &mut s.bias,
            &mut s.work,
            cancel,
            &mut cooperate,
        )?;
        emit(
            &mut observer,
            "merger_up_b",
            plan.output_tokens,
            self.config.projector_hidden,
            &s.norm,
        )?;
        kernels::merger_activation(&mut s.norm, cancel)?;
        emit(
            &mut observer,
            "merger_gelu",
            plan.output_tokens,
            self.config.projector_hidden,
            &s.norm,
        )?;
        self.affine(
            "mm.2",
            &s.norm,
            &mut output,
            &mut s.bias,
            &mut s.work,
            cancel,
            &mut cooperate,
        )?;
        emit(
            &mut observer,
            "projected_embeddings",
            plan.output_tokens,
            plan.output_width,
            &output,
        )?;
        self.storage.unchanged()?;
        cancel.check()?;
        let encoding_ms = start.elapsed().as_secs_f64() * 1000.;
        drop(s);
        let output_reservation = held.split(plan.output_bytes)?;
        Ok(VisionResult {
            model_sha256: self.sha256.clone(),
            plan,
            encoding_ms,
            memory: self.budget.stats(),
            data: output,
            _buffers: output_buffers,
            _reservation: output_reservation,
        })
    }
    fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.descriptor
            .tensor(name)
            .ok_or_else(|| Error::invalid("validated vision tensor disappeared"))
    }
    // Keep explicit disjoint workspace buffers beside their scheduling checkpoint.
    #[allow(clippy::too_many_arguments)]
    fn affine(
        &self,
        name: &str,
        input: &[f32],
        output: &mut [f32],
        bias: &mut [f32],
        work: &mut Workspace,
        cancel: &Cancellation,
        cooperate: &mut Option<&mut crate::multimodal::WorkYield<'_>>,
    ) -> Result<()> {
        let t = self.tensor(&format!("{name}.weight"))?;
        tensor::matmul_cooperative(
            &self.storage,
            self.descriptor.data_offset,
            t,
            input,
            output,
            work,
            cancel,
            cooperate,
        )?;
        let width = t.dimensions[1] as usize;
        tensor::read_row(
            &self.storage,
            self.descriptor.data_offset,
            self.tensor(&format!("{name}.bias"))?,
            0,
            &mut bias[..width],
            work,
        )?;
        for row in output.chunks_exact_mut(width) {
            cancel.check()?;
            for (value, bias) in row.iter_mut().zip(&bias[..width]) {
                *value += bias;
                if !value.is_finite() {
                    return Err(Error::invalid("non-finite vision bias result"));
                }
            }
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn norm(
        &self,
        name: &str,
        states: &mut [f32],
        weight: &mut [f32],
        bias: &mut [f32],
        work: &mut Workspace,
        cancel: &Cancellation,
        cooperate: &mut Option<&mut crate::multimodal::WorkYield<'_>>,
    ) -> Result<()> {
        let h = self.config.hidden;
        tensor::read_row(
            &self.storage,
            self.descriptor.data_offset,
            self.tensor(&format!("{name}.weight"))?,
            0,
            &mut weight[..h],
            work,
        )?;
        tensor::read_row(
            &self.storage,
            self.descriptor.data_offset,
            self.tensor(&format!("{name}.bias"))?,
            0,
            &mut bias[..h],
            work,
        )?;
        for row in states.chunks_exact_mut(h) {
            cancel.check()?;
            cpu::layer_norm(row, &weight[..h], &bias[..h], self.config.norm_epsilon)?;
        }
        if let Some(yield_work) = cooperate.as_deref_mut() {
            yield_work()?;
        }
        Ok(())
    }
}
fn emit(
    observer: &mut Option<&mut Observer<'_>>,
    name: &str,
    rows: usize,
    width: usize,
    values: &[f32],
) -> Result<()> {
    if let Some(observer) = observer {
        observer(VisionCheckpoint {
            name,
            rows,
            width,
            values,
        })?;
    }
    Ok(())
}
struct Session {
    work: Workspace,
    states: Vec<f32>,
    norm: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    delta: Vec<f32>,
    ff: Vec<f32>,
    patches: Vec<f32>,
    rope: Vec<f32>,
    weight: Vec<f32>,
    bias: Vec<f32>,
    attention: Vec<AttentionScratch>,
    _buffers: BufferAccounting,
}
impl Session {
    fn new(c: &VisionConfig, p: &VisionPlan, budget: &Budget) -> Result<Self> {
        let n = p.patch_rows;
        let hidden = memory::product(&[n, c.hidden])?;
        let width = c
            .hidden
            .max(c.intermediate)
            .max(c.projector_hidden)
            .max(c.output_width)
            .max(588);
        let mut attention = Vec::new();
        attention
            .try_reserve_exact(p.threads)
            .map_err(|_| Error::resource("vision worker descriptors allocation failed"))?;
        for _ in 0..p.threads {
            attention.push(AttentionScratch::new(n, c.head_dim)?);
        }
        let mut s = Self {
            work: Workspace::new(width, IO, p.threads, p.tile_tokens)?,
            states: memory::zeros(hidden)?,
            norm: memory::zeros(hidden)?,
            q: memory::zeros(hidden)?,
            k: memory::zeros(hidden)?,
            v: memory::zeros(hidden)?,
            delta: memory::zeros(hidden)?,
            ff: memory::zeros(memory::product(&[n, c.intermediate])?)?,
            patches: memory::zeros(memory::product(&[n, 588])?)?,
            rope: memory::zeros(memory::product(&[n, c.head_dim])?)?,
            weight: memory::zeros(width)?,
            bias: memory::zeros(width)?,
            attention,
            _buffers: budget.track_buffers(p.workspace_bytes),
        };
        let mut bytes = s.work.capacity_bytes();
        for v in [
            &s.states, &s.norm, &s.q, &s.k, &s.v, &s.delta, &s.ff, &s.patches, &s.rope, &s.weight,
            &s.bias,
        ] {
            bytes = sum(&[bytes, v.capacity() * 4])?;
        }
        bytes = sum(&[
            bytes,
            s.attention.capacity() * std::mem::size_of::<AttentionScratch>(),
        ])?;
        for a in &s.attention {
            bytes = sum(&[bytes, a.capacity_bytes()])?;
        }
        s._buffers.grow(bytes as u64)?;
        Ok(s)
    }
}

impl VisionConfig {
    pub fn plan_image(&self, p: &ImagePlan, options: &VisionOptions) -> Result<VisionPlan> {
        options.validate()?;
        if p.profile != Processor::Qwen2VlStill || p.patch_features != 1176 {
            return Err(Error::request("unsupported vision processor geometry"));
        }
        let span = ImagePositions::new(p, 0)?;
        let n = p.patch_rows;
        if n > options.max_patches {
            return Err(Error::resource(
                "image patch count exceeds the requested vision limit",
            ));
        }
        if span.physical_tokens() != n / 4 {
            return Err(Error::invalid("vision feature geometry mismatch"));
        }
        let c = self;
        let width = c
            .hidden
            .max(c.intermediate)
            .max(c.projector_hidden)
            .max(c.output_width)
            .max(588);
        let floats = sum(&[
            memory::product(&[n, c.hidden, 6])?,
            memory::product(&[n, c.intermediate])?,
            memory::product(&[n, 588])?,
            memory::product(&[n, c.head_dim])?,
            width * 2,
        ])?;
        let lanes = memory::product(&[
            options.tile_tokens,
            sum(&[
                width * 4,
                width.div_ceil(256) * 292,
                std::mem::size_of::<crate::quant::QuantizedInput>(),
                128,
            ])?,
        ])?;
        let attention = memory::product(&[
            options.threads,
            sum(&[
                n * 4,
                c.head_dim * 8,
                std::mem::size_of::<AttentionScratch>(),
            ])?,
        ])?;
        let workspace = sum(&[
            floats * 4,
            IO * 3,
            lanes,
            attention,
            memory::worker_bytes(options.threads)?,
            2 << 20,
        ])?;
        let output = sum(&[memory::product(&[n / 4, c.output_width, 4])?, 4096])?;
        Ok(VisionPlan {
            profile: PROFILE,
            patch_rows: n,
            output_tokens: n / 4,
            output_width: c.output_width,
            grid: p.grid,
            threads: options.threads,
            tile_tokens: options.tile_tokens,
            workspace_bytes: workspace as u64,
            output_bytes: output as u64,
            peak_reserved_bytes: sum(&[workspace, output])? as u64,
        })
    }
}

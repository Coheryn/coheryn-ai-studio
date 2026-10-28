use super::{
    EmbeddingConfig, EmbeddingOptions, EmbeddingPlan, EmbeddingPlanInput, EmbeddingTokenizer,
    config, plan,
};
use crate::{
    Cancellation, Descriptor, Error, ParseLimits, ResourceLimits, Result, cpu, gguf,
    memory::{self, Budget, BufferAccounting, MemoryStats, Reservation},
    planner::HardwareSnapshot,
    storage::Storage,
    tensor::{self, Workspace},
};
use rayon::prelude::*;
use serde::Serialize;
use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};

struct Slots {
    count: AtomicUsize,
    maximum: usize,
}
struct Slot(Arc<Slots>);
impl Drop for Slot {
    fn drop(&mut self) {
        self.0.count.fetch_sub(1, Ordering::AcqRel);
    }
}
impl Slots {
    fn acquire(self: &Arc<Self>) -> Result<Slot> {
        self.count
            .try_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < self.maximum).then_some(n + 1)
            })
            .map_err(|_| Error::resource("embedding inference concurrency limit reached"))?;
        Ok(Slot(self.clone()))
    }
}

pub struct EmbeddingModel {
    descriptor: Descriptor,
    config: EmbeddingConfig,
    tokenizer: EmbeddingTokenizer,
    storage: Storage,
    limits: ResourceLimits,
    budget: Budget,
    slots: Arc<Slots>,
    identity: Arc<()>,
    _metadata_reservation: Reservation,
}
pub struct EmbeddingAdmission {
    plan: EmbeddingPlan,
    options: EmbeddingOptions,
    input_bytes: Vec<usize>,
    identity: Arc<()>,
    reservation: Reservation,
    _slot: Option<Slot>,
}
#[derive(Debug, Serialize)]
pub struct EmbeddingItem {
    pub index: usize,
    pub embedding: Vec<f32>,
    pub prompt_tokens: usize,
    pub truncated: bool,
}
#[derive(Debug, Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}
#[derive(Serialize)]
pub struct EmbeddingResult {
    pub dimension: usize,
    pub data: Vec<EmbeddingItem>,
    pub usage: EmbeddingUsage,
    pub plan: EmbeddingPlan,
    pub tokenization_ms: f64,
    pub encoding_ms: f64,
    pub memory: MemoryStats,
    #[serde(skip)]
    _buffers: BufferAccounting,
    #[serde(skip)]
    _output_reservation: Reservation,
}
/// A bounded, resumable request. Its reservations outlive every workspace and output.
pub struct EmbeddingJob {
    session: Session,
    tokens: Vec<Vec<u32>>,
    truncated: Vec<bool>,
    groups: Vec<(usize, usize, usize)>,
    next_group: usize,
    failed: bool,
    output: Vec<EmbeddingItem>,
    output_buffers: BufferAccounting,
    tokenization_ms: f64,
    encoding_ms: f64,
    admission: EmbeddingAdmission,
}
impl EmbeddingJob {
    pub fn is_finished(&self) -> bool {
        !self.failed && self.next_group == self.groups.len()
    }
    pub fn prompt_tokens(&self) -> usize {
        self.tokens.iter().map(Vec::len).sum()
    }
}

/// Borrowed diagnostic view; no activation copies are retained by the engine.
pub struct EmbeddingCheckpoint<'a> {
    pub name: &'a str,
    pub first_input: usize,
    pub sequence_lengths: &'a [usize],
    pub width: usize,
    pub values: &'a [f32],
}
type Observer<'a> = dyn FnMut(EmbeddingCheckpoint<'_>) -> Result<()> + 'a;

impl EmbeddingAdmission {
    pub fn plan(&self) -> &EmbeddingPlan {
        &self.plan
    }
}

impl EmbeddingModel {
    pub fn descriptor(&self) -> &Descriptor {
        &self.descriptor
    }
    pub fn context_limit(&self) -> usize {
        self.limits.context
    }
    pub fn config(&self) -> &EmbeddingConfig {
        &self.config
    }
    pub fn load(path: impl AsRef<Path>, limits: ResourceLimits) -> Result<Self> {
        Self::load_shared(path, limits, Budget::new(limits.max_memory), 1)
    }
    pub fn load_shared(
        path: impl AsRef<Path>,
        limits: ResourceLimits,
        budget: Budget,
        max_concurrency: usize,
    ) -> Result<Self> {
        if !(2..=512).contains(&limits.context) || !(1..=16).contains(&max_concurrency) {
            return Err(Error::request(
                "embedding load requires context 2..=512 and concurrency 1..=16",
            ));
        }
        if budget.stats().limit > limits.max_memory {
            return Err(Error::request(
                "shared embedding budget exceeds the configured memory limit",
            ));
        }
        let mut parse = ParseLimits::default();
        parse.max_metadata_allocation = parse.max_metadata_allocation.min(limits.max_memory / 2);
        let descriptor = gguf::inspect(&path, parse)?;
        let config = config::configuration(&descriptor)?;
        if limits.context > config.qualified_context {
            return Err(Error::request(
                "embedding context exceeds the qualified model limit",
            ));
        }
        let metadata_reservation = budget.reserve(plan::model_bytes(&descriptor, &config)?)?;
        let tokenizer = EmbeddingTokenizer::from_descriptor(&descriptor)?;
        let storage = Storage::open(path.as_ref(), &descriptor)?;
        Ok(Self {
            descriptor,
            config,
            tokenizer,
            storage,
            limits,
            budget,
            slots: Arc::new(Slots {
                count: AtomicUsize::new(0),
                maximum: max_concurrency,
            }),
            identity: Arc::new(()),
            _metadata_reservation: metadata_reservation,
        })
    }
    pub fn memory(&self) -> MemoryStats {
        self.budget.stats()
    }
    pub fn tokenize(
        &self,
        text: &str,
        options: &EmbeddingOptions,
        cancel: &Cancellation,
    ) -> Result<(Vec<u32>, bool)> {
        options.validate(self.limits.context)?;
        // Standalone token diagnostics also reserve the bounded library allowance.
        let _reservation = self.budget.reserve(
            (text.len().min(super::tokenizer::MAX_TEXT_BYTES) * 128 + options.max_input_tokens * 4)
                as u64,
        )?;
        self.tokenizer
            .encode(text, options.max_input_tokens, options.truncate, cancel)
    }
    pub fn plan(
        &self,
        input_bytes: &[usize],
        options: &EmbeddingOptions,
        hardware: &HardwareSnapshot,
    ) -> Result<EmbeddingPlan> {
        plan::estimate(EmbeddingPlanInput {
            descriptor: &self.descriptor,
            config: &self.config,
            context: self.limits.context,
            input_bytes,
            options,
            budget: self.budget.stats(),
            hardware,
            max_concurrency: self.slots.maximum,
        })
    }
    pub fn reserve_request(
        &self,
        input_bytes: &[usize],
        options: &EmbeddingOptions,
        hardware: &HardwareSnapshot,
    ) -> Result<EmbeddingAdmission> {
        self.reserve(input_bytes, options, hardware, true, None)
    }
    /// Queue-safe: holds memory now; compute slots are acquired per microbatch.
    pub fn reserve_queued_request(
        &self,
        input_bytes: &[usize],
        options: &EmbeddingOptions,
        hardware: &HardwareSnapshot,
    ) -> Result<EmbeddingAdmission> {
        self.reserve(input_bytes, options, hardware, false, None)
    }
    /// Cap this request's additional engine reservation without creating a second budget.
    pub fn reserve_queued_capped(
        &self,
        input_bytes: &[usize],
        options: &EmbeddingOptions,
        hardware: &HardwareSnapshot,
        ceiling: Option<u64>,
    ) -> Result<EmbeddingAdmission> {
        self.reserve(input_bytes, options, hardware, false, ceiling)
    }
    fn reserve(
        &self,
        input_bytes: &[usize],
        options: &EmbeddingOptions,
        hardware: &HardwareSnapshot,
        acquire_slot: bool,
        ceiling: Option<u64>,
    ) -> Result<EmbeddingAdmission> {
        let mut budget = self.budget.stats();
        if let Some(ceiling) = ceiling {
            budget.limit = budget.limit.min(budget.reserved.saturating_add(ceiling));
        }
        let plan = plan::estimate(EmbeddingPlanInput {
            descriptor: &self.descriptor,
            config: &self.config,
            context: self.limits.context,
            input_bytes,
            options,
            budget,
            hardware,
            max_concurrency: self.slots.maximum,
        })?;
        plan.require_admission()?;
        let slot = if acquire_slot {
            Some(self.slots.acquire()?)
        } else {
            None
        };
        let reservation = self.budget.reserve(plan.additional_reserved_bytes)?;
        let mut lengths = Vec::new();
        lengths
            .try_reserve_exact(input_bytes.len())
            .map_err(|_| Error::resource("request identity allocation failed"))?;
        lengths.extend_from_slice(input_bytes);
        Ok(EmbeddingAdmission {
            plan,
            options: options.clone(),
            input_bytes: lengths,
            identity: self.identity.clone(),
            reservation,
            _slot: slot,
        })
    }
    pub fn embed(
        &self,
        inputs: &[&str],
        options: &EmbeddingOptions,
        cancel: &Cancellation,
    ) -> Result<EmbeddingResult> {
        let hardware = local_cpu_snapshot();
        self.embed_with_hardware(inputs, options, &hardware, cancel)
    }
    pub fn embed_with_hardware(
        &self,
        inputs: &[&str],
        options: &EmbeddingOptions,
        hardware: &HardwareSnapshot,
        cancel: &Cancellation,
    ) -> Result<EmbeddingResult> {
        let lengths = input_lengths(inputs)?;
        let admission = self.reserve_request(&lengths, options, hardware)?;
        self.embed_admitted(inputs, admission, cancel)
    }
    pub fn embed_admitted(
        &self,
        inputs: &[&str],
        admission: EmbeddingAdmission,
        cancel: &Cancellation,
    ) -> Result<EmbeddingResult> {
        self.execute(inputs, admission, cancel, None)
    }
    pub fn embed_observed(
        &self,
        inputs: &[&str],
        options: &EmbeddingOptions,
        cancel: &Cancellation,
        observer: &mut Observer<'_>,
    ) -> Result<EmbeddingResult> {
        let admission =
            self.reserve_request(&input_lengths(inputs)?, options, &local_cpu_snapshot())?;
        self.execute(inputs, admission, cancel, Some(observer))
    }
    fn execute(
        &self,
        inputs: &[&str],
        admission: EmbeddingAdmission,
        cancel: &Cancellation,
        mut observer: Option<&mut Observer<'_>>,
    ) -> Result<EmbeddingResult> {
        let mut job = self.prepare_admitted(inputs, admission, cancel)?;
        while !job.is_finished() {
            self.step_observed(&mut job, cancel, &mut observer)?;
        }
        self.finish(job)
    }
    /// Tokenize once and allocate one reusable workspace; callers may yield between steps.
    pub fn prepare_admitted(
        &self,
        inputs: &[&str],
        admission: EmbeddingAdmission,
        cancel: &Cancellation,
    ) -> Result<EmbeddingJob> {
        if !Arc::ptr_eq(&self.identity, &admission.identity)
            || input_lengths(inputs)? != admission.input_bytes
            || !admission
                .reservation
                .covers(&self.budget, admission.plan.additional_reserved_bytes)
        {
            return Err(Error::request(
                "embedding admission does not belong to this model/request",
            ));
        }
        cancel.check()?;
        let start = Instant::now();
        let mut tokens = Vec::new();
        tokens
            .try_reserve_exact(inputs.len())
            .map_err(|_| Error::resource("embedding token batch allocation failed"))?;
        let mut truncated = Vec::new();
        truncated
            .try_reserve_exact(inputs.len())
            .map_err(|_| Error::resource("embedding truncation allocation failed"))?;
        for (index, text) in inputs.iter().enumerate() {
            let (ids, shortened) = self
                .tokenizer
                .encode(
                    text,
                    admission.options.max_input_tokens,
                    admission.options.truncate,
                    cancel,
                )
                .map_err(|e| {
                    crate::Error::new(
                        e.code,
                        format!("embedding input at index {index}: {}", e.message),
                    )
                })?;
            tokens.push(ids);
            truncated.push(shortened);
        }
        let tokenization_ms = start.elapsed().as_secs_f64() * 1000.;
        let groups = microbatches(
            &tokens,
            admission.plan.batch_size,
            admission.plan.token_capacity,
        )?;
        let largest = groups
            .iter()
            .map(|g| g.2)
            .max()
            .ok_or_else(|| Error::request("empty embedding job"))?;
        let mut output_buffers = self.budget.track_buffers(admission.plan.output_bytes);
        let mut output = Vec::new();
        output
            .try_reserve_exact(inputs.len())
            .map_err(|_| Error::resource("embedding output allocation failed"))?;
        output_buffers.grow((output.capacity() * std::mem::size_of::<EmbeddingItem>()) as u64)?;
        let session = Session::new(&self.config, largest, &admission.plan, &self.budget)?;
        Ok(EmbeddingJob {
            session,
            tokens,
            truncated,
            groups,
            next_group: 0,
            failed: false,
            output,
            output_buffers,
            tokenization_ms,
            encoding_ms: 0.,
            admission,
        })
    }
    pub fn step(&self, job: &mut EmbeddingJob, cancel: &Cancellation) -> Result<()> {
        self.step_observed(job, cancel, &mut None)
    }
    fn step_observed(
        &self,
        job: &mut EmbeddingJob,
        cancel: &Cancellation,
        observer: &mut Option<&mut Observer<'_>>,
    ) -> Result<()> {
        if job.failed || !Arc::ptr_eq(&self.identity, &job.admission.identity) {
            return Err(Error::request("embedding job belongs to a different model"));
        }
        job.failed = true;
        let _slot = if job.admission._slot.is_none() {
            Some(self.slots.acquire()?)
        } else {
            None
        };
        let (first, end, count) = *job
            .groups
            .get(job.next_group)
            .ok_or_else(|| Error::request("embedding job is already complete"))?;
        let start = Instant::now();
        cancel.check()?;
        self.storage.unchanged()?;
        let lengths: Vec<usize> = job.tokens[first..end].iter().map(Vec::len).collect();
        self.encode(
            &job.tokens[first..end],
            &mut job.session,
            &job.admission.plan,
            cancel,
            first,
            &lengths,
            observer,
        )?;
        let mut offset = 0;
        for index in first..end {
            cancel.check()?;
            let n = job.tokens[index].len();
            let mut vector = memory::zeros(self.config.hidden)?;
            let inverse = 1. / n as f32;
            for (column, value) in vector.iter_mut().enumerate() {
                if column.is_multiple_of(64) {
                    cancel.check()?;
                }
                *value = (0..n)
                    .map(|row| {
                        f64::from(
                            job.session.states[(offset + row) * self.config.hidden + column]
                                * inverse,
                        )
                    })
                    .sum::<f64>() as f32;
            }
            observe(observer, "pooled", index, &[n], self.config.hidden, &vector)?;
            let norm = vector
                .iter()
                .map(|v| f64::from(*v) * f64::from(*v))
                .sum::<f64>()
                .sqrt();
            if !norm.is_finite() || norm == 0. {
                return Err(Error::invalid(
                    "embedding output has a non-finite or zero norm",
                ));
            }
            for value in &mut vector {
                *value = (f64::from(*value) / norm) as f32;
            }
            job.output_buffers.grow((vector.capacity() * 4) as u64)?;
            job.output.push(EmbeddingItem {
                index,
                embedding: vector,
                prompt_tokens: n,
                truncated: job.truncated[index],
            });
            offset += n;
        }
        debug_assert_eq!(offset, count);
        cancel.check()?;
        job.encoding_ms += start.elapsed().as_secs_f64() * 1000.;
        job.next_group += 1;
        job.failed = false;
        Ok(())
    }
    pub fn finish(&self, job: EmbeddingJob) -> Result<EmbeddingResult> {
        if !job.is_finished() || !Arc::ptr_eq(&self.identity, &job.admission.identity) {
            return Err(Error::request(
                "embedding job is incomplete or belongs to another model",
            ));
        }
        let EmbeddingJob {
            session,
            tokens,
            truncated,
            groups,
            output,
            output_buffers,
            tokenization_ms,
            encoding_ms,
            mut admission,
            ..
        } = job;
        let prompt_tokens = tokens.iter().map(Vec::len).sum();
        drop(session);
        drop(tokens);
        drop(truncated);
        drop(groups);
        let output_reservation = admission.reservation.split(admission.plan.output_bytes)?;
        let plan = admission.plan.clone();
        drop(admission);
        Ok(EmbeddingResult {
            dimension: self.config.hidden,
            data: output,
            usage: EmbeddingUsage {
                prompt_tokens,
                total_tokens: prompt_tokens,
            },
            plan,
            tokenization_ms,
            encoding_ms,
            memory: self.budget.stats(),
            _buffers: output_buffers,
            _output_reservation: output_reservation,
        })
    }
    fn row(&self, name: &str, row: usize, out: &mut [f32], work: &mut Workspace) -> Result<()> {
        let t = self
            .descriptor
            .tensor(name)
            .ok_or_else(|| Error::invalid("validated embedding tensor disappeared"))?;
        tensor::read_row(
            &self.storage,
            self.descriptor.data_offset,
            t,
            row,
            out,
            work,
        )
    }
    fn matrix(
        &self,
        name: &str,
        input: &[f32],
        out: &mut [f32],
        work: &mut Workspace,
        cancel: &Cancellation,
    ) -> Result<()> {
        let t = self
            .descriptor
            .tensor(name)
            .ok_or_else(|| Error::invalid("validated embedding matrix disappeared"))?;
        tensor::matmul(
            &self.storage,
            self.descriptor.data_offset,
            t,
            input,
            out,
            work,
            cancel,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn encode(
        &self,
        tokens: &[Vec<u32>],
        s: &mut Session,
        plan: &EmbeddingPlan,
        cancel: &Cancellation,
        first: usize,
        lengths: &[usize],
        observer: &mut Option<&mut Observer<'_>>,
    ) -> Result<()> {
        let c = &self.config;
        let h = c.hidden;
        let n: usize = lengths.iter().sum();
        self.row("token_types.weight", 0, &mut s.type_embedding, &mut s.work)?;
        for (row, id) in tokens.iter().flatten().enumerate() {
            cancel.check()?;
            if *id as usize >= c.vocabulary {
                return Err(Error::invalid("embedding token outside vocabulary"));
            }
            self.row(
                "token_embd.weight",
                *id as usize,
                &mut s.states[row * h..(row + 1) * h],
                &mut s.work,
            )?;
            for (v, t) in s.states[row * h..(row + 1) * h]
                .iter_mut()
                .zip(&s.type_embedding)
            {
                *v += t;
            }
        }
        self.norm(
            "token_embd_norm",
            &mut s.states[..n * h],
            &mut s.weight,
            &mut s.bias,
            &mut s.work,
        )?;
        for layer in 0..c.layers {
            cancel.check()?;
            self.storage.unchanged()?;
            self.matrix(
                &format!("blk.{layer}.attn_qkv.weight"),
                &s.states[..n * h],
                &mut s.qkv[..n * 3 * h],
                &mut s.work,
                cancel,
            )?;
            let mut row = 0;
            for len in lengths {
                for pos in 0..*len {
                    let base = row * 3 * h;
                    cpu::rope(&mut s.qkv[base..base + h], c.head_dim, pos, c.rope_base);
                    cpu::rope(
                        &mut s.qkv[base + h..base + 2 * h],
                        c.head_dim,
                        pos,
                        c.rope_base,
                    );
                    row += 1;
                }
            }
            attention(
                &s.qkv[..n * 3 * h],
                &mut s.attention[..n * h],
                &mut s.attention_work,
                lengths,
                c,
                &s.work,
                plan.threads,
                cancel,
            )?;
            self.matrix(
                &format!("blk.{layer}.attn_output.weight"),
                &s.attention[..n * h],
                &mut s.projection[..n * h],
                &mut s.work,
                cancel,
            )?;
            for (v, delta) in s.states[..n * h].iter_mut().zip(&s.projection) {
                *v += delta;
            }
            self.norm(
                &format!("blk.{layer}.attn_output_norm"),
                &mut s.states[..n * h],
                &mut s.weight,
                &mut s.bias,
                &mut s.work,
            )?;
            self.matrix(
                &format!("blk.{layer}.ffn_up.weight"),
                &s.states[..n * h],
                &mut s.up[..n * c.intermediate],
                &mut s.work,
                cancel,
            )?;
            self.matrix(
                &format!("blk.{layer}.ffn_gate.weight"),
                &s.states[..n * h],
                &mut s.gate[..n * c.intermediate],
                &mut s.work,
                cancel,
            )?;
            cpu::swiglu(
                &mut s.gate[..n * c.intermediate],
                &s.up[..n * c.intermediate],
            );
            self.matrix(
                &format!("blk.{layer}.ffn_down.weight"),
                &s.gate[..n * c.intermediate],
                &mut s.projection[..n * h],
                &mut s.work,
                cancel,
            )?;
            if observer.is_some() {
                observe(
                    observer,
                    &format!("ffn_out-{layer}"),
                    first,
                    lengths,
                    h,
                    &s.projection[..n * h],
                )?;
            }
            for (v, delta) in s.states[..n * h].iter_mut().zip(&s.projection) {
                *v += delta;
            }
            self.norm(
                &format!("blk.{layer}.layer_output_norm"),
                &mut s.states[..n * h],
                &mut s.weight,
                &mut s.bias,
                &mut s.work,
            )?;
        }
        observe(
            observer,
            "result_embd",
            first,
            lengths,
            h,
            &s.states[..n * h],
        )
    }
    fn norm(
        &self,
        name: &str,
        states: &mut [f32],
        weight: &mut [f32],
        bias: &mut [f32],
        work: &mut Workspace,
    ) -> Result<()> {
        self.row(&format!("{name}.weight"), 0, weight, work)?;
        self.row(&format!("{name}.bias"), 0, bias, work)?;
        for row in states.chunks_exact_mut(self.config.hidden) {
            layer_norm(row, weight, bias, self.config.norm_epsilon)?;
        }
        Ok(())
    }
}

fn input_lengths(inputs: &[&str]) -> Result<Vec<usize>> {
    if inputs.is_empty() || inputs.len() > plan::MAX_INPUTS {
        return Err(Error::request("embedding request requires 1..=256 inputs"));
    }
    Ok(inputs.iter().map(|s| s.len()).collect())
}
fn microbatches(
    tokens: &[Vec<u32>],
    items: usize,
    capacity: usize,
) -> Result<Vec<(usize, usize, usize)>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(tokens.len())
        .map_err(|_| Error::resource("microbatch descriptors allocation failed"))?;
    let mut first = 0;
    while first < tokens.len() {
        let mut end = first;
        let mut count = 0;
        while end < tokens.len() && end - first < items && tokens[end].len() <= capacity - count {
            count += tokens[end].len();
            end += 1;
        }
        if end == first {
            return Err(Error::resource(
                "individual embedding input exceeds admitted microbatch capacity",
            ));
        }
        result.push((first, end, count));
        first = end;
    }
    Ok(result)
}
fn observe(
    observer: &mut Option<&mut Observer<'_>>,
    name: &str,
    first_input: usize,
    lengths: &[usize],
    width: usize,
    values: &[f32],
) -> Result<()> {
    if let Some(observer) = observer {
        observer(EmbeddingCheckpoint {
            name,
            first_input,
            sequence_lengths: lengths,
            width,
            values,
        })?;
    }
    Ok(())
}
fn local_cpu_snapshot() -> HardwareSnapshot {
    HardwareSnapshot {
        os: std::env::consts::OS.into(),
        architecture: std::env::consts::ARCH.into(),
        logical_cpus: std::thread::available_parallelism().map_or(1, usize::from),
        cpu_features: Vec::new(),
        ram_total_bytes: None,
        ram_available_bytes: None,
        swap_total_bytes: None,
        swap_used_bytes: None,
        observed_unix_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis().min(u64::MAX as u128) as u64),
        gpu_available_bytes: None,
        gpu_backend_qualified: false,
    }
}

pub(crate) fn layer_norm(x: &mut [f32], weight: &[f32], bias: &[f32], epsilon: f32) -> Result<()> {
    cpu::layer_norm(x, weight, bias, epsilon)
}

struct AttentionScratch {
    scores: Vec<f32>,
    accumulator: Vec<f64>,
}
#[allow(clippy::too_many_arguments)]
fn attention(
    qkv: &[f32],
    output: &mut [f32],
    scratch: &mut [AttentionScratch],
    lengths: &[usize],
    c: &EmbeddingConfig,
    work: &Workspace,
    threads: usize,
    cancel: &Cancellation,
) -> Result<()> {
    let rows = output.len() / c.hidden;
    let group = rows.div_ceil(threads);
    let scale = 1. / (c.head_dim as f32).sqrt();
    let compute = |(chunk_index, (out, scratch)): (usize, (&mut [f32], &mut AttentionScratch))| -> Result<()> {
        for (local, out) in out.chunks_exact_mut(c.hidden).enumerate() {
            cancel.check()?;
            let row = chunk_index * group + local;
            let mut start = 0;
            let mut length = 0;
            for n in lengths {
                if row < start + n {
                    length = *n;
                    break;
                }
                start += n;
            }
            if length == 0 {
                return Err(Error::invalid("attention sequence boundary missing"));
            }
            for head in 0..c.heads {
                let query_base = row * 3 * c.hidden + head * c.head_dim;
                let query = &qkv[query_base..query_base + c.head_dim];
                let scores = &mut scratch.scores[..length];
                let mut maximum = f32::NEG_INFINITY;
                for (i, score) in scores.iter_mut().enumerate() {
                    if i.is_multiple_of(64) { cancel.check()?; }
                    let base = (start + i) * 3 * c.hidden + c.hidden + head * c.head_dim;
                    *score = cpu::dot(query, &qkv[base..base + c.head_dim]) * scale;
                    if !score.is_finite() {
                        return Err(Error::invalid("non-finite embedding attention score"));
                    }
                    maximum = maximum.max(*score);
                }
                let mut denominator = 0f64;
                for score in scores.iter_mut() {
                    *score = (*score - maximum).exp();
                    denominator += f64::from(*score);
                }
                let inverse = (1. / denominator) as f32;
                for score in scores.iter_mut() { *score *= inverse; }
                scratch.accumulator.fill(0.);
                for (i, score) in scores.iter().enumerate() {
                    if i.is_multiple_of(64) { cancel.check()?; }
                    let base = (start + i) * 3 * c.hidden + 2 * c.hidden + head * c.head_dim;
                    for (sum, value) in scratch.accumulator.iter_mut().zip(&qkv[base..base + c.head_dim]) {
                        *sum += f64::from(*score * *value);
                    }
                }
                for (dst, value) in out[head * c.head_dim..(head + 1) * c.head_dim].iter_mut().zip(&scratch.accumulator) {
                    *dst = *value as f32;
                    if !dst.is_finite() {
                        return Err(Error::invalid("non-finite embedding attention output"));
                    }
                }
            }
        }
        Ok(())
    };
    if threads > 1 {
        work.install(|| {
            output
                .par_chunks_mut(group * c.hidden)
                .zip(scratch.par_iter_mut())
                .enumerate()
                .try_for_each(compute)
        })
    } else {
        for pair in output
            .chunks_mut(group * c.hidden)
            .zip(scratch.iter_mut())
            .enumerate()
        {
            compute(pair)?;
        }
        Ok(())
    }
}
struct Session {
    work: Workspace,
    states: Vec<f32>,
    qkv: Vec<f32>,
    attention: Vec<f32>,
    projection: Vec<f32>,
    up: Vec<f32>,
    gate: Vec<f32>,
    weight: Vec<f32>,
    bias: Vec<f32>,
    type_embedding: Vec<f32>,
    attention_work: Vec<AttentionScratch>,
    _buffers: BufferAccounting,
}
impl Session {
    fn new(
        c: &EmbeddingConfig,
        tokens: usize,
        plan: &EmbeddingPlan,
        budget: &Budget,
    ) -> Result<Self> {
        let hidden = memory::product(&[tokens, c.hidden])?;
        let intermediate = memory::product(&[tokens, c.intermediate])?;
        let mut attention_work = Vec::new();
        attention_work
            .try_reserve_exact(plan.threads)
            .map_err(|_| Error::resource("attention workspace handles allocation failed"))?;
        for _ in 0..plan.threads {
            attention_work.push(AttentionScratch {
                scores: memory::zeros(plan.max_input_tokens)?,
                accumulator: memory::zeros_f64(c.head_dim)?,
            });
        }
        let mut session = Self {
            work: Workspace::new(
                c.hidden.max(c.intermediate),
                plan.io_bytes,
                plan.threads,
                plan.tensor_tile_tokens,
            )?,
            states: memory::zeros(hidden)?,
            qkv: memory::zeros(memory::product(&[hidden, 3])?)?,
            attention: memory::zeros(hidden)?,
            projection: memory::zeros(hidden)?,
            up: memory::zeros(intermediate)?,
            gate: memory::zeros(intermediate)?,
            weight: memory::zeros(c.hidden)?,
            bias: memory::zeros(c.hidden)?,
            type_embedding: memory::zeros(c.hidden)?,
            attention_work,
            _buffers: budget.track_buffers(plan.workspace_bytes),
        };
        let mut bytes = session.work.capacity_bytes();
        for v in [
            &session.states,
            &session.qkv,
            &session.attention,
            &session.projection,
            &session.up,
            &session.gate,
            &session.weight,
            &session.bias,
            &session.type_embedding,
        ] {
            bytes = bytes
                .checked_add(v.capacity() * 4)
                .ok_or_else(|| Error::resource("embedding observation overflow"))?;
        }
        for w in &session.attention_work {
            bytes = bytes
                .checked_add(w.scores.capacity() * 4 + w.accumulator.capacity() * 8)
                .ok_or_else(|| Error::resource("attention observation overflow"))?;
        }
        bytes += session.attention_work.capacity() * std::mem::size_of::<AttentionScratch>();
        session._buffers.grow(bytes as u64)?;
        Ok(session)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_norm_has_independent_affine_and_constant_fixtures() {
        let mut values = [1., 2., 3., 4.];
        layer_norm(&mut values, &[1., 2., 0., -1.], &[0., 1., 2., 3.], 1e-5).unwrap();
        let expected = [-1.3416355, 0.105576396, 2., 1.6583645];
        for (value, reference) in values.into_iter().zip(expected) {
            assert!((value - reference).abs() < 1e-6);
        }
        let mut constant = [9.; 4];
        layer_norm(&mut constant, &[1.; 4], &[1., 2., 3., 4.], 1e-12).unwrap();
        assert_eq!(constant, [1., 2., 3., 4.]);
        assert!(layer_norm(&mut [], &[], &[], 1e-5).is_err());
        assert!(layer_norm(&mut [1., 2.], &[1.], &[0.], 1e-5).is_err());
        assert!(layer_norm(&mut [f32::NAN], &[1.], &[0.], 1e-5).is_err());
    }

    #[test]
    fn bidirectional_attention_is_block_diagonal_and_cancellable() {
        let c = EmbeddingConfig {
            layers: 1,
            hidden: 2,
            intermediate: 2,
            heads: 1,
            head_dim: 2,
            vocabulary: 4,
            token_types: 1,
            model_context: 512,
            qualified_context: 512,
            rope_base: 1000.,
            norm_epsilon: 1e-5,
        };
        // Zero Q/K gives uniform attention: first sequence means [2,4];
        // the second sequence has one [100,200] value and cannot affect it.
        let qkv = [
            0., 0., 0., 0., 1., 3., 0., 0., 0., 0., 3., 5., 0., 0., 0., 0., 100., 200.,
        ];
        for threads in [1, 2] {
            let work = Workspace::new(2, 4096, threads, 1).unwrap();
            let mut scratch: Vec<_> = (0..threads)
                .map(|_| AttentionScratch {
                    scores: vec![0.; 2],
                    accumulator: vec![0.; 2],
                })
                .collect();
            let mut out = [0.; 6];
            attention(
                &qkv,
                &mut out,
                &mut scratch,
                &[2, 1],
                &c,
                &work,
                threads,
                &Cancellation::default(),
            )
            .unwrap();
            assert_eq!(out, [2., 4., 2., 4., 100., 200.]);
            let cancel = Cancellation::default();
            cancel.cancel();
            assert_eq!(
                attention(
                    &qkv,
                    &mut out,
                    &mut scratch,
                    &[2, 1],
                    &c,
                    &work,
                    threads,
                    &cancel
                )
                .unwrap_err()
                .code,
                crate::ErrorCode::Cancelled
            );
        }
    }
}

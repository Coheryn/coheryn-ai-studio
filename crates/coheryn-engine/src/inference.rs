use crate::quant;
use crate::tensor::{self, Workspace};
use crate::{
    Descriptor, Error, ErrorCode, ModelConfig, ParseLimits, ResourceLimits, Result, compat, cpu,
    gguf::{self, Tensor},
    memory::{self, Budget, MemoryStats, Reservation},
    sampling::{Sampler, Sampling},
    storage::Storage,
    tokenizer::{Message, TextTokenizer, Utf8Stream},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    path::Path,
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

mod batch;
mod generation;
pub(crate) mod vlm;
pub use batch::BatchStep;
pub use generation::{BATCH_PROFILE, BatchingMetadata, GenerationState};

const IO_BYTES: usize = 256 << 10;
type LayerObserver<'a> = dyn FnMut(&str, &[f32]) -> Result<()> + 'a;
pub type TraceObserver<'a> = dyn FnMut(usize, &str, &[f32]) -> Result<()> + 'a;
#[derive(Default, Clone)]
pub struct Cancellation(Arc<AtomicBool>, Option<Instant>);
impl Cancellation {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
    pub fn check(&self) -> Result<()> {
        if self.0.load(Ordering::Relaxed) {
            Err(Error::new(ErrorCode::Cancelled, "generation cancelled"))
        } else if self.1.is_some_and(|deadline| Instant::now() >= deadline) {
            Err(Error::new(
                ErrorCode::DeadlineExceeded,
                "generation deadline exceeded",
            ))
        } else {
            Ok(())
        }
    }
    pub fn with_timeout(&self, seconds: Option<u64>) -> Result<Self> {
        let deadline = match seconds {
            Some(n) if (1..=86400).contains(&n) => Some(
                Instant::now()
                    .checked_add(std::time::Duration::from_secs(n))
                    .ok_or_else(|| Error::request("deadline overflow"))?,
            ),
            Some(_) => return Err(Error::request("timeout_seconds must be in 1..=86400")),
            None => None,
        };
        Ok(Self(
            self.0.clone(),
            match (self.1, deadline) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            },
        ))
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationOptions {
    pub max_tokens: usize,
    pub sampling: Sampling,
    pub stop: Vec<String>,
    #[serde(default)]
    pub stop_tokens: Vec<u32>,
    #[serde(default)]
    pub penalties: crate::sampling::Penalties,
    #[serde(default)]
    pub response_format: crate::constraints::ResponseFormat,
    /// Internal, derived only from validated single-function strict tool contracts.
    #[serde(skip)]
    pub strict_tool: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_seconds: Option<u64>,
    /// Preserve qualified tool delimiters for the native parser; other control tokens stay hidden.
    #[serde(default)]
    pub tool_markers: bool,
}
impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            sampling: Sampling::default(),
            stop: vec![],
            stop_tokens: vec![],
            penalties: crate::sampling::Penalties::default(),
            response_format: crate::constraints::ResponseFormat::Text,
            strict_tool: false,
            timeout_seconds: None,
            tool_markers: false,
        }
    }
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
    ToolCalls,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct TokenEvent {
    pub token_id: Option<u32>,
    pub text: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vision: Option<Box<VisionMetadata>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batching: Option<Box<BatchingMetadata>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_metadata: Option<Box<SamplingMetadata>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constraint_validation: Option<Box<ConstraintValidation>>,
    pub text: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<crate::tools::ToolCall>,
    pub token_ids: Vec<u32>,
    pub prompt_tokens: usize,
    #[serde(default)]
    pub reused_prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: FinishReason,
    pub prefill_ms: f64,
    pub tokenization_ms: f64,
    pub decode_ms: f64,
    pub output_delivery_ms: f64,
    pub first_token_ms: Option<f64>,
    pub memory: MemoryStats,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cooperative: Option<Box<VisionExecution>>,
    pub profile: String,
    pub language_sha256: String,
    pub vision_sha256: String,
    pub images: usize,
    pub vision_ms: f64,
    pub effective_positions: u32,
    pub peak_request_reserved_bytes: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionExecution {
    pub profile: String,
    pub units: u64,
    pub yields: u64,
    pub compute_lease_ms: f64,
    pub waiting_for_compute_ms: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintValidation {
    pub profile: String,
    pub valid: bool,
    pub target: String,
    pub schema_fingerprint: String,
    pub compile_ms: f64,
    pub mask_ms: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingMetadata {
    pub sampling_ms: f64,
    pub profile: String,
    pub settings: Sampling,
    pub penalties: crate::sampling::Penalties,
    pub history_scope: String,
    pub tokenizer_fingerprint: String,
    pub backend: String,
    pub cache_dtype: String,
    pub threads: usize,
    pub prefill_chunk_tokens: usize,
    pub engine_version: String,
}

pub struct Model {
    cache_owner: Arc<()>,
    pub descriptor: Descriptor,
    pub config: ModelConfig,
    pub tokenizer: TextTokenizer,
    storage: Storage,
    tensors: HashMap<String, Tensor>,
    budget: Budget,
    limits: ResourceLimits,
    _model_reservation: Reservation,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionSettings {
    pub threads: usize,
    pub prefill_chunk_tokens: usize,
}
impl Default for ExecutionSettings {
    fn default() -> Self {
        Self {
            threads: 1,
            prefill_chunk_tokens: 1,
        }
    }
}
impl ExecutionSettings {
    pub fn validate(&self) -> Result<()> {
        memory::worker_bytes(self.threads)?;
        if !(1..=64).contains(&self.prefill_chunk_tokens) {
            return Err(Error::request("prefill chunks must contain 1..=64 tokens"));
        }
        Ok(())
    }
}
/// A queue-held reservation transferred to execution. Validated against the model's budget.
pub struct SessionAdmission {
    pub constraints: Option<Reservation>,
    pub context: usize,
    pub reservation: Reservation,
    pub io_bytes: usize,
    pub execution: ExecutionSettings,
}
enum RowInput<'a> {
    Token(u32),
    Embedding(&'a [f32]),
}
struct ForwardRow<'a> {
    input: RowInput<'a>,
    rotary: Option<vlm::RotaryPosition>,
    logits: bool,
}
impl Model {
    pub fn load(path: impl AsRef<Path>, limits: ResourceLimits) -> Result<Self> {
        Self::load_with_budget(path, limits, Budget::new(limits.max_memory))
    }
    /// Share one admission budget across resident models without copying their weights.
    pub fn load_with_budget(
        path: impl AsRef<Path>,
        limits: ResourceLimits,
        budget: Budget,
    ) -> Result<Self> {
        if limits.context == 0 {
            return Err(Error::request("context must be positive"));
        }
        let mut parse_limits = ParseLimits::default();
        parse_limits.max_metadata_allocation = parse_limits
            .max_metadata_allocation
            .min(limits.max_memory / 2);
        let descriptor = gguf::inspect(&path, parse_limits)?;
        let report = compat::validate(&descriptor);
        if !report.executable {
            return Err(report.diagnostics.into_iter().next().unwrap());
        }
        let config = report.config.ok_or_else(|| {
            Error::new(
                ErrorCode::UnsupportedModelArchitecture,
                "this model requires the independent embedding interface, not text generation",
            )
        })?;
        if limits.context > config.context {
            return Err(Error::request(format!(
                "requested context exceeds model context {}",
                config.context
            )));
        }
        // Conservative reservation for parser structures, BPE maps, merge tables and library overhead.
        // This is an admission estimate, not instrumentation of the allocator inside tokenizers.
        let metadata_bytes = memory::model_bytes(&descriptor, &config)?;
        let model_reservation = budget.reserve(metadata_bytes)?;
        // Check feasibility before constructing the tokenizer or allocating a session.
        drop(budget.reserve(memory::session_bytes(&config, limits.context)?)?);
        let tokenizer = TextTokenizer::from_descriptor(&descriptor)?;
        let storage = Storage::open(path.as_ref(), &descriptor)?;
        let tensors = descriptor
            .tensors
            .iter()
            .map(|t| (t.name.clone(), t.clone()))
            .collect();
        Ok(Self {
            cache_owner: Arc::new(()),
            descriptor,
            config,
            tokenizer,
            storage,
            tensors,
            budget,
            limits,
            _model_reservation: model_reservation,
        })
    }
    pub fn memory(&self) -> MemoryStats {
        self.budget.stats()
    }
    pub fn context_limit(&self) -> usize {
        self.limits.context
    }
    pub fn reserve_session(&self, context: usize, overhead_bytes: u64) -> Result<SessionAdmission> {
        self.reserve_session_with_io(context, overhead_bytes, IO_BYTES)
    }
    pub fn reserve_session_with_io(
        &self,
        context: usize,
        overhead_bytes: u64,
        io_bytes: usize,
    ) -> Result<SessionAdmission> {
        self.reserve_session_with_execution(
            context,
            overhead_bytes,
            io_bytes,
            ExecutionSettings::default(),
        )
    }
    pub fn reserve_session_with_execution(
        &self,
        context: usize,
        overhead_bytes: u64,
        io_bytes: usize,
        execution: ExecutionSettings,
    ) -> Result<SessionAdmission> {
        execution.validate()?;
        if context == 0 || context > self.limits.context {
            return Err(Error::request("context exceeds loaded capacity"));
        }
        let bytes = memory::session_bytes_for(&self.config, context, io_bytes, execution.threads)?
            .checked_add(overhead_bytes)
            .ok_or_else(|| Error::resource("session admission overflow"))?;
        Ok(SessionAdmission {
            constraints: None,
            context,
            reservation: self.budget.reserve(bytes)?,
            io_bytes,
            execution,
        })
    }
    pub fn render(&self, messages: &[Message]) -> Result<String> {
        self.tokenizer.render(messages)
    }
    fn tensor(&self, name: &str) -> Result<Tensor> {
        self.tensors
            .get(name)
            .cloned()
            .ok_or_else(|| Error::invalid(format!("missing tensor {name}")))
    }
    fn vector(
        &self,
        name: &str,
        row: usize,
        output: &mut [f32],
        workspace: &mut Workspace,
    ) -> Result<()> {
        let t = self.tensor(name)?;
        tensor::read_row(
            &self.storage,
            self.descriptor.data_offset,
            &t,
            row,
            output,
            workspace,
        )
    }
    fn matvec(
        &self,
        name: &str,
        input: &[f32],
        out: &mut [f32],
        cancel: &Cancellation,
        workspace: &mut Workspace,
    ) -> Result<()> {
        let t = self.tensor(name)?;
        tensor::matmul(
            &self.storage,
            self.descriptor.data_offset,
            &t,
            input,
            out,
            workspace,
            cancel,
        )
    }
    fn forward(
        &self,
        token: u32,
        session: &mut Session,
        cancel: &Cancellation,
        trace: Option<&mut LayerObserver<'_>>,
    ) -> Result<()> {
        self.forward_row(
            ForwardRow {
                input: RowInput::Token(token),
                rotary: None,
                logits: true,
            },
            session,
            cancel,
            trace,
        )
    }
    fn forward_row(
        &self,
        row: ForwardRow<'_>,
        session: &mut Session,
        cancel: &Cancellation,
        mut trace: Option<&mut LayerObserver<'_>>,
    ) -> Result<()> {
        cancel.check()?;
        self.storage.unchanged()?;
        if session.position >= session.context {
            return Err(Error::resource("context capacity exhausted"));
        }
        let ledger_token = match row.input {
            RowInput::Token(token) => token,
            RowInput::Embedding(_) => u32::MAX,
        };
        let c = self.config.clone();
        let s = &mut session.scratch;
        let workspace = &mut session.workspace;
        let rotary = row
            .rotary
            .map(|position| vlm::RotaryCache::new(c.head_dim, c.rope_base, position))
            .transpose()?;
        match row.input {
            RowInput::Token(token) => {
                if token as usize >= c.vocabulary {
                    return Err(Error::request("token ID outside vocabulary"));
                }
                self.vector("token_embd.weight", token as usize, &mut s.x, workspace)?;
            }
            RowInput::Embedding(values) => {
                if values.len() != c.hidden || values.iter().any(|v| !v.is_finite()) {
                    return Err(Error::invalid("invalid projected input embedding"));
                }
                s.x.copy_from_slice(values);
            }
        }
        for i in 0..c.layers {
            cancel.check()?;
            let prefix = format!("blk.{i}");
            self.vector(
                &format!("{prefix}.attn_norm.weight"),
                0,
                &mut s.norm_weight,
                workspace,
            )?;
            cpu::rms_norm(&s.x, &s.norm_weight, c.norm_epsilon, &mut s.normalized);
            for (name, out) in [
                ("attn_q", &mut s.q),
                ("attn_k", &mut s.k),
                ("attn_v", &mut s.v),
            ] {
                self.matvec(
                    &format!("{prefix}.{name}.weight"),
                    &s.normalized,
                    out,
                    cancel,
                    workspace,
                )?;
                self.vector(
                    &format!("{prefix}.{name}.bias"),
                    0,
                    &mut s.bias[..out.len()],
                    workspace,
                )?;
                for (x, b) in out.iter_mut().zip(&s.bias) {
                    *x += b;
                }
            }
            if let Some(rotation) = &rotary {
                rotation.apply(&mut s.q)?;
                rotation.apply(&mut s.k)?;
            } else {
                cpu::rope(&mut s.q, c.head_dim, session.position, c.rope_base);
                cpu::rope(&mut s.k, c.head_dim, session.position, c.rope_base);
            }
            session.keys[i].append(&s.k, session.context, &mut session.buffers)?;
            session.values[i].append(&s.v, session.context, &mut session.buffers)?;
            // A non-output prompt row needs the final block's K/V, but its final
            // attention/FFN state cannot affect any later layer or token.
            if !row.logits && i + 1 == c.layers {
                break;
            }
            cpu::attention_rows(
                &s.q,
                &session.keys[i].data,
                &session.values[i].data,
                c.kv_heads,
                c.head_dim,
                cpu::AttentionWorkspace {
                    output: &mut s.attention,
                    scores: &mut s.scores,
                    accumulator: &mut s.attention_accumulator,
                },
                || cancel.check(),
            )?;
            self.matvec(
                &format!("{prefix}.attn_output.weight"),
                &s.attention,
                &mut s.projection,
                cancel,
                workspace,
            )?;
            for (x, p) in s.x.iter_mut().zip(&s.projection) {
                *x += p;
            }
            self.vector(
                &format!("{prefix}.ffn_norm.weight"),
                0,
                &mut s.norm_weight,
                workspace,
            )?;
            cpu::rms_norm(&s.x, &s.norm_weight, c.norm_epsilon, &mut s.normalized);
            self.matvec(
                &format!("{prefix}.ffn_gate.weight"),
                &s.normalized,
                &mut s.gate,
                cancel,
                workspace,
            )?;
            self.matvec(
                &format!("{prefix}.ffn_up.weight"),
                &s.normalized,
                &mut s.up,
                cancel,
                workspace,
            )?;
            cpu::swiglu(&mut s.gate, &s.up);
            self.matvec(
                &format!("{prefix}.ffn_down.weight"),
                &s.gate,
                &mut s.projection,
                cancel,
                workspace,
            )?;
            for (x, p) in s.x.iter_mut().zip(&s.projection) {
                *x += p;
            }
            if let Some(ref mut cb) = trace {
                cb(&format!("blk.{i}.output"), &s.x)?;
            }
        }
        if row.logits {
            self.vector("output_norm.weight", 0, &mut s.norm_weight, workspace)?;
            cpu::rms_norm(&s.x, &s.norm_weight, c.norm_epsilon, &mut s.normalized);
            let output = if self.tensors.contains_key("output.weight") {
                "output.weight"
            } else {
                "token_embd.weight"
            };
            self.matvec(output, &s.normalized, &mut s.logits, cancel, workspace)?;
            if let Some(ref mut cb) = trace {
                cb("logits", &s.logits)?;
            }
        }
        session.position += 1;
        session.tokens.push(ledger_token);
        Ok(())
    }
    /// Teacher-forced full logits and optional intermediate activations for oracle tests.
    pub fn trace(
        &self,
        tokens: &[u32],
        cancel: &Cancellation,
        callback: &mut TraceObserver<'_>,
    ) -> Result<()> {
        if tokens.is_empty() || tokens.len() > self.limits.context {
            return Err(Error::request(
                "trace token count exceeds context or is empty",
            ));
        }
        let mut session = Session::new(&self.config, self.limits.context, &self.budget)?;
        for (i, token) in tokens.iter().enumerate() {
            let mut cb = |name: &str, values: &[f32]| callback(i, name, values);
            self.forward(*token, &mut session, cancel, Some(&mut cb))?;
        }
        Ok(())
    }
    pub fn generate(
        &self,
        prompt: &str,
        options: &GenerationOptions,
        cancel: &Cancellation,
        emit: impl FnMut(TokenEvent) -> Result<()>,
    ) -> Result<GenerationResult> {
        self.generate_at_context(prompt, options, self.limits.context, cancel, emit)
    }
    pub fn generate_at_context(
        &self,
        prompt: &str,
        options: &GenerationOptions,
        context: usize,
        cancel: &Cancellation,
        emit: impl FnMut(TokenEvent) -> Result<()>,
    ) -> Result<GenerationResult> {
        if context == 0 || context > self.limits.context {
            return Err(Error::request(
                "requested context exceeds the loaded context limit",
            ));
        }
        let reservation = self
            .budget
            .reserve(memory::session_bytes(&self.config, context)?)?;
        self.generate_admitted(
            prompt,
            options,
            SessionAdmission {
                constraints: None,
                context,
                reservation,
                io_bytes: IO_BYTES,
                execution: ExecutionSettings::default(),
            },
            cancel,
            emit,
        )
    }
    pub fn generate_admitted(
        &self,
        prompt: &str,
        options: &GenerationOptions,
        admission: SessionAdmission,
        cancel: &Cancellation,
        emit: impl FnMut(TokenEvent) -> Result<()>,
    ) -> Result<GenerationResult> {
        self.generate_impl(prompt, options, admission, cancel, None, emit)
    }
    pub fn generate_cached_admitted(
        &self,
        prompt: &str,
        options: &GenerationOptions,
        admission: SessionAdmission,
        cancel: &Cancellation,
        cache: &mut Option<PrefixCache>,
        emit: impl FnMut(TokenEvent) -> Result<()>,
    ) -> Result<GenerationResult> {
        self.generate_impl(prompt, options, admission, cancel, Some(cache), emit)
    }
    fn generate_impl(
        &self,
        prompt: &str,
        options: &GenerationOptions,
        admission: SessionAdmission,
        cancel: &Cancellation,
        mut cache: Option<&mut Option<PrefixCache>>,
        mut emit: impl FnMut(TokenEvent) -> Result<()>,
    ) -> Result<GenerationResult> {
        let prior = cache.as_deref_mut().and_then(Option::take);
        let mut state = self.begin_generation(prompt, options, admission, cancel, prior)?;
        while !state.is_finished() {
            let result = if state.needs_forward() {
                self.step_generation(&mut state)
            } else {
                Ok(())
            };
            if let Err(error) = result {
                if error.code == ErrorCode::Cancelled && !state.is_prefilling() {
                    state.cancelled();
                    break;
                }
                return Err(error);
            }
            match state.sample(self) {
                Ok(Some(event)) => {
                    let start = Instant::now();
                    emit(event)?;
                    state.record_delivery(start.elapsed().as_secs_f64() * 1000.);
                }
                Ok(None) => {}
                Err(error) if error.code == ErrorCode::Cancelled && !state.is_prefilling() => {
                    state.cancelled();
                    break;
                }
                Err(error) => return Err(error),
            }
        }
        let (result, new_cache) = state.finish(self, cache.is_some(), emit)?;
        if let Some(output) = cache {
            *output = new_cache;
        }
        Ok(result)
    }
}

pub struct PrefixCache {
    keys: Vec<Kv>,
    values: Vec<Kv>,
    tokens: Vec<u32>,
    position: usize,
    model: Weak<()>,
    key: [u8; 32],
    context: usize,
    io_bytes: usize,
    execution: ExecutionSettings,
    buffers: memory::BufferAccounting,
    reservation: Reservation,
}
impl PrefixCache {
    pub fn is_live(&self) -> bool {
        self.model.strong_count() > 0
    }
    pub fn reserved_bytes(&self) -> u64 {
        self.reservation.bytes()
    }
    pub fn token_count(&self) -> usize {
        self.position
    }
}

#[cfg(test)]
#[path = "../tests/common/mod.rs"]
pub(crate) mod cache_model_fixture;
#[cfg(test)]
mod cache_tests {
    use super::*;
    fn fixture() -> (tempfile::TempDir, Model) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("cache.gguf");
        let mut bytes = cache_model_fixture::tiny_model();
        std::fs::write(&path, &bytes).unwrap();
        let descriptor = crate::inspect(&path, Default::default()).unwrap();
        for tensor in &descriptor.tensors {
            let offset = (descriptor.data_offset + tensor.offset) as usize;
            for index in 0..tensor.bytes as usize / 4 {
                let value = if tensor.dimensions.len() == 1 && tensor.name.ends_with("norm.weight")
                {
                    1.0
                } else {
                    ((index * 7 + 3) % 29) as f32 * 0.015625 - 0.1875
                };
                bytes[offset + index * 4..offset + index * 4 + 4]
                    .copy_from_slice(&value.to_le_bytes());
            }
        }
        std::fs::write(&path, bytes).unwrap();
        let model = Model::load(
            path,
            ResourceLimits {
                context: 128,
                max_memory: 64 << 20,
            },
        )
        .unwrap();
        (directory, model)
    }
    fn run(
        model: &Model,
        prompt: &str,
        cache: &mut Option<PrefixCache>,
        maximum: usize,
    ) -> GenerationResult {
        let admission = model
            .reserve_cached_session(128, IO_BYTES, ExecutionSettings::default(), cache)
            .unwrap();
        model
            .generate_cached_admitted(
                prompt,
                &GenerationOptions {
                    max_tokens: maximum,
                    ..Default::default()
                },
                admission,
                &Cancellation::default(),
                cache,
                |_| Ok(()),
            )
            .unwrap()
    }
    #[test]
    fn cached_rows_and_teacher_forced_logits_match_fresh_nonconstant_graph() {
        let (_directory, model) = fixture();
        let baseline = model.memory().reserved;
        let mut reused = None;
        let mut fresh = None;
        run(&model, "Hello", &mut reused, 0);
        let result = run(&model, "Hello world", &mut reused, 0);
        assert!(result.reused_prompt_tokens > 0);
        run(&model, "Hello world", &mut fresh, 0);
        let one = reused.as_ref().unwrap();
        let two = fresh.as_ref().unwrap();
        assert_eq!(one.tokens, two.tokens);
        assert!(one.keys[0].data.iter().any(|v| *v != 0.));
        for (a, b) in one
            .keys
            .iter()
            .chain(&one.values)
            .zip(two.keys.iter().chain(&two.values))
        {
            assert_eq!(a.data, b.data);
        }
        let tokens = one.tokens.clone();
        let admission = model
            .reserve_cached_session(128, IO_BYTES, ExecutionSettings::default(), &mut reused)
            .unwrap();
        let mut session = Session::admitted_cached(
            &model.config,
            128,
            &model.budget,
            admission.reservation,
            IO_BYTES,
            ExecutionSettings::default(),
            reused.take(),
        )
        .unwrap();
        session.position = tokens.len() - 1;
        session.tokens.truncate(tokens.len() - 1);
        for kv in session.keys.iter_mut().chain(&mut session.values) {
            kv.data
                .truncate((tokens.len() - 1) * model.config.kv_heads * model.config.head_dim);
        }
        model
            .forward(
                *tokens.last().unwrap(),
                &mut session,
                &Cancellation::default(),
                None,
            )
            .unwrap();
        let mut logits = Vec::new();
        model
            .trace(&tokens, &Cancellation::default(), &mut |_, name, values| {
                if name == "logits" {
                    logits = values.to_vec();
                }
                Ok(())
            })
            .unwrap();
        assert_eq!(session.scratch.logits, logits);
        assert!(logits.windows(2).any(|p| p[0] != p[1]));
        drop(session);
        drop(fresh);
        assert_eq!(model.memory().reserved, baseline);
    }
    #[test]
    fn prefix_edits_sampling_cancellation_and_model_reload_invalidate_safely() {
        let (_directory, model) = fixture();
        let mut cache = None;
        run(&model, "first", &mut cache, 4);
        let cold = model
            .generate(
                "first again",
                &GenerationOptions {
                    max_tokens: 4,
                    ..Default::default()
                },
                &Cancellation::default(),
                |_| Ok(()),
            )
            .unwrap();
        let warm = run(&model, "first again", &mut cache, 4);
        assert_eq!(cold.token_ids, warm.token_ids);
        assert!(warm.reused_prompt_tokens > 0);
        assert_eq!(
            run(&model, "different", &mut cache, 0).reused_prompt_tokens,
            0
        );
        let admission = model
            .reserve_cached_session(128, IO_BYTES, ExecutionSettings::default(), &mut cache)
            .unwrap();
        let cancellation = Cancellation::default();
        let trigger = cancellation.clone();
        let result = model.generate_cached_admitted(
            "different suffix",
            &GenerationOptions {
                max_tokens: 4,
                ..Default::default()
            },
            admission,
            &cancellation,
            &mut cache,
            |_| {
                trigger.cancel();
                Err(Error::new(ErrorCode::Cancelled, "test consumer closed"))
            },
        );
        assert!(result.is_err());
        assert!(cache.is_none());
        run(&model, "reload", &mut cache, 0);
        let (_other, other) = fixture();
        assert_eq!(run(&other, "reload", &mut cache, 0).reused_prompt_tokens, 0);
        let changed = ExecutionSettings {
            threads: 2,
            prefill_chunk_tokens: 1,
        };
        let admission = other
            .reserve_cached_session(128, IO_BYTES, changed, &mut cache)
            .unwrap();
        assert!(cache.is_none());
        drop(admission);
        let settings = GenerationOptions {
            tool_markers: false,
            max_tokens: 5,
            sampling: Sampling {
                temperature: 0.7,
                top_k: 8,
                top_p: 0.95,
                seed: 77,
            },
            stop: vec![],
            ..GenerationOptions::default()
        };
        let cold = other
            .generate("seeded", &settings, &Cancellation::default(), |_| Ok(()))
            .unwrap();
        run(&other, "seeded", &mut cache, 0);
        let admission = other
            .reserve_cached_session(128, IO_BYTES, ExecutionSettings::default(), &mut cache)
            .unwrap();
        let warm = other
            .generate_cached_admitted(
                "seeded",
                &settings,
                admission,
                &Cancellation::default(),
                &mut cache,
                |_| Ok(()),
            )
            .unwrap();
        assert_eq!(cold.token_ids, warm.token_ids);
    }
}
impl Model {
    fn cache_key(&self) -> Result<[u8; 32]> {
        let data = serde_json::to_vec(&(
            &self.config,
            &self.descriptor.header_sha256,
            self.descriptor.data_offset,
            self.tokenizer.cache_identity(),
        ))
        .map_err(|_| Error::request("invalid cache graph identity"))?;
        Ok(Sha256::digest(data).into())
    }
    pub fn reserve_cached_session(
        &self,
        context: usize,
        io_bytes: usize,
        execution: ExecutionSettings,
        cache: &mut Option<PrefixCache>,
    ) -> Result<SessionAdmission> {
        if context == 0 || context > self.context_limit() {
            return Err(Error::request("cache context exceeds loaded capacity"));
        }
        if let Some(old) = cache
            && !self.cache_compatible(old, context, io_bytes, execution)?
        {
            *cache = None;
        }
        let credit = cache.as_ref().map_or(0, PrefixCache::reserved_bytes);
        let total = memory::session_bytes_for(&self.config, context, io_bytes, execution.threads)?;
        let needed = total
            .checked_sub(credit)
            .ok_or_else(|| Error::resource("invalid cache credit"))?;
        Ok(SessionAdmission {
            context,
            io_bytes,
            execution,
            reservation: self.budget.reserve(needed)?,
            constraints: None,
        })
    }
    pub fn cache_compatible(
        &self,
        cache: &PrefixCache,
        context: usize,
        io_bytes: usize,
        execution: ExecutionSettings,
    ) -> Result<bool> {
        Ok(
            Weak::ptr_eq(&cache.model, &Arc::downgrade(&self.cache_owner))
                && cache.key == self.cache_key()?
                && cache.context == context
                && cache.io_bytes == io_bytes
                && cache.execution == execution
                && cache
                    .reservation
                    .covers(&self.budget, cache.reservation.bytes()),
        )
    }
}

struct Scratch {
    x: Vec<f32>,
    norm_weight: Vec<f32>,
    normalized: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    bias: Vec<f32>,
    attention: Vec<f32>,
    projection: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    logits: Vec<f32>,
    scores: Vec<f32>,
    attention_accumulator: Vec<f64>,
}
impl Scratch {
    fn new(c: &ModelConfig, context: usize) -> Result<Self> {
        Ok(Self {
            x: memory::zeros(c.hidden)?,
            norm_weight: memory::zeros(c.hidden)?,
            normalized: memory::zeros(c.hidden)?,
            q: memory::zeros(c.hidden)?,
            k: memory::zeros(c.kv_heads * c.head_dim)?,
            v: memory::zeros(c.kv_heads * c.head_dim)?,
            bias: memory::zeros(c.hidden)?,
            attention: memory::zeros(c.hidden)?,
            projection: memory::zeros(c.hidden)?,
            gate: memory::zeros(c.intermediate)?,
            up: memory::zeros(c.intermediate)?,
            logits: memory::zeros(c.vocabulary)?,
            scores: memory::zeros(context)?,
            attention_accumulator: memory::zeros_f64(c.head_dim)?,
        })
    }
}
struct Kv {
    data: Vec<f32>,
}
impl Kv {
    fn append(
        &mut self,
        row: &[f32],
        context: usize,
        buffers: &mut memory::BufferAccounting,
    ) -> Result<()> {
        if self.data.len() / row.len() >= context {
            return Err(Error::resource("KV capacity exceeded"));
        }
        if self.data.capacity() < self.data.len() + row.len() {
            let rows = (self.data.len() / row.len() + 64).min(context);
            let before = self.data.capacity();
            self.data
                .try_reserve_exact(rows * row.len() - self.data.len())
                .map_err(|_| Error::resource("KV allocation failed"))?;
            buffers.grow(((self.data.capacity() - before) * std::mem::size_of::<f32>()) as u64)?;
        }
        self.data.extend_from_slice(row);
        Ok(())
    }
}
struct Session {
    tokens: Vec<u32>,
    position: usize,
    context: usize,
    workspace: Workspace,
    keys: Vec<Kv>,
    values: Vec<Kv>,
    scratch: Scratch,
    buffers: memory::BufferAccounting,
    // Drop observations after the buffers, then release admission capacity last.
    _reservation: Reservation,
}
impl Session {
    fn into_cache(
        mut self,
        model: Weak<()>,
        key: [u8; 32],
        io_bytes: usize,
        execution: ExecutionSettings,
    ) -> Result<PrefixCache> {
        let observed = (self.keys.capacity() + self.values.capacity()) as u64
            * std::mem::size_of::<Kv>() as u64
            + self
                .keys
                .iter()
                .chain(&self.values)
                .map(|k| k.data.capacity() as u64 * 4)
                .sum::<u64>()
            + self.tokens.capacity() as u64 * 4;
        let cache = PrefixCache {
            keys: std::mem::take(&mut self.keys),
            values: std::mem::take(&mut self.values),
            tokens: std::mem::take(&mut self.tokens),
            position: self.position,
            context: self.context,
            model,
            key,
            io_bytes,
            execution,
            buffers: self.buffers.split(observed)?,
            reservation: self._reservation.split(observed + 4096)?,
        };
        Ok(cache)
    }
    fn new(c: &ModelConfig, context: usize, budget: &Budget) -> Result<Self> {
        let reservation = budget.reserve(memory::session_bytes(c, context)?)?;
        Self::admitted(
            c,
            context,
            budget,
            reservation,
            IO_BYTES,
            ExecutionSettings::default(),
        )
    }
    fn admitted(
        c: &ModelConfig,
        context: usize,
        budget: &Budget,
        reservation: Reservation,
        io_bytes: usize,
        execution: ExecutionSettings,
    ) -> Result<Self> {
        Self::admitted_cached(c, context, budget, reservation, io_bytes, execution, None)
    }
    fn admitted_cached(
        c: &ModelConfig,
        context: usize,
        budget: &Budget,
        mut reservation: Reservation,
        io_bytes: usize,
        execution: ExecutionSettings,
        cache: Option<PrefixCache>,
    ) -> Result<Self> {
        execution.validate()?;
        let limit = memory::session_bytes_for(c, context, io_bytes, execution.threads)?;
        let (keys, values, mut tokens, position, mut buffers) = if let Some(cache) = cache {
            reservation.absorb(cache.reservation)?;
            (
                cache.keys,
                cache.values,
                cache.tokens,
                cache.position,
                cache.buffers,
            )
        } else {
            (
                (0..c.layers).map(|_| Kv { data: Vec::new() }).collect(),
                (0..c.layers).map(|_| Kv { data: Vec::new() }).collect(),
                Vec::new(),
                0,
                budget.track_buffers(limit),
            )
        };
        if tokens.capacity() < context {
            tokens
                .try_reserve_exact(context - tokens.len())
                .map_err(|_| Error::resource("cache token ledger allocation failed"))?;
        }
        buffers.relimit(limit)?;
        if !reservation.covers(budget, limit) {
            return Err(Error::resource(
                "session reservation belongs to another budget or is too small",
            ));
        }
        let mut session = Self {
            tokens,
            position,
            context,
            workspace: Workspace::new(
                c.hidden.max(c.intermediate),
                io_bytes,
                execution.threads,
                1,
            )?,
            keys,
            values,
            scratch: Scratch::new(c, context)?,
            _reservation: reservation,
            buffers,
        };
        let s = &session.scratch;
        let floats = [
            &s.x,
            &s.norm_weight,
            &s.normalized,
            &s.q,
            &s.k,
            &s.v,
            &s.bias,
            &s.attention,
            &s.projection,
            &s.gate,
            &s.up,
            &s.logits,
            &s.scores,
            &session.workspace.floats,
            &session.workspace.operand,
        ];
        let bytes = floats
            .iter()
            .try_fold(
                session.workspace.io.capacity() + session.workspace.quantized.capacity_bytes(),
                |sum, v| {
                    v.capacity()
                        .checked_mul(std::mem::size_of::<f32>())
                        .and_then(|n| sum.checked_add(n))
                },
            )
            .and_then(|n| {
                n.checked_add(s.attention_accumulator.capacity() * std::mem::size_of::<f64>())
            })
            .and_then(|n| {
                n.checked_add(
                    (session.keys.capacity() + session.values.capacity())
                        * std::mem::size_of::<Kv>(),
                )
            })
            .ok_or_else(|| Error::resource("buffer capacity accounting overflow"))?;
        let kv_bytes = session
            .keys
            .iter()
            .chain(&session.values)
            .map(|kv| kv.data.capacity() as u64 * 4)
            .sum::<u64>()
            + session.tokens.capacity() as u64 * 4;
        session
            .buffers
            .grow(bytes as u64 + kv_bytes - session.buffers.bytes())?;
        Ok(session)
    }
}
/// Holds possible stop prefixes so a stop split across tokens is never emitted.
struct StopBuffer {
    pending: String,
    stops: Vec<String>,
    matched: bool,
}
impl StopBuffer {
    fn new(stops: Vec<String>) -> Self {
        Self {
            pending: String::new(),
            stops,
            matched: false,
        }
    }
    fn push(&mut self, text: &str, finish: bool) -> (String, bool) {
        if self.matched {
            return (String::new(), true);
        }
        self.pending.push_str(text);
        if let Some(index) = self.stops.iter().filter_map(|s| self.pending.find(s)).min() {
            let safe = self.pending[..index].to_owned();
            self.pending.clear();
            self.matched = true;
            return (safe, true);
        }
        let hold = if finish {
            0
        } else {
            self.stops
                .iter()
                .flat_map(|s| s.char_indices().map(|(n, _)| n))
                .filter(|n| *n > 0)
                .filter(|n| {
                    self.stops.iter().any(|s| {
                        s.is_char_boundary(*n) && *n < s.len() && self.pending.ends_with(&s[..*n])
                    })
                })
                .max()
                .unwrap_or(0)
        };
        let safe = self.pending[..self.pending.len() - hold].to_owned();
        self.pending.drain(..self.pending.len() - hold);
        (safe, false)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stop_sequences_cannot_leak_across_chunks() {
        let mut b = StopBuffer::new(vec!["END".into()]);
        assert_eq!(b.push("hello E", false), ("hello ".into(), false));
        assert_eq!(b.push("N", false), ("".into(), false));
        assert_eq!(b.push("D rest", false), ("".into(), true));
    }
    #[test]
    fn incomplete_stop_is_flushed_at_length() {
        let mut b = StopBuffer::new(vec!["終わり".into()]);
        assert_eq!(b.push("abc終", false), ("abc".into(), false));
        assert_eq!(b.push("", true), ("終".into(), false));
    }
    #[test]
    fn cancellation_is_shared() {
        let c = Cancellation::default();
        let d = c.clone();
        d.cancel();
        assert!(c.check().is_err());
    }
    #[test]
    fn deadlines_are_scoped_preserve_earlier_limits_and_do_not_poison_parent() {
        let parent = Cancellation::default();
        assert!(parent.with_timeout(Some(0)).is_err());
        assert!(parent.with_timeout(Some(86401)).is_err());
        let expired = Cancellation(parent.0.clone(), Some(Instant::now()));
        assert_eq!(
            expired.check().unwrap_err().code,
            ErrorCode::DeadlineExceeded
        );
        assert_eq!(expired.with_timeout(Some(10)).unwrap().1, expired.1);
        assert!(parent.check().is_ok());
        parent.cancel();
        assert_eq!(expired.check().unwrap_err().code, ErrorCode::Cancelled);
    }
}

//! Coheryn-owned Qwen3.5 CPU text adapter with bounded recurrent/KV state.
mod cache;
mod config;
use cache::CacheSlot;
pub use cache::HybridCache;
mod execution;
mod generation;
pub(crate) mod gguf;
mod image_generation;
pub mod image_plan;
pub use generation::GenerationState;
pub use image_generation::PreparedImages;
mod kernels;
mod plan;
pub mod vision;
use crate::{
    Cancellation, Error, ErrorCode, FinishReason, GenerationOptions, GenerationResult,
    ResourceLimits, Result, TokenEvent,
    memory::{Budget, Reservation},
    safetensors::{Limits, TensorSet},
    tokenizer::Message,
};
pub use config::Config;
use execution::{Graph, Session};
pub use plan::{estimate, estimate_shared};
use sha2::{Digest, Sha256};
use std::{io::Write, path::Path, time::Instant};
use tokenizers::Tokenizer;

pub const RESIDENT_METADATA_BYTES: u64 = 256 << 20;
pub const TOKENIZER_SHA256: &str =
    "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42";
pub const TEMPLATE_SHA256: &str =
    "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80";
impl Config {
    /// Uses the same checked reservation arithmetic as actual session construction.
    pub fn session_estimate(
        &self,
        context: usize,
        threads: usize,
    ) -> Result<crate::planner::SessionEstimate> {
        if context == 0 || context > self.context.min(8192) {
            return Err(Error::request(
                "Qwen3.5 context exceeds source/development limit",
            ));
        }
        let total = Session::bytes(self, context, threads)?;
        let kv = crate::memory::product(&[
            self.layers.iter().filter(|v| **v).count(),
            context,
            self.kv_heads,
            self.head_dim,
            2,
            4,
        ])? as u64;
        Ok(crate::planner::SessionEstimate {
            kv_capacity_bytes: kv,
            workspace_bytes: total - kv,
            total_bytes: total,
        })
    }
}
pub type TraceObserver<'a> = dyn FnMut(usize, &str, &[f32]) -> Result<()> + 'a;

pub struct PreparedGeneration {
    admission: plan::Admission,
    options: GenerationOptions,
    context: usize,
    plan: crate::planner::ExecutionPlan,
    owner: std::sync::Arc<()>,
    cache: Option<CacheSlot>,
}
impl PreparedGeneration {
    pub fn plan(&self) -> &crate::planner::ExecutionPlan {
        &self.plan
    }
    pub fn request_reserved_bytes(&self) -> u64 {
        self.admission.reservation.bytes() + self.cache.as_ref().map_or(0, CacheSlot::bytes)
    }
}

pub struct Model {
    config: Config,
    set: TensorSet,
    graph: Graph,
    vision: Option<vision::VisionConfig>,
    tokenizer: Tokenizer,
    template: String,
    budget: Budget,
    limits: ResourceLimits,
    _resident: Reservation,
    owner: std::sync::Arc<()>,
}
fn unsupported(m: impl Into<String>) -> Error {
    Error::new(ErrorCode::UnsupportedFeature, m)
}
fn asset(root: &Path, name: &str, limit: u64) -> Result<Vec<u8>> {
    crate::safetensors::bounded_bytes(
        &root.join(name),
        usize::try_from(limit).map_err(|_| Error::resource("asset byte limit overflow"))?,
    )
}
impl Model {
    pub fn load(
        path: impl AsRef<Path>,
        limits: ResourceLimits,
        cancel: &Cancellation,
    ) -> Result<Self> {
        Self::load_with_budget(path, limits, Budget::new(limits.max_memory), cancel)
    }
    pub fn load_with_budget(
        path: impl AsRef<Path>,
        limits: ResourceLimits,
        budget: Budget,
        cancel: &Cancellation,
    ) -> Result<Self> {
        if budget.stats().limit != limits.max_memory {
            return Err(Error::request("model limit differs from shared budget"));
        }
        cancel.check()?;
        if limits.context == 0 || limits.context > 8192 {
            return Err(Error::request(
                "Qwen3.5 development context must be in 1..=8192",
            ));
        }
        let resident = budget.reserve(RESIDENT_METADATA_BYTES)?;
        let selected = path.as_ref();
        let root = if selected.is_dir() {
            selected
        } else {
            selected.parent().unwrap_or(Path::new("."))
        };
        let source = asset(root, "config.json", 1 << 20)?;
        let json = crate::safetensors::configuration(&source)?;
        let (config, set) = if gguf::is_source(selected)? {
            let descriptor = crate::inspect(selected, crate::ParseLimits::default())?;
            gguf::open(selected, &descriptor, &json, cancel)?
        } else {
            (
                Config::parse(&json)?,
                TensorSet::open(selected, Limits::default(), cancel)?,
            )
        };
        if limits.context > config.context {
            return Err(Error::request(
                "requested context exceeds source model context",
            ));
        }
        config.validate_tensors(&set)?;
        // Reject infeasible context before constructing tokenizer or session buffers.
        drop(budget.reserve(Session::bytes(&config, limits.context, 1)?)?);
        let graph = Graph::bind(&set, &config)?;
        // A missing or unqualified image component never disables compatible text execution.
        let vision = if root.join("preprocessor_config.json").try_exists()? {
            let bytes = asset(root, "preprocessor_config.json", 1 << 20)?;
            vision::VisionConfig::parse(&json, &bytes, &set).ok()
        } else {
            None
        };
        cancel.check()?;
        let token_bytes = asset(root, "tokenizer.json", 16 << 20)?;
        if crate::storage::digest_hex(Sha256::digest(&token_bytes).into()) != TOKENIZER_SHA256 {
            return Err(Error::new(
                ErrorCode::UnsupportedTokenizer,
                "unqualified Qwen3.5 tokenizer profile",
            ));
        }
        let tokenizer = Tokenizer::from_bytes(&token_bytes)
            .map_err(|e| Error::new(ErrorCode::UnsupportedTokenizer, e.to_string()))?;
        if tokenizer.get_vocab_size(true) > config.vocabulary
            || tokenizer.token_to_id("<|endoftext|>") != Some(config.eos)
            || tokenizer.token_to_id("<|im_end|>").is_none()
        {
            return Err(Error::new(
                ErrorCode::UnsupportedTokenizer,
                "Qwen3.5 vocabulary/special-token mismatch",
            ));
        }
        let template = asset(root, "chat_template.jinja", 16 << 10)?;
        if crate::storage::digest_hex(Sha256::digest(&template).into()) != TEMPLATE_SHA256 {
            return Err(Error::new(
                ErrorCode::UnsupportedChatTemplate,
                "unqualified Qwen3.5 chat template",
            ));
        }
        let template = String::from_utf8(template).map_err(|_| {
            Error::new(ErrorCode::UnsupportedChatTemplate, "invalid template UTF-8")
        })?;
        cancel.check()?;
        Ok(Self {
            config,
            set,
            graph,
            vision,
            tokenizer,
            template,
            budget,
            limits,
            _resident: resident,
            owner: std::sync::Arc::new(()),
        })
    }
    pub fn is_gguf(&self) -> bool {
        self.set.is_gguf()
    }
    pub fn configuration(&self) -> &Config {
        &self.config
    }
    pub fn memory(&self) -> crate::memory::MemoryStats {
        self.budget.stats()
    }
    pub fn count_tokens(&self, prompt: &str) -> Result<usize> {
        Ok(self.tokenize(prompt)?.len())
    }
    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>> {
        let ids = self.tokenize(prompt)?;
        if ids.len() > self.limits.context {
            return Err(Error::request("prompt tokens exceed loaded context"));
        }
        Ok(ids)
    }
    fn tokenize(&self, prompt: &str) -> Result<Vec<u32>> {
        if prompt.is_empty() || prompt.len() > 1 << 20 {
            return Err(Error::request(
                "prompt must contain 1..=1048576 UTF-8 bytes",
            ));
        }
        let ids = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| Error::new(ErrorCode::UnsupportedTokenizer, e.to_string()))?
            .get_ids()
            .to_vec();
        if ids.is_empty() || ids.iter().any(|t| *t as usize >= self.config.vocabulary) {
            return Err(Error::request("prompt tokens exceed context/vocabulary"));
        }
        Ok(ids)
    }
    pub fn render(&self, messages: &[Message]) -> Result<String> {
        if messages.is_empty() || messages.len() > 256 || !messages.iter().any(|m| m.role == "user")
        {
            return Err(Error::request(
                "Qwen3.5 chat requires bounded messages and a user turn",
            ));
        }
        let mut total = 0usize;
        let mut values = Vec::new();
        for (i, m) in messages.iter().enumerate() {
            total = total
                .checked_add(m.content.len())
                .filter(|n| *n <= 1 << 20)
                .ok_or_else(|| Error::resource("chat source exceeds one MiB"))?;
            if !matches!(m.role.as_str(), "user" | "assistant" | "system")
                || (m.role == "system" && i != 0)
                || !m.tool_calls.is_empty()
                || m.tool_call_id.is_some()
                || m.content.contains("<|")
                || m.content.contains("<tool_")
                || m.content.contains("<think>")
                || m.content.contains("</think>")
            {
                return Err(unsupported(
                    "Qwen3.5 text profile rejects unsupported roles/tools/control delimiters",
                ));
            }
            values.push(serde_json::json!({"role":m.role,"content":m.content,"reasoning_content":"","tool_calls":[]}));
        }
        self.render_values(values)
    }
    pub(super) fn render_values(&self, values: Vec<serde_json::Value>) -> Result<String> {
        let mut env = minijinja::Environment::new();
        env.set_fuel(Some(100_000));
        env.set_recursion_limit(32);
        // Only these string methods are reachable in the qualified text-only profile.
        // Other methods/roles require an explicit profile extension and parity evidence.
        env.set_unknown_method_callback(|_, value, method, args| {
            let invalid = || {
                minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    "unsupported bounded string method",
                )
            };
            let text = value.as_str().ok_or_else(invalid)?;
            let [argument] = args else {
                return Err(invalid());
            };
            let pattern = argument.as_str().ok_or_else(invalid)?;
            match method {
                "startswith" => Ok(text.starts_with(pattern).into()),
                "endswith" => Ok(text.ends_with(pattern).into()),
                _ => Err(invalid()),
            }
        });
        env.add_template("chat", &self.template)
            .map_err(|e| Error::new(ErrorCode::UnsupportedChatTemplate, e.to_string()))?;
        env.add_function(
            "raise_exception",
            |message: String| -> std::result::Result<String, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    message,
                ))
            },
        );
        let mut output = BoundedOutput(Vec::new());
        env.get_template("chat").map_err(|e|Error::new(ErrorCode::UnsupportedChatTemplate,e.to_string()))?.render_captured_to(minijinja::context!(messages=>minijinja::value::Serde(values),tools=>minijinja::value::Serde(Vec::<serde_json::Value>::new()),add_generation_prompt=>true,enable_thinking=>false,add_vision_id=>false),&mut output).map_err(|e|Error::new(ErrorCode::UnsupportedChatTemplate,e.to_string()))?;
        String::from_utf8(output.0)
            .map_err(|_| Error::new(ErrorCode::UnsupportedChatTemplate, "rendered invalid UTF-8"))
    }
    pub fn trace(
        &self,
        tokens: &[u32],
        threads: usize,
        cancel: &Cancellation,
        callback: &mut TraceObserver<'_>,
    ) -> Result<()> {
        if tokens.is_empty()
            || tokens.len() > self.limits.context
            || tokens
                .iter()
                .any(|id| *id as usize >= self.config.vocabulary)
        {
            return Err(Error::request("invalid trace token/context range"));
        }
        let _reservation =
            self.budget
                .reserve(Session::bytes(&self.config, self.limits.context, threads)?)?;
        let mut session = Session::new(&self.config, self.limits.context, threads)?;
        let mut accounting =
            self.budget
                .track_buffers(Session::bytes(&self.config, self.limits.context, threads)?);
        accounting.grow(session.capacity_bytes())?;
        for (position, token) in tokens.iter().enumerate() {
            self.graph.forward(
                &self.set,
                &self.config,
                *token,
                &mut session,
                true,
                cancel,
                Some(&mut |name, data| callback(position, name, data)),
                None,
            )?;
        }
        Ok(())
    }
    pub fn generate(
        &self,
        prompt: &str,
        options: &GenerationOptions,
        threads: usize,
        cancel: &Cancellation,
        emit: &mut dyn FnMut(TokenEvent) -> Result<()>,
    ) -> Result<GenerationResult> {
        Self::check_options(options)?;
        let cancel = cancel.with_timeout(options.timeout_seconds)?;
        cancel.check()?;
        let minimum = self.generation_bytes(self.limits.context, threads)?;
        let admission = plan::Admission::reserve(&self.budget, minimum, minimum, threads)?;
        self.generate_admitted(
            prompt,
            options,
            admission,
            self.limits.context,
            &cancel,
            None,
            emit,
            None,
            &mut None,
        )
    }
    pub fn check_options(options: &GenerationOptions) -> Result<()> {
        if options.max_tokens == 0
            || options.max_tokens > 256
            || options.sampling.temperature != 0.
            || options.sampling.top_k != 0
            || options.sampling.top_p != 1.
            || !options.stop.is_empty()
            || !options.stop_tokens.is_empty()
            || options.penalties.repetition_penalty != 1.
            || options.penalties.frequency_penalty != 0.
            || options.penalties.presence_penalty != 0.
            || options.response_format.enabled()
            || options.strict_tool
            || options.tool_markers
        {
            return Err(unsupported(
                "Qwen3.5 initial CPU text profile supports bounded greedy generation without tools/constraints/custom stops",
            ));
        }
        Ok(())
    }
    fn generation_bytes(&self, context: usize, threads: usize) -> Result<u64> {
        Session::bytes(&self.config, context, threads)?
            .checked_add(crate::planner::REQUEST_BYTES)
            .ok_or_else(|| Error::resource("generation request byte overflow"))
    }
    pub(crate) fn generate_profiled(
        &self,
        prompt: &str,
        options: &GenerationOptions,
        execution: crate::adapters::Execution<'_>,
        cancel: &Cancellation,
        emit: &mut dyn FnMut(TokenEvent) -> Result<()>,
    ) -> Result<GenerationResult> {
        Self::check_options(options)?;
        let cancel = cancel.with_timeout(options.timeout_seconds)?;
        cancel.check()?;
        let request = crate::catalog::PlanRequest {
            context: self.limits.context,
            concurrency: 1,
            max_memory: self.limits.max_memory,
            profile: execution.profile,
            threads: execution.threads,
            constraints: false,
        };
        let plan = estimate(
            &self.config,
            self.set.tensor_bytes()?,
            &request,
            execution.hardware,
            self.budget.stats(),
            true,
        )?;
        plan.require_admission()?;
        cancel.check()?;
        let admission = plan::Admission::reserve(
            &self.budget,
            self.generation_bytes(self.limits.context, plan.threads_per_session)?,
            plan.additional_reserved_bytes,
            plan.threads_per_session,
        )?;
        self.generate_admitted(
            prompt,
            options,
            admission,
            self.limits.context,
            &cancel,
            None,
            emit,
            None,
            &mut None,
        )
    }
    pub fn context_limit(&self) -> usize {
        self.limits.context
    }
    pub fn weight_bytes(&self) -> Result<u64> {
        self.set.tensor_bytes()
    }
    pub fn prepare_generation(
        &self,
        options: GenerationOptions,
        context: usize,
        execution: crate::adapters::Execution<'_>,
        max_concurrency: usize,
    ) -> Result<PreparedGeneration> {
        Self::check_options(&options)?;
        if context == 0 || context > self.limits.context {
            return Err(Error::request("request context exceeds loaded capacity"));
        }
        let request = crate::catalog::PlanRequest {
            context,
            concurrency: 1,
            max_memory: self.limits.max_memory,
            profile: execution.profile,
            threads: execution.threads,
            constraints: false,
        };
        let plan = estimate_shared(
            &self.config,
            self.set.tensor_bytes()?,
            &request,
            execution.hardware,
            self.budget.stats(),
            true,
            max_concurrency,
        )?;
        plan.require_admission()?;
        let admission = plan::Admission::reserve(
            &self.budget,
            self.generation_bytes(context, plan.threads_per_session)?,
            plan.additional_reserved_bytes,
            plan.threads_per_session,
        )?;
        Ok(PreparedGeneration {
            admission,
            options,
            context,
            plan,
            owner: self.owner.clone(),
            cache: None,
        })
    }
    pub fn generate_prepared(
        &self,
        prompt: &str,
        prepared: PreparedGeneration,
        cancel: &Cancellation,
        cooperate: &mut crate::multimodal::WorkYield<'_>,
        emit: &mut dyn FnMut(TokenEvent) -> Result<()>,
    ) -> Result<GenerationResult> {
        self.generate_prepared_cached(prompt, prepared, cancel, cooperate, emit)
            .map(|(r, _)| r)
    }
    /// Consume an owned admission from this model's budget. The caller may hold it while queued.
    pub fn generate_prepared_cached(
        &self,
        prompt: &str,
        prepared: PreparedGeneration,
        cancel: &Cancellation,
        cooperate: &mut crate::multimodal::WorkYield<'_>,
        emit: &mut dyn FnMut(TokenEvent) -> Result<()>,
    ) -> Result<(GenerationResult, Option<HybridCache>)> {
        if !std::sync::Arc::ptr_eq(&self.owner, &prepared.owner)
            || prepared.context > self.limits.context
        {
            return Err(Error::request(
                "generation admission belongs to another model instance",
            ));
        }
        let cancel = cancel.with_timeout(prepared.options.timeout_seconds)?;
        let mut cache = None;
        let result = self.generate_admitted(
            prompt,
            &prepared.options,
            prepared.admission,
            prepared.context,
            &cancel,
            Some(cooperate),
            emit,
            prepared.cache,
            &mut cache,
        )?;
        Ok((result, cache))
    }
    #[allow(clippy::too_many_arguments)]
    fn generate_admitted(
        &self,
        prompt: &str,
        options: &GenerationOptions,
        admission: plan::Admission,
        context: usize,
        cancel: &Cancellation,
        mut cooperate: Option<&mut crate::multimodal::WorkYield<'_>>,
        emit: &mut dyn FnMut(TokenEvent) -> Result<()>,
        cache_slot: Option<CacheSlot>,
        cache_output: &mut Option<HybridCache>,
    ) -> Result<GenerationResult> {
        let mut state = GenerationState::begin(
            self,
            prompt,
            options.clone(),
            admission,
            context,
            cache_slot,
            cancel,
        )?;
        while !state.is_done() {
            if state.needs_forward() {
                let mut checkpoint = || match cooperate.as_deref_mut() {
                    Some(y) => y(),
                    None => Ok(()),
                };
                self.step_single(&mut state, Some(&mut checkpoint))?;
            }
            if let Some(token) = state.sample(self)? {
                let start = Instant::now();
                emit(token)?;
                state.record_delivery(start.elapsed().as_secs_f64() * 1000.);
            }
        }
        let (result, cache) = self.finish_generation(state)?;
        *cache_output = cache;
        Ok(result)
    }
}
struct BoundedOutput(Vec<u8>);
impl Write for BoundedOutput {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self
            .0
            .len()
            .checked_add(bytes.len())
            .is_none_or(|n| n > 1 << 20)
        {
            return Err(std::io::Error::other("render exceeds one MiB"));
        }
        self.0
            .try_reserve(bytes.len())
            .map_err(|_| std::io::Error::other("render allocation failed"))?;
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

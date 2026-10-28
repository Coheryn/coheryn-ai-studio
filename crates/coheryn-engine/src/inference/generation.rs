//! Resumable request-owned generation. A scheduling unit consumes one model token.
use super::*;

pub const BATCH_PROFILE: &str = "coheryn-cpu-batch-v1";
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BatchingMetadata {
    pub profile: String,
    pub strategy: String,
    pub forward_units: usize,
    pub batched_forward_units: usize,
    pub maximum_batch_width: usize,
    pub prefilled_tokens: usize,
    pub scheduler_wait_ms: f64,
}

/// Owns all mutable sequence state and its reservations. It may move between workers,
/// but cannot execute against another model or be advanced concurrently.
pub struct GenerationState {
    pub(super) session: Option<Session>,
    pub(super) cancel: Cancellation,
    pub(super) error: Option<Error>,
    pub(super) model: Weak<()>,
    options: GenerationOptions,
    execution: ExecutionSettings,
    io_bytes: usize,
    prompt_tokens: Vec<u32>,
    cursor: usize,
    reused: usize,
    pending_forward: Option<u32>,
    sampler: Option<Sampler>,
    stream: Utf8Stream,
    stop: StopBuffer,
    text: String,
    ids: Vec<u32>,
    done: bool,
    reason: FinishReason,
    constraint: Option<crate::constraints::ConstraintDecoder>,
    constraint_buffers: Option<memory::BufferAccounting>,
    constraint_reservation: Option<Reservation>,
    sampling_buffers: Option<memory::BufferAccounting>,
    started: Instant,
    first_token_ms: Option<f64>,
    tokenization_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    output_delivery_ms: f64,
    compile_ms: f64,
    mask_ms: f64,
    sampling_ms: f64,
    pub(super) batching: BatchingMetadata,
}
impl Model {
    pub fn begin_generation(
        &self,
        prompt: &str,
        options: &GenerationOptions,
        admission: SessionAdmission,
        cancel: &Cancellation,
        cache: Option<PrefixCache>,
    ) -> Result<GenerationState> {
        let timed_cancel = cancel.with_timeout(options.timeout_seconds)?;
        let cancel = &timed_cancel;
        let SessionAdmission {
            constraints,
            context,
            reservation,
            io_bytes,
            execution,
        } = admission;
        execution.validate()?;
        for tensor in &self.descriptor.tensors {
            if quant::row_bytes(tensor.dtype, tensor.dimensions[0] as usize)? > io_bytes {
                return Err(Error::resource(
                    "I/O tile cannot hold a validated tensor row",
                ));
            }
        }
        if context == 0 || context > self.limits.context {
            return Err(Error::request(
                "requested context exceeds the loaded context limit",
            ));
        }
        let request_start = Instant::now();
        if options.max_tokens > context
            || options.stop.len() > 16
            || options.stop.iter().any(|s| s.is_empty() || s.len() > 4096)
        {
            return Err(Error::request(
                "invalid generation length or stop sequences",
            ));
        }
        if options.stop_tokens.len() > 16
            || options
                .stop_tokens
                .iter()
                .any(|id| *id as usize >= self.tokenizer.vocabulary())
        {
            return Err(Error::request(
                "stop_tokens must contain at most 16 valid token IDs",
            ));
        }
        if options.response_format.enabled()
            && ((options.tool_markers && !options.strict_tool)
                || !options.stop.is_empty()
                || !options.stop_tokens.is_empty())
        {
            return Err(Error::new(
                ErrorCode::UnsupportedFeature,
                "response_format cannot be combined with tools or custom stops",
            ));
        }
        if options.strict_tool && (!options.tool_markers || !options.response_format.enabled()) {
            return Err(Error::request(
                "strict tool framing requires a constrained tool envelope",
            ));
        }
        let constraint_reservation = if options.response_format.enabled() {
            match constraints {
                Some(held) if held.covers(&self.budget, crate::constraints::RESERVATION_BYTES) => {
                    Some(held)
                }
                Some(_) => return Err(Error::resource("invalid constraint admission reservation")),
                None => Some(self.budget.reserve(crate::constraints::RESERVATION_BYTES)?),
            }
        } else {
            drop(constraints);
            None
        };
        let constraint_start = Instant::now();
        let constraint = match options.response_format.compile()? {
            Some(grammar) => Some(crate::constraints::ConstraintDecoder::with_tool_framing(
                grammar,
                &self.tokenizer,
                cancel,
                options.strict_tool,
            )?),
            None => None,
        };
        let mut constraint_buffers = self
            .budget
            .track_buffers(crate::constraints::RESERVATION_BYTES);
        if let Some(decoder) = &constraint {
            constraint_buffers.grow(decoder.capacity_bytes())?;
        }
        let mut sampler = Sampler::with_penalties(options.sampling.clone(), options.penalties)?;
        let constraint_compile_ms = constraint_start.elapsed().as_secs_f64() * 1000.;
        cancel.check()?;
        let tokenizer_start = Instant::now();
        let prompt_tokens = self.tokenizer.encode(prompt)?;
        let tokenization_ms = tokenizer_start.elapsed().as_secs_f64() * 1000.;
        sampler.history(self.tokenizer.vocabulary(), &prompt_tokens)?;
        if prompt_tokens.is_empty()
            || prompt_tokens
                .len()
                .checked_add(options.max_tokens)
                .is_none_or(|n| n > context)
        {
            return Err(Error::request(
                "prompt must be nonempty and prompt plus max_tokens must fit context",
            ));
        }
        let mut prior = cache;
        if let Some(old) = &prior
            && !self.cache_compatible(old, context, io_bytes, execution)?
        {
            prior = None;
        }
        let mut session = Session::admitted_cached(
            &self.config,
            context,
            &self.budget,
            reservation,
            io_bytes,
            execution,
            prior,
        )?;
        // The existing vocabulary workspace reservation includes logits and sampling.
        // Partition observations too, so the two trackers cannot exceed that reservation.
        let sampling_limit = memory::product(&[self.config.vocabulary, 20])? as u64;
        session.buffers.relimit(
            session
                ._reservation
                .bytes()
                .checked_sub(sampling_limit)
                .ok_or_else(|| Error::resource("sampling workspace reservation underflow"))?,
        )?;
        let mut sampling_buffers = self.budget.track_buffers(sampling_limit);
        sampling_buffers.grow(sampler.capacity_bytes())?;
        // Recompute the last prompt token so logits never need a second retained copy.
        let reused_prompt_tokens = session
            .tokens
            .iter()
            .zip(&prompt_tokens)
            .take_while(|(a, b)| a == b)
            .count()
            .min(prompt_tokens.len() - 1);
        session.position = reused_prompt_tokens;
        session.tokens.truncate(reused_prompt_tokens);
        let row = self.config.kv_heads * self.config.head_dim;
        for kv in session.keys.iter_mut().chain(&mut session.values) {
            kv.data.truncate(reused_prompt_tokens * row);
        }

        Ok(GenerationState {
            session: Some(session),
            cancel: timed_cancel,
            error: None,
            model: Arc::downgrade(&self.cache_owner),
            options: options.clone(),
            execution,
            io_bytes,
            cursor: reused_prompt_tokens,
            reused: reused_prompt_tokens,
            prompt_tokens,
            pending_forward: None,
            sampler: Some(sampler),
            stream: Utf8Stream::default(),
            stop: StopBuffer::new(options.stop.clone()),
            text: String::new(),
            ids: Vec::new(),
            done: false,
            reason: FinishReason::Length,
            constraint,
            constraint_buffers: Some(constraint_buffers),
            constraint_reservation,
            sampling_buffers: Some(sampling_buffers),
            started: request_start,
            first_token_ms: None,
            tokenization_ms,
            prefill_ms: 0.,
            decode_ms: 0.,
            output_delivery_ms: 0.,
            compile_ms: constraint_compile_ms,
            mask_ms: 0.,
            sampling_ms: 0.,
            batching: BatchingMetadata {
                profile: BATCH_PROFILE.into(),
                strategy: "single-sequence".into(),
                ..Default::default()
            },
        })
    }
    /// Original one-sequence graph, retained for standalone and batch parity checks.
    pub fn step_generation(&self, state: &mut GenerationState) -> Result<()> {
        state.check_model(self)?;
        let token = state.next_token()?;
        let prefilling = state.cursor < state.prompt_tokens.len();
        let start = Instant::now();
        self.forward(
            token,
            state.session.as_mut().expect("live session"),
            &state.cancel,
            None,
        )?;
        state.forwarded(start.elapsed().as_secs_f64() * 1000., prefilling, 1);
        Ok(())
    }
}
impl GenerationState {
    fn check_model(&self, model: &Model) -> Result<()> {
        if !Weak::ptr_eq(&self.model, &Arc::downgrade(&model.cache_owner)) {
            return Err(Error::request(
                "generation state belongs to another model instance",
            ));
        }
        Ok(())
    }
    pub fn compatible(&self, other: &Self) -> bool {
        Weak::ptr_eq(&self.model, &other.model)
            && self.execution == other.execution
            && self.io_bytes == other.io_bytes
    }
    pub fn is_finished(&self) -> bool {
        self.done
    }
    pub fn is_prefilling(&self) -> bool {
        self.cursor < self.prompt_tokens.len()
    }
    pub fn needs_forward(&self) -> bool {
        !self.done && (self.is_prefilling() || self.pending_forward.is_some())
    }
    pub fn record_scheduler_wait(&mut self, milliseconds: f64) {
        self.batching.scheduler_wait_ms += milliseconds.max(0.);
    }
    pub fn record_delivery(&mut self, milliseconds: f64) {
        self.output_delivery_ms += milliseconds;
    }
    pub(super) fn next_token(&mut self) -> Result<u32> {
        if let Some(error) = self.error.take() {
            return Err(error);
        }
        self.cancel.check()?;
        if self.done {
            return Err(Error::request("generation already completed"));
        }
        self.prompt_tokens
            .get(self.cursor)
            .copied()
            .or(self.pending_forward)
            .ok_or_else(|| Error::request("sample existing logits before advancing generation"))
    }
    pub(super) fn forwarded(&mut self, milliseconds: f64, prefilling: bool, width: usize) {
        if prefilling {
            self.cursor += 1;
            self.prefill_ms += milliseconds;
            self.batching.prefilled_tokens += 1;
        } else {
            self.pending_forward = None;
            self.decode_ms += milliseconds;
        }
        self.batching.forward_units += 1;
        self.batching.maximum_batch_width = self.batching.maximum_batch_width.max(width);
        self.batching.batched_forward_units += usize::from(width > 1);
    }
    /// Sample at most one token. Prefill work and waiting never advance the RNG.
    pub fn sample(&mut self, model: &Model) -> Result<Option<TokenEvent>> {
        self.check_model(model)?;
        if let Some(error) = self.error.take() {
            return Err(error);
        }
        self.cancel.check()?;
        if self.done || self.needs_forward() {
            return Ok(None);
        }
        if self.ids.len() == self.options.max_tokens {
            self.done = true;
            return Ok(None);
        }
        let start = Instant::now();
        let sampler = self.sampler.as_mut().expect("live sampler");
        let logits = &self.session.as_ref().expect("live session").scratch.logits;
        let token = if let Some(decoder) = &mut self.constraint {
            let mask_start = Instant::now();
            let mask = decoder.mask(&self.cancel)?;
            self.mask_ms += mask_start.elapsed().as_secs_f64() * 1000.;
            let sampling_start = Instant::now();
            let token = sampler.sample_allowed(logits, Some(mask))?;
            self.sampling_ms += sampling_start.elapsed().as_secs_f64() * 1000.;
            token
        } else {
            let sampling_start = Instant::now();
            let token = sampler.sample(logits)?;
            self.sampling_ms += sampling_start.elapsed().as_secs_f64() * 1000.;
            token
        };
        sampler.observe(token)?;
        let buffers = self.sampling_buffers.as_mut().expect("live buffers");
        if sampler.capacity_bytes() > buffers.bytes() {
            buffers.grow(sampler.capacity_bytes() - buffers.bytes())?;
        }
        self.ids.push(token);
        self.first_token_ms
            .get_or_insert_with(|| self.started.elapsed().as_secs_f64() * 1000.);
        if model.tokenizer.is_eog(token) || self.options.stop_tokens.contains(&token) {
            self.done = true;
            self.reason = FinishReason::Stop;
            self.decode_ms += start.elapsed().as_secs_f64() * 1000.;
            return Ok(None);
        }
        if let Some(decoder) = &mut self.constraint {
            decoder.advance(model.tokenizer.token_bytes(token)?)?;
        }
        let decoded = if model.tokenizer.is_special(token)
            && !(self.options.tool_markers && model.tokenizer.is_tool_marker(token))
        {
            String::new()
        } else {
            self.stream.push(model.tokenizer.token_bytes(token)?, false)
        };
        let (safe, matched) = self.stop.push(&decoded, false);
        if self.text.len().saturating_add(safe.len()) > 1 << 20 {
            return Err(Error::resource(
                "generated text exceeds one MiB output limit",
            ));
        }
        self.text.push_str(&safe);
        self.done = matched
            || self.constraint.as_ref().is_some_and(|c| c.complete())
            || self.ids.len() == self.options.max_tokens;
        if matched || self.constraint.as_ref().is_some_and(|c| c.complete()) {
            self.reason = FinishReason::Stop;
        }
        if !self.done {
            self.pending_forward = Some(token);
        }
        self.decode_ms += start.elapsed().as_secs_f64() * 1000.;
        Ok(Some(TokenEvent {
            token_id: Some(token),
            text: safe,
        }))
    }
    pub(super) fn cancelled(&mut self) {
        self.done = true;
        self.reason = FinishReason::Cancelled;
    }
    pub fn finish(
        mut self,
        model: &Model,
        keep_cache: bool,
        mut emit: impl FnMut(TokenEvent) -> Result<()>,
    ) -> Result<(GenerationResult, Option<PrefixCache>)> {
        self.check_model(model)?;
        if let Some(error) = self.error.take() {
            return Err(error);
        }
        if !self.done {
            return Err(Error::request("generation is not finished"));
        }
        if let Some(decoder) = &self.constraint {
            if self.reason == FinishReason::Cancelled {
                return Err(Error::new(
                    ErrorCode::Cancelled,
                    "constrained generation cancelled with incomplete output",
                ));
            }
            if !decoder.complete() {
                return Err(Error::new(
                    ErrorCode::ConstraintIncomplete,
                    "token limit reached before a complete constrained object",
                ));
            }
        }
        if !self.stop.matched {
            let tail = self.stream.push(&[], true);
            let (safe, _) = self.stop.push(&tail, true);
            if self.text.len().saturating_add(safe.len()) > 1 << 20 {
                return Err(Error::resource(
                    "generated text exceeds one MiB output limit",
                ));
            }
            self.text.push_str(&safe);
            if !safe.is_empty() {
                let start = Instant::now();
                emit(TokenEvent {
                    token_id: None,
                    text: safe,
                })?;
                self.output_delivery_ms += start.elapsed().as_secs_f64() * 1000.;
            }
        }
        if let Some(decoder) = &self.constraint {
            decoder.validate_output(&self.text)?;
        }
        self.sampler.take();
        self.sampling_buffers.take();
        let session = self.session.take().expect("live session");
        let cache = if keep_cache
            && self.reason != FinishReason::Cancelled
            && self.cancel.check().is_ok()
        {
            Some(session.into_cache(
                Arc::downgrade(&model.cache_owner),
                model.cache_key()?,
                self.io_bytes,
                self.execution,
            )?)
        } else {
            drop(session);
            None
        };
        let constraint_validation = self.constraint.as_ref().map(|c| {
            Box::new(ConstraintValidation {
                profile: crate::constraints::PROFILE.into(),
                valid: true,
                schema_fingerprint: c.schema_fingerprint().into(),
                compile_ms: self.compile_ms,
                mask_ms: self.mask_ms,
                target: if self.options.strict_tool {
                    "tool_arguments"
                } else {
                    "response"
                }
                .into(),
            })
        });
        self.constraint.take();
        self.constraint_buffers.take();
        self.constraint_reservation.take();
        Ok((
            GenerationResult {
                vision: None,
                batching: Some(Box::new(self.batching)),
                sampling_metadata: Some(Box::new(SamplingMetadata {
                    sampling_ms: self.sampling_ms,
                    profile: crate::sampling::PROFILE.into(),
                    settings: self.options.sampling,
                    penalties: self.options.penalties,
                    history_scope:
                        "exact selected prompt plus generated token IDs, including control tokens"
                            .into(),
                    tokenizer_fingerprint: model.tokenizer.cache_identity().into(),
                    backend: "native-cpu".into(),
                    cache_dtype: "f32".into(),
                    threads: self.execution.threads,
                    prefill_chunk_tokens: self.execution.prefill_chunk_tokens,
                    engine_version: env!("CARGO_PKG_VERSION").into(),
                })),
                constraint_validation,
                tool_calls: Vec::new(),
                text: self.text,
                completion_tokens: self.ids.len(),
                token_ids: self.ids,
                prompt_tokens: self.prompt_tokens.len(),
                reused_prompt_tokens: self.reused,
                finish_reason: self.reason,
                prefill_ms: self.prefill_ms,
                tokenization_ms: self.tokenization_ms,
                decode_ms: self.decode_ms,
                output_delivery_ms: self.output_delivery_ms,
                first_token_ms: self.first_token_ms,
                memory: model.budget.stats(),
            },
            cache,
        ))
    }
}

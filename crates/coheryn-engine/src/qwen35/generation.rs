//! Owned hybrid generation state for token-level scheduling and shared weight tiles.
use super::*;
use crate::{
    inference::{BatchStep, BatchingMetadata},
    memory::BufferAccounting,
};
use std::sync::{Arc, Weak};

pub struct GenerationState {
    session: Option<Session>,
    admission: Option<plan::Admission>,
    buffers: Option<BufferAccounting>,
    cache: Option<HybridCache>,
    owner: Weak<()>,
    cancel: Cancellation,
    error: Option<Error>,
    options: GenerationOptions,
    pub(super) threads: usize,
    pub(super) tokens: Vec<u32>,
    pub(super) condition: Option<super::image_generation::Conditioning>,
    cursor: usize,
    reused: usize,
    capture: usize,
    pending: Option<u32>,
    text: String,
    ids: Vec<u32>,
    decode_ids: Vec<u32>,
    decode_prefix: String,
    decode_index: usize,
    done: bool,
    reason: FinishReason,
    tokenization_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    delivery_ms: f64,
    first_token_ms: Option<f64>,
    started: Instant,
    batching: BatchingMetadata,
}
impl GenerationState {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn begin(
        model: &Model,
        prompt: &str,
        options: GenerationOptions,
        admission: plan::Admission,
        context: usize,
        cache_slot: Option<CacheSlot>,
        cancel: &Cancellation,
    ) -> Result<Self> {
        let cancel = cancel.with_timeout(options.timeout_seconds)?;
        cancel.check()?;
        let threads = admission.threads;
        admission.check(&model.budget, model.generation_bytes(context, threads)?)?;
        let started = Instant::now();
        let tokens = model.encode(prompt)?;
        let tokenization_ms = started.elapsed().as_secs_f64() * 1000.;
        if tokens
            .len()
            .checked_add(options.max_tokens)
            .is_none_or(|n| n > context)
        {
            return Err(Error::new(
                ErrorCode::ContextLimitExceeded,
                "prompt plus output reserve exceeds requested context",
            ));
        }
        let mut session = Session::new(&model.config, context, threads)?;
        let mut buffers =
            model
                .budget
                .track_buffers(Session::bytes(&model.config, context, threads)?);
        buffers.grow(session.capacity_bytes())?;
        let capture = if cache_slot.is_some() {
            model.cache_boundary(prompt, &tokens)?
        } else {
            0
        };
        let mut cache = if capture > 0 {
            cache_slot
                .map(|c| c.initialize(model, context, threads))
                .transpose()?
        } else {
            None
        };
        let reused = if let Some(c) = &cache {
            if !c.tokens.is_empty()
                && c.tokens.len() < tokens.len()
                && tokens.starts_with(&c.tokens)
            {
                c.snapshot.restore(&mut session)?;
                c.tokens.len()
            } else {
                0
            }
        } else {
            0
        };
        if capture < reused {
            cache = None;
        }
        Ok(Self {
            session: Some(session),
            admission: Some(admission),
            buffers: Some(buffers),
            cache,
            owner: Arc::downgrade(&model.owner),
            cancel,
            error: None,
            options,
            threads,
            tokens,
            condition: None,
            cursor: reused,
            reused,
            capture,
            pending: None,
            text: String::new(),
            ids: Vec::new(),
            decode_ids: Vec::new(),
            decode_prefix: String::new(),
            decode_index: 0,
            done: false,
            reason: FinishReason::Length,
            tokenization_ms,
            prefill_ms: 0.,
            decode_ms: 0.,
            delivery_ms: 0.,
            first_token_ms: None,
            started,
            batching: BatchingMetadata {
                profile: "coheryn-qwen35-batch-v1".into(),
                strategy: "single".into(),
                ..Default::default()
            },
        })
    }
    pub(super) fn set_images(
        &mut self,
        tokens: Vec<u32>,
        condition: super::image_generation::Conditioning,
    ) -> Result<()> {
        let context = self
            .session
            .as_ref()
            .expect("live image session")
            .context_capacity();
        if tokens.len() + self.options.max_tokens > context {
            return Err(Error::new(
                ErrorCode::ContextLimitExceeded,
                "expanded image context exceeded",
            ));
        }
        self.tokens = tokens;
        self.condition = Some(condition);
        self.cache = None;
        self.capture = 0;
        self.reused = 0;
        self.cursor = 0;
        Ok(())
    }
    pub fn prompt_token_ids(&self) -> &[u32] {
        &self.tokens
    }
    fn rotary(&self) -> Option<[usize; 3]> {
        self.condition.as_ref().map(|c| {
            c.positions
                .get(self.cursor)
                .copied()
                .unwrap_or([c.next_position + self.ids.len().saturating_sub(1); 3])
        })
    }
    pub fn compatible(&self, other: &Self) -> bool {
        Weak::ptr_eq(&self.owner, &other.owner) && self.threads == other.threads
    }
    pub fn is_done(&self) -> bool {
        self.done
    }
    pub fn needs_forward(&self) -> bool {
        self.cursor < self.tokens.len() || self.pending.is_some()
    }
    pub fn record_scheduler_wait(&mut self, ms: f64) {
        self.batching.scheduler_wait_ms += ms;
    }
    pub fn record_delivery(&mut self, ms: f64) {
        self.delivery_ms += ms;
    }
    pub fn cancellation(&self) -> Cancellation {
        self.cancel.clone()
    }
    fn token(&self) -> Result<u32> {
        self.cancel.check()?;
        if let Some(e) = &self.error {
            return Err(e.clone());
        }
        if self.done {
            return Err(Error::request("cannot advance finished hybrid generation"));
        }
        if self.cursor < self.tokens.len() {
            Ok(self.tokens[self.cursor])
        } else {
            self.pending
                .ok_or_else(|| Error::request("sample before advancing hybrid generation"))
        }
    }
    fn forwarded(&mut self, ms: f64, width: usize) -> Result<()> {
        if self.cursor < self.tokens.len() {
            self.cursor += 1;
            self.prefill_ms += ms;
            self.batching.prefilled_tokens += 1;
            if self.cursor == self.capture
                && let Some(cache) = &mut self.cache
            {
                cache
                    .snapshot
                    .capture(self.session.as_ref().expect("live session"))?;
                cache.tokens.clear();
                cache.tokens.extend_from_slice(&self.tokens[..self.capture]);
            }
        } else {
            self.pending = None;
            self.decode_ms += ms;
        }
        self.batching.forward_units += 1;
        self.batching.batched_forward_units += usize::from(width > 1);
        self.batching.maximum_batch_width = self.batching.maximum_batch_width.max(width);
        Ok(())
    }
    pub fn sample(&mut self, model: &Model) -> Result<Option<TokenEvent>> {
        self.cancel.check()?;
        if let Some(e) = &self.error {
            return Err(e.clone());
        }
        if !Weak::ptr_eq(&self.owner, &Arc::downgrade(&model.owner)) {
            return Err(Error::request(
                "hybrid state belongs to another model instance",
            ));
        }
        if self.done || self.cursor < self.tokens.len() {
            return Ok(None);
        }
        if self.pending.is_some() {
            return Err(Error::request("forward pending token before sampling"));
        }
        let start = Instant::now();
        let s = self
            .session
            .as_ref()
            .ok_or_else(|| Error::request("finished hybrid session"))?;
        let next = s
            .logits
            .iter()
            .enumerate()
            .max_by(|(ai, a), (bi, b)| a.total_cmp(b).then_with(|| bi.cmp(ai)))
            .map(|(i, _)| i as u32)
            .ok_or_else(|| Error::request("empty logits"))?;
        let stop =
            next == model.config.eos || Some(next) == model.tokenizer.token_to_id("<|im_end|>");
        self.ids.push(next);
        self.done = stop || self.ids.len() == self.options.max_tokens;
        if stop {
            self.reason = FinishReason::Stop;
        }
        let mut piece = if stop {
            String::new()
        } else {
            tokenizers::tokenizer::step_decode_stream(
                &*model.tokenizer,
                vec![next],
                true,
                &mut self.decode_ids,
                &mut self.decode_prefix,
                &mut self.decode_index,
            )
            .map_err(|e| Error::new(ErrorCode::UnsupportedTokenizer, e.to_string()))?
            .unwrap_or_default()
        };
        if self.done {
            let full = model
                .tokenizer
                .decode(&self.ids, true)
                .map_err(|e| Error::new(ErrorCode::UnsupportedTokenizer, e.to_string()))?;
            piece = full
                .strip_prefix(&self.text)
                .ok_or_else(|| {
                    unsupported("incremental token decoding disagrees with complete decoding")
                })?
                .to_owned();
        } else {
            self.pending = Some(next);
        }
        if self
            .text
            .len()
            .checked_add(piece.len())
            .is_none_or(|n| n > 1 << 20)
        {
            return Err(Error::resource("Qwen3.5 output exceeds one MiB"));
        }
        self.text.push_str(&piece);
        self.decode_ms += start.elapsed().as_secs_f64() * 1000.;
        if piece.is_empty() {
            return Ok(None);
        }
        self.first_token_ms
            .get_or_insert(self.started.elapsed().as_secs_f64() * 1000.);
        Ok(Some(TokenEvent {
            token_id: if stop { None } else { Some(next) },
            text: piece,
        }))
    }
}
impl Model {
    pub fn begin_generation(
        &self,
        prompt: &str,
        prepared: PreparedGeneration,
        cancel: &Cancellation,
    ) -> Result<GenerationState> {
        if !Arc::ptr_eq(&self.owner, &prepared.owner) || prepared.context > self.limits.context {
            return Err(Error::request(
                "generation admission belongs to another model instance",
            ));
        }
        GenerationState::begin(
            self,
            prompt,
            prepared.options,
            prepared.admission,
            prepared.context,
            prepared.cache,
            cancel,
        )
    }
    pub fn step_batch(
        &self,
        states: &mut [&mut GenerationState],
        mut cooperate: Option<&mut crate::multimodal::WorkYield<'_>>,
    ) -> Result<BatchStep> {
        if states.is_empty() || states.len() > 16 {
            return Err(Error::request("hybrid batch width must be 1..=16"));
        }
        if states.iter().any(|s| {
            !Weak::ptr_eq(&s.owner, &Arc::downgrade(&self.owner)) || !s.compatible(states[0])
        }) {
            return Err(Error::request("incompatible hybrid generation batch"));
        }
        let start = Instant::now();
        let mut lanes = Vec::with_capacity(states.len());
        for state in states.iter_mut() {
            let token = match state.token() {
                Ok(t) => t,
                Err(e) => {
                    state.error = Some(e);
                    0
                }
            };
            let rotary = state.rotary();
            if let Some(session) = &mut state.session {
                session.rotary = rotary;
            }
            lanes.push(execution::Lane {
                embedding: state
                    .condition
                    .as_ref()
                    .and_then(|c| c.embedding(state.cursor, self.config.hidden)),
                token,
                logits: state.cursor + 1 >= state.tokens.len(),
                session: state
                    .session
                    .as_mut()
                    .ok_or_else(|| Error::request("finished hybrid generation"))?,
                cancel: &state.cancel,
                error: &mut state.error,
            });
        }
        let result = self.graph.forward_batch(
            &self.set,
            &self.config,
            &mut lanes,
            &mut None,
            &mut cooperate,
        );
        drop(lanes);
        let elapsed = start.elapsed().as_secs_f64() * 1000.;
        let stats = match result {
            Ok(s) => s,
            Err(e) => {
                for state in states.iter_mut() {
                    if state.error.is_none() {
                        state.error = Some(e.clone());
                    }
                }
                return Err(e);
            }
        };
        for state in states.iter_mut() {
            state.batching.strategy = "continuous-weight-tiled".into();
            if state.error.is_none()
                && let Err(e) = state.forwarded(elapsed, stats.width)
            {
                state.error = Some(e);
            }
        }
        Ok(stats)
    }
    pub(super) fn step_single(
        &self,
        state: &mut GenerationState,
        cooperate: Option<&mut crate::multimodal::WorkYield<'_>>,
    ) -> Result<()> {
        let token = state.token()?;
        let start = Instant::now();
        let rotary = state.rotary();
        if let Some(session) = &mut state.session {
            session.rotary = rotary;
        }
        self.graph.forward_input(
            &self.set,
            &self.config,
            token,
            state
                .condition
                .as_ref()
                .and_then(|c| c.embedding(state.cursor, self.config.hidden)),
            state
                .session
                .as_mut()
                .ok_or_else(|| Error::request("finished hybrid state"))?,
            state.cursor + 1 >= state.tokens.len(),
            &state.cancel,
            None,
            cooperate,
        )?;
        state.forwarded(start.elapsed().as_secs_f64() * 1000., 1)
    }
    pub fn finish_generation(
        &self,
        mut state: GenerationState,
    ) -> Result<(GenerationResult, Option<HybridCache>)> {
        state.cancel.check()?;
        if let Some(e) = state.error.take() {
            return Err(e);
        }
        if !state.done || !Weak::ptr_eq(&state.owner, &Arc::downgrade(&self.owner)) {
            return Err(Error::request("hybrid generation is incomplete or foreign"));
        }
        drop(state.session.take());
        drop(state.buffers.take());
        drop(state.admission.take());
        let cache = state
            .cache
            .take()
            .filter(|c| !c.tokens.is_empty() && c.tokens.len() == state.capture);
        let vision = state.condition.take().map(|c| Box::new(c.metadata.clone()));
        let result = GenerationResult {
            vision,
            batching: if state.batching.strategy == "single" {
                None
            } else {
                Some(Box::new(state.batching))
            },
            sampling_metadata: None,
            constraint_validation: None,
            tool_calls: vec![],
            text: state.text,
            completion_tokens: state.ids.len(),
            token_ids: state.ids,
            prompt_tokens: state.tokens.len(),
            reused_prompt_tokens: state.reused,
            finish_reason: state.reason,
            prefill_ms: state.prefill_ms,
            tokenization_ms: state.tokenization_ms,
            decode_ms: state.decode_ms,
            output_delivery_ms: state.delivery_ms,
            first_token_ms: state.first_token_ms,
            memory: self.budget.stats(),
        };
        Ok((result, cache))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires the separately acquired pinned Qwen3.5 checkpoint"]
    fn hybrid_weight_batch_matches_frozen_outputs_and_isolates_cancelled_lane() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let model = Model::load(
            root.join(".cache/qwen3.5-0.8b-source"),
            ResourceLimits {
                context: 128,
                max_memory: 1 << 30,
            },
            &Default::default(),
        )
        .unwrap();
        let reference: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("fixtures/qwen35/reference-v1.json")).unwrap(),
        )
        .unwrap();
        let hardware = crate::planner::HardwareSnapshot {
            os: "test".into(),
            architecture: "test".into(),
            logical_cpus: 2,
            cpu_features: vec![],
            ram_total_bytes: None,
            ram_available_bytes: None,
            swap_total_bytes: None,
            swap_used_bytes: None,
            observed_unix_ms: 0,
            gpu_available_bytes: None,
            gpu_backend_qualified: false,
        };
        let execution = || crate::adapters::Execution {
            profile: crate::planner::Profile::Reference,
            threads: Some(1),
            hardware: &hardware,
        };
        let options = GenerationOptions {
            max_tokens: 4,
            ..Default::default()
        };
        let prompts: Vec<_> = reference["cases"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["prompt"].as_str().unwrap().to_owned())
            .collect();
        let mut states: Vec<_> = prompts
            .iter()
            .map(|p| {
                model
                    .begin_generation(
                        p,
                        model
                            .prepare_generation(options.clone(), 128, execution(), 2)
                            .unwrap(),
                        &Default::default(),
                    )
                    .unwrap()
            })
            .collect();
        let mut group_units = 0;
        while states.iter().any(|s| !s.is_done()) {
            let mut live: Vec<_> = states
                .iter_mut()
                .filter(|s| !s.is_done() && s.needs_forward())
                .collect();
            if !live.is_empty() {
                let step = model.step_batch(&mut live, None).unwrap();
                group_units += usize::from(step.width > 1);
            }
            for state in states.iter_mut().filter(|s| !s.is_done()) {
                state.sample(&model).unwrap();
            }
        }
        assert!(group_units > 0);
        for (state, case) in states
            .into_iter()
            .zip(reference["cases"].as_array().unwrap())
        {
            let (result, cache) = model.finish_generation(state).unwrap();
            assert!(cache.is_none());
            assert_eq!(
                serde_json::json!(result.token_ids),
                case["greedy_token_ids"]
            );
            assert_eq!(result.text, case["greedy_text"].as_str().unwrap());
            assert_eq!(result.batching.unwrap().maximum_batch_width, 2);
        }
        let cancel = Cancellation::default();
        let mut a = model
            .begin_generation(
                &prompts[0],
                model
                    .prepare_generation(options.clone(), 128, execution(), 2)
                    .unwrap(),
                &cancel,
            )
            .unwrap();
        let mut b = model
            .begin_generation(
                &prompts[1],
                model
                    .prepare_generation(options, 128, execution(), 2)
                    .unwrap(),
                &Default::default(),
            )
            .unwrap();
        cancel.cancel();
        let step = model.step_batch(&mut [&mut a, &mut b], None).unwrap();
        assert_eq!(step.width, 1);
        assert_eq!(a.sample(&model).unwrap_err().code, ErrorCode::Cancelled);
        assert!(b.sample(&model).is_ok());
        drop(a);
        drop(b);
        assert_eq!(model.memory().reserved, RESIDENT_METADATA_BYTES);
        assert_eq!(model.memory().session_buffer_capacity_bytes, 0);
    }
}

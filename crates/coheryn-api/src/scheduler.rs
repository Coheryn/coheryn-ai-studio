use crate::{
    config::ServerConfig,
    error::{ApiError, Result},
    wire::GenerationRequest,
};
use coheryn_engine::{
    Cancellation, Error, ErrorCode, FinishReason, GenerationResult, Model, TokenEvent,
    inference::SessionAdmission,
    memory::Budget,
    planner::{self, PlanInput, Profile},
};
use serde::Serialize;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc};

mod batching;
mod compute;
mod embedding;
mod qwen35;
mod vision;
pub use embedding::{EmbeddingDelivery, EmbeddingHandle};

async fn finalize_turn(
    mut turn: Option<Box<crate::sessions::TurnLease>>,
    mut event: GenerationEvent,
) -> GenerationEvent {
    if turn.is_none() {
        return event;
    }
    match tokio::task::spawn_blocking(move || {
        if let Some(turn) = &mut turn
            && let Err(error) = turn.finish(&event)
        {
            event = GenerationEvent::Failed(error);
            let _ = turn.finish(&event);
        }
        event
    })
    .await
    {
        Ok(event) => event,
        Err(error) => GenerationEvent::Failed(error.into()),
    }
}

type ExecutionPermits = (Option<OwnedSemaphorePermit>, OwnedSemaphorePermit);
const TOKEN_EVENT_SLOTS: usize = 16;
const DELIVERY_CHANNEL_SLOTS: usize = TOKEN_EVENT_SLOTS + 1;
fn constraint_error(mut error: ApiError, constrained: bool) -> ApiError {
    if constrained {
        let status = match error.error.code {
            ErrorCode::ConstraintDeadEnd => "dead_end",
            ErrorCode::ConstraintViolation => "invalid",
            ErrorCode::Cancelled => "cancelled",
            ErrorCode::DeadlineExceeded => "deadline",
            _ => "incomplete",
        };
        let mut details = error
            .details
            .take()
            .unwrap_or_else(|| serde_json::json!({}));
        details["constraint_validation"] = serde_json::json!({"profile":coheryn_engine::constraints::PROFILE,"valid":false,"status":status});
        error.details = Some(details);
    }
    error
}
use tokio_util::sync::CancellationToken;

pub enum GenerationEvent {
    Token(TokenEvent),
    ToolDelta(coheryn_engine::tools::ToolDelta),
    Finished(GenerationResult),
    Failed(ApiError),
}
pub struct GenerationHandle {
    _vision: Option<vision::VisionOwner>,
    _qwen35: Option<qwen35::TextOwner>,
    receiver: mpsc::Receiver<GenerationEvent>,
    cancel: Cancellation,
}
#[cfg(test)]
pub(crate) fn protocol_fixture(events: Vec<GenerationEvent>) -> GenerationHandle {
    let (sender, receiver) = mpsc::channel(events.len().max(1));
    for event in events {
        assert!(sender.try_send(event).is_ok());
    }
    GenerationHandle {
        receiver,
        _vision: None,
        _qwen35: None,
        cancel: Cancellation::default(),
    }
}
impl GenerationHandle {
    pub async fn next(&mut self) -> Option<GenerationEvent> {
        self.receiver.recv().await
    }
}
impl Drop for GenerationHandle {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}
#[derive(Default)]
pub struct Counters {
    waiting_generation_units: AtomicU64,
    generation_units: AtomicU64,
    batched_generation_units: AtomicU64,
    forwarded_tokens: AtomicU64,
    maximum_batch_width: AtomicU64,
    matrix_tile_reads: AtomicU64,
    matrix_read_bytes: AtomicU64,
    embedding_microbatches: AtomicU64,
    vision_backpressure_waits: AtomicU64,
    vision_units: AtomicU64,
    vision_yields: AtomicU64,
    vision_waiting_units: AtomicU64,
    vision_max_unit_micros: AtomicU64,
    active: AtomicU64,
    queued: AtomicU64,
    completed: AtomicU64,
    cancelled: AtomicU64,
    failed: AtomicU64,
    rejected: AtomicU64,
}
#[derive(Serialize)]
pub struct SchedulerStats {
    pub waiting_generation_units: u64,
    pub generation_units: u64,
    pub batched_generation_units: u64,
    pub forwarded_tokens: u64,
    pub maximum_batch_width: u64,
    pub matrix_tile_reads: u64,
    pub matrix_read_bytes: u64,
    pub embedding_microbatches: u64,
    pub vision_backpressure_waits: u64,
    pub vision_units: u64,
    pub vision_yields: u64,
    pub vision_waiting_units: u64,
    pub vision_max_unit_micros: u64,
    pub active: u64,
    pub queued: u64,
    pub completed: u64,
    pub cancelled: u64,
    pub failed: u64,
    pub rejected: u64,
}
impl Counters {
    pub fn snapshot(&self) -> SchedulerStats {
        SchedulerStats {
            waiting_generation_units: self.waiting_generation_units.load(Ordering::Relaxed),
            generation_units: self.generation_units.load(Ordering::Relaxed),
            batched_generation_units: self.batched_generation_units.load(Ordering::Relaxed),
            forwarded_tokens: self.forwarded_tokens.load(Ordering::Relaxed),
            maximum_batch_width: self.maximum_batch_width.load(Ordering::Relaxed),
            matrix_tile_reads: self.matrix_tile_reads.load(Ordering::Relaxed),
            matrix_read_bytes: self.matrix_read_bytes.load(Ordering::Relaxed),
            embedding_microbatches: self.embedding_microbatches.load(Ordering::Relaxed),
            vision_backpressure_waits: self.vision_backpressure_waits.load(Ordering::Relaxed),
            vision_units: self.vision_units.load(Ordering::Relaxed),
            vision_yields: self.vision_yields.load(Ordering::Relaxed),
            vision_waiting_units: self.vision_waiting_units.load(Ordering::Relaxed),
            vision_max_unit_micros: self.vision_max_unit_micros.load(Ordering::Relaxed),
            active: self.active.load(Ordering::Relaxed),
            queued: self.queued.load(Ordering::Relaxed),
            completed: self.completed.load(Ordering::Relaxed),
            cancelled: self.cancelled.load(Ordering::Relaxed),
            failed: self.failed.load(Ordering::Relaxed),
            rejected: self.rejected.load(Ordering::Relaxed),
        }
    }
}
#[derive(Clone)]
pub struct Scheduler {
    compute: Arc<Semaphore>,
    batcher: batching::Batcher,
    hybrid_batcher: batching::Batcher<coheryn_engine::qwen35::Model>,
    sessions: Option<crate::sessions::Sessions>,
    slots: Arc<Semaphore>,
    active: Arc<Semaphore>,
    balanced: Arc<Semaphore>,
    waiting: Arc<Semaphore>,
    embedding_slots: Arc<Semaphore>,
    embedding_active: Arc<Semaphore>,
    vision_active: Arc<Semaphore>,
    pub counters: Arc<Counters>,
    shutdown: CancellationToken,
    config: ServerConfig,
    budget: Budget,
}
impl Scheduler {
    pub fn new(mut config: ServerConfig, budget: Budget, shutdown: CancellationToken) -> Self {
        config.max_concurrency = config.max_concurrency.min(
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
        );
        let compute = Arc::new(Semaphore::new(1));
        let counters = Arc::new(Counters::default());
        let batcher = batching::Batcher::new(compute.clone(), counters.clone(), shutdown.clone());
        Self {
            hybrid_batcher: batching::Batcher::new(
                compute.clone(),
                counters.clone(),
                shutdown.clone(),
            ),
            compute,
            batcher,
            sessions: None,
            slots: Arc::new(Semaphore::new(
                config.queue_capacity + config.max_concurrency,
            )),
            active: Arc::new(Semaphore::new(config.max_concurrency)),
            vision_active: Arc::new(Semaphore::new(config.max_concurrency)),
            balanced: Arc::new(Semaphore::new(config.max_concurrency.min(2))),
            waiting: Arc::new(Semaphore::new(config.queue_capacity)),
            embedding_slots: Arc::new(Semaphore::new(config.embeddings.requests)),
            embedding_active: Arc::new(Semaphore::new(
                config.embeddings.concurrency.min(config.max_concurrency),
            )),
            counters,
            config,
            budget,
            shutdown,
        }
    }
    pub fn with_sessions(mut self, sessions: crate::sessions::Sessions) -> Self {
        self.sessions = Some(sessions);
        self
    }
    pub fn max_concurrency(&self) -> usize {
        self.config.max_concurrency
    }
    fn try_execution(&self, profile: Profile) -> Option<ExecutionPermits> {
        let profile_permit = if profile == Profile::Balanced {
            Some(self.balanced.clone().try_acquire_owned().ok()?)
        } else {
            None
        };
        let count = if profile == Profile::LowMemory {
            self.config.max_concurrency as u32
        } else {
            1
        };
        let active = self.active.clone().try_acquire_many_owned(count).ok()?;
        Some((profile_permit, active))
    }
    pub fn submit(
        &self,
        model: Arc<Model>,
        request: GenerationRequest,
    ) -> Result<GenerationHandle> {
        self.submit_inner(model, request, None)
    }
    pub fn submit_managed(
        &self,
        model: Arc<Model>,
        request: GenerationRequest,
        turn: Box<crate::sessions::TurnLease>,
    ) -> Result<GenerationHandle> {
        self.submit_inner(model, request, Some(turn))
    }
    fn submit_inner(
        &self,
        model: Arc<Model>,
        request: GenerationRequest,
        mut turn: Option<Box<crate::sessions::TurnLease>>,
    ) -> Result<GenerationHandle> {
        let managed_overhead = turn.as_ref().map_or(0, |t| t.reserved_bytes());
        let managed = turn.is_some();
        let mut cache = turn
            .as_mut()
            .and_then(|t| t.cache.take())
            .and_then(|c| match c {
                crate::sessions::TextCache::Gguf(c) => Some(c),
                _ => None,
            });
        let mut admit = || -> Result<_> {
            request.validate()?;
            let constraint_bytes = if request.options().response_format.enabled() {
                coheryn_engine::constraints::RESERVATION_BYTES
            } else {
                0
            };
            if request
                .stop_tokens
                .iter()
                .any(|id| *id as usize >= model.tokenizer.vocabulary())
                || request.top_k > model.tokenizer.vocabulary()
            {
                return Err(ApiError::bad(
                    "stop token or top_k is outside the loaded vocabulary",
                ));
            }
            if (request.tool_config().enabled()
                || request.messages.as_ref().is_some_and(|ms| {
                    ms.iter()
                        .any(|m| m.role == "tool" || !m.tool_calls.is_empty())
                }))
                && !model.tokenizer.supports_tools()
            {
                return Err(Error::new(
                    ErrorCode::UnsupportedChatTemplate,
                    "loaded template does not support tools",
                )
                .into());
            }
            let context = request.context.unwrap_or(model.context_limit());
            if context == 0 || context > model.context_limit() || request.max_tokens > context {
                return Err(ApiError::bad(
                    "context/max_tokens exceed the loaded model capacity",
                ));
            }
            if self.shutdown.is_cancelled() {
                return Err(Error::new(ErrorCode::Unavailable, "server is shutting down").into());
            }
            let profile = Profile::parse(&request.profile)?;
            let estimate = |credit: u64| {
                planner::estimate(PlanInput {
                    constraints: constraint_bytes > 0,
                    descriptor: &model.descriptor,
                    config: &model.config,
                    model: &request.model,
                    profile,
                    context,
                    concurrency: 1,
                    max_concurrency: self.config.max_concurrency,
                    loaded: true,
                    budget: {
                        let mut stats = self.budget.stats();
                        stats.reserved = stats.reserved.saturating_sub(credit);
                        stats
                    },
                    hardware: crate::hardware::probe(),
                    images: None,
                    threads: request.threads,
                })
            };
            let mut credit = cache.as_ref().map_or(0, |c| c.reserved_bytes());
            let mut plan = estimate(credit)?;
            let execution = coheryn_engine::inference::ExecutionSettings {
                threads: plan.threads_per_session,
                prefill_chunk_tokens: plan.prefill_chunk_tokens,
            };
            if let Some(c) = &cache
                && !model.cache_compatible(c, context, plan.io_buffer_bytes, execution)?
            {
                cache = None;
                credit = 0;
                plan = estimate(0)?;
            }
            if plan.require_admission().is_err()
                && let Some(sessions) = &self.sessions
            {
                sessions.evict_idle()?;
                plan = estimate(credit)?;
            }
            plan.require_admission().map_err(|error| {
                let mut error: ApiError = error.into();
                error.details = Some(serde_json::json!(plan));
                self.counters.rejected.fetch_add(1, Ordering::Relaxed);
                error
            })?;
            if let Some(limit) = request.max_memory {
                let required =
                    coheryn_engine::memory::model_bytes(&model.descriptor, &model.config)?
                        .checked_add(plan.additional_reserved_bytes)
                        .and_then(|bytes| bytes.checked_add(managed_overhead))
                        .ok_or_else(|| Error::resource("request estimate overflow"))?;
                if required > limit {
                    self.counters.rejected.fetch_add(1, Ordering::Relaxed);
                    return Err(Error::resource(format!(
                        "request needs an estimated {required} bytes, exceeding max_memory={limit}"
                    ))
                    .into());
                }
            }
            let slot = self.slots.clone().try_acquire_owned().map_err(|_| {
                self.counters.rejected.fetch_add(1, Ordering::Relaxed);
                ApiError::from(Error::new(ErrorCode::QueueFull, "generation queue is full"))
            })?;
            // One transaction holds payload + full session capacity + headroom, including while queued.
            // The session slice transfers directly to the engine; no check/drop/re-reserve window.
            let mut held = self
                .budget
                .reserve(
                    plan.additional_reserved_bytes
                        .checked_sub(credit)
                        .ok_or_else(|| Error::resource("invalid cache admission credit"))?,
                )
                .map_err(|error| {
                    self.counters.rejected.fetch_add(1, Ordering::Relaxed);
                    ApiError::from(error)
                })?;
            let session = SessionAdmission {
                constraints: if constraint_bytes > 0 {
                    Some(held.split(constraint_bytes)?)
                } else {
                    None
                },
                context,
                reservation: held.split(
                    plan.session_reserved_bytes
                        .checked_sub(credit)
                        .ok_or_else(|| Error::resource("cache exceeds session reservation"))?,
                )?,
                io_bytes: plan.io_buffer_bytes,
                execution: coheryn_engine::inference::ExecutionSettings {
                    threads: plan.threads_per_session,
                    prefill_chunk_tokens: plan.prefill_chunk_tokens,
                },
            };
            let initial = self.try_execution(profile);
            let waiting = if initial.is_none() {
                Some(self.waiting.clone().try_acquire_owned().map_err(|_| {
                    self.counters.rejected.fetch_add(1, Ordering::Relaxed);
                    ApiError::from(Error::new(
                        ErrorCode::QueueFull,
                        "generation waiting queue is full",
                    ))
                })?)
            } else {
                None
            };
            let queued = waiting.is_some();
            Ok((slot, held, session, initial, waiting, queued, profile))
        };
        let (slot, held, session, initial, waiting, queued, profile) = match admit() {
            Ok(admitted) => admitted,
            Err(error) => {
                if let Some(turn) = &mut turn {
                    turn.finish(&GenerationEvent::Failed(error.clone()))?;
                }
                return Err(error);
            }
        };
        // Sixteen token events plus a reserved terminal slot.
        let constrained = request.options().response_format.enabled();
        let (sender, receiver) = mpsc::channel(DELIVERY_CHANNEL_SLOTS);
        let timeout_seconds = request
            .timeout_seconds
            .unwrap_or(self.config.generation_timeout_seconds)
            .min(self.config.generation_timeout_seconds);
        let cancel = Cancellation::default().with_timeout(Some(timeout_seconds))?;
        let engine_cancel = cancel.clone();
        let scheduler = self.clone();
        if queued {
            self.counters.queued.fetch_add(1, Ordering::Relaxed);
        }
        tokio::spawn(async move {
            let _slot = slot;
            let _held = held;
            let acquire = async {
                let profile_permit = if profile == Profile::Balanced {
                    Some(scheduler.balanced.clone().acquire_owned().await.ok()?)
                } else {
                    None
                };
                let count = if profile == Profile::LowMemory {
                    scheduler.config.max_concurrency as u32
                } else {
                    1
                };
                let active = scheduler
                    .active
                    .clone()
                    .acquire_many_owned(count)
                    .await
                    .ok()?;
                Some((profile_permit, active))
            };
            let permit = if initial.is_some() {
                initial
            } else {
                tokio::select! {
                    _=sender.closed()=>None,
                    _=scheduler.shutdown.cancelled()=>None,
                    result=tokio::time::timeout(scheduler.config.queue_timeout().min(std::time::Duration::from_secs(timeout_seconds)),acquire)=>result.ok().flatten(),
                }
            };
            if queued {
                scheduler.counters.queued.fetch_sub(1, Ordering::Relaxed);
            }
            drop(waiting);
            let Some(_permits) = permit else {
                scheduler.counters.cancelled.fetch_add(1, Ordering::Relaxed);
                let event = GenerationEvent::Failed(constraint_error(
                    engine_cancel
                        .check()
                        .err()
                        .unwrap_or_else(|| {
                            Error::new(
                                ErrorCode::Cancelled,
                                "queued request cancelled or timed out",
                            )
                        })
                        .into(),
                    constrained,
                ));
                let event = finalize_turn(turn, event).await;
                let _ = sender.send(event).await;
                return;
            };
            scheduler.counters.active.fetch_add(1, Ordering::Relaxed);
            let compute_sender = sender.clone();
            let compute_cancel = engine_cancel.clone();
            let output_limit = turn.as_ref().map(|t| t.output_limit());
            let consumer_timeout =
                std::time::Duration::from_secs(scheduler.config.consumer_timeout_seconds);
            let batcher = scheduler.batcher.clone();
            let mut job = tokio::task::spawn_blocking(move || -> Result<(GenerationResult, Option<coheryn_engine::inference::PrefixCache>)> {
                compute_cancel.check()?;
                let prompt = match &request.prompt {
                    Some(prompt) => prompt.clone(),
                    None => model.tokenizer.render_tools(request.messages.as_ref().expect("validated messages"),&request.tool_config())?,
                };
                let options = request.options();
                let config=request.tool_config();
                let mut parser=if config.enabled(){Some(coheryn_engine::tools::ToolParser::new(config,format!("call_{}",uuid::Uuid::new_v4().simple()))?)}else{None};
                let deliver = |mut pending:GenerationEvent| -> coheryn_engine::Result<()> {
                        let start = std::time::Instant::now();

                        loop {
                            compute_cancel.check()?;
                            if compute_sender.capacity() <= 1 {
                                if compute_sender.is_closed() {
                                    return Err(Error::new(
                                        ErrorCode::Cancelled,
                                        "consumer disconnected",
                                    ));
                                }
                                if start.elapsed() > consumer_timeout {
                                    compute_cancel.cancel();
                                    return Err(Error::new(
                                        ErrorCode::Cancelled,
                                        "consumer backpressure timeout",
                                    ));
                                }
                                std::thread::park_timeout(std::time::Duration::from_millis(5));
                                continue;
                            }
                            match compute_sender.try_send(pending) {
                                Ok(()) => return Ok(()),
                                Err(mpsc::error::TrySendError::Closed(_)) => {
                                    return Err(Error::new(
                                        ErrorCode::Cancelled,
                                        "consumer disconnected",
                                    ));
                                }
                                Err(mpsc::error::TrySendError::Full(event)) => pending = event,
                            }
                            if start.elapsed() > consumer_timeout {
                                compute_cancel.cancel();
                                return Err(Error::new(
                                    ErrorCode::Cancelled,
                                    "consumer backpressure timeout",
                                ));
                            }
                            std::thread::park_timeout(std::time::Duration::from_millis(5));
                        }
                };
                let mut delivered_bytes=0usize;
                let mut sink=|event:TokenEvent| -> coheryn_engine::Result<()> {
                    delivered_bytes=delivered_bytes.checked_add(event.text.len()).ok_or_else(||Error::resource("output size overflow"))?;
                    if output_limit.is_some_and(|n|delivered_bytes>n){return Err(Error::resource("assistant output exceeds session history limit"));}
                    if let Some(parser)=&mut parser {
                        for event in parser.push(&event.text)? {
                            match event {coheryn_engine::tools::ParsedEvent::Text(text)=>deliver(GenerationEvent::Token(TokenEvent{token_id:None,text}))?,coheryn_engine::tools::ParsedEvent::Delta(delta)=>deliver(GenerationEvent::ToolDelta(delta))?}
                        } Ok(())
                    } else {deliver(GenerationEvent::Token(event))}
                };
                let mut state=model.begin_generation(&prompt,&options,session,&compute_cancel,cache.take())?;
                while !state.is_finished() {
                    compute_cancel.check()?;
                    if state.needs_forward(){state=batcher.advance(model.clone(),state)?;}
                    if let Some(event)=state.sample(&model)? {
                        let start=std::time::Instant::now();sink(event)?;
                        state.record_delivery(start.elapsed().as_secs_f64()*1000.);
                    }
                }
                let (mut result,new_cache)=state.finish(&model,managed,&mut sink)?;
                cache=new_cache;
                drop(sink);
                if let Some(parser)=parser {
                    let(text,calls,tail)=parser.finish(result.finish_reason)?;
                    if !tail.is_empty(){deliver(GenerationEvent::Token(TokenEvent{token_id:None,text:tail}))?;}
                    result.text=text;
                    if !calls.is_empty(){result.finish_reason=FinishReason::ToolCalls;}
                    result.tool_calls=calls;
                }
                Ok((result,cache))
            });
            let timeout = tokio::time::sleep(std::time::Duration::from_secs(timeout_seconds));
            tokio::pin!(timeout);
            let result = tokio::select! {
                result=&mut job=>result,
                _=scheduler.shutdown.cancelled()=>{engine_cancel.cancel();job.await},
                _=&mut timeout=>job.await,
                _=sender.closed()=>{engine_cancel.cancel();job.await},
            };
            let result = result.map(|r| {
                r.map(|(result, cache)| {
                    if let Some(turn) = &mut turn {
                        turn.cache = cache.map(crate::sessions::TextCache::Gguf);
                    }
                    result
                })
            });
            scheduler.counters.active.fetch_sub(1, Ordering::Relaxed);
            let mut event = match result {
                Ok(Ok(result)) if result.finish_reason != FinishReason::Cancelled => {
                    scheduler.counters.completed.fetch_add(1, Ordering::Relaxed);
                    GenerationEvent::Finished(result)
                }
                Ok(Ok(_)) => {
                    scheduler.counters.cancelled.fetch_add(1, Ordering::Relaxed);
                    GenerationEvent::Failed(
                        Error::new(ErrorCode::Cancelled, "generation cancelled").into(),
                    )
                }
                Ok(Err(error)) => {
                    if matches!(
                        error.error.code,
                        ErrorCode::Cancelled | ErrorCode::DeadlineExceeded
                    ) {
                        scheduler.counters.cancelled.fetch_add(1, Ordering::Relaxed);
                    } else {
                        scheduler.counters.failed.fetch_add(1, Ordering::Relaxed);
                    }
                    GenerationEvent::Failed(constraint_error(error, constrained))
                }
                Err(error) => {
                    scheduler.counters.failed.fetch_add(1, Ordering::Relaxed);
                    GenerationEvent::Failed(error.into())
                }
            };
            let prior_success = matches!(event, GenerationEvent::Finished(_));
            event = finalize_turn(turn, event).await;
            if prior_success && matches!(event, GenerationEvent::Failed(_)) {
                scheduler.counters.completed.fetch_sub(1, Ordering::Relaxed);
                scheduler.counters.failed.fetch_add(1, Ordering::Relaxed);
            }
            // A terminal message cannot keep admission/model residency pinned forever.
            let _ = tokio::time::timeout(consumer_timeout, sender.send(event)).await;
        });
        Ok(GenerationHandle {
            receiver,
            cancel,
            _vision: None,
            _qwen35: None,
        })
    }
    pub async fn drained(&self) {
        while self.slots.available_permits()
            != self.config.queue_capacity + self.config.max_concurrency
        {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }
}

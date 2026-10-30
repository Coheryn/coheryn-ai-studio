use super::*;
use coheryn_engine::{adapters::Execution, memory::Reservation, qwen35::Model as HybridModel};
use std::time::{Duration, Instant};

/// Pin model files and bounded transport until the response is drained or dropped.
pub(super) struct TextOwner {
    _model: Arc<HybridModel>,
    _transport: Reservation,
}

impl Scheduler {
    fn try_hybrid_execution(&self, profile: Profile, image: bool) -> Option<ExecutionPermits> {
        if !image {
            return self.try_execution(profile);
        }
        let count = if profile == Profile::LowMemory {
            self.config.max_concurrency as u32
        } else {
            1
        };
        self.vision_active
            .clone()
            .try_acquire_many_owned(count)
            .ok()
            .map(|permit| (None, permit))
    }
    async fn acquire_hybrid_execution(
        &self,
        profile: Profile,
        image: bool,
    ) -> Option<ExecutionPermits> {
        let balanced = if !image && profile == Profile::Balanced {
            Some(self.balanced.clone().acquire_owned().await.ok()?)
        } else {
            None
        };
        let count = if profile == Profile::LowMemory {
            self.config.max_concurrency as u32
        } else {
            1
        };
        let phase = if image {
            &self.vision_active
        } else {
            &self.active
        };
        Some((
            balanced,
            phase.clone().acquire_many_owned(count).await.ok()?,
        ))
    }
    pub fn submit_qwen35(
        &self,
        model: Arc<HybridModel>,
        request: GenerationRequest,
        payload: Option<Reservation>,
    ) -> Result<GenerationHandle> {
        self.submit_qwen35_inner(model, request, payload, &mut None, None)
    }
    pub fn submit_managed_qwen35(
        &self,
        model: Arc<HybridModel>,
        request: GenerationRequest,
        payload: Option<Reservation>,
        turn: Box<crate::sessions::TurnLease>,
    ) -> Result<GenerationHandle> {
        let mut turn = Some(turn);
        let result = self.submit_qwen35_inner(model, request, payload, &mut turn, None);
        if let (Err(error), Some(turn)) = (&result, &mut turn) {
            turn.finish(&GenerationEvent::Failed(error.clone()))?;
        }
        result
    }
    pub fn submit_managed_qwen35_images(
        &self,
        model: Arc<HybridModel>,
        request: GenerationRequest,
        payload: Reservation,
        prepared: coheryn_engine::qwen35::PreparedImages,
        turn: Box<crate::sessions::TurnLease>,
    ) -> Result<GenerationHandle> {
        let mut turn = Some(turn);
        let result =
            self.submit_qwen35_inner(model, request, Some(payload), &mut turn, Some(prepared));
        if let (Err(error), Some(turn)) = (&result, &mut turn) {
            turn.finish(&GenerationEvent::Failed(error.clone()))?;
        }
        result
    }
    pub fn submit_qwen35_images(
        &self,
        model: Arc<HybridModel>,
        request: GenerationRequest,
        messages: Vec<coheryn_engine::multimodal::qwen2vl::ChatMessage>,
        images: Vec<Vec<u8>>,
        payload: Reservation,
    ) -> Result<GenerationHandle> {
        request.validate()?;
        crate::vision::validate_controls(&request)?;
        let hardware = crate::hardware::probe();
        let prepared = model.prepare_images(
            &messages,
            images,
            self.config.vision.images,
            request.options(),
            request.context.unwrap_or(model.context_limit()),
            Execution {
                profile: Profile::parse(&request.profile)?,
                threads: request.threads,
                hardware: &hardware,
            },
            self.config.max_concurrency,
            &Default::default(),
        )?;
        self.submit_qwen35_inner(model, request, Some(payload), &mut None, Some(prepared))
    }
    fn submit_qwen35_inner(
        &self,
        model: Arc<HybridModel>,
        request: GenerationRequest,
        payload: Option<Reservation>,
        turn: &mut Option<Box<crate::sessions::TurnLease>>,
        image: Option<coheryn_engine::qwen35::PreparedImages>,
    ) -> Result<GenerationHandle> {
        let managed_overhead = turn.as_ref().map_or(0, |t| t.reserved_bytes());
        let output_limit = turn.as_ref().map(|t| t.output_limit());
        request.validate()?;
        HybridModel::check_options(&request.options())?;
        if request.tool_config().enabled()
            || request.messages.as_ref().is_some_and(|m| {
                m.iter()
                    .any(|m| m.role == "tool" || !m.tool_calls.is_empty())
            })
        {
            return Err(Error::new(
                ErrorCode::UnsupportedFeature,
                "Qwen3.5 greedy text profile does not support tools or tool histories",
            )
            .into());
        }
        if self.shutdown.is_cancelled() {
            return Err(Error::new(ErrorCode::Unavailable, "server is shutting down").into());
        }
        let context = request.context.unwrap_or(model.context_limit());
        let profile = Profile::parse(&request.profile)?;
        let seconds = request
            .timeout_seconds
            .unwrap_or(self.config.generation_timeout_seconds)
            .min(self.config.generation_timeout_seconds);
        let cancel = Cancellation::default().with_timeout(Some(seconds))?;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(seconds);
        let slot = self
            .slots
            .clone()
            .try_acquire_owned()
            .map_err(|_| Error::new(ErrorCode::QueueFull, "shared inference queue is full"))?;
        let transport = if let Some(mut payload) = payload {
            if !payload.covers(&self.budget, payload.bytes()) {
                return Err(Error::resource("text ingress belongs to another budget").into());
            }
            payload.extend((8u64 << 20).saturating_sub(payload.bytes()))?;
            let transport = payload.split(8 << 20)?;
            // Parsed source buffers have retired. Transfer bounded response capacity
            // without an uncharged request or duplicate ingress reservation.
            drop(payload);
            transport
        } else {
            self.budget.reserve(8 << 20)?
        };
        let hardware = crate::hardware::probe();
        let cache = turn
            .as_mut()
            .and_then(|t| t.cache.take())
            .and_then(|c| match c {
                crate::sessions::TextCache::Qwen35(c) => Some(c),
                _ => None,
            });
        let cache_limit = if turn.is_some() && image.is_none() {
            self.sessions.as_ref().map_or(0, |s| s.cache_limit())
        } else {
            0
        };
        let prepared = if image.is_none() {
            Some(model.prepare_cached_generation(
                request.options(),
                context,
                Execution {
                    profile,
                    threads: request.threads,
                    hardware: &hardware,
                },
                self.config.max_concurrency,
                cache,
                cache_limit,
            )?)
        } else {
            None
        };
        if request.max_memory.is_some_and(|limit| {
            coheryn_engine::qwen35::RESIDENT_METADATA_BYTES
                .saturating_add(
                    prepared.as_ref().map_or(0, |p| p.request_reserved_bytes())
                        + image.as_ref().map_or(0, |p| p.request_reserved_bytes()),
                )
                .saturating_add(transport.bytes())
                .saturating_add(managed_overhead)
                > limit
        }) {
            return Err(Error::resource("max_memory cannot cover Qwen3.5 metadata, execution, transport and profile headroom").into());
        }
        let is_image = image.is_some();
        let initial = self.try_hybrid_execution(profile, is_image);
        let waiting =
            if initial.is_none() {
                Some(self.waiting.clone().try_acquire_owned().map_err(|_| {
                    Error::new(ErrorCode::QueueFull, "shared inference queue is full")
                })?)
            } else {
                None
            };
        let queued = waiting.is_some();
        if queued {
            self.counters.queued.fetch_add(1, Ordering::Relaxed);
        }
        let (sender, receiver) = mpsc::channel(DELIVERY_CHANNEL_SLOTS);
        let owner = TextOwner {
            _model: model.clone(),
            _transport: transport,
        };
        let scheduler = self.clone();
        let engine_cancel = cancel.clone();
        let mut turn = turn.take();
        tokio::spawn(async move {
            let _slot = slot;
            // Keep admission and source ownership through cancellation acknowledgement.
            let acquire = async {
                if let Some(p) = initial {
                    return Some(p);
                }
                scheduler.acquire_hybrid_execution(profile, is_image).await
            };
            let permits = tokio::select! {
                p = tokio::time::timeout(scheduler.config.queue_timeout(),acquire) => p.ok().flatten(),
                _ = sender.closed() => None,
                _ = scheduler.shutdown.cancelled() => None,
                _ = tokio::time::sleep_until(deadline) => None,
            };
            if queued {
                scheduler.counters.queued.fetch_sub(1, Ordering::Relaxed);
            }
            drop(waiting);
            let result: Result<(GenerationResult,Option<coheryn_engine::qwen35::HybridCache>)> = async {
                let _permits = permits.ok_or_else(|| {
                    Error::new(
                        ErrorCode::Cancelled,
                        "queued Qwen3.5 request cancelled or timed out",
                    )
                })?;
                engine_cancel.check()?;
                let c = engine_cancel.clone();
                let tx = sender.clone();
                let job_scheduler = scheduler.clone();
                scheduler.counters.active.fetch_add(1, Ordering::Relaxed);
                let mut job = tokio::task::spawn_blocking(move || -> Result<(GenerationResult,Option<coheryn_engine::qwen35::HybridCache>)> {
                    let is_image = image.is_some();
                    let prompt = if image.is_some(){String::new()} else if let Some(prompt) = &request.prompt {
                        prompt.clone()
                    } else {
                        model.render(request.messages.as_deref().unwrap_or_default())?
                    };
                    let mut delivered = 0usize;
                    let mut state=if let Some(image)=image {
                        let runtime=tokio::runtime::Handle::current();
                        let permit=runtime.block_on(async {tokio::select! {
                            p=job_scheduler.compute.clone().acquire_owned()=>p.ok(),
                            _=tx.closed()=>None,
                            _=job_scheduler.shutdown.cancelled()=>None,
                            _=tokio::time::sleep_until(deadline)=>None,
                        }}).ok_or_else(||Error::new(ErrorCode::Cancelled,"image request cancelled waiting for compute"))?;
                        let mut lease=super::compute::ComputeLease::new(permit,&job_scheduler,c.clone(),tx.clone(),deadline,true);
                        let state=model.begin_images(image,&c,Some(&mut ||lease.yield_unit()),None)?;
                        drop(lease);state
                    }else{model.begin_generation(&prompt,prepared.expect("text admission"),&c)?};
                    while !state.is_done() {
                        if state.needs_forward(){state=job_scheduler.hybrid_batcher.advance(model.clone(),state)?;}
                        if let Some(token)=state.sample(&model)? {
                            let delivery_start=Instant::now();
                            delivered = delivered
                                .checked_add(token.text.len())
                                .ok_or_else(|| Error::resource("text output byte overflow"))?;
                            if output_limit.is_some_and(|limit| delivered > limit) {
                                return Err(Error::resource(
                                    "text response exceeds session history quota",
                                ).into());
                            }
                            let mut pending = GenerationEvent::Token(token);
                            let start = Instant::now();
                            let mut reported = false;
                            loop {
                                c.check()?;
                                // Leave a bounded terminal slot; no compute permit during delivery.
                                if tx.capacity() > 1 {
                                    match tx.try_send(pending) {
                                        Ok(()) => {
                                            break;
                                        }
                                        Err(mpsc::error::TrySendError::Closed(_)) => {
                                            return Err(Error::new(
                                                ErrorCode::Cancelled,
                                                "text consumer disconnected",
                                            ).into());
                                        }
                                        Err(mpsc::error::TrySendError::Full(event)) => {
                                            pending = event
                                        }
                                    }
                                } else if tx.is_closed() {
                                    return Err(Error::new(
                                        ErrorCode::Cancelled,
                                        "text consumer disconnected",
                                    ).into());
                                }
                                if is_image && !reported {
                                    job_scheduler.counters.vision_backpressure_waits.fetch_add(1, Ordering::Relaxed);
                                    reported = true;
                                }
                                if start.elapsed()
                                    >= Duration::from_secs(
                                        job_scheduler.config.consumer_timeout_seconds,
                                    )
                                {
                                    c.cancel();
                                    return Err(Error::new(
                                        ErrorCode::Cancelled,
                                        "text consumer backpressure timeout",
                                    ).into());
                                }
                                std::thread::park_timeout(Duration::from_millis(5));
                            }
                            state.record_delivery(delivery_start.elapsed().as_secs_f64()*1000.);
                        }
                    }
                    Ok(model.finish_generation(state)?)
                });
                let result = tokio::select! {
                    r = &mut job => r,
                    _ = sender.closed() => {engine_cancel.cancel(); job.await},
                    _ = scheduler.shutdown.cancelled() => {engine_cancel.cancel(); job.await},
                    _ = tokio::time::sleep_until(deadline) => {engine_cancel.cancel(); job.await},
                };
                scheduler.counters.active.fetch_sub(1, Ordering::Relaxed);
                result?
            }
            .await;
            // No blocking job still owns compute or session buffers at this point.
            let event = match result {
                Ok((result, cache)) => {
                    if let Some(turn) = &mut turn {
                        turn.cache = cache.map(crate::sessions::TextCache::Qwen35);
                    }
                    scheduler.counters.completed.fetch_add(1, Ordering::Relaxed);
                    GenerationEvent::Finished(result)
                }
                Err(error) => {
                    if matches!(
                        error.error.code,
                        ErrorCode::Cancelled | ErrorCode::DeadlineExceeded
                    ) {
                        scheduler.counters.cancelled.fetch_add(1, Ordering::Relaxed);
                    } else {
                        scheduler.counters.failed.fetch_add(1, Ordering::Relaxed);
                    }
                    GenerationEvent::Failed(error)
                }
            };
            let event = finalize_turn(turn, event).await;
            let _ = tokio::time::timeout(
                Duration::from_secs(scheduler.config.consumer_timeout_seconds),
                sender.send(event),
            )
            .await;
        });
        Ok(GenerationHandle {
            receiver,
            cancel,
            _vision: None,
            _qwen35: Some(owner),
        })
    }
}

#[cfg(test)]
mod phase_tests {
    use super::*;
    fn scheduler(concurrency: usize) -> Scheduler {
        Scheduler::new(
            ServerConfig {
                max_concurrency: concurrency,
                ..Default::default()
            },
            Budget::new(512 << 20),
            CancellationToken::new(),
        )
    }
    #[tokio::test]
    async fn image_delivery_does_not_consume_text_phase_or_balanced_permits() {
        for concurrency in [1, 2] {
            for profile in [Profile::Balanced, Profile::LowMemory] {
                let scheduler = scheduler(concurrency);
                let image = scheduler.try_hybrid_execution(profile, true).unwrap();
                let text = scheduler
                    .try_hybrid_execution(profile, false)
                    .expect("an admitted text peer must have its own phase");
                if profile == Profile::LowMemory || scheduler.max_concurrency() == 1 {
                    assert!(scheduler.try_hybrid_execution(profile, true).is_none());
                    assert!(scheduler.try_hybrid_execution(profile, false).is_none());
                }
                drop(image);
                drop(text);
                assert!(scheduler.try_hybrid_execution(profile, true).is_some());
                assert!(scheduler.try_hybrid_execution(profile, false).is_some());
            }
        }
    }
    #[tokio::test]
    async fn queued_image_acquires_its_phase_while_text_is_still_pinned() {
        for profile in [Profile::Balanced, Profile::LowMemory] {
            let scheduler = scheduler(1);
            let image = scheduler.try_hybrid_execution(profile, true).unwrap();
            let text = scheduler.try_hybrid_execution(profile, false).unwrap();
            let waiting = scheduler.clone();
            let request = tokio::spawn(async move {
                waiting
                    .acquire_hybrid_execution(profile, true)
                    .await
                    .unwrap()
            });
            tokio::task::yield_now().await;
            assert!(!request.is_finished());
            drop(image);
            let next = tokio::time::timeout(std::time::Duration::from_secs(1), request)
                .await
                .unwrap()
                .unwrap();
            assert!(scheduler.try_hybrid_execution(profile, false).is_none());
            drop(next);
            drop(text);
        }
    }
}

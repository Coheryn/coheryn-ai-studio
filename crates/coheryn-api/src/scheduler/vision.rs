use super::compute::ComputeLease;
use super::*;
use coheryn_engine::{
    memory::Reservation,
    multimodal::{
        qwen2vl::{self, Qwen2Vl},
        vision::VisionOptions,
    },
};
use std::{
    cell::RefCell,
    time::{Duration, Instant},
};

/// Held by the response stream after compute has ended, including slow consumers.
pub(super) struct VisionOwner {
    pub _model: Arc<Qwen2Vl>,
    pub _transport: Reservation,
}
struct CancelOnDrop(Option<Cancellation>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if let Some(c) = &self.0 {
            c.cancel();
        }
    }
}
struct VisionSubmission {
    model: Arc<Qwen2Vl>,
    request: GenerationRequest,
    prepared: qwen2vl::PreparedRequest,
    transport: Reservation,
    cancel: Cancellation,
    deadline: tokio::time::Instant,
}
impl Scheduler {
    pub fn vision_options(
        &self,
        model: &Qwen2Vl,
        request: &GenerationRequest,
    ) -> Result<qwen2vl::Options> {
        request.validate()?;
        crate::vision::validate_controls(request)?;
        let profile = Profile::parse(&request.profile)?;
        let threads = planner::execution_threads(
            profile,
            request.threads,
            crate::hardware::probe().logical_cpus,
            self.config.max_concurrency,
        )?;
        Ok(qwen2vl::Options {
            context: Some(request.context.unwrap_or(model.context_limit())),
            max_tokens: request.max_tokens,
            threads,
            vision: VisionOptions {
                threads,
                ..Default::default()
            },
        })
    }
    pub async fn submit_vision(
        &self,
        model: Arc<Qwen2Vl>,
        request: GenerationRequest,
        messages: Vec<qwen2vl::ChatMessage>,
        images: Vec<Vec<u8>>,
        payload: Reservation,
    ) -> Result<GenerationHandle> {
        if self.shutdown.is_cancelled() {
            return Err(Error::new(ErrorCode::Unavailable, "server is shutting down").into());
        }
        let options = self.vision_options(&model, &request)?;
        let seconds = request
            .timeout_seconds
            .unwrap_or(self.config.generation_timeout_seconds)
            .min(self.config.generation_timeout_seconds);
        let deadline = tokio::time::Instant::now() + Duration::from_secs(seconds);
        let cancel = Cancellation::default().with_timeout(Some(seconds))?;
        let mut guard = CancelOnDrop(Some(cancel.clone()));
        // Bound preprocessing/tokenization jobs with the same shared request slots.
        let slot = self
            .slots
            .clone()
            .try_acquire_owned()
            .map_err(|_| Error::new(ErrorCode::QueueFull, "shared inference queue is full"))?;
        let transport = self.budget.reserve(8 << 20)?;
        let engine = model.clone();
        let c = cancel.clone();
        let mut prepare = tokio::task::spawn_blocking(move || {
            engine.prepare_owned(&messages, images, &options, &c)
        });
        let prepared = tokio::select! {
            result = &mut prepare => result??,
            _ = self.shutdown.cancelled() => { cancel.cancel(); let _ = prepare.await; return Err(Error::new(ErrorCode::Cancelled, "server shutdown cancelled image preparation").into()); },
            _ = tokio::time::sleep_until(deadline) => { cancel.cancel(); let _ = prepare.await; return Err(Error::new(ErrorCode::DeadlineExceeded, "image preparation deadline exceeded").into()); },
        };
        if request.max_memory.is_some_and(|n| {
            prepared
                .plan()
                .peak_request_bytes
                .saturating_add(model.resident_bytes())
                .saturating_add(transport.bytes())
                .saturating_add(payload.bytes())
                > n
        }) {
            return Err(Error::resource(
                "max_memory cannot cover image model metadata, transport and request capacity",
            )
            .into());
        }
        // The owned prepared job now charges the decoded attachments and prompt buffers.
        // Release the retired JSON/Base64 ingress stage before it waits for compute.
        drop(payload);
        drop(slot);
        let result = self.enqueue_vision(
            VisionSubmission {
                model,
                request,
                prepared,
                transport,
                cancel,
                deadline,
            },
            None,
        );
        if result.is_ok() {
            guard.0.take();
        }
        result
    }
    pub fn submit_managed_vision(
        &self,
        model: Arc<Qwen2Vl>,
        request: GenerationRequest,
        prepared: qwen2vl::PreparedRequest,
        payload: Reservation,
        mut turn: Box<crate::sessions::TurnLease>,
    ) -> Result<GenerationHandle> {
        let admission = (|| -> Result<_> {
            self.vision_options(&model, &request)?;
            let transport = self.budget.reserve(8 << 20)?;
            if request.max_memory.is_some_and(|n| {
                prepared
                    .plan()
                    .peak_request_bytes
                    .saturating_add(model.resident_bytes())
                    .saturating_add(transport.bytes())
                    .saturating_add(payload.bytes())
                    .saturating_add(turn.reserved_bytes())
                    > n
            }) {
                return Err(Error::resource("max_memory cannot cover managed image turn and complete execution/transport capacity").into());
            }
            let seconds = request
                .timeout_seconds
                .unwrap_or(self.config.generation_timeout_seconds)
                .min(self.config.generation_timeout_seconds);
            let cancel = Cancellation::default().with_timeout(Some(seconds))?;
            Ok((
                transport,
                cancel,
                tokio::time::Instant::now() + Duration::from_secs(seconds),
            ))
        })();
        let (transport, cancel, deadline) = match admission {
            Ok(v) => v,
            Err(error) => {
                turn.finish(&GenerationEvent::Failed(error.clone()))?;
                return Err(error);
            }
        };
        drop(payload);
        self.enqueue_vision(
            VisionSubmission {
                model,
                request,
                prepared,
                transport,
                cancel,
                deadline,
            },
            Some(turn),
        )
    }
    fn enqueue_vision(
        &self,
        submission: VisionSubmission,
        mut turn: Option<Box<crate::sessions::TurnLease>>,
    ) -> Result<GenerationHandle> {
        let VisionSubmission {
            model,
            request,
            prepared,
            transport,
            cancel,
            deadline,
        } = submission;
        let mut guard = CancelOnDrop(Some(cancel.clone()));
        let admission = (|| -> Result<_> {
            if self.shutdown.is_cancelled() {
                return Err(Error::new(ErrorCode::Unavailable, "server shutting down").into());
            }
            let profile = Profile::parse(&request.profile)?;
            let slot =
                self.slots.clone().try_acquire_owned().map_err(|_| {
                    Error::new(ErrorCode::QueueFull, "shared inference queue is full")
                })?;
            let n = if profile == Profile::LowMemory {
                self.config.max_concurrency as u32
            } else {
                1
            };
            let initial = self.vision_active.clone().try_acquire_many_owned(n).ok();
            let waiting = if initial.is_none() {
                Some(self.waiting.clone().try_acquire_owned().map_err(|_| {
                    Error::new(ErrorCode::QueueFull, "inference waiting queue is full")
                })?)
            } else {
                None
            };
            Ok((profile, slot, initial, waiting))
        })();
        let (profile, slot, initial, waiting) = match admission {
            Ok(v) => v,
            Err(error) => {
                if let Some(turn) = &mut turn {
                    turn.finish(&GenerationEvent::Failed(error.clone()))?;
                }
                return Err(error);
            }
        };
        let queued = waiting.is_some();
        if queued {
            self.counters.queued.fetch_add(1, Ordering::Relaxed);
        }
        let (sender, receiver) = mpsc::channel(DELIVERY_CHANNEL_SLOTS);
        let scheduler = self.clone();
        let engine_cancel = cancel.clone();
        let output_limit = turn.as_ref().map(|t| t.output_limit());
        let owner = VisionOwner {
            _model: model.clone(),
            _transport: transport,
        };
        tokio::spawn(async move {
            let _slot = slot;
            let acquire = async {
                if let Some(p) = initial {
                    return Some(p);
                }
                let n = if profile == Profile::LowMemory {
                    scheduler.config.max_concurrency as u32
                } else {
                    1
                };
                scheduler
                    .vision_active
                    .clone()
                    .acquire_many_owned(n)
                    .await
                    .ok()
            };
            let permits = tokio::select! {
                p = tokio::time::timeout(scheduler.config.queue_timeout(), acquire) => p.ok().flatten(),
                _ = sender.closed() => None,
                _ = scheduler.shutdown.cancelled() => None,
                _ = tokio::time::sleep_until(deadline) => None,
            };
            if queued {
                scheduler.counters.queued.fetch_sub(1, Ordering::Relaxed);
            }
            drop(waiting);
            let result: Result<GenerationResult> = async {
                let _permits = permits.ok_or_else(|| {
                    Error::new(
                        ErrorCode::Cancelled,
                        "queued image request cancelled or timed out",
                    )
                })?;
                // Tensor units share the FIFO gate. Reservations and model/session ownership
                // remain held while this phase yields to text/embedding requests.
                let compute = tokio::select! {
                    p = scheduler.compute.clone().acquire_owned() => p.ok(),
                    _ = sender.closed() => None,
                    _ = scheduler.shutdown.cancelled() => None,
                    _ = tokio::time::sleep_until(deadline) => None,
                }
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::Cancelled,
                        "image request cancelled waiting for compute",
                    )
                })?;
                engine_cancel.check()?;
                let c = engine_cancel.clone();
                let tx = sender.clone();
                let counters = scheduler.counters.clone();
                let consumer_timeout =
                    Duration::from_secs(scheduler.config.consumer_timeout_seconds);
                scheduler.counters.active.fetch_add(1, Ordering::Relaxed);
                let scheduler_for_job = scheduler.clone();
                let mut job = tokio::task::spawn_blocking(move || -> Result<GenerationResult> {
                    let gate = RefCell::new(ComputeLease::new(
                        compute,
                        &scheduler_for_job,
                        c.clone(),
                        tx.clone(),
                        deadline,
                        true,
                    ));
                    let mut cooperate = || gate.borrow_mut().yield_unit();
                    let mut delivered = 0usize;
                    let response =
                        model.generate_cooperative(prepared, &c, &mut cooperate, |token| {
                            gate.borrow_mut().release();
                            delivered = delivered
                                .checked_add(token.text.len())
                                .ok_or_else(|| Error::resource("image output byte overflow"))?;
                            if output_limit.is_some_and(|limit| delivered > limit) {
                                return Err(Error::resource(
                                    "image response exceeds session history quota",
                                ));
                            }
                            let mut pending = GenerationEvent::Token(token);
                            let start = Instant::now();
                            let mut reported = false;
                            loop {
                                c.check()?;
                                // Keep one channel slot for the terminal result/error.
                                if tx.capacity() > 1 {
                                    match tx.try_send(pending) {
                                        Ok(()) => {
                                            gate.borrow_mut().acquire()?;
                                            return Ok(());
                                        }
                                        Err(mpsc::error::TrySendError::Closed(_)) => {
                                            return Err(Error::new(
                                                ErrorCode::Cancelled,
                                                "image consumer disconnected",
                                            ));
                                        }
                                        Err(mpsc::error::TrySendError::Full(event)) => {
                                            pending = event
                                        }
                                    }
                                } else if tx.is_closed() {
                                    return Err(Error::new(
                                        ErrorCode::Cancelled,
                                        "image consumer disconnected",
                                    ));
                                }
                                if !reported {
                                    counters
                                        .vision_backpressure_waits
                                        .fetch_add(1, Ordering::Relaxed);
                                    reported = true;
                                }
                                if start.elapsed() >= consumer_timeout {
                                    c.cancel();
                                    return Err(Error::new(
                                        ErrorCode::Cancelled,
                                        "image consumer backpressure timeout",
                                    ));
                                }
                                std::thread::park_timeout(Duration::from_millis(5));
                            }
                        })?;
                    let mut result = crate::vision::generation_result(response);
                    let mut gate = gate.borrow_mut();
                    gate.release();
                    if let Some(vision) = &mut result.vision {
                        vision.cooperative =
                            Some(Box::new(coheryn_engine::inference::VisionExecution {
                                profile: "coheryn-vlm-cooperative-v1".into(),
                                units: gate.units,
                                yields: gate.yields,
                                compute_lease_ms: gate.compute_ms,
                                waiting_for_compute_ms: gate.wait_ms,
                            }));
                    }
                    Ok(result)
                });
                let result = tokio::select! {
                    r = &mut job => r,
                    _ = sender.closed() => { engine_cancel.cancel(); job.await },
                    _ = scheduler.shutdown.cancelled() => { engine_cancel.cancel(); job.await },
                    _ = tokio::time::sleep_until(deadline) => { engine_cancel.cancel(); job.await },
                };
                scheduler.counters.active.fetch_sub(1, Ordering::Relaxed);
                result?
            }
            .await;
            let event = match result {
                Ok(result) => {
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
        // Transfer cancellation responsibility to the response handle without cancelling it.
        guard.0.take();
        Ok(GenerationHandle {
            receiver,
            cancel,
            _vision: Some(owner),
            _qwen35: None,
        })
    }
}

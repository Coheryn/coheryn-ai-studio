use super::*;
use crate::{
    embeddings::{self, EmbeddingRequest, EmbeddingSpace},
    resident::ResidentEmbedding,
};
use coheryn_engine::{embedding::EmbeddingJob, memory::Reservation};
use tokio::{sync::oneshot, task::JoinHandle, time::Instant};

pub struct EmbeddingDelivery {
    pub bytes: Vec<u8>,
    pub space: EmbeddingSpace,
    pub token_counts: String,
    pub truncated: String,
    pub diagnostics: serde_json::Value,
    pub(crate) _model: Arc<ResidentEmbedding>,
    pub(crate) _serialization: Reservation,
}
pub struct EmbeddingHandle {
    receiver: oneshot::Receiver<Result<EmbeddingDelivery>>,
    cancel: Cancellation,
}
impl Drop for EmbeddingHandle {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}
impl EmbeddingHandle {
    pub async fn receive(&mut self) -> Result<EmbeddingDelivery> {
        (&mut self.receiver).await.map_err(|_| {
            Error::new(
                ErrorCode::Internal,
                "embedding worker ended without a result",
            )
        })?
    }
}
type EmbeddingPermits = (OwnedSemaphorePermit, ExecutionPermits);
impl Scheduler {
    fn try_embedding_execution(&self, profile: Profile) -> Option<EmbeddingPermits> {
        let embedding = self.embedding_active.clone().try_acquire_owned().ok()?;
        Some((embedding, self.try_execution(profile)?))
    }
    async fn acquire_embedding_execution(&self, profile: Profile) -> Option<EmbeddingPermits> {
        let embedding = self.embedding_active.clone().acquire_owned().await.ok()?;
        let balanced = if profile == Profile::Balanced {
            Some(self.balanced.clone().acquire_owned().await.ok()?)
        } else {
            None
        };
        let count = if profile == Profile::LowMemory {
            self.config.max_concurrency as u32
        } else {
            1
        };
        let active = self.active.clone().acquire_many_owned(count).await.ok()?;
        Some((embedding, (balanced, active)))
    }
    pub fn submit_embedding(
        &self,
        model: Arc<ResidentEmbedding>,
        request: EmbeddingRequest,
        payload: Reservation,
    ) -> Result<EmbeddingHandle> {
        let result = self.submit_embedding_inner(model, request, payload);
        if result.is_err() {
            self.counters.rejected.fetch_add(1, Ordering::Relaxed);
        }
        result
    }
    fn submit_embedding_inner(
        &self,
        model: Arc<ResidentEmbedding>,
        request: EmbeddingRequest,
        payload: Reservation,
    ) -> Result<EmbeddingHandle> {
        request.validate(&self.config.embeddings)?;
        if self.shutdown.is_cancelled() {
            return Err(Error::new(ErrorCode::Unavailable, "server is shutting down").into());
        }
        if request
            .dimensions
            .is_some_and(|n| n != model.space.dimension)
        {
            return Err(Error::new(
                ErrorCode::UnsupportedEmbeddingDimensions,
                format!(
                    "model supports exactly {} embedding dimensions",
                    model.space.dimension
                ),
            )
            .into());
        }
        let options = request.options(model.model.context_limit(), &self.config.embeddings)?;
        let bytes: Vec<_> = request.input.texts().iter().map(String::len).collect();
        let allowance = embeddings::response_allowance(
            bytes.len(),
            model.space.dimension,
            self.config.embeddings.response_bytes,
        )?;
        // Reserve transport first so adaptive packing sees every live stage.
        let serialization = self.budget.reserve(allowance)?;
        let ceiling = if let Some(limit) = request.max_memory {
            let fixed = coheryn_engine::embedding::estimate_model_bytes(
                model.model.descriptor(),
                model.model.config(),
            )?
            .checked_add(allowance)
            .and_then(|n| n.checked_add(payload.bytes()))
            .ok_or_else(|| Error::resource("embedding request estimate overflow"))?;
            Some(limit.checked_sub(fixed).ok_or_else(|| {
                Error::resource("max_memory cannot cover embedding metadata and transport")
            })?)
        } else {
            None
        };
        let admission = model.model.reserve_queued_capped(
            &bytes,
            &options,
            &crate::hardware::probe(),
            ceiling,
        )?;
        let embedding_slot = self
            .embedding_slots
            .clone()
            .try_acquire_owned()
            .map_err(|_| Error::new(ErrorCode::QueueFull, "embedding request queue is full"))?;
        let slot = self
            .slots
            .clone()
            .try_acquire_owned()
            .map_err(|_| Error::new(ErrorCode::QueueFull, "shared inference queue is full"))?;
        let initial = self.try_embedding_execution(options.profile);
        let waiting =
            if initial.is_none() {
                Some(self.waiting.clone().try_acquire_owned().map_err(|_| {
                    Error::new(ErrorCode::QueueFull, "inference waiting queue is full")
                })?)
            } else {
                None
            };
        let queued = waiting.is_some();
        if queued {
            self.counters.queued.fetch_add(1, Ordering::Relaxed);
        }
        let (mut sender, receiver) = oneshot::channel();
        let cancel = Cancellation::default();
        let compute_cancel = cancel.clone();
        let scheduler = self.clone();
        tokio::spawn(async move {
            let (_slot, _embedding_slot, _payload) = (slot, embedding_slot, payload);
            let deadline = Instant::now()
                + std::time::Duration::from_secs(scheduler.config.embedding_timeout_seconds);
            let permits = match initial {
                Some(p) => Some(p),
                None => tokio::select! {
                    _=sender.closed()=>None,
                    _=scheduler.shutdown.cancelled()=>None,
                    p=tokio::time::timeout(scheduler.config.queue_timeout(),scheduler.acquire_embedding_execution(options.profile))=>p.ok().flatten(),
                },
            };
            if queued {
                scheduler.counters.queued.fetch_sub(1, Ordering::Relaxed);
            }
            drop(waiting);
            let result=async {
                let mut permits=permits.ok_or_else(||ApiError::from(Error::new(ErrorCode::Cancelled,"queued embedding cancelled or timed out")))?;
                let engine=model.clone();let cancel=compute_cancel.clone();
                let total_limit=scheduler.config.embeddings.total_tokens;
                let encoding=request.encoding_format;let model_id=request.model.clone();
                scheduler.counters.active.fetch_add(1,Ordering::Relaxed);
                let prepare=tokio::task::spawn_blocking(move||->Result<EmbeddingJob> {
                    let texts:Vec<_>=request.input.texts().iter().map(String::as_str).collect();
                    let job=engine.model.prepare_admitted(&texts,admission,&cancel)?;
                    if job.prompt_tokens()>total_limit {return Err(embeddings::batch_limit("aggregate embedding token count exceeds configured limit"));}
                    Ok(job)
                });
                let prepared=wait_job(prepare,&mut sender,&compute_cancel,&scheduler.shutdown,deadline).await;
                scheduler.counters.active.fetch_sub(1,Ordering::Relaxed);
                let mut job=prepared?;
                drop(_payload);
                // Tokenization/session allocation is a bounded phase; execution retains these permits
                // for its first microbatch, then returns to the shared FIFO semaphore.
                while !job.is_finished() {
                    let engine=model.clone();let cancel=compute_cancel.clone();
                    scheduler.counters.active.fetch_add(1,Ordering::Relaxed);
                    let step=tokio::task::spawn_blocking(move||->Result<EmbeddingJob> {engine.model.step(&mut job,&cancel)?;Ok(job)});
                    let stepped=wait_job(step,&mut sender,&compute_cancel,&scheduler.shutdown,deadline).await;
                    scheduler.counters.active.fetch_sub(1,Ordering::Relaxed);
                    job=stepped?;
                    if job.is_finished(){break;}
                    drop(permits);
                    scheduler.counters.queued.fetch_add(1,Ordering::Relaxed);
                    let acquire=scheduler.acquire_embedding_execution(options.profile);
                    let next=tokio::select! {
                        _=sender.closed()=>None,
                        _=scheduler.shutdown.cancelled()=>None,
                        _=tokio::time::sleep_until(deadline)=>None,
                        result=tokio::time::timeout(scheduler.config.queue_timeout(),acquire)=>result.ok().flatten(),
                    };
                    scheduler.counters.queued.fetch_sub(1,Ordering::Relaxed);
                    permits=next.ok_or_else(||ApiError::from(Error::new(ErrorCode::Cancelled,"embedding cancelled while yielding to shared scheduler")))?;
                }
                let engine=model.clone();let cancel=compute_cancel.clone();
                scheduler.counters.active.fetch_add(1,Ordering::Relaxed);
                let serialization_job=tokio::task::spawn_blocking(move||->Result<EmbeddingDelivery> {
                    cancel.check()?;
                    let result=engine.model.finish(job)?;
                    let token_counts=result.data.iter().map(|i|i.prompt_tokens.to_string()).collect::<Vec<_>>().join(",");
                    let truncated=result.data.iter().filter(|i|i.truncated).map(|i|i.index.to_string()).collect::<Vec<_>>().join(",");
                    let bytes=embeddings::serialize_response(&result,&model_id,encoding,allowance)?;
                    let diagnostics=serde_json::json!({"plan":result.plan,"tokenization_ms":result.tokenization_ms,"encoding_ms":result.encoding_ms});
                    cancel.check()?;
                    Ok(EmbeddingDelivery {bytes,space:engine.space.clone(),token_counts,truncated,diagnostics,_model:engine,_serialization:serialization})
                });
                let result=wait_job(serialization_job,&mut sender,&compute_cancel,&scheduler.shutdown,deadline).await;
                scheduler.counters.active.fetch_sub(1,Ordering::Relaxed);
                result
            }.await;
            match &result {
                Ok(_) => {
                    scheduler.counters.completed.fetch_add(1, Ordering::Relaxed);
                }
                Err(e) if e.error.code == ErrorCode::Cancelled => {
                    scheduler.counters.cancelled.fetch_add(1, Ordering::Relaxed);
                }
                Err(_) => {
                    scheduler.counters.failed.fetch_add(1, Ordering::Relaxed);
                }
            }
            let _ = sender.send(result);
        });
        Ok(EmbeddingHandle { receiver, cancel })
    }
}
async fn wait_job<T: Send + 'static>(
    mut job: JoinHandle<Result<T>>,
    sender: &mut oneshot::Sender<Result<EmbeddingDelivery>>,
    cancel: &Cancellation,
    shutdown: &CancellationToken,
    deadline: Instant,
) -> Result<T> {
    tokio::select! {
        result=&mut job=>result?,
        _=sender.closed()=>{cancel.cancel();let _=job.await;Err(Error::new(ErrorCode::Cancelled,"embedding consumer disconnected").into())},
        _=shutdown.cancelled()=>{cancel.cancel();let _=job.await;Err(Error::new(ErrorCode::Cancelled,"server shutdown cancelled embedding").into())},
        _=tokio::time::sleep_until(deadline)=>{cancel.cancel();let _=job.await;Err(Error::new(ErrorCode::Cancelled,"embedding deadline exceeded").into())},
    }
}

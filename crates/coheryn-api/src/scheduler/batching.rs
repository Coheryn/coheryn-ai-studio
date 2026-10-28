//! Bounded dynamic packing of live, compatible model-token work.
use super::*;
use coheryn_engine::inference::GenerationState;
use std::{collections::VecDeque, sync::OnceLock, time::Instant};
use tokio::sync::oneshot;
struct Queued(Arc<Counters>);
impl Drop for Queued {
    fn drop(&mut self) {
        self.0
            .waiting_generation_units
            .fetch_sub(1, Ordering::Relaxed);
    }
}
struct Work {
    waiting: Option<Queued>,
    model: Arc<Model>,
    state: GenerationState,
    reply: oneshot::Sender<coheryn_engine::Result<GenerationState>>,
    enqueued: Instant,
}
#[derive(Clone)]
pub(super) struct Batcher {
    sender: Arc<OnceLock<mpsc::Sender<Work>>>,
    compute: Arc<Semaphore>,
    counters: Arc<Counters>,
    shutdown: CancellationToken,
}
impl Batcher {
    pub fn new(
        compute: Arc<Semaphore>,
        counters: Arc<Counters>,
        shutdown: CancellationToken,
    ) -> Self {
        Self {
            sender: Arc::new(OnceLock::new()),
            compute,
            counters,
            shutdown,
        }
    }
    /// Called by a request-owned blocking worker. Moving the state transfers its
    /// reservations with it, and the reply acknowledges completion before release.
    pub fn advance(&self, model: Arc<Model>, state: GenerationState) -> Result<GenerationState> {
        let sender = self.sender.get_or_init(|| {
            let (sender, receiver) = mpsc::channel(16);
            tokio::spawn(run(
                receiver,
                self.compute.clone(),
                self.counters.clone(),
                self.shutdown.clone(),
            ));
            sender
        });
        let (reply, response) = oneshot::channel();
        self.counters
            .waiting_generation_units
            .fetch_add(1, Ordering::Relaxed);
        sender
            .blocking_send(Work {
                waiting: Some(Queued(self.counters.clone())),
                model,
                state,
                reply,
                enqueued: Instant::now(),
            })
            .map_err(|_| Error::new(ErrorCode::Cancelled, "native batch worker stopped"))?;
        response
            .blocking_recv()
            .map_err(|_| Error::new(ErrorCode::Internal, "native batch worker failed"))?
            .map_err(Into::into)
    }
}
async fn run(
    mut receiver: mpsc::Receiver<Work>,
    compute: Arc<Semaphore>,
    counters: Arc<Counters>,
    shutdown: CancellationToken,
) {
    let mut pending = VecDeque::<Work>::new();
    loop {
        if shutdown.is_cancelled() {
            receiver.close();
            while let Ok(work) = receiver.try_recv() {
                pending.push_back(work);
            }
            for work in pending.drain(..) {
                let _ = work.reply.send(Err(Error::new(
                    ErrorCode::Cancelled,
                    "server shutdown cancelled batch work",
                )));
            }
            return;
        }
        if pending.is_empty() {
            tokio::select! {
                work=receiver.recv()=>match work {Some(work)=>pending.push_back(work),None=>return},
                _=shutdown.cancelled()=>continue,
            }
        }
        while pending.len() < 16 {
            match receiver.try_recv() {
                Ok(work) => pending.push_back(work),
                Err(_) => break,
            }
        }
        pending.retain(|work| !work.reply.is_closed());
        let Some(first) = pending.pop_front() else {
            continue;
        };
        let mut batch = vec![first];
        // The oldest request selects the group. Incompatible groups keep FIFO order;
        // compatible arrivals join at this boundary without resetting existing lanes.
        let count = pending.len();
        for _ in 0..count {
            let work = pending.pop_front().expect("bounded pending length");
            if Arc::ptr_eq(&work.model, &batch[0].model) && work.state.compatible(&batch[0].state) {
                batch.push(work);
            } else {
                pending.push_back(work);
            }
        }
        let permit = tokio::select! {
            p=compute.clone().acquire_owned()=>p.ok(),
            _=shutdown.cancelled()=>None,
        };
        let Some(permit) = permit else {
            for work in batch {
                let _ = work.reply.send(Err(Error::new(
                    ErrorCode::Cancelled,
                    "batch compute cancelled",
                )));
            }
            continue;
        };
        // Requests that arrived while another task held compute can join the same
        // compatible unit. Keep incompatible work in its original FIFO position.
        while pending.len() + batch.len() < 16 {
            match receiver.try_recv() {
                Ok(work) => pending.push_back(work),
                Err(_) => break,
            }
        }
        let count = pending.len();
        for _ in 0..count {
            let work = pending.pop_front().expect("bounded pending length");
            if !work.reply.is_closed()
                && Arc::ptr_eq(&work.model, &batch[0].model)
                && work.state.compatible(&batch[0].state)
            {
                batch.push(work);
            } else {
                pending.push_back(work);
            }
        }
        let stats = counters.clone();
        // Responses are sent from the owning worker itself, so a panic closes every
        // affected reply and drops its buffers instead of stranding a receiver.
        let _ = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            for work in &mut batch {
                work.waiting.take();
                work.state
                    .record_scheduler_wait(work.enqueued.elapsed().as_secs_f64() * 1000.);
            }
            let model = batch[0].model.clone();
            let result = {
                let mut states: Vec<_> = batch.iter_mut().map(|w| &mut w.state).collect();
                model.step_batch(&mut states)
            };
            if let Ok(step) = &result {
                stats.generation_units.fetch_add(1, Ordering::Relaxed);
                stats
                    .batched_generation_units
                    .fetch_add(u64::from(step.width > 1), Ordering::Relaxed);
                stats
                    .forwarded_tokens
                    .fetch_add(step.completed_tokens as u64, Ordering::Relaxed);
                stats
                    .maximum_batch_width
                    .fetch_max(step.width as u64, Ordering::Relaxed);
                stats
                    .matrix_tile_reads
                    .fetch_add(step.matrix_tiles, Ordering::Relaxed);
                stats
                    .matrix_read_bytes
                    .fetch_add(step.matrix_bytes, Ordering::Relaxed);
            }
            for work in batch {
                let response = match &result {
                    Ok(_) => Ok(work.state),
                    Err(error) => {
                        drop(work.state);
                        Err(error.clone())
                    }
                };
                let _ = work.reply.send(response);
            }
        })
        .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_model as fixture;
    #[tokio::test]
    async fn arrivals_waiting_for_compute_pack_into_one_real_shared_weight_unit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, fixture::tiny_model()).unwrap();
        let model = Arc::new(
            Model::load(
                path,
                coheryn_engine::ResourceLimits {
                    context: 128,
                    max_memory: 64 << 20,
                },
            )
            .unwrap(),
        );
        let baseline = model.memory().reserved;
        let compute = Arc::new(Semaphore::new(1));
        let hold = compute.clone().acquire_owned().await.unwrap();
        let counters = Arc::new(Counters::default());
        let shutdown = CancellationToken::new();
        let (sender, receiver) = mpsc::channel(16);
        tokio::spawn(run(receiver, compute, counters.clone(), shutdown.clone()));
        let mut replies = Vec::new();
        for _ in 0..3 {
            let state = model
                .begin_generation(
                    "hi",
                    &coheryn_engine::GenerationOptions {
                        max_tokens: 4,
                        ..Default::default()
                    },
                    model.reserve_session(128, 0).unwrap(),
                    &Cancellation::default(),
                    None,
                )
                .unwrap();
            let (reply, response) = oneshot::channel();
            counters
                .waiting_generation_units
                .fetch_add(1, Ordering::Relaxed);
            assert!(
                sender
                    .send(Work {
                        waiting: Some(Queued(counters.clone())),
                        model: model.clone(),
                        state,
                        reply,
                        enqueued: Instant::now(),
                    })
                    .await
                    .is_ok()
            );
            replies.push(response);
        }
        // All three sends are acknowledged before releasing compute, so this does
        // not depend on a timing window between a blocking sender and the actor.
        drop(hold);
        for reply in replies {
            drop(reply.await.unwrap().unwrap());
        }
        let stats = counters.snapshot();
        assert_eq!(stats.maximum_batch_width, 3);
        assert_eq!(stats.generation_units, 1);
        assert_eq!(stats.forwarded_tokens, 3);
        let matrix_bytes: u64 = model
            .descriptor
            .tensors
            .iter()
            .filter(|t| t.dimensions.len() == 2 && t.name != "token_embd.weight")
            .map(|t| t.bytes)
            .sum();
        assert_eq!(
            stats.matrix_read_bytes, matrix_bytes,
            "each matrix tile read once for three independent sequences"
        );
        assert_eq!(model.memory().reserved, baseline);
        assert_eq!(model.memory().session_buffer_capacity_bytes, 0);
        shutdown.cancel();
    }
}

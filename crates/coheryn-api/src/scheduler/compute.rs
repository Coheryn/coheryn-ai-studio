use super::*;
use std::time::Instant;

/// A blocking worker parks only after all of its numeric workers have joined.
/// The FIFO semaphore is returned during every checkpoint and delivery wait.
pub(super) struct ComputeLease {
    vision: bool,
    permit: Option<OwnedSemaphorePermit>,
    compute: Arc<Semaphore>,
    counters: Arc<Counters>,
    cancel: Cancellation,
    shutdown: CancellationToken,
    sender: mpsc::Sender<GenerationEvent>,
    deadline: tokio::time::Instant,
    runtime: tokio::runtime::Handle,
    unit_start: Instant,
    pub(super) compute_ms: f64,
    pub(super) wait_ms: f64,
    pub(super) units: u64,
    pub(super) yields: u64,
}
impl ComputeLease {
    pub(super) fn new(
        permit: OwnedSemaphorePermit,
        scheduler: &Scheduler,
        cancel: Cancellation,
        sender: mpsc::Sender<GenerationEvent>,
        deadline: tokio::time::Instant,
        vision: bool,
    ) -> Self {
        if vision {
            scheduler
                .counters
                .vision_units
                .fetch_add(1, Ordering::Relaxed);
        } else {
            scheduler
                .counters
                .generation_units
                .fetch_add(1, Ordering::Relaxed);
        }
        Self {
            vision,
            permit: Some(permit),
            compute: scheduler.compute.clone(),
            counters: scheduler.counters.clone(),
            cancel,
            shutdown: scheduler.shutdown.clone(),
            sender,
            deadline,
            runtime: tokio::runtime::Handle::current(),
            unit_start: Instant::now(),
            compute_ms: 0.,
            wait_ms: 0.,
            units: 1,
            yields: 0,
        }
    }
    pub(super) fn release(&mut self) {
        if self.permit.take().is_some() {
            self.compute_ms += self.unit_start.elapsed().as_secs_f64() * 1000.;
            if self.vision {
                self.counters.vision_max_unit_micros.fetch_max(
                    self.unit_start.elapsed().as_micros().min(u64::MAX as u128) as u64,
                    Ordering::Relaxed,
                );
            }
        }
    }
    pub(super) fn acquire(&mut self) -> coheryn_engine::Result<()> {
        self.cancel.check()?;
        if self.permit.is_some() {
            return Ok(());
        }
        let waiting_counter = if self.vision {
            &self.counters.vision_waiting_units
        } else {
            &self.counters.waiting_generation_units
        };
        waiting_counter.fetch_add(1, Ordering::Relaxed);
        let waiting = Instant::now();
        let permit = self.runtime.block_on(async {
            tokio::select! {
                p=self.compute.clone().acquire_owned()=>p.ok(),
                _=self.sender.closed()=>None,
                _=self.shutdown.cancelled()=>None,
                _=tokio::time::sleep_until(self.deadline)=>None,
            }
        });
        self.wait_ms += waiting.elapsed().as_secs_f64() * 1000.;
        waiting_counter.fetch_sub(1, Ordering::Relaxed);
        match permit {
            Some(p) => {
                self.permit = Some(p);
                self.units += 1;
                self.unit_start = Instant::now();
                if self.vision {
                    self.counters.vision_units.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.counters
                        .generation_units
                        .fetch_add(1, Ordering::Relaxed);
                }
                self.cancel.check()
            }
            None => {
                self.cancel.cancel();
                Err(Error::new(
                    ErrorCode::Cancelled,
                    "request cancelled while yielding shared compute",
                ))
            }
        }
    }
    pub(super) fn yield_unit(&mut self) -> coheryn_engine::Result<()> {
        self.release();
        self.yields += 1;
        if self.vision {
            self.counters.vision_yields.fetch_add(1, Ordering::Relaxed);
        }
        self.acquire()
    }
}
impl Drop for ComputeLease {
    fn drop(&mut self) {
        self.release();
    }
}

//! Bounded, request-owned CPU workers. Join before releasing the session reservation.
use crate::{Error, Result};
use std::thread::JoinHandle;

pub(crate) struct CpuPool {
    pool: Option<rayon::ThreadPool>,
    workers: Vec<JoinHandle<()>>,
}
impl CpuPool {
    pub(crate) fn new(threads: usize) -> Result<Self> {
        crate::memory::worker_bytes(threads)?;
        let mut workers = Vec::new();
        workers
            .try_reserve_exact(threads)
            .map_err(|_| Error::resource("CPU worker handle allocation failed"))?;
        let result = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .stack_size(512 << 10)
            .thread_name(|i| format!("coheryn-cpu-{i}"))
            .spawn_handler(|worker| {
                let handle = std::thread::Builder::new()
                    .name(worker.name().unwrap_or("coheryn-cpu").to_owned())
                    .stack_size(worker.stack_size().unwrap_or(512 << 10))
                    .spawn(move || worker.run())?;
                workers.push(handle);
                Ok(())
            })
            .build();
        match result {
            Ok(pool) => Ok(Self {
                pool: Some(pool),
                workers,
            }),
            Err(error) => {
                for worker in workers {
                    let _ = worker.join();
                }
                Err(Error::resource(format!(
                    "CPU worker creation failed: {error}"
                )))
            }
        }
    }
    pub(crate) fn current_num_threads(&self) -> usize {
        self.workers.len()
    }
    pub(crate) fn install<T: Send>(&self, work: impl FnOnce() -> T + Send) -> T {
        self.pool.as_ref().expect("live CPU pool").install(work)
    }
}
impl Drop for CpuPool {
    fn drop(&mut self) {
        drop(self.pool.take());
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

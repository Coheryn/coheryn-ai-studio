use super::{Profile, Result, require};
use coheryn_api::service::Service;
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

struct Writer {
    stream: BufWriter<File>,
    bytes: usize,
}
#[derive(Clone)]
pub struct Log {
    writer: Arc<Mutex<Writer>>,
    context: serde_json::Map<String, Value>,
}
impl Log {
    pub fn new(path: &Path) -> Result<Self> {
        Ok(Self {
            writer: Arc::new(Mutex::new(Writer {
                stream: BufWriter::new(OpenOptions::new().create_new(true).write(true).open(path)?),
                bytes: 0,
            })),
            context: Default::default(),
        })
    }
    pub fn scoped(&self, context: Value) -> Self {
        let mut scoped = self.clone();
        scoped.context.extend(context.as_object().unwrap().clone());
        scoped
    }
    pub fn write(&self, mut value: Value) -> Result<()> {
        let row = value
            .as_object_mut()
            .ok_or("observer record must be an object")?;
        for (key, field) in &self.context {
            require(
                row.get(key).is_none_or(|existing| existing == field),
                "observer context collision",
            )?;
            row.insert(key.clone(), field.clone());
        }
        let bytes = serde_json::to_vec(&value)?;
        require(bytes.len() <= 2 << 20, "observer record exceeds bound")?;
        let mut writer = self.writer.lock().map_err(|_| "observer log poisoned")?;
        require(
            writer.bytes.saturating_add(bytes.len() + 1) <= 256 << 20,
            "observer log exceeds 256 MiB bound",
        )?;
        writer.stream.write_all(&bytes)?;
        writer.stream.write_all(b"\n")?;
        writer.bytes += bytes.len() + 1;
        Ok(())
    }
    pub fn flush(&self) -> Result<()> {
        self.writer
            .lock()
            .map_err(|_| "observer log poisoned")?
            .stream
            .flush()?;
        Ok(())
    }
}

fn process(system: &mut sysinfo::System) -> (Option<u64>, Option<u64>) {
    let pid = sysinfo::Pid::from_u32(std::process::id());
    system.refresh_processes_specifics(
        sysinfo::ProcessesToUpdate::Some(&[pid]),
        false,
        sysinfo::ProcessRefreshKind::nothing()
            .with_memory()
            .without_tasks(),
    );
    system
        .process(pid)
        .map(|p| {
            (
                (p.memory() > 0).then(|| p.memory()),
                (p.virtual_memory() > 0).then(|| p.virtual_memory()),
            )
        })
        .unwrap_or_default()
}
pub fn resident_memory() -> Option<u64> {
    sysinfo::System::new()
        .ok()
        .and_then(|mut s| process(&mut s).0)
}

#[derive(Default, Serialize)]
pub struct Observation {
    samples: u64,
    rss_samples: u64,
    sampled_peak_rss_bytes: Option<u64>,
    sampled_peak_virtual_bytes: Option<u64>,
    peak_reserved_bytes: u64,
    peak_engine_buffer_bytes: u64,
}
pub struct Monitor {
    stop: Arc<AtomicBool>,
    failure: Arc<Mutex<Option<String>>>,
    worker: Option<JoinHandle<Result<Observation>>>,
}
impl Monitor {
    pub fn start(service: Arc<Service>, profile: Profile, log: Log) -> Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let failure = Arc::new(Mutex::new(None));
        let (flag, failed, weak) = (stop.clone(), failure.clone(), Arc::downgrade(&service));
        let worker = std::thread::Builder::new().name("cpu-soak-observer".into()).spawn(move || {
            let mut observed = Observation::default(); let mut system = sysinfo::System::new().ok(); let start = Instant::now();
            while !flag.load(Ordering::Acquire) {
                let Some(service) = weak.upgrade() else { break; };
                let (rss, virtual_bytes) = system.as_mut().map(process).unwrap_or_default();
                let memory = service.budget.stats(); let scheduler = service.scheduler.counters.snapshot();
                observed.samples += 1; observed.rss_samples += u64::from(rss.is_some());
                observed.sampled_peak_rss_bytes = observed.sampled_peak_rss_bytes.max(rss);
                observed.sampled_peak_virtual_bytes = observed.sampled_peak_virtual_bytes.max(virtual_bytes);
                observed.peak_reserved_bytes = observed.peak_reserved_bytes.max(memory.peak_reserved);
                observed.peak_engine_buffer_bytes = observed.peak_engine_buffer_bytes.max(memory.peak_session_buffer_capacity_bytes);
                log.write(json!({"kind":"sample","elapsed_ms":start.elapsed().as_secs_f64()*1000.,"rss_bytes":rss,"virtual_bytes":virtual_bytes,"private_bytes":null,"memory":memory,"scheduler":scheduler}))?;
                let error = if profile.require_rss && rss.is_none() { Some("required process RSS sample is unavailable") }
                    else if rss.is_some_and(|n| n > profile.max_rss_bytes) { Some("sampled RSS exceeds frozen ceiling") }
                    else if memory.peak_reserved > profile.max_memory_bytes { Some("reservation exceeds frozen ceiling") } else { None };
                if let Some(error) = error {
                    *failed.lock().map_err(|_| "observer state poisoned")? = Some(error.into());
                    log.write(json!({"kind":"observer_failure","reason":error}))?; service.shutdown.cancel(); break;
                }
                drop(service);
                std::thread::park_timeout(Duration::from_millis(profile.sample_ms));
            }
            Ok(observed)
        })?;
        Ok(Self {
            stop,
            failure,
            worker: Some(worker),
        })
    }
    pub fn check(&self) -> Result<()> {
        if let Some(error) = self
            .failure
            .lock()
            .map_err(|_| "observer state poisoned")?
            .as_ref()
        {
            return Err(std::io::Error::other(error.clone()).into());
        }
        Ok(())
    }
    pub fn finish(mut self) -> Result<Observation> {
        self.stop.store(true, Ordering::Release);
        let worker = self.worker.take().unwrap();
        worker.thread().unpark();
        let result = worker.join().map_err(|_| "observer thread failed")??;
        self.check()?;
        Ok(result)
    }
}
impl Drop for Monitor {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            worker.thread().unpark();
            let _ = worker.join();
        }
    }
}

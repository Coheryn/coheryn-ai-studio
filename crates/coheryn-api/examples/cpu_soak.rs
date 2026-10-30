//! Development-only mixed native workload driver. No external inference process.
#[path = "../../coheryn-engine/tests/common/mod.rs"]
mod fixture;
#[path = "cpu_soak/metrics.rs"]
mod metrics;
#[path = "cpu_soak/models.rs"]
mod models;
#[path = "../../coheryn-engine/tests/common/vision.rs"]
mod vision_fixture;
#[path = "cpu_soak/workload.rs"]
mod workload;

use coheryn_api::{auth::Credentials, config::ServerConfig, http::router, service::Service};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{fs::OpenOptions, io::Read, path::Path, sync::Arc, time::Duration};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
fn require(condition: bool, message: &str) -> Result<()> {
    if !condition {
        return Err(std::io::Error::other(message).into());
    }
    Ok(())
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Profile {
    schema_version: u32,
    name: String,
    cycles: usize,
    warmup_cycles: usize,
    restart_every: usize,
    fault_every: usize,
    concurrency: Vec<usize>,
    execution_profiles: Vec<String>,
    max_memory_bytes: u64,
    max_rss_bytes: u64,
    max_drained_rss_growth_bytes: u64,
    require_rss: bool,
    sample_ms: u64,
    request_timeout_seconds: u64,
    cleanup_timeout_seconds: u64,
    consumer_timeout_seconds: u64,
}
impl Profile {
    fn validate(&self) -> Result<()> {
        require(
            self.schema_version == 1 && !self.name.is_empty() && self.name.len() <= 128,
            "invalid profile identity",
        )?;
        require(
            (1..=10000).contains(&self.cycles) && (1..=10).contains(&self.warmup_cycles),
            "invalid bounded cycle counts",
        )?;
        require(
            (1..=self.cycles).contains(&self.restart_every)
                && (1..=self.cycles).contains(&self.fault_every),
            "invalid restart/fault cadence",
        )?;
        require(
            !self.concurrency.is_empty()
                && self.concurrency.len() <= 2
                && self.concurrency.iter().all(|n| (1..=2).contains(n)),
            "soak concurrency must be one or two",
        )?;
        require(
            !self.execution_profiles.is_empty()
                && self.execution_profiles.len() <= 2
                && self
                    .execution_profiles
                    .iter()
                    .all(|p| matches!(p.as_str(), "balanced" | "low-memory")),
            "unsupported execution profile",
        )?;
        require(
            (128 << 20..=8u64 << 30).contains(&self.max_memory_bytes)
                && self.max_rss_bytes > 0
                && self.max_drained_rss_growth_bytes <= self.max_rss_bytes,
            "invalid memory acceptance limits",
        )?;
        require(
            (10..=1000).contains(&self.sample_ms)
                && (1..=1800).contains(&self.request_timeout_seconds)
                && (1..=60).contains(&self.cleanup_timeout_seconds)
                && (1..=60).contains(&self.consumer_timeout_seconds),
            "invalid timeout/sampling bounds",
        )
    }
}

fn read_json(path: &Path) -> Result<Value> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take((64 << 10) + 1)
        .read_to_end(&mut bytes)?;
    require(bytes.len() <= 64 << 10, "configuration exceeds 64 KiB")?;
    Ok(serde_json::from_slice(&bytes)?)
}

struct Shutdown(Arc<Service>);
impl Drop for Shutdown {
    fn drop(&mut self) {
        self.0.shutdown.cancel();
    }
}

async fn bootstrap(
    config: &ServerConfig,
    sources: &models::Sources,
    profile: &Profile,
    log: &metrics::Log,
) -> Result<Value> {
    let service = Service::open(
        config.clone(),
        Credentials {
            inference_token: "i".repeat(64),
            admin_token: "a".repeat(64),
        },
    )?;
    let guard = Shutdown(service.clone());
    let app = router(service.clone());
    let monitor = metrics::Monitor::start(
        service.clone(),
        profile.clone(),
        log.scoped(json!({"phase":"bootstrap"})),
    )?;
    sources.install(&service).await?;
    service.load("text".into(), 512).await?;
    let receipt = workload::seed_history(&app, profile).await?;
    workload::idle(&service, profile).await?;
    service.unload("text")?;
    let observations = monitor.finish()?;
    let budget = service.budget.clone();
    service.shutdown.cancel();
    drop(app);
    drop(guard);
    drop(service);
    require(
        budget.stats().reserved == 0,
        "bootstrap retained reservations",
    )?;
    log.write(json!({"kind":"bootstrap_complete","observations":observations,"reserved_after_close":budget.stats().reserved}))?;
    Ok(receipt)
}

async fn execute(
    profile: &Profile,
    sources: &models::Sources,
    directory: &Path,
    log: &metrics::Log,
) -> Result<Value> {
    let mut results = Vec::new();
    let ids = sources.ids();
    for &concurrency in &profile.concurrency {
        for execution_profile in &profile.execution_profiles {
            let scoped = log
                .scoped(json!({"concurrency":concurrency,"execution_profile":execution_profile}));
            let log = &scoped;
            let store = tempfile::Builder::new()
                .prefix("store-")
                .tempdir_in(directory)?;
            let mut config = ServerConfig {
                data_dir: store.path().join("vault"),
                max_memory: profile.max_memory_bytes,
                max_context: 512,
                queue_capacity: 3,
                max_concurrency: concurrency,
                max_loaded_models: 5,
                generation_timeout_seconds: profile.request_timeout_seconds,
                embedding_timeout_seconds: profile.request_timeout_seconds,
                consumer_timeout_seconds: profile.consumer_timeout_seconds,
                queue_timeout_seconds: profile.request_timeout_seconds,
                ..Default::default()
            };
            config.persistence.enabled = true;
            config.cache.global_bytes = 0;
            config.cache.session_bytes = 0;
            config.embeddings.batch_size = 2;
            config.embeddings.batch_tokens = 512;
            let mut expected = None;
            let mut rss_baseline = None;
            let mut history_baseline = None;
            let mut loaded_baseline = None;
            let mut counters = json!({});
            let receipt = bootstrap(&config, sources, profile, log).await?;
            let epochs = profile.cycles.div_ceil(profile.restart_every);
            for epoch in 0..epochs {
                let service = Service::open(
                    config.clone(),
                    Credentials {
                        inference_token: "i".repeat(64),
                        admin_token: "a".repeat(64),
                    },
                )?;
                let guard = Shutdown(service.clone());
                require(
                    service.scheduler.max_concurrency() == concurrency,
                    "host cannot supply requested native concurrency",
                )?;
                let app = router(service.clone());
                let monitor = metrics::Monitor::start(
                    service.clone(),
                    profile.clone(),
                    log.scoped(json!({"epoch":epoch})),
                )?;
                // Every measured epoch starts from restored pages. Their capacity
                // can legitimately differ from newly created history pages.
                require(
                    service.loaded()?.as_array().is_some_and(Vec::is_empty),
                    "restart unexpectedly retained weights",
                )?;
                workload::replay(&app, &receipt, profile).await?;
                workload::idle(&service, profile).await?;
                if epoch == 0 {
                    history_baseline = Some(service.budget.stats().reserved);
                    log.write(
                        json!({"kind":"history_baseline","reserved_bytes":history_baseline}),
                    )?;
                } else {
                    workload::quiescent(&service, history_baseline.unwrap(), profile).await?;
                }
                for id in &ids {
                    service.load((*id).into(), 512).await?;
                }
                let resident_baseline = service.budget.stats().reserved;
                if let Some(expected) = loaded_baseline {
                    require(
                        resident_baseline == expected,
                        "resident reservation changed across restart",
                    )?;
                } else {
                    loaded_baseline = Some(resident_baseline);
                    log.write(
                        json!({"kind":"resident_baseline","reserved_bytes":resident_baseline}),
                    )?;
                }
                if epoch == 0 {
                    expected = Some(
                        workload::mixed(&app, sources, execution_profile, profile, log).await?,
                    );
                    workload::quiescent(&service, resident_baseline, profile).await?;
                    for warmup in 0..profile.warmup_cycles {
                        let warmup_log =
                            log.scoped(json!({"phase":"warmup","cycle":warmup,"epoch":epoch}));
                        workload::cycle(
                            &service,
                            &app,
                            &warmup_log,
                            &workload::Cycle {
                                sources,
                                execution_profile,
                                profile,
                                expected: expected.as_ref().unwrap(),
                                resident_baseline,
                                history_baseline: history_baseline.unwrap(),
                            },
                            true,
                        )
                        .await?;
                    }
                    rss_baseline = metrics::resident_memory();
                    require(
                        !profile.require_rss || rss_baseline.is_some(),
                        "required process RSS is unavailable",
                    )?;
                    log.write(json!({"kind":"warmup_complete","concurrency":concurrency,"profile":execution_profile,"rss_baseline_bytes":rss_baseline}))?;
                }
                for cycle in epoch * profile.restart_every
                    ..((epoch + 1) * profile.restart_every).min(profile.cycles)
                {
                    let cycle_log =
                        log.scoped(json!({"phase":"measured","cycle":cycle,"epoch":epoch}));
                    workload::cycle(
                        &service,
                        &app,
                        &cycle_log,
                        &workload::Cycle {
                            sources,
                            execution_profile,
                            profile,
                            expected: expected.as_ref().unwrap(),
                            resident_baseline,
                            history_baseline: history_baseline.unwrap(),
                        },
                        cycle % profile.fault_every == 0,
                    )
                    .await?;
                    monitor.check()?;
                    let rss = metrics::resident_memory();
                    require(
                        !profile.require_rss || rss.is_some(),
                        "required drained process RSS is unavailable",
                    )?;
                    require(
                        rss.is_none_or(|n| n <= profile.max_rss_bytes),
                        "drained RSS exceeds frozen ceiling",
                    )?;
                    if let (Some(base), Some(now)) = (rss_baseline, rss) {
                        require(
                            now <= base.saturating_add(profile.max_drained_rss_growth_bytes),
                            "drained RSS growth exceeds frozen limit",
                        )?;
                    }
                    log.write(json!({"kind":"cycle_complete","cycle":cycle,"epoch":epoch,"concurrency":concurrency,"profile":execution_profile,"rss_bytes":rss,"memory":service.budget.stats()}))?;
                    if (cycle + 1) % 10 == 0 {
                        eprintln!(
                            "completed {concurrency}/{execution_profile} cycle {}",
                            cycle + 1
                        );
                    }
                }
                counters = serde_json::to_value(service.scheduler.counters.snapshot())?;
                for id in &ids {
                    service.unload(id)?;
                }
                workload::quiescent(&service, history_baseline.unwrap(), profile).await?;
                let observed = monitor.finish()?;
                log.write(json!({"kind":"epoch_complete","epoch":epoch,"concurrency":concurrency,"profile":execution_profile,"observations":observed,"scheduler":counters}))?;
                let budget = service.budget.clone();
                service.shutdown.cancel();
                tokio::time::timeout(
                    Duration::from_secs(profile.cleanup_timeout_seconds),
                    service.scheduler.drained(),
                )
                .await?;
                drop(app);
                drop(guard);
                drop(service);
                require(
                    budget.stats().reserved == 0
                        && budget.stats().session_buffer_capacity_bytes == 0,
                    "service teardown retained reservations or engine buffers",
                )?;
                log.write(json!({"kind":"service_closed","epoch":epoch,"memory":budget.stats()}))?;
            }
            results.push(json!({"concurrency":concurrency,"execution_profile":execution_profile,"cycles":profile.cycles,"epochs":epochs,"last_epoch_scheduler":counters,"rss_baseline_bytes":rss_baseline}));
            store.close()?;
        }
    }
    Ok(json!({"configurations":results}))
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    require(
        args.len() == 4,
        "usage: cpu_soak PROFILE.json SOURCES.json OUTPUT_DIRECTORY SUMMARY.json",
    )?;
    let profile: Profile = serde_json::from_value(read_json(Path::new(&args[0]))?)?;
    profile.validate()?;
    let sources: models::Sources = serde_json::from_value(read_json(Path::new(&args[1]))?)?;
    sources.validate()?;
    let directory = Path::new(&args[2]);
    std::fs::create_dir(directory)?;
    let log = metrics::Log::new(&directory.join("observations.jsonl"))?;
    log.write(json!({"kind":"contract","profile":profile,"sources":sources,"pid":std::process::id(),"hardware":coheryn_api::hardware::probe()}))?;
    let result = execute(&profile, &sources, directory, &log).await;
    let report = json!({"schema_version":1,"kind":"cpu-mixed-soak-runtime","outcome":if result.is_ok(){"pass"}else{"fail"},"profile":profile,"sources":sources,"results":result.as_ref().ok(),"error":result.as_ref().err().map(ToString::to_string),"os":std::env::consts::OS,"architecture":std::env::consts::ARCH,"private_memory_bytes":null,"private_memory_reason":"Not available through the portable process probe; RSS and virtual memory are separate sampled observations.","scope":"In-process authenticated HTTP bodies and actual native engine/scheduler. Repetition checks detect state contamination; they are not an independent numerical oracle, TCP transport test or full production qualification."});
    serde_json::to_writer_pretty(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(Path::new(&args[3]))?,
        &report,
    )?;
    log.flush()?;
    result.map(|_| ())
}

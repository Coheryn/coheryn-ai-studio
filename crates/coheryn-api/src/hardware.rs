//! Local, read-only probes. Missing platform counters stay null, never synthetic zeroes.
use coheryn_engine::planner::HardwareSnapshot;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn probe() -> HardwareSnapshot {
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    let total = system.total_memory();
    let mut features = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        for (name, supported) in [
            ("sse2", std::is_x86_feature_detected!("sse2")),
            ("avx2", std::is_x86_feature_detected!("avx2")),
            ("fma", std::is_x86_feature_detected!("fma")),
            ("f16c", std::is_x86_feature_detected!("f16c")),
        ] {
            if supported {
                features.push(name.to_owned());
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        for (name, supported) in [
            ("neon", std::arch::is_aarch64_feature_detected!("neon")),
            ("fp16", std::arch::is_aarch64_feature_detected!("fp16")),
        ] {
            if supported {
                features.push(name.to_owned());
            }
        }
    }
    HardwareSnapshot {
        os: std::env::consts::OS.into(),
        architecture: std::env::consts::ARCH.into(),
        logical_cpus: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        cpu_features: features,
        ram_total_bytes: (total > 0).then_some(total),
        // Some sandbox/platform probes return zero when unavailable. Preserve uncertainty.
        ram_available_bytes: (system.available_memory() > 0).then_some(system.available_memory()),
        swap_total_bytes: (total > 0).then_some(system.total_swap()),
        swap_used_bytes: (total > 0).then_some(system.used_swap()),
        observed_unix_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis().min(u64::MAX as u128) as u64)
            .unwrap_or(0),
        gpu_available_bytes: None,
        gpu_backend_qualified: false,
    }
}

//! Local, read-only probes. Missing platform counters stay null, never synthetic zeroes.
use coheryn_engine::planner::HardwareSnapshot;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn probe() -> HardwareSnapshot {
    let counters = sysinfo::System::new().ok().map(|mut system| {
        system.refresh_memory();
        (
            system.total_memory(),
            system.available_memory(),
            system.total_swap(),
            system.used_swap(),
        )
    });
    let (total, available, swap_total, swap_used) = counters.unwrap_or_default();
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
        ram_available_bytes: (available > 0).then_some(available),
        swap_total_bytes: (total > 0).then_some(swap_total),
        swap_used_bytes: (total > 0).then_some(swap_used),
        observed_unix_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis().min(u64::MAX as u128) as u64)
            .unwrap_or(0),
        gpu_available_bytes: None,
        gpu_backend_qualified: false,
    }
}

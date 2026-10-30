//! Shared Qwen3.5 cost policy and atomic session admission.
use super::{Config, RESIDENT_METADATA_BYTES};
use crate::{
    Error, Result,
    catalog::PlanRequest,
    memory::{Budget, MemoryStats, Reservation},
    planner::{self, ExecutionPlan, HardwareSnapshot},
};

pub fn estimate(
    config: &Config,
    weight_bytes: u64,
    request: &PlanRequest,
    hardware: &HardwareSnapshot,
    budget: MemoryStats,
    loaded: bool,
) -> Result<ExecutionPlan> {
    estimate_shared(
        config,
        weight_bytes,
        request,
        hardware,
        budget,
        loaded,
        request.concurrency,
    )
}

pub fn estimate_shared(
    config: &Config,
    weight_bytes: u64,
    request: &PlanRequest,
    hardware: &HardwareSnapshot,
    budget: MemoryStats,
    loaded: bool,
    max_concurrency: usize,
) -> Result<ExecutionPlan> {
    if request.constraints {
        return Err(Error::new(
            crate::ErrorCode::UnsupportedFeature,
            "Qwen3.5 text constraints are not implemented",
        ));
    }
    if request.max_memory != budget.limit {
        return Err(Error::request(
            "plan budget differs from the model/request limit",
        ));
    }
    let threads = planner::execution_threads(
        request.profile,
        request.threads,
        hardware.logical_cpus,
        max_concurrency,
    )?;
    let session = config.session_estimate(request.context, threads)?;
    let mut plan = planner::estimate_prepared(planner::PreparedPlan {
        model: "local",
        profile: request.profile,
        context: request.context,
        concurrency: request.concurrency,
        max_concurrency,
        budget,
        hardware: hardware.clone(),
        images: None,
        constraints: false,
        shared_weight_bytes: weight_bytes,
        metadata: if loaded { 0 } else { RESIDENT_METADATA_BYTES },
        session,
        threads_per_session: threads,
        prefill_chunk_tokens: 1,
        io_buffer_bytes: 256 << 10,
    })?;
    plan.limitations = vec![
        "Development CPU greedy text estimate; image workloads require the image planner; optional hybrid idle caches are separately reserved",
        "Workspace includes bounded recurrent/convolution state, scratch, and worker allowances",
        "Resident metadata/tokenizer bytes are an estimate, not observed library allocation/RSS",
        "Source weight bytes include all components as an advisory file-cache upper bound; buffered weights are not copied",
        "Concurrency projects independently admitted states; continuous text batching shares weight tiles without sharing request state",
        "Plans hold no reservation; execution must atomically reserve the full candidate request including profile headroom",
    ];
    Ok(plan)
}

pub(super) struct Admission {
    pub(super) reservation: Reservation,
    pub(super) threads: usize,
}
impl Admission {
    pub(super) fn reserve(
        budget: &Budget,
        minimum: u64,
        planned: u64,
        threads: usize,
    ) -> Result<Self> {
        crate::memory::worker_bytes(threads)?;
        if planned < minimum {
            return Err(Error::resource(
                "generation plan under-reserves session/request buffers",
            ));
        }
        Ok(Self {
            reservation: budget.reserve(planned)?,
            threads,
        })
    }
    pub(super) fn check(&self, budget: &Budget, minimum: u64) -> Result<()> {
        if self.reservation.covers(budget, minimum) {
            Ok(())
        } else {
            Err(Error::resource(
                "generation reservation belongs to another budget or is too small",
            ))
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    #[test]
    fn racing_admissions_hold_headroom_and_release_only_after_work_retires() {
        let budget = Budget::new(200);
        let start = Arc::new(Barrier::new(3));
        let finish = Arc::new(Barrier::new(3));
        let handles = (0..2)
            .map(|_| {
                let b = budget.clone();
                let start = start.clone();
                let finish = finish.clone();
                std::thread::spawn(move || {
                    start.wait();
                    let guard = Admission::reserve(&b, 100, 120, 1);
                    finish.wait();
                    let success = guard.is_ok();
                    drop(guard);
                    success
                })
            })
            .collect::<Vec<_>>();
        start.wait();
        finish.wait();
        let admitted = handles
            .into_iter()
            .map(|h| usize::from(h.join().unwrap()))
            .sum::<usize>();
        assert_eq!(admitted, 1);
        assert_eq!(budget.stats().reserved, 0);
        assert_eq!(budget.stats().peak_reserved, 120);
    }
    #[test]
    fn foreign_and_under_reserved_admissions_are_rejected_and_failed_growth_leaves_budget_unchanged()
     {
        let a = Budget::new(120);
        let b = Budget::new(120);
        assert!(Admission::reserve(&a, 100, 99, 1).is_err());
        assert_eq!(a.stats().reserved, 0);
        let admission = Admission::reserve(&a, 100, 120, 1).unwrap();
        assert!(admission.check(&b, 100).is_err());
        assert!(admission.check(&a, 121).is_err());
        assert!(Admission::reserve(&a, 1, 1, 1).is_err());
        assert_eq!(a.stats().reserved, 120);
        drop(admission);
        assert_eq!(a.stats().reserved, 0);
    }
}

#[cfg(test)]
mod cost_tests {
    use super::*;
    use crate::planner::Profile;
    #[test]
    fn loaded_and_unloaded_costs_match_and_profile_headroom_is_required() {
        let config = Config {
            hidden: 1024,
            intermediate: 3584,
            vocabulary: 248320,
            heads: 8,
            kv_heads: 2,
            head_dim: 256,
            linear_key_heads: 16,
            linear_value_heads: 16,
            key_dim: 128,
            value_dim: 128,
            conv_kernel: 4,
            epsilon: 1e-6,
            rope_base: 10000000.,
            rotary_dim: 64,
            context: 262144,
            eos: 248044,
            tied: true,
            layers: (0..24).map(|i| i % 4 == 3).collect(),
        };
        let hardware = HardwareSnapshot {
            os: "test".into(),
            architecture: "test".into(),
            logical_cpus: 8,
            cpu_features: vec![],
            ram_total_bytes: None,
            ram_available_bytes: None,
            swap_total_bytes: None,
            swap_used_bytes: None,
            observed_unix_ms: 0,
            gpu_available_bytes: None,
            gpu_backend_qualified: false,
        };
        for profile in [
            Profile::Reference,
            Profile::Balanced,
            Profile::LowMemory,
            Profile::Performance,
        ] {
            let mut request = PlanRequest {
                context: 256,
                concurrency: 1,
                max_memory: 1 << 30,
                profile,
                threads: None,
                constraints: false,
            };
            let cold = estimate(
                &config,
                1 << 30,
                &request,
                &hardware,
                MemoryStats {
                    limit: request.max_memory,
                    ..Default::default()
                },
                false,
            )
            .unwrap();
            let hot = estimate(
                &config,
                1 << 30,
                &request,
                &hardware,
                MemoryStats {
                    reserved: RESIDENT_METADATA_BYTES,
                    limit: request.max_memory,
                    ..Default::default()
                },
                true,
            )
            .unwrap();
            assert_eq!(cold.required_reserved_bytes, hot.required_reserved_bytes);
            assert_eq!(hot.model_metadata_additional, 0);
            assert_eq!(cold.model_metadata_additional, RESIDENT_METADATA_BYTES);
            assert!(hot.admissible);
            assert_eq!(
                cold.safety_headroom_bytes == 0,
                profile == Profile::Reference
            );
            request.max_memory = cold.required_reserved_bytes - 1;
            let rejected = estimate(
                &config,
                1 << 30,
                &request,
                &hardware,
                MemoryStats {
                    reserved: RESIDENT_METADATA_BYTES,
                    limit: request.max_memory,
                    ..Default::default()
                },
                true,
            )
            .unwrap();
            assert!(!rejected.admissible);
        }
        let mut hardware = hardware;
        hardware.ram_available_bytes = Some(1);
        let request = PlanRequest {
            context: 256,
            concurrency: 1,
            max_memory: 1 << 30,
            profile: Profile::Reference,
            threads: Some(1),
            constraints: false,
        };
        let rejected = estimate(
            &config,
            1 << 30,
            &request,
            &hardware,
            MemoryStats {
                reserved: RESIDENT_METADATA_BYTES,
                limit: request.max_memory,
                ..Default::default()
            },
            true,
        )
        .unwrap();
        assert!(!rejected.admissible);
        assert_eq!(rejected.host_allocation_feasible, Some(false));
    }
}

#[cfg(test)]
mod model_tests {
    use super::*;
    use crate::{
        Cancellation, GenerationOptions, ResourceLimits, adapters::Execution, planner::Profile,
        tokenizer::Message,
    };
    #[test]
    #[ignore = "requires the separately acquired pinned Qwen3.5 checkpoint"]
    fn profiled_admission_cleans_up_after_consumer_failure_cancel_and_model_drop() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let path = root.join(".cache/qwen3.5-0.8b-source");
        assert_eq!(
            crate::storage::sha256_file(path.join("model.safetensors-00001-of-00001.safetensors"))
                .unwrap(),
            "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
        );
        let model = super::super::Model::load(
            &path,
            ResourceLimits {
                context: 64,
                max_memory: 768 << 20,
            },
            &Cancellation::default(),
        )
        .unwrap();
        let long_prompt = "hello ".repeat(128);
        assert!(model.count_tokens(&long_prompt).unwrap() > model.context_limit());
        assert!(model.encode(&long_prompt).is_err());
        let budget = model.budget.clone();
        let baseline = budget.stats().reserved;
        let prompt = model
            .render(&[Message {
                role: "user".into(),
                content: "Say hello.".into(),
                ..Default::default()
            }])
            .unwrap();
        let cpus = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        let threads = cpus.min(2);
        let hardware = HardwareSnapshot {
            os: std::env::consts::OS.into(),
            architecture: std::env::consts::ARCH.into(),
            logical_cpus: cpus,
            cpu_features: vec![],
            ram_total_bytes: None,
            ram_available_bytes: None,
            swap_total_bytes: None,
            swap_used_bytes: None,
            observed_unix_ms: 0,
            gpu_available_bytes: None,
            gpu_backend_qualified: false,
        };
        let options = GenerationOptions {
            max_tokens: 4,
            ..Default::default()
        };
        let error = model
            .generate_profiled(
                &prompt,
                &options,
                Execution {
                    profile: Profile::Balanced,
                    threads: Some(threads),
                    hardware: &hardware,
                },
                &Cancellation::default(),
                &mut |_| Err(Error::request("consumer failed")),
            )
            .unwrap_err();
        assert_eq!(error.code, crate::ErrorCode::InvalidRequest);
        assert_eq!(budget.stats().reserved, baseline);
        assert_eq!(budget.stats().session_buffer_capacity_bytes, 0);
        let request = PlanRequest {
            context: 64,
            concurrency: 1,
            max_memory: 768 << 20,
            profile: Profile::Balanced,
            threads: Some(threads),
            constraints: false,
        };
        let plan = estimate(
            model.configuration(),
            model.set.tensor_bytes().unwrap(),
            &request,
            &hardware,
            budget.stats(),
            true,
        )
        .unwrap();
        assert_eq!(budget.stats().peak_reserved, plan.required_reserved_bytes);
        assert!(plan.safety_headroom_bytes > 0);
        let cancel = Cancellation::default();
        let error = model
            .generate_profiled(
                &prompt,
                &options,
                Execution {
                    profile: Profile::Balanced,
                    threads: Some(threads),
                    hardware: &hardware,
                },
                &cancel,
                &mut |_| {
                    cancel.cancel();
                    Ok(())
                },
            )
            .unwrap_err();
        assert_eq!(error.code, crate::ErrorCode::Cancelled);
        assert_eq!(budget.stats().reserved, baseline);
        assert_eq!(budget.stats().session_buffer_capacity_bytes, 0);
        let other = super::super::Model::load_with_budget(
            &path,
            ResourceLimits {
                context: 64,
                max_memory: 768 << 20,
            },
            budget.clone(),
            &Default::default(),
        )
        .unwrap();
        let prepared = model
            .prepare_generation(
                options.clone(),
                64,
                Execution {
                    profile: Profile::Reference,
                    threads: Some(1),
                    hardware: &hardware,
                },
                1,
            )
            .unwrap();
        let error = other
            .generate_prepared(
                &prompt,
                prepared,
                &Default::default(),
                &mut || panic!("foreign model must fail before compute"),
                &mut |_| panic!("foreign model must not emit"),
            )
            .unwrap_err();
        assert_eq!(error.code, crate::ErrorCode::InvalidRequest);
        drop(other);
        assert_eq!(budget.stats().reserved, baseline);
        let prepared = model
            .prepare_generation(
                options.clone(),
                64,
                Execution {
                    profile: Profile::Reference,
                    threads: Some(1),
                    hardware: &hardware,
                },
                1,
            )
            .unwrap();
        let error = model
            .generate_prepared(
                &prompt,
                prepared,
                &Default::default(),
                &mut || {
                    Err(Error::new(
                        crate::ErrorCode::Cancelled,
                        "cooperative test cancellation",
                    ))
                },
                &mut |_| panic!("cancelled graph must not emit"),
            )
            .unwrap_err();
        assert_eq!(error.code, crate::ErrorCode::Cancelled);
        assert_eq!(budget.stats().reserved, baseline);
        assert_eq!(budget.stats().session_buffer_capacity_bytes, 0);
        drop(model);
        assert_eq!(budget.stats().reserved, 0);
    }
}

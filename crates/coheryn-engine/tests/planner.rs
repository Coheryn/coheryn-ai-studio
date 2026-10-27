mod common;
use coheryn_engine::{
    Cancellation, ErrorCode, GenerationOptions, Model, ResourceLimits, gguf,
    memory::{Budget, MemoryStats},
    planner::{self, *},
};
use std::{
    io::Cursor,
    sync::{Arc, Barrier},
};

fn hardware() -> HardwareSnapshot {
    HardwareSnapshot {
        os: "fixture".into(),
        architecture: "fixture".into(),
        logical_cpus: 8,
        cpu_features: vec![],
        ram_total_bytes: Some(8 << 30),
        ram_available_bytes: Some(4 << 30),
        swap_total_bytes: Some(0),
        swap_used_bytes: Some(0),
        observed_unix_ms: 0,
        gpu_available_bytes: None,
        gpu_backend_qualified: false,
    }
}
fn with_input(f: impl FnOnce(PlanInput<'_>)) {
    let bytes = common::tiny_model();
    let d = gguf::parse(Cursor::new(&bytes), bytes.len() as u64, Default::default()).unwrap();
    let c = coheryn_engine::validate(&d).config.unwrap();
    f(PlanInput {
        descriptor: &d,
        config: &c,
        model: "tiny",
        profile: Profile::Balanced,
        context: 64,
        concurrency: 1,
        max_concurrency: 4,
        loaded: false,
        budget: MemoryStats {
            limit: 128 << 20,
            ..Default::default()
        },
        hardware: hardware(),
        images: None,
        threads: None,
    });
}
#[test]
fn plans_are_deterministic_and_do_not_mutate_context_or_backend() {
    with_input(|mut input| {
        let a = planner::estimate(PlanInput {
            hardware: hardware(),
            ..input
        })
        .unwrap();
        let b = planner::estimate(PlanInput {
            hardware: hardware(),
            ..input
        })
        .unwrap();
        assert_eq!(
            serde_json::to_value(&a).unwrap(),
            serde_json::to_value(&b).unwrap()
        );
        assert!(a.admissible);
        assert_eq!(a.session.kv_capacity_bytes, 1024);
        assert_eq!(a.context, 64);
        assert_eq!(a.cache_dtype, "f32");
        assert_eq!(a.backend, "cpu-reference");
        input.profile = Profile::LowMemory;
        input.concurrency = 2;
        let low = planner::estimate(input).unwrap();
        assert!(!low.admissible);
        assert_eq!(low.context, 64);
        assert_eq!(low.concurrency, 2);
        assert!(low.reasons.iter().any(|r| r.contains("limit 1")));
    });
}
#[test]
fn concurrency_shares_weights_and_metadata_but_duplicates_request_capacity() {
    with_input(|input| {
        let one = planner::estimate(PlanInput {
            hardware: hardware(),
            ..input
        })
        .unwrap();
        let two = planner::estimate(PlanInput {
            hardware: hardware(),
            concurrency: 2,
            ..input
        })
        .unwrap();
        assert_eq!(two.shared_weight_bytes, one.shared_weight_bytes);
        assert_eq!(two.model_metadata_additional, one.model_metadata_additional);
        assert_eq!(
            two.additional_reserved_bytes - one.model_metadata_additional,
            2 * (one.additional_reserved_bytes - one.model_metadata_additional)
        );
        assert_eq!(two.mapped_weight_bytes, 0);
        assert_eq!(two.gpu_allocation_bytes, 0);
    });
}
#[test]
fn residency_budget_host_pressure_and_unknown_vision_are_explicit() {
    with_input(|input| {
        let loaded = planner::estimate(PlanInput {
            loaded: true,
            hardware: hardware(),
            ..input
        })
        .unwrap();
        assert_eq!(loaded.model_metadata_additional, 0);
        let low_budget = planner::estimate(PlanInput {
            budget: MemoryStats {
                limit: 1,
                ..Default::default()
            },
            hardware: hardware(),
            ..input
        })
        .unwrap();
        assert!(!low_budget.admissible);
        let mut host = hardware();
        host.ram_available_bytes = Some(1);
        assert!(
            !planner::estimate(PlanInput {
                hardware: host,
                ..input
            })
            .unwrap()
            .admissible
        );
        let mut unknown = hardware();
        unknown.ram_available_bytes = None;
        assert_eq!(
            planner::estimate(PlanInput {
                hardware: unknown,
                ..input
            })
            .unwrap()
            .host_memory_feasible,
            None
        );
        let images = ImageWorkload {
            images: vec![ImageDescriptor {
                width: 336,
                height: 336,
            }],
            processor_profile: "unqualified".into(),
        };
        let vision = planner::estimate(PlanInput {
            hardware: hardware(),
            images: Some(&images),
            ..input
        })
        .unwrap();
        assert!(!vision.admissible);
        let stages = vision.image_stages.unwrap();
        assert!(!stages.qualified);
        assert_eq!(stages.vision_workspace_bytes, None);
        assert_eq!(stages.effective_image_positions, None);
    });
}
#[test]
fn overflow_and_invalid_planning_inputs_are_recoverable() {
    with_input(|input| {
        assert!(
            planner::estimate(PlanInput {
                hardware: hardware(),
                context: 129,
                ..input
            })
            .is_err()
        );
        assert!(
            planner::estimate(PlanInput {
                hardware: hardware(),
                concurrency: 0,
                ..input
            })
            .is_err()
        );
        assert!(
            planner::estimate(PlanInput {
                hardware: hardware(),
                budget: MemoryStats {
                    reserved: u64::MAX,
                    limit: u64::MAX,
                    ..Default::default()
                },
                ..input
            })
            .is_err()
        );
        let mut oversized = input.config.clone();
        oversized.intermediate = usize::MAX;
        assert!(planner::session_estimate(&oversized, 64).is_err());
    });
    assert!(Profile::parse("auto-magic").is_err());
    assert!(
        ImageWorkload {
            images: vec![ImageDescriptor {
                width: 0,
                height: 1
            }],
            processor_profile: "x".into()
        }
        .validate()
        .is_err()
    );
}
#[test]
fn concurrent_reservations_never_both_admit_and_splitting_has_no_release_gap() {
    let budget = Budget::new(100);
    let begin = Arc::new(Barrier::new(3));
    let held = Arc::new(Barrier::new(3));
    let jobs: Vec<_> = (0..2)
        .map(|_| {
            let budget = budget.clone();
            let begin = begin.clone();
            let held = held.clone();
            std::thread::spawn(move || {
                begin.wait();
                let reservation = budget.reserve(60);
                let accepted = reservation.is_ok();
                held.wait();
                drop(reservation);
                accepted
            })
        })
        .collect();
    begin.wait();
    held.wait();
    assert_eq!(
        jobs.into_iter()
            .map(|j| usize::from(j.join().unwrap()))
            .sum::<usize>(),
        1
    );
    let mut whole = budget.reserve(100).unwrap();
    let session = whole.split(60).unwrap();
    assert_eq!(budget.stats().reserved, 100);
    assert!(budget.reserve(1).is_err());
    drop(session);
    assert_eq!(budget.stats().reserved, 40);
    drop(whole);
    assert_eq!(budget.stats().reserved, 0);
}
#[test]
fn session_handoff_uses_one_reservation_and_observed_buffers_are_released() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    std::fs::write(&path, common::tiny_model()).unwrap();
    let model = Model::load(
        path,
        ResourceLimits {
            context: 64,
            max_memory: 32 << 20,
        },
    )
    .unwrap();
    let baseline = model.memory().reserved;
    let admission = model.reserve_session(64, REQUEST_BYTES).unwrap();
    let held = model.memory().reserved;
    let mut observed = false;
    model
        .generate_admitted(
            "Hi",
            &GenerationOptions {
                max_tokens: 4,
                ..Default::default()
            },
            admission,
            &Cancellation::default(),
            |_| {
                let stats = model.memory();
                assert_eq!(stats.reserved, held);
                assert!(stats.session_buffer_capacity_bytes > 0);
                observed = true;
                Ok(())
            },
        )
        .unwrap();
    assert!(observed);
    assert_eq!(model.memory().reserved, baseline);
    assert_eq!(model.memory().session_buffer_capacity_bytes, 0);
    assert!(model.memory().peak_session_buffer_capacity_bytes > 0);
    let foreign = Budget::new(32 << 20).reserve(16 << 20).unwrap();
    let error = model
        .generate_admitted(
            "Hi",
            &GenerationOptions {
                max_tokens: 1,
                ..Default::default()
            },
            coheryn_engine::inference::SessionAdmission {
                context: 64,
                reservation: foreign,
                io_bytes: 256 << 10,
                execution: Default::default(),
            },
            &Cancellation::default(),
            |_| Ok(()),
        )
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::ResourceExhausted);
    assert_eq!(model.memory().reserved, baseline);
}

#[test]
fn smaller_io_tiles_preserve_tokens_and_reduce_observed_workspace_capacity() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    std::fs::write(&path, common::tiny_model()).unwrap();
    let model = Model::load(
        path,
        ResourceLimits {
            context: 64,
            max_memory: 32 << 20,
        },
    )
    .unwrap();
    let mut reference = None;
    let mut capacities = vec![];
    for io_bytes in [16 << 10, 256 << 10] {
        let mut capacity = 0;
        let admission = model.reserve_session_with_io(64, 0, io_bytes).unwrap();
        let result = model
            .generate_admitted(
                "Hi",
                &GenerationOptions {
                    max_tokens: 4,
                    ..Default::default()
                },
                admission,
                &Cancellation::default(),
                |_| {
                    capacity = capacity.max(model.memory().session_buffer_capacity_bytes);
                    Ok(())
                },
            )
            .unwrap();
        if let Some(ids) = &reference {
            assert_eq!(&result.token_ids, ids);
        } else {
            reference = Some(result.token_ids);
        }
        capacities.push(capacity);
        assert_eq!(model.memory().session_buffer_capacity_bytes, 0);
    }
    assert_eq!(capacities[1] - capacities[0], 3 * ((256 - 16) << 10));
}

#[test]
fn buffered_weights_do_not_require_full_file_cache_residency_for_admission() {
    with_input(|input| {
        let baseline = planner::estimate(PlanInput {
            hardware: hardware(),
            ..input
        })
        .unwrap();
        let mut host = hardware();
        host.ram_available_bytes =
            Some(baseline.additional_reserved_bytes + baseline.shared_weight_bytes / 2);
        let plan = planner::estimate(PlanInput {
            hardware: host,
            ..input
        })
        .unwrap();
        assert_eq!(plan.host_memory_feasible, Some(false));
        assert_eq!(plan.host_allocation_feasible, Some(true));
        assert!(plan.admissible);
        assert!(!plan.warnings.is_empty());
    });
}

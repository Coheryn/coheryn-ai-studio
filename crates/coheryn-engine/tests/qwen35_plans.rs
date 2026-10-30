use coheryn_engine::{
    catalog::PlanRequest,
    memory::{Budget, MemoryStats},
    multimodal::image::{self, ImageLimits, Processor},
    planner::{HardwareSnapshot, ImageDescriptor, ImageWorkload, Profile},
    qwen35::{
        self, Config,
        image_plan::{dimension_stages, estimate_images, image_stages},
        vision::VisionConfig,
    },
};
fn config() -> Config {
    Config {
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
        rope_base: 1e7,
        rotary_dim: 64,
        context: 262144,
        eos: 248044,
        tied: true,
        layers: (0..24).map(|i| i % 4 == 3).collect(),
    }
}
fn vision() -> VisionConfig {
    VisionConfig {
        hidden: 768,
        intermediate: 3072,
        heads: 12,
        layers: 12,
        output: 1024,
        positions: 2304,
    }
}
fn hardware() -> HardwareSnapshot {
    HardwareSnapshot {
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
    }
}
fn workload() -> ImageWorkload {
    ImageWorkload {
        processor_profile: Processor::Qwen35Still.name().into(),
        images: vec![
            ImageDescriptor {
                width: 32,
                height: 32,
            },
            ImageDescriptor {
                width: 32,
                height: 64,
            },
        ],
    }
}
fn request() -> PlanRequest {
    PlanRequest {
        context: 256,
        concurrency: 1,
        max_memory: 2 << 30,
        profile: Profile::Reference,
        threads: Some(1),
        constraints: false,
    }
}
#[test]
fn dimension_bound_covers_exact_inputs_without_constructing_execution_state() {
    let budget = Budget::new(1 << 30);
    let before = budget.stats();
    let limits = ImageLimits::default();
    let v = vision();
    let dim = dimension_stages(&v, &workload(), limits, 1).unwrap();
    let plans = [
        include_bytes!("../../../fixtures/qwen35/vision/red.png").as_slice(),
        include_bytes!("../../../fixtures/qwen35/vision/blue.png").as_slice(),
    ]
    .iter()
    .map(|bytes| {
        image::plan(
            image::inspect(bytes, limits, &Default::default()).unwrap(),
            Processor::Qwen35Still,
            limits,
        )
        .unwrap()
    })
    .collect::<Vec<_>>();
    let exact = image_stages(&v, &plans, 1, limits).unwrap();
    assert_eq!(dim.image_feature_tokens, 64 + 72);
    assert_eq!(exact.image_feature_tokens, dim.image_feature_tokens);
    assert!(dim.extra_reserved_bytes > exact.extra_reserved_bytes);
    // Encoded-byte decoder allowances and conservative RGBA descriptors also grow.
    assert!(
        dim.extra_reserved_bytes - exact.extra_reserved_bytes
            >= dim.encoded_bytes_upper_bound - exact.encoded_bytes_upper_bound
    );
    for (a, b) in dim.images.iter().zip(&exact.images) {
        assert_eq!(a.encoder_reserved_bytes, b.encoder_reserved_bytes);
        assert_eq!(a.feature_embedding_bytes, b.feature_embedding_bytes);
    }
    assert_eq!(budget.stats().reserved, before.reserved);
    assert_eq!(budget.stats().session_buffer_capacity_bytes, 0);
}
#[test]
fn loaded_unloaded_concurrency_and_transport_costs_share_weights_once() {
    let c = config();
    let v = vision();
    let mut r = request();
    let h = hardware();
    let limits = ImageLimits::default();
    let unloaded = estimate_images(
        &c,
        &v,
        1 << 30,
        &r,
        &h,
        MemoryStats {
            limit: r.max_memory,
            ..Default::default()
        },
        false,
        2,
        &workload(),
        limits,
    )
    .unwrap();
    let mut loaded = estimate_images(
        &c,
        &v,
        1 << 30,
        &r,
        &h,
        MemoryStats {
            limit: r.max_memory,
            reserved: qwen35::RESIDENT_METADATA_BYTES,
            ..Default::default()
        },
        true,
        2,
        &workload(),
        limits,
    )
    .unwrap();
    assert_eq!(
        loaded.execution.required_reserved_bytes,
        unloaded.execution.required_reserved_bytes
    );
    assert_eq!(loaded.execution.model_metadata_additional, 0);
    assert!(loaded.execution.admissible);
    r.concurrency = 2;
    let two = estimate_images(
        &c,
        &v,
        1 << 30,
        &r,
        &h,
        MemoryStats {
            limit: r.max_memory,
            ..Default::default()
        },
        false,
        2,
        &workload(),
        limits,
    )
    .unwrap();
    assert_eq!(
        two.execution.additional_reserved_bytes - qwen35::RESIDENT_METADATA_BYTES,
        2 * (unloaded.execution.additional_reserved_bytes - qwen35::RESIDENT_METADATA_BYTES)
    );
    assert_eq!(
        two.execution.shared_weight_bytes,
        unloaded.execution.shared_weight_bytes
    );
    let previous = loaded.execution.required_reserved_bytes;
    loaded.add_capacity(8 << 20).unwrap();
    assert_eq!(
        loaded.execution.required_reserved_bytes,
        previous + (8 << 20)
    );
}
#[test]
fn invalid_profiles_limits_context_and_budget_are_explicit() {
    let c = config();
    let v = vision();
    let mut r = request();
    let mut w = workload();
    let mut h = hardware();
    let limits = ImageLimits::default();
    w.processor_profile = "qwen2vl-still-v1".into();
    assert!(dimension_stages(&v, &w, limits, 1).is_err());
    w = workload();
    w.images.clear();
    assert!(dimension_stages(&v, &w, limits, 1).is_err());
    r.context = 128;
    let p = estimate_images(
        &c,
        &v,
        1 << 30,
        &r,
        &h,
        MemoryStats {
            limit: r.max_memory,
            ..Default::default()
        },
        false,
        2,
        &workload(),
        limits,
    )
    .unwrap();
    assert!(!p.execution.admissible);
    assert!(p.execution.reasons.iter().any(|r| r.contains("features")));
    r = request();
    r.max_memory = 300 << 20;
    let p = estimate_images(
        &c,
        &v,
        1 << 30,
        &r,
        &h,
        MemoryStats {
            limit: r.max_memory,
            ..Default::default()
        },
        false,
        2,
        &workload(),
        limits,
    )
    .unwrap();
    assert!(!p.execution.admissible);
    r = request();
    h.ram_available_bytes = Some(1);
    let p = estimate_images(
        &c,
        &v,
        1 << 30,
        &r,
        &h,
        MemoryStats {
            limit: r.max_memory,
            ..Default::default()
        },
        false,
        2,
        &workload(),
        limits,
    )
    .unwrap();
    assert!(!p.execution.admissible);
    r.constraints = true;
    assert!(
        estimate_images(
            &c,
            &v,
            1 << 30,
            &r,
            &h,
            MemoryStats {
                limit: r.max_memory,
                ..Default::default()
            },
            false,
            2,
            &workload(),
            limits
        )
        .is_err()
    );
}

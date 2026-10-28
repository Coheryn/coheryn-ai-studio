use coheryn_engine::{
    Cancellation, Error, ErrorCode, ParseLimits, gguf,
    memory::Budget,
    multimodal::{
        image::{self, Processor},
        vision::{VisionConfig, VisionModel, VisionOptions},
    },
};

#[path = "common/vision.rs"]
mod vision_fixture;
fn fixture() -> tempfile::NamedTempFile {
    vision_fixture::component(12)
}
fn prepare(budget: &Budget, profile: Processor) -> image::PreparedImage {
    image::prepare(
        include_bytes!("../../../fixtures/vlm/images/minimum.png"),
        profile,
        Default::default(),
        budget,
        &Cancellation::default(),
    )
    .unwrap()
}
#[test]
fn validates_actual_shapes_instead_of_misleading_ffn_metadata() {
    let f = fixture();
    let d = gguf::inspect(f.path(), ParseLimits::default()).unwrap();
    let c = VisionConfig::from_descriptor(&d).unwrap();
    assert_eq!(c.intermediate, 32);
    assert_eq!(c.declared_intermediate, Some(17));
    for kind in 0..7 {
        let mut d = d.clone();
        match kind {
            0 => {
                d.tensors.pop();
            }
            1 => d.tensors.push(d.tensors[0].clone()),
            2 => d.tensors[0].dtype = 0,
            3 => d.tensors[0].dimensions[0] = 16,
            4 => {
                d.metadata
                    .insert("clip.vision.window_size".into(), gguf::Value::Unsigned(112));
            }
            5 => {
                d.metadata
                    .insert("clip.use_gelu".into(), gguf::Value::Bool(true));
            }
            _ => d.byte_order = "big".into(),
        }
        assert!(VisionConfig::from_descriptor(&d).is_err(), "case {kind}");
    }
}
#[test]
fn native_graph_is_thread_and_tile_consistent_and_outputs_own_their_reservation() {
    let f = fixture();
    let budget = Budget::new(64 << 20);
    let cancel = Cancellation::default();
    let model = VisionModel::load(f.path(), budget.clone(), &cancel).unwrap();
    let image = prepare(&budget, Processor::Qwen2VlStill);
    let baseline = budget.stats().reserved;
    let a = model
        .encode(&image, &VisionOptions::default(), &cancel)
        .unwrap();
    assert_eq!(a.embeddings().len(), a.plan.output_tokens * 12);
    assert!(a.embeddings().iter().all(|v| v.is_finite()));
    assert!(a.embeddings().iter().any(|v| v.abs() > 0.01));
    let b = model
        .encode(
            &image,
            &VisionOptions {
                threads: 2,
                tile_tokens: 3,
                ..Default::default()
            },
            &cancel,
        )
        .unwrap();
    assert_eq!(a.embeddings(), b.embeddings());
    drop(b);
    assert_eq!(budget.stats().reserved, baseline + a.plan.output_bytes);
    drop(image);
    drop(model);
    assert_eq!(budget.stats().reserved, a.plan.output_bytes);
    assert!(!a.embeddings().is_empty());
    drop(a);
    assert_eq!(budget.stats().reserved, 0);
    assert_eq!(budget.stats().session_buffer_capacity_bytes, 0);
}
#[test]
fn admission_profile_budget_and_cancellation_fail_without_leaks() {
    let f = fixture();
    let budget = Budget::new(64 << 20);
    let model = VisionModel::load(f.path(), budget.clone(), &Cancellation::default()).unwrap();
    let image = prepare(&budget, Processor::Qwen2VlStill);
    let options = VisionOptions::default();
    let before = budget.stats();
    let plan = model.plan(&image, &options).unwrap();
    let held = budget
        .reserve((64 << 20) - before.reserved - plan.peak_reserved_bytes + 1)
        .unwrap();
    assert_eq!(
        model
            .encode(&image, &options, &Cancellation::default())
            .unwrap_err()
            .code,
        ErrorCode::ResourceExhausted
    );
    drop(held);
    assert_eq!(budget.stats().reserved, before.reserved);
    let other = prepare(&Budget::new(64 << 20), Processor::Qwen2VlStill);
    assert!(model.plan(&other, &options).is_err());
    let wrong = prepare(&budget, Processor::Llava15Pad336);
    assert!(model.plan(&wrong, &options).is_err());
    drop(wrong);
    assert!(
        model
            .plan(
                &image,
                &VisionOptions {
                    threads: 0,
                    ..Default::default()
                }
            )
            .is_err()
    );
    assert!(
        model
            .plan(
                &image,
                &VisionOptions {
                    max_patches: 4,
                    ..Default::default()
                }
            )
            .is_err()
    );
    for checkpoint in ["patch_embeddings", "layer_out-0"] {
        let cancel = Cancellation::default();
        let result = model.trace(&image, &options, &cancel, &mut |c| {
            if c.name == checkpoint {
                cancel.cancel();
            }
            Ok(())
        });
        assert_eq!(result.unwrap_err().code, ErrorCode::Cancelled);
        assert_eq!(budget.stats().reserved, before.reserved);
        assert_eq!(
            budget.stats().session_buffer_capacity_bytes,
            before.session_buffer_capacity_bytes
        );
    }
    let result = model.trace(&image, &options, &Cancellation::default(), &mut |_| {
        Err(Error::request("observer stopped"))
    });
    assert!(result.is_err());
    assert_eq!(budget.stats().reserved, before.reserved);
    drop(image);
    drop(model);
    assert_eq!(budget.stats().reserved, 0);
}
#[test]
fn changed_and_truncated_components_are_rejected() {
    let mut f = fixture();
    let budget = Budget::new(64 << 20);
    let cancel = Cancellation::default();
    let model = VisionModel::load(f.path(), budget.clone(), &cancel).unwrap();
    let image = prepare(&budget, Processor::Qwen2VlStill);
    f.as_file()
        .set_len(f.as_file().metadata().unwrap().len() + 1)
        .unwrap();
    assert!(
        model
            .encode(&image, &VisionOptions::default(), &cancel)
            .is_err()
    );
    drop(model);
    drop(image);
    f.as_file_mut().set_len(128).unwrap();
    assert!(VisionModel::load(f.path(), budget.clone(), &cancel).is_err());
    assert_eq!(budget.stats().reserved, 0);
    cancel.cancel();
    assert!(VisionModel::load(f.path(), budget.clone(), &cancel).is_err());
    assert_eq!(budget.stats().reserved, 0);
}

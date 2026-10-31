mod common;
use coheryn_engine::{
    Cancellation, Error, ErrorCode, ResourceLimits, Result,
    adapters::{Capabilities, Registry, TextAdapter, TextModel},
    catalog::{Format, PlanRequest, Source, StructuralStatus, TextMetadata},
    planner::{HardwareSnapshot, Profile},
};
use serde_json::json;
use std::{
    io::{Seek, SeekFrom, Write},
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
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
fn request() -> PlanRequest {
    PlanRequest {
        context: 64,
        concurrency: 1,
        max_memory: u64::MAX,
        profile: Profile::Reference,
        threads: Some(1),
        constraints: false,
    }
}
fn write_safe(path: &Path, name: &str) {
    let h = json!({name:{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}).to_string();
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&(h.len() as u64).to_le_bytes()).unwrap();
    f.write_all(h.as_bytes()).unwrap();
    f.write_all(&f32::NAN.to_le_bytes()).unwrap();
}
#[test]
fn gguf_metadata_plan_preserves_old_costs_and_never_reads_numeric_payloads() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("misleading.safetensors");
    std::fs::write(&path, common::tiny_model()).unwrap();
    let cancel = Cancellation::default();
    let before = Registry::default().inspect(&path, &cancel).unwrap();
    let identity = before.descriptor().metadata_identity.clone();
    let (gguf, compat) = before.gguf().unwrap();
    let hw = hardware();
    let req = request();
    let plan = before.plan(&req, &hw, &cancel).unwrap();
    let old = coheryn_engine::planner::estimate(coheryn_engine::planner::PlanInput {
        descriptor: gguf,
        config: compat.config.as_ref().unwrap(),
        model: "local",
        profile: req.profile,
        context: req.context,
        concurrency: 1,
        max_concurrency: 1,
        loaded: false,
        budget: coheryn_engine::memory::MemoryStats {
            limit: req.max_memory,
            ..Default::default()
        },
        hardware: hw.clone(),
        images: None,
        threads: req.threads,
        constraints: false,
    })
    .unwrap();
    assert_eq!(
        serde_json::to_value(&plan).unwrap(),
        serde_json::to_value(old).unwrap()
    );
    let offset = gguf.data_offset;
    drop(before);
    let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.seek(SeekFrom::Start(offset)).unwrap();
    file.write_all(&f32::NAN.to_le_bytes()).unwrap();
    drop(file);
    let inspected = Registry::default().inspect(&path, &cancel).unwrap();
    let d = inspected.descriptor();
    assert_eq!(d.format, Format::Gguf);
    assert!(d.executable);
    assert_eq!(d.weight_payloads, "not-scanned");
    assert_eq!(d.metadata_identity, identity);
    assert!(d.identity_scope.contains("not a model/cache identity"));
    assert!(inspected.plan(&req, &hw, &cancel).unwrap().admissible);
    assert_eq!(
        inspected.deep_validate(&cancel).unwrap_err().code,
        ErrorCode::InvalidGguf
    );
    let mut exact = req.clone();
    exact.max_memory = plan.required_reserved_bytes;
    assert!(inspected.plan(&exact, &hw, &cancel).unwrap().admissible);
    exact.max_memory -= 1;
    assert!(!inspected.plan(&exact, &hw, &cancel).unwrap().admissible);
    assert_eq!(d.admission_status, "not-evaluated");
    let mut bad = req;
    bad.context = 0;
    assert!(inspected.plan(&bad, &hw, &cancel).is_err());
    bad.context = 64;
    bad.concurrency = 0;
    assert!(inspected.plan(&bad, &hw, &cancel).is_err());
}
#[test]
fn valid_unsupported_sources_and_missing_duplicate_assets_have_precise_reports() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    write_safe(&path, "x");
    let cancel = Cancellation::default();
    let missing = Registry::default().inspect(&path, &cancel).unwrap();
    assert!(missing.descriptor().container_valid);
    assert!(matches!(
        missing.descriptor().structural_status,
        StructuralStatus::Invalid
    ));
    assert!(
        missing.descriptor().diagnostics[0]
            .message
            .contains("config.json")
    );
    std::fs::write(dir.path().join("config.json"), r#"{"model_type":"future"}"#).unwrap();
    let future = Registry::default().inspect(&path, &cancel).unwrap();
    assert_eq!(future.descriptor().tensor_bytes, 4);
    assert_eq!(future.descriptor().tensor_count, 1);
    assert!(matches!(
        future.descriptor().structural_status,
        StructuralStatus::Unsupported
    ));
    assert_eq!(
        future.require_compatible().unwrap_err().code,
        ErrorCode::UnsupportedModelArchitecture
    );
    assert_eq!(future.descriptor().source_files.len(), 2);
    assert!(
        future
            .descriptor()
            .source_files
            .iter()
            .any(|f| f.role == "asset" && f.hash_scope == "complete-file")
    );
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type":"a","model_type":"b"}"#,
    )
    .unwrap();
    let duplicate = Registry::default().inspect(&path, &cancel).unwrap();
    assert!(
        duplicate.descriptor().diagnostics[0]
            .message
            .contains("duplicate")
    );
    cancel.cancel();
    assert_eq!(
        Registry::default()
            .inspect("does-not-exist", &cancel)
            .err()
            .unwrap()
            .code,
        ErrorCode::Cancelled
    );
}
#[test]
fn indexed_metadata_records_all_shards_and_exact_index_identity() {
    let dir = tempfile::tempdir().unwrap();
    write_safe(&dir.path().join("a.safetensors"), "a");
    write_safe(&dir.path().join("b.safetensors"), "b");
    std::fs::write(dir.path().join("config.json"), r#"{"model_type":"future"}"#).unwrap();
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        r#"{"weight_map":{"a":"a.safetensors","b":"b.safetensors"}}"#,
    )
    .unwrap();
    let inspection = Registry::default()
        .inspect(dir.path(), &Cancellation::default())
        .unwrap();
    let d = inspection.descriptor();
    assert_eq!(d.layout, "indexed");
    assert_eq!(d.tensor_count, 2);
    assert_eq!(d.tensor_bytes, 8);
    assert_eq!(
        d.source_files
            .iter()
            .filter(|f| f.role == "weights")
            .count(),
        2
    );
    assert!(
        d.source_files
            .iter()
            .any(|f| f.role == "index" && f.hash_scope == "complete-file")
    );
}
#[test]
fn embedding_metadata_keeps_its_independent_task_and_context_bounds() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("encoder.gguf");
    std::fs::write(&path, common::embedding_model(8)).unwrap();
    let inspected = Registry::default()
        .inspect(&path, &Cancellation::default())
        .unwrap();
    let d = inspected.descriptor();
    assert_eq!(d.task, Some(coheryn_engine::ModelTask::Embedding));
    assert!(d.executable);
    assert!(d.text_adapter.is_none());
    assert_eq!(
        inspected
            .plan(&request(), &hardware(), &Cancellation::default())
            .unwrap_err()
            .code,
        ErrorCode::UnsupportedFeature
    );
}
struct MetadataAdapter {
    id: &'static str,
    loads: Arc<AtomicUsize>,
}
impl TextAdapter for MetadataAdapter {
    fn id(&self) -> &'static str {
        self.id
    }
    fn accepts(&self, _: &Path, _: ResourceLimits) -> Result<bool> {
        Ok(true)
    }
    fn load(&self, _: &Path, _: ResourceLimits, _: &Cancellation) -> Result<Box<dyn TextModel>> {
        self.loads.fetch_add(1, Ordering::SeqCst);
        Err(Error::request("unexpected load"))
    }
    fn describe(&self, source: &mut Source, _: &Cancellation) -> Option<Result<TextMetadata>> {
        if source.architecture() != Some("qwen2") {
            return None;
        }
        Some(Ok(TextMetadata::new(
            Capabilities {
                adapter: self.id,
                architecture: "qwen2",
                format: "gguf",
                text_generation: true,
                images: false,
                managed_api: false,
                continuous_batching: false,
                qualification: "test",
            },
            json!({}),
            256,
            128,
            |_, _, _| {
                Err(Error::new(
                    ErrorCode::UnsupportedFeature,
                    "test planning contract",
                ))
            },
        )))
    }
}
#[test]
fn metadata_factories_are_extensible_ambiguous_claims_fail_and_no_execution_load_occurs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    std::fs::write(&path, common::tiny_model()).unwrap();
    let cancel = Cancellation::default();
    assert!(
        !Registry::new()
            .inspect(&path, &cancel)
            .unwrap()
            .descriptor()
            .executable
    );
    let loads = Arc::new(AtomicUsize::new(0));
    let mut registry = Registry::new();
    registry
        .register(Box::new(MetadataAdapter {
            id: "test-one",
            loads: loads.clone(),
        }))
        .unwrap();
    let inspected = registry.inspect(&path, &cancel).unwrap();
    assert_eq!(
        inspected
            .descriptor()
            .text_adapter
            .as_ref()
            .unwrap()
            .adapter,
        "test-one"
    );
    assert_eq!(inspected.descriptor().runtime_context_limit, Some(128));
    assert_eq!(
        inspected
            .plan(&request(), &hardware(), &cancel)
            .unwrap_err()
            .code,
        ErrorCode::UnsupportedFeature
    );
    registry
        .register(Box::new(MetadataAdapter {
            id: "test-two",
            loads: loads.clone(),
        }))
        .unwrap();
    assert!(
        registry
            .inspect(&path, &cancel)
            .err()
            .unwrap()
            .message
            .contains("ambiguous")
    );
    assert_eq!(loads.load(Ordering::SeqCst), 0);
}
#[test]
fn qwen35_session_estimates_include_fixed_recurrent_state_and_linear_kv_growth() {
    let c = coheryn_engine::qwen35::Config {
        zero_centered_norms: true,
        ssm_pre_exponentiated: false,
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
    let a = c.session_estimate(128, 1).unwrap();
    let b = c.session_estimate(256, 1).unwrap();
    assert_eq!(a.kv_capacity_bytes, 6 * 128 * 2 * 256 * 2 * 4);
    assert_eq!(b.total_bytes - a.total_bytes, a.kv_capacity_bytes);
    assert_eq!(a.workspace_bytes, b.workspace_bytes);
    assert!(a.workspace_bytes > 18 * 16 * 128 * 128 * 4);
    assert!(c.session_estimate(8193, 1).is_err());
    assert!(c.session_estimate(0, 1).is_err());
    assert!(c.session_estimate(128, 0).is_err());
}

struct CancelledMetadata(ErrorCode);
impl TextAdapter for CancelledMetadata {
    fn id(&self) -> &'static str {
        "cancelled-metadata"
    }
    fn accepts(&self, _: &Path, _: ResourceLimits) -> Result<bool> {
        Ok(true)
    }
    fn load(&self, _: &Path, _: ResourceLimits, _: &Cancellation) -> Result<Box<dyn TextModel>> {
        panic!("inspection must not load")
    }
    fn describe(&self, _: &mut Source, _: &Cancellation) -> Option<Result<TextMetadata>> {
        Some(Err(Error::new(self.0, "metadata cancelled")))
    }
}
#[test]
fn metadata_cancellation_is_not_downgraded_to_a_successful_inspection_report() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    std::fs::write(&path, common::tiny_model()).unwrap();
    let mut registry = Registry::new();
    registry
        .register(Box::new(CancelledMetadata(ErrorCode::Cancelled)))
        .unwrap();
    assert_eq!(
        registry
            .inspect(&path, &Cancellation::default())
            .err()
            .unwrap()
            .code,
        ErrorCode::Cancelled
    );
}

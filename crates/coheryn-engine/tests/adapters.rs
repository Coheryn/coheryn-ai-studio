use coheryn_engine::{
    Cancellation, Error, ErrorCode, ResourceLimits, Result,
    adapters::{Registry, TextAdapter, TextModel},
};
use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
struct TestAdapter {
    id: &'static str,
    claims: bool,
    loads: Arc<AtomicUsize>,
}
impl TextAdapter for TestAdapter {
    fn id(&self) -> &'static str {
        self.id
    }
    fn accepts(&self, _: &Path, _: ResourceLimits) -> Result<bool> {
        Ok(self.claims)
    }
    fn load(&self, _: &Path, _: ResourceLimits, _: &Cancellation) -> Result<Box<dyn TextModel>> {
        self.loads.fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::UnsupportedFeature,
            "selected adapter rejected source configuration",
        ))
    }
}
fn adapter(id: &'static str, claims: bool, calls: &Arc<AtomicUsize>) -> Box<dyn TextAdapter> {
    Box::new(TestAdapter {
        id,
        claims,
        loads: calls.clone(),
    })
}
fn limits() -> ResourceLimits {
    ResourceLimits {
        context: 128,
        max_memory: 512 << 20,
    }
}
#[test]
fn registry_rejects_duplicates_ambiguity_unknown_and_cancelled_loads() {
    let loads = Arc::new(AtomicUsize::new(0));
    let mut registry = Registry::new();
    assert_eq!(
        registry
            .load("unused", limits(), &Cancellation::default())
            .err()
            .unwrap()
            .code,
        ErrorCode::UnsupportedModelArchitecture
    );
    registry.register(adapter("one", true, &loads)).unwrap();
    assert!(registry.register(adapter("one", true, &loads)).is_err());
    registry.register(adapter("two", true, &loads)).unwrap();
    assert!(
        registry
            .load("unused", limits(), &Cancellation::default())
            .err()
            .unwrap()
            .message
            .contains("ambiguous")
    );
    assert_eq!(loads.load(Ordering::SeqCst), 0);
    let cancel = Cancellation::default();
    cancel.cancel();
    assert_eq!(
        registry
            .load("unused", limits(), &cancel)
            .err()
            .unwrap()
            .code,
        ErrorCode::Cancelled
    );
    assert_eq!(loads.load(Ordering::SeqCst), 0);
}
#[test]
fn selected_adapter_error_is_preserved_without_fallback() {
    let loads = Arc::new(AtomicUsize::new(0));
    let mut registry = Registry::new();
    registry
        .register(adapter("selected", true, &loads))
        .unwrap();
    registry.register(adapter("other", false, &loads)).unwrap();
    assert_eq!(
        registry
            .load("unused", limits(), &Cancellation::default())
            .err()
            .unwrap()
            .code,
        ErrorCode::UnsupportedFeature
    );
    assert_eq!(loads.load(Ordering::SeqCst), 1);
}
#[test]
fn builtin_detection_uses_actual_format_and_configuration() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("misleading.safetensors");
    std::fs::write(&path, b"GGUF").unwrap();
    // Container identification reaches the existing GGUF validator even under another extension.
    assert_ne!(
        Registry::default()
            .load(&path, limits(), &Cancellation::default())
            .err()
            .unwrap()
            .code,
        ErrorCode::UnsupportedModelArchitecture
    );
    std::fs::write(&path, b"not a model").unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type":"unsupported"}"#,
    )
    .unwrap();
    assert_eq!(
        Registry::default()
            .load(dir.path(), limits(), &Cancellation::default())
            .err()
            .unwrap()
            .code,
        ErrorCode::UnsupportedModelArchitecture
    );
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type":"qwen3_5"}"#,
    )
    .unwrap();
    assert!(
        Registry::default()
            .load(dir.path(), limits(), &Cancellation::default())
            .err()
            .unwrap()
            .message
            .contains("text configuration")
    );
}

#[test]
fn duplicate_configuration_keys_fail_before_model_loading() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type":"other","model_type":"qwen3_5"}"#,
    )
    .unwrap();
    assert!(
        Registry::default()
            .load(dir.path(), limits(), &Cancellation::default())
            .err()
            .unwrap()
            .message
            .contains("duplicate")
    );
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type":"qwen3_5","text_config":{"hidden_size":4,"hidden_size":8}}"#,
    )
    .unwrap();
    assert!(
        Registry::default()
            .load(dir.path(), limits(), &Cancellation::default())
            .err()
            .unwrap()
            .message
            .contains("duplicate")
    );
}

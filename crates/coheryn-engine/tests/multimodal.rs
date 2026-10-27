use coheryn_engine::multimodal::{Content, ContentPart, SelectionManifest};
use serde_json::{Value, json};

const MANIFEST: &[u8] = include_bytes!("../../../fixtures/vlm/models.json");
#[test]
fn selected_vlm_components_are_exact_bounded_and_explicitly_unqualified() {
    let manifest = SelectionManifest::parse(MANIFEST).unwrap();
    assert_eq!(manifest.models.len(), 2);
    assert!(!manifest.oracle.vision_reference_outputs_captured);
    assert!(!manifest.full_weight_downloaded);
    for model in manifest.models {
        assert_eq!(model.components.len(), 2);
        assert!(
            model
                .source_asset_names
                .contains(&"preprocessor_config.json".to_owned())
        );
    }
}
#[test]
fn hostile_component_paths_duplicate_roles_and_false_provenance_are_rejected() {
    for path in [
        "../record.json",
        "resources/../record.json",
        "/resources/record.json",
    ] {
        let mut manifest: Value = serde_json::from_slice(MANIFEST).unwrap();
        manifest["oracle"]["source_record"] = json!(path);
        assert!(SelectionManifest::parse(&serde_json::to_vec(&manifest).unwrap()).is_err());
    }
    for replacement in [
        json!("../weights.gguf"),
        json!("/etc/passwd"),
        json!("C:\\secret.gguf"),
    ] {
        let mut manifest: Value = serde_json::from_slice(MANIFEST).unwrap();
        manifest["models"][0]["components"][0]["file"] = replacement;
        assert!(SelectionManifest::parse(&serde_json::to_vec(&manifest).unwrap()).is_err());
    }
    let mut manifest: Value = serde_json::from_slice(MANIFEST).unwrap();
    manifest["models"][0]["components"][1]["role"] = json!("language");
    assert!(SelectionManifest::parse(&serde_json::to_vec(&manifest).unwrap()).is_err());
    let mut manifest: Value = serde_json::from_slice(MANIFEST).unwrap();
    manifest["models"][0]["components"][0]["hash_status"] = json!("verified");
    assert!(SelectionManifest::parse(&serde_json::to_vec(&manifest).unwrap()).is_err());
    let mut manifest: Value = serde_json::from_slice(MANIFEST).unwrap();
    manifest["runtime_dependency"] = json!(true);
    assert!(SelectionManifest::parse(&serde_json::to_vec(&manifest).unwrap()).is_err());
}
#[test]
fn content_parts_preserve_interleaving_and_enforce_aggregate_limits() {
    let content: Content = serde_json::from_value(json!([
        {"type":"text","text":"Compare"},
        {"type":"image_url","image_url":{"url":"data:image/png;base64,first"}},
        {"type":"text","text":"with"},
        {"type":"image_url","image_url":{"url":"https://example.org/second.png"}}
    ]))
    .unwrap();
    content.validate_bounds(1024, 2).unwrap();
    assert!(content.validate_bounds(1024, 1).is_err());
    assert!(content.validate_bounds(16, 2).is_err());
    match content {
        Content::Parts(parts) => {
            assert!(matches!(&parts[0],ContentPart::Text{text} if text=="Compare"));
            assert!(
                matches!(&parts[3],ContentPart::ImageUrl{image_url} if image_url.url.ends_with("second.png"))
            );
        }
        _ => panic!("ordered parts expected"),
    }
    assert!(
        serde_json::from_value::<Content>(
            json!([{"type":"image_url","image_url":{"url":"x","execute":"code"}}])
        )
        .is_err()
    );
    assert!(serde_json::from_value::<Content>(json!([{"type":"video","url":"x"}])).is_err());
}

#[path = "../../coheryn-engine/tests/common/mod.rs"]
mod common;
#[path = "../../coheryn-engine/tests/common/vision.rs"]
mod vision_fixture;
use axum::{
    Router,
    body::{Body, Bytes, to_bytes},
    http::{Request, StatusCode},
};
use coheryn_api::{
    auth::Credentials,
    config::ServerConfig,
    http::router,
    service::Service,
    vision::{VisionLimits, data_url},
};
use coheryn_engine::{Cancellation, ErrorCode};
use futures_util::stream;
use serde_json::{Value, json};
use std::{sync::Arc, time::Duration};
use tower::ServiceExt;

fn setup(limits: VisionLimits) -> (tempfile::TempDir, Arc<Service>, Router) {
    let dir = tempfile::tempdir().unwrap();
    let service = Service::open(
        ServerConfig {
            data_dir: dir.path().join("daemon"),
            max_memory: 512 << 20,
            max_context: 512,
            consumer_timeout_seconds: 1,
            generation_timeout_seconds: 10,
            queue_timeout_seconds: 3,
            queue_capacity: 1,
            max_model_bytes: 8 << 20,
            disk_quota: 32 << 20,
            vision: limits,
            ..Default::default()
        },
        Credentials {
            inference_token: "i".repeat(64),
            admin_token: "a".repeat(64),
        },
    )
    .unwrap();
    let app = router(service.clone());
    (dir, service, app)
}
fn request(path: &str, value: Value, admin: bool) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .header(
            "authorization",
            format!(
                "Bearer {}",
                if admin {
                    "a".repeat(64)
                } else {
                    "i".repeat(64)
                }
            ),
        )
        .body(Body::from(serde_json::to_vec(&value).unwrap()))
        .unwrap()
}
async fn call(app: &Router, path: &str, value: Value, admin: bool) -> (StatusCode, Value) {
    let r = app
        .clone()
        .oneshot(request(path, value, admin))
        .await
        .unwrap();
    let status = r.status();
    let b = to_bytes(r.into_body(), 4 << 20).await.unwrap();
    (status, serde_json::from_slice(&b).unwrap())
}
async fn install(service: &Arc<Service>, id: &str, bytes: Vec<u8>) {
    service
        .ingest(
            id.into(),
            None,
            None,
            stream::once(async move { Ok::<_, std::io::Error>(Bytes::from(bytes)) }),
        )
        .await
        .unwrap();
}
async fn pair(service: &Arc<Service>, app: &Router) {
    install(service, "language", common::tiny_vlm_model()).await;
    let vision = vision_fixture::component(256);
    install(service, "projector", std::fs::read(vision.path()).unwrap()).await;
    let (status, result) = call(
        app,
        "/api/v1/models/register-pair",
        json!({"model":"tiny-vlm","language":"language","vision":"projector"}),
        true,
    )
    .await;
    assert_eq!(status, 200, "{result}");
}
fn input(streaming: bool, count: usize) -> Value {
    let url = data_url(
        include_bytes!("../../../fixtures/vlm/images/minimum.png"),
        &Cancellation::default(),
    )
    .unwrap();
    let mut content: Vec<_> = (0..count)
        .map(|_| json!({"type":"image_url","image_url":{"url":url,"detail":"auto"}}))
        .collect();
    content.push(json!({"type":"text","text":"Describe. বাংলা 中文 العربية 🙂"}));
    json!({"model":"tiny-vlm","messages":[{"role":"user","content":content}],"max_tokens":2,"context":512,"threads":1,"stream":streaming})
}
#[tokio::test]
async fn pair_registry_plans_without_loading_and_protects_components() {
    let (_dir, service, app) = setup(Default::default());
    pair(&service, &app).await;
    let baseline = service.budget.stats().reserved;
    let inspected = service.inspect("tiny-vlm".into(), true).await.unwrap();
    assert_eq!(inspected["compatibility"]["executable"], true);
    assert_eq!(inspected["hashes_verified"], true);
    let (status, plan) = call(&app, "/api/v1/plans/estimate", json!({"model":"tiny-vlm","task":"image-to-text","context":512,"images":{"processor_profile":"qwen2vl-still-v1","images":[{"width":56,"height":56}]}}), true).await;
    assert_eq!(status, 200, "{plan}");
    assert_eq!(plan["image_feature_tokens"], 4);
    assert!(plan["weight_file_bytes"].as_u64().unwrap() > 0);
    assert_eq!(service.loaded().unwrap(), json!([]));
    assert_eq!(service.budget.stats().reserved, baseline);
    assert_eq!(
        service.remove("language").unwrap_err().error.code,
        ErrorCode::Conflict
    );
    assert!(!service.registry.model_path("tiny-vlm").unwrap().exists());
    let disk = service.registry.disk_bytes().unwrap();
    assert_eq!(disk, service.registry.get("tiny-vlm").unwrap().bytes);
    let nested = service
        .registry
        .register_pair("nested", "tiny-vlm", "projector")
        .unwrap_err();
    assert_eq!(nested.error.code, ErrorCode::InvalidRequest);
    service.remove("tiny-vlm").unwrap();
    service.remove("language").unwrap();
    service.remove("projector").unwrap();
    assert_eq!(service.registry.disk_bytes().unwrap(), 0);
}
#[tokio::test]
async fn native_openai_streaming_and_multiple_images_have_exact_parity() {
    let (_dir, service, app) = setup(Default::default());
    pair(&service, &app).await;
    service.load("tiny-vlm".into(), 512).await.unwrap();
    let baseline = service.budget.stats().reserved;
    let (status, native) = call(&app, "/api/v1/generate", input(false, 2), false).await;
    assert_eq!(status, 200, "{native}");
    assert_eq!(native["vision"]["images"], 2);
    assert_eq!(native["completion_tokens"], 2);
    let (status, openai) = call(&app, "/v1/chat/completions", input(false, 2), false).await;
    assert_eq!(status, 200, "{openai}");
    assert_eq!(openai["choices"][0]["message"]["content"], native["text"]);
    assert_eq!(openai["usage"]["prompt_tokens"], native["prompt_tokens"]);
    assert_eq!(
        openai["coheryn"]["vision"]["language_sha256"],
        native["vision"]["language_sha256"]
    );
    let mut value = input(true, 2);
    value["stream_options"] = json!({"include_usage":true});
    let response = app
        .clone()
        .oneshot(request("/v1/chat/completions", value, false))
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body = to_bytes(response.into_body(), 4 << 20).await.unwrap();
    let body = std::str::from_utf8(&body).unwrap();
    assert!(body.ends_with("data: [DONE]\n\n"));
    let events: Vec<Value> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|l| *l != "[DONE]")
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let text: String = events
        .iter()
        .filter_map(|v| v["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(text, native["text"]);
    assert_eq!(events.last().unwrap()["usage"]["completion_tokens"], 2);
    service.scheduler.drained().await;
    assert_eq!(service.budget.stats().reserved, baseline);
    service.unload("tiny-vlm").unwrap();
}
#[tokio::test]
async fn malformed_unsupported_and_excessive_inputs_fail_before_compute() {
    let (_dir, service, app) = setup(Default::default());
    pair(&service, &app).await;
    service.load("tiny-vlm".into(), 512).await.unwrap();
    let baseline = service.budget.stats().reserved;
    for change in [
        json!({"temperature":0.7}),
        json!({"stop":"x"}),
        json!({"context":513}),
        json!({"max_tokens":257}),
        json!({"unknown":true}),
        json!({"tools":[]}),
    ] {
        let mut v = input(false, 1);
        v.as_object_mut()
            .unwrap()
            .extend(change.as_object().unwrap().clone());
        let (status, e) = call(&app, "/v1/chat/completions", v, false).await;
        assert!(status.is_client_error(), "{e}");
    }
    for url in [
        "file:///etc/passwd",
        "https://127.0.0.1/private",
        "data:image/png;base64,!!!!",
        "data:image/gif;base64,AAAA",
        "data:image/jpeg;base64,iVBORw0KGgo=",
    ] {
        let mut v = input(false, 1);
        v["messages"][0]["content"][0]["image_url"]["url"] = json!(url);
        let (status, e) = call(&app, "/v1/chat/completions", v, false).await;
        assert!(status.is_client_error(), "{e}");
    }
    let (status, _) = call(&app, "/v1/chat/completions", input(false, 5), false).await;
    assert!(status.is_client_error());
    let mut v = input(false, 1);
    v["messages"][0]["role"] = json!("assistant");
    assert!(
        call(&app, "/v1/chat/completions", v, false)
            .await
            .0
            .is_client_error()
    );
    assert_eq!(service.scheduler.counters.snapshot().completed, 0);
    assert_eq!(service.budget.stats().reserved, baseline);
    let held = service
        .budget
        .reserve(service.budget.stats().limit - baseline - 1024)
        .unwrap();
    assert_eq!(
        call(&app, "/v1/chat/completions", input(false, 1), false)
            .await
            .0,
        503
    );
    drop(held);
    assert_eq!(service.budget.stats().reserved, baseline);
}
#[tokio::test]
async fn queued_disconnects_pin_both_components_and_release_all_admission() {
    let (_dir, service, app) = setup(Default::default());
    pair(&service, &app).await;
    service.load("tiny-vlm".into(), 512).await.unwrap();
    let baseline = service.budget.stats().reserved;
    let mut v = input(true, 1);
    v["max_tokens"] = json!(128);
    let first = app
        .clone()
        .oneshot(request("/v1/chat/completions", v, false))
        .await
        .unwrap();
    assert_eq!(first.status(), 200);
    let second = app
        .clone()
        .oneshot(request("/v1/chat/completions", input(true, 1), false))
        .await
        .unwrap();
    assert_eq!(second.status(), 200);
    assert!(service.budget.stats().reserved > baseline);
    assert_eq!(
        service.unload("tiny-vlm").unwrap_err().error.code,
        ErrorCode::Conflict
    );
    assert_eq!(
        service.remove("projector").unwrap_err().error.code,
        ErrorCode::Conflict
    );
    let third = call(&app, "/v1/chat/completions", input(false, 1), false).await;
    assert_eq!(third.0, 429, "{}", third.1);
    drop(second);
    drop(first);
    tokio::time::timeout(Duration::from_secs(5), service.scheduler.drained())
        .await
        .unwrap();
    assert_eq!(service.budget.stats().reserved, baseline);
    assert!(service.scheduler.counters.snapshot().cancelled >= 1);
    service.unload("tiny-vlm").unwrap();
}
#[tokio::test]
async fn slow_consumer_receives_error_and_done_after_worker_acknowledges_cancel() {
    let (_dir, service, app) = setup(Default::default());
    pair(&service, &app).await;
    service.load("tiny-vlm".into(), 512).await.unwrap();
    let baseline = service.budget.stats().reserved;
    let mut v = input(true, 1);
    v["max_tokens"] = json!(128);
    let response = app
        .clone()
        .oneshot(request("/v1/chat/completions", v, false))
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    tokio::time::timeout(Duration::from_secs(10), async {
        while service
            .scheduler
            .counters
            .snapshot()
            .vision_backpressure_waits
            == 0
        {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(1200)).await;
    let bytes = to_bytes(response.into_body(), 4 << 20).await.unwrap();
    let body = std::str::from_utf8(&bytes).unwrap();
    assert!(body.contains("\"error\""), "{body}");
    assert!(body.contains("[DONE]"));
    assert!(!body.contains("\"finish_reason\":\"length\""));
    service.scheduler.drained().await;
    assert_eq!(service.budget.stats().reserved, baseline);
}
#[tokio::test]
async fn operator_limits_auth_and_remote_url_restrictions_are_enforced() {
    let limits = VisionLimits {
        allow_urls: true,
        images: coheryn_engine::multimodal::image::ImageLimits {
            max_encoded_bytes: 256,
            ..Default::default()
        },
        ..Default::default()
    };
    let (_dir, service, app) = setup(limits);
    pair(&service, &app).await;
    service.load("tiny-vlm".into(), 512).await.unwrap();
    assert_eq!(
        call(
            &app,
            "/api/v1/models/register-pair",
            json!({"model":"x","language":"language","vision":"projector"}),
            false
        )
        .await
        .0,
        403
    );
    for url in [
        "http://example.com/photo.png",
        "https://user:secret@example.com/photo.png",
        "https://example.com:444/photo.png",
        "https://127.0.0.1/photo.png",
        "https://[::1]/photo.png",
    ] {
        let mut v = input(false, 1);
        v["messages"][0]["content"][0]["image_url"]["url"] = json!(url);
        let (status, e) = call(&app, "/v1/chat/completions", v, false).await;
        assert_eq!(status, 400, "{e}");
        assert!(!e.to_string().contains("secret"));
    }
    let mut v = input(false, 1);
    v["messages"][0]["content"][0]["image_url"]["url"] =
        json!(format!("data:image/png;base64,{}", "A".repeat(1024)));
    assert_eq!(call(&app, "/v1/chat/completions", v, false).await.0, 413);
    // Same-size mutation is caught by the pinned component hash before residency is installed.
    service.unload("tiny-vlm").unwrap();
    let path = service.registry.model_path("projector").unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    std::fs::write(path, bytes).unwrap();
    assert_eq!(
        service
            .load("tiny-vlm".into(), 512)
            .await
            .unwrap_err()
            .error
            .code,
        ErrorCode::ModelChanged
    );
    assert_eq!(service.loaded().unwrap(), json!([]));
}

#[path = "../../coheryn-engine/tests/common/mod.rs"]
mod common;
use axum::{
    Router,
    body::{Body, Bytes, to_bytes},
    http::{Request, StatusCode},
};
use coheryn_api::{
    auth::Credentials, config::ServerConfig, http::router, scheduler::GenerationEvent,
    service::Service, wire::GenerationRequest,
};
use coheryn_engine::{Cancellation, ErrorCode, GenerationOptions, Model, ResourceLimits};
use futures_util::stream;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{sync::Arc, time::Duration};
use tower::ServiceExt;

fn credentials() -> Credentials {
    Credentials {
        inference_token: "i".repeat(64),
        admin_token: "a".repeat(64),
    }
}
fn setup(queue_capacity: usize) -> (tempfile::TempDir, Arc<Service>, Router) {
    setup_limits(queue_capacity, 1, 128 << 20)
}
fn setup_limits(
    queue_capacity: usize,
    max_concurrency: usize,
    max_memory: u64,
) -> (tempfile::TempDir, Arc<Service>, Router) {
    let dir = tempfile::tempdir().unwrap();
    let config = ServerConfig {
        data_dir: dir.path().join("service"),
        max_memory,
        max_concurrency,
        max_context: 128,
        queue_capacity,
        max_model_bytes: 1 << 20,
        disk_quota: 4 << 20,
        consumer_timeout_seconds: 1,
        generation_timeout_seconds: 10,
        queue_timeout_seconds: 2,
        ..Default::default()
    };
    let service = Service::open(config, credentials()).unwrap();
    let app = router(service.clone());
    (dir, service, app)
}
fn request(method: &str, path: &str, admin: Option<bool>, body: Value) -> Request<Body> {
    let mut builder = Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", "application/json");
    if let Some(admin) = admin {
        builder = builder.header(
            "authorization",
            format!(
                "Bearer {}",
                if admin {
                    credentials().admin_token
                } else {
                    credentials().inference_token
                }
            ),
        );
    }
    builder
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap()
}
async fn call(
    app: &Router,
    method: &str,
    path: &str,
    admin: Option<bool>,
    body: Value,
) -> (StatusCode, Value) {
    let response = app
        .clone()
        .oneshot(request(method, path, admin, body))
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 16 << 20).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}
async fn install(service: &Arc<Service>, name: &str) {
    let bytes = common::tiny_model();
    let hash = coheryn_engine::storage::digest_hex(Sha256::digest(&bytes).into());
    service
        .ingest(
            name.into(),
            Some(hash),
            None,
            stream::once(async move { Ok::<_, std::io::Error>(Bytes::from(bytes)) }),
        )
        .await
        .unwrap();
}
async fn loaded(service: &Arc<Service>) {
    install(service, "tiny").await;
    service.load("tiny".into(), 128).await.unwrap();
}
fn generation(max_tokens: usize) -> GenerationRequest {
    GenerationRequest::completion(
        "tiny".into(),
        "Hi".into(),
        &GenerationOptions {
            max_tokens,
            ..Default::default()
        },
        128,
        true,
    )
}

#[tokio::test]
async fn credentials_are_required_and_roles_are_separate() {
    let (_dir, _s, app) = setup(1);
    assert_eq!(call(&app, "GET", "/health", None, Value::Null).await.0, 200);
    assert_eq!(
        call(&app, "GET", "/v1/models", None, Value::Null).await.0,
        401
    );
    assert_eq!(
        call(&app, "GET", "/v1/models", Some(false), Value::Null)
            .await
            .0,
        200
    );
    assert_eq!(
        call(
            &app,
            "GET",
            "/api/v1/server/status",
            Some(false),
            Value::Null
        )
        .await
        .0,
        403
    );
    assert_eq!(
        call(
            &app,
            "GET",
            "/api/v1/server/status",
            Some(true),
            Value::Null
        )
        .await
        .0,
        200
    );
    let unsupported = call(&app, "POST", "/v1/models", Some(false), Value::Null).await;
    assert_eq!(unsupported.0, 405);
    assert_eq!(unsupported.1["error"]["code"], "INVALID_REQUEST");
    let mut duplicate = request("GET", "/v1/models", Some(false), Value::Null);
    duplicate
        .headers_mut()
        .append("authorization", "Bearer invalid".parse().unwrap());
    assert_eq!(app.oneshot(duplicate).await.unwrap().status(), 401);
}
#[tokio::test]
async fn managed_import_load_validate_unload_remove_lifecycle() {
    let (_dir, s, app) = setup(1);
    let bytes = common::tiny_model();
    let hash = coheryn_engine::storage::digest_hex(Sha256::digest(&bytes).into());
    let import = || {
        Request::builder()
            .method("POST")
            .uri(format!("/api/v1/models/import?name=tiny&sha256={hash}"))
            .header(
                "authorization",
                format!("Bearer {}", credentials().admin_token),
            )
            .header("content-type", "application/octet-stream")
            .body(Body::from(bytes.clone()))
            .unwrap()
    };
    assert_eq!(app.clone().oneshot(import()).await.unwrap().status(), 200);
    assert_eq!(app.clone().oneshot(import()).await.unwrap().status(), 409);
    let list = call(&app, "GET", "/v1/models", Some(false), Value::Null)
        .await
        .1;
    assert_eq!(list["data"][0]["id"], "tiny");
    assert_eq!(
        call(
            &app,
            "POST",
            "/api/v1/models/validate",
            Some(true),
            json!({"model":"tiny","deep":true})
        )
        .await
        .0,
        200
    );
    assert_eq!(
        call(
            &app,
            "GET",
            "/api/v1/models/tiny/inspect",
            Some(true),
            Value::Null
        )
        .await
        .0,
        200
    );
    assert_eq!(
        call(
            &app,
            "POST",
            "/api/v1/models/load",
            Some(true),
            json!({"model":"tiny","context":128})
        )
        .await
        .0,
        200
    );
    assert_eq!(
        call(
            &app,
            "POST",
            "/api/v1/models/load",
            Some(true),
            json!({"model":"tiny","context":64})
        )
        .await
        .0,
        409
    );
    assert_eq!(
        call(
            &app,
            "DELETE",
            "/api/v1/models/tiny",
            Some(true),
            Value::Null
        )
        .await
        .0,
        409
    );
    assert_eq!(
        call(
            &app,
            "POST",
            "/api/v1/models/unload",
            Some(true),
            json!({"model":"tiny"})
        )
        .await
        .0,
        200
    );
    assert_eq!(s.budget.stats().reserved, 0);
    assert_eq!(
        call(
            &app,
            "DELETE",
            "/api/v1/models/tiny",
            Some(true),
            Value::Null
        )
        .await
        .0,
        200
    );
    assert!(s.registry.list().unwrap().is_empty());
    assert_eq!(s.registry.disk_bytes().unwrap(), 0);
}

#[tokio::test]
async fn embedding_inspection_is_task_aware_and_generation_planning_requires_its_task() {
    let (_dir, s, app) = setup(1);
    let bytes = common::embedding_model(1);
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/models/import?name=encoder")
                .header(
                    "authorization",
                    format!("Bearer {}", credentials().admin_token),
                )
                .header("content-type", "application/octet-stream")
                .body(Body::from(bytes))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let (status, report) = call(
        &app,
        "POST",
        "/api/v1/models/validate",
        Some(true),
        json!({"model":"encoder","deep":true}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let report = &report["compatibility"];
    assert_eq!(report["task"], "embedding");
    assert_eq!(report["executable"], true);
    assert_eq!(report["embedding_config"]["hidden"], 32);
    let (status, value) = call(
        &app,
        "POST",
        "/api/v1/plans/estimate",
        Some(true),
        json!({"model":"encoder","context":32}),
    )
    .await;
    assert!(status.is_client_error(), "{value}");
    assert!(
        value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("embedding")
    );
    assert_eq!(
        call(
            &app,
            "POST",
            "/api/v1/models/load",
            Some(true),
            json!({"model":"encoder","context":32})
        )
        .await
        .0,
        StatusCode::OK
    );
    assert!(s.model("encoder").is_err());
}
#[tokio::test]
async fn standalone_native_openai_and_chat_outputs_match() {
    let (_dir, s, app) = setup(2);
    loaded(&s).await;
    let standalone = Model::load(
        s.registry.model_path("tiny").unwrap(),
        ResourceLimits {
            context: 128,
            max_memory: 128 << 20,
        },
    )
    .unwrap();
    let expected = standalone
        .generate(
            "Hi",
            &GenerationOptions {
                max_tokens: 4,
                ..Default::default()
            },
            &Cancellation::default(),
            |_| Ok(()),
        )
        .unwrap();
    let body = json!({"model":"tiny","prompt":"Hi","max_tokens":4,"context":128});
    let native = call(&app, "POST", "/api/v1/generate", Some(false), body.clone()).await;
    assert_eq!(native.0, 200);
    assert_eq!(native.1["text"], expected.text);
    assert_eq!(native.1["token_ids"], json!(expected.token_ids));
    let completion = call(&app, "POST", "/v1/completions", Some(false), body).await;
    assert_eq!(completion.0, 200);
    assert_eq!(completion.1["choices"][0]["text"], expected.text);
    assert_eq!(completion.1["usage"]["completion_tokens"], 4);
    let chat = call(
        &app,
        "POST",
        "/v1/chat/completions",
        Some(false),
        json!({"model":"tiny","messages":[{"role":"user","content":"Hi"}],"max_tokens":4}),
    )
    .await;
    assert_eq!(chat.0, 200);
    assert_eq!(chat.1["choices"][0]["message"]["content"], expected.text);
}
#[tokio::test]
async fn sse_chunks_have_delta_finish_usage_and_done() {
    let (_dir, s, app) = setup(1);
    loaded(&s).await;
    let response=app.oneshot(request("POST","/v1/chat/completions",Some(false),json!({"model":"tiny","messages":[{"role":"user","content":"Hi"}],"max_tokens":4,"stream":true,"stream_options":{"include_usage":true}}))).await.unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(text.ends_with("data: [DONE]\n\n"));
    let events: Vec<Value> = text
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|l| *l != "[DONE]")
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(events[0]["choices"][0]["delta"]["role"], "assistant");
    let output: String = events
        .iter()
        .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(output, "!!!!");
    assert!(
        events
            .iter()
            .any(|e| e["choices"][0]["finish_reason"] == "length")
    );
    assert_eq!(events.last().unwrap()["usage"]["completion_tokens"], 4);
    assert!(
        events[..events.len() - 1]
            .iter()
            .all(|e| e.get("usage") == Some(&Value::Null))
    );
}
#[tokio::test]
async fn request_limits_and_unsupported_fields_fail_explicitly() {
    let (_dir, _s, app) = setup(1);
    for body in [
        json!({"model":"tiny","prompt":"Hi","tools":[]}),
        json!({"model":"tiny","prompt":"Hi","n":2}),
        json!({"model":"tiny","prompt":["Hi"]}),
        json!({"model":"tiny","prompt":"Hi","temperature":-1}),
    ] {
        assert_eq!(
            call(&app, "POST", "/v1/completions", Some(false), body)
                .await
                .0,
            400
        );
    }
    assert_eq!(
        call(
            &app,
            "POST",
            "/api/v1/models/validate",
            Some(true),
            json!({"path":"/etc/passwd"})
        )
        .await
        .0,
        400
    );
    assert_eq!(
        call(
            &app,
            "DELETE",
            "/api/v1/models/..%2Fsecret",
            Some(true),
            Value::Null
        )
        .await
        .0,
        400
    );
    let mut req = request("GET", "/v1/models", Some(false), Value::Null);
    for i in 0..65 {
        req.headers_mut().insert(
            format!("x-test-{i}")
                .parse::<axum::http::HeaderName>()
                .unwrap(),
            "x".parse().unwrap(),
        );
    }
    assert_eq!(app.clone().oneshot(req).await.unwrap().status(), 431);
    let req = Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header(
            "authorization",
            format!("Bearer {}", credentials().inference_token),
        )
        .header("content-type", "application/json")
        .body(Body::from(vec![b'x'; (1 << 20) + 1]))
        .unwrap();
    assert_eq!(app.oneshot(req).await.unwrap().status(), 413);
}
#[tokio::test]
async fn failed_hash_interrupted_stream_and_quota_leave_no_partial_model() {
    let (_dir, s, _app) = setup(1);
    let bytes = common::tiny_model();
    assert!(
        s.ingest(
            "bad".into(),
            Some("0".repeat(64)),
            None,
            stream::once(async move { Ok::<_, std::io::Error>(Bytes::from(bytes)) })
        )
        .await
        .is_err()
    );
    let broken = stream::iter(vec![
        Ok(Bytes::from_static(b"GGUF")),
        Err(std::io::Error::other("disconnected")),
    ]);
    assert!(s.ingest("broken".into(), None, None, broken).await.is_err());
    let too_big =
        stream::once(async { Ok::<_, std::io::Error>(Bytes::from(vec![0; (1 << 20) + 1])) });
    assert!(s.ingest("huge".into(), None, None, too_big).await.is_err());
    assert_eq!(s.registry.disk_bytes().unwrap(), 0);
    assert!(s.registry.list().unwrap().is_empty());
}
#[tokio::test]
async fn bounded_queue_pins_models_and_disconnect_reclaims_reservations() {
    let (_dir, s, _app) = setup(1);
    loaded(&s).await;
    let baseline = s.budget.stats().reserved;
    let first = s
        .scheduler
        .submit(s.model("tiny").unwrap(), generation(100))
        .unwrap();
    tokio::time::timeout(Duration::from_secs(2), async {
        while s.scheduler.counters.snapshot().active == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let second = s
        .scheduler
        .submit(s.model("tiny").unwrap(), generation(100))
        .unwrap();
    let error = s
        .scheduler
        .submit(s.model("tiny").unwrap(), generation(100))
        .err()
        .unwrap();
    assert_eq!(error.error.code, ErrorCode::QueueFull);
    assert_eq!(s.unload("tiny").unwrap_err().status, 409);
    drop(first);
    drop(second);
    tokio::time::timeout(Duration::from_secs(3), s.scheduler.drained())
        .await
        .unwrap();
    assert_eq!(s.budget.stats().reserved, baseline);
    s.unload("tiny").unwrap();
    assert_eq!(s.budget.stats().reserved, 0);
}
#[tokio::test]
async fn slow_consumer_is_cancelled_with_a_retained_terminal_error() {
    let (_dir, s, _app) = setup(0);
    loaded(&s).await;
    let baseline = s.budget.stats().reserved;
    let mut handle = s
        .scheduler
        .submit(s.model("tiny").unwrap(), generation(100))
        .unwrap();
    tokio::time::timeout(Duration::from_secs(4), s.scheduler.drained())
        .await
        .unwrap();
    assert_eq!(s.budget.stats().reserved, baseline);
    assert_eq!(s.scheduler.counters.snapshot().cancelled, 1);
    let mut tokens = 0;
    let mut terminal = false;
    while let Some(event) = handle.next().await {
        match event {
            GenerationEvent::Token(_) | GenerationEvent::ToolDelta(_) => tokens += 1,
            GenerationEvent::Failed(e) => {
                assert_eq!(e.error.code, ErrorCode::Cancelled);
                terminal = true;
            }
            GenerationEvent::Finished(_) => panic!("slow consumer must not report success"),
        }
    }
    assert_eq!(tokens, 16);
    assert!(terminal);
}
#[tokio::test]
async fn sse_body_drop_cancels_and_http_capacity_lasts_for_body_lifetime() {
    let (_dir, s, app) = setup(1);
    loaded(&s).await;
    let baseline = s.budget.stats().reserved;
    let response = app
        .oneshot(request(
            "POST",
            "/v1/completions",
            Some(false),
            json!({"model":"tiny","prompt":"Hi","max_tokens":100,"stream":true}),
        ))
        .await
        .unwrap();
    assert_eq!(s.http_slots.available_permits(), 63);
    drop(response);
    tokio::time::timeout(Duration::from_secs(3), s.scheduler.drained())
        .await
        .unwrap();
    assert_eq!(s.http_slots.available_permits(), 64);
    assert_eq!(s.budget.stats().reserved, baseline);
}
#[tokio::test]
async fn shutdown_cancels_active_and_queued_requests() {
    let (_dir, s, _app) = setup(1);
    loaded(&s).await;
    let _first = s
        .scheduler
        .submit(s.model("tiny").unwrap(), generation(100))
        .unwrap();
    let _second = s
        .scheduler
        .submit(s.model("tiny").unwrap(), generation(100))
        .unwrap();
    s.shutdown.cancel();
    tokio::time::timeout(Duration::from_secs(3), s.scheduler.drained())
        .await
        .unwrap();
    assert!(
        s.scheduler
            .submit(s.model("tiny").unwrap(), generation(1))
            .is_err()
    );
    s.unload("tiny").unwrap();
    assert_eq!(s.budget.stats().reserved, 0);
}
#[tokio::test]
async fn registry_lock_and_private_download_targets_are_rejected() {
    let (_dir, s, app) = setup(1);
    assert!(Service::open(s.config.clone(), credentials()).is_err());
    for url in [
        "http://127.0.0.1/model",
        "https://127.0.0.1/model",
        "https://user:pass@example.com/model",
    ] {
        assert_eq!(
            call(
                &app,
                "POST",
                "/api/v1/models/pull",
                Some(true),
                json!({"name":"nope","url":url,"sha256":"0".repeat(64)})
            )
            .await
            .0,
            400
        );
    }
}
#[tokio::test]
async fn estimate_exposes_reservations_without_loading_or_changing_context() {
    let (_dir, s, app) = setup(1);
    install(&s, "tiny").await;
    let result = call(
        &app,
        "POST",
        "/api/v1/plans/estimate",
        Some(true),
        json!({"model":"tiny","context":64}),
    )
    .await;
    assert_eq!(result.0, 200);
    assert_eq!(result.1["context"], 64);
    assert!(result.1["admissible"].as_bool().unwrap());
    assert_eq!(s.budget.stats().reserved, 0);
    assert_eq!(s.loaded().unwrap(), json!([]));
    assert_eq!(
        call(
            &app,
            "POST",
            "/api/v1/plans/estimate",
            Some(true),
            json!({"model":"tiny","context":129})
        )
        .await
        .0,
        400
    );
    assert_eq!(
        call(
            &app,
            "POST",
            "/api/v1/plans/estimate",
            Some(true),
            json!({"model":"tiny","context":64,"profile":"unknown"})
        )
        .await
        .0,
        400
    );
}

#[tokio::test]
async fn explicit_request_memory_cap_and_missing_models_are_not_ignored() {
    let (_dir, s, app) = setup(1);
    loaded(&s).await;
    let baseline = s.budget.stats().reserved;
    let request = json!({"model":"tiny","prompt":"Hi","max_tokens":1,"max_memory":1});
    let result = call(&app, "POST", "/api/v1/generate", Some(false), request).await;
    assert_eq!(result.0, 503);
    assert_eq!(result.1["error"]["code"], "RESOURCE_EXHAUSTED");
    assert_eq!(s.budget.stats().reserved, baseline);
    assert_eq!(
        call(
            &app,
            "POST",
            "/v1/completions",
            Some(false),
            json!({"model":"missing","prompt":"Hi","max_tokens":1})
        )
        .await
        .0,
        404
    );
}
#[tokio::test]
async fn real_loopback_transport_supports_client_stream_and_load_lifecycle() {
    let (dir, s, _app) = setup(1);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let service = s.clone();
    let server =
        tokio::spawn(async move { coheryn_api::http::serve(listener, service).await.unwrap() });
    let client =
        coheryn_api::client::Client::with_credentials(&format!("http://{address}"), credentials())
            .unwrap();
    let cancel = Cancellation::default();
    let path = dir.path().join("source.gguf");
    std::fs::write(&path, common::tiny_model()).unwrap();
    client.import(&path, "tiny", None, &cancel).await.unwrap();
    client
        .json(
            "POST",
            "/api/v1/models/load",
            Some(json!({"model":"tiny","context":128})),
            true,
            &cancel,
        )
        .await
        .unwrap();
    let mut output = String::new();
    let result = client
        .generate(&generation(4), &cancel, |event| {
            output += &event.text;
            Ok(())
        })
        .await
        .unwrap();
    assert_eq!(result.text, "!!!!");
    assert_eq!(result.text, output);
    client
        .json(
            "POST",
            "/api/v1/models/unload",
            Some(json!({"model":"tiny"})),
            true,
            &cancel,
        )
        .await
        .unwrap();
    client
        .json("DELETE", "/api/v1/models/tiny", None, true, &cancel)
        .await
        .unwrap();
    s.shutdown.cancel();
    tokio::time::timeout(Duration::from_secs(3), server)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
#[ignore = "requires outbound HTTPS; reads only the GGUF magic from the pinned publisher file"]
async fn pinned_public_https_download_follows_verified_redirects() {
    let url = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-fp16.gguf";
    let mut response = coheryn_api::download::response(url).await.unwrap();
    let mut prefix: Vec<u8> = Vec::new();
    while prefix.len() < 4 {
        let chunk = response.chunk().await.unwrap().unwrap();
        prefix.extend(chunk.iter().take(4 - prefix.len()));
    }
    assert_eq!(prefix, b"GGUF");
}

#[tokio::test]
async fn caep_profiles_share_metadata_and_report_unqualified_images_without_loading() {
    let (_dir, s, app) = setup_limits(2, 2, 128 << 20);
    install(&s, "tiny").await;
    for profile in ["reference", "performance", "balanced", "low-memory"] {
        let (status, plan) = call(
            &app,
            "POST",
            "/api/v1/plans/estimate",
            Some(true),
            json!({"model":"tiny","context":64,"profile":profile,"concurrency":1}),
        )
        .await;
        assert_eq!(status, 200);
        assert_eq!(plan["profile"], profile);
        assert!(plan["admissible"].as_bool().unwrap());
        let cpus = plan["hardware"]["logical_cpus"].as_u64().unwrap() as usize;
        let share = (cpus / s.scheduler.max_concurrency()).clamp(1, 16);
        let threads = match profile {
            "performance" => share.min(4),
            "balanced" => share.min(2),
            _ => 1,
        };
        assert_eq!(plan["threads_per_session"], threads);
        assert_eq!(plan["cache_dtype"], "f32");
        assert_eq!(s.budget.stats().reserved, 0);
    }
    let (_, one) = call(
        &app,
        "POST",
        "/api/v1/plans/estimate",
        Some(true),
        json!({"model":"tiny","context":64}),
    )
    .await;
    let (_, two) = call(
        &app,
        "POST",
        "/api/v1/plans/estimate",
        Some(true),
        json!({"model":"tiny","context":64,"concurrency":2}),
    )
    .await;
    assert_eq!(one["shared_weight_bytes"], two["shared_weight_bytes"]);
    assert_eq!(
        one["model_metadata_additional"],
        two["model_metadata_additional"]
    );
    assert_eq!(
        two["additional_reserved_bytes"].as_u64().unwrap()
            - two["model_metadata_additional"].as_u64().unwrap(),
        2 * (one["additional_reserved_bytes"].as_u64().unwrap()
            - one["model_metadata_additional"].as_u64().unwrap())
    );
    let (_,vision)=call(&app,"POST","/api/v1/plans/estimate",Some(true),
        json!({"model":"tiny","context":64,"images":{"processor_profile":"candidate","images":[{"width":336,"height":336}]}})).await;
    assert_eq!(vision["admissible"], false);
    assert_eq!(vision["image_stages"]["qualified"], false);
    assert!(vision["image_stages"]["vision_workspace_bytes"].is_null());
    s.load("tiny".into(), 128).await.unwrap();
    let baseline = s.budget.stats().reserved;
    let (_, loaded) = call(
        &app,
        "POST",
        "/api/v1/plans/estimate",
        Some(true),
        json!({"model":"tiny","context":64}),
    )
    .await;
    assert_eq!(loaded["model_metadata_additional"], 0);
    assert_eq!(loaded["temporary_inspection_reserved_bytes"], 0);
    assert_eq!(s.budget.stats().reserved, baseline);
}

#[tokio::test]
async fn two_sessions_run_together_and_low_memory_acquires_exclusive_execution() {
    let (_dir, s, _app) = setup_limits(2, 2, 128 << 20);
    loaded(&s).await;
    let baseline = s.budget.stats().reserved;
    let first = s
        .scheduler
        .submit(s.model("tiny").unwrap(), generation(100))
        .unwrap();
    let second = s
        .scheduler
        .submit(s.model("tiny").unwrap(), generation(100))
        .unwrap();
    tokio::time::timeout(Duration::from_secs(2), async {
        while s.scheduler.counters.snapshot().active != 2
            || s.budget.stats().session_buffer_capacity_bytes == 0
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(s.budget.stats().reserved > baseline);
    assert!(s.budget.stats().session_buffer_capacity_bytes > 0);
    drop(first);
    drop(second);
    s.scheduler.drained().await;
    let mut low = generation(100);
    low.profile = "low-memory".into();
    let first = s.scheduler.submit(s.model("tiny").unwrap(), low).unwrap();
    tokio::time::timeout(Duration::from_secs(2), async {
        while s.scheduler.counters.snapshot().active != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let second = s
        .scheduler
        .submit(s.model("tiny").unwrap(), generation(100))
        .unwrap();
    assert_eq!(s.scheduler.counters.snapshot().active, 1);
    assert_eq!(s.scheduler.counters.snapshot().queued, 1);
    drop(first);
    drop(second);
    s.scheduler.drained().await;
    assert_eq!(s.budget.stats().reserved, baseline);
    assert_eq!(s.budget.stats().session_buffer_capacity_bytes, 0);
}

#[tokio::test]
async fn queued_session_reservations_cannot_overcommit_the_global_budget() {
    let (_dir, s, _app) = setup_limits(8, 1, 32 << 20);
    loaded(&s).await;
    let first = s
        .scheduler
        .submit(s.model("tiny").unwrap(), generation(100))
        .unwrap();
    let second = s
        .scheduler
        .submit(s.model("tiny").unwrap(), generation(100))
        .unwrap();
    let reserved = s.budget.stats().reserved;
    let error = s
        .scheduler
        .submit(s.model("tiny").unwrap(), generation(100))
        .err()
        .unwrap();
    assert_eq!(error.error.code, ErrorCode::ResourceExhausted);
    assert_eq!(s.budget.stats().reserved, reserved);
    assert!(s.budget.stats().reserved <= s.config.max_memory);
    drop(first);
    drop(second);
    s.scheduler.drained().await;
    s.unload("tiny").unwrap();
    assert_eq!(s.budget.stats().reserved, 0);
}

#[tokio::test]
async fn profile_execution_limits_do_not_expand_a_zero_capacity_waiting_queue() {
    let (_dir, s, _app) = setup_limits(0, 2, 128 << 20);
    loaded(&s).await;
    let mut low = generation(100);
    low.profile = "low-memory".into();
    let first = s.scheduler.submit(s.model("tiny").unwrap(), low).unwrap();
    let error = s
        .scheduler
        .submit(s.model("tiny").unwrap(), generation(100))
        .err()
        .unwrap();
    assert_eq!(error.error.code, ErrorCode::QueueFull);
    assert_eq!(s.scheduler.counters.snapshot().queued, 0);
    drop(first);
    s.scheduler.drained().await;
}

#[tokio::test]
async fn explicit_threads_are_admitted_with_stack_reservations_and_invalid_counts_fail_early() {
    let (_dir, s, app) = setup(1);
    loaded(&s).await;
    let (_, one) = call(
        &app,
        "POST",
        "/api/v1/plans/estimate",
        Some(true),
        json!({"model":"tiny","context":64,"profile":"performance","threads":1}),
    )
    .await;
    let (status, two) = call(
        &app,
        "POST",
        "/api/v1/plans/estimate",
        Some(true),
        json!({"model":"tiny","context":64,"profile":"performance","threads":2}),
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(two["threads_per_session"], 2);
    assert!(
        two["session_reserved_bytes"].as_u64().unwrap()
            > one["session_reserved_bytes"].as_u64().unwrap()
    );
    for threads in [0, 17] {
        let result = call(
            &app,
            "POST",
            "/api/v1/generate",
            Some(false),
            json!({"model":"tiny","prompt":"Hi","max_tokens":4,"threads":threads}),
        )
        .await;
        assert_eq!(result.0, 400);
    }
    let baseline = s.budget.stats().reserved;
    let result = call(
        &app,
        "POST",
        "/api/v1/generate",
        Some(false),
        json!({"model":"tiny","prompt":"Hi","max_tokens":4,"profile":"performance","threads":2}),
    )
    .await;
    assert_eq!(result.0, 200);
    assert_eq!(result.1["text"], "!!!!");
    s.scheduler.drained().await;
    assert_eq!(s.budget.stats().reserved, baseline);
    assert_eq!(s.budget.stats().session_buffer_capacity_bytes, 0);
}

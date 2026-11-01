#![cfg(all(feature = "metal", target_os = "macos"))]
//! Pinned model checks through the authenticated router; no external executor or sockets.
use axum::{
    Router,
    body::{Body, Bytes, to_bytes},
    http::Request,
};
use base64::{Engine, engine::general_purpose::STANDARD};
use coheryn_api::{
    auth::Credentials,
    checkpoint::{CheckpointManifest, MAGIC},
    config::ServerConfig,
    http::router,
    service::Service,
};
use futures_util::{StreamExt, TryStreamExt, stream};
use serde_json::{Value, json};
use std::{fs, io::Write, path::Path, sync::Arc, time::Duration};
use tokio_util::io::ReaderStream;
use tower::ServiceExt;

fn read(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}
fn request(path: &str, body: Value) -> Request<Body> {
    let admin = path.starts_with("/api/") && path != "/api/v1/generate";
    Request::builder()
        .method("POST")
        .uri(path)
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
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap()
}
async fn call(app: &Router, path: &str, body: Value, expected: u16) -> Value {
    let response = app.clone().oneshot(request(path, body)).await.unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 2 << 20).await.unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(status.as_u16(), expected, "{path}: {value}");
    value
}
async fn idle(service: &Arc<Service>) {
    tokio::time::timeout(Duration::from_secs(600), async {
        loop {
            let counters = service.scheduler.counters.snapshot();
            let loaded = service.loaded().unwrap();
            let unpinned = loaded
                .as_array()
                .unwrap()
                .iter()
                .all(|model| model["pinned"] == false);
            if counters.active == 0 && counters.queued == 0 && unpinned {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("owned workers must retire within the test bound");
}
fn check(result: &Value, expected: &Value) {
    assert_eq!(result["token_ids"], expected["greedy_token_ids"]);
    assert_eq!(result["text"], expected["greedy_text"]);
    assert_eq!(
        result["sampling_metadata"]["backend"],
        "native-metal-matmul-f32-reference"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires pinned pretrained assets, native Metal and the owned host guard"]
async fn managed_metal_router_preserves_answers_streaming_history_and_cleanup() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let suite = std::env::var("COHERYN_METAL_MODEL_SUITE").unwrap();
    let (source, reference) = match suite.as_str() {
        "qwen35-gguf-q4_k_m-text" => {
            let manifest = read(&root.join("fixtures/qwen35/gguf/models-v1.json"));
            let item = manifest["files"]
                .as_array()
                .unwrap()
                .iter()
                .find(|item| item["role"] == "language" && item["label"] == "q4_k_m")
                .unwrap();
            (
                root.join(manifest["cache_directory"].as_str().unwrap())
                    .join(item["file"].as_str().unwrap()),
                read(&root.join("fixtures/qwen35/gguf/q4_k_m-text.reference-v1.json")),
            )
        }
        "qwen35-text" => {
            let manifest = read(&root.join("fixtures/qwen35/model.json"));
            (
                root.join(manifest["cache_directory"].as_str().unwrap()),
                read(&root.join("fixtures/qwen35/reference-v1.json")),
            )
        }
        _ => panic!("explicit pinned suite required"),
    };
    let temporary = tempfile::tempdir_in(root.join(".cache")).unwrap();
    let service = Service::open(
        ServerConfig {
            data_dir: temporary.path().join("vault"),
            max_memory: 2 << 30,
            max_context: 128,
            max_concurrency: 2,
            cache: coheryn_api::sessions::CacheLimits {
                idle_seconds: 3600,
                ..Default::default()
            },
            generation_timeout_seconds: 600,
            consumer_timeout_seconds: 600,
            ..Default::default()
        },
        Credentials {
            inference_token: "i".repeat(64),
            admin_token: "a".repeat(64),
        },
    )
    .unwrap();
    let (manifest, base) = CheckpointManifest::from_local(&source, &Default::default()).unwrap();
    let bytes = serde_json::to_vec(&manifest).unwrap();
    let mut prefix = MAGIC.to_vec();
    prefix.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    prefix.extend_from_slice(&bytes);
    let paths = manifest
        .files
        .iter()
        .map(|f| base.join(&f.name))
        .collect::<Vec<_>>();
    let files = stream::iter(paths)
        .then(|path| async move { tokio::fs::File::open(path).await.map(ReaderStream::new) })
        .try_flatten();
    let first = stream::once(async move { Ok::<_, std::io::Error>(Bytes::from(prefix)) });
    let record = {
        let _admin = service.admin.lock().await;
        service
            .ingest_checkpoint(
                "gpu".into(),
                Some(manifest.identity().unwrap()),
                first.chain(files),
            )
            .await
            .unwrap()
    };
    let app = router(service.clone());
    let loaded = call(
        &app,
        "/api/v1/models/load",
        json!({"model":"gpu","context":128,"backend":"metal"}),
        200,
    )
    .await;
    assert_eq!(loaded["backend"], "metal");
    assert_eq!(service.loaded().unwrap()[0]["backend"], "metal");
    let baseline = service.budget.stats().reserved;
    let plan = call(
        &app,
        "/api/v1/plans/estimate",
        json!({"model":"gpu","context":128,"profile":"reference","threads":1,"concurrency":2}),
        200,
    )
    .await;
    assert_eq!(plan["backend"], "native-metal-matmul-f32-reference");
    assert!(plan["gpu_allocation_bytes"].as_u64().unwrap() > 0);
    assert_eq!(service.budget.stats().reserved, baseline);
    call(
        &app,
        "/api/v1/models/load",
        json!({"model":"gpu","context":128,"backend":"cpu"}),
        409,
    )
    .await;
    assert_eq!(service.budget.stats().reserved, baseline);
    let cases = reference["cases"].as_array().unwrap();
    let body = |case: &Value| json!({"model":"gpu","messages":case["messages"],"max_tokens":4,"context":128,"profile":"reference","threads":1});
    let (a, b) = tokio::join!(
        call(&app, "/api/v1/generate", body(&cases[0]), 200),
        call(&app, "/api/v1/generate", body(&cases[1]), 200)
    );
    check(&a, &cases[0]);
    check(&b, &cases[1]);
    idle(&service).await;
    assert_eq!(service.budget.stats().reserved, baseline);
    let mut streams = Vec::new();
    for expected in cases {
        let mut input = body(expected);
        input["stream"] = json!(true);
        input["stream_options"] = json!({"include_usage":true});
        let response = app
            .clone()
            .oneshot(request("/v1/chat/completions", input))
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let bytes = to_bytes(response.into_body(), 2 << 20).await.unwrap();
        let text = std::str::from_utf8(&bytes).unwrap();
        assert_eq!(
            text.lines().filter(|line| *line == "data: [DONE]").count(),
            1
        );
        assert!(text.ends_with("data: [DONE]\n\n"));
        let frames = text
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|line| *line != "[DONE]")
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        let answer: String = frames
            .iter()
            .filter_map(|frame| frame["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(answer, expected["greedy_text"]);
        assert_eq!(
            frames
                .iter()
                .filter(|frame| frame["choices"][0]["finish_reason"].is_string())
                .count(),
            1
        );
        let finish = frames
            .iter()
            .find(|frame| frame["choices"][0]["finish_reason"].is_string())
            .unwrap();
        assert_eq!(
            finish["coheryn"]["sampling"]["backend"],
            "native-metal-matmul-f32-reference"
        );
        let usage = frames
            .iter()
            .filter(|frame| frame["usage"].is_object())
            .collect::<Vec<_>>();
        assert_eq!(usage.len(), 1);
        assert_eq!(usage[0]["usage"]["completion_tokens"], 4);
        streams.push(json!({"text":answer,"done":1,"finish":finish,"usage":usage[0]["usage"]}));
    }
    idle(&service).await;
    assert_eq!(service.budget.stats().reserved, baseline);
    let mut input = body(&cases[0]);
    input["stream"] = json!(true);
    input["max_tokens"] = json!(64);
    let pending = app
        .clone()
        .oneshot(request("/v1/chat/completions", input))
        .await
        .unwrap();
    assert_eq!(pending.status(), 200);
    call(&app, "/api/v1/models/unload", json!({"model":"gpu"}), 409).await;
    drop(pending);
    idle(&service).await;
    assert_eq!(service.budget.stats().reserved, baseline);
    assert_eq!(service.budget.stats().session_buffer_capacity_bytes, 0);
    let healthy = call(&app, "/api/v1/generate", body(&cases[0]), 200).await;
    check(&healthy, &cases[0]);
    let image = STANDARD.encode(fs::read(root.join("fixtures/vlm/images/minimum.png")).unwrap());
    let error = call(&app, "/v1/chat/completions", json!({"model":"gpu","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":format!("data:image/png;base64,{image}")}},{"type":"text","text":"Describe."}]}],"max_tokens":4}), 400).await;
    assert_eq!(error["error"]["code"], "UNSUPPORTED_FEATURE");
    let session = call(
        &app,
        "/coheryn/v1/sessions",
        json!({"model":"gpu","context":128}),
        200,
    )
    .await;
    let path = format!(
        "/coheryn/v1/sessions/{}/messages",
        session["id"].as_str().unwrap()
    );
    let settings = json!({"max_tokens":4,"threads":1,"profile":"reference"});
    let one = call(&app, &path, json!({"expected_revision":0,"idempotency_key":"one","content":cases[0]["messages"][0]["content"],"settings":settings}), 200).await;
    check(&one["result"], &cases[0]);
    let followup = "你好. বাংলা again.";
    let two = call(&app, &path, json!({"expected_revision":2,"idempotency_key":"two","content":followup,"settings":settings}), 200).await;
    assert!(two["result"]["reused_prompt_tokens"].as_u64().unwrap() > 0);
    let fresh = call(&app, "/api/v1/generate", json!({"model":"gpu","messages":[cases[0]["messages"][0],{"role":"assistant","content":one["result"]["text"]},{"role":"user","content":followup}],"max_tokens":4,"context":128,"threads":1,"profile":"reference"}), 200).await;
    assert_eq!(two["result"]["token_ids"], fresh["token_ids"]);
    assert_eq!(two["result"]["text"], fresh["text"]);
    assert_eq!(
        fresh["sampling_metadata"]["backend"],
        "native-metal-matmul-f32-reference"
    );
    idle(&service).await;
    let cache = service.sessions.cache_stats().unwrap();
    assert!(cache["reserved_bytes"].as_u64().unwrap() > 0);
    assert!(
        service.budget.stats().session_buffer_capacity_bytes
            <= cache["reserved_bytes"].as_u64().unwrap()
    );
    let gpu = service.loaded().unwrap()[0]["gpu_execution"].clone();
    assert!(gpu["completed_model_dispatches"].as_u64().unwrap() > 0);
    assert!(gpu["encoded_weight_bytes"].as_u64().unwrap() > 0);
    assert!(
        gpu["driver_weight_allocation_bytes"].as_u64().unwrap()
            >= gpu["encoded_weight_bytes"].as_u64().unwrap()
    );
    service.unload("gpu").unwrap();
    assert_eq!(service.loaded().unwrap(), json!([]));
    assert_eq!(service.budget.stats().session_buffer_capacity_bytes, 0);
    let retained_history = service.budget.stats().reserved;
    assert!(retained_history > 0 && retained_history < baseline);
    let mut deletion = request(
        &format!("/coheryn/v1/sessions/{}", session["id"].as_str().unwrap()),
        json!({"expected_revision":4}),
    );
    *deletion.method_mut() = axum::http::Method::DELETE;
    let deleted = app.clone().oneshot(deletion).await.unwrap();
    assert_eq!(deleted.status(), 200);
    to_bytes(deleted.into_body(), 2 << 20).await.unwrap();
    assert_eq!(service.budget.stats().reserved, 0);
    service.remove("gpu").unwrap();
    service.shutdown.cancel();
    let report = json!({"schema_version":1,"outcome":"pass","suite":suite,"managed_record":record,"plan":plan,"gpu":gpu,
        "cases":[a,b],"streams":streams,"healthy_after_disconnect":healthy,"cached_result":two["result"],"fresh_result":fresh,
        "backend_conflict_rejected":true,"pinned_unload_rejected":true,"disconnect_cleanup":true,"image_profile_rejected":true,
        "plan_without_reservation":true,"unload_cache_release":true,"history_deleted":true,"post_unload_history_reserved":retained_history,"final_memory":service.budget.stats(),
        "scope":"Authenticated in-process managed Metal router, independent greedy outputs, SSE and history/lifecycle checks; no TCP, throughput or production qualification."});
    let destination = std::env::var("COHERYN_METAL_API_REPORT").unwrap();
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)
        .unwrap();
    file.write_all(serde_json::to_string_pretty(&report).unwrap().as_bytes())
        .unwrap();
}

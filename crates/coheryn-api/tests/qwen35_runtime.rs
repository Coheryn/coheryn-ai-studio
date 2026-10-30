//! Opt-in real checkpoint qualification through the HTTP router, without sockets.
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
use std::{path::Path, sync::Arc, time::Duration};
use tokio_util::io::ReaderStream;
use tower::ServiceExt;
fn digest(path: &Path) -> String {
    coheryn_api::checkpoint::hash_file(path, &Default::default()).unwrap()
}
fn request(method: &str, path: &str, body: Value) -> Request<Body> {
    let admin = path.starts_with("/api/") && path != "/api/v1/generate";
    Request::builder()
        .method(method)
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
async fn call(app: &Router, method: &str, path: &str, body: Value, expected: u16) -> Value {
    let response = app
        .clone()
        .oneshot(request(method, path, body))
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 16 << 20).await.unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(status.as_u16(), expected, "{path}: {value}");
    value
}
async fn idle(service: &Arc<Service>) {
    for _ in 0..3000 {
        let status = service.status().unwrap();
        if status["scheduler"]["active"] == 0 && status["scheduler"]["queued"] == 0 {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("workers did not retire");
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires separately acquired pinned Qwen3.5 checkpoint and private managed copy disk capacity"]
async fn native_hybrid_router_cache_batch_images_and_cleanup() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output = std::env::var_os("COHERYN_Q35_REPORT").map(std::path::PathBuf::from);
    if let Some(p) = &output {
        assert!(!p.exists(), "never overwrite qualification records");
    }
    let source = root.join(".cache/qwen3.5-0.8b-source");
    let manifest_path = root.join("fixtures/qwen35/model.json");
    let model_manifest: Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    for file in model_manifest["files"].as_array().unwrap() {
        let path = source.join(file["file"].as_str().unwrap());
        assert_eq!(digest(&path), file["sha256"]);
        assert_eq!(
            std::fs::metadata(path).unwrap().len(),
            file["bytes"].as_u64().unwrap()
        );
    }
    let temp = tempfile::tempdir_in(root.join(".cache")).unwrap();
    let config = ServerConfig {
        data_dir: temp.path().join("vault"),
        max_memory: 1 << 30,
        max_context: 256,
        max_concurrency: 2,
        generation_timeout_seconds: 600,
        ..Default::default()
    };
    let service = Service::open(
        config,
        Credentials {
            inference_token: "i".repeat(64),
            admin_token: "a".repeat(64),
        },
    )
    .unwrap();
    let (manifest, base) = CheckpointManifest::from_local(&source, &Default::default()).unwrap();
    assert!(
        manifest
            .files
            .iter()
            .any(|f| f.name == "preprocessor_config.json")
    );
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
                "q35".into(),
                Some(manifest.identity().unwrap()),
                first.chain(files),
            )
            .await
            .unwrap()
    };
    service.load("q35".into(), 256).await.unwrap();
    let app = router(service.clone());
    let inspect = call(&app, "GET", "/api/v1/models/q35/inspect", Value::Null, 200).await;
    for flag in [
        "images",
        "prefix_kv_reuse",
        "recurrent_state_reuse",
        "continuous_batching",
    ] {
        assert_eq!(inspect["compatibility"][flag], true);
    }
    let reference: Value = serde_json::from_slice(
        &std::fs::read(root.join("fixtures/qwen35/reference-v1.json")).unwrap(),
    )
    .unwrap();
    let cases = reference["cases"].as_array().unwrap();
    let body = |case: &Value| json!({"model":"q35","messages":case["messages"],"max_tokens":4,"context":256,"threads":1,"profile":"reference"});
    let (a, b) = tokio::join!(
        call(&app, "POST", "/api/v1/generate", body(&cases[0]), 200),
        call(&app, "POST", "/api/v1/generate", body(&cases[1]), 200)
    );
    for (result, case) in [(&a, &cases[0]), (&b, &cases[1])] {
        assert_eq!(result["token_ids"], case["greedy_token_ids"]);
        assert_eq!(result["text"], case["greedy_text"]);
        assert_eq!(result["batching"]["maximum_batch_width"], 2);
    }
    let created = call(
        &app,
        "POST",
        "/coheryn/v1/sessions",
        json!({"model":"q35","context":256}),
        200,
    )
    .await;
    let route = format!("/coheryn/v1/sessions/{}", created["id"].as_str().unwrap());
    let settings = json!({"max_tokens":4,"threads":1,"profile":"reference"});
    let first=call(&app,"POST",&format!("{route}/messages"),json!({"expected_revision":0,"idempotency_key":"one","content":cases[0]["messages"][0]["content"],"settings":settings}),200).await;
    assert_eq!(first["result"]["token_ids"], cases[0]["greedy_token_ids"]);
    let view = call(&app, "GET", &route, Value::Null, 200).await;
    assert!(view["cached_tokens"].as_u64().unwrap() > 0);
    assert!(view["cache_reserved_bytes"].as_u64().unwrap() > 0);
    let followup = "你好. বাংলা again.";
    let second=call(&app,"POST",&format!("{route}/messages"),json!({"expected_revision":2,"idempotency_key":"two","content":followup,"settings":settings}),200).await;
    assert!(second["result"]["reused_prompt_tokens"].as_u64().unwrap() > 0);
    let fresh=call(&app,"POST","/api/v1/generate",json!({"model":"q35","messages":[cases[0]["messages"][0],{"role":"assistant","content":first["result"]["text"]},{"role":"user","content":followup}],"max_tokens":4,"threads":1,"profile":"reference","context":256}),200).await;
    assert_eq!(second["result"]["token_ids"], fresh["token_ids"]);
    assert_eq!(second["result"]["text"], fresh["text"]);
    let exported = call(&app, "GET", &format!("{route}/export"), Value::Null, 200).await;
    assert_eq!(exported["session"]["cache_reserved_bytes"], 0);
    call(&app, "DELETE", &route, json!({"expected_revision":4}), 200).await;
    idle(&service).await;
    assert_eq!(service.budget.stats().session_buffer_capacity_bytes, 0);
    let image_reference_path = root.join("fixtures/qwen35/vision/reference-v1.json");
    let vision: Value =
        serde_json::from_slice(&std::fs::read(&image_reference_path).unwrap()).unwrap();
    let url = |name: &str| {
        format!(
            "data:image/png;base64,{}",
            STANDARD.encode(
                std::fs::read(root.join(format!("fixtures/qwen35/vision/{name}.png"))).unwrap()
            )
        )
    };
    let image_body = json!({"model":"q35","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":url("red")}},{"type":"text","text":"What color is the picture?"}]}],"max_tokens":4,"threads":1,"profile":"reference","context":256});
    let mut short = image_body.clone();
    short["context"] = json!(64);
    call(&app, "POST", "/v1/chat/completions", short, 413).await;
    let mut scarce = image_body.clone();
    scarce["max_memory"] = json!(1);
    call(&app, "POST", "/v1/chat/completions", scarce, 503).await;
    let mut malformed = image_body.clone();
    malformed["messages"][0]["content"][0]["image_url"]["url"] =
        json!("data:image/png;base64,AAAA");
    call(&app, "POST", "/v1/chat/completions", malformed, 400).await;
    let mut blocked = image_body.clone();
    blocked["messages"][0]["content"][0]["image_url"]["url"] =
        json!("https://example.com/image.png");
    call(&app, "POST", "/v1/chat/completions", blocked, 400).await;
    let image_result = call(&app, "POST", "/api/v1/generate", image_body.clone(), 200).await;
    assert_eq!(
        image_result["token_ids"],
        vision["cases"][0]["greedy_token_ids"]
    );
    assert_eq!(image_result["vision"]["images"], 1);
    let mut stream_body = image_body.clone();
    stream_body["stream"] = json!(true);
    stream_body["stream_options"] = json!({"include_usage":true});
    let response = app
        .clone()
        .oneshot(request("POST", "/v1/chat/completions", stream_body))
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let stream = to_bytes(response.into_body(), 16 << 20).await.unwrap();
    let stream = std::str::from_utf8(&stream).unwrap();
    assert_eq!(stream.matches("data: [DONE]").count(), 1);
    let frames: Vec<Value> = stream
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|l| *l != "[DONE]")
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let joined: String = frames
        .iter()
        .filter_map(|v| v["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(joined, vision["cases"][0]["greedy_text"]);
    // Drop an image response before computation completes. A healthy text lane must survive.
    let mut streaming = image_body;
    streaming["stream"] = json!(true);
    let response = app
        .clone()
        .oneshot(request("POST", "/v1/chat/completions", streaming))
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    drop(response);
    let healthy = call(&app, "POST", "/api/v1/generate", body(&cases[0]), 200).await;
    assert_eq!(healthy["token_ids"], cases[0]["greedy_token_ids"]);
    idle(&service).await;
    assert_eq!(service.budget.stats().session_buffer_capacity_bytes, 0);
    assert_eq!(service.budget.stats().reserved, 256 << 20);
    let status = service.status().unwrap();
    service.unload("q35").unwrap();
    service.remove("q35").unwrap();
    assert_eq!(service.budget.stats().reserved, 0);
    service.shutdown.cancel();
    let report = json!({"schema_version":1,"scope":"Native pinned checkpoint through authenticated in-process HTTP router; no sockets/network. Manifest import, hybrid cache parity, two-lane frozen-output batching, PNG API/SSE parity, resource/context/malformed/URL rejection, disconnect cleanup and healthy text survival. TCP/CLI client transport and broader image semantics remain separately gated.","model_manifest_sha256":digest(&manifest_path),"image_reference_sha256":digest(&image_reference_path),"test_executable_sha256":digest(&std::env::current_exe().unwrap()),"managed_record":record,"concurrent_results":[a,b],"cached_result":second,"fresh_result":fresh,"image_result":image_result,"stream_frames":frames.len(),"status_before_unload":status,"final_memory":service.budget.stats(),"cache_condition":"uncontrolled","external_executor_invoked":false});
    if let Some(output) = output {
        std::fs::write(output, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    }
}

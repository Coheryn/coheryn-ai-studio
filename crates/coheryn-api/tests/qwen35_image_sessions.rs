//! Opt-in native checkpoint, real loopback transport and CLI qualification.
use axum::body::Bytes;
use coheryn_api::{
    auth::Credentials,
    checkpoint::{CheckpointManifest, MAGIC},
    config::ServerConfig,
    http::router,
    service::Service,
};
use futures_util::{StreamExt, TryStreamExt, stream};
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};
use tokio_util::io::ReaderStream;
fn sha(p: &Path) -> String {
    coheryn_api::checkpoint::hash_file(p, &Default::default()).unwrap()
}
async fn start(
    config: ServerConfig,
) -> (
    Arc<Service>,
    String,
    tokio::task::JoinHandle<std::io::Result<()>>,
) {
    let s = Service::open(
        config,
        Credentials {
            inference_token: "i".repeat(64),
            admin_token: "a".repeat(64),
        },
    )
    .unwrap();
    let socket = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", socket.local_addr().unwrap());
    let shutdown = s.shutdown.clone();
    let app = router(s.clone());
    let task = tokio::spawn(async move {
        axum::serve(socket, app)
            .with_graceful_shutdown(shutdown.cancelled_owned())
            .await
    });
    (s, endpoint, task)
}
async fn call(endpoint: &str, method: &str, path: &str, value: Value, expected: u16) -> Value {
    let admin = path.starts_with("/api/") && path != "/api/v1/generate";
    let response = reqwest::Client::new()
        .request(method.parse().unwrap(), format!("{endpoint}{path}"))
        .bearer_auth(if admin {
            "a".repeat(64)
        } else {
            "i".repeat(64)
        })
        .json(&value)
        .send()
        .await
        .unwrap();
    let status = response.status();
    let v: Value = response.json().await.unwrap();
    assert_eq!(status.as_u16(), expected, "{path}: {v}");
    v
}
async fn cli(root: &Path, data: &Path, endpoint: &str, args: Vec<String>) -> String {
    let binary = root.join("target/release/coheryn");
    let data = data.to_owned();
    let endpoint = endpoint.to_owned();
    let empty = root.join(".cache/b4-empty-path");
    std::fs::create_dir_all(&empty).unwrap();
    let result = tokio::task::spawn_blocking(move || {
        std::process::Command::new(binary)
            .args([
                "--data-dir",
                data.to_str().unwrap(),
                "--endpoint",
                &endpoint,
            ])
            .args(args)
            .arg("--json")
            .env("COHERYN_API_TOKEN", "i".repeat(64))
            .env("COHERYN_ADMIN_TOKEN", "a".repeat(64))
            .env("PATH", empty)
            .output()
            .unwrap()
    })
    .await
    .unwrap();
    assert!(
        result.status.success(),
        "CLI: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    String::from_utf8(result.stdout).unwrap()
}
async fn idle(s: &Service) {
    for _ in 0..3000 {
        let v = s.status().unwrap();
        if v["scheduler"]["active"] == 0 && v["scheduler"]["queued"] == 0 {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("workers did not retire");
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires separately acquired pinned source, release CLI and loopback access"]
async fn qwen35_plans_and_durable_image_conversations_match_stateless_execution() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cli_sha = sha(&root.join("target/release/coheryn"));
    let output = std::env::var_os("COHERYN_Q35_IMAGE_SESSION_REPORT").map(PathBuf::from);
    if let Some(p) = &output {
        assert!(!p.exists(), "never overwrite qualification evidence");
    }
    let source = root.join(".cache/qwen3.5-0.8b-source");
    let model_manifest_path = root.join("fixtures/qwen35/model.json");
    let pins: Value =
        serde_json::from_slice(&std::fs::read(&model_manifest_path).unwrap()).unwrap();
    let (manifest, base) = CheckpointManifest::from_local(&source, &Default::default()).unwrap();
    for file in &manifest.files {
        let pin = pins["files"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["file"] == file.name)
            .unwrap();
        assert_eq!(file.sha256, pin["sha256"]);
        assert_eq!(file.bytes, pin["bytes"].as_u64().unwrap());
    }
    let temp = tempfile::tempdir_in(root.join(".cache")).unwrap();
    let mut config = ServerConfig {
        data_dir: temp.path().join("vault"),
        max_memory: 1 << 30,
        max_context: 512,
        max_concurrency: 1,
        generation_timeout_seconds: 600,
        ..Default::default()
    };
    config.persistence.enabled = true;
    let (s, endpoint, server) = start(config.clone()).await;
    let m = serde_json::to_vec(&manifest).unwrap();
    let mut prefix = MAGIC.to_vec();
    prefix.extend_from_slice(&(m.len() as u32).to_le_bytes());
    prefix.extend_from_slice(&m);
    let paths = manifest
        .files
        .iter()
        .map(|f| base.join(&f.name))
        .collect::<Vec<_>>();
    let files = stream::iter(paths)
        .then(|p| async move { tokio::fs::File::open(p).await.map(ReaderStream::new) })
        .try_flatten();
    let record = {
        let _admin = s.admin.lock().await;
        s.ingest_checkpoint(
            "q35".into(),
            Some(manifest.identity().unwrap()),
            stream::once(async move { Ok::<_, std::io::Error>(Bytes::from(prefix)) }).chain(files),
        )
        .await
        .unwrap()
    };
    let planning = json!({"model":"q35","task":"image-to-text","context":256,"concurrency":1,"profile":"reference","threads":1,"images":{"processor_profile":"qwen35-still-v1","images":[{"width":32,"height":32}]}});
    let before = s.budget.stats();
    let unloaded = call(
        &endpoint,
        "POST",
        "/api/v1/plans/estimate",
        planning.clone(),
        200,
    )
    .await;
    assert_eq!(unloaded["image_feature_tokens"], 64);
    assert_eq!(s.budget.stats().reserved, before.reserved);
    assert_eq!(s.budget.stats().session_buffer_capacity_bytes, 0);
    assert_eq!(s.loaded().unwrap(), json!([]));
    let local_args = vec![
        "plan".into(),
        source.display().to_string(),
        "--standalone".into(),
        "--task".into(),
        "image-to-text".into(),
        "--image".into(),
        root.join("fixtures/qwen35/vision/red.png")
            .display()
            .to_string(),
        "--context".into(),
        "256".into(),
        "--profile".into(),
        "reference".into(),
        "--threads".into(),
        "1".into(),
    ];
    let local: Value =
        serde_json::from_str(&cli(&root, &config.data_dir, &endpoint, local_args).await).unwrap();
    assert_eq!(
        local["plan"]["image_extra_reserved_bytes"],
        unloaded["image_extra_reserved_bytes"]
    );
    s.load("q35".into(), 256).await.unwrap();
    let loaded = call(&endpoint, "POST", "/api/v1/plans/estimate", planning, 200).await;
    assert_eq!(
        loaded["required_reserved_bytes"],
        unloaded["required_reserved_bytes"]
    );
    assert_eq!(loaded["model_metadata_additional"], 0);
    let remote: Value = serde_json::from_str(
        &cli(
            &root,
            &config.data_dir,
            &endpoint,
            vec![
                "plan".into(),
                "q35".into(),
                "--task".into(),
                "image-to-text".into(),
                "--image".into(),
                root.join("fixtures/qwen35/vision/red.png")
                    .display()
                    .to_string(),
                "--context".into(),
                "256".into(),
                "--profile".into(),
                "reference".into(),
                "--threads".into(),
                "1".into(),
            ],
        )
        .await,
    )
    .unwrap();
    assert_eq!(
        remote["image_extra_reserved_bytes"],
        loaded["image_extra_reserved_bytes"]
    );
    let text_session = call(
        &endpoint,
        "POST",
        "/coheryn/v1/sessions",
        json!({"model":"q35","context":256}),
        200,
    )
    .await;
    assert!(text_session.get("task").is_none());
    call(
        &endpoint,
        "DELETE",
        &format!(
            "/coheryn/v1/sessions/{}",
            text_session["id"].as_str().unwrap()
        ),
        json!({"expected_revision":0}),
        200,
    )
    .await;
    let prompt = "What color is the picture?";
    let streamed = cli(
        &root,
        &config.data_dir,
        &endpoint,
        vec![
            "chat".into(),
            "q35".into(),
            "--session".into(),
            "photos".into(),
            "--persist".into(),
            "--image".into(),
            root.join("fixtures/qwen35/vision/red.png")
                .display()
                .to_string(),
            "--prompt".into(),
            prompt.into(),
            "--context".into(),
            "256".into(),
            "--max-tokens".into(),
            "4".into(),
            "--profile".into(),
            "reference".into(),
            "--threads".into(),
            "1".into(),
            "--stream".into(),
        ],
    )
    .await;
    let frames: Vec<Value> = streamed
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let initial = frames.last().unwrap()["result"].clone();
    let reference: Value = serde_json::from_slice(
        &std::fs::read(root.join("fixtures/qwen35/vision/reference-v1.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        initial["token_ids"],
        reference["cases"][0]["greedy_token_ids"]
    );
    assert_eq!(
        initial["text"],
        frames
            .iter()
            .filter_map(|v| v["token"]["text"].as_str())
            .collect::<String>()
    );
    let route = "/coheryn/v1/sessions/photos";
    let history = call(&endpoint, "GET", route, Value::Null, 200).await;
    assert_eq!(history["task"], "image-to-text");
    assert_eq!(history["cached_tokens"], 0);
    assert_eq!(history["image_assets"].as_array().unwrap().len(), 1);
    let exported = call(
        &endpoint,
        "GET",
        &format!("{route}/export"),
        Value::Null,
        200,
    )
    .await;
    assert_eq!(exported["version"], 3);
    assert_eq!(exported["assets"].as_array().unwrap().len(), 1);
    let first_receipt = exported["turns"][0]["receipt"].clone();
    let first_settings = exported["turns"][0]["settings"].clone();
    let replay_body = json!({"expected_revision":1,"idempotency_key":exported["turns"][0]["key"],"content":prompt,"images":history["messages"][0]["images"],"settings":first_settings,"stream":true});
    idle(&s).await;
    s.unload("q35").unwrap();
    s.shutdown.cancel();
    server.await.unwrap().unwrap();
    drop(s);
    let (s, endpoint, server) = start(config.clone()).await;
    assert_eq!(s.loaded().unwrap(), json!([]));
    let restored = call(&endpoint, "GET", route, Value::Null, 200).await;
    assert_eq!(restored["messages"], history["messages"]);
    assert_eq!(
        call(
            &endpoint,
            "POST",
            &format!("{route}/messages"),
            replay_body,
            200
        )
        .await,
        first_receipt
    );
    s.load("q35".into(), 256).await.unwrap();
    let second_body = json!({"expected_revision":restored["revision"],"idempotency_key":"followup","content":"Name the color again.","settings":{"max_tokens":4,"threads":1,"profile":"reference"}});
    let request_file = temp.path().join("followup.json");
    std::fs::write(&request_file, serde_json::to_vec(&second_body).unwrap()).unwrap();
    let second: Value = serde_json::from_str(
        &cli(
            &root,
            &config.data_dir,
            &endpoint,
            vec![
                "sessions".into(),
                "submit".into(),
                "photos".into(),
                "--request".into(),
                request_file.display().to_string(),
            ],
        )
        .await,
    )
    .unwrap();
    assert_eq!(second["status"], "completed");
    assert_eq!(second["result"]["reused_prompt_tokens"], 0);
    assert_eq!(second["result"]["vision"]["images"], 1);
    let mut content = Vec::new();
    content.push(json!({"type":"image_url","image_url":{"url":coheryn_api::vision::data_url(include_bytes!("../../../fixtures/qwen35/vision/red.png"),&Default::default()).unwrap()}}));
    content.push(json!({"type":"text","text":prompt}));
    let fresh=call(&endpoint,"POST","/api/v1/generate",json!({"model":"q35","messages":[{"role":"user","content":content},{"role":"assistant","content":initial["text"]},{"role":"user","content":"Name the color again."}],"max_tokens":4,"context":256,"threads":1,"profile":"reference"}),200).await;
    assert_eq!(fresh["token_ids"], second["result"]["token_ids"]);
    assert_eq!(fresh["text"], second["result"]["text"]);
    assert_eq!(fresh["prompt_tokens"], second["context"]["prompt_tokens"]);
    let after = call(&endpoint, "GET", route, Value::Null, 200).await;
    assert_eq!(after["image_assets"], restored["image_assets"]);
    assert_eq!(after["message_count"], 4);
    let portable = call(
        &endpoint,
        "GET",
        &format!("{route}/export"),
        Value::Null,
        200,
    )
    .await;
    let imported = call(
        &endpoint,
        "POST",
        "/coheryn/v1/sessions/import",
        json!({"document":portable,"name":"copy","persist":true}),
        200,
    )
    .await;
    assert_eq!(
        imported["image_assets"][0]["sha256"],
        after["image_assets"][0]["sha256"]
    );
    call(
        &endpoint,
        "DELETE",
        "/coheryn/v1/sessions/copy",
        json!({"expected_revision":imported["revision"]}),
        200,
    )
    .await;
    let response=reqwest::Client::new().post(format!("{endpoint}{route}/messages")).bearer_auth("i".repeat(64)).json(&json!({"expected_revision":after["revision"],"idempotency_key":"cancel","content":"Continue.","stream":true,"settings":{"max_tokens":64,"threads":1,"profile":"reference"}})).send().await.unwrap();
    assert!(response.status().is_success());
    drop(response);
    idle(&s).await;
    let cancelled = tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let view = call(&endpoint, "GET", route, Value::Null, 200).await;
            if view["active_turn"].is_null() {
                break view;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("cancelled writer must acknowledge cleanup");
    let reset = call(
        &endpoint,
        "POST",
        &format!("{route}/reset"),
        json!({"expected_revision":cancelled["revision"]}),
        200,
    )
    .await;
    assert!(reset["image_assets"].as_array().is_none_or(Vec::is_empty));
    call(
        &endpoint,
        "DELETE",
        route,
        json!({"expected_revision":reset["revision"]}),
        200,
    )
    .await;
    idle(&s).await;
    assert_eq!(s.budget.stats().session_buffer_capacity_bytes, 0);
    s.unload("q35").unwrap();
    s.remove("q35").unwrap();
    assert_eq!(s.budget.stats().reserved, 16 << 20);
    assert_eq!(s.sessions.image_stats().unwrap()["disk_bytes"], 0);
    let final_memory = s.budget.stats();
    s.shutdown.cancel();
    server.await.unwrap().unwrap();
    drop(s);
    assert_eq!(
        sha(&root.join("target/release/coheryn")),
        cli_sha,
        "CLI must remain immutable during qualification"
    );
    let report = json!({"schema_version":1,"scope":"Pinned native Qwen3.5 source; real authenticated loopback HTTP and empty-PATH release CLI. Shared metadata-only image planning, image chat/SSE, durable no-resend followup, unloaded replay, stateless parity, portable original bytes, cancellation/reset/delete and final store-only reservation. Current host; broader semantic/backend/platform qualification remains open.","model_manifest_sha256":sha(&model_manifest_path),"binary_sha256":sha(&root.join("target/release/coheryn")),"test_binary_sha256":sha(&std::env::current_exe().unwrap()),"managed_record":record,"unloaded_plan":unloaded,"loaded_plan":loaded,"standalone_plan":local,"managed_cli_plan":remote,"initial_result":initial,"followup_receipt":second,"fresh_result":fresh,"final_memory":final_memory,"cache_condition":"uncontrolled","external_executor_invoked":false});
    if let Some(p) = output {
        std::fs::write(p, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    }
}

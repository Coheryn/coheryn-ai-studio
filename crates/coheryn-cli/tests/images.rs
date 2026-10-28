#[path = "../../coheryn-engine/tests/common/mod.rs"]
mod common;
#[path = "../../coheryn-engine/tests/common/vision.rs"]
mod vision_fixture;
use coheryn_api::{auth::Credentials, config::ServerConfig, service::Service};
use serde_json::{Value, json};
use std::{path::Path, process::Command};

fn run(endpoint: &str, data: &Path, args: &[&str]) -> Value {
    let out = Command::new(env!("CARGO_BIN_EXE_coheryn"))
        .env("PATH", "")
        .args(["--json", "--endpoint", endpoint, "--data-dir"])
        .arg(data)
        .args(args)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_cli_model_pair_plan_images_and_stream_match_standalone() {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("daemon");
    let service = Service::open(
        ServerConfig {
            data_dir: data.clone(),
            max_memory: 512 << 20,
            max_context: 512,
            max_model_bytes: 8 << 20,
            disk_quota: 32 << 20,
            ..Default::default()
        },
        Credentials::load_or_create(&data).unwrap(),
    )
    .unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let s = service.clone();
    let server = tokio::spawn(async move { coheryn_api::http::serve(listener, s).await.unwrap() });
    let language = dir.path().join("language.gguf");
    std::fs::write(&language, common::tiny_vlm_model()).unwrap();
    let vision = vision_fixture::component(256);
    let image = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/vlm/images/minimum.png");
    let image = image.to_str().unwrap();
    run(
        &endpoint,
        &data,
        &[
            "models",
            "import",
            language.to_str().unwrap(),
            "--name",
            "language",
        ],
    );
    run(
        &endpoint,
        &data,
        &[
            "models",
            "import",
            vision.path().to_str().unwrap(),
            "--name",
            "vision",
        ],
    );
    let pair = run(
        &endpoint,
        &data,
        &[
            "models",
            "register-pair",
            "--name",
            "pair",
            "--language",
            "language",
            "--vision",
            "vision",
        ],
    );
    assert_eq!(pair["components"]["vision"]["model"], "vision");
    let plan = run(
        &endpoint,
        &data,
        &[
            "plan",
            "pair",
            "--task",
            "image-to-text",
            "--context",
            "256",
            "--image",
            image,
        ],
    );
    assert_eq!(plan["image_feature_tokens"], 4);
    run(
        &endpoint,
        &data,
        &["models", "load", "pair", "--context", "256"],
    );
    let tail = [
        "--image",
        image,
        "--image",
        image,
        "--prompt",
        "Describe. বাংলা 中文 العربية 🙂",
        "--context",
        "256",
        "--max-tokens",
        "2",
        "--threads",
        "1",
        "--profile",
        "reference",
    ];
    let mut remote = vec!["chat", "pair"];
    remote.extend(tail);
    let result = run(&endpoint, &data, &remote);
    let mut local = vec![
        "chat",
        language.to_str().unwrap(),
        "--standalone",
        "--vision-model",
        vision.path().to_str().unwrap(),
    ];
    local.extend(tail);
    let standalone = run(&endpoint, &data, &local);
    assert_eq!(result["text"], standalone["text"]);
    assert_eq!(result["token_ids"], standalone["token_ids"]);
    assert_eq!(result["prompt_tokens"], standalone["prompt_tokens"]);
    remote.push("--stream");
    let out = Command::new(env!("CARGO_BIN_EXE_coheryn"))
        .env("PATH", "")
        .args(["--json", "--endpoint", &endpoint, "--data-dir"])
        .arg(&data)
        .args(remote)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let events: Vec<Value> = std::str::from_utf8(&out.stdout)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let text: String = events
        .iter()
        .filter_map(|v| v["token"]["text"].as_str())
        .collect();
    assert_eq!(json!(text), result["text"]);
    assert_eq!(
        events.last().unwrap()["result"]["token_ids"],
        result["token_ids"]
    );
    service.shutdown.cancel();
    server.await.unwrap();
}
#[test]
fn incompatible_image_controls_fail_before_model_loading() {
    let image = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/vlm/images/minimum.png");
    let out = Command::new(env!("CARGO_BIN_EXE_coheryn"))
        .args([
            "--json",
            "chat",
            "missing.gguf",
            "--standalone",
            "--vision-model",
            "missing-projector.gguf",
            "--image",
        ])
        .arg(image)
        .args(["--prompt", "Hello", "--temperature", "0.5"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let error: Value = serde_json::from_slice(&out.stderr).unwrap();
    assert_eq!(error["code"], "UNSUPPORTED_FEATURE");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn managed_image_cli_attaches_once_and_resumes_after_restart_without_resending() {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("daemon");
    let config = ServerConfig {
        data_dir: data.clone(),
        max_memory: 512 << 20,
        max_context: 512,
        max_model_bytes: 8 << 20,
        disk_quota: 32 << 20,
        persistence: coheryn_api::session_store::PersistenceLimits {
            enabled: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let s = Service::open(config.clone(), Credentials::load_or_create(&data).unwrap()).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let service = s.clone();
    let server =
        tokio::spawn(async move { coheryn_api::http::serve(listener, service).await.unwrap() });
    let language = dir.path().join("language.gguf");
    std::fs::write(&language, common::tiny_vlm_model()).unwrap();
    let vision = vision_fixture::component(256);
    let image = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/vlm/images/minimum.png");
    let image = image.to_str().unwrap();
    run(
        &endpoint,
        &data,
        &[
            "models",
            "import",
            language.to_str().unwrap(),
            "--name",
            "language",
        ],
    );
    run(
        &endpoint,
        &data,
        &[
            "models",
            "import",
            vision.path().to_str().unwrap(),
            "--name",
            "vision",
        ],
    );
    run(
        &endpoint,
        &data,
        &[
            "models",
            "register-pair",
            "--name",
            "pair",
            "--language",
            "language",
            "--vision",
            "vision",
        ],
    );
    run(
        &endpoint,
        &data,
        &["models", "load", "pair", "--context", "512"],
    );
    let first = run(
        &endpoint,
        &data,
        &[
            "chat",
            "pair",
            "--session",
            "saved",
            "--persist",
            "--image",
            image,
            "--prompt",
            "Describe this image.",
            "--context",
            "512",
            "--max-tokens",
            "2",
        ],
    );
    assert_eq!(first["revision"], 3);
    assert_eq!(first["result"]["vision"]["images"], 1);
    let second = run(
        &endpoint,
        &data,
        &[
            "sessions",
            "send",
            "saved",
            "--input",
            "What else? বাংলা 中文 العربية",
            "--expected-revision",
            "3",
            "--max-tokens",
            "2",
        ],
    );
    assert_eq!(second["revision"], 5);
    assert_eq!(second["result"]["vision"]["images"], 1);
    s.shutdown.cancel();
    server.await.unwrap();
    drop(s);
    let s = Service::open(config, Credentials::load_or_create(&data).unwrap()).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let service = s.clone();
    let server =
        tokio::spawn(async move { coheryn_api::http::serve(listener, service).await.unwrap() });
    run(
        &endpoint,
        &data,
        &["models", "load", "pair", "--context", "512"],
    );
    let resumed = run(
        &endpoint,
        &data,
        &[
            "sessions",
            "send",
            "saved",
            "--input",
            "Describe it again.",
            "--expected-revision",
            "5",
            "--max-tokens",
            "2",
        ],
    );
    assert_eq!(resumed["revision"], 7);
    assert_eq!(resumed["result"]["vision"]["images"], 1);
    let shown = run(&endpoint, &data, &["sessions", "show", "saved"]);
    assert_eq!(shown["image_assets"].as_array().unwrap().len(), 1);
    let out = Command::new(env!("CARGO_BIN_EXE_coheryn"))
        .env("PATH", "")
        .args(["--json", "--endpoint", &endpoint, "--data-dir"])
        .arg(&data)
        .args([
            "chat",
            "pair",
            "--session",
            "saved",
            "--resume",
            "--image",
            image,
            "--prompt",
            "Describe both attachments.",
            "--max-tokens",
            "2",
            "--stream",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let events: Vec<Value> = std::str::from_utf8(&out.stdout)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(events.last().unwrap()["result"]["vision"]["images"], 2);
    s.shutdown.cancel();
    server.await.unwrap();
}

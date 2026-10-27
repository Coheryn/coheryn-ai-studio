#[path = "../../coheryn-engine/tests/common/mod.rs"]
mod common;
use serde_json::Value;
use std::{
    io::{BufRead, BufReader, Write},
    path::Path,
    process::{Child, Command, Output, Stdio},
    time::Duration,
};
struct Daemon(Child);
impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn command(data: &Path, endpoint: &str, args: &[&str]) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_coheryn"));
    command
        .env_remove("COHERYN_API_TOKEN")
        .env_remove("COHERYN_ADMIN_TOKEN")
        .args(["--data-dir", data.to_str().unwrap(), "--endpoint", endpoint])
        .args(args);
    command
}
fn json(output: Output) -> Value {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}
fn start(data: &Path, persistent: bool) -> (Daemon, String, std::thread::JoinHandle<()>) {
    start_context(data, persistent, "512")
}
fn start_context(
    data: &Path,
    persistent: bool,
    context: &str,
) -> (Daemon, String, std::thread::JoinHandle<()>) {
    let mut arguments = vec!["serve", "--port", "0", "--max-context", context];
    if persistent {
        arguments.push("--session-persistence");
    }
    let mut daemon = Daemon(
        command(data, "http://127.0.0.1:1", &arguments)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
    );
    let stderr = daemon.0.stderr.take().unwrap();
    let (send, receive) = std::sync::mpsc::sync_channel(1);
    let reader = std::thread::spawn(move || {
        let mut sent = false;
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            if !sent
                && let Ok(value) = serde_json::from_str::<Value>(&line)
                && value["event"] == "listening"
            {
                let _ = send.send(value["address"].as_str().unwrap().to_owned());
                sent = true;
            }
        }
    });
    let endpoint = format!(
        "http://{}",
        receive
            .recv_timeout(Duration::from_secs(15))
            .expect("native daemon failed to start")
    );
    (daemon, endpoint, reader)
}
#[test]
fn cli_session_lifecycle_resume_streaming_and_idempotent_automation() {
    let directory = tempfile::tempdir().unwrap();
    let data = directory.path().join("service");
    let model = directory.path().join("chat.gguf");
    std::fs::write(&model, common::conversation_model()).unwrap();
    let (daemon, endpoint, reader) = start(&data, false);
    let invalid = command(
        &data,
        &endpoint,
        &[
            "chat",
            "chat",
            "--session",
            "project",
            "--resume",
            "--context",
            "512",
        ],
    )
    .output()
    .unwrap();
    assert_eq!(invalid.status.code(), Some(2));
    json(
        command(
            &data,
            &endpoint,
            &[
                "models",
                "import",
                model.to_str().unwrap(),
                "--name",
                "chat",
            ],
        )
        .output()
        .unwrap(),
    );
    json(
        command(
            &data,
            &endpoint,
            &["models", "load", "chat", "--context", "512"],
        )
        .output()
        .unwrap(),
    );
    let created = json(
        command(
            &data,
            &endpoint,
            &[
                "sessions",
                "create",
                "chat",
                "--name",
                "project",
                "--context",
                "512",
                "--system",
                "S",
            ],
        )
        .output()
        .unwrap(),
    );
    assert_eq!(created["revision"], 0);
    let args = [
        "sessions",
        "send",
        "project",
        "--input",
        "Bonjour 日本語 العربية",
        "--expected-revision",
        "0",
        "--idempotency-key",
        "first",
        "--max-tokens",
        "2",
    ];
    let first = json(command(&data, &endpoint, &args).output().unwrap());
    let replay = json(command(&data, &endpoint, &args).output().unwrap());
    assert_eq!(first, replay);
    assert_eq!(first["revision"], 2);
    let mut chat = command(
        &data,
        &endpoint,
        &[
            "chat",
            "chat",
            "--session",
            "project",
            "--resume",
            "--max-tokens",
            "2",
        ],
    )
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .spawn()
    .unwrap();
    chat.stdin
        .take()
        .unwrap()
        .write_all("مرحبا\n/exit\n".as_bytes())
        .unwrap();
    let output = chat.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "You: !!\nYou: ");
    let shown = json(
        command(&data, &endpoint, &["sessions", "show", "project"])
            .output()
            .unwrap(),
    );
    assert_eq!(shown["revision"], 4);
    assert_eq!(shown["messages"].as_array().unwrap().len(), 5);
    let conflict = command(
        &data,
        &endpoint,
        &["chat", "chat", "--session", "project", "--context", "512"],
    )
    .output()
    .unwrap();
    assert_eq!(conflict.status.code(), Some(7));
    let reset = json(
        command(
            &data,
            &endpoint,
            &["sessions", "reset", "project", "--expected-revision", "4"],
        )
        .output()
        .unwrap(),
    );
    assert_eq!(reset["revision"], 5);
    assert_eq!(reset["messages"].as_array().unwrap().len(), 1);
    json(
        command(
            &data,
            &endpoint,
            &["sessions", "delete", "project", "--expected-revision", "5"],
        )
        .output()
        .unwrap(),
    );
    assert_eq!(
        json(
            command(&data, &endpoint, &["sessions", "list"])
                .output()
                .unwrap()
        )["sessions"],
        serde_json::json!([])
    );
    drop(daemon);
    reader.join().unwrap();
}

#[test]
fn cli_persistence_survives_restart_and_atomic_exports_can_restore_a_new_session() {
    let directory = tempfile::tempdir().unwrap();
    let data = directory.path().join("service");
    let model = directory.path().join("chat.gguf");
    std::fs::write(&model, common::conversation_model()).unwrap();
    let (daemon, endpoint, reader) = start(&data, true);
    json(
        command(
            &data,
            &endpoint,
            &[
                "models",
                "import",
                model.to_str().unwrap(),
                "--name",
                "chat",
            ],
        )
        .output()
        .unwrap(),
    );
    json(
        command(
            &data,
            &endpoint,
            &["models", "load", "chat", "--context", "512"],
        )
        .output()
        .unwrap(),
    );
    json(
        command(
            &data,
            &endpoint,
            &[
                "sessions",
                "create",
                "chat",
                "--name",
                "durable",
                "--context",
                "512",
                "--system",
                "S",
                "--persist",
            ],
        )
        .output()
        .unwrap(),
    );
    let args = [
        "sessions",
        "send",
        "durable",
        "--input",
        "こんにちは العربية",
        "--expected-revision",
        "0",
        "--idempotency-key",
        "first",
        "--max-tokens",
        "2",
    ];
    let original = json(command(&data, &endpoint, &args).output().unwrap());
    let exported = directory.path().join("history.json");
    json(
        command(
            &data,
            &endpoint,
            &[
                "sessions",
                "export",
                "durable",
                "--output",
                exported.to_str().unwrap(),
            ],
        )
        .output()
        .unwrap(),
    );
    let before = std::fs::read(&exported).unwrap();
    assert_eq!(
        command(
            &data,
            &endpoint,
            &[
                "sessions",
                "export",
                "durable",
                "--output",
                exported.to_str().unwrap()
            ]
        )
        .output()
        .unwrap()
        .status
        .code(),
        Some(7)
    );
    assert_eq!(std::fs::read(&exported).unwrap(), before);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&exported).unwrap().permissions().mode() & 0o077,
            0
        );
    }
    drop(daemon);
    reader.join().unwrap();
    let (daemon, endpoint, reader) = start(&data, true);
    let restored = json(
        command(&data, &endpoint, &["sessions", "show", "durable"])
            .output()
            .unwrap(),
    );
    assert_eq!(restored["revision"], 2);
    assert_eq!(restored["cached_tokens"], 0);
    assert_eq!(
        json(command(&data, &endpoint, &args).output().unwrap()),
        original
    );
    let imported = json(
        command(
            &data,
            &endpoint,
            &[
                "sessions",
                "import",
                exported.to_str().unwrap(),
                "--name",
                "copy",
                "--persist",
            ],
        )
        .output()
        .unwrap(),
    );
    assert_eq!(imported["persistent"], true);
    assert_ne!(imported["id"], restored["id"]);
    json(
        command(
            &data,
            &endpoint,
            &["sessions", "reset", "durable", "--expected-revision", "2"],
        )
        .output()
        .unwrap(),
    );
    json(
        command(
            &data,
            &endpoint,
            &["sessions", "delete", "durable", "--expected-revision", "3"],
        )
        .output()
        .unwrap(),
    );
    drop(daemon);
    reader.join().unwrap();
    let (daemon, endpoint, reader) = start(&data, false);
    assert_eq!(
        json(
            command(&data, &endpoint, &["sessions", "list"])
                .output()
                .unwrap()
        )["sessions"],
        serde_json::json!([])
    );
    drop(daemon);
    reader.join().unwrap();
    let (daemon, endpoint, reader) = start(&data, true);
    assert_eq!(
        json(
            command(&data, &endpoint, &["sessions", "list"])
                .output()
                .unwrap()
        )["sessions"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    json(
        command(
            &data,
            &endpoint,
            &["models", "load", "chat", "--context", "512"],
        )
        .output()
        .unwrap(),
    );
    let mut chat = command(
        &data,
        &endpoint,
        &[
            "chat",
            "chat",
            "--session",
            "copy",
            "--resume",
            "--max-tokens",
            "2",
        ],
    )
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .spawn()
    .unwrap();
    chat.stdin
        .take()
        .unwrap()
        .write_all(b"again\n/exit\n")
        .unwrap();
    let result = chat.wait_with_output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(String::from_utf8(result.stdout).unwrap(), "You: !!\nYou: ");
    assert_eq!(
        json(
            command(&data, &endpoint, &["sessions", "show", "copy"])
                .output()
                .unwrap()
        )["revision"],
        4
    );
    drop(daemon);
    reader.join().unwrap();
}

#[test]
fn headless_tool_requests_and_structured_session_submission_work_through_cli() {
    use serde_json::json;
    let temp = tempfile::tempdir().unwrap();
    let data = temp.path().join("service");
    let model = temp.path().join("model.gguf");
    std::fs::write(&model, common::conversation_model()).unwrap();
    let (daemon, endpoint, reader) = start_context(&data, false, "1024");
    json(
        command(
            &data,
            &endpoint,
            &[
                "models",
                "import",
                model.to_str().unwrap(),
                "--name",
                "chat",
            ],
        )
        .output()
        .unwrap(),
    );
    json(
        command(
            &data,
            &endpoint,
            &["models", "load", "chat", "--context", "1024"],
        )
        .output()
        .unwrap(),
    );
    let tools =
        json!([{"type":"function","function":{"name":"add","parameters":{"type":"object"}}}]);
    let file = temp.path().join("request.json");
    std::fs::write(&file,serde_json::to_vec(&json!({"messages":[{"role":"user","content":"Hi"}],"tools":tools,"tool_choice":"none","context":1024,"max_tokens":1})).unwrap()).unwrap();
    let out = command(
        &data,
        &endpoint,
        &[
            "chat",
            "chat",
            "--request",
            file.to_str().unwrap(),
            "--stream",
            "--json",
        ],
    )
    .output()
    .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let rows: Vec<Value> = String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    assert_eq!(rows.last().unwrap()["result"]["text"], "!");
    json(
        command(
            &data,
            &endpoint,
            &[
                "sessions",
                "create",
                "chat",
                "--name",
                "work",
                "--context",
                "1024",
            ],
        )
        .output()
        .unwrap(),
    );
    std::fs::write(&file,serde_json::to_vec(&json!({"expected_revision":0,"idempotency_key":"one","content":"Hi","settings":{"tools":tools,"tool_choice":"none","max_tokens":1}})).unwrap()).unwrap();
    let receipt = json(
        command(
            &data,
            &endpoint,
            &[
                "sessions",
                "submit",
                "work",
                "--request",
                file.to_str().unwrap(),
            ],
        )
        .output()
        .unwrap(),
    );
    assert_eq!(receipt["revision"], 2);
    assert_eq!(receipt["result"]["text"], "!");
    std::fs::write(&file,serde_json::to_vec(&json!({"expected_revision":2,"idempotency_key":"bad","tool_results":[{"tool_call_id":"unknown","content":"4"}]})).unwrap()).unwrap();
    let out = command(
        &data,
        &endpoint,
        &[
            "sessions",
            "submit",
            "work",
            "--request",
            file.to_str().unwrap(),
        ],
    )
    .output()
    .unwrap();
    assert_eq!(out.status.code(), Some(2));
    let view = json(
        command(&data, &endpoint, &["sessions", "show", "work"])
            .output()
            .unwrap(),
    );
    assert_eq!(view["revision"], 2);
    assert!(view["active_turn"].is_null());
    json(
        command(
            &data,
            &endpoint,
            &["sessions", "delete", "work", "--expected-revision", "2"],
        )
        .output()
        .unwrap(),
    );
    json(
        command(&data, &endpoint, &["models", "unload", "chat"])
            .output()
            .unwrap(),
    );
    drop(daemon);
    reader.join().unwrap();
}

#[path = "../../coheryn-engine/tests/common/mod.rs"]
mod common;
use std::{
    io::Write,
    process::{Command, Stdio},
};

#[test]
fn native_standalone_execution_needs_no_executable_search_path_or_service() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("native.gguf");
    std::fs::write(&path, common::tiny_model()).unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_coheryn"));
    command
        .env_clear()
        .env("PATH", "")
        .env("HOME", directory.path());
    #[cfg(windows)]
    if let Some(root) = std::env::var_os("SystemRoot") {
        command.env("SystemRoot", root);
    }
    let result = command
        .args([
            "run",
            path.to_str().unwrap(),
            "--standalone",
            "--prompt",
            "Hi",
            "--context",
            "64",
            "--max-tokens",
            "2",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(value["text"], "!!");
}

#[test]
fn deep_validate_identifies_encoder_task_without_rendering_a_chat_template() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("encoder.gguf");
    std::fs::write(&path, common::embedding_model(8)).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_coheryn"))
        .args([
            "models",
            "validate",
            path.to_str().unwrap(),
            "--deep",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["task"], "embedding");
    assert_eq!(report["executable"], true);
}

#[test]
fn interactive_standalone_chat_and_integrated_bench_execute_real_engine() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    std::fs::write(&path, common::tiny_model()).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_coheryn"))
        .args([
            "chat",
            path.to_str().unwrap(),
            "--standalone",
            "--context",
            "64",
            "--max-tokens",
            "4",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"Hi\n/exit\n")
        .unwrap();
    let result = child.wait_with_output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "You: !!!!\nYou: "
    );
    let bench = Command::new(env!("CARGO_BIN_EXE_coheryn"))
        .args([
            "--json",
            "bench",
            "--model",
            path.to_str().unwrap(),
            "--standalone",
            "--context",
            "64",
            "--max-tokens",
            "2",
        ])
        .output()
        .unwrap();
    assert!(bench.status.success());
    let value: serde_json::Value = serde_json::from_slice(&bench.stdout).unwrap();
    assert_eq!(value["runs"][0]["text"], "!!");
    assert!(value["load_ms"].as_f64().unwrap() >= 0.);
}
#[cfg(unix)]
#[test]
fn interactive_chat_can_cancel_while_waiting_for_input() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    std::fs::write(&path, common::tiny_model()).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_coheryn"))
        .args([
            "chat",
            path.to_str().unwrap(),
            "--standalone",
            "--context",
            "64",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    use std::io::Read;
    let mut prompt = [0; 5];
    child
        .stdout
        .as_mut()
        .unwrap()
        .read_exact(&mut prompt)
        .unwrap();
    assert_eq!(&prompt, b"You: ");
    assert!(
        Command::new("kill")
            .args(["-INT", &child.id().to_string()])
            .status()
            .unwrap()
            .success()
    );
    let start = std::time::Instant::now();
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert_eq!(status.code(), Some(130));
            break;
        }
        if start.elapsed() > std::time::Duration::from_secs(5) {
            let _ = child.kill();
            panic!("chat did not cancel while waiting for input");
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

#[test]
fn headless_chat_request_streams_json_offline_and_rejects_unsupported_tools() {
    use serde_json::{Value, json};
    let temp = tempfile::tempdir().unwrap();
    let model = temp.path().join("model.gguf");
    std::fs::write(&model, common::tiny_model()).unwrap();
    let file = temp.path().join("request.json");
    let mut request =
        json!({"messages":[{"role":"user","content":"Hi"}],"context":64,"max_tokens":2});
    std::fs::write(&file, serde_json::to_vec(&request).unwrap()).unwrap();
    let run = || {
        let mut c = Command::new(env!("CARGO_BIN_EXE_coheryn"));
        c.env_remove("COHERYN_API_TOKEN")
            .env("PATH", "")
            .args([
                "chat",
                model.to_str().unwrap(),
                "--standalone",
                "--request",
                file.to_str().unwrap(),
                "--stream",
                "--json",
            ])
            .output()
            .unwrap()
    };
    let out = run();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let records: Vec<Value> = String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    assert_eq!(records.last().unwrap()["result"]["text"], "!!");
    assert_eq!(
        records.iter().filter(|r| r.get("token").is_some()).count(),
        2
    );
    request["tools"] =
        json!([{"type":"function","function":{"name":"add","parameters":{"type":"object"}}}]);
    std::fs::write(&file, serde_json::to_vec(&request).unwrap()).unwrap();
    let out = run();
    assert_eq!(out.status.code(), Some(4));
    assert!(String::from_utf8_lossy(&out.stderr).contains("template"));
    request["tools"][0]["function"]["strict"] = json!(true);
    std::fs::write(&file, serde_json::to_vec(&request).unwrap()).unwrap();
    let out = run();
    assert_eq!(out.status.code(), Some(4));
    assert!(String::from_utf8_lossy(&out.stderr).contains("strict"));
}

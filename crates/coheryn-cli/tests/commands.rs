#[path = "../../coheryn-engine/tests/common/mod.rs"]
mod common;
#[test]
fn schema_flags_constrain_native_output_and_reject_conflicts_before_loading() {
    use serde_json::{Value, json};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    let schema = dir.path().join("schema.json");
    std::fs::write(&path, common::tiny_model()).unwrap();
    std::fs::write(&schema,serde_json::to_vec(&json!({"type":"object","properties":{"answer":{"type":"string","const":"ok"}},"required":["answer"],"additionalProperties":false})).unwrap()).unwrap();
    let run = |extra: &[&str]| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_coheryn"));
        command
            .arg("--json")
            .arg("run")
            .arg(&path)
            .args([
                "--standalone",
                "--prompt",
                "json",
                "--context",
                "128",
                "--max-tokens",
                "64",
                "--response-format",
                "json_schema",
                "--schema",
            ])
            .arg(&schema)
            .args(extra);
        command.output().unwrap()
    };
    let output = run(&[]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(value["text"].as_str().unwrap()).unwrap(),
        json!({"answer":"ok"})
    );
    assert_eq!(value["constraint_validation"]["valid"], true);
    let output = run(&["--stop", "}"]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("custom stops"));
    let output = run(&["--max-memory", "16777216"]);
    assert_eq!(output.status.code(), Some(5));
    let output = run(&["--presence-penalty", "3"]);
    assert_eq!(output.status.code(), Some(2));
}
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

#[test]
fn local_inspection_validation_and_planning_share_metadata_without_a_daemon() {
    use serde_json::Value;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    std::fs::write(&path, common::tiny_model()).unwrap();
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_coheryn"))
            .env("PATH", "")
            .args(args)
            .output()
            .unwrap()
    };
    let p = path.to_str().unwrap();
    let inspected = run(&["models", "inspect", p, "--json"]);
    assert!(inspected.status.success());
    let i: Value = serde_json::from_slice(&inspected.stdout).unwrap();
    assert_eq!(i["version"], 3);
    assert_eq!(i["model_descriptor"]["format"], "gguf");
    assert_eq!(i["model_descriptor"]["weight_payloads"], "not-scanned");
    let validated = run(&["models", "validate", p, "--json"]);
    assert!(validated.status.success());
    let v: Value = serde_json::from_slice(&validated.stdout).unwrap();
    assert_eq!(v["executable"], true);
    assert_eq!(v["payload_scan_completed"], false);
    let plan = run(&[
        "plan",
        p,
        "--standalone",
        "--context",
        "64",
        "--profile",
        "reference",
        "--threads",
        "1",
        "--json",
    ]);
    assert!(
        plan.status.success(),
        "{}",
        String::from_utf8_lossy(&plan.stderr)
    );
    let value: Value = serde_json::from_slice(&plan.stdout).unwrap();
    assert_eq!(value["mode"], "standalone-metadata");
    assert_eq!(value["plan"]["admissible"], true);
    assert_eq!(
        value["model_descriptor"]["metadata_identity"],
        i["model_descriptor"]["metadata_identity"]
    );
    let rejected = run(&[
        "plan",
        p,
        "--standalone",
        "--context",
        "64",
        "--profile",
        "reference",
        "--max-memory",
        "1",
        "--json",
    ]);
    assert_eq!(rejected.status.code(), Some(5));
    assert_eq!(
        serde_json::from_slice::<Value>(&rejected.stdout).unwrap()["plan"]["admissible"],
        false
    );
    let wrong_task = run(&["plan", p, "--standalone", "--task", "embedding", "--json"]);
    assert_eq!(wrong_task.status.code(), Some(4));
}
#[test]
fn unsupported_safetensors_are_inspectable_and_validation_fails_with_structured_details() {
    use serde_json::{Value, json};
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    let h = json!({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}).to_string();
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&(h.len() as u64).to_le_bytes()).unwrap();
    f.write_all(h.as_bytes()).unwrap();
    f.write_all(&f32::NAN.to_le_bytes()).unwrap();
    drop(f);
    std::fs::write(dir.path().join("config.json"), r#"{"model_type":"future"}"#).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_coheryn"))
        .args(["models", "inspect", path.to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["format"], "safetensors");
    assert_eq!(report["container_valid"], true);
    assert_eq!(report["structural_status"], "unsupported");
    let output = Command::new(env!("CARGO_BIN_EXE_coheryn"))
        .args(["models", "validate", path.to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(4));
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["executable"], false);
    assert_eq!(report["payload_scan_completed"], false);
    assert!(String::from_utf8_lossy(&output.stderr).contains("UNSUPPORTED_MODEL_ARCHITECTURE"));
}

#[test]
fn managed_plan_defaults_still_require_credentials_rather_than_standalone_flags() {
    let dir = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_coheryn"))
        .env_remove("COHERYN_API_TOKEN")
        .env_remove("COHERYN_ADMIN_TOKEN")
        .arg("--data-dir")
        .arg(dir.path())
        .args(["plan", "managed-model", "--json"])
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(6),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

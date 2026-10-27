#[path = "../../coheryn-engine/tests/common/mod.rs"]
mod common;
use serde_json::{Value, json};
use std::{
    path::Path,
    process::{Command, Output},
};
fn command(model: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_coheryn"))
        .args(["--json", "embed", model.to_str().unwrap()])
        .args(args)
        .output()
        .unwrap()
}
fn model(dir: &Path) -> std::path::PathBuf {
    let path = dir.join("encoder.gguf");
    std::fs::write(&path, common::embedding_model(1)).unwrap();
    path
}
fn parse(output: &Output) -> Value {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}
#[test]
fn short_local_form_single_text_and_bounded_file_windows_preserve_vectors_usage_and_whitespace() {
    let dir = tempfile::tempdir().unwrap();
    let model = model(dir.path());
    let single = parse(&command(&model, &["--input", "hello world"]));
    let input = dir.path().join("documents.txt");
    std::fs::write(&input, "hello world\r\n café 中 \nhello world").unwrap();
    let result = parse(&command(
        &model,
        &["--file", input.to_str().unwrap(), "--request-size", "1"],
    ));
    assert_eq!(result["data"].as_array().unwrap().len(), 3);
    assert_eq!(result["usage"]["prompt_tokens"], 12);
    assert_eq!(
        result["data"][0]["embedding"],
        single["data"][0]["embedding"]
    );
    assert_eq!(
        result["data"][2]["embedding"],
        single["data"][0]["embedding"]
    );
    assert_eq!(result["data"][2]["index"], 2);
    assert_eq!(
        result["coheryn"]["embedding_space"],
        single["coheryn"]["embedding_space"]
    );
    assert!(result["coheryn"]["model_sha256"].as_str().unwrap().len() == 64);
}
#[test]
fn atomic_file_publication_overwrite_and_failed_late_records_leave_no_partial_output() {
    let dir = tempfile::tempdir().unwrap();
    let model = model(dir.path());
    let input = dir.path().join("input.txt");
    let target = dir.path().join("embeddings.json");
    std::fs::write(&input, "hello\nworld\n").unwrap();
    let run = || {
        command(
            &model,
            &[
                "--file",
                input.to_str().unwrap(),
                "--request-size",
                "1",
                "--output",
                target.to_str().unwrap(),
            ],
        )
    };
    assert!(run().status.success());
    let before = std::fs::read(&target).unwrap();
    assert_eq!(run().status.code(), Some(7));
    assert_eq!(std::fs::read(&target).unwrap(), before);
    std::fs::write(&input, "hello\n\n").unwrap();
    let failed = command(
        &model,
        &[
            "--file",
            input.to_str().unwrap(),
            "--request-size",
            "1",
            "--output",
            target.to_str().unwrap(),
            "--overwrite",
        ],
    );
    assert_eq!(failed.status.code(), Some(2));
    assert!(failed.stdout.is_empty());
    assert_eq!(std::fs::read(&target).unwrap(), before);
    assert!(
        !std::fs::read_dir(dir.path()).unwrap().any(|e| e
            .unwrap()
            .file_name()
            .to_string_lossy()
            .ends_with(".part"))
    );
    let no_file = dir.path().join("not-published.json");
    let failed = command(
        &model,
        &[
            "--file",
            input.to_str().unwrap(),
            "--request-size",
            "1",
            "--output",
            no_file.to_str().unwrap(),
        ],
    );
    assert!(!failed.status.success());
    assert!(!no_file.exists());
}
#[test]
fn output_quota_invalid_utf8_empty_records_and_explicit_truncation_are_bounded() {
    let dir = tempfile::tempdir().unwrap();
    let model = model(dir.path());
    let out = command(&model, &["--input", "hello", "--max-output-bytes", "32"]);
    assert_eq!(out.status.code(), Some(5));
    assert!(out.stdout.is_empty());
    let file = dir.path().join("input.txt");
    for bytes in [vec![0xff, 0x0a], vec![], vec![b'\n'], vec![b'a'; 65537]] {
        std::fs::write(&file, bytes).unwrap();
        let out = command(&model, &["--file", file.to_str().unwrap()]);
        assert_eq!(out.status.code(), Some(2));
        assert!(out.stdout.is_empty());
    }
    let out = command(
        &model,
        &["--input", "hello world", "--max-input-tokens", "3"],
    );
    assert_eq!(out.status.code(), Some(2));
    assert!(out.stdout.is_empty());
    let out = parse(&command(
        &model,
        &[
            "--input",
            "hello world",
            "--max-input-tokens",
            "3",
            "--truncate",
        ],
    ));
    assert_eq!(out["data"][0]["truncated"], true);
    assert_eq!(out["data"][0]["prompt_tokens"], 3);
}
#[test]
fn base64_export_and_embedding_benchmark_report_task_identity() {
    let dir = tempfile::tempdir().unwrap();
    let model = model(dir.path());
    let out = parse(&command(
        &model,
        &["--input", "hello", "--encoding-format", "base64"],
    ));
    let vector = coheryn_api::embeddings::Vector::Base64(
        out["data"][0]["embedding"].as_str().unwrap().into(),
    )
    .floats()
    .unwrap();
    assert_eq!(vector.len(), 32);
    let result = Command::new(env!("CARGO_BIN_EXE_coheryn"))
        .args([
            "--json",
            "bench",
            "--task",
            "embedding",
            "--model",
            model.to_str().unwrap(),
            "--standalone",
            "--batch-items",
            "2",
            "--context",
            "32",
            "--prompt",
            "hello",
        ])
        .output()
        .unwrap();
    let report = parse(&result);
    assert_eq!(report["task"], "embedding");
    assert_eq!(report["runs"][0]["items"], 2);
    assert_eq!(report["runs"][0]["prompt_tokens"], 6);
    assert_eq!(report["cache_condition"], "uncontrolled");
    assert_eq!(report["embedding_space"], out["coheryn"]["embedding_space"]);
    let invalid = Command::new(env!("CARGO_BIN_EXE_coheryn"))
        .args([
            "--json",
            "bench",
            "--task",
            "embedding",
            "--model",
            model.to_str().unwrap(),
            "--standalone",
            "--context",
            "4096",
        ])
        .output()
        .unwrap();
    assert_eq!(invalid.status.code(), Some(2)); // Explicit context is never silently reduced.
}
#[test]
fn input_contract_and_missing_daemon_never_fall_back_to_standalone() {
    let dir = tempfile::tempdir().unwrap();
    let model = model(dir.path());
    assert_eq!(
        command(&model, &["--input", "hello", "--file", "irrelevant.txt"])
            .status
            .code(),
        Some(2)
    );
    let result = Command::new(env!("CARGO_BIN_EXE_coheryn"))
        .args([
            "--json",
            "--endpoint",
            "http://127.0.0.1:1",
            "--data-dir",
            dir.path().to_str().unwrap(),
            "embed",
            "managed",
            "--input",
            "hello",
        ])
        .env("COHERYN_API_TOKEN", "i".repeat(64))
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(8));
    assert!(result.stdout.is_empty());
    let failure: Value = serde_json::from_slice(&result.stderr).unwrap();
    assert_eq!(failure["code"], json!("UNAVAILABLE"));
}

#[test]
fn model_switch_between_client_file_windows_aborts_atomic_publication() {
    use std::{
        io::{Read, Write},
        net::TcpListener,
        time::{Duration, Instant},
    };
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("records.txt");
    let target = dir.path().join("result.json");
    std::fs::write(&input, "hello\nworld\n").unwrap();
    std::fs::write(&target, "previous export").unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = std::thread::spawn(move || {
        for ch in ['a', 'b'] {
            let deadline = Instant::now() + Duration::from_secs(10);
            let mut socket = loop {
                match listener.accept() {
                    Ok((s, _)) => break s,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(
                            Instant::now() < deadline,
                            "client did not send the next bounded window"
                        );
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(e) => panic!("{e}"),
                }
            };
            socket.set_nonblocking(false).unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            let mut request = Vec::new();
            let mut buffer = [0u8; 1024];
            loop {
                let n = socket.read(&mut buffer).unwrap();
                assert!(n > 0);
                request.extend_from_slice(&buffer[..n]);
                if let Some(end) = request.windows(4).position(|p| p == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&request[..end]);
                    let length = headers
                        .lines()
                        .find_map(|line| {
                            let (key, value) = line.split_once(':')?;
                            key.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().unwrap())
                        })
                        .unwrap_or(0);
                    if request.len() >= end + 4 + length {
                        break;
                    }
                }
                assert!(request.len() < 65536);
            }
            let body = r#"{"object":"list","model":"remote","data":[{"object":"embedding","index":0,"embedding":[0.6,0.8]}],"usage":{"prompt_tokens":3,"total_tokens":3}}"#;
            let hash = ch.to_string().repeat(64);
            write!(socket,"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\nX-Coheryn-Embedding-Space: coheryn-embedding-v1-{}\r\nX-Coheryn-Model-Sha256: {}\r\nX-Coheryn-Embedding-Dimension: 2\r\nX-Coheryn-Token-Counts: 3\r\nX-Coheryn-Truncated-Indices: \r\nX-Coheryn-Batch-Size: 1\r\nX-Coheryn-Batch-Tokens: 512\r\n\r\n{}",body.len(),hash,hash,body).unwrap();
        }
    });
    let out = Command::new(env!("CARGO_BIN_EXE_coheryn"))
        .args([
            "--json",
            "--endpoint",
            &endpoint,
            "--data-dir",
            dir.path().to_str().unwrap(),
            "embed",
            "remote",
            "--file",
            input.to_str().unwrap(),
            "--request-size",
            "1",
            "--output",
            target.to_str().unwrap(),
            "--overwrite",
        ])
        .env("COHERYN_API_TOKEN", "i".repeat(64))
        .output()
        .unwrap();
    server.join().unwrap();
    assert_eq!(
        out.status.code(),
        Some(3),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out.stdout.is_empty());
    assert_eq!(std::fs::read_to_string(target).unwrap(), "previous export");
    assert!(
        !std::fs::read_dir(dir.path()).unwrap().any(|e| e
            .unwrap()
            .file_name()
            .to_string_lossy()
            .ends_with(".part"))
    );
}

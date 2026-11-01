#![cfg(all(feature = "metal", target_os = "macos"))]
//! Opt-in real executable checks; acquisition and host ownership belong to the runner.
use serde_json::{Value, json};
use std::{fs, path::Path, process::Command};

fn read(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

#[test]
#[ignore = "requires pinned pretrained assets, native Metal and the owned host guard"]
fn standalone_metal_cli_matches_frozen_outputs_and_rejects_unsupported_routes() {
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
            let source = root
                .join(manifest["cache_directory"].as_str().unwrap())
                .join(item["file"].as_str().unwrap());
            assert!(source.is_file() && !source.is_symlink());
            assert_eq!(
                source.metadata().unwrap().len(),
                item["bytes"].as_u64().unwrap()
            );
            assert_eq!(
                coheryn_engine::storage::sha256_file(&source).unwrap(),
                item["sha256"].as_str().unwrap()
            );
            (
                source,
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
        _ => panic!("explicit supported suite required"),
    };
    let directory = tempfile::tempdir().unwrap();
    let mut cases = Vec::new();
    let mut streams = Vec::new();
    for expected in reference["cases"].as_array().unwrap() {
        let messages = directory.path().join("messages.json");
        fs::write(&messages, serde_json::to_vec(&expected["messages"]).unwrap()).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_coheryn"))
            .current_dir(&root)
            .args(["--json", "run"])
            .arg(&source)
            .args(["--standalone", "--backend", "metal", "--messages"])
            .arg(&messages)
            .args([
                "--profile",
                "reference",
                "--context",
                "128",
                "--max-tokens",
                "4",
                "--threads",
                "1",
                "--max-memory",
                "2147483648",
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stdout.len() < 2 << 20);
        let result: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(result["token_ids"], expected["greedy_token_ids"]);
        assert_eq!(result["text"], expected["greedy_text"]);
        assert_eq!(
            result["prompt_tokens"].as_u64().unwrap() as usize,
            expected["prompt_token_ids"].as_array().unwrap().len()
        );
        assert_eq!(
            result["sampling_metadata"]["backend"],
            "native-metal-matmul-f32-reference"
        );
        cases.push(result);
        let streamed = Command::new(env!("CARGO_BIN_EXE_coheryn"))
            .current_dir(&root)
            .arg("run")
            .arg(&source)
            .args(["--standalone", "--backend", "metal", "--prompt"])
            .arg(expected["prompt"].as_str().unwrap())
            .args([
                "--stream",
                "--profile",
                "reference",
                "--context",
                "128",
                "--max-tokens",
                "4",
                "--threads",
                "1",
                "--max-memory",
                "2147483648",
            ])
            .output()
            .unwrap();
        assert!(
            streamed.status.success(),
            "{}",
            String::from_utf8_lossy(&streamed.stderr)
        );
        let text = String::from_utf8(streamed.stdout).unwrap();
        assert_eq!(text, format!("{}\n", expected["greedy_text"].as_str().unwrap()));
        streams.push(text);
    }
    let mut rejected = Vec::new();
    for arguments in [
        vec!["run", "missing.gguf", "--backend", "metal", "--prompt", "hello"],
        vec![
            "chat",
            "missing.gguf",
            "--standalone",
            "--backend",
            "metal",
            "--image",
            "missing.png",
        ],
        vec![
            "chat",
            "missing.gguf",
            "--standalone",
            "--backend",
            "metal",
            "--request",
            "missing.json",
        ],
        vec!["chat", "missing.gguf", "--standalone", "--backend", "metal", "--session", "test"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_coheryn"))
            .current_dir(directory.path())
            .arg("--json")
            .args(&arguments)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        let error: Value = serde_json::from_slice(&output.stderr).unwrap();
        let expected = if arguments[0] == "run" {
            "INVALID_REQUEST"
        } else {
            "UNSUPPORTED_FEATURE"
        };
        assert_eq!(error["code"], expected);
        assert_eq!(
            output.status.code(),
            Some(if arguments[0] == "run" { 2 } else { 4 })
        );
        rejected.push(json!({"arguments":arguments,"error":error,"exit":output.status.code()}));
    }
    let report = json!({"schema_version":1,"outcome":"pass","suite":suite,"cases":cases,"streams":streams,"rejected":rejected,
        "scope":"Standalone Metal executable outputs and unsupported-route errors only; no managed API, performance or production qualification."});
    let destination = std::env::var("COHERYN_METAL_CLI_REPORT").unwrap();
    use std::io::Write;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)
        .unwrap();
    file.write_all(serde_json::to_string_pretty(&report).unwrap().as_bytes())
        .unwrap();
}

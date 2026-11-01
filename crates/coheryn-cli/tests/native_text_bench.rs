//! Explicit pinned-model benchmark qualification; ordinary tests never acquire models.
use serde_json::{Value, json};
use std::{fs, path::Path, process::Command};

fn read(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

fn verify(root: &Path, path: &Path, expected: &Value, inputs: &mut Vec<Value>) {
    assert!(
        path.is_file() && !path.is_symlink(),
        "missing input: {}",
        path.display()
    );
    assert_eq!(
        fs::metadata(path).unwrap().len(),
        expected["bytes"].as_u64().unwrap()
    );
    let digest = coheryn_engine::storage::sha256_file(path).unwrap();
    assert_eq!(digest, expected["sha256"].as_str().unwrap());
    inputs.push(json!({"path":path.strip_prefix(root).unwrap().to_str().unwrap().replace('\\', "/"),"sha256":digest,"bytes":expected["bytes"]}));
}

#[test]
#[ignore = "requires pinned model assets and an owned host-guarded runner"]
fn native_text_adapters_benchmark_with_frozen_outputs_and_repeated_fresh_sessions() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let suite =
        std::env::var("COHERYN_TEXT_BENCH_SUITE").expect("explicit benchmark suite required");
    let mut inputs = Vec::new();
    let (models, reference, architecture, format) = match suite.as_str() {
        "gguf-text" => {
            let manifest = read(&root.join("fixtures/models.json"));
            let item = &manifest["models"][0];
            let path = root
                .join(".cache/models")
                .join(item["file"].as_str().unwrap());
            verify(&root, &path, item, &mut inputs);
            let reference = read(&root.join("fixtures/golden/qwen2.5-0.5b-instruct-fp16.json"));
            assert_eq!(reference["model_sha256"], item["sha256"]);
            (vec![path], reference, "qwen2", "gguf")
        }
        "qwen35-gguf-q4_k_m-text" | "qwen35-text" => {
            let manifest = read(&root.join("fixtures/qwen35/model.json"));
            let source = root.join(manifest["cache_directory"].as_str().unwrap());
            if suite == "qwen35-text" {
                for item in manifest["files"].as_array().unwrap() {
                    verify(
                        &root,
                        &source.join(item["file"].as_str().unwrap()),
                        item,
                        &mut inputs,
                    );
                }
                let weight = manifest["files"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|item| item["file"].as_str().unwrap().ends_with(".safetensors"))
                    .unwrap();
                let reference = read(&root.join("fixtures/qwen35/reference-v1.json"));
                assert_eq!(reference["model_revision"], manifest["revision"]);
                (
                    vec![
                        source.clone(),
                        source.join("model.safetensors.index.json"),
                        source.join(weight["file"].as_str().unwrap()),
                    ],
                    reference,
                    "qwen3_5",
                    "safetensors",
                )
            } else {
                let gguf = read(&root.join("fixtures/qwen35/gguf/models-v1.json"));
                let directory = root.join(gguf["cache_directory"].as_str().unwrap());
                for item in manifest["files"].as_array().unwrap().iter().filter(|item| {
                    matches!(
                        item["file"].as_str().unwrap(),
                        "LICENSE"
                            | "config.json"
                            | "tokenizer.json"
                            | "tokenizer_config.json"
                            | "chat_template.jinja"
                            | "preprocessor_config.json"
                    )
                }) {
                    verify(
                        &root,
                        &directory.join(item["file"].as_str().unwrap()),
                        item,
                        &mut inputs,
                    );
                }
                let item = gguf["files"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|item| item["role"] == "language" && item["label"] == "q4_k_m")
                    .unwrap();
                let path = directory.join(item["file"].as_str().unwrap());
                verify(&root, &path, item, &mut inputs);
                let reference =
                    read(&root.join("fixtures/qwen35/gguf/q4_k_m-text.reference-v1.json"));
                assert_eq!(
                    reference["provenance"]["files"][0]["sha256"],
                    item["sha256"]
                );
                (vec![path], reference, "qwen35", "gguf")
            }
        }
        _ => panic!("unsupported benchmark suite"),
    };
    let available = std::thread::available_parallelism().unwrap().get();
    assert!(available >= 2, "qualification requires workers one and two");
    let threads = if available >= 4 {
        vec![1, 2, 4]
    } else {
        vec![1, 2]
    };
    let temporary = tempfile::tempdir().unwrap();
    let mut runs = Vec::new();
    for (entry, model) in models.iter().enumerate() {
        for case in reference["cases"].as_array().unwrap().iter().take(2) {
            let expected = if architecture == "qwen2" {
                &case["generated_token_ids"]
            } else {
                &case["greedy_token_ids"]
            };
            let expected_text = if architecture == "qwen2" {
                &case["text"]
            } else {
                &case["greedy_text"]
            };
            for workers in &threads {
                let mut command = Command::new(env!("CARGO_BIN_EXE_coheryn"));
                command.env_clear();
                for key in ["SystemRoot", "WINDIR", "TMP", "TEMP", "TMPDIR"] {
                    if let Some(value) = std::env::var_os(key) {
                        command.env(key, value);
                    }
                }
                let result = command
                    .env("PATH", "")
                    .env("COHERYN_ENDPOINT", "http://127.0.0.1:1")
                    .arg("--json")
                    .arg("--data-dir")
                    .arg(temporary.path())
                    .arg("bench")
                    .arg("--model")
                    .arg(model)
                    .arg("--standalone")
                    .args([
                        "--context",
                        "256",
                        "--max-memory",
                        "1073741824",
                        "--profile",
                        "balanced",
                        "--iterations",
                        "2",
                        "--threads",
                        &workers.to_string(),
                        "--max-tokens",
                        &expected.as_array().unwrap().len().to_string(),
                        "--prompt",
                        case["prompt"].as_str().unwrap(),
                    ])
                    .output()
                    .unwrap();
                assert!(
                    result.status.success(),
                    "{}",
                    String::from_utf8_lossy(&result.stderr)
                );
                let value: Value = serde_json::from_slice(&result.stdout).unwrap();
                assert_eq!(value["mode"], "standalone");
                assert_eq!(value["adapter"]["architecture"], architecture);
                assert_eq!(value["adapter"]["format"], format);
                assert_eq!(value["cache_condition"], "uncontrolled");
                assert_eq!(value["requested_threads"], *workers);
                assert_eq!(value["runs"].as_array().unwrap().len(), 2);
                assert!(value["load_ms"].as_f64().unwrap().is_finite());
                for run in value["runs"].as_array().unwrap() {
                    assert_eq!(&run["token_ids"], expected);
                    assert_eq!(&run["text"], expected_text);
                    assert_eq!(
                        run["prompt_tokens"].as_u64().unwrap() as usize,
                        case["prompt_token_ids"].as_array().unwrap().len()
                    );
                    assert_eq!(
                        run["completion_tokens"].as_u64().unwrap() as usize,
                        expected.as_array().unwrap().len()
                    );
                    assert_eq!(run["reused_prompt_tokens"], 0);
                    for field in [
                        "prefill_ms",
                        "decode_ms",
                        "tokenization_ms",
                        "output_delivery_ms",
                        "first_token_ms",
                    ] {
                        let timing = run[field].as_f64().expect("required timing");
                        assert!(timing.is_finite() && timing >= 0.);
                    }
                    assert!(run["memory"]["peak_reserved"].as_u64().unwrap() <= 1 << 30);
                }
                runs.push(json!({"entrypoint":entry,"case":case["id"],"workers":workers,"benchmark":value}));
            }
        }
    }
    for item in &inputs {
        assert_eq!(
            coheryn_engine::storage::sha256_file(&root.join(item["path"].as_str().unwrap()))
                .unwrap(),
            item["sha256"].as_str().unwrap()
        );
    }
    let report =
        std::env::var_os("COHERYN_TEXT_BENCH_REPORT").expect("exclusive benchmark report required");
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(report)
        .unwrap();
    serde_json::to_writer_pretty(file, &json!({"schema_version":1,"suite":suite,"outcome":"pass",
        "architecture":std::env::consts::ARCH,"os":std::env::consts::OS,"available_cpus":available,
        "threads":threads,"inputs":inputs,"source_entrypoints":models.len(),"cases":2,"runs":runs,
        "scope":"Pinned greedy/tokenizer and native benchmark routing; repeated fresh sessions. Timing observations under uncontrolled cache conditions; no full-array, performance-SLA or production qualification."})).unwrap();
}

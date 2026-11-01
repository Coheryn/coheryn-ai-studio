#![cfg(all(feature = "metal", target_os = "macos"))]
//! Opt-in pretrained GPU graph checks. No acquisition or external executor here.
use coheryn_engine::{
    Cancellation, GenerationOptions, ResourceLimits, memory::Budget, qwen35::Model,
};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
};

fn read(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}
fn verify(path: &Path, identity: &Value) {
    assert!(path.is_file() && !path.is_symlink());
    assert_eq!(
        path.metadata().unwrap().len(),
        identity["bytes"].as_u64().unwrap()
    );
    assert_eq!(
        coheryn_engine::storage::sha256_file(path).unwrap(),
        identity["sha256"].as_str().unwrap()
    );
}
fn compare(actual: &[f32], expected: &[f32], absolute: f64, relative: f64, l2: f64) {
    assert_eq!(actual.len(), expected.len());
    let (mut squared, mut scale) = (0., 0.);
    for (&actual, &expected) in actual.iter().zip(expected) {
        let (actual, expected) = (f64::from(actual), f64::from(expected));
        assert!(actual.is_finite() && expected.is_finite());
        assert!(
            (actual - expected).abs() <= absolute + relative * expected.abs(),
            "{actual} != {expected}"
        );
        squared += (actual - expected).powi(2);
        scale += expected.powi(2);
    }
    assert!((squared / scale.max(1e-30)).sqrt() <= l2);
}
fn sampled(actual: &[f32], expected: &Value, absolute: f64, relative: f64) -> usize {
    assert_eq!(actual.len(), expected["length"].as_u64().unwrap() as usize);
    let indices = expected["indices"].as_array().unwrap();
    let values = expected["values"].as_array().unwrap();
    assert_eq!(indices.len(), values.len());
    for (index, expected) in indices.iter().zip(values) {
        let actual = f64::from(actual[index.as_u64().unwrap() as usize]);
        let expected = expected.as_f64().unwrap();
        assert!(
            actual.is_finite() && (actual - expected).abs() <= absolute + relative * expected.abs()
        );
    }
    indices.len()
}

#[test]
#[ignore = "requires pinned pretrained assets, native Metal and the owned host guard"]
fn native_qwen35_gpu_graph_matches_frozen_answers_and_cpu_arrays() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let suite = std::env::var("COHERYN_METAL_MODEL_SUITE").expect("explicit pinned suite required");
    let manifest = read(&root.join("fixtures/qwen35/model.json"));
    let (source, reference) = match suite.as_str() {
        "qwen35-gguf-q4_k_m-text" => {
            let gguf = read(&root.join("fixtures/qwen35/gguf/models-v1.json"));
            let folder = root.join(gguf["cache_directory"].as_str().unwrap());
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
                verify(&folder.join(item["file"].as_str().unwrap()), item);
            }
            let identity = gguf["files"]
                .as_array()
                .unwrap()
                .iter()
                .find(|item| item["role"] == "language" && item["label"] == "q4_k_m")
                .unwrap();
            let source = folder.join(identity["file"].as_str().unwrap());
            verify(&source, identity);
            let reference = read(&root.join("fixtures/qwen35/gguf/q4_k_m-text.reference-v1.json"));
            assert_eq!(
                reference["provenance"]["files"][0]["sha256"],
                identity["sha256"]
            );
            (source, reference)
        }
        "qwen35-text" => {
            let source = root.join(manifest["cache_directory"].as_str().unwrap());
            for item in manifest["files"].as_array().unwrap() {
                verify(&source.join(item["file"].as_str().unwrap()), item);
            }
            let reference = read(&root.join("fixtures/qwen35/reference-v1.json"));
            assert_eq!(reference["model_revision"], manifest["revision"]);
            (source, reference)
        }
        _ => panic!("unsupported model suite"),
    };
    let absolute = reference["tolerances"]["logits_absolute"].as_f64().unwrap();
    let relative = reference["tolerances"]["logits_relative"].as_f64().unwrap();
    let l2 = reference["tolerances"]["normalized_l2"].as_f64().unwrap();
    let limits = ResourceLimits {
        context: 128,
        max_memory: 2 << 30,
    };
    let budget = Budget::new(limits.max_memory);
    let cpu =
        Model::load_with_budget(&source, limits, budget.clone(), &Cancellation::default()).unwrap();
    let first = &reference["cases"][0];
    let ids: Vec<_> = first["prompt_token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|id| id.as_u64().unwrap() as u32)
        .collect();
    let mut arrays = Vec::new();
    cpu.trace(
        &ids[..2],
        1,
        &Cancellation::default(),
        &mut |position, name, values| {
            arrays.push((position, name.to_string(), values.to_vec()));
            Ok(())
        },
    )
    .unwrap();
    let model = cpu.enable_metal(&Cancellation::default()).unwrap();
    let baseline = budget.stats().reserved;
    let hardware = coheryn_engine::planner::HardwareSnapshot {
        os: std::env::consts::OS.into(),
        architecture: std::env::consts::ARCH.into(),
        logical_cpus: 1,
        cpu_features: vec![],
        ram_total_bytes: None,
        ram_available_bytes: None,
        swap_total_bytes: None,
        swap_used_bytes: None,
        observed_unix_ms: 0,
        gpu_available_bytes: None,
        gpu_backend_qualified: false,
    };
    let info = model.metal_info().unwrap().unwrap();
    let prepared = model
        .prepare_generation(
            GenerationOptions {
                max_tokens: 4,
                ..Default::default()
            },
            128,
            coheryn_engine::adapters::Execution {
                profile: coheryn_engine::planner::Profile::Reference,
                threads: Some(1),
                hardware: &hardware,
            },
            1,
        )
        .unwrap();
    assert_eq!(prepared.plan().backend, model.execution_backend());
    assert_eq!(
        prepared.plan().gpu_allocation_bytes,
        info.encoded_weight_bytes + info.request_workspace_bytes
    );
    assert_eq!(
        prepared.plan().session.total_bytes,
        model
            .configuration()
            .session_estimate(128, 1)
            .unwrap()
            .total_bytes
            + info.request_workspace_bytes
    );
    assert_eq!(
        budget.stats().reserved,
        baseline + prepared.request_reserved_bytes()
    );
    drop(prepared);
    assert_eq!(budget.stats().reserved, baseline);
    let mut checked_arrays = 0;
    let mut checked_elements = 0;
    model
        .trace(
            &ids[..2],
            1,
            &Cancellation::default(),
            &mut |position, name, values| {
                let (expected_position, expected_name, expected) = &arrays[checked_arrays];
                assert_eq!(
                    (position, name),
                    (*expected_position, expected_name.as_str())
                );
                compare(values, expected, absolute, relative, l2);
                checked_arrays += 1;
                checked_elements += values.len();
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(checked_arrays, arrays.len());
    drop(arrays);
    let mut sampled_values = 0;
    let mut results = Vec::new();
    for case in reference["cases"].as_array().unwrap() {
        let prompt = case["prompt"].as_str().unwrap();
        let mut ids = model.encode(prompt).unwrap();
        let prompt_tokens = ids.len();
        assert_eq!(
            serde_json::to_value(&ids).unwrap(),
            case["prompt_token_ids"]
        );
        let greedy: Vec<_> = case["greedy_token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        ids.extend(&greedy[..greedy.len() - 1]);
        model
            .trace(
                &ids,
                1,
                &Cancellation::default(),
                &mut |position, name, values| {
                    if position == prompt_tokens - 1 && name.starts_with("layer.") {
                        let expected = case["checkpoints"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .find(|value| value["name"].as_str().unwrap().ends_with(name))
                            .unwrap();
                        sampled_values += sampled(values, expected, absolute, relative);
                    }
                    if name == "logits" && position >= prompt_tokens - 1 {
                        sampled_values += sampled(
                            values,
                            &case["teacher_forced"][position + 1 - prompt_tokens],
                            absolute,
                            relative,
                        );
                    }
                    Ok(())
                },
            )
            .unwrap();
        let result = model
            .generate(
                prompt,
                &GenerationOptions {
                    max_tokens: greedy.len(),
                    ..Default::default()
                },
                1,
                &Cancellation::default(),
                &mut |_| Ok(()),
            )
            .unwrap();
        assert_eq!(result.token_ids, greedy);
        assert_eq!(result.text, case["greedy_text"].as_str().unwrap());
        assert_eq!(
            result.sampling_metadata.as_ref().unwrap().backend,
            model.execution_backend()
        );
        assert_eq!(budget.stats().reserved, baseline);
        assert_eq!(budget.stats().session_buffer_capacity_bytes, 0);
        results.push(result);
    }
    let error = model
        .trace(&ids[..2], 1, &Cancellation::default(), &mut |_, _, _| {
            Err(coheryn_engine::Error::request("controlled trace consumer"))
        })
        .unwrap_err();
    assert_eq!(error.code, coheryn_engine::ErrorCode::InvalidRequest);
    assert_eq!(budget.stats().reserved, baseline);
    assert_eq!(budget.stats().session_buffer_capacity_bytes, 0);
    let cancelled = Cancellation::default();
    let error = model
        .trace(&ids[..2], 1, &cancelled, &mut |_, _, _| {
            cancelled.cancel();
            Ok(())
        })
        .unwrap_err();
    assert_eq!(error.code, coheryn_engine::ErrorCode::Cancelled);
    assert_eq!(budget.stats().reserved, baseline);
    assert_eq!(budget.stats().session_buffer_capacity_bytes, 0);
    let info = model.metal_info().unwrap().unwrap();
    assert!(info.completed_model_dispatches > 0 && info.encoded_weight_bytes > 0);
    assert!(sampled_values > 0 && checked_elements > 0);
    assert!(model.vision_configuration().is_err());
    drop(model);
    assert_eq!(budget.stats().reserved, 0);
    let output = PathBuf::from(
        std::env::var_os("COHERYN_METAL_MODEL_REPORT").expect("explicit report path required"),
    );
    fs::write(output, serde_json::to_vec_pretty(&json!({"schema_version":1,"outcome":"pass","suite":suite,"source":source,
        "scope":"Selected native library text graph: frozen sampled checkpoints/teacher-forced logits and greedy answers; full CPU/GPU arrays at two positions; no CLI/API, throughput or full release claim",
        "full_cpu_gpu_arrays":checked_arrays,"full_cpu_gpu_elements":checked_elements,"independent_sampled_values":sampled_values,
        "cases":results,"gpu":info,"memory_after_unload":budget.stats(),
        "trace_failure_cleanup":true,"in_graph_cancellation_cleanup":true,"unqualified_image_profile_rejected":true,
        "gpu_plan_prepaid":true})).unwrap()).unwrap();
}

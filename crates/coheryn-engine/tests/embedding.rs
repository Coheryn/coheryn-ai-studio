mod common;
use coheryn_engine::{
    Cancellation, Error, ErrorCode, Model, ResourceLimits,
    embedding::{self, EmbeddingModel, EmbeddingOptions, EmbeddingPlanInput, EmbeddingTokenizer},
    gguf::{self, Value},
    memory::{Budget, MemoryStats},
    planner::{HardwareSnapshot, Profile},
};
use std::io::Cursor;

fn hardware() -> HardwareSnapshot {
    HardwareSnapshot {
        os: "fixture".into(),
        architecture: "fixture".into(),
        logical_cpus: 8,
        cpu_features: vec![],
        ram_total_bytes: Some(8 << 30),
        ram_available_bytes: Some(4 << 30),
        swap_total_bytes: None,
        swap_used_bytes: None,
        observed_unix_ms: 1,
        gpu_available_bytes: None,
        gpu_backend_qualified: false,
    }
}
fn options() -> EmbeddingOptions {
    EmbeddingOptions {
        max_input_tokens: 32,
        max_batch_tokens: 128,
        profile: Profile::Reference,
        threads: Some(1),
        ..Default::default()
    }
}
fn fixture(dtype: u32) -> tempfile::NamedTempFile {
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), common::embedding_model(dtype)).unwrap();
    file
}
fn load(file: &tempfile::NamedTempFile) -> EmbeddingModel {
    EmbeddingModel::load(
        file.path(),
        ResourceLimits {
            context: 512,
            max_memory: 64 << 20,
        },
    )
    .unwrap()
}
fn descriptor() -> gguf::Descriptor {
    let bytes = common::embedding_model(1);
    gguf::parse(Cursor::new(&bytes), bytes.len() as u64, Default::default()).unwrap()
}
fn error<T>(result: coheryn_engine::Result<T>) -> Error {
    match result {
        Err(e) => e,
        Ok(_) => panic!("expected failure"),
    }
}

#[test]
fn encoder_is_independent_and_generation_rejects_its_task() {
    let file = fixture(1);
    let model = load(&file);
    let report = coheryn_engine::validate(model.descriptor());
    assert!(report.executable);
    assert!(report.config.is_none());
    assert_eq!(report.embedding_config.unwrap().hidden, 32);
    assert_eq!(serde_json::to_value(report.task).unwrap(), "embedding");
    coheryn_engine::validate_tokenizer(model.descriptor()).unwrap();
    assert_eq!(
        error(Model::load(
            file.path(),
            ResourceLimits {
                context: 32,
                max_memory: 64 << 20
            }
        ))
        .code,
        ErrorCode::UnsupportedModelArchitecture
    );
    let baseline = model.memory().reserved;
    let result = model
        .embed_with_hardware(
            &["hello world"],
            &options(),
            &hardware(),
            &Cancellation::default(),
        )
        .unwrap();
    assert_eq!(result.dimension, 32);
    assert_eq!(result.usage.total_tokens, 4);
    let norm = result.data[0]
        .embedding
        .iter()
        .map(|v| f64::from(*v).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!((norm - 1.).abs() < 1e-6);
    assert_eq!(model.memory().reserved, baseline + result.plan.output_bytes);
    assert!(model.memory().session_buffer_capacity_bytes > 0);
    drop(result);
    assert_eq!(model.memory().reserved, baseline);
    assert_eq!(model.memory().session_buffer_capacity_bytes, 0);
}

#[test]
fn wordpiece_unicode_special_literals_long_words_and_explicit_truncation() {
    let tokenizer = EmbeddingTokenizer::from_descriptor(&descriptor()).unwrap();
    let cancel = Cancellation::default();
    assert_eq!(
        tokenizer
            .encode("HELLO café cafe\u{301} 中 !", 32, false, &cancel)
            .unwrap(),
        (vec![2, 5, 9, 9, 10, 11, 3], false)
    );
    assert_eq!(
        tokenizer.encode("[CLS]", 32, false, &cancel).unwrap().0,
        vec![2, 12, 14, 13, 3]
    );
    assert_eq!(
        tokenizer.encode("  \t\n", 2, false, &cancel).unwrap().0,
        vec![2, 3]
    );
    assert_eq!(
        tokenizer
            .encode(&"a".repeat(200), 512, false, &cancel)
            .unwrap()
            .0
            .len(),
        202
    );
    // A failed suffix rolls back the entire word, even beyond the token limit.
    assert_eq!(
        tokenizer
            .encode(&("a".repeat(200) + "🙂"), 3, false, &cancel)
            .unwrap()
            .0,
        vec![2, 1, 3]
    );
    assert!(tokenizer.encode("hello world", 3, false, &cancel).is_err());
    assert_eq!(
        tokenizer.encode("hello world", 3, true, &cancel).unwrap(),
        (vec![2, 5, 3], true)
    );
    assert!(tokenizer.encode("", 32, false, &cancel).is_err());
    assert!(
        tokenizer
            .encode(&"a".repeat(65537), 512, false, &cancel)
            .is_err()
    );
    cancel.cancel();
    assert_eq!(
        error(tokenizer.encode("hello", 32, false, &cancel)).code,
        ErrorCode::Cancelled
    );
}

#[test]
fn published_vocabulary_matches_all_frozen_multilingual_token_ids_without_model_downloads() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/embedding");
    let vocabulary = std::fs::read_to_string(
        root.join("../../resources/reference-archive/fixtures/embedding/source/vocab.txt"),
    )
    .unwrap();
    let special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"];
    let tokens = vocabulary
        .lines()
        .map(|word| {
            Value::String(if special.contains(&word) {
                word.to_owned()
            } else if let Some(tail) = word.strip_prefix("##") {
                tail.to_owned()
            } else {
                format!("▁{word}")
            })
        })
        .collect();
    let mut d = descriptor();
    d.metadata
        .insert("tokenizer.ggml.tokens".into(), Value::Array(tokens));
    for (key, id) in [("bos", 101), ("eos", 102), ("unknown", 100), ("padding", 0)] {
        d.metadata.insert(
            format!("tokenizer.ggml.{key}_token_id"),
            Value::Unsigned(id),
        );
    }
    let tokenizer = EmbeddingTokenizer::from_descriptor(&d).unwrap();
    for filename in [
        "nomic-embed-text-v1.5.f16.reference.json",
        "nomic-embed-text-v1.5.f16.multilingual.json",
    ] {
        let reference: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(root.join(filename)).unwrap()).unwrap();
        for case in reference["cases"].as_array().unwrap() {
            let text = case
                .get("input")
                .or_else(|| case.get("text"))
                .unwrap()
                .as_str()
                .unwrap();
            let (tokens, truncated) = tokenizer
                .encode(text, 512, false, &Cancellation::default())
                .unwrap();
            assert!(!truncated);
            assert_eq!(
                serde_json::to_value(tokens).unwrap(),
                case["token_ids"],
                "case {}",
                case["id"]
            );
        }
    }
}

#[test]
fn mixed_batches_preserve_order_masks_and_dimensions_in_all_profiles_and_encodings() {
    for dtype in [0, 1, 8] {
        let file = fixture(dtype);
        let model = load(&file);
        let baseline = model.memory().reserved;
        let inputs = ["hello", "hello world a", "hello", "café 中"];
        let mut singles = Vec::new();
        for text in inputs {
            let result = model
                .embed_with_hardware(&[text], &options(), &hardware(), &Cancellation::default())
                .unwrap();
            singles.push(result.data[0].embedding.clone());
        }
        for (profile, threads, batch_size) in [
            (Profile::Reference, 1, 4),
            (Profile::Balanced, 2, 4),
            (Profile::Performance, 4, 4),
            (Profile::LowMemory, 1, 4),
            (Profile::Balanced, 2, 1),
        ] {
            let settings = EmbeddingOptions {
                profile,
                threads: Some(threads),
                batch_size,
                ..options()
            };
            let result = model
                .embed_with_hardware(&inputs, &settings, &hardware(), &Cancellation::default())
                .unwrap();
            for (i, item) in result.data.iter().enumerate() {
                assert_eq!(item.index, i);
                assert!(!item.truncated);
                assert_eq!(
                    item.embedding, singles[i],
                    "dtype {dtype}, profile {profile:?}"
                );
            }
            assert_eq!(result.usage.prompt_tokens, 15);
            assert!(
                result.memory.peak_session_buffer_capacity_bytes
                    <= result.plan.additional_reserved_bytes
            );
        }
        assert_eq!(model.memory().reserved, baseline);
        assert_eq!(model.memory().session_buffer_capacity_bytes, 0);
    }
}

#[test]
fn cancellation_observer_failure_and_input_failure_release_every_reservation() {
    let file = fixture(1);
    let model = load(&file);
    let baseline = model.memory().reserved;
    let cancel = Cancellation::default();
    cancel.cancel();
    assert_eq!(
        error(model.embed(&["hello"], &options(), &cancel)).code,
        ErrorCode::Cancelled
    );
    let cancel = Cancellation::default();
    let other = cancel.clone();
    let result = model.embed_observed(&["hello world"], &options(), &cancel, &mut |_| {
        other.cancel();
        Ok(())
    });
    assert_eq!(error(result).code, ErrorCode::Cancelled);
    assert!(
        model
            .embed_observed(
                &["hello"],
                &options(),
                &Cancellation::default(),
                &mut |_| { Err(Error::request("observer failed")) }
            )
            .is_err()
    );
    assert!(
        model
            .embed(
                &["hello world"],
                &EmbeddingOptions {
                    max_input_tokens: 3,
                    ..options()
                },
                &Cancellation::default()
            )
            .is_err()
    );
    assert_eq!(model.memory().reserved, baseline);
    assert_eq!(model.memory().session_buffer_capacity_bytes, 0);
    assert!(
        model
            .embed(&["hello"], &options(), &Cancellation::default())
            .is_ok()
    );
}

#[test]
fn admission_is_bound_to_the_model_and_holds_concurrency_until_execution_or_drop() {
    let file = fixture(1);
    let model = load(&file);
    let other = load(&file);
    let baseline = model.memory().reserved;
    let admission = model
        .reserve_request(&[5], &options(), &hardware())
        .unwrap();
    assert!(admission.plan().admissible);
    assert!(
        model
            .reserve_request(&[5], &options(), &hardware())
            .is_err()
    );
    assert!(
        other
            .embed_admitted(&["hello"], admission, &Cancellation::default())
            .is_err()
    );
    assert_eq!(model.memory().reserved, baseline);
    let admission = model
        .reserve_request(&[5], &options(), &hardware())
        .unwrap();
    assert!(
        model
            .embed_admitted(&["hello world"], admission, &Cancellation::default())
            .is_err()
    );
    assert_eq!(model.memory().reserved, baseline);
    let admission = model
        .reserve_request(&[5], &options(), &hardware())
        .unwrap();
    let result = model
        .embed_admitted(&["hello"], admission, &Cancellation::default())
        .unwrap();
    // Completed vectors keep output memory; compute slot is already released.
    let next = model
        .reserve_request(&[5], &options(), &hardware())
        .unwrap();
    drop(next);
    drop(result);
    assert_eq!(model.memory().reserved, baseline);
}

#[test]
fn embedding_planner_adapts_microbatches_and_rejects_overflow_and_pressure() {
    let d = descriptor();
    let c = coheryn_engine::validate(&d).embedding_config.unwrap();
    let h = hardware();
    let settings = EmbeddingOptions {
        max_input_tokens: 512,
        ..options()
    };
    let settings = EmbeddingOptions {
        max_batch_tokens: 2048,
        ..settings
    };
    let estimate = |budget, hardware: &HardwareSnapshot, config: &embedding::EmbeddingConfig| {
        embedding::estimate(EmbeddingPlanInput {
            descriptor: &d,
            config,
            context: 512,
            input_bytes: &[5; 8],
            options: &settings,
            budget,
            hardware,
            max_concurrency: 1,
        })
    };
    let large = estimate(
        MemoryStats {
            limit: 128 << 20,
            ..Default::default()
        },
        &h,
        &c,
    )
    .unwrap();
    assert_eq!(large.token_capacity, 2048);
    let small = estimate(
        MemoryStats {
            limit: large.additional_reserved_bytes - 1,
            ..Default::default()
        },
        &h,
        &c,
    )
    .unwrap();
    assert!(small.admissible);
    assert!(small.token_capacity < large.token_capacity);
    assert_eq!(small.max_input_tokens, 512);
    let denied = estimate(
        MemoryStats {
            limit: 1,
            ..Default::default()
        },
        &h,
        &c,
    )
    .unwrap();
    assert!(!denied.admissible);
    assert!(denied.require_admission().is_err());
    let mut pressure = h.clone();
    pressure.ram_available_bytes = Some(1);
    assert!(
        !estimate(
            MemoryStats {
                limit: 128 << 20,
                ..Default::default()
            },
            &pressure,
            &c
        )
        .unwrap()
        .admissible
    );
    let mut invalid = c;
    invalid.hidden = usize::MAX;
    assert!(
        estimate(
            MemoryStats {
                limit: u64::MAX,
                ..Default::default()
            },
            &h,
            &invalid
        )
        .is_err()
    );
}

#[test]
fn incompatible_graphs_and_tokenizers_fail_before_execution_allocation() {
    for (key, value) in [
        ("nomic-bert.attention.causal", Value::Bool(true)),
        ("nomic-bert.pooling_type", Value::Unsigned(2)),
        ("nomic-bert.rope.scaling.factor", Value::Float(2.)),
        ("tokenizer.ggml.normalizer.lowercase", Value::Bool(false)),
    ] {
        let mut d = descriptor();
        d.metadata.insert(key.into(), value);
        let report = coheryn_engine::validate(&d);
        assert!(!report.executable);
        assert!(!report.diagnostics.is_empty());
    }
    let mut d = descriptor();
    d.tensors[0].dimensions[0] += 1;
    assert!(!coheryn_engine::validate(&d).executable);
    let mut d = descriptor();
    d.tensors[0].dtype = 2;
    assert!(!coheryn_engine::validate(&d).executable);
}

#[test]
fn shared_budget_admission_is_atomic_across_embedding_requests() {
    let file = fixture(1);
    let budget = Budget::new(64 << 20);
    let model = EmbeddingModel::load_shared(
        file.path(),
        ResourceLimits {
            context: 512,
            max_memory: 64 << 20,
        },
        budget.clone(),
        2,
    )
    .unwrap();
    let baseline = budget.stats().reserved;
    let plan = model.plan(&[5], &options(), &hardware()).unwrap();
    let blocker = budget
        .reserve(64 * 1024 * 1024 - baseline - plan.additional_reserved_bytes)
        .unwrap();
    let barrier = std::sync::Barrier::new(2);
    std::thread::scope(|scope| {
        let a = scope.spawn(|| {
            let admission = model.reserve_request(&[5], &options(), &hardware());
            barrier.wait();
            admission
        });
        let b = scope.spawn(|| {
            let admission = model.reserve_request(&[5], &options(), &hardware());
            barrier.wait();
            admission
        });
        let a = a.join().unwrap();
        let b = b.join().unwrap();
        assert_ne!(a.is_ok(), b.is_ok());
    });
    drop(blocker);
    assert_eq!(budget.stats().reserved, baseline);
}

#[test]
fn queued_jobs_yield_without_retokenization_and_failed_steps_cannot_resume() {
    let file = fixture(1);
    let model = load(&file);
    let baseline = model.memory().reserved;
    let opts = EmbeddingOptions {
        batch_size: 1,
        ..options()
    };
    let texts = ["hello", "hello world", "hello"];
    let expected = model
        .embed(&texts, &opts, &Cancellation::default())
        .unwrap()
        .data
        .iter()
        .map(|i| i.embedding.clone())
        .collect::<Vec<_>>();
    let admission = model
        .reserve_queued_request(&[5, 11, 5], &opts, &hardware())
        .unwrap();
    let mut job = model
        .prepare_admitted(&texts, admission, &Cancellation::default())
        .unwrap();
    assert_eq!(job.prompt_tokens(), 10);
    assert!(!job.is_finished());
    model.step(&mut job, &Cancellation::default()).unwrap();
    // An independent compute admission can run while this job retains its scratch and output.
    drop(model.reserve_request(&[5], &opts, &hardware()).unwrap());
    while !job.is_finished() {
        model.step(&mut job, &Cancellation::default()).unwrap();
    }
    let result = model.finish(job).unwrap();
    assert_eq!(
        result
            .data
            .iter()
            .map(|i| i.embedding.clone())
            .collect::<Vec<_>>(),
        expected
    );
    drop(result);
    let admission = model
        .reserve_queued_request(&[5], &opts, &hardware())
        .unwrap();
    let mut job = model
        .prepare_admitted(&["hello"], admission, &Cancellation::default())
        .unwrap();
    let cancel = Cancellation::default();
    cancel.cancel();
    assert!(model.step(&mut job, &cancel).is_err());
    assert!(model.step(&mut job, &Cancellation::default()).is_err());
    assert!(model.finish(job).is_err());
    assert_eq!(model.memory().reserved, baseline);
    assert_eq!(model.memory().session_buffer_capacity_bytes, 0);
}

#[test]
fn per_request_ceiling_adapts_packing_without_changing_input_limits_or_shared_budget() {
    let file = fixture(1);
    let model = load(&file);
    let baseline = model.memory().reserved;
    let opts = EmbeddingOptions {
        max_input_tokens: 512,
        max_batch_tokens: 2048,
        ..options()
    };
    let plan = model.plan(&[5; 8], &opts, &hardware()).unwrap();
    let admission = model
        .reserve_queued_capped(
            &[5; 8],
            &opts,
            &hardware(),
            Some(plan.additional_reserved_bytes - 1),
        )
        .unwrap();
    assert!(admission.plan().token_capacity < plan.token_capacity);
    assert_eq!(admission.plan().max_input_tokens, 512);
    assert_eq!(model.memory().limit, 64 << 20);
    drop(admission);
    assert_eq!(model.memory().reserved, baseline);
}

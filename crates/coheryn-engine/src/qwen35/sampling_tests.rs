//! Sampler/state integration against frozen scalar answers and complete tiny native forwards.
use super::*;
use crate::sampling::{Penalties, Sampler, Sampling};
use serde_json::json;

fn model(vocabulary: usize) -> (tempfile::TempDir, Model) {
    let directory = tempfile::tempdir().unwrap();
    let hidden = 4;
    let embedding_bytes = vocabulary * hidden * 4;
    let mut header = serde_json::to_vec(&json!({
        "model.language_model.embed_tokens.weight": {
            "dtype":"F32", "shape":[vocabulary,hidden], "data_offsets":[0,embedding_bytes]
        },
        "model.language_model.norm.weight": {
            "dtype":"F32", "shape":[hidden], "data_offsets":[embedding_bytes,embedding_bytes+hidden*4]
        }
    })).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.resize(bytes.len() + embedding_bytes, 0);
    for _ in 0..hidden {
        bytes.extend(1f32.to_le_bytes());
    }
    let path = directory.path().join("model.safetensors");
    std::fs::write(&path, bytes).unwrap();
    let set = TensorSet::open(&path, Limits::default(), &Default::default()).unwrap();
    let vocabulary_map: tokenizers::models::bpe::Vocab = (0..vocabulary)
        .map(|id| (((b'a' + id as u8) as char).to_string(), id as u32))
        .collect();
    let tokenizer = Tokenizer::new(
        tokenizers::models::bpe::BPE::builder()
            .vocab_and_merges(vocabulary_map, vec![])
            .build()
            .unwrap(),
    );
    // Deliberately a zero-layer test graph, not a structurally supported publisher model.
    // Sampling equations are checked separately against immutable independent fixtures.
    let config = Config {
        zero_centered_norms: false,
        ssm_pre_exponentiated: false,
        hidden,
        intermediate: 8,
        vocabulary,
        heads: 1,
        kv_heads: 1,
        head_dim: 4,
        linear_key_heads: 1,
        linear_value_heads: 1,
        key_dim: 4,
        value_dim: 4,
        conv_kernel: 2,
        epsilon: 1e-6,
        rope_base: 10000.,
        rotary_dim: 4,
        context: 128,
        eos: (vocabulary - 1) as u32,
        tied: true,
        layers: vec![],
    };
    let graph = Graph::bind(&set, &config).unwrap();
    let limits = ResourceLimits {
        context: 128,
        max_memory: 512 << 20,
    };
    let budget = Budget::new(limits.max_memory);
    let resident = budget.reserve(RESIDENT_METADATA_BYTES).unwrap();
    let model = Model {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        gpu: None,
        config,
        set,
        graph,
        vision: None,
        tokenizer,
        template: String::new(),
        budget,
        limits,
        _resident: resident,
        owner: Arc::new(()),
    };
    (directory, model)
}

fn begin(model: &Model, prompt: &str, options: GenerationOptions) -> GenerationState {
    let bytes = model.generation_bytes(128, 1).unwrap();
    let admission = plan::Admission::reserve(&model.budget, bytes, bytes, 1).unwrap();
    GenerationState::begin(
        model,
        prompt,
        options,
        admission,
        128,
        None,
        &Default::default(),
    )
    .unwrap()
}

#[test]
fn frozen_scalar_answers_survive_hybrid_state_history_and_incremental_decoding() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../fixtures/constraints/sampling-v2.json"
    ))
    .unwrap();
    let mut decisions = 0;
    for case in fixture["cases"].as_array().unwrap() {
        if case["allowed"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v.as_bool() != Some(true))
        {
            continue; // Constraints/masks remain unsupported by this adapter.
        }
        let mut logits: Vec<f32> = serde_json::from_value(case["logits"].clone()).unwrap();
        let (directory, model) = model(logits.len() + 1);
        // Extra EOS has zero probability under these finite fixtures, including all penalties.
        logits.push(-10000.);
        let history: Vec<u32> = serde_json::from_value(case["history"].clone()).unwrap();
        let prompt = if history.is_empty() {
            ((b'a' + model.config.eos as u8) as char).to_string()
        } else {
            history
                .iter()
                .map(|id| (b'a' + *id as u8) as char)
                .collect()
        };
        let options = GenerationOptions {
            max_tokens: case["expected_ids"].as_array().unwrap().len(),
            sampling: serde_json::from_value(case["settings"].clone()).unwrap(),
            penalties: serde_json::from_value(case["penalties"].clone()).unwrap(),
            ..Default::default()
        };
        let mut state = begin(&model, &prompt, options);
        state.cursor = state.tokens.len();
        state
            .session
            .as_mut()
            .unwrap()
            .logits
            .copy_from_slice(&logits);
        let mut streamed = String::new();
        for expected in case["expected_ids"].as_array().unwrap() {
            state.pending = None; // Injected frozen logits replace the next diagnostic forward.
            if let Some(event) = state.sample(&model).unwrap() {
                streamed.push_str(&event.text);
            }
            assert_eq!(
                state.ids.last().copied(),
                Some(expected.as_u64().unwrap() as u32)
            );
            decisions += 1;
        }
        let (result, cache) = model.finish_generation(state).unwrap();
        assert!(cache.is_none());
        assert_eq!(result.text, streamed);
        assert_eq!(result.finish_reason, FinishReason::Length);
        assert_eq!(
            result.sampling_metadata.unwrap().profile,
            crate::sampling::PROFILE
        );
        assert_eq!(model.memory().reserved, RESIDENT_METADATA_BYTES);
        assert_eq!(model.memory().session_buffer_capacity_bytes, 0);
        drop(model);
        drop(directory);
    }
    assert_eq!(decisions, 72);
}

#[test]
fn seeded_native_forwards_match_batched_lanes_and_release_private_scratch() {
    let (_directory, model) = model(8);
    let options = |seed| GenerationOptions {
        max_tokens: 12,
        sampling: Sampling {
            temperature: 0.8,
            top_k: 6,
            top_p: 0.85,
            seed,
        },
        penalties: Penalties {
            repetition_penalty: 1.15,
            frequency_penalty: 0.1,
            presence_penalty: 0.2,
        },
        ..Default::default()
    };
    let prompts = ["abba", "ccd"];
    let mut expected = Vec::new();
    for (index, prompt) in prompts.iter().enumerate() {
        let result = model
            .generate(
                prompt,
                &options(index as u64 + 7),
                1,
                &Default::default(),
                &mut |_| Ok(()),
            )
            .unwrap();
        expected.push(result.token_ids);
    }
    let mut states: Vec<_> = prompts
        .iter()
        .enumerate()
        .map(|(index, prompt)| begin(&model, prompt, options(index as u64 + 7)))
        .collect();
    while states.iter().any(|s| !s.is_done()) {
        let mut live: Vec<_> = states
            .iter_mut()
            .filter(|s| !s.is_done() && s.needs_forward())
            .collect();
        if !live.is_empty() {
            model.step_batch(&mut live, None).unwrap();
        }
        for state in states.iter_mut().filter(|s| !s.is_done()) {
            state.sample(&model).unwrap();
        }
    }
    for (state, expected) in states.into_iter().zip(expected) {
        assert_eq!(
            model.finish_generation(state).unwrap().0.token_ids,
            expected
        );
    }
    let state = begin(&model, "abba", options(7));
    assert!(model.memory().session_buffer_capacity_bytes > 0);
    state.cancel.cancel();
    drop(state);
    assert_eq!(model.memory().reserved, RESIDENT_METADATA_BYTES);
    assert_eq!(model.memory().session_buffer_capacity_bytes, 0);
}

#[test]
fn expanded_image_tokens_replace_text_penalty_history_before_sampling() {
    let (_directory, model) = model(5);
    let mut state = begin(
        &model,
        "aaaa",
        GenerationOptions {
            max_tokens: 1,
            penalties: Penalties {
                repetition_penalty: 2.,
                frequency_penalty: 0.25,
                presence_penalty: 0.5,
            },
            ..Default::default()
        },
    );
    state
        .set_images(
            vec![1, 1, 2],
            super::super::image_generation::Conditioning {
                features: vec![],
                positions: vec![],
                spans: vec![],
                next_position: 3,
                metadata: serde_json::from_value(
                    json!({"profile":"test","language_sha256":"","vision_sha256":"",
            "images":0,"vision_ms":0.,"effective_positions":3,"peak_request_reserved_bytes":0}),
                )
                .unwrap(),
            },
        )
        .unwrap();
    state.cursor = state.tokens.len();
    state
        .session
        .as_mut()
        .unwrap()
        .logits
        .copy_from_slice(&[5., 4., 3., 2., -10000.]);
    state.sample(&model).unwrap();
    assert_eq!(state.ids, [0]); // Exact adjusted scores: [5, 1, .75, 2, -10000].
    drop(state);
    assert_eq!(model.memory().session_buffer_capacity_bytes, 0);
}

#[test]
fn invalid_settings_and_under_reserved_sampling_fail_without_leaking_admission() {
    let (_directory, model) = model(8);
    let default = GenerationOptions::default();
    for options in [
        GenerationOptions {
            sampling: Sampling {
                temperature: f32::NAN,
                ..Default::default()
            },
            ..default.clone()
        },
        GenerationOptions {
            sampling: Sampling {
                top_p: 0.,
                ..Default::default()
            },
            ..default.clone()
        },
        GenerationOptions {
            penalties: Penalties {
                frequency_penalty: 3.,
                ..Default::default()
            },
            ..default.clone()
        },
        GenerationOptions {
            stop: vec!["stop".into()],
            ..default.clone()
        },
    ] {
        assert!(Model::check_options(&options).is_err());
    }
    assert!(
        model
            .check_request_options(&GenerationOptions {
                sampling: Sampling {
                    top_k: 9,
                    ..Default::default()
                },
                ..default.clone()
            })
            .is_err()
    );
    let too_small = Session::bytes(&model.config, 128, 1).unwrap() + crate::planner::REQUEST_BYTES;
    let admission = plan::Admission::reserve(&model.budget, too_small, too_small, 1).unwrap();
    assert!(
        GenerationState::begin(
            &model,
            "ab",
            default,
            admission,
            128,
            None,
            &Default::default()
        )
        .is_err()
    );
    assert_eq!(model.memory().reserved, RESIDENT_METADATA_BYTES);
    assert_eq!(model.memory().session_buffer_capacity_bytes, 0);
    assert_eq!(
        model.config.session_estimate(128, 1).unwrap().total_bytes,
        Session::bytes(&model.config, 128, 1).unwrap() + Sampler::workspace_bytes(8).unwrap()
    );
    assert!(Sampler::workspace_bytes(usize::MAX).is_err());
}

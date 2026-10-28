mod common;
use coheryn_engine::{
    Cancellation, ErrorCode, GenerationOptions, Model, ResourceLimits,
    inference::GenerationState,
    sampling::{Penalties, Sampling},
};
fn fixture() -> (tempfile::TempDir, Model) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    let mut bytes = common::tiny_model();
    std::fs::write(&path, &bytes).unwrap();
    let d = coheryn_engine::inspect(&path, Default::default()).unwrap();
    for t in &d.tensors {
        let start = (d.data_offset + t.offset) as usize;
        for i in 0..t.bytes as usize / 4 {
            let value = if t.dimensions.len() == 1 && t.name.ends_with("norm.weight") {
                1.
            } else {
                ((i * 7 + 3) % 29) as f32 * 0.015625 - 0.1875
            };
            bytes[start + i * 4..start + i * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
    std::fs::write(path.clone(), bytes).unwrap();
    let m = Model::load(
        path,
        ResourceLimits {
            context: 128,
            max_memory: 128 << 20,
        },
    )
    .unwrap();
    (dir, m)
}
fn begin(m: &Model, prompt: &str, o: &GenerationOptions, c: &Cancellation) -> GenerationState {
    m.begin_generation(prompt, o, m.reserve_session(128, 0).unwrap(), c, None)
        .unwrap()
}
#[test]
fn dynamic_arrival_retirement_seeded_history_and_streams_match_individual_nonconstant_graph() {
    let (_dir, m) = fixture();
    let baseline = m.memory().reserved;
    let prompts = ["hi", "a longer prompt", "joined later"];
    let options: Vec<_> = [2, 5, 7]
        .into_iter()
        .enumerate()
        .map(|(i, max_tokens)| GenerationOptions {
            max_tokens,
            sampling: Sampling {
                temperature: 0.7,
                seed: 47 + i as u64,
                top_k: 16,
                top_p: 0.9,
            },
            penalties: Penalties {
                repetition_penalty: 1.1,
                frequency_penalty: 0.2,
                presence_penalty: 0.3,
            },
            ..Default::default()
        })
        .collect();
    let expected: Vec<_> = prompts
        .iter()
        .zip(&options)
        .map(|(p, o)| {
            m.generate(p, o, &Cancellation::default(), |_| Ok(()))
                .unwrap()
        })
        .collect();
    let mut states: Vec<_> = prompts[..2]
        .iter()
        .zip(&options)
        .map(|(p, o)| Some(begin(&m, p, o, &Cancellation::default())))
        .collect();
    states.push(None);
    let mut outputs = [String::new(), String::new(), String::new()];
    let mut results = [None, None, None];
    let mut saved_bytes = 0;
    for round in 0..100 {
        if round == 4 {
            states[2] = Some(begin(&m, prompts[2], &options[2], &Cancellation::default()));
        }
        let mut lanes: Vec<_> = states
            .iter_mut()
            .filter_map(Option::as_mut)
            .filter(|s| s.needs_forward())
            .collect();
        if !lanes.is_empty() {
            let stats = m.step_batch(&mut lanes).unwrap();
            saved_bytes += stats.matrix_bytes * (stats.width - 1) as u64;
        }
        for i in 0..3 {
            if let Some(state) = &mut states[i] {
                if let Some(event) = state.sample(&m).unwrap() {
                    outputs[i].push_str(&event.text);
                }
                if state.is_finished() {
                    let (result, _) = states[i]
                        .take()
                        .unwrap()
                        .finish(&m, false, |event| {
                            outputs[i].push_str(&event.text);
                            Ok(())
                        })
                        .unwrap();
                    results[i] = Some(result);
                }
            }
        }
        if results.iter().all(Option::is_some) {
            break;
        }
    }
    assert!(saved_bytes > 0, "batch must actually reuse weight reads");
    for i in 0..3 {
        let actual = results[i].as_ref().unwrap();
        assert_eq!(actual.token_ids, expected[i].token_ids);
        assert_eq!(actual.text, expected[i].text);
        assert_eq!(outputs[i], actual.text);
        assert!(actual.batching.as_ref().unwrap().batched_forward_units > 0);
    }
    assert_eq!(m.memory().reserved, baseline);
    assert_eq!(m.memory().session_buffer_capacity_bytes, 0);
}
#[test]
fn lane_cancellation_and_incomplete_schema_do_not_change_peer_outputs_or_leak_buffers() {
    let (_dir, m) = fixture();
    let baseline = m.memory().reserved;
    let options = GenerationOptions {
        max_tokens: 8,
        ..Default::default()
    };
    let expected = m
        .generate("healthy", &options, &Cancellation::default(), |_| Ok(()))
        .unwrap();
    let cancel = Cancellation::default();
    let mut failed = begin(&m, "cancel me", &options, &cancel);
    let mut healthy = begin(&m, "healthy", &options, &Cancellation::default());
    m.step_batch(&mut [&mut failed, &mut healthy]).unwrap();
    cancel.cancel();
    let step = m.step_batch(&mut [&mut failed, &mut healthy]).unwrap();
    assert_eq!(step.width, 1);
    assert_eq!(step.completed_tokens, 1);
    assert_eq!(failed.sample(&m).unwrap_err().code, ErrorCode::Cancelled);
    drop(failed);
    while !healthy.is_finished() {
        if healthy.needs_forward() {
            m.step_batch(&mut [&mut healthy]).unwrap();
        }
        healthy.sample(&m).unwrap();
    }
    let (actual, _) = healthy.finish(&m, false, |_| Ok(())).unwrap();
    assert_eq!(actual.token_ids, expected.token_ids);
    let mut short = GenerationOptions {
        max_tokens: 1,
        ..Default::default()
    };
    short.response_format =
        serde_json::from_value(serde_json::json!({"type":"json_object"})).unwrap();
    let mut invalid = begin(&m, "json", &short, &Cancellation::default());
    let mut peer = begin(&m, "healthy", &options, &Cancellation::default());
    while !invalid.is_finished() {
        let mut states = vec![&mut invalid];
        if peer.needs_forward() {
            states.push(&mut peer);
        }
        m.step_batch(&mut states).unwrap();
        invalid.sample(&m).unwrap();
        peer.sample(&m).unwrap();
    }
    assert_eq!(
        invalid.finish(&m, false, |_| Ok(())).err().unwrap().code,
        ErrorCode::ConstraintIncomplete
    );
    while !peer.is_finished() {
        if peer.needs_forward() {
            m.step_batch(&mut [&mut peer]).unwrap();
        }
        peer.sample(&m).unwrap();
    }
    assert_eq!(
        peer.finish(&m, false, |_| Ok(())).unwrap().0.token_ids,
        expected.token_ids
    );
    assert_eq!(m.memory().reserved, baseline);
    assert_eq!(m.memory().session_buffer_capacity_bytes, 0);
}
#[test]
fn foreign_models_and_worker_contracts_cannot_share_a_batch() {
    let (_dir, m) = fixture();
    let (_other, n) = fixture();
    let options = GenerationOptions {
        max_tokens: 1,
        ..Default::default()
    };
    let mut a = begin(&m, "first", &options, &Cancellation::default());
    let mut b = begin(&n, "second", &options, &Cancellation::default());
    assert!(!a.compatible(&b));
    assert_eq!(
        m.step_batch(&mut [&mut a, &mut b]).unwrap_err().code,
        ErrorCode::InvalidRequest
    );
    let settings = coheryn_engine::inference::ExecutionSettings {
        threads: 1,
        prefill_chunk_tokens: 2,
    };
    let mut c = m
        .begin_generation(
            "third",
            &options,
            m.reserve_session_with_execution(128, 0, 256 << 10, settings)
                .unwrap(),
            &Cancellation::default(),
            None,
        )
        .unwrap();
    assert!(!a.compatible(&c));
    assert!(m.step_batch(&mut [&mut a, &mut c]).is_err());
}

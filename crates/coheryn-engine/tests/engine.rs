//! A complete tiny GGUF exercises public execution without downloads.
mod common;
use coheryn_engine::{Cancellation, FinishReason, GenerationOptions, Model, ResourceLimits};
use common::tiny_model;
use std::path::PathBuf;
struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "coheryn-{}-{}.gguf",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::write(&path, tiny_model()).unwrap();
        Self(path)
    }
    fn load(&self) -> Model {
        Model::load(
            &self.0,
            ResourceLimits {
                context: 64,
                max_memory: 32 << 20,
            },
        )
        .unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
#[test]
fn complete_cpu_model_greedy_stream_and_cleanup() {
    let f = Fixture::new();
    let m = f.load();
    let before = m.memory().reserved;
    let mut output = String::new();
    let r = m
        .generate(
            "Hi",
            &GenerationOptions {
                max_tokens: 4,
                ..Default::default()
            },
            &Cancellation::default(),
            |e| {
                output += &e.text;
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(r.token_ids, vec![0; 4]);
    assert_eq!(r.text, "!!!!");
    assert_eq!(output, r.text);
    assert_eq!(r.finish_reason, FinishReason::Length);
    assert_eq!(m.memory().reserved, before);
}
#[test]
fn cancellation_and_sink_failure_release_session() {
    let f = Fixture::new();
    let m = f.load();
    let before = m.memory().reserved;
    let c = Cancellation::default();
    let other = c.clone();
    let r = m
        .generate(
            "Hi",
            &GenerationOptions {
                max_tokens: 4,
                ..Default::default()
            },
            &c,
            |_| {
                other.cancel();
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(r.finish_reason, FinishReason::Cancelled);
    assert_eq!(m.memory().reserved, before);
    assert!(
        m.generate(
            "Hi",
            &GenerationOptions {
                max_tokens: 4,
                ..Default::default()
            },
            &Cancellation::default(),
            |_| Err(coheryn_engine::Error::request("closed sink"))
        )
        .is_err()
    );
    assert_eq!(m.memory().reserved, before);
}
#[test]
fn infeasible_requests_and_file_truncation_fail_safely() {
    let f = Fixture::new();
    assert!(
        Model::load(
            &f.0,
            ResourceLimits {
                context: 64,
                max_memory: 1
            }
        )
        .is_err()
    );
    let m = f.load();
    assert!(
        m.generate(
            "Hi",
            &GenerationOptions {
                max_tokens: 64,
                ..Default::default()
            },
            &Cancellation::default(),
            |_| Ok(())
        )
        .is_err()
    );
    std::fs::OpenOptions::new()
        .write(true)
        .open(&f.0)
        .unwrap()
        .set_len(0)
        .unwrap();
    assert!(
        m.generate(
            "Hi",
            &GenerationOptions {
                max_tokens: 1,
                ..Default::default()
            },
            &Cancellation::default(),
            |_| Ok(())
        )
        .is_err()
    );
}
#[test]
fn teacher_forcing_exposes_layers_and_logits() {
    let f = Fixture::new();
    let m = f.load();
    let mut seen = vec![];
    m.trace(
        &[1, 2],
        &Cancellation::default(),
        &mut |position, name, values| {
            seen.push((position, name.to_owned(), values.len()));
            if name == "logits" {
                assert!(values.iter().all(|x| *x == 0.));
            }
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(
        seen,
        vec![
            (0, "blk.0.output".into(), 4),
            (0, "logits".into(), 259),
            (1, "blk.0.output".into(), 4),
            (1, "logits".into(), 259)
        ]
    );
}

#[test]
fn unsupported_variants_and_hostile_templates_fail_early() {
    let f = Fixture::new();
    let mut d = coheryn_engine::inspect(&f.0, Default::default()).unwrap();
    d.metadata.insert(
        "qwen2.rope.scaling.type".into(),
        coheryn_engine::gguf::Value::String("linear".into()),
    );
    assert!(!coheryn_engine::validate(&d).executable);
    d.metadata.remove("qwen2.rope.scaling.type");
    d.metadata.insert(
        "tokenizer.chat_template".into(),
        coheryn_engine::gguf::Value::String("{{ 'x' * 999999999999 }}".into()),
    );
    assert!(coheryn_engine::tokenizer::TextTokenizer::from_descriptor(&d).is_err());
}
#[test]
fn eos_stops_without_emitting_special_text() {
    use std::io::{Seek, SeekFrom, Write};
    let f = Fixture::new();
    let d = coheryn_engine::inspect(&f.0, Default::default()).unwrap();
    let t = d.tensor("output.weight").unwrap();
    let mut file = std::fs::OpenOptions::new().write(true).open(&f.0).unwrap();
    file.seek(SeekFrom::Start(d.data_offset + t.offset + 258 * 4 * 4))
        .unwrap();
    for _ in 0..4 {
        file.write_all(&1f32.to_le_bytes()).unwrap();
    }
    drop(file);
    let m = f.load();
    let r = m
        .generate(
            "Hi",
            &GenerationOptions {
                max_tokens: 4,
                ..Default::default()
            },
            &Cancellation::default(),
            |e| {
                assert!(e.text.is_empty());
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(r.token_ids, vec![258]);
    assert_eq!(r.finish_reason, FinishReason::Stop);
    assert_eq!(r.text, "");
}
#[test]
fn deep_validation_scans_payload_values() {
    use std::io::{Seek, SeekFrom, Write};
    let f = Fixture::new();
    let d = coheryn_engine::inspect(&f.0, Default::default()).unwrap();
    let mut file = std::fs::OpenOptions::new().write(true).open(&f.0).unwrap();
    file.seek(SeekFrom::Start(d.data_offset)).unwrap();
    file.write_all(&f32::NAN.to_le_bytes()).unwrap();
    drop(file);
    assert!(coheryn_engine::deep_validate(&f.0, &d, &Cancellation::default()).is_err());
}

#[test]
fn immutable_model_shares_weights_but_keeps_sessions_and_admission_independent() {
    use std::sync::{Arc, Barrier};
    let fixture = Fixture::new();
    let model = Arc::new(fixture.load());
    let barrier = Arc::new(Barrier::new(2));
    let baseline = model.memory().reserved;
    let threads: Vec<_> = [(4, 2), (6, 4)]
        .into_iter()
        .map(|(context, max_tokens)| {
            let model = model.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let mut first = true;
                model
                    .generate_at_context(
                        "Hi",
                        &GenerationOptions {
                            max_tokens,
                            ..Default::default()
                        },
                        context,
                        &Cancellation::default(),
                        |_| {
                            if first {
                                first = false;
                                barrier.wait();
                            }
                            Ok(())
                        },
                    )
                    .unwrap()
                    .text
            })
        })
        .collect();
    let output: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    assert_eq!(output, ["!!", "!!!!"]);
    assert_eq!(model.memory().reserved, baseline);
    let sum = coheryn_engine::memory::session_bytes(&model.config, 4).unwrap()
        + coheryn_engine::memory::session_bytes(&model.config, 6).unwrap();
    assert!(model.memory().peak_reserved >= baseline + sum);
}

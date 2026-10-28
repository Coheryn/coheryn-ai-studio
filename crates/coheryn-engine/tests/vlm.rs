mod common;
#[path = "common/vision.rs"]
mod vision_fixture;
use coheryn_engine::{
    Cancellation, Error, ErrorCode, ParseLimits, ResourceLimits, gguf,
    memory::Budget,
    multimodal::qwen2vl::{self, ChatContent, ChatMessage, ChatPart, Options, Qwen2Vl},
};
use std::io::Write;
fn language() -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(&common::tiny_vlm_model()).unwrap();
    f.flush().unwrap();
    f
}
fn messages(count: usize) -> Vec<ChatMessage> {
    let mut parts = (0..count)
        .map(|index| ChatPart::Image { index })
        .collect::<Vec<_>>();
    parts.push(ChatPart::Text {
        text: "Describe. বাংলা 中文 العربية 🙂".into(),
    });
    vec![ChatMessage {
        role: "user".into(),
        content: ChatContent::Parts(parts),
    }]
}
#[test]
fn rendered_prompts_match_all_six_frozen_reference_cases() {
    let reference: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/vlm/image-conditioned-reference-v1.json"
    ))
    .unwrap();
    for c in reference["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["model"] == "qwen2-vl-2b")
    {
        let mut count = 0;
        let mut messages = Vec::new();
        for m in c["messages"].as_array().unwrap() {
            let content = if let Some(text) = m["content"].as_str() {
                ChatContent::Text(text.into())
            } else {
                ChatContent::Parts(
                    m["content"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|part| {
                            if part["type"] == "image" {
                                let index = count;
                                count += 1;
                                ChatPart::Image { index }
                            } else {
                                ChatPart::Text {
                                    text: part["text"].as_str().unwrap().into(),
                                }
                            }
                        })
                        .collect(),
                )
            };
            messages.push(ChatMessage {
                role: m["role"].as_str().unwrap().into(),
                content,
            });
        }
        assert_eq!(
            qwen2vl::render(&messages, count).unwrap(),
            c["rendered_prompt"].as_str().unwrap()
        );
    }
}
#[test]
fn invalid_roles_media_markers_indices_and_bounds_fail_explicitly() {
    assert!(qwen2vl::render(&[], 0).is_err());
    assert!(qwen2vl::render(&messages(1), 0).is_err());
    assert!(qwen2vl::render(&messages(0), 1).is_err());
    assert!(qwen2vl::render(&messages(5), 5).is_err());
    for text in [
        "<|image_pad|>".to_owned(),
        "<|im_start|>".to_owned(),
        "x".repeat(65537),
    ] {
        assert!(
            qwen2vl::render(
                &[ChatMessage {
                    role: "user".into(),
                    content: ChatContent::Text(text)
                }],
                0
            )
            .is_err()
        );
    }
    let mut m = messages(1);
    m[0].role = "assistant".into();
    assert!(qwen2vl::render(&m, 1).is_err());
    let m = vec![ChatMessage {
        role: "user".into(),
        content: ChatContent::Parts(vec![
            ChatPart::Image { index: 0 },
            ChatPart::Image { index: 0 },
        ]),
    }];
    assert!(qwen2vl::render(&m, 1).is_err());
}
#[test]
fn configuration_rejects_unknown_variants_tensors_sections_and_encodings() {
    let f = language();
    let d = gguf::inspect(f.path(), ParseLimits::default()).unwrap();
    let c = qwen2vl::language_configuration(&d).unwrap();
    assert_eq!(c.sections, [16, 24, 24, 0]);
    assert_eq!(c.decoder.vocabulary, 262);
    assert!(
        !coheryn_engine::validate(&d).executable,
        "product loader must not opt into offline multimodal execution"
    );
    for kind in 0..7 {
        let mut d = d.clone();
        match kind {
            0 => {
                d.metadata.insert(
                    "qwen2vl.rope.dimension_sections".into(),
                    gguf::Value::Array(vec![gguf::Value::Unsigned(0); 4]),
                );
            }
            1 => {
                d.metadata.insert(
                    "qwen2vl.attention.sliding_window".into(),
                    gguf::Value::Unsigned(12),
                );
            }
            2 => d.tensors[0].dtype = 8,
            3 => {
                d.tensors.pop();
            }
            4 => {
                d.metadata
                    .insert("split.count".into(), gguf::Value::Unsigned(1));
            }
            5 => d.byte_order = "big".into(),
            _ => d.tensors[0].dimensions[0] += 1,
        }
        assert!(
            qwen2vl::language_configuration(&d).is_err(),
            "variant {kind}"
        );
    }
}
#[test]
fn two_images_use_reserved_stages_and_keep_physical_and_rotary_counts_separate() {
    let file = language();
    let vision = vision_fixture::component(256);
    let budget = Budget::new(512 << 20);
    let cancel = Cancellation::default();
    let model = Qwen2Vl::load(
        file.path(),
        vision.path(),
        ResourceLimits {
            context: 256,
            max_memory: 512 << 20,
        },
        budget.clone(),
        &cancel,
    )
    .unwrap();
    let before = budget.stats();
    let a = include_bytes!("../../../fixtures/vlm/images/landscape.png").as_slice();
    let b = include_bytes!("../../../fixtures/vlm/images/minimum.png").as_slice();
    let options = Options {
        max_tokens: 2,
        ..Default::default()
    };
    let mut observed = 0;
    let response = model
        .trace(
            &messages(2),
            &[a, b],
            &options,
            &cancel,
            &mut |_| {
                observed += 1;
                Ok(())
            },
            |_| Ok(()),
        )
        .unwrap();
    assert!(observed >= 3);
    assert!(response.plan.physical_prompt_tokens > response.plan.effective_positions as usize);
    assert_eq!(response.token_ids.len(), 2);
    assert!(budget.stats().reserved > before.reserved);
    let expected = response.token_ids.clone();
    drop(response);
    assert_eq!(budget.stats().reserved, before.reserved);
    let response = model
        .generate(
            &messages(2),
            &[a, b],
            &Options {
                max_tokens: 2,
                threads: 2,
                ..Default::default()
            },
            &cancel,
            |_| Ok(()),
        )
        .unwrap();
    assert_eq!(response.token_ids, expected);
    drop(response);
    assert_eq!(budget.stats().reserved, before.reserved);
    drop(model);
    assert_eq!(budget.stats().reserved, 0);
    assert_eq!(budget.stats().session_buffer_capacity_bytes, 0);
}
#[test]
fn failures_cancel_and_atomic_admission_release_all_temporary_state() {
    let file = language();
    let vision = vision_fixture::component(256);
    let budget = Budget::new(512 << 20);
    let cancel = Cancellation::default();
    let model = Qwen2Vl::load(
        file.path(),
        vision.path(),
        ResourceLimits {
            context: 128,
            max_memory: 512 << 20,
        },
        budget.clone(),
        &cancel,
    )
    .unwrap();
    let baseline = budget.stats();
    let png = include_bytes!("../../../fixtures/vlm/images/minimum.png").as_slice();
    let options = Options {
        max_tokens: 2,
        ..Default::default()
    };
    let held = budget
        .reserve((512 << 20) - baseline.reserved - (8 << 20))
        .unwrap();
    assert_eq!(
        model
            .generate(&messages(1), &[png], &options, &cancel, |_| Ok(()))
            .unwrap_err()
            .code,
        ErrorCode::ResourceExhausted
    );
    drop(held);
    assert_eq!(budget.stats().reserved, baseline.reserved);
    assert!(
        model
            .generate(&messages(1), &[b"broken"], &options, &cancel, |_| Ok(()))
            .is_err()
    );
    assert_eq!(budget.stats().reserved, baseline.reserved);
    assert!(
        model
            .generate(
                &messages(1),
                &[png],
                &Options {
                    max_tokens: 128,
                    ..Default::default()
                },
                &cancel,
                |_| Ok(())
            )
            .is_err()
    );
    let cancelled = Cancellation::default();
    let result = model.trace(
        &messages(1),
        &[png],
        &options,
        &cancelled,
        &mut |_| {
            cancelled.cancel();
            Ok(())
        },
        |_| Ok(()),
    );
    assert_eq!(result.unwrap_err().code, ErrorCode::Cancelled);
    assert_eq!(budget.stats().reserved, baseline.reserved);
    assert_eq!(
        budget.stats().session_buffer_capacity_bytes,
        baseline.session_buffer_capacity_bytes
    );
    assert!(
        model
            .generate(&messages(1), &[png], &options, &cancel, |_| Err(
                Error::request("client stopped")
            ))
            .is_err()
    );
    assert_eq!(budget.stats().reserved, baseline.reserved);
    assert!(
        model
            .trace(
                &messages(1),
                &[png],
                &options,
                &cancel,
                &mut |_| Err(Error::request("observer stopped")),
                |_| Ok(())
            )
            .is_err()
    );
    assert_eq!(budget.stats().reserved, baseline.reserved);
    drop(model);
    assert_eq!(budget.stats().reserved, 0);
}
#[test]
fn incompatible_component_width_and_changed_language_files_are_rejected() {
    let file = language();
    let wrong = vision_fixture::component(12);
    let budget = Budget::new(512 << 20);
    let cancel = Cancellation::default();
    assert!(
        Qwen2Vl::load(
            file.path(),
            wrong.path(),
            ResourceLimits {
                context: 128,
                max_memory: 512 << 20
            },
            budget.clone(),
            &cancel
        )
        .is_err()
    );
    assert_eq!(budget.stats().reserved, 0);
    let vision = vision_fixture::component(256);
    let model = Qwen2Vl::load(
        file.path(),
        vision.path(),
        ResourceLimits {
            context: 128,
            max_memory: 512 << 20,
        },
        budget.clone(),
        &cancel,
    )
    .unwrap();
    file.as_file()
        .set_len(file.as_file().metadata().unwrap().len() + 1)
        .unwrap();
    assert!(
        model
            .generate(
                &messages(0),
                &[],
                &Options {
                    max_tokens: 1,
                    ..Default::default()
                },
                &cancel,
                |_| Ok(())
            )
            .is_err()
    );
    drop(model);
    assert_eq!(budget.stats().reserved, 0);
}

#[test]
fn prepared_request_holds_full_capacity_while_queued_and_binds_model_instance() {
    let file = language();
    let vision = vision_fixture::component(256);
    let budget = Budget::new(512 << 20);
    let cancel = Cancellation::default();
    let load = || {
        Qwen2Vl::load(
            file.path(),
            vision.path(),
            ResourceLimits {
                context: 128,
                max_memory: 512 << 20,
            },
            budget.clone(),
            &cancel,
        )
        .unwrap()
    };
    let model = load();
    let other = load();
    let baseline = budget.stats().reserved;
    let png = include_bytes!("../../../fixtures/vlm/images/minimum.png").to_vec();
    let options = Options {
        context: Some(128),
        max_tokens: 2,
        ..Default::default()
    };
    let prepared = model
        .prepare_owned(&messages(1), vec![png.clone()], &options, &cancel)
        .unwrap();
    let request_bytes = prepared.plan().peak_request_bytes;
    assert_eq!(budget.stats().reserved, baseline + request_bytes);
    let peak_before_execution = budget.stats().peak_reserved;
    let response = model
        .generate_prepared(prepared, &cancel, |_| Ok(()))
        .unwrap();
    assert_eq!(response.plan.context, 128);
    assert_eq!(budget.stats().peak_reserved, peak_before_execution);
    drop(response);
    assert_eq!(budget.stats().reserved, baseline);
    let prepared = model
        .prepare_owned(&messages(1), vec![png.clone()], &options, &cancel)
        .unwrap();
    assert_eq!(
        other
            .generate_prepared(prepared, &cancel, |_| Ok(()))
            .unwrap_err()
            .code,
        ErrorCode::InvalidRequest
    );
    assert_eq!(budget.stats().reserved, baseline);
    let prepared = model
        .prepare_owned(&messages(1), vec![png], &options, &cancel)
        .unwrap();
    cancel.cancel();
    assert_eq!(
        model
            .generate_prepared(prepared, &cancel, |_| Ok(()))
            .unwrap_err()
            .code,
        ErrorCode::Cancelled
    );
    assert_eq!(budget.stats().reserved, baseline);
}

#[test]
fn cooperative_generation_matches_uninterrupted_execution_and_cleans_up_on_yield_failure() {
    let file = language();
    let vision = vision_fixture::component(256);
    let budget = Budget::new(512 << 20);
    let cancel = Cancellation::default();
    let model = Qwen2Vl::load(
        file.path(),
        vision.path(),
        ResourceLimits {
            context: 256,
            max_memory: 512 << 20,
        },
        budget.clone(),
        &cancel,
    )
    .unwrap();
    let baseline = budget.stats().reserved;
    let png = include_bytes!("../../../fixtures/vlm/images/minimum.png").to_vec();
    let options = Options {
        max_tokens: 3,
        ..Default::default()
    };
    let expected = model
        .generate(&messages(1), &[&png], &options, &cancel, |_| Ok(()))
        .unwrap();
    let ids = expected.token_ids.clone();
    let text = expected.text.clone();
    drop(expected);
    let prepared = model
        .prepare_owned(&messages(1), vec![png.clone()], &options, &cancel)
        .unwrap();
    let mut checkpoints = 0;
    let result = model
        .generate_cooperative(
            prepared,
            &cancel,
            &mut || {
                checkpoints += 1;
                std::thread::yield_now();
                Ok(())
            },
            |_| Ok(()),
        )
        .unwrap();
    assert_eq!(result.token_ids, ids);
    assert_eq!(result.text, text);
    assert!(checkpoints > 100);
    drop(result);
    assert_eq!(budget.stats().reserved, baseline);
    let prepared = model
        .prepare_owned(&messages(1), vec![png], &options, &cancel)
        .unwrap();
    let e = model
        .generate_cooperative(
            prepared,
            &cancel,
            &mut || {
                Err(Error::new(
                    ErrorCode::Cancelled,
                    "scheduler cancelled at checkpoint",
                ))
            },
            |_| Ok(()),
        )
        .unwrap_err();
    assert_eq!(e.code, ErrorCode::Cancelled);
    assert_eq!(budget.stats().reserved, baseline);
}

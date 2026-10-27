mod common;
use coheryn_engine::{FinishReason, Model, ResourceLimits, tokenizer::Message, tools::*};
use serde_json::{Value, json};
fn config() -> ToolConfig {
    serde_json::from_value(json!({"tools":[{"type":"function","function":{"name":"add","parameters":{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}}}}}]})).unwrap()
}
fn parse(
    s: &str,
    step: usize,
    cfg: ToolConfig,
) -> coheryn_engine::Result<(String, Vec<ToolCall>, Vec<ToolDelta>)> {
    let mut p = ToolParser::new(cfg, "call_test".into())?;
    let mut deltas = vec![];
    let mut text = String::new();
    let chars: Vec<_> = s.chars().collect();
    for chunk in chars.chunks(step) {
        for event in p.push(&chunk.iter().collect::<String>())? {
            match event {
                ParsedEvent::Text(s) => text.push_str(&s),
                ParsedEvent::Delta(d) => deltas.push(d),
            }
        }
    }
    let (out, calls, tail) = p.finish(FinishReason::Stop)?;
    text.push_str(&tail);
    assert_eq!(text, out);
    Ok((out, calls, deltas))
}
#[test]
fn independently_rendered_tool_prompts_match_exactly() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model.gguf");
    std::fs::write(&path, common::conversation_model()).unwrap();
    let model = Model::load(
        path,
        ResourceLimits {
            max_memory: 64 << 20,
            context: 512,
        },
    )
    .unwrap();
    let reference: Value = serde_json::from_str(include_str!(
        "../../../fixtures/tools/qwen2.5-reference.json"
    ))
    .unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let mut messages = case["messages"].clone();
        for message in messages.as_array_mut().unwrap() {
            for call in message["tool_calls"].as_array_mut().unwrap() {
                if !call["function"]["arguments"].is_string() {
                    call["function"]["arguments"] =
                        json!(template_json(call["function"]["arguments"].clone()));
                }
            }
        }
        let messages: Vec<Message> = serde_json::from_value(messages).unwrap();
        let cfg: ToolConfig = serde_json::from_value(json!({"tools":case["tools"]})).unwrap();
        assert_eq!(
            model.tokenizer.render_tools(&messages, &cfg).unwrap(),
            case["prompt"].as_str().unwrap(),
            "{}",
            case["id"]
        );
    }
}
#[test]
fn fragmented_unicode_and_multiple_calls_preserve_indices_ids_arguments_and_text() {
    let raw = "Before 🌍<tool_call>\n{\"name\":\"add\",\"arguments\":{\"label\":\"বাংলা </tool_call> \\\" 🌍\",\"a\":2,\"b\":2}}\n</tool_call><tool_call>{\"name\":\"add\",\"arguments\":{\"a\":3,\"b\":4}}</tool_call>After";
    let whole = parse(raw, raw.len(), config()).unwrap();
    assert_eq!(whole.0, "Before 🌍After");
    assert_eq!(whole.1.len(), 2);
    for step in [1, 2, 7, 31, 256] {
        let result = parse(raw, step, config()).unwrap();
        assert_eq!(result.0, whole.0);
        assert_eq!(result.1, whole.1);
    }
    for (i, c) in whole.1.iter().enumerate() {
        let fragments: Vec<_> = whole.2.iter().filter(|d| d.index == i).collect();
        assert_eq!(fragments[0].id.as_ref(), Some(&c.id));
        assert_eq!(
            fragments
                .iter()
                .map(|d| d.function.arguments.as_str())
                .collect::<String>(),
            c.function.arguments
        );
    }
}
#[test]
fn declarations_choices_strict_references_and_malformed_calls_fail_explicitly() {
    for value in [
        json!({"tool_choice":"required"}),
        json!({"tools":config().tools,"tool_choice":{"type":"function","function":{"name":"absent"}}}),
        json!({"tools":[{"type":"function","function":{"name":"bad name","parameters":{"type":"object"}}}]}),
        json!({"tools":[{"type":"function","function":{"name":"add","strict":true,"parameters":{"type":"object"}}}]}),
        json!({"tools":[{"type":"function","function":{"name":"add","parameters":{"type":"object","$ref":"https://invalid.example/schema"}}}]}),
    ] {
        let c: ToolConfig = serde_json::from_value(value).unwrap();
        assert!(c.validate().is_err());
    }
    for raw in [
        "<tool_call>{\"name\":\"unknown\",\"arguments\":{}}</tool_call>",
        "<tool_call>{\"name\":\"add\",\"arguments\":[]}</tool_call>",
        "<tool_call>{\"name\":\"add\",\"arguments\":{}}",
        "<tool_call>{{\"name\":\"add\",\"arguments\":{}}}</tool_call>",
    ] {
        assert!(parse(raw, 1, config()).is_err(), "{raw}");
    }
    let mut c = config();
    c.tool_choice = Some(ToolChoice::Mode(ChoiceMode::Required));
    assert!(parse("plain text", 1, c).is_err());
    let mut c = config();
    c.parallel_tool_calls = Some(false);
    assert!(parse("<tool_call>{\"name\":\"add\",\"arguments\":{}}</tool_call><tool_call>{\"name\":\"add\",\"arguments\":{}}</tool_call>",3,c).is_err());
}
#[test]
fn original_model_negative_fixture_is_rejected_without_repair() {
    let reference: Value = serde_json::from_str(include_str!(
        "../../../fixtures/tools/qwen2.5-original-negative.json"
    ))
    .unwrap();
    assert!(
        parse(
            reference["cases"][0]["generated_text"].as_str().unwrap(),
            1,
            config()
        )
        .is_err()
    );
}
#[test]
fn result_ids_roles_and_length_termination_cannot_be_claimed_complete() {
    let (_, calls, _) = parse(
        "<tool_call>{\"name\":\"add\",\"arguments\":{}}</tool_call>",
        1,
        config(),
    )
    .unwrap();
    let cfg = config();
    cfg.validate_output(&calls, FinishReason::ToolCalls)
        .unwrap();
    assert!(cfg.validate_output(&calls, FinishReason::Length).is_err());
    let assistant = Message {
        role: "assistant".into(),
        tool_calls: calls.clone(),
        ..Default::default()
    };
    assert!(
        cfg.validate_messages(std::slice::from_ref(&assistant), false)
            .is_err()
    );
    let tool = Message {
        role: "tool".into(),
        content: "4".into(),
        tool_call_id: Some(calls[0].id.clone()),
        ..Default::default()
    };
    assert!(
        cfg.validate_messages(&[assistant.clone(), tool.clone()], false)
            .is_ok()
    );
    assert!(
        cfg.validate_messages(&[assistant, tool.clone(), tool], false)
            .is_err()
    );
}
#[test]
fn argument_deltas_are_bounded_utf8_and_output_limits_fail_closed() {
    let text = "🌍".repeat(1000);
    let raw = format!(
        "<tool_call>{{\"name\":\"add\",\"arguments\":{{\"label\":{}}}}}</tool_call>",
        serde_json::to_string(&text).unwrap()
    );
    let (_, calls, deltas) = parse(&raw, 1, config()).unwrap();
    assert!(deltas.len() > 1);
    assert!(deltas.iter().all(|d| d.function.arguments.len() <= 256));
    assert_eq!(
        deltas
            .iter()
            .map(|d| d.function.arguments.as_str())
            .collect::<String>(),
        calls[0].function.arguments
    );
    let mut parser = ToolParser::new(config(), "test".into()).unwrap();
    assert!(parser.push(&"x".repeat(MAX_OUTPUT_BYTES + 1)).is_err());
    let mut parser = ToolParser::new(config(), "test".into()).unwrap();
    parser.push("<tool_call>").unwrap();
    assert!(parser.push(&"x".repeat(MAX_ARGUMENT_BYTES + 33)).is_err());
}

#[test]
fn malformed_protocol_mutations_remain_bounded_and_never_panic() {
    let raw = b"<tool_call>{\"name\":\"add\",\"arguments\":{\"a\":2,\"b\":2}}</tool_call>";
    let mut seed = 12345u64;
    for _ in 0..2000 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut bytes = raw.to_vec();
        let index = (seed as usize) % bytes.len();
        if seed & 1 == 0 {
            bytes.remove(index);
        } else {
            bytes[index] = (seed as u8) % 128;
        }
        if let Ok(raw) = std::str::from_utf8(&bytes)
            && let Ok((_, calls, _)) = parse(raw, 3, config())
        {
            config()
                .validate_output(
                    &calls,
                    if calls.is_empty() {
                        FinishReason::Stop
                    } else {
                        FinishReason::ToolCalls
                    },
                )
                .unwrap();
        }
    }
}

#[test]
fn arguments_preserve_large_numbers_whitespace_and_escaped_unicode_without_repair() {
    let arguments = r#"{ "large":123456789012345678901234567890, "zero":-0, "escaped":"\u09ac\u09be", "literal":"🌍" }"#;
    let raw = format!("<tool_call>{{\"name\":\"add\",\"arguments\":{arguments}}}</tool_call>");
    let (_, calls, deltas) = parse(&raw, 1, config()).unwrap();
    assert_eq!(calls[0].function.arguments, arguments);
    assert_eq!(
        deltas
            .iter()
            .map(|d| d.function.arguments.as_str())
            .collect::<String>(),
        arguments
    );
    validate_call(&calls[0]).unwrap();
    let mut parser = ToolParser::new(config(), "cancel".into()).unwrap();
    parser.push("<tool_call>").unwrap();
    assert_eq!(
        parser.finish(FinishReason::Cancelled).unwrap_err().code,
        coheryn_engine::ErrorCode::Cancelled
    );
}

#[test]
fn native_decode_preserves_only_tool_delimiters_under_explicit_protocol_policy() {
    use coheryn_engine::{Cancellation, GenerationOptions, ParseLimits, inspect};
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("markers.gguf");
    std::fs::write(&path, common::tool_marker_model()).unwrap();
    let descriptor = inspect(&path, ParseLimits::default()).unwrap();
    let tensor = descriptor
        .tensors
        .iter()
        .find(|t| t.name == "output.weight")
        .unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    let start = (descriptor.data_offset + tensor.offset) as usize;
    bytes[start..start + tensor.bytes as usize].fill(0);
    for value in bytes[start + 256 * 4 * 4..start + 257 * 4 * 4]
        .as_chunks_mut::<4>()
        .0
    {
        value.copy_from_slice(&1f32.to_le_bytes());
    }
    std::fs::write(&path, bytes).unwrap();
    let model = Model::load(
        path,
        ResourceLimits {
            max_memory: 64 << 20,
            context: 128,
        },
    )
    .unwrap();
    assert!(model.tokenizer.is_tool_marker(256));
    assert!(model.tokenizer.is_tool_marker(257));
    assert!(!model.tokenizer.is_tool_marker(258));
    let plain = model
        .generate(
            "Hi",
            &GenerationOptions {
                max_tokens: 1,
                ..Default::default()
            },
            &Cancellation::default(),
            |_| Ok(()),
        )
        .unwrap();
    let tools = model
        .generate(
            "Hi",
            &GenerationOptions {
                max_tokens: 1,
                tool_markers: true,
                ..Default::default()
            },
            &Cancellation::default(),
            |_| Ok(()),
        )
        .unwrap();
    assert_eq!(plain.token_ids, vec![256]);
    assert_eq!(plain.token_ids, tools.token_ids);
    assert_eq!(plain.text, "");
    assert_eq!(tools.text, "<tool_call>");
}

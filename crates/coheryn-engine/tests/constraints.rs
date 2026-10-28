mod common;
use coheryn_engine::{
    Cancellation, ErrorCode, FinishReason, GenerationOptions, Model, ResourceLimits,
    constraints::{ConstraintDecoder, ResponseFormat, SchemaRequest},
    tools::{ToolConfig, ToolParser},
};
use serde_json::{Value, json};

fn format() -> ResponseFormat {
    ResponseFormat::JsonSchema {
        json_schema: SchemaRequest {
            name: "answer".into(),
            strict: true,
            schema: json!({"type":"object","properties":{"answer":{"type":"string","const":"ok"}},"required":["answer"],"additionalProperties":false}),
        },
    }
}
fn model(tools: bool, memory: u64) -> (tempfile::TempDir, Model) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    std::fs::write(
        &path,
        if tools {
            common::tool_marker_model()
        } else {
            common::tiny_model()
        },
    )
    .unwrap();
    let m = Model::load(
        path,
        ResourceLimits {
            context: 128,
            max_memory: memory,
        },
    )
    .unwrap();
    (dir, m)
}
#[test]
fn constrained_native_generation_matches_stream_validates_and_releases_memory() {
    let (_dir, m) = model(false, 64 << 20);
    let baseline = m.memory().reserved;
    let mut stream = String::new();
    let result = m
        .generate(
            "json",
            &GenerationOptions {
                max_tokens: 64,
                response_format: format(),
                ..Default::default()
            },
            &Cancellation::default(),
            |event| {
                stream.push_str(&event.text);
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&result.text).unwrap(),
        json!({"answer":"ok"})
    );
    assert_eq!(result.text, stream);
    assert_eq!(result.finish_reason, FinishReason::Stop);
    assert!(result.constraint_validation.unwrap().valid);
    assert_eq!(m.memory().reserved, baseline);
    assert!(m.memory().peak_session_buffer_capacity_bytes >= 16 << 20);
    assert_eq!(m.memory().session_buffer_capacity_bytes, 0);
}
#[test]
fn incomplete_cancelled_incompatible_and_unaffordable_constraints_fail_without_leaks() {
    let (_dir, m) = model(false, 64 << 20);
    let baseline = m.memory().reserved;
    let options = GenerationOptions {
        max_tokens: 1,
        response_format: format(),
        ..Default::default()
    };
    assert_eq!(
        m.generate("json", &options, &Cancellation::default(), |_| Ok(()))
            .unwrap_err()
            .code,
        ErrorCode::ConstraintIncomplete
    );
    assert_eq!(m.memory().reserved, baseline);
    let cancel = Cancellation::default();
    let options = GenerationOptions {
        max_tokens: 64,
        response_format: format(),
        ..Default::default()
    };
    assert_eq!(
        m.generate("json", &options, &cancel, |_| {
            cancel.cancel();
            Ok(())
        })
        .unwrap_err()
        .code,
        ErrorCode::Cancelled
    );
    assert_eq!(m.memory().reserved, baseline);
    assert_eq!(
        m.generate(
            "json",
            &GenerationOptions {
                stop: vec!["}".into()],
                ..options
            },
            &Cancellation::default(),
            |_| Ok(())
        )
        .unwrap_err()
        .code,
        ErrorCode::UnsupportedFeature
    );
    let (_dir, small) = model(false, 16 << 20);
    let baseline = small.memory().reserved;
    assert_eq!(
        small
            .generate(
                "json",
                &GenerationOptions {
                    max_tokens: 64,
                    response_format: format(),
                    ..Default::default()
                },
                &Cancellation::default(),
                |_| Ok(())
            )
            .unwrap_err()
            .code,
        ErrorCode::ResourceExhausted
    );
    assert_eq!(small.memory().reserved, baseline);
}
#[test]
fn byte_trie_masks_utf8_fragments_and_control_tokens_using_real_token_bytes() {
    let (_dir, m) = model(false, 64 << 20);
    let grammar = ResponseFormat::JsonObject.compile().unwrap().unwrap();
    let mut decoder =
        ConstraintDecoder::new(grammar, &m.tokenizer, &Cancellation::default()).unwrap();
    decoder.advance(b"{\"x\":\"").unwrap();
    decoder.advance(&[0xf0]).unwrap();
    let mask = decoder.mask(&Cancellation::default()).unwrap();
    for (id, allowed) in mask.iter().enumerate() {
        if m.tokenizer.is_special(id as u32) || m.tokenizer.is_eog(id as u32) {
            assert!(!allowed);
            continue;
        }
        let piece = m.tokenizer.token_bytes(id as u32).unwrap();
        if piece.len() == 1 {
            assert_eq!(*allowed, (0x90..=0xbf).contains(&piece[0]));
        }
    }
    decoder.advance(&[0x9f, 0x8c, 0x8d, b'"', b'}']).unwrap();
    assert!(decoder.complete());
    decoder.validate_output("{\"x\":\"🌍\"}").unwrap();
}
#[test]
fn required_single_strict_tool_is_constrained_through_its_markers_and_arguments() {
    let (_dir, m) = model(true, 64 << 20);
    let config:ToolConfig=serde_json::from_value(json!({"tools":[{"type":"function","function":{"name":"answer","strict":true,"parameters":{"type":"object","properties":{"value":{"type":"integer","const":4}},"required":["value"],"additionalProperties":false}}}],"tool_choice":"required","parallel_tool_calls":false})).unwrap();
    config.validate().unwrap();
    let mut parser = ToolParser::new(config, "call_strict".into()).unwrap();
    let mut raw = String::new();
    let cfg:ToolConfig=serde_json::from_value(json!({"tools":[{"type":"function","function":{"name":"answer","strict":true,"parameters":{"type":"object","properties":{"value":{"type":"integer","const":4}},"required":["value"],"additionalProperties":false}}}],"tool_choice":"required","parallel_tool_calls":false})).unwrap();
    let result = m
        .generate(
            "call",
            &GenerationOptions {
                max_tokens: 100,
                tool_markers: true,
                strict_tool: true,
                response_format: cfg.strict_format().unwrap(),
                ..Default::default()
            },
            &Cancellation::default(),
            |event| {
                raw += &event.text;
                parser.push(&event.text)?;
                Ok(())
            },
        )
        .unwrap();
    let (_, calls, _) = parser.finish(result.finish_reason).unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "answer");
    assert_eq!(
        serde_json::from_str::<Value>(&calls[0].function.arguments).unwrap(),
        json!({"value":4})
    );
    assert!(raw.starts_with("<tool_call>\n") && raw.ends_with("\n</tool_call>"));
    assert_eq!(
        result.constraint_validation.unwrap().target,
        "tool_arguments"
    );
}

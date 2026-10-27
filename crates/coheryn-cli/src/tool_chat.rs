//! Headless native chat requests. Suggested functions are returned as data only.
use crate::{Cli, daemon, print_json};
use coheryn_api::wire::GenerationRequest;
use coheryn_engine::{
    Cancellation, Error, Result, TokenEvent,
    tools::{ParsedEvent, ToolParser},
};
use serde::de::DeserializeOwned;
use serde_json::json;
use std::{io::Read, path::Path};
pub(crate) fn read_json<T: DeserializeOwned>(path: &Path, limit: usize) -> Result<T> {
    let file = std::fs::File::open(path)?;
    if !file.metadata()?.is_file() || file.metadata()?.len() > limit as u64 {
        return Err(Error::resource(
            "request file is not regular or exceeds its byte limit",
        ));
    }
    let mut bytes = Vec::new();
    file.take(limit as u64 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err(Error::resource("request file grew beyond its byte limit"));
    }
    serde_json::from_slice(&bytes).map_err(|e| Error::request(format!("invalid request JSON: {e}")))
}
pub async fn run(cli: &Cli, args: &daemon::ChatArgs, cancel: &Cancellation) -> Result<()> {
    let mut value: serde_json::Value = read_json(args.request.as_ref().unwrap(), 1 << 20)?;
    let id = if args.standalone {
        "standalone"
    } else {
        args.model
            .to_str()
            .ok_or_else(|| Error::request("managed model ID must be UTF-8"))?
    };
    let object = value
        .as_object_mut()
        .ok_or_else(|| Error::request("request must be a JSON object"))?;
    if !args.standalone && object.get("model").is_some_and(|v| v.as_str() != Some(id)) {
        return Err(Error::request("request model differs from CLI model"));
    }
    object.insert("model".into(), json!(id));
    let mut request: GenerationRequest = serde_json::from_value(value)
        .map_err(|e| Error::request(format!("invalid chat request: {e}")))?;
    if request.messages.is_none() {
        return Err(Error::request("chat --request requires messages"));
    }
    request.context = Some(request.context.unwrap_or(args.resources.context));
    request.max_memory = Some(
        request
            .max_memory
            .unwrap_or(args.resources.max_memory)
            .min(args.resources.max_memory),
    );
    if args.stream {
        request.stream = true;
    }
    request.validate().map_err(|e| e.error)?;
    let stream = request.stream;
    let emit = |token: TokenEvent| -> Result<()> {
        if stream {
            print_json(&json!({"token":token}))?;
        }
        Ok(())
    };
    let tool = |delta| -> Result<()> {
        if stream {
            print_json(&json!({"tool_call":delta}))?;
        }
        Ok(())
    };
    let result = if args.standalone {
        let mut limits = args.resources.limits();
        limits.context = request.context.unwrap_or(limits.context);
        limits.max_memory = request.max_memory.unwrap();
        let model = coheryn_engine::Model::load(&args.model, limits)?;
        let config = request.tool_config();
        let prompt = model
            .tokenizer
            .render_tools(request.messages.as_ref().unwrap(), &config)?;
        if (config.enabled()
            || request
                .messages
                .as_ref()
                .unwrap()
                .iter()
                .any(|m| m.role == "tool" || !m.tool_calls.is_empty()))
            && !coheryn_engine::tools::qualified_model_hash(&coheryn_engine::storage::sha256_file(
                &args.model,
            )?)
        {
            return Err(Error::new(
                coheryn_engine::ErrorCode::UnsupportedFeature,
                "model bytes are not qualified for qwen-tools-v1",
            ));
        }
        let mut parser = if config.enabled() {
            Some(ToolParser::new(
                config,
                format!("call_{}", crate::sessions::fresh_key()),
            )?)
        } else {
            None
        };
        let mut result = daemon::generate_local(
            &model,
            &prompt,
            &request.options(),
            (&request.profile, request.threads),
            cancel,
            |event| {
                if let Some(p) = &mut parser {
                    for e in p.push(&event.text)? {
                        match e {
                            ParsedEvent::Text(text) => emit(TokenEvent {
                                token_id: None,
                                text,
                            })?,
                            ParsedEvent::Delta(delta) => tool(delta)?,
                        }
                    }
                    Ok(())
                } else {
                    emit(event)
                }
            },
        )?;
        if let Some(parser) = parser {
            let (text, calls, tail) = parser.finish(result.finish_reason)?;
            if !tail.is_empty() {
                emit(TokenEvent {
                    token_id: None,
                    text: tail,
                })?;
            }
            result.text = text;
            if !calls.is_empty() {
                result.finish_reason = coheryn_engine::FinishReason::ToolCalls;
            }
            result.tool_calls = calls;
        }
        result
    } else {
        daemon::client(cli)?
            .generate_with_tools(&request, cancel, emit, tool)
            .await?
    };
    if stream {
        print_json(&json!({"result":result}))
    } else {
        print_json(&result)
    }
}

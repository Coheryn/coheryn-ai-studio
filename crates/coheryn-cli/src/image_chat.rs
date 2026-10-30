use crate::{
    Cli,
    daemon::{self, ChatArgs},
    print_json,
};
use coheryn_engine::{
    Cancellation, Error, Result,
    memory::Budget,
    multimodal::{
        image::ImageLimits,
        qwen2vl::{self, ChatContent, ChatMessage, ChatPart, Qwen2Vl},
        vision::VisionOptions,
    },
};
use serde_json::json;
use std::{
    fs::File,
    io::{self, Read, Write},
};

pub async fn run(cli: &Cli, args: &ChatArgs, cancel: &Cancellation) -> Result<()> {
    let limits = ImageLimits::default();
    if args.images.len() > limits.max_images {
        return Err(Error::resource("too many image attachments"));
    }
    if args.standalone
        && args.vision_model.is_none()
        && args.model.extension().is_some_and(|e| e == "gguf")
    {
        return Err(Error::request(
            "standalone image chat requires --vision-model",
        ));
    }
    let prompt = if let Some(s) = &args.prompt {
        s.clone()
    } else {
        let mut s = String::new();
        io::stdin().take((64 << 10) + 1).read_to_string(&mut s)?;
        s
    };
    if prompt.is_empty() || prompt.len() > 64 << 10 {
        return Err(Error::request(
            "image prompt requires 1..=65536 UTF-8 bytes",
        ));
    }
    if args.session.is_some() {
        return managed(cli, args, prompt, cancel).await;
    }
    let options = coheryn_engine::GenerationOptions {
        max_tokens: args.max_tokens,
        ..args.controls.options()?
    };
    let mut controls = coheryn_api::wire::GenerationRequest::completion(
        args.model.to_string_lossy().into(),
        prompt.clone(),
        &options,
        args.resources.context,
        args.stream,
    );
    // Local paths belong only to explicit standalone loading; the remote request always uses a managed ID.
    if args.standalone {
        controls.model = "standalone-image".into();
    }
    controls.prompt = None;
    controls.messages = Some(vec![]);
    controls.stop = None; // The ordinary CLI default represents no custom stops as an empty vector.
    if !options.stop.is_empty() {
        return Err(Error::request(
            "custom image stop sequences are not qualified",
        ));
    }
    controls.profile = args.profile.clone();
    controls.threads = args.threads;
    controls.max_memory = Some(args.resources.max_memory);
    let mut images = Vec::new();
    let mut total = 0u64;
    let mut parts = Vec::new();
    for path in &args.images {
        cancel.check()?;
        let metadata = std::fs::symlink_metadata(path)?;
        if !metadata.is_file()
            || metadata.file_type().is_symlink()
            || metadata.len() > limits.max_encoded_bytes
        {
            return Err(Error::request(
                "image attachment must be a regular non-symlink file within the byte limit",
            ));
        }
        let mut bytes = Vec::new();
        File::open(path)?
            .take(limits.max_encoded_bytes + 1)
            .read_to_end(&mut bytes)?;
        total = total
            .checked_add(bytes.len() as u64)
            .filter(|n| {
                *n <= limits.max_total_encoded_bytes
                    && bytes.len() as u64 <= limits.max_encoded_bytes
            })
            .ok_or_else(|| Error::resource("image attachments exceed aggregate byte limit"))?;
        let url = if args.standalone {
            coheryn_engine::multimodal::image::inspect(&bytes, limits, cancel)?;
            "local-attachment".to_owned()
        } else {
            coheryn_api::vision::data_url(&bytes, cancel)?
        };
        parts.push(json!({"type":"image_url","image_url":{"url":url}}));
        images.push(bytes);
    }
    parts.push(json!({"type":"text","text":prompt}));
    let mut messages = Vec::new();
    if let Some(system) = &args.system {
        messages.push(json!({"role":"system","content":system}));
    }
    messages.push(json!({"role":"user","content":parts}));
    let mut value = serde_json::to_value(&controls).map_err(|e| Error::request(e.to_string()))?;
    value["messages"] = json!(messages);
    coheryn_api::vision::VisionRequest::from_value(value.clone()).map_err(|e| e.error)?;
    let mut emit = |event: coheryn_engine::TokenEvent| -> Result<()> {
        if args.stream {
            if cli.json {
                println!("{}", json!({"token":event}));
            } else {
                let mut out = io::stdout().lock();
                write!(out, "{}", event.text)?;
                out.flush()?;
            }
        }
        Ok(())
    };
    if args.standalone {
        let mut native = Vec::new();
        if let Some(system) = &args.system {
            native.push(ChatMessage {
                role: "system".into(),
                content: ChatContent::Text(system.clone()),
            });
        }
        let mut parts: Vec<_> = (0..images.len())
            .map(|index| ChatPart::Image { index })
            .collect();
        parts.push(ChatPart::Text { text: prompt });
        native.push(ChatMessage {
            role: "user".into(),
            content: ChatContent::Parts(parts),
        });
        let profile = coheryn_engine::planner::Profile::parse(&args.profile)?;
        let threads = coheryn_engine::planner::execution_threads(
            profile,
            args.threads,
            std::thread::available_parallelism().map_or(1, |n| n.get()),
            1,
        )?;
        let cancel = cancel.with_timeout(options.timeout_seconds)?;
        let budget = Budget::new(args.resources.max_memory);
        let _transport = budget.reserve(8 << 20)?;
        let response = match &args.vision_model {
            None => {
                let model = coheryn_engine::qwen35::Model::load_with_budget(
                    &args.model,
                    args.resources.limits(),
                    budget,
                    &cancel,
                )?;
                let hardware = coheryn_api::hardware::probe();
                let prepared = model.prepare_images(
                    &native,
                    images,
                    limits,
                    options.clone(),
                    args.resources.context,
                    coheryn_engine::adapters::Execution {
                        profile,
                        threads: Some(threads),
                        hardware: &hardware,
                    },
                    1,
                    &cancel,
                )?;
                let mut state = model.begin_images(prepared, &cancel, None, None)?;
                while !state.is_done() {
                    if state.needs_forward() {
                        model.step_batch(&mut [&mut state], None)?;
                    }
                    if let Some(event) = state.sample(&model)? {
                        emit(event)?;
                    }
                }
                model.finish_generation(state)?.0
            }
            Some(vision_model) => {
                let model = Qwen2Vl::load(
                    &args.model,
                    vision_model,
                    args.resources.limits(),
                    budget,
                    &cancel,
                )?;
                let refs: Vec<_> = images.iter().map(Vec::as_slice).collect();
                let response = model.generate(
                    &native,
                    &refs,
                    &qwen2vl::Options {
                        context: Some(args.resources.context),
                        max_tokens: args.max_tokens,
                        threads,
                        vision: VisionOptions {
                            threads,
                            ..Default::default()
                        },
                    },
                    &cancel,
                    &mut emit,
                )?;
                coheryn_api::vision::generation_result(response)
            }
        };
        if cli.json {
            if args.stream {
                println!("{}", json!({"result":response}));
            } else {
                print_json(&response)?;
            }
        } else if !args.stream {
            println!("{}", response.text);
        }
    } else {
        drop(images); // Encoded attachments are now owned by the wire request.
        let result = daemon::client(cli)?
            .vision_generate(&value, cancel, &mut emit)
            .await?;
        if cli.json {
            if args.stream {
                println!("{}", json!({"result":result}));
            } else {
                print_json(&result)?;
            }
        } else if !args.stream {
            println!("{}", result.text);
        }
    }
    if args.stream && !cli.json {
        println!();
    }
    Ok(())
}

async fn managed(cli: &Cli, args: &ChatArgs, prompt: String, cancel: &Cancellation) -> Result<()> {
    let session = crate::sessions::interactive_session(cli, args, cancel)
        .await?
        .ok_or_else(|| Error::request("managed image chat requires --session"))?;
    let client = daemon::client(cli)?;
    let mut revision = crate::sessions::revision(&client, &session, cancel).await?;
    let mut ids = Vec::new();
    for path in &args.images {
        let attachment = client
            .attach_session_image(&session, path, revision, cancel)
            .await?;
        revision = attachment["revision"]
            .as_u64()
            .ok_or_else(|| Error::request("invalid attachment revision"))?;
        ids.push(
            attachment["image"]["id"]
                .as_str()
                .ok_or_else(|| Error::request("invalid attachment ID"))?
                .to_owned(),
        );
    }
    let request = coheryn_api::sessions::SubmitTurn {
        images: ids,
        expected_revision: revision,
        idempotency_key: crate::sessions::fresh_key(),
        content: prompt,
        tool_results: vec![],
        settings: coheryn_api::sessions::TurnSettings {
            profile: args.profile.clone(),
            threads: args.threads,
            max_memory: Some(args.resources.max_memory),
            ..args.controls.turn_settings(args.max_tokens)?
        },
        stream: args.stream,
    };
    if !args.stream && cli.json {
        let value = client
            .json(
                "POST",
                &format!("/coheryn/v1/sessions/{session}/messages"),
                Some(
                    serde_json::to_value(request)
                        .map_err(|_| Error::request("invalid image turn"))?,
                ),
                false,
                cancel,
            )
            .await?;
        return print_json(&value);
    }
    let result = client
        .session_generate(&session, &request, cancel, |token| {
            if args.stream {
                if cli.json {
                    print_json(&json!({"token":token}))?;
                } else {
                    let mut out = io::stdout().lock();
                    write!(out, "{}", token.text)?;
                    out.flush()?;
                }
            }
            Ok(())
        })
        .await?;
    if cli.json {
        print_json(&json!({"result":result}))?;
    } else if args.stream {
        println!();
    } else {
        println!("{}", result.text);
    }
    Ok(())
}

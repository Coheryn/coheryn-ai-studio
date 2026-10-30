//! Native-only decoder qualification runner. Expected outputs are inert fixture data.
use coheryn_engine::{
    Cancellation, Error, GenerationOptions, ResourceLimits, Result, qwen35::Model,
    tokenizer::Message,
};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs::{File, create_dir_all},
    io::{Read, Write},
    path::PathBuf,
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    messages: Vec<Message>,
    #[serde(default)]
    lifecycle: bool,
    #[serde(default)]
    teacher_tokens: Vec<u32>,
    #[serde(default)]
    trace_directory: Option<PathBuf>,
    #[serde(default = "max_tokens")]
    max_tokens: usize,
    #[serde(default = "threads")]
    threads: usize,
}
fn max_tokens() -> usize {
    4
}
fn threads() -> usize {
    1
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err(Error::request(
            "usage: qwen35_probe CHECKPOINT REQUEST_JSON",
        ));
    }
    let mut source = Vec::new();
    File::open(&args[1])?
        .take((1 << 20) + 1)
        .read_to_end(&mut source)?;
    if source.len() > 1 << 20 {
        return Err(Error::request("probe input exceeds one MiB"));
    }
    let request: Request =
        serde_json::from_slice(&source).map_err(|e| Error::request(e.to_string()))?;
    if request.teacher_tokens.len() > 256 {
        return Err(Error::request("too many teacher tokens"));
    }
    let cancel = Cancellation::default();
    let model = Model::load(
        &args[0],
        ResourceLimits {
            max_memory: 768 << 20,
            context: 256,
        },
        &cancel,
    )?;
    let prompt = model.render(&request.messages)?;
    let prompt_ids = model.encode(&prompt)?;
    let mut checkpoints = Vec::new();
    if let Some(directory) = &request.trace_directory {
        create_dir_all(directory)?;
        let mut tokens = prompt_ids.clone();
        tokens.extend_from_slice(&request.teacher_tokens);
        model.trace(&tokens,request.threads,&cancel,&mut |position,name,data| {
            if position+1<prompt_ids.len(){return Ok(());}
            if position+1>prompt_ids.len()&&name!="logits"{return Ok(());}
            let path=directory.join(format!("{position}-{name}.f32"));
            let mut bytes=Vec::with_capacity(data.len()*4);
            for value in data{bytes.extend_from_slice(&value.to_le_bytes());}
            let hash=Sha256::digest(&bytes);let sha256=hash.iter().map(|b|format!("{b:02x}")).collect::<String>();
            File::create(&path)?.write_all(&bytes)?;
            checkpoints.push(json!({"position":position,"name":name,"length":data.len(),"path":path,"sha256":sha256}));Ok(())
        })?;
    }
    let resident = model.memory();
    let mut events = Vec::new();
    let result = model.generate(
        &prompt,
        &GenerationOptions {
            max_tokens: request.max_tokens,
            ..Default::default()
        },
        request.threads,
        &cancel,
        &mut |e| {
            events.push(e);
            Ok(())
        },
    )?;
    let after = model.memory();
    let mut lifecycle = Vec::new();
    if request.lifecycle {
        let baseline = model.memory().reserved;
        let stopped = Cancellation::default();
        stopped.cancel();
        let error = model
            .generate(
                &prompt,
                &GenerationOptions::default(),
                1,
                &stopped,
                &mut |_| Ok(()),
            )
            .unwrap_err();
        if error.code != coheryn_engine::ErrorCode::Cancelled {
            return Err(error);
        }
        lifecycle.push(json!({"case":"pre-cancel","error":error}));
        let error = model
            .trace(&[prompt_ids[0]], 1, &cancel, &mut |_, _, _| {
                Err(Error::request("probe observer failure"))
            })
            .unwrap_err();
        if error.code != coheryn_engine::ErrorCode::InvalidRequest {
            return Err(error);
        }
        lifecycle.push(json!({"case":"observer-failure","error":error}));
        let during = Cancellation::default();
        let error = model
            .trace(&[prompt_ids[0]], 2, &during, &mut |_, _, _| {
                during.cancel();
                Ok(())
            })
            .unwrap_err();
        if error.code != coheryn_engine::ErrorCode::Cancelled {
            return Err(error);
        }
        lifecycle.push(json!({"case":"in-compute-cancel","error":error}));
        let error = model
            .generate(
                &prompt,
                &GenerationOptions {
                    max_tokens: 256,
                    ..Default::default()
                },
                1,
                &cancel,
                &mut |_| Ok(()),
            )
            .unwrap_err();
        if error.code != coheryn_engine::ErrorCode::ContextLimitExceeded {
            return Err(error);
        }
        lifecycle.push(json!({"case":"context-reservation-overflow","error":error}));
        let mut sampled = GenerationOptions::default();
        sampled.sampling.temperature = 0.7;
        let error = model
            .generate(&prompt, &sampled, 1, &cancel, &mut |_| Ok(()))
            .unwrap_err();
        if error.code != coheryn_engine::ErrorCode::UnsupportedFeature {
            return Err(error);
        }
        lifecycle.push(json!({"case":"unsupported-sampling","error":error}));
        if model.memory().reserved != baseline || model.memory().session_buffer_capacity_bytes != 0
        {
            return Err(Error::resource("probe detected unreleased session state"));
        }
    }
    println!("{}",serde_json::to_string(&json!({"prompt":prompt,"prompt_token_ids":prompt_ids,"config":model.configuration(),"checkpoints":checkpoints,"generation":result,"events":events,"resident":resident,"after":after,"lifecycle":lifecycle})).map_err(|e|Error::request(e.to_string()))?);
    Ok(())
}

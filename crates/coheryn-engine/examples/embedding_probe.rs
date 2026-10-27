//! Development-only E1 qualification driver, not the E2 CLI/API contract.
use coheryn_engine::{
    Cancellation, Error, ResourceLimits, Result,
    embedding::{EmbeddingModel, EmbeddingOptions},
};
use serde::Deserialize;
use serde_json::json;
use std::{
    io::{Read, Write},
    path::PathBuf,
    process::ExitCode,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    inputs: Vec<String>,
    #[serde(default)]
    options: EmbeddingOptions,
    #[serde(default = "memory_limit")]
    max_memory: u64,
    #[serde(default)]
    trace: bool,
    #[serde(default)]
    tokenize_only: bool,
}
fn memory_limit() -> u64 {
    256 << 20
}
fn emit(value: &impl serde::Serialize) -> Result<()> {
    let mut out = std::io::stdout().lock();
    serde_json::to_writer(&mut out, value).map_err(|e| Error::request(e.to_string()))?;
    writeln!(out)?;
    Ok(())
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 {
        return Err(Error::request("usage: embedding_probe MODEL REQUEST.json"));
    }
    let mut bytes = Vec::new();
    let file = std::fs::File::open(PathBuf::from(&args[1]))?;
    if file.metadata()?.len() > 9 << 20 {
        return Err(Error::request("probe request exceeds nine MiB"));
    }
    file.take((9 << 20) + 1).read_to_end(&mut bytes)?;
    let request: Request =
        serde_json::from_slice(&bytes).map_err(|e| Error::request(e.to_string()))?;
    let model = EmbeddingModel::load(
        PathBuf::from(&args[0]),
        ResourceLimits {
            max_memory: request.max_memory,
            context: request.options.max_input_tokens,
        },
    )?;
    let cancel = Cancellation::default();
    if request.tokenize_only {
        let tokens: Vec<_> = request
            .inputs
            .iter()
            .map(|text| model.tokenize(text, &request.options, &cancel))
            .collect::<Result<_>>()?;
        return emit(&json!({"event":"tokens","tokens":tokens}));
    }
    let inputs: Vec<_> = request.inputs.iter().map(String::as_str).collect();
    let result = if request.trace {
        let last = format!("ffn_out-{}", model.config().layers - 1);
        model.embed_observed(&inputs,&request.options,&cancel,&mut |frame| {
            if frame.name=="ffn_out-0" || frame.name==last || frame.name=="result_embd" || frame.name=="pooled" {
                emit(&json!({"event":"checkpoint","name":frame.name,"first_input":frame.first_input,"sequence_lengths":frame.sequence_lengths,"width":frame.width,"rows":frame.values.len()/frame.width,"values":frame.values}))?;
            }
            Ok(())
        })?
    } else {
        model.embed(&inputs, &request.options, &cancel)?
    };
    emit(&json!({"event":"result","result":result}))?;
    drop(result);
    emit(&json!({"event":"cleanup","memory":model.memory()}))
}
fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{}", serde_json::to_string(&e).unwrap());
            ExitCode::FAILURE
        }
    }
}

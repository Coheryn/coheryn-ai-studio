//! Development-only descriptor inventory from a bounded prefix and publisher-declared length.
//! This cannot qualify or execute a tensor payload: the remainder is never downloaded here.
use coheryn_engine::{Error, ParseLimits, Result, gguf};
use serde_json::json;
use std::{
    fs::File,
    io::{Cursor, Read},
};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err(Error::request(
            "usage: inspect_header PREFIX DECLARED_TOTAL_BYTES",
        ));
    }
    let length = args[1]
        .parse::<u64>()
        .map_err(|_| Error::request("invalid declared length"))?;
    let file = File::open(&args[0])?;
    if file.metadata()?.len() > 16 << 20 {
        return Err(Error::resource("prefix exceeds 16 MiB"));
    }
    let mut prefix = Vec::new();
    file.take((16 << 20) + 1).read_to_end(&mut prefix)?;
    let d = gguf::parse(Cursor::new(prefix), length, ParseLimits::default())?;
    let processor_metadata: serde_json::Map<_, _> = d
        .metadata
        .iter()
        .filter(|(key, _)| {
            key.starts_with("clip.")
                || key.starts_with("qwen2vl.")
                || key.starts_with("qwen2_5_vl.")
                || key.starts_with("llama.")
                || key.starts_with("nomic-bert.")
                || key.starts_with("bert.")
                || key.starts_with("general.")
                || matches!(
                    key.as_str(),
                    "tokenizer.ggml.model"
                        | "tokenizer.ggml.pre"
                        | "tokenizer.ggml.bos_token_id"
                        | "tokenizer.ggml.eos_token_id"
                        | "tokenizer.ggml.padding_token_id"
                        | "tokenizer.ggml.unknown_token_id"
                        | "tokenizer.ggml.seperator_token_id"
                        | "tokenizer.ggml.mask_token_id"
                        | "tokenizer.ggml.add_bos_token"
                        | "tokenizer.ggml.add_eos_token"
                )
        })
        .map(|(key, value)| {
            Ok((
                key.clone(),
                serde_json::to_value(value).map_err(|e| Error::invalid(e.to_string()))?,
            ))
        })
        .collect::<Result<_>>()?;
    serde_json::to_writer_pretty(std::io::stdout(), &json!({"descriptor":d,
        "architecture_and_processor_metadata":processor_metadata,"tensor_payload_verified":false,
        "note":"Header inventory only; tensor extents use a publisher-declared length, not downloaded payload evidence"}))
        .map_err(|e|Error::invalid(e.to_string()))?;
    Ok(())
}

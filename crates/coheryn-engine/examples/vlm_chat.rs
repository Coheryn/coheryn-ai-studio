//! Offline local-file image-to-text runner. No daemon or network acquisition.
use coheryn_engine::{
    Cancellation, Error, ResourceLimits, Result,
    memory::Budget,
    multimodal::qwen2vl::{ChatMessage, Options, Qwen2Vl},
};
use serde::Deserialize;
use serde_json::json;
use std::{
    fs::{File, create_dir_all},
    io::{BufWriter, Read, Write},
    path::PathBuf,
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    #[serde(default)]
    cooperative: bool,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    images: Vec<PathBuf>,
    #[serde(default)]
    options: Options,
    #[serde(default = "context")]
    context: usize,
    #[serde(default = "max_memory")]
    max_memory: u64,
    timeout_seconds: Option<u64>,
    trace_directory: Option<PathBuf>,
    #[serde(default)]
    stream: bool,
}
fn context() -> usize {
    256
}
fn max_memory() -> u64 {
    768 << 20
}
fn bounded(path: &PathBuf, limit: u64) -> Result<Vec<u8>> {
    let file = File::open(path)?;
    let bytes = file.metadata()?.len();
    if bytes > limit {
        return Err(Error::resource("runner input exceeds byte limit"));
    }
    let mut out = Vec::new();
    out.try_reserve_exact(bytes as usize)
        .map_err(|_| Error::resource("runner input allocation failed"))?;
    file.take(limit + 1).read_to_end(&mut out)?;
    if out.len() as u64 > limit {
        return Err(Error::resource("runner input grew beyond byte limit"));
    }
    Ok(out)
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err(Error::request(
            "usage: vlm_chat LANGUAGE_GGUF VISION_GGUF REQUEST_JSON",
        ));
    }
    let request: Request = serde_json::from_slice(&bounded(&PathBuf::from(&args[2]), 256 << 10)?)
        .map_err(|e| Error::request(e.to_string()))?;
    if request.images.len() > 4 {
        return Err(Error::request("runner supports at most four images"));
    }
    let mut encoded = Vec::new();
    let mut total = 0usize;
    for path in &request.images {
        let bytes = bounded(path, 16 << 20)?;
        total = total
            .checked_add(bytes.len())
            .filter(|n| *n <= 32 << 20)
            .ok_or_else(|| Error::resource("runner images exceed aggregate byte limit"))?;
        encoded.push(bytes);
    }
    let budget = Budget::new(request.max_memory);
    let cancel = Cancellation::default().with_timeout(request.timeout_seconds)?;
    let model = Qwen2Vl::load(
        &args[0],
        &args[1],
        ResourceLimits {
            max_memory: request.max_memory,
            context: request.context,
        },
        budget.clone(),
        &cancel,
    )?;
    let mut emit = |event| {
        if request.stream {
            writeln!(
                std::io::stderr(),
                "{}",
                serde_json::to_string(&event).map_err(|e| Error::request(e.to_string()))?
            )?;
        }
        Ok(())
    };
    let prepared = model.prepare_owned(&request.messages, encoded, &request.options, &cancel)?;
    let mut yielded = 0u64;
    let mut cooperate = || {
        yielded += 1;
        std::thread::yield_now();
        Ok(())
    };
    let response = if let Some(directory) = request.trace_directory {
        create_dir_all(&directory)?;
        model.trace_prepared(
            prepared,
            &cancel,
            &mut |checkpoint| {
                let mut out = BufWriter::new(File::create(
                    directory.join(format!("{}.f32", checkpoint.name)),
                )?);
                for value in checkpoint.values {
                    out.write_all(&value.to_le_bytes())?;
                }
                out.flush()?;
                Ok(())
            },
            if request.cooperative {
                Some(&mut cooperate)
            } else {
                None
            },
            &mut emit,
        )?
    } else if request.cooperative {
        model.generate_cooperative(prepared, &cancel, &mut cooperate, &mut emit)?
    } else {
        model.generate_prepared(prepared, &cancel, &mut emit)?
    };
    let output = json!({"scope":"offline native Qwen2-VL CPU image-to-text development runner","response":response,"memory_while_response_retained":budget.stats(),"product_image_cli_api_available":true,"cooperative_yields":yielded});
    drop(response);
    drop(model);
    println!(
        "{}",
        json!({"result":output,"memory_after_release":budget.stats()})
    );
    Ok(())
}

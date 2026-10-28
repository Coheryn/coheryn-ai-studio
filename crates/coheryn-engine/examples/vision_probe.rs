//! Native image-to-embedding diagnostics; never executes a language model.
use coheryn_engine::{
    Cancellation, Error, Result,
    memory::Budget,
    multimodal::{
        image::{self, ImageLimits, Processor},
        vision::{VisionModel, VisionOptions},
    },
};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs::{File, create_dir_all},
    io::{BufWriter, Read, Write},
    path::PathBuf,
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    image: PathBuf,
    #[serde(default)]
    options: VisionOptions,
    #[serde(default = "budget")]
    max_memory: u64,
    trace_directory: Option<PathBuf>,
}
fn budget() -> u64 {
    128 << 20
}
fn read_bounded(path: &PathBuf, limit: u64) -> Result<Vec<u8>> {
    let file = File::open(path)?;
    let bytes = file.metadata()?.len();
    if bytes > limit {
        return Err(Error::resource("diagnostic input exceeds byte limit"));
    }
    let mut data = Vec::new();
    data.try_reserve_exact(bytes as usize)
        .map_err(|_| Error::resource("diagnostic read allocation failed"))?;
    file.take(limit + 1).read_to_end(&mut data)?;
    if data.len() as u64 > limit {
        return Err(Error::resource(
            "diagnostic input changed beyond byte limit",
        ));
    }
    Ok(data)
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err(Error::request(
            "usage: vision_probe VISION_GGUF REQUEST_JSON",
        ));
    }
    let request: Request = serde_json::from_slice(&read_bounded(&PathBuf::from(&args[1]), 65536)?)
        .map_err(|e| Error::request(e.to_string()))?;
    let budget = Budget::new(request.max_memory);
    let cancel = Cancellation::default();
    let model = VisionModel::load(&args[0], budget.clone(), &cancel)?;
    let encoded = read_bounded(&request.image, ImageLimits::default().max_encoded_bytes)?;
    let prepared = image::prepare(
        &encoded,
        Processor::Qwen2VlStill,
        ImageLimits::default(),
        &budget,
        &cancel,
    )?;
    let result = if let Some(directory) = request.trace_directory {
        create_dir_all(&directory)?;
        model.trace(&prepared, &request.options, &cancel, &mut |checkpoint| {
            let mut file = BufWriter::new(File::create(
                directory.join(format!("{}.f32", checkpoint.name)),
            )?);
            for value in checkpoint.values {
                file.write_all(&value.to_le_bytes())?;
            }
            file.flush()?;
            Ok(())
        })?
    } else {
        model.encode(&prepared, &request.options, &cancel)?
    };
    let mut hash = Sha256::new();
    for value in result.embeddings() {
        hash.update(value.to_le_bytes());
    }
    let mut sum = 0f64;
    let mut square = 0f64;
    for value in result.embeddings() {
        sum += f64::from(*value);
        square += f64::from(*value) * f64::from(*value);
    }
    let output = json!({"scope":"native Qwen2-VL vision encoder and merger only","image_to_text_supported":false,"model_sha256":result.model_sha256,"config":model.config(),"plan":result.plan,"encoding_ms":result.encoding_ms,"sha256":coheryn_engine::storage::digest_hex(hash.finalize().into()),"count":result.embeddings().len(),"sum":sum,"l2":square.sqrt(),"memory_while_output_retained":budget.stats()});
    drop(result);
    drop(prepared);
    drop(model);
    println!(
        "{}",
        json!({"result":output,"memory_after_release":budget.stats()})
    );
    Ok(())
}

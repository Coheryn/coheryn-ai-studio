use clap::Parser;
use coheryn_engine::{Cancellation, GenerationOptions, Model, ResourceLimits};
use std::{path::PathBuf, time::Instant};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value = "Hello")]
    prompt: String,
    #[arg(long, default_value_t = 8)]
    max_tokens: usize,
    #[arg(long, default_value_t = 512)]
    context: usize,
    #[arg(long, default_value_t = 1073741824)]
    max_memory: u64,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let model_sha256 = coheryn_engine::storage::sha256_file(&args.model)?;
    let start = Instant::now();
    let model = Model::load(
        &args.model,
        ResourceLimits {
            max_memory: args.max_memory,
            context: args.context,
        },
    )?;
    let load_ms = start.elapsed().as_secs_f64() * 1000.;
    let result = model.generate(
        &args.prompt,
        &GenerationOptions {
            max_tokens: args.max_tokens,
            ..GenerationOptions::default()
        },
        &Cancellation::default(),
        |_| Ok(()),
    )?;
    println!(
        "{}",
        serde_json::json!({
            "schema_version":1,"engine_version":env!("CARGO_PKG_VERSION"),"compiler":env!("COHERYN_RUSTC"),
            "backend":"cpu-reference","os":std::env::consts::OS,"architecture":std::env::consts::ARCH,"threads":1,
            "context":args.context,"kv_dtype":"F32","profile":"reference","cache_condition":"uncontrolled",
            "model_file":args.model,"model_sha256":model_sha256,"model_file_bytes":model.descriptor.file_bytes,
            "header_sha256":model.descriptor.header_sha256,"tokenizer_fingerprints":model.descriptor.metadata_fingerprints,
            "load_ms":load_ms,"generation":result,
            "measurements":{"rss":null,"private_bytes":null,"mapped_bytes":0,"page_faults":null,"swap":null,"io_bytes":null,"gpu_bytes":null},
            "limitations":["allocation reservations are not allocator measurements","OS counters unavailable in portable harness","model hashing warms file cache"]
        })
    );
    Ok(())
}

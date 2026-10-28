//! Offline image-stage diagnostics only; no vision encoder or generation.
use coheryn_engine::{Cancellation, Error, Result, memory::Budget, multimodal::image::*};
use serde_json::json;
use std::{fs::File, io::Read, time::Instant};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err(Error::request(
            "usage: prepare_image PROFILE LOCAL_PNG_OR_JPEG",
        ));
    }
    let profile = Processor::parse(&args[0])?;
    let limits = ImageLimits::default();
    let file = File::open(&args[1])?;
    let capacity = file.metadata()?.len();
    if capacity > limits.max_encoded_bytes {
        return Err(Error::resource("image exceeds encoded byte ceiling"));
    }
    let mut encoded = Vec::new();
    encoded
        .try_reserve_exact(capacity as usize)
        .map_err(|_| Error::resource("input allocation failed"))?;
    file.take(limits.max_encoded_bytes + 1)
        .read_to_end(&mut encoded)?;
    let budget = Budget::new(256 << 20);
    let start = Instant::now();
    let prepared = prepare(&encoded, profile, limits, &budget, &Cancellation::default())?;
    let output = json!({"scope":"native image preprocessing only", "image_understanding_supported":false,
        "plan":prepared.plan(), "processed_rgb_sha256":prepared.processed_rgb_sha256(),
        "elapsed_seconds":start.elapsed().as_secs_f64(), "memory_while_retained":budget.stats()});
    drop(prepared);
    println!(
        "{}",
        json!({"result":output,"memory_after_release":budget.stats()})
    );
    Ok(())
}

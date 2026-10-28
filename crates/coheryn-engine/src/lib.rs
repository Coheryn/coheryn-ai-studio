//! Independent, portable reference CPU execution. No external inference runtime.
pub mod compat;
pub mod constraints;
pub mod cpu;
pub mod embedding;
pub mod error;
pub mod gguf;
pub mod inference;
pub mod memory;
pub mod multimodal;
mod parallel;
pub mod planner;
pub mod quant;
pub mod sampling;
pub mod storage;
mod tensor;
pub mod tokenizer;
pub mod tools;

pub use compat::{CompatibilityReport, ModelConfig, ModelTask, validate};
pub use error::{Error, ErrorCode, Result};
pub use gguf::{Descriptor, ParseLimits, inspect};
pub use inference::{
    Cancellation, FinishReason, GenerationOptions, GenerationResult, Model, TokenEvent,
};
pub use memory::ResourceLimits;

/// Validate the task-specific tokenizer without treating an encoder as a decoder.
pub fn validate_tokenizer(d: &Descriptor) -> Result<()> {
    if d.architecture.as_deref() == Some("nomic-bert") {
        embedding::EmbeddingTokenizer::from_descriptor(d).map(drop)
    } else {
        tokenizer::TextTokenizer::from_descriptor(d)?
            .render(&[tokenizer::Message {
                role: "user".into(),
                content: "Hi".into(),
                ..Default::default()
            }])
            .map(drop)
    }
}

/// Scan all executable float/packed payloads with bounded workspace. Does not infer.
pub fn deep_validate(
    path: impl AsRef<std::path::Path>,
    descriptor: &Descriptor,
    cancel: &Cancellation,
) -> Result<()> {
    let report = validate(descriptor);
    if !report.executable {
        return Err(report
            .diagnostics
            .into_iter()
            .next()
            .expect("compatibility diagnostic"));
    }
    let storage = storage::Storage::open(path.as_ref(), descriptor)?;
    let mut bytes = vec![0u8; 256 << 10];
    let mut floats = memory::zeros(bytes.len() / 2)?;
    for tensor in &descriptor.tensors {
        let (_, block, size) = gguf::encoding(tensor.dtype).expect("validated dtype");
        let mut position = 0;
        while position < tensor.bytes {
            cancel.check()?;
            storage.unchanged()?;
            let blocks = ((tensor.bytes - position) / size)
                .min((bytes.len() as u64) / size)
                .min((floats.len() as u64) / block);
            let count = (blocks * size) as usize;
            storage.read(
                descriptor.data_offset + tensor.offset + position,
                &mut bytes[..count],
            )?;
            cpu::decode(
                &bytes[..count],
                tensor.dtype,
                &mut floats[..(blocks * block) as usize],
            )?;
            position += count as u64;
        }
    }
    Ok(())
}

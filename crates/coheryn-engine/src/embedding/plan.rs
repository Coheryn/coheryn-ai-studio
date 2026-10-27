use super::{EmbeddingConfig, EmbeddingOptions, tokenizer::MAX_TEXT_BYTES};
use crate::{
    Descriptor, Error, Result,
    memory::{self, MemoryStats},
    planner::{HardwareSnapshot, Profile},
};
use serde::Serialize;

pub const MAX_INPUTS: usize = 256;
pub const MAX_REQUEST_BYTES: usize = 8 << 20;

#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingPlan {
    pub schema_version: u32,
    pub task: &'static str,
    pub backend: &'static str,
    pub profile: Profile,
    pub input_count: usize,
    pub max_input_tokens: usize,
    pub batch_size: usize,
    pub token_capacity: usize,
    pub tensor_tile_tokens: usize,
    pub threads: usize,
    pub io_bytes: usize,
    pub output_bytes: u64,
    pub tokenization_bytes: u64,
    pub workspace_bytes: u64,
    pub request_reserved_bytes: u64,
    pub safety_headroom_bytes: u64,
    pub shared_weight_bytes: u64,
    pub additional_reserved_bytes: u64,
    pub required_reserved_bytes: u64,
    pub admissible: bool,
    pub reasons: Vec<String>,
    pub limitations: Vec<&'static str>,
}
impl EmbeddingPlan {
    pub fn require_admission(&self) -> Result<()> {
        if self.admissible {
            Ok(())
        } else {
            Err(Error::resource(self.reasons.join("; ")))
        }
    }
}
pub struct EmbeddingPlanInput<'a> {
    pub descriptor: &'a Descriptor,
    pub config: &'a EmbeddingConfig,
    pub context: usize,
    pub input_bytes: &'a [usize],
    pub options: &'a EmbeddingOptions,
    pub budget: MemoryStats,
    pub hardware: &'a HardwareSnapshot,
    pub max_concurrency: usize,
}
pub(crate) fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| Error::resource("embedding estimate overflow"))
}
pub fn model_bytes(d: &Descriptor, c: &EmbeddingConfig) -> Result<u64> {
    d.data_offset
        .checked_mul(16)
        .and_then(|n| n.checked_add((c.vocabulary as u64).checked_mul(512)?))
        .ok_or_else(|| Error::resource("embedding metadata estimate overflow"))
}
pub(crate) fn workspace_bytes(
    c: &EmbeddingConfig,
    tokens: usize,
    context: usize,
    io: usize,
    threads: usize,
    tile: usize,
) -> Result<u64> {
    let states = memory::product(&[
        tokens,
        c.hidden
            .checked_mul(6)
            .and_then(|n| n.checked_add(c.intermediate * 2))
            .ok_or_else(|| Error::resource("embedding workspace overflow"))?,
        4,
    ])?;
    let weights = memory::product(&[4, c.hidden, 4])?;
    let attention = memory::product(&[
        threads,
        context
            .checked_mul(4)
            .and_then(|n| n.checked_add(c.head_dim * 8))
            .ok_or_else(|| Error::resource("embedding attention overflow"))?,
    ])?;
    let activation = memory::product(&[
        tile,
        c.hidden
            .max(c.intermediate)
            .checked_mul(4)
            .and_then(|n| n.checked_add(c.hidden.max(c.intermediate).div_ceil(256) * 292 + 128))
            .ok_or_else(|| Error::resource("embedding activation overflow"))?,
    ])?;
    [
        states,
        weights,
        attention,
        activation,
        3 * io,
        memory::worker_bytes(threads)?,
        2 << 20,
    ]
    .into_iter()
    .try_fold(0u64, |sum, n| add(sum, n as u64))
}
pub fn estimate(input: EmbeddingPlanInput<'_>) -> Result<EmbeddingPlan> {
    let EmbeddingPlanInput {
        descriptor: d,
        config: c,
        context,
        input_bytes: lengths,
        options,
        budget,
        hardware,
        max_concurrency,
    } = input;
    c.check_bounds()?;
    if context > c.qualified_context {
        return Err(Error::request(
            "embedding planning context exceeds qualified configuration",
        ));
    }
    options.validate(context)?;
    if lengths.is_empty()
        || lengths.len() > MAX_INPUTS
        || lengths.iter().any(|n| *n == 0 || *n > MAX_TEXT_BYTES)
        || hardware.logical_cpus == 0
        || !(1..=16).contains(&max_concurrency)
    {
        return Err(Error::request(
            "embedding request exceeds bounded input/concurrency limits",
        ));
    }
    let total_bytes = lengths.iter().try_fold(0usize, |a, n| {
        a.checked_add(*n)
            .filter(|n| *n <= MAX_REQUEST_BYTES)
            .ok_or_else(|| Error::request("embedding request exceeds eight MiB"))
    })?;
    let share = (hardware.logical_cpus / max_concurrency).clamp(1, 16);
    let automatic = match options.profile {
        Profile::Reference | Profile::LowMemory => 1,
        Profile::Balanced => share.min(2),
        Profile::Performance => share.min(4),
    };
    let threads = options.threads.unwrap_or(automatic);
    if threads > share
        || (matches!(options.profile, Profile::Reference | Profile::LowMemory) && threads != 1)
    {
        return Err(Error::request(
            "embedding workers exceed the CPU share/profile",
        ));
    }
    let largest_row = d.tensors.iter().try_fold(0usize, |n, t| {
        let width = t
            .dimensions
            .first()
            .copied()
            .and_then(|n| usize::try_from(n).ok())
            .filter(|n| *n > 0 && *n <= 32768)
            .ok_or_else(|| Error::invalid("embedding tensor row exceeds planning bounds"))?;
        Ok::<_, Error>(n.max(crate::quant::row_bytes(t.dtype, width)?))
    })?;
    let io_bytes = if options.profile == Profile::LowMemory {
        largest_row.max(16 << 10).next_power_of_two()
    } else {
        256 << 10
    };
    let tile = if options.profile == Profile::LowMemory {
        4
    } else if options.profile == Profile::Performance {
        32
    } else {
        16
    };
    let output_bytes = memory::product(&[lengths.len(), c.hidden * 4 + 256])? as u64;
    let tokenization_bytes = add(
        memory::product(&[*lengths.iter().max().unwrap(), 128])? as u64,
        memory::product(&[lengths.len(), options.max_input_tokens * 4 + 64])? as u64,
    )?;
    let request_bytes = add(
        add(output_bytes, tokenization_bytes)?,
        total_bytes as u64 + (1 << 20),
    )?;
    let mut capacity = options
        .max_batch_tokens
        .min(options.batch_size.min(lengths.len()) * options.max_input_tokens);
    if options.profile == Profile::LowMemory {
        capacity = options.max_input_tokens;
    }
    let (workspace, headroom, additional) = loop {
        let work = workspace_bytes(
            c,
            capacity,
            options.max_input_tokens,
            io_bytes,
            threads,
            tile,
        )?;
        let subtotal = add(request_bytes, work)?;
        let percent = match options.profile {
            Profile::Reference => 0,
            Profile::Balanced => 10,
            _ => 5,
        };
        let headroom = subtotal
            .checked_mul(percent)
            .ok_or_else(|| Error::resource("embedding headroom overflow"))?
            .div_ceil(100);
        let additional = add(subtotal, headroom)?;
        let fits = add(budget.reserved, additional)? <= budget.limit
            && hardware.ram_available_bytes.is_none_or(|n| additional <= n);
        if fits || capacity == options.max_input_tokens {
            break (work, headroom, additional);
        }
        capacity = (capacity / 2).max(options.max_input_tokens);
    };
    let required = add(budget.reserved, additional)?;
    let mut reasons = Vec::new();
    if required > budget.limit {
        reasons.push(format!("embedding request needs {additional} additional reserved bytes; {required} exceeds engine limit {}", budget.limit));
    }
    if hardware.ram_available_bytes.is_some_and(|n| additional > n) {
        reasons.push("embedding allocation estimate exceeds observed available RAM".into());
    }
    let admissible = reasons.is_empty();
    if admissible {
        reasons.push(format!("admitted up to {capacity} packed tokens per microbatch; input limits and dimensions are preserved"));
    }
    Ok(EmbeddingPlan {
        schema_version: 1,
        task: "embedding",
        backend: "cpu-reference",
        profile: options.profile,
        input_count: lengths.len(),
        max_input_tokens: options.max_input_tokens,
        batch_size: options.batch_size.min(lengths.len()),
        token_capacity: capacity,
        tensor_tile_tokens: tile,
        threads,
        io_bytes,
        output_bytes,
        tokenization_bytes,
        workspace_bytes: workspace,
        request_reserved_bytes: request_bytes,
        safety_headroom_bytes: headroom,
        shared_weight_bytes: d.tensor_bytes,
        additional_reserved_bytes: additional,
        required_reserved_bytes: required,
        admissible,
        reasons,
        limitations: vec![
            "Estimates reserve engine capacity; they are not allocator instrumentation or an OS-wide RAM cap",
            "Tokenizer library allocations are bounded by a 64 KiB input cap and a conservative allowance",
            "The encoder stores no generation KV cache or quadratic attention matrix",
            "CPU only; hardware RAM may be unavailable in the convenience library entrypoint",
        ],
    })
}

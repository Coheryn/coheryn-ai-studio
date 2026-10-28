//! Pure, versioned planning from validated metadata and an explicit hardware snapshot.
//! Estimates describe reservations, not allocator measurements or an OS-wide RAM cap.
use crate::{
    Descriptor, Error, ModelConfig, Result,
    memory::{self, MemoryStats},
};
use serde::{Deserialize, Serialize};

pub const REQUEST_BYTES: u64 = 8 << 20;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Profile {
    Reference,
    Performance,
    #[default]
    Balanced,
    LowMemory,
}
impl Profile {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "reference" => Ok(Self::Reference),
            "performance" => Ok(Self::Performance),
            "balanced" => Ok(Self::Balanced),
            "low-memory" => Ok(Self::LowMemory),
            _ => Err(Error::request(
                "profile must be reference, performance, balanced or low-memory",
            )),
        }
    }
    pub fn concurrency_limit(self, configured: usize) -> usize {
        match self {
            Self::LowMemory => 1,
            Self::Balanced => configured.min(2),
            _ => configured,
        }
    }
    fn headroom_percent(self) -> u64 {
        match self {
            Self::Reference => 0,
            Self::Performance => 5,
            Self::Balanced => 10,
            Self::LowMemory => 5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareSnapshot {
    pub os: String,
    pub architecture: String,
    pub logical_cpus: usize,
    pub cpu_features: Vec<String>,
    pub ram_total_bytes: Option<u64>,
    pub ram_available_bytes: Option<u64>,
    pub swap_total_bytes: Option<u64>,
    pub swap_used_bytes: Option<u64>,
    pub observed_unix_ms: u64,
    /// No GPU backend is qualified yet. Null is deliberately distinct from zero.
    pub gpu_available_bytes: Option<u64>,
    pub gpu_backend_qualified: bool,
}

/// Shared CPU policy for text, image and metadata-only execution planning.
pub fn execution_threads(
    profile: Profile,
    threads: Option<usize>,
    logical_cpus: usize,
    max_concurrency: usize,
) -> Result<usize> {
    let cpu_share = (logical_cpus / max_concurrency.max(1)).clamp(1, 16);
    let automatic_threads = match profile {
        Profile::Reference | Profile::LowMemory => 1,
        Profile::Balanced => cpu_share.min(2),
        Profile::Performance => cpu_share.min(4),
    };
    let threads_per_session = threads.unwrap_or(automatic_threads);
    if threads_per_session == 0
        || threads_per_session > cpu_share
        || (matches!(profile, Profile::Reference | Profile::LowMemory) && threads_per_session != 1)
    {
        return Err(Error::request(
            "threads exceed the per-session CPU share or the reference/low-memory single-thread contract",
        ));
    }
    Ok(threads_per_session)
}

/// M0/M1 boundary: caller-provided dimensions cannot invent an executable vision graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageDescriptor {
    pub width: u32,
    pub height: u32,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageWorkload {
    pub images: Vec<ImageDescriptor>,
    pub processor_profile: String,
}
impl ImageWorkload {
    pub fn validate(&self) -> Result<()> {
        if self.images.is_empty()
            || self.images.len() > 16
            || self.processor_profile.is_empty()
            || self.processor_profile.len() > 128
            || self
                .images
                .iter()
                .any(|i| i.width == 0 || i.height == 0 || i.width > 32768 || i.height > 32768)
        {
            return Err(Error::request(
                "image estimate requires 1..=16 bounded dimensions and a processor profile",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SessionEstimate {
    pub kv_capacity_bytes: u64,
    pub workspace_bytes: u64,
    pub total_bytes: u64,
}
pub fn session_estimate(c: &ModelConfig, context: usize) -> Result<SessionEstimate> {
    session_estimate_with_io(c, context, 256 << 10)
}
pub fn session_estimate_with_io(
    c: &ModelConfig,
    context: usize,
    io_bytes: usize,
) -> Result<SessionEstimate> {
    let total = memory::session_bytes_with_io(c, context, io_bytes)?;
    let kv = memory::product(&[2, c.layers, c.kv_heads, c.head_dim, context, 4])? as u64;
    Ok(SessionEstimate {
        kv_capacity_bytes: kv,
        workspace_bytes: total - kv,
        total_bytes: total,
    })
}

pub struct PlanInput<'a> {
    pub constraints: bool,
    pub descriptor: &'a Descriptor,
    pub config: &'a ModelConfig,
    pub model: &'a str,
    pub profile: Profile,
    pub context: usize,
    pub concurrency: usize,
    pub max_concurrency: usize,
    pub loaded: bool,
    pub budget: MemoryStats,
    pub hardware: HardwareSnapshot,
    pub images: Option<&'a ImageWorkload>,
    pub threads: Option<usize>,
}
#[derive(Debug, Clone, Serialize)]
pub struct ExecutionPlan {
    /// Per-request grammar/trie reservation, separate from shared weights and KV.
    pub constraint_reserved_bytes: u64,
    pub schema_version: u32,
    pub model: String,
    pub profile: Profile,
    pub context: usize,
    pub concurrency: usize,
    pub concurrency_limit: usize,
    pub backend: &'static str,
    pub weight_storage: &'static str,
    pub cache_dtype: &'static str,
    pub threads_per_session: usize,
    pub prefill_chunk_tokens: usize,
    pub io_buffer_bytes: usize,
    pub shared_weight_bytes: u64,
    pub mapped_weight_bytes: u64,
    pub gpu_allocation_bytes: u64,
    pub cpu_gpu_transfer_bytes: u64,
    pub model_metadata_additional: u64,
    pub session: SessionEstimate,
    pub session_reserved_bytes: u64,
    pub request_reserved_bytes: u64,
    pub safety_headroom_bytes: u64,
    pub additional_reserved_bytes: u64,
    pub required_reserved_bytes: u64,
    pub conservative_host_working_set_additional_bytes: u64,
    pub host_memory_feasible: Option<bool>,
    pub host_allocation_feasible: Option<bool>,
    pub warnings: Vec<String>,
    pub memory: MemoryStats,
    pub hardware: HardwareSnapshot,
    pub image_stages: Option<UnqualifiedImageStages>,
    pub admissible: bool,
    pub reasons: Vec<String>,
    pub limitations: Vec<&'static str>,
}
#[derive(Debug, Clone, Serialize)]
pub struct UnqualifiedImageStages {
    pub image_count: usize,
    pub processor_profile: String,
    pub acquisition_bytes: Option<u64>,
    pub decoded_bytes: Option<u64>,
    pub preprocessing_bytes: Option<u64>,
    pub vision_workspace_bytes: Option<u64>,
    pub embedding_bytes: Option<u64>,
    pub effective_image_positions: Option<u64>,
    pub qualified: bool,
}
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| Error::resource("plan byte arithmetic overflow"))
}
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .ok_or_else(|| Error::resource("plan byte arithmetic overflow"))
}
impl ExecutionPlan {
    pub fn require_admission(&self) -> Result<()> {
        if self.admissible {
            Ok(())
        } else {
            Err(Error::resource(self.reasons.join("; ")))
        }
    }
}
pub fn estimate(input: PlanInput<'_>) -> Result<ExecutionPlan> {
    let PlanInput {
        constraints,
        descriptor: d,
        config: c,
        model,
        profile,
        context,
        concurrency,
        max_concurrency,
        loaded,
        budget,
        hardware,
        images,
        threads,
    } = input;
    if context == 0
        || context > c.context
        || concurrency == 0
        || concurrency > 16
        || max_concurrency == 0
        || max_concurrency > 16
        || hardware.logical_cpus == 0
    {
        return Err(Error::request(
            "context/concurrency exceed supported planning bounds",
        ));
    }
    if let Some(images) = images {
        images.validate()?;
    }
    let largest_row = d.tensors.iter().try_fold(0u64, |largest, tensor| {
        let width = tensor
            .dimensions
            .first()
            .ok_or_else(|| Error::invalid("missing tensor dimensions"))?;
        let row = crate::quant::row_bytes(tensor.dtype, *width as usize)? as u64;
        Ok::<u64, Error>(largest.max(row))
    })?;
    if largest_row > 256 << 10 {
        return Err(Error::request("tensor row exceeds bounded CPU workspace"));
    }
    let io_buffer_bytes = if profile == Profile::LowMemory {
        (largest_row.max(16 << 10) as usize).next_power_of_two()
    } else {
        256 << 10
    };
    let threads_per_session =
        execution_threads(profile, threads, hardware.logical_cpus, max_concurrency)?;
    let prefill_chunk_tokens = match profile {
        Profile::Reference => 1,
        Profile::LowMemory => 8,
        Profile::Balanced => 16,
        Profile::Performance => 32,
    };
    let mut session = session_estimate_with_io(c, context, io_buffer_bytes)?;
    session.workspace_bytes = add(
        session.workspace_bytes,
        memory::worker_bytes(threads_per_session)? as u64,
    )?;
    session.total_bytes = add(
        session.total_bytes,
        memory::worker_bytes(threads_per_session)? as u64,
    )?;
    let shared_weight_bytes = d
        .tensors
        .iter()
        .try_fold(0u64, |sum, t| add(sum, t.bytes))?;
    let metadata = if loaded {
        0
    } else {
        memory::model_bytes(d, c)?
    };
    let constraint_reserved_bytes = if constraints {
        crate::constraints::RESERVATION_BYTES
    } else {
        0
    };
    let per_request = add(
        add(session.total_bytes, REQUEST_BYTES)?,
        constraint_reserved_bytes,
    )?;
    let per_request_headroom = mul(per_request, profile.headroom_percent())?.div_ceil(100);
    let headroom = mul(per_request_headroom, concurrency as u64)?;
    let additional = add(
        metadata,
        mul(add(per_request, per_request_headroom)?, concurrency as u64)?,
    )?;
    let required = add(budget.reserved, additional)?;
    // Buffered weights have no engine-owned full-weight allocation. OS page-cache residency
    // is a separate conservative full-weight working-set upper bound, counted once per model.
    let host_additional = add(shared_weight_bytes, additional)?;
    let host_memory_feasible = hardware
        .ram_available_bytes
        .map(|available| host_additional <= available);
    let host_allocation_feasible = hardware
        .ram_available_bytes
        .map(|available| additional <= available);
    let mut warnings = Vec::new();
    if host_memory_feasible == Some(false) {
        warnings.push("Full-weight file-cache residency may not fit observed available RAM; bounded buffered reads remain usable and may incur additional I/O".into());
    }
    let concurrency_limit = profile.concurrency_limit(max_concurrency);
    let mut reasons = vec![];
    if required > budget.limit {
        reasons.push(format!(
            "need {additional} additional reserved bytes ({required} total); engine limit is {}",
            budget.limit
        ));
    }
    if concurrency > concurrency_limit {
        reasons.push(format!("requested concurrency {concurrency} exceeds profile/configured limit {concurrency_limit}; request fewer sessions explicitly"));
    }
    if host_allocation_feasible == Some(false) {
        reasons.push(format!("additional engine allocation reservation {additional} exceeds observed available RAM; reduce concurrency/context explicitly or use a smaller qualified model"));
    }
    let image_stages = images.map(|images| UnqualifiedImageStages {
        image_count: images.images.len(),
        processor_profile: images.processor_profile.clone(),
        acquisition_bytes: None,
        decoded_bytes: None,
        preprocessing_bytes: None,
        vision_workspace_bytes: None,
        embedding_bytes: None,
        effective_image_positions: None,
        qualified: false,
    });
    if image_stages.is_some() {
        reasons.push("vision graph and processor are not qualified; image stage estimates and execution are unavailable".into());
    }
    let admissible = reasons.is_empty();
    if admissible {
        reasons.push("request fits the atomic engine reservation budget and configured concurrency; context and sampling semantics are preserved".into());
    }
    Ok(ExecutionPlan {
        constraint_reserved_bytes,
        schema_version: 1,
        model: model.into(),
        profile,
        context,
        concurrency,
        concurrency_limit,
        backend: "cpu-reference",
        weight_storage: "read-only-buffered",
        cache_dtype: "f32",
        threads_per_session,
        prefill_chunk_tokens,
        io_buffer_bytes,
        shared_weight_bytes,
        mapped_weight_bytes: 0,
        gpu_allocation_bytes: 0,
        cpu_gpu_transfer_bytes: 0,
        model_metadata_additional: metadata,
        session_reserved_bytes: session.total_bytes,
        request_reserved_bytes: REQUEST_BYTES,
        safety_headroom_bytes: headroom,
        additional_reserved_bytes: additional,
        required_reserved_bytes: required,
        conservative_host_working_set_additional_bytes: host_additional,
        host_memory_feasible,
        host_allocation_feasible,
        warnings,
        session,
        memory: budget,
        hardware,
        image_stages,
        admissible,
        reasons,
        limitations: vec![
            "Reservations are conservative estimates, not allocator measurements or an OS-wide RAM cap",
            "Full-weight page-cache residency is an advisory upper bound, counted once; only engine allocation needs enter buffered-backend admission",
            "Available RAM is a point-in-time observation; only engine budget admission is atomic",
            "Row-parallel CPU workers preserve per-row arithmetic; packed inputs use Q8_0/Q8_K activation quantization; prefill remains sequential within bounded chunks; F32 KV and CPU-only placement are retained",
            "Vision stage costs remain unavailable until a qualified model/processor manifest exists",
        ],
    })
}

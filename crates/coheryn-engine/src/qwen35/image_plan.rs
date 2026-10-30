//! Pure still-image costs shared by metadata planning and actual request admission.
use super::{Config, vision::VisionConfig};
use crate::{
    Error, Result,
    catalog::PlanRequest,
    memory::{self, MemoryStats},
    multimodal::image::{self, ImageFormat, ImageInfo, ImageLimits, ImagePlan, Processor},
    planner::{self, ExecutionPlan, HardwareSnapshot, ImageWorkload},
};
use serde::Serialize;
#[derive(Clone, Serialize)]
pub struct ImageStage {
    pub preprocessor: ImagePlan,
    pub encoder_reserved_bytes: u64,
    pub feature_embedding_bytes: u64,
}
#[derive(Clone, Serialize)]
pub struct ImageStages {
    pub images: Vec<ImageStage>,
    pub image_feature_tokens: usize,
    pub encoded_bytes_upper_bound: u64,
    #[serde(rename = "image_extra_reserved_bytes")]
    pub extra_reserved_bytes: u64,
}
#[derive(Serialize)]
pub struct ImageExecutionPlan {
    #[serde(flatten)]
    pub execution: ExecutionPlan,
    pub task: &'static str,
    #[serde(flatten)]
    pub images: ImageStages,
}
impl ImageExecutionPlan {
    pub fn require_admission(&self) -> Result<()> {
        self.execution.require_admission()
    }
}
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| Error::resource("image plan byte overflow"))
}
/// Every stage remains reserved while queued; images execute sequentially. This conservative
/// sum includes retained output ownership without relying on a future recycling optimization.
pub fn image_stages(
    config: &VisionConfig,
    plans: &[ImagePlan],
    threads: usize,
    limits: ImageLimits,
) -> Result<ImageStages> {
    limits.validate()?;
    if plans.len() > limits.max_images {
        return Err(Error::new(
            crate::ErrorCode::ImageInputLimitExceeded,
            "image count exceeds operator limit",
        ));
    }
    let mut images = Vec::new();
    let mut extra = planner::REQUEST_BYTES;
    let mut encoded = 0;
    let mut features = 0usize;
    for p in plans {
        if p.profile != Processor::Qwen35Still {
            return Err(Error::request(
                "image processor differs from validated model profile",
            ));
        }
        encoded = add(encoded, p.info.encoded_bytes)?;
        features = features
            .checked_add(p.image_feature_tokens)
            .ok_or_else(|| Error::resource("image feature overflow"))?;
        let encoder = config.estimate(p.patch_rows, threads)?;
        let embedding = memory::product(&[p.image_feature_tokens, config.output, 4])? as u64;
        extra = add(
            extra,
            add(
                p.peak_reserved_bytes
                    .checked_sub(p.info.encoded_bytes)
                    .ok_or_else(|| Error::resource("invalid preprocessing stage"))?,
                encoder,
            )?,
        )?;
        images.push(ImageStage {
            preprocessor: p.clone(),
            encoder_reserved_bytes: encoder,
            feature_embedding_bytes: embedding,
        });
    }
    if features > limits.max_total_image_tokens {
        return Err(Error::new(
            crate::ErrorCode::ImageInputLimitExceeded,
            "aggregate image features exceed operator limit",
        ));
    }
    // Dimension-only inputs use each per-image ceiling; aggregate ingress is independently capped.
    let encoded_bytes_upper_bound = encoded.min(limits.max_total_encoded_bytes);
    Ok(ImageStages {
        images,
        image_feature_tokens: features,
        encoded_bytes_upper_bound,
        extra_reserved_bytes: add(extra, encoded_bytes_upper_bound)?,
    })
}
pub fn dimension_stages(
    config: &VisionConfig,
    workload: &ImageWorkload,
    limits: ImageLimits,
    threads: usize,
) -> Result<ImageStages> {
    workload.validate()?;
    if workload.processor_profile != Processor::Qwen35Still.name() {
        return Err(Error::new(
            crate::ErrorCode::UnsupportedImageProcessor,
            "Qwen3.5 image planning requires qwen35-still-v1",
        ));
    }
    let plans = workload
        .images
        .iter()
        .map(|i| {
            image::plan(
                ImageInfo {
                    format: ImageFormat::Png,
                    width: i.width,
                    height: i.height,
                    channels: 4,
                    orientation: 1,
                    encoded_bytes: limits.max_encoded_bytes,
                },
                Processor::Qwen35Still,
                limits,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    image_stages(config, &plans, threads, limits)
}
#[allow(clippy::too_many_arguments)]
pub fn estimate_images(
    config: &Config,
    vision: &VisionConfig,
    weights: u64,
    request: &PlanRequest,
    hardware: &HardwareSnapshot,
    budget: MemoryStats,
    loaded: bool,
    max_concurrency: usize,
    workload: &ImageWorkload,
    limits: ImageLimits,
) -> Result<ImageExecutionPlan> {
    let execution = super::estimate_shared(
        config,
        weights,
        request,
        hardware,
        budget,
        loaded,
        max_concurrency,
    )?;
    let images = dimension_stages(vision, workload, limits, execution.threads_per_session)?;
    let mut plan = ImageExecutionPlan {
        execution,
        task: "image-to-text",
        images,
    };
    let extra = plan
        .images
        .extra_reserved_bytes
        .checked_mul(request.concurrency as u64)
        .ok_or_else(|| Error::resource("image concurrency overflow"))?;
    plan.execution.request_reserved_bytes = add(
        plan.execution.request_reserved_bytes,
        plan.images.extra_reserved_bytes,
    )?;
    plan.add_capacity(extra)?;
    if plan.images.image_feature_tokens >= request.context {
        plan.reject("image features leave no capacity for text and output tokens");
    }
    plan.execution.image_stages = Some(planner::UnqualifiedImageStages {
        image_count: plan.images.images.len(),
        processor_profile: Processor::Qwen35Still.name().into(),
        acquisition_bytes: Some(plan.images.encoded_bytes_upper_bound),
        decoded_bytes: Some(
            plan.images
                .images
                .iter()
                .map(|i| i.preprocessor.decoder_stage_bytes)
                .max()
                .unwrap_or(0),
        ),
        preprocessing_bytes: Some(
            plan.images
                .images
                .iter()
                .map(|i| i.preprocessor.peak_reserved_bytes)
                .max()
                .unwrap_or(0),
        ),
        vision_workspace_bytes: Some(
            plan.images
                .images
                .iter()
                .map(|i| i.encoder_reserved_bytes)
                .max()
                .unwrap_or(0),
        ),
        embedding_bytes: Some(
            plan.images
                .images
                .iter()
                .map(|i| i.feature_embedding_bytes)
                .sum(),
        ),
        effective_image_positions: Some(plan.images.image_feature_tokens as u64),
        qualified: true,
    });
    plan.execution.limitations = vec![
        "Validated native CPU still-image profile; exact-file numerical evidence remains separate from structural compatibility",
        "Dimension-only estimates bound encoded input; exact text/output/physical context is checked at submission without resizing or context reduction",
        "All processor, encoder, feature, decoder and worker reservations are held before queueing; no image prefix cache",
        "Resident metadata and host working set are estimates, distinct from observed allocation/RSS; plans hold no reservation",
    ];
    Ok(plan)
}
impl ImageExecutionPlan {
    fn reject(&mut self, reason: &str) {
        if self.execution.admissible {
            self.execution.reasons.clear();
        }
        self.execution.admissible = false;
        if !self.execution.reasons.iter().any(|r| r == reason) {
            self.execution.reasons.push(reason.into());
        }
    }
    /// Add managed ingress/delivery outside the shared engine estimator, using the same
    /// checked totals and RAM feasibility rules. This never acquires a reservation.
    pub fn add_capacity(&mut self, bytes: u64) -> Result<()> {
        let p = &mut self.execution;
        p.additional_reserved_bytes = add(p.additional_reserved_bytes, bytes)?;
        p.required_reserved_bytes = add(p.required_reserved_bytes, bytes)?;
        p.conservative_host_working_set_additional_bytes =
            add(p.conservative_host_working_set_additional_bytes, bytes)?;
        p.host_allocation_feasible = p
            .hardware
            .ram_available_bytes
            .map(|n| p.additional_reserved_bytes <= n);
        p.host_memory_feasible = p
            .hardware
            .ram_available_bytes
            .map(|n| p.conservative_host_working_set_additional_bytes <= n);
        if p.required_reserved_bytes > p.memory.limit {
            self.reject("complete image workload exceeds the engine reservation budget");
        }
        if self.execution.host_allocation_feasible == Some(false) {
            self.reject("image workload allocation exceeds observed available RAM");
        }
        Ok(())
    }
}

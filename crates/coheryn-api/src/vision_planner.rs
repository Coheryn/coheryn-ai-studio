//! Metadata-only estimates for the qualified paired graph; never opens tensor views.
use crate::{
    error::{ApiError, Result},
    registry::ModelRecord,
    service::Service,
    wire::EstimateRequest,
};
use coheryn_engine::{
    Error, ParseLimits, memory,
    multimodal::{
        image::{self, ImageFormat, ImageInfo, Processor},
        qwen2vl,
        vision::{VisionConfig, VisionOptions},
    },
    planner::Profile,
};
use serde_json::{Value, json};

pub(crate) fn descriptors(
    service: &Service,
    record: &ModelRecord,
) -> Result<(coheryn_engine::Descriptor, coheryn_engine::Descriptor)> {
    let pair = record
        .components
        .as_ref()
        .ok_or_else(|| ApiError::bad("model is not a language/vision pair"))?;
    let limits = ParseLimits {
        max_metadata_entries: 512,
        max_tensors: 2048,
        max_array_items: 200000,
        max_string_bytes: 16384,
        max_header_bytes: 16 << 20,
        max_metadata_allocation: 64 << 20,
        max_depth: 2,
    };
    let language =
        coheryn_engine::inspect(service.registry.model_path(&pair.language.model)?, limits)?;
    let vision = coheryn_engine::inspect(service.registry.model_path(&pair.vision.model)?, limits)?;
    let c = qwen2vl::language_configuration(&language)?;
    qwen2vl::validate_language_metadata(&language)?;
    let v = VisionConfig::from_descriptor(&vision)?;
    if c.decoder.hidden != v.output_width {
        return Err(ApiError::bad("language and vision embedding widths differ"));
    }
    Ok((language, vision))
}
pub(crate) fn inspect(service: &Service, record: &ModelRecord, deep: bool) -> Result<Value> {
    let _inspection = service.budget.reserve(128 << 20)?;
    let (language, vision) = descriptors(service, record)?;
    if deep {
        let _tokenizer = service.budget.reserve(256 << 20)?;
        qwen2vl::validate_language_tokenizer(&language)?;
        let pair = record.components.as_ref().expect("checked pair");
        for component in [&pair.language, &pair.vision] {
            if coheryn_engine::storage::sha256_file(service.registry.model_path(&component.model)?)?
                != component.sha256
            {
                return Err(coheryn_engine::Error::new(
                    coheryn_engine::ErrorCode::ModelChanged,
                    "model pair component hash changed",
                )
                .into());
            }
        }
    }
    Ok(
        json!({"model":record,"task":"image-to-text","compatibility":{"executable":true,"scope":"qwen2vl-still-f16-cpu-greedy","language":qwen2vl::language_configuration(&language)?,"vision":VisionConfig::from_descriptor(&vision)?,"exact_file_qualification":"fixtures/compatibility.json"},"language_descriptor":language,"vision_descriptor":vision,"hashes_verified":deep}),
    )
}
pub(crate) fn estimate(
    service: &Service,
    request: EstimateRequest,
    record: ModelRecord,
    loaded: bool,
) -> Result<Value> {
    if request.task.as_deref() != Some("image-to-text")
        || request.constraints
        || request.input_bytes.is_some()
        || request.batch_size.is_some()
        || request.max_input_tokens.is_some()
        || request.max_batch_tokens.is_some()
    {
        return Err(ApiError::bad(
            "model pairs require task=image-to-text and image dimension descriptors; text/embedding constraints are unsupported",
        ));
    }
    let profile = Profile::parse(&request.profile)?;
    if request.concurrency == 0
        || request.concurrency > profile.concurrency_limit(service.scheduler.max_concurrency())
    {
        return Err(ApiError::bad(
            "image concurrency exceeds the selected scheduler profile",
        ));
    }
    let _inspection = service.budget.reserve(128 << 20)?;
    let (language, vision) = descriptors(service, &record)?;
    let c = qwen2vl::language_configuration(&language)?;
    let v = VisionConfig::from_descriptor(&vision)?;
    if request.context > c.decoder.context || request.context > 4096 {
        return Err(ApiError::bad(
            "image context exceeds qualified model capacity",
        ));
    }
    let workload = request
        .images
        .ok_or_else(|| ApiError::bad("image plans require image dimension descriptors"))?;
    if workload.processor_profile != "qwen2vl-still-v1"
        || workload.images.len() > service.config.vision.images.max_images
    {
        return Err(ApiError::bad("unsupported image processor or image count"));
    }
    let threads = coheryn_engine::planner::execution_threads(
        profile,
        request.threads,
        crate::hardware::probe().logical_cpus,
        service.config.max_concurrency,
    )?;
    let options = VisionOptions {
        threads,
        ..Default::default()
    };
    let limits = service.config.vision.images;
    let mut stages = Vec::new();
    let mut stage = 0;
    let mut tokens = 0;
    for i in workload.images {
        // Four channels and the per-image byte ceiling give a conservative dimension-only bound.
        // Dimensions describe the oriented image. Exact encoded bytes/tokenization are checked at submission.
        let p = image::plan(
            ImageInfo {
                format: ImageFormat::Png,
                width: i.width,
                height: i.height,
                channels: 4,
                orientation: 1,
                encoded_bytes: limits.max_encoded_bytes,
            },
            Processor::Qwen2VlStill,
            limits,
        )?;
        let encoder = v.plan_image(&p, &options)?;
        tokens += p.image_feature_tokens;
        stage = stage.max(p.peak_reserved_bytes).max(
            p.output_reserved_bytes
                .checked_add(encoder.peak_reserved_bytes)
                .ok_or_else(|| Error::resource("vision estimate overflow"))?,
        );
        stages.push(json!({"preprocessor":p,"encoder":encoder}));
    }
    if tokens > limits.max_total_image_tokens {
        return Err(Error::resource("image features exceed operator limit").into());
    }
    let decoder = memory::session_bytes_for(&c.decoder, request.context, 256 << 10, threads)?;
    let per_request = decoder
        .checked_add(stage)
        .and_then(|n| n.checked_add(limits.max_total_encoded_bytes))
        .and_then(|n| n.checked_add(16 << 20))
        .ok_or_else(|| Error::resource("image request estimate overflow"))?;
    let metadata = if loaded {
        0
    } else {
        memory::model_bytes(&language, &c.decoder)?
            .checked_add(vision.data_offset * 16 + vision.tensors.len() as u64 * 512 + (64 << 10))
            .ok_or_else(|| Error::resource("pair metadata overflow"))?
    };
    let ingress = service
        .config
        .vision
        .ingress_bytes(service.config.vision.body_bytes)?;
    let additional = per_request
        .checked_add(ingress)
        .and_then(|n| n.checked_mul(request.concurrency as u64))
        .and_then(|n| n.checked_add(metadata))
        .ok_or_else(|| Error::resource("image concurrency overflow"))?;
    let stats = service.budget.stats();
    let reserved_before_inspection = stats.reserved.saturating_sub(128 << 20);
    let required = reserved_before_inspection
        .checked_add(additional)
        .ok_or_else(|| Error::resource("image admission overflow"))?;
    let context_fits_images = tokens < request.context;
    let admissible = required <= stats.limit && context_fits_images;
    let mut reasons = Vec::new();
    if required > stats.limit {
        reasons
            .push("complete image request and ingress reservations exceed the configured budget");
    }
    if !context_fits_images {
        reasons.push("image features leave no room for text and output tokens");
    }
    Ok(
        json!({"schema_version":1,"task":"image-to-text","model":request.model,"backend":"cpu-reference","profile":request.profile,"context":request.context,"threads":threads,"concurrency":request.concurrency,"image_feature_tokens":tokens,"images":stages,"decoder_reserved_bytes":decoder,"request_reserved_bytes":per_request,"ingress_reserved_bytes":ingress,"model_metadata_additional_bytes":metadata,"weight_file_bytes":record.bytes,"weight_storage":"bounded buffered reads; file bytes are not an equivalent heap reservation","additional_reserved_bytes":additional,"required_reserved_bytes":required,"admissible":admissible,"rejection_reasons":reasons,"temporary_inspection_reserved_bytes":128u64<<20,"hardware":crate::hardware::probe(),"note":"Conservative dimension-only estimate. Exact text tokens, attachment bytes, physical/rotary capacity and atomic admission are checked before enqueueing; no automatic resizing or context reduction."}),
    )
}

pub(crate) fn estimate_qwen35(
    service: &Service,
    request: EstimateRequest,
    record: &ModelRecord,
    loaded: Option<&crate::resident::Resident>,
) -> Result<Value> {
    use coheryn_engine::{catalog::PlanRequest, qwen35};
    if request.task.as_deref() != Some("image-to-text")
        || request.constraints
        || request.input_bytes.is_some()
        || request.batch_size.is_some()
        || request.max_input_tokens.is_some()
        || request.max_batch_tokens.is_some()
    {
        return Err(ApiError::bad(
            "image planning requires image-to-text without text constraints or embedding fields",
        ));
    }
    let workload = request
        .images
        .as_ref()
        .ok_or_else(|| ApiError::bad("image plans require bounded dimension descriptors"))?;
    let budget = service.budget.stats();
    let hardware = crate::hardware::probe();
    let input = PlanRequest {
        context: request.context,
        concurrency: request.concurrency,
        max_memory: service.config.max_memory,
        profile: Profile::parse(&request.profile)?,
        threads: request.threads,
        constraints: false,
    };
    let mut inspection_bytes = 0;
    let mut plan = if let Some(crate::resident::Resident::Qwen35(model)) = loaded {
        qwen35::image_plan::estimate_images(
            model.configuration(),
            model.vision_configuration()?,
            model.weight_bytes()?,
            &input,
            &hardware,
            budget,
            true,
            service.scheduler.max_concurrency(),
            workload,
            service.config.vision.images,
        )?
    } else {
        inspection_bytes = qwen35::RESIDENT_METADATA_BYTES;
        let _inspection = service.budget.reserve(inspection_bytes)?;
        let mut inspection = record
            .checkpoint
            .as_ref()
            .ok_or_else(|| ApiError::bad("missing checkpoint manifest"))?
            .bind(
                &service.registry.checkpoint_root(&record.id)?,
                &Default::default(),
            )?;
        let vision = inspection.qwen35_vision_configuration(&Default::default())?;
        let config = qwen35::Config::parse(
            inspection
                .source_configuration()
                .expect("source configuration"),
        )?;
        qwen35::image_plan::estimate_images(
            &config,
            &vision,
            inspection.descriptor().tensor_bytes,
            &input,
            &hardware,
            budget,
            false,
            service.scheduler.max_concurrency(),
            workload,
            service.config.vision.images,
        )?
    };
    let transport = (8u64 << 20)
        .checked_mul(request.concurrency as u64)
        .ok_or_else(|| Error::resource("image transport overflow"))?;
    let ingress = service
        .config
        .vision
        .ingress_bytes(service.config.vision.body_bytes)?
        .checked_mul(request.concurrency as u64)
        .ok_or_else(|| Error::resource("image ingress overflow"))?;
    plan.add_capacity(
        transport
            .checked_add(ingress)
            .ok_or_else(|| Error::resource("image delivery/ingress overflow"))?,
    )?;
    plan.execution.model = request.model;
    let mut value = serde_json::to_value(plan).map_err(|e| Error::request(e.to_string()))?;
    value["transport_reserved_bytes"] = json!(transport);
    value["ingress_reserved_bytes"] = json!(ingress);
    value["temporary_inspection_reserved_bytes"] = json!(inspection_bytes);
    value["source_set_sha256"] = json!(record.sha256);
    value["payload_integrity"] = json!("not-rehashed-by-plan");
    Ok(value)
}

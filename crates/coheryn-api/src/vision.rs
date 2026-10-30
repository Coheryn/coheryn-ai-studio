//! Bounded image-chat transport. Image bytes never select a server filesystem path.
use crate::{
    error::{ApiError, Result},
    wire::GenerationRequest,
};
use base64::{Engine, engine::general_purpose::STANDARD};
use coheryn_engine::{
    Cancellation, Error, ErrorCode,
    multimodal::{
        image::{self, ImageFormat, ImageLimits},
        qwen2vl,
    },
};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;

#[derive(Debug, Clone, Serialize)]
pub struct VisionLimits {
    pub images: ImageLimits,
    pub body_bytes: usize,
    pub allow_urls: bool,
    pub acquisition_timeout_seconds: u64,
    pub redirects: usize,
}
impl Default for VisionLimits {
    fn default() -> Self {
        Self {
            images: ImageLimits::default(),
            body_bytes: 48 << 20,
            allow_urls: false,
            acquisition_timeout_seconds: 30,
            redirects: 3,
        }
    }
}
impl VisionLimits {
    pub fn validate(&self) -> coheryn_engine::Result<()> {
        self.images.validate()?;
        let d = ImageLimits::default();
        if self.images.max_encoded_bytes > d.max_encoded_bytes
            || self.images.max_total_encoded_bytes > d.max_total_encoded_bytes
            || self.images.max_images > d.max_images
            || self.images.max_edge > d.max_edge
            || self.images.max_source_pixels > d.max_source_pixels
            || self.images.max_processed_pixels > d.max_processed_pixels
            || self.images.max_image_tokens > d.max_image_tokens
            || self.images.max_total_image_tokens > d.max_total_image_tokens
            || !(1024..=64 << 20).contains(&self.body_bytes)
            || !(1..=120).contains(&self.acquisition_timeout_seconds)
            || self.redirects > 5
        {
            return Err(Error::request(
                "image transport limits exceed the qualified finite profile",
            ));
        }
        Ok(())
    }
    /// Covers wire bytes, JSON string storage, parsed values and decoded attachments.
    /// The engine separately reserves execution and retained output before queueing.
    pub fn ingress_bytes(&self, wire_bytes: usize) -> coheryn_engine::Result<u64> {
        (wire_bytes as u64)
            .checked_mul(4)
            .and_then(|n| n.checked_add(self.images.max_total_encoded_bytes))
            .and_then(|n| n.checked_add(16 << 20))
            .ok_or_else(|| Error::resource("image ingress size overflow"))
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Message {
    pub role: String,
    pub content: Content,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(#[serde(deserialize_with = "crate::bounded::sequence::<_,_,128>")] Vec<Part>),
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Part {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageUrl {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}
pub struct VisionRequest {
    pub controls: GenerationRequest,
    pub messages: Vec<Message>,
}
pub fn has_parts(value: &Value) -> bool {
    value["messages"]
        .as_array()
        .is_some_and(|m| m.iter().any(|m| m["content"].is_array()))
}
impl VisionRequest {
    pub fn from_value(mut value: Value) -> Result<Self> {
        let raw = value
            .get_mut("messages")
            .ok_or_else(|| ApiError::bad("image chat requires messages"))?
            .take();
        let messages: Vec<Message> = serde_json::from_value(raw)
            .map_err(|_| ApiError::bad("invalid multimodal messages"))?;
        if messages.is_empty() || messages.len() > 64 {
            return Err(ApiError::bad("image chat requires 1..=64 messages"));
        }
        // Validate every ordinary generation field through the shared strict schema.
        value["messages"] = serde_json::json!([{"role":"user","content":"image request"}]);
        let controls: GenerationRequest = serde_json::from_value(value)
            .map_err(|_| ApiError::bad("invalid or unsupported image generation field"))?;
        controls.validate()?;
        let request = Self { controls, messages };
        validate_controls(&request.controls)?;
        request.validate_structure()?;
        Ok(request)
    }
    fn validate_structure(&self) -> Result<()> {
        let mut count = 0;
        let mut text_bytes = 0usize;
        let mut skeleton = Vec::new();
        for m in &self.messages {
            if m.role.len() > 16 {
                return Err(ApiError::bad("invalid image chat role"));
            }
            let content = match &m.content {
                Content::Text(text) => {
                    text_bytes = text_bytes.saturating_add(text.len());
                    if text_bytes > 64 << 10 {
                        return Err(ApiError::bad("image chat text exceeds 64 KiB"));
                    }
                    qwen2vl::ChatContent::Text(text.clone())
                }
                Content::Parts(parts) => {
                    let mut native = Vec::new();
                    for p in parts {
                        match p {
                            Part::Text { text } => {
                                text_bytes = text_bytes.saturating_add(text.len());
                                if text_bytes > 64 << 10 {
                                    return Err(ApiError::bad("image chat text exceeds 64 KiB"));
                                }
                                native.push(qwen2vl::ChatPart::Text { text: text.clone() });
                            }
                            Part::ImageUrl { image_url } => {
                                if image_url.detail.as_deref().is_some_and(|d| d != "auto") {
                                    return Err(ApiError::bad("image detail supports only auto"));
                                }
                                native.push(qwen2vl::ChatPart::Image { index: count });
                                count += 1;
                            }
                        }
                    }
                    qwen2vl::ChatContent::Parts(native)
                }
            };
            skeleton.push(qwen2vl::ChatMessage {
                role: m.role.clone(),
                content,
            });
        }
        qwen2vl::render(&skeleton, count)?;
        Ok(())
    }
    pub async fn acquire(
        self,
        limits: &VisionLimits,
        cancel: &Cancellation,
    ) -> Result<(GenerationRequest, Vec<qwen2vl::ChatMessage>, Vec<Vec<u8>>)> {
        self.acquire_with_processor(limits, cancel, image::Processor::Qwen2VlStill)
            .await
    }
    pub async fn acquire_with_processor(
        self,
        limits: &VisionLimits,
        cancel: &Cancellation,
        processor: image::Processor,
    ) -> Result<(GenerationRequest, Vec<qwen2vl::ChatMessage>, Vec<Vec<u8>>)> {
        let mut messages = Vec::with_capacity(self.messages.len());
        let mut images = Vec::new();
        let mut total = 0u64;
        let mut total_features = 0;
        for message in self.messages {
            let content = match message.content {
                Content::Text(s) => qwen2vl::ChatContent::Text(s),
                Content::Parts(parts) => {
                    let mut out = Vec::with_capacity(parts.len());
                    for part in parts {
                        cancel.check()?;
                        match part {
                            Part::Text { text } => out.push(qwen2vl::ChatPart::Text { text }),
                            Part::ImageUrl { image_url } => {
                                if message.role != "user" {
                                    return Err(ApiError::bad(
                                        "images are permitted only in user messages",
                                    ));
                                }
                                if images.len() >= limits.images.max_images {
                                    return Err(Error::resource(
                                        "image count exceeds operator limit",
                                    )
                                    .into());
                                }
                                if image_url.detail.as_deref().is_some_and(|d| d != "auto") {
                                    return Err(Error::new(ErrorCode::UnsupportedFeature, "image detail supports only auto for the qualified processor").into());
                                }
                                let remaining = limits
                                    .images
                                    .max_total_encoded_bytes
                                    .saturating_sub(total)
                                    .min(limits.images.max_encoded_bytes);
                                let bytes =
                                    acquire_image(&image_url.url, remaining as usize, limits)
                                        .await?;
                                cancel.check()?;
                                let info = image::inspect(&bytes, limits.images, cancel)?;
                                let plan = image::plan(info, processor, limits.images)?;
                                total_features += plan.image_feature_tokens;
                                if total_features > limits.images.max_total_image_tokens {
                                    return Err(Error::new(
                                        ErrorCode::ImageInputLimitExceeded,
                                        "aggregate image features exceed operator limit",
                                    )
                                    .into());
                                }
                                total += bytes.len() as u64;
                                out.push(qwen2vl::ChatPart::Image {
                                    index: images.len(),
                                });
                                images.push(bytes);
                            }
                        }
                    }
                    qwen2vl::ChatContent::Parts(out)
                }
            };
            messages.push(qwen2vl::ChatMessage {
                role: message.role,
                content,
            });
        }
        // Validate roles, aggregate text and reserved delimiters before execution.
        qwen2vl::render(&messages, images.len())?;
        Ok((self.controls, messages, images))
    }
}
fn media_type(mime: &str) -> Result<ImageFormat> {
    match mime {
        "image/png" => Ok(ImageFormat::Png),
        "image/jpeg" => Ok(ImageFormat::Jpeg),
        _ => Err(ApiError::bad(
            "image media type must be image/png or image/jpeg",
        )),
    }
}
async fn acquire_image(url: &str, cap: usize, limits: &VisionLimits) -> Result<Vec<u8>> {
    let (bytes, expected) = if let Some(data) = url.strip_prefix("data:") {
        let (mime, payload) = data
            .split_once(";base64,")
            .ok_or_else(|| ApiError::bad("images require a PNG/JPEG Base64 data URL"))?;
        let format = media_type(mime)?;
        if payload.len() > cap.div_ceil(3).saturating_mul(4) {
            return Err(Error::new(
                ErrorCode::ImageInputLimitExceeded,
                "encoded image exceeds operator byte limit",
            )
            .into());
        }
        let bytes = STANDARD
            .decode(payload)
            .map_err(|_| ApiError::bad("invalid canonical Base64 image"))?;
        if bytes.len() > cap {
            return Err(Error::new(
                ErrorCode::ImageInputLimitExceeded,
                "image exceeds operator byte limit",
            )
            .into());
        }
        (bytes, format)
    } else {
        if !limits.allow_urls {
            return Err(ApiError::bad(
                "remote image URLs are disabled by the operator; use a Base64 data URL",
            ));
        }
        let fetch = async {
            let response = crate::download::response_with_limits(
                url,
                limits.redirects,
                limits.acquisition_timeout_seconds,
            )
            .await?;
            let mime = response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .split(';')
                .next()
                .unwrap_or("");
            let format = media_type(mime)?;
            if response.content_length().is_some_and(|n| n > cap as u64) {
                return Err(Error::new(
                    ErrorCode::ImageInputLimitExceeded,
                    "remote image exceeds byte limit",
                )
                .into());
            }
            let mut stream = response.bytes_stream();
            let mut bytes = Vec::new();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|_| ApiError::bad("image download interrupted"))?;
                if bytes.len().checked_add(chunk.len()).is_none_or(|n| n > cap) {
                    return Err(Error::new(
                        ErrorCode::ImageInputLimitExceeded,
                        "remote image exceeds byte limit",
                    )
                    .into());
                }
                bytes
                    .try_reserve_exact(chunk.len())
                    .map_err(|_| Error::resource("image download allocation failed"))?;
                bytes.extend_from_slice(&chunk);
            }
            Ok::<_, ApiError>((bytes, format))
        };
        tokio::time::timeout(
            Duration::from_secs(limits.acquisition_timeout_seconds),
            fetch,
        )
        .await
        .map_err(|_| ApiError::bad("image acquisition timeout"))??
    };
    let actual = if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some(ImageFormat::Png)
    } else if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        Some(ImageFormat::Jpeg)
    } else {
        None
    };
    if actual != Some(expected) {
        return Err(ApiError::bad(
            "image bytes disagree with the declared media type",
        ));
    }
    Ok(bytes)
}
/// Used by the CLI to attach validated local bytes without exposing local paths.
pub fn data_url(bytes: &[u8], cancel: &Cancellation) -> coheryn_engine::Result<String> {
    let info = image::inspect(bytes, ImageLimits::default(), cancel)?;
    let mime = match info.format {
        ImageFormat::Png => "image/png",
        ImageFormat::Jpeg => "image/jpeg",
    };
    Ok(format!("data:{mime};base64,{}", STANDARD.encode(bytes)))
}

pub fn generation_result(response: qwen2vl::Response) -> coheryn_engine::GenerationResult {
    let images = response
        .plan
        .chunks
        .iter()
        .filter(|c| matches!(c, qwen2vl::PromptChunk::Image { .. }))
        .count();
    coheryn_engine::GenerationResult {
        vision: Some(Box::new(coheryn_engine::inference::VisionMetadata {
            cooperative: None,
            profile: "qwen2vl-still-f16-cpu-greedy".into(),
            language_sha256: response.language_sha256,
            vision_sha256: response.vision_sha256,
            images,
            vision_ms: response.vision_ms,
            effective_positions: response.plan.effective_positions,
            peak_request_reserved_bytes: response.plan.peak_request_bytes,
        })),
        batching: None,
        sampling_metadata: None,
        constraint_validation: None,
        prompt_tokens: response.plan.physical_prompt_tokens,
        reused_prompt_tokens: 0,
        completion_tokens: response.token_ids.len(),
        token_ids: response.token_ids,
        text: response.text,
        tool_calls: vec![],
        finish_reason: response.finish_reason,
        prefill_ms: response.prefill_ms + response.vision_ms,
        tokenization_ms: response.plan.tokenization_ms,
        decode_ms: response.decode_ms,
        output_delivery_ms: 0.,
        first_token_ms: None,
        memory: response.memory,
    }
}

/// Grow admission with received bytes, so chunked text clients keep their small-request budget.
/// A lexical structural bound keeps generic JSON trees within the fixed metadata allowance.
pub(crate) async fn read_body(
    request: axum::extract::Request,
    budget: &coheryn_engine::memory::Budget,
    limits: &VisionLimits,
) -> Result<(Value, usize, coheryn_engine::memory::Reservation)> {
    use axum::http::{StatusCode, header};
    if request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_none_or(|v| v.split(';').next() != Some("application/json"))
    {
        return Err(ApiError::new(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            ErrorCode::InvalidRequest,
            "Content-Type must be application/json",
        ));
    }
    if let Some(v) = request.headers().get(header::CONTENT_LENGTH) {
        let n = v
            .to_str()
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .ok_or_else(|| ApiError::bad("invalid Content-Length"))?;
        if n > limits.body_bytes {
            return Err(ApiError::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                ErrorCode::InvalidRequest,
                "image chat body exceeds operator limit",
            ));
        }
    }
    let mut held = budget.reserve(16 << 20)?;
    let mut stream = request.into_body().into_data_stream();
    let mut bytes = Vec::new();
    let read = async {
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|_| ApiError::bad("image chat body interrupted"))?;
            if bytes
                .len()
                .checked_add(chunk.len())
                .is_none_or(|n| n > limits.body_bytes)
            {
                return Err(ApiError::new(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    ErrorCode::InvalidRequest,
                    "image chat body exceeds operator limit",
                ));
            }
            held.extend(chunk.len() as u64 * 4)?;
            bytes
                .try_reserve_exact(chunk.len())
                .map_err(|_| Error::resource("image body allocation failed"))?;
            bytes.extend_from_slice(&chunk);
        }
        Ok::<_, ApiError>(())
    };
    tokio::time::timeout(Duration::from_secs(30), read)
        .await
        .map_err(|_| ApiError::bad("image chat body timeout"))??;
    let mut quoted = false;
    let mut escaped = false;
    let mut structure = 0;
    for b in &bytes {
        if quoted {
            if escaped {
                escaped = false;
            } else if *b == b'\\' {
                escaped = true;
            } else if *b == b'"' {
                quoted = false;
            }
        } else if *b == b'"' {
            quoted = true;
        } else if b"{}[],:".contains(b) {
            structure += 1;
            if structure > 32768 {
                return Err(ApiError::bad(
                    "chat JSON exceeds structural complexity limit",
                ));
            }
        }
    }
    let value =
        serde_json::from_slice(&bytes).map_err(|_| ApiError::bad("invalid chat request JSON"))?;
    Ok((value, bytes.len(), held))
}

pub(crate) fn validate_controls(r: &GenerationRequest) -> Result<()> {
    if r.max_tokens > 256
        || r.temperature != 0.
        || r.top_p != 1.
        || r.top_k != 0
        || r.repetition_penalty != 1.
        || r.frequency_penalty != 0.
        || r.presence_penalty != 0.
        || r.response_format.enabled()
        || r.tools.is_some()
        || r.tool_choice.is_some()
        || r.parallel_tool_calls.is_some()
        || r.stop
            .as_ref()
            .is_some_and(|s| !matches!(s,crate::wire::Stops::Many(v) if v.is_empty()))
        || !r.stop_tokens.is_empty()
    {
        return Err(Error::new(ErrorCode::UnsupportedFeature, "image chat currently qualifies greedy decoding with 1..=256 output tokens; sampling, tools, constraints and custom stops are unsupported").into());
    }
    Ok(())
}

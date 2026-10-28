//! Offline Qwen2-VL adapter. Product transports and managed sessions remain separate.
use super::*;
use crate::gguf::Value;
use crate::multimodal::{
    image::{self, ImageLimits, ImagePlan, Processor},
    positions::ImagePositions,
    vision::{VisionModel, VisionOptions},
};
use serde::{Deserialize, Serialize};

const REQUEST_BYTES: u64 = 8 << 20;
#[derive(Clone, Debug, Serialize)]
pub struct LanguageConfig {
    pub decoder: ModelConfig,
    pub sections: [usize; 4],
}
pub fn language_configuration(d: &Descriptor) -> Result<LanguageConfig> {
    if d.metadata.len() > 256
        || d.tensors.len() > 1024
        || d.byte_order != "little"
        || d.metadata.keys().any(|k| k.starts_with("split."))
    {
        return Err(Error::invalid(
            "unsupported language component bounds, byte order or shards",
        ));
    }
    let c = compat::configuration_for(d, "qwen2vl")?;
    if c.layers > 64
        || c.hidden > 8192
        || c.intermediate > 32768
        || c.head_dim > 256
        || c.vocabulary > 200000
    {
        return Err(Error::resource(
            "Qwen2-VL decoder exceeds supported dimensions",
        ));
    }
    compat::graph(d, &c)?;
    for t in &d.tensors {
        let dtype = if t.dimensions.len() == 1 { 0 } else { 1 };
        if t.dtype != dtype || quant::row_bytes(t.dtype, t.dimensions[0] as usize)? > IO_BYTES {
            return Err(Error::new(
                ErrorCode::UnsupportedGgufTensorType,
                format!("unqualified Qwen2-VL tensor encoding: {}", t.name),
            ));
        }
    }
    let values = d
        .get("qwen2vl.rope.dimension_sections")
        .and_then(Value::array)
        .filter(|a| a.len() == 4)
        .ok_or_else(|| Error::invalid("missing four rotary sections"))?;
    let mut sections = [0; 4];
    for (dst, value) in sections.iter_mut().zip(values) {
        *dst = value
            .integer()
            .filter(|n| *n <= c.head_dim as u64)
            .ok_or_else(|| Error::invalid("invalid rotary section"))? as usize;
    }
    if sections.iter().sum::<usize>() != c.head_dim / 2 || sections[3] != 0 {
        return Err(Error::new(
            ErrorCode::UnsupportedFeature,
            "unqualified multimodal rotary sections",
        ));
    }
    Ok(LanguageConfig {
        decoder: c,
        sections,
    })
}
#[derive(Clone, Copy)]
pub(super) struct RotaryPosition {
    pub axes: [u32; 4],
    pub sections: [usize; 4],
}
#[inline(never)]
fn paired_trig(angle: f32) -> (f32, f32) {
    angle.sin_cos()
}
pub(super) struct RotaryCache {
    head_dim: usize,
    pairs: [[f32; 2]; 128],
}
impl RotaryCache {
    pub(super) fn new(head_dim: usize, base: f32, position: RotaryPosition) -> Result<Self> {
        if head_dim == 0
            || head_dim > 256
            || !head_dim.is_multiple_of(2)
            || position
                .sections
                .iter()
                .try_fold(0usize, |n, x| n.checked_add(*x))
                != Some(head_dim / 2)
            || !base.is_finite()
            || base <= 0.
            || position.axes.iter().any(|p| *p > i32::MAX as u32)
        {
            return Err(Error::invalid("invalid multidimensional rotary geometry"));
        }
        let mut cache = Self {
            head_dim,
            pairs: [[0.; 2]; 128],
        };
        let step = base.powf(-2. / head_dim as f32);
        let mut angles = position.axes.map(|n| n as f32);
        let mut axis = 0;
        let mut end = position.sections[0];
        // Decoder frequencies progress globally across sections; vision frequencies
        // reset at spatial boundaries. Compute once per row, reuse across all layers.
        for (i, pair) in cache.pairs[..head_dim / 2].iter_mut().enumerate() {
            while i >= end {
                axis += 1;
                end += position.sections[axis];
            }
            let (sin, cos) = paired_trig(angles[axis]);
            *pair = [cos, sin];
            for theta in &mut angles {
                *theta *= step;
            }
        }
        Ok(cache)
    }
    pub(super) fn apply(&self, values: &mut [f32]) -> Result<()> {
        if !values.len().is_multiple_of(self.head_dim) {
            return Err(Error::invalid("rotary vector shape mismatch"));
        }
        for head in values.chunks_exact_mut(self.head_dim) {
            for (i, [cos, sin]) in self.pairs[..self.head_dim / 2].iter().enumerate() {
                let a = head[i];
                let b = head[i + self.head_dim / 2];
                head[i] = a * cos - b * sin;
                head[i + self.head_dim / 2] = a * sin + b * cos;
            }
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(Error::invalid("nonfinite spatial rotary output"));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatMessage {
    pub role: String,
    pub content: ChatContent,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatContent {
    Text(String),
    Parts(Vec<ChatPart>),
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ChatPart {
    Text { text: String },
    Image { index: usize },
}
/// Implements the pinned template's text/still-image subset, with no executable
/// template program, URLs, tools, videos or implicit attachment ordering.
pub fn render(messages: &[ChatMessage], images: usize) -> Result<String> {
    if messages.is_empty() || messages.len() > 64 || images > 4 {
        return Err(Error::request(
            "chat requires 1..=64 messages and at most four images",
        ));
    }
    let mut out = String::new();
    let mut seen = vec![false; images];
    let mut bytes = 0usize;
    let mut attachments = 0;
    let append_text = |out: &mut String, text: &str, bytes: &mut usize| -> Result<()> {
        *bytes = bytes
            .checked_add(text.len())
            .filter(|n| *n <= 65536)
            .ok_or_else(|| Error::resource("chat text exceeds 64 KiB"))?;
        if [
            "<|vision_",
            "<|image_pad|>",
            "<|video_pad|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
        ]
        .iter()
        .any(|m| text.contains(m))
        {
            return Err(Error::request(
                "reserved chat/media delimiters are not accepted inside text",
            ));
        }
        out.push_str(text);
        Ok(())
    };
    for (i, message) in messages.iter().enumerate() {
        if !matches!(message.role.as_str(), "system" | "user" | "assistant")
            || (message.role == "system" && i != 0)
        {
            return Err(Error::request(
                "unsupported chat role or system-message placement",
            ));
        }
        if i == 0 && message.role != "system" {
            out.push_str("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n");
        }
        out.push_str("<|im_start|>");
        out.push_str(&message.role);
        out.push('\n');
        match &message.content {
            ChatContent::Text(text) => append_text(&mut out, text, &mut bytes)?,
            ChatContent::Parts(parts) => {
                if parts.is_empty() || parts.len() > 128 {
                    return Err(Error::request("chat parts require 1..=128 entries"));
                }
                for part in parts {
                    match part {
                        ChatPart::Text { text } => append_text(&mut out, text, &mut bytes)?,
                        ChatPart::Image { index } => {
                            if message.role != "user" || *index >= images || seen[*index] {
                                return Err(Error::request(
                                    "images require unique valid indices in user messages",
                                ));
                            }
                            seen[*index] = true;
                            attachments += 1;
                            out.push_str("<|vision_start|><|image_pad|><|vision_end|>");
                        }
                    }
                }
            }
        }
        out.push_str("<|im_end|>\n");
    }
    if attachments != images || messages.last().unwrap().role != "user" {
        return Err(Error::request(
            "every attachment must be used and the final message must be a user turn",
        ));
    }
    out.push_str("<|im_start|>assistant\n");
    Ok(out)
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    pub context: Option<usize>,
    pub max_tokens: usize,
    pub threads: usize,
    pub vision: VisionOptions,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            context: None,
            max_tokens: 128,
            threads: 1,
            vision: VisionOptions::default(),
        }
    }
}
#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PromptChunk {
    Text {
        token_ids: Vec<u32>,
        start_position: u32,
    },
    Image {
        index: usize,
        positions: ImagePositions,
    },
}
#[derive(Debug, Serialize)]
pub struct RequestPlan {
    pub profile: &'static str,
    pub rendered_prompt: String,
    pub chunks: Vec<PromptChunk>,
    pub physical_prompt_tokens: usize,
    pub effective_positions: u32,
    pub context: usize,
    pub image_stage_bytes: u64,
    pub decoder_stage_bytes: u64,
    pub peak_request_bytes: u64,
    pub tokenization_ms: f64,
}
pub struct Checkpoint<'a> {
    pub name: &'a str,
    pub values: &'a [f32],
}
pub type Observer<'a> = dyn FnMut(Checkpoint<'_>) -> Result<()> + 'a;
#[derive(Serialize)]
pub struct Response {
    pub language_sha256: String,
    pub vision_sha256: String,
    pub plan: RequestPlan,
    pub token_ids: Vec<u32>,
    pub text: String,
    pub finish_reason: FinishReason,
    pub vision_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub memory: MemoryStats,
    #[serde(skip)]
    _reservation: Reservation,
}
impl std::fmt::Debug for Response {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VlmResponse")
            .field("token_ids", &self.token_ids)
            .field("finish_reason", &self.finish_reason)
            .finish_non_exhaustive()
    }
}
/// An owned request whose complete execution capacity is held while queued.
/// Its private admission can only be consumed by the model instance that created it.
pub struct PreparedRequest {
    images: Vec<Vec<u8>>,
    admitted: AdmittedRequest,
}
struct AdmittedRequest {
    owner: Arc<()>,
    plan: RequestPlan,
    image_plans: Vec<ImagePlan>,
    options: Options,
    output: Reservation,
    execution: Reservation,
}
impl PreparedRequest {
    pub fn plan(&self) -> &RequestPlan {
        &self.admitted.plan
    }
}
pub struct Qwen2Vl {
    language: Model,
    vision: VisionModel,
    configuration: LanguageConfig,
    language_sha256: String,
    budget: Budget,
}
/// Header-only compatibility preflight; construction/deep validation checks the complete BPE graph.
pub fn validate_language_metadata(descriptor: &Descriptor) -> Result<()> {
    crate::tokenizer::check_qwen_vl_metadata(descriptor)?;
    let tokens = descriptor
        .get("tokenizer.ggml.tokens")
        .and_then(crate::gguf::Value::array)
        .ok_or_else(|| Error::invalid("missing image vocabulary"))?;
    let types = descriptor
        .get("tokenizer.ggml.token_type")
        .and_then(crate::gguf::Value::array)
        .ok_or_else(|| Error::invalid("missing image token types"))?;
    if tokens.len() != types.len() {
        return Err(Error::invalid("image token type count mismatch"));
    }
    for marker in ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>"] {
        let matches = tokens
            .iter()
            .zip(types)
            .filter(|(t, _)| t.string() == Some(marker))
            .collect::<Vec<_>>();
        if matches.len() != 1 || !matches[0].1.integer().is_some_and(|ty| matches!(ty, 3 | 4)) {
            return Err(Error::invalid("missing unique special vision marker"));
        }
    }
    Ok(())
}
/// Validate the exact qualified tokenizer/template and atomic image markers without loading weights.
pub fn validate_language_tokenizer(descriptor: &Descriptor) -> Result<()> {
    let tokenizer = TextTokenizer::from_qwen_vl_descriptor(descriptor)?;
    for marker in ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>"] {
        let ids = tokenizer.encode(marker)?;
        if ids.len() != 1 || !tokenizer.is_special(ids[0]) {
            return Err(Error::invalid("missing atomic vision marker token"));
        }
    }
    Ok(())
}
fn language_limits() -> ParseLimits {
    ParseLimits {
        max_metadata_entries: 256,
        max_tensors: 1024,
        max_array_items: 200000,
        max_string_bytes: 16384,
        max_header_bytes: 16 << 20,
        max_metadata_allocation: 64 << 20,
        max_depth: 2,
    }
}
impl Qwen2Vl {
    pub fn load(
        language: impl AsRef<Path>,
        vision: impl AsRef<Path>,
        limits: ResourceLimits,
        budget: Budget,
        cancel: &Cancellation,
    ) -> Result<Self> {
        cancel.check()?;
        if !(1..=4096).contains(&limits.context) {
            return Err(Error::request("offline VLM context must be in 1..=4096"));
        }
        let stat = std::fs::metadata(language.as_ref())?;
        if !stat.is_file() || stat.len() > 8 << 30 {
            return Err(Error::invalid(
                "language component must be a regular file within 8 GiB",
            ));
        }
        let loading = budget.reserve(256 << 20)?;
        let descriptor = gguf::inspect(language.as_ref(), language_limits())?;
        let configuration = language_configuration(&descriptor)?;
        let config = configuration.decoder.clone();
        if limits.context > config.context {
            return Err(Error::request(
                "requested context exceeds language capacity",
            ));
        }
        let model_reservation = budget.reserve(memory::model_bytes(&descriptor, &config)?)?;
        let tokenizer = TextTokenizer::from_qwen_vl_descriptor(&descriptor)?;
        for marker in ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>"] {
            let tokens = tokenizer.encode(marker)?;
            if tokens.len() != 1 || !tokenizer.is_special(tokens[0]) {
                return Err(Error::invalid("missing atomic vision marker token"));
            }
        }
        let storage = Storage::open_with_limits(language.as_ref(), &descriptor, language_limits())?;
        let mut hash = Sha256::new();
        let mut buffer = vec![0u8; IO_BYTES];
        let mut offset = 0;
        while offset < descriptor.file_bytes {
            cancel.check()?;
            let n = (descriptor.file_bytes - offset).min(IO_BYTES as u64) as usize;
            storage.read(offset, &mut buffer[..n])?;
            hash.update(&buffer[..n]);
            offset += n as u64;
        }
        storage.unchanged()?;
        cancel.check()?;
        let language_sha256 = crate::storage::digest_hex(hash.finalize().into());
        let tensors = descriptor
            .tensors
            .iter()
            .map(|t| (t.name.clone(), t.clone()))
            .collect();
        let language = Model {
            cache_owner: Arc::new(()),
            descriptor,
            config,
            tokenizer,
            storage,
            tensors,
            budget: budget.clone(),
            limits,
            _model_reservation: model_reservation,
        };
        drop(buffer);
        drop(loading);
        let vision = VisionModel::load(vision, budget.clone(), cancel)?;
        if vision.config().output_width != language.config.hidden {
            return Err(Error::invalid(
                "vision merger width differs from the language input width",
            ));
        }
        Ok(Self {
            language,
            vision,
            configuration,
            language_sha256,
            budget,
        })
    }
    pub fn context_limit(&self) -> usize {
        self.language.limits.context
    }
    pub fn descriptor(&self) -> &Descriptor {
        &self.language.descriptor
    }
    pub fn resident_bytes(&self) -> u64 {
        self.language._model_reservation.bytes() + self.vision.resident_bytes()
    }
    pub fn configuration(&self) -> &LanguageConfig {
        &self.configuration
    }
    pub fn model_identities(&self) -> (&str, &str) {
        (&self.language_sha256, self.vision.model_sha256())
    }
    fn plan(
        &self,
        messages: &[ChatMessage],
        images: &[&[u8]],
        options: &Options,
        cancel: &Cancellation,
    ) -> Result<(RequestPlan, Vec<ImagePlan>)> {
        if !(1..=256).contains(&options.max_tokens) {
            return Err(Error::request("offline generation requires 1..=256 tokens"));
        }
        memory::worker_bytes(options.threads)?;
        options.vision.validate()?;
        let context = options.context.unwrap_or(self.context_limit());
        if context == 0 || context > self.context_limit() {
            return Err(Error::request("context exceeds loaded VLM capacity"));
        }
        let limits = ImageLimits::default();
        let rendered_prompt = render(messages, images.len())?;
        let mut plans = Vec::with_capacity(images.len());
        let mut total_encoded = 0u64;
        let mut features = 0;
        let mut stage = 0u64;
        for bytes in images {
            cancel.check()?;
            total_encoded = total_encoded
                .checked_add(bytes.len() as u64)
                .filter(|n| *n <= limits.max_total_encoded_bytes)
                .ok_or_else(|| Error::resource("aggregate image bytes exceeded"))?;
            let p = image::plan(
                image::inspect(bytes, limits, cancel)?,
                Processor::Qwen2VlStill,
                limits,
            )?;
            features += p.image_feature_tokens;
            if features > limits.max_total_image_tokens {
                return Err(Error::resource("aggregate image features exceeded"));
            }
            let v = self.vision.plan_image(&p, &options.vision)?;
            stage = stage.max(p.peak_reserved_bytes).max(
                p.output_reserved_bytes
                    .checked_add(v.peak_reserved_bytes)
                    .ok_or_else(|| Error::resource("vision stage overflow"))?,
            );
            plans.push(p);
        }
        // Caller-owned encoded bytes remain charged throughout sequential reuse.
        stage = stage
            .checked_add(total_encoded)
            .ok_or_else(|| Error::resource("image reservation overflow"))?;
        let mut tokenization_ms = 0.;
        let mut chunks = Vec::new();
        let mut physical = 0usize;
        let mut next = 0u32;
        let order = messages
            .iter()
            .flat_map(|m| match &m.content {
                ChatContent::Text(_) => vec![],
                ChatContent::Parts(p) => p
                    .iter()
                    .filter_map(|p| {
                        if let ChatPart::Image { index } = p {
                            Some(*index)
                        } else {
                            None
                        }
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        for (i, text) in rendered_prompt.split("<|image_pad|>").enumerate() {
            cancel.check()?;
            let tokenization_start = Instant::now();
            let ids = self.language.tokenizer.encode(text)?;
            tokenization_ms += tokenization_start.elapsed().as_secs_f64() * 1000.;
            physical = physical
                .checked_add(ids.len())
                .ok_or_else(|| Error::resource("physical token overflow"))?;
            let start = next;
            next = next
                .checked_add(ids.len() as u32)
                .ok_or_else(|| Error::resource("rotary position overflow"))?;
            chunks.push(PromptChunk::Text {
                token_ids: ids,
                start_position: start,
            });
            if i < order.len() {
                let index = order[i];
                let span = ImagePositions::new(&plans[index], next)?;
                physical += span.physical_tokens();
                next = span.next_position();
                chunks.push(PromptChunk::Image {
                    index,
                    positions: span,
                });
            }
        }
        if physical
            .checked_add(options.max_tokens)
            .is_none_or(|n| n > context)
            || next as usize + options.max_tokens > self.language.config.context
        {
            return Err(Error::new(
                ErrorCode::ContextLimitExceeded,
                "image-expanded prompt plus output exceeds physical or rotary context capacity",
            ));
        }
        let decoder =
            memory::session_bytes_for(&self.language.config, context, IO_BYTES, options.threads)?;
        let peak = decoder
            .checked_add(stage)
            .and_then(|n| n.checked_add(REQUEST_BYTES))
            .ok_or_else(|| Error::resource("VLM request reservation overflow"))?;
        Ok((
            RequestPlan {
                profile: "coheryn-qwen2vl-offline-v1",
                rendered_prompt,
                chunks,
                physical_prompt_tokens: physical,
                effective_positions: next,
                context,
                image_stage_bytes: stage,
                decoder_stage_bytes: decoder,
                peak_request_bytes: peak,
                tokenization_ms,
            },
            plans,
        ))
    }
    pub fn generate(
        &self,
        messages: &[ChatMessage],
        images: &[&[u8]],
        options: &Options,
        cancel: &Cancellation,
        emit: impl FnMut(TokenEvent) -> Result<()>,
    ) -> Result<Response> {
        self.run(messages, images, options, cancel, None, emit)
    }
    pub fn trace(
        &self,
        messages: &[ChatMessage],
        images: &[&[u8]],
        options: &Options,
        cancel: &Cancellation,
        observer: &mut Observer<'_>,
        emit: impl FnMut(TokenEvent) -> Result<()>,
    ) -> Result<Response> {
        self.run(messages, images, options, cancel, Some(observer), emit)
    }
    fn run(
        &self,
        messages: &[ChatMessage],
        images: &[&[u8]],
        options: &Options,
        cancel: &Cancellation,
        observer: Option<&mut Observer<'_>>,
        emit: impl FnMut(TokenEvent) -> Result<()>,
    ) -> Result<Response> {
        let admitted = self.admit(messages, images, options, cancel)?;
        self.execute(admitted, images, cancel, observer, None, emit)
    }
    fn admit(
        &self,
        messages: &[ChatMessage],
        images: &[&[u8]],
        options: &Options,
        cancel: &Cancellation,
    ) -> Result<AdmittedRequest> {
        cancel.check()?;
        let output = self.budget.reserve(REQUEST_BYTES)?;
        let (plan, image_plans) = self.plan(messages, images, options, cancel)?;
        let execution = self
            .budget
            .reserve(plan.peak_request_bytes - REQUEST_BYTES)?;
        Ok(AdmittedRequest {
            owner: self.language.cache_owner.clone(),
            plan,
            image_plans,
            options: options.clone(),
            output,
            execution,
        })
    }
    pub fn prepare_owned(
        &self,
        messages: &[ChatMessage],
        images: Vec<Vec<u8>>,
        options: &Options,
        cancel: &Cancellation,
    ) -> Result<PreparedRequest> {
        let refs: Vec<_> = images.iter().map(Vec::as_slice).collect();
        let admitted = self.admit(messages, &refs, options, cancel)?;
        Ok(PreparedRequest { images, admitted })
    }
    pub fn generate_prepared(
        &self,
        request: PreparedRequest,
        cancel: &Cancellation,
        emit: impl FnMut(TokenEvent) -> Result<()>,
    ) -> Result<Response> {
        let PreparedRequest { images, admitted } = request;
        let refs: Vec<_> = images.iter().map(Vec::as_slice).collect();
        self.execute(admitted, &refs, cancel, None, None, emit)
    }
    /// Scheduler checkpoints never release the request's buffers or admission.
    pub fn generate_cooperative(
        &self,
        request: PreparedRequest,
        cancel: &Cancellation,
        cooperate: &mut crate::multimodal::WorkYield<'_>,
        emit: impl FnMut(TokenEvent) -> Result<()>,
    ) -> Result<Response> {
        let PreparedRequest { images, admitted } = request;
        let refs: Vec<_> = images.iter().map(Vec::as_slice).collect();
        self.execute(admitted, &refs, cancel, None, Some(cooperate), emit)
    }
    pub fn trace_prepared(
        &self,
        request: PreparedRequest,
        cancel: &Cancellation,
        observer: &mut Observer<'_>,
        cooperate: Option<&mut crate::multimodal::WorkYield<'_>>,
        emit: impl FnMut(TokenEvent) -> Result<()>,
    ) -> Result<Response> {
        let PreparedRequest { images, admitted } = request;
        let refs: Vec<_> = images.iter().map(Vec::as_slice).collect();
        self.execute(admitted, &refs, cancel, Some(observer), cooperate, emit)
    }
    fn execute(
        &self,
        admitted: AdmittedRequest,
        images: &[&[u8]],
        cancel: &Cancellation,
        mut observer: Option<&mut Observer<'_>>,
        mut cooperate: Option<&mut crate::multimodal::WorkYield<'_>>,
        mut emit: impl FnMut(TokenEvent) -> Result<()>,
    ) -> Result<Response> {
        cancel.check()?;
        if !Arc::ptr_eq(&admitted.owner, &self.language.cache_owner) {
            return Err(Error::request(
                "prepared image request belongs to another model instance",
            ));
        }
        let AdmittedRequest {
            plan,
            image_plans,
            options,
            output: output_reservation,
            execution: mut held,
            ..
        } = admitted;
        let session_reservation = held.split(plan.decoder_stage_bytes)?;
        let execution = ExecutionSettings {
            threads: options.threads,
            prefill_chunk_tokens: 1,
        };
        let mut session = Session::admitted(
            &self.language.config,
            plan.context,
            &self.budget,
            session_reservation,
            IO_BYTES,
            execution,
        )?;
        let start = Instant::now();
        let mut vision_ms = 0.;
        for (chunk_index, chunk) in plan.chunks.iter().enumerate() {
            match chunk {
                PromptChunk::Text {
                    token_ids,
                    start_position,
                } => {
                    for (i, token) in token_ids.iter().enumerate() {
                        let last = chunk_index + 1 == plan.chunks.len() && i + 1 == token_ids.len();
                        let cooperates = cooperate.is_some();
                        let mut checkpoint = |name: &str, values: &[f32]| {
                            if last
                                && name != "logits"
                                && let Some(cb) = observer.as_deref_mut()
                            {
                                cb(Checkpoint {
                                    name: &format!("prompt_{name}"),
                                    values,
                                })?;
                            }
                            yield_work(&mut cooperate)
                        };
                        self.language.forward_row(
                            ForwardRow {
                                input: RowInput::Token(*token),
                                rotary: Some(RotaryPosition {
                                    axes: [*start_position + i as u32; 4],
                                    sections: self.configuration.sections,
                                }),
                                logits: last,
                            },
                            &mut session,
                            cancel,
                            if last || cooperates {
                                Some(&mut checkpoint)
                            } else {
                                None
                            },
                        )?;
                        yield_work(&mut cooperate)?;
                    }
                }
                PromptChunk::Image { index, positions } => {
                    yield_work(&mut cooperate)?;
                    let prepared = image::prepare_admitted(
                        images[*index],
                        image_plans[*index].clone(),
                        &self.budget,
                        &mut held,
                        cancel,
                    )?;
                    yield_work(&mut cooperate)?;
                    let encoded = if let Some(yield_cpu) = cooperate.as_deref_mut() {
                        self.vision.encode_cooperative(
                            &prepared,
                            &options.vision,
                            cancel,
                            &mut held,
                            yield_cpu,
                        )?
                    } else {
                        self.vision.encode_reserved(
                            &prepared,
                            &options.vision,
                            cancel,
                            &mut held,
                        )?
                    };
                    vision_ms += encoded.encoding_ms;
                    if let Some(cb) = observer.as_deref_mut() {
                        cb(Checkpoint {
                            name: &format!("image_{index}_embeddings"),
                            values: encoded.embeddings(),
                        })?;
                    }
                    prepared.recycle(&mut held)?;
                    for (i, row) in encoded
                        .embeddings()
                        .chunks_exact(self.language.config.hidden)
                        .enumerate()
                    {
                        let cooperates = cooperate.is_some();
                        let mut checkpoint = |_: &str, _: &[f32]| yield_work(&mut cooperate);
                        self.language.forward_row(
                            ForwardRow {
                                input: RowInput::Embedding(row),
                                rotary: Some(RotaryPosition {
                                    axes: positions.at(i)?,
                                    sections: self.configuration.sections,
                                }),
                                logits: false,
                            },
                            &mut session,
                            cancel,
                            if cooperates {
                                Some(&mut checkpoint)
                            } else {
                                None
                            },
                        )?;
                        yield_work(&mut cooperate)?;
                    }
                    encoded.recycle(&mut held)?;
                }
            }
        }
        let prefill_ms = start.elapsed().as_secs_f64() * 1000. - vision_ms;
        let start = Instant::now();
        let mut token_ids = Vec::new();
        let mut text = String::new();
        let mut utf8 = Utf8Stream::default();
        let mut finish_reason = FinishReason::Length;
        let mut next = plan.effective_positions;
        for step in 0..options.max_tokens {
            cancel.check()?;
            if let Some(cb) = observer.as_deref_mut() {
                cb(Checkpoint {
                    name: &format!("logits_{step}"),
                    values: &session.scratch.logits,
                })?;
            }
            cancel.check()?;
            let mut best = None;
            for (i, logit) in session.scratch.logits.iter().enumerate() {
                if !logit.is_finite() {
                    return Err(Error::invalid("nonfinite language logits"));
                }
                if best.is_none_or(|(_, v)| *logit > v) {
                    best = Some((i, *logit));
                }
            }
            let id = best
                .ok_or_else(|| Error::invalid("empty language logits"))?
                .0 as u32;
            token_ids.push(id);
            if self.language.tokenizer.is_eog(id) {
                finish_reason = FinishReason::Stop;
                break;
            }
            let bytes = if self.language.tokenizer.is_special(id) {
                &[][..]
            } else {
                self.language.tokenizer.token_bytes(id)?
            };
            let piece = utf8.push(bytes, false);
            if text.len() + piece.len() > 1 << 20 {
                return Err(Error::resource("generated text exceeds one MiB"));
            }
            text.push_str(&piece);
            cancel.check()?;
            emit(TokenEvent {
                token_id: Some(id),
                text: piece,
            })?;
            if step + 1 < options.max_tokens {
                let cooperates = cooperate.is_some();
                let mut checkpoint = |_: &str, _: &[f32]| yield_work(&mut cooperate);
                self.language.forward_row(
                    ForwardRow {
                        input: RowInput::Token(id),
                        rotary: Some(RotaryPosition {
                            axes: [next; 4],
                            sections: self.configuration.sections,
                        }),
                        logits: true,
                    },
                    &mut session,
                    cancel,
                    if cooperates {
                        Some(&mut checkpoint)
                    } else {
                        None
                    },
                )?;
                yield_work(&mut cooperate)?;
                next += 1;
            }
        }
        let tail = utf8.push(&[], true);
        if text
            .len()
            .checked_add(tail.len())
            .is_none_or(|n| n > 1 << 20)
        {
            return Err(Error::resource("generated text exceeds one MiB"));
        }
        if !tail.is_empty() {
            text.push_str(&tail);
            emit(TokenEvent {
                token_id: None,
                text: tail,
            })?;
        }
        self.language.storage.unchanged()?;
        cancel.check()?;
        let decode_ms = start.elapsed().as_secs_f64() * 1000.;
        drop(session);
        drop(held);
        Ok(Response {
            language_sha256: self.language_sha256.clone(),
            vision_sha256: self.vision.model_sha256().into(),
            plan,
            token_ids,
            text,
            finish_reason,
            vision_ms,
            prefill_ms,
            decode_ms,
            memory: self.budget.stats(),
            _reservation: output_reservation,
        })
    }
}

fn yield_work(cooperate: &mut Option<&mut crate::multimodal::WorkYield<'_>>) -> Result<()> {
    if let Some(f) = cooperate.as_deref_mut() {
        f()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn decoder_rotary_sections_use_global_frequencies_and_distinct_axes() {
        let p = RotaryPosition {
            axes: [1, 2, 3, 0],
            sections: [1, 1, 2, 0],
        };
        let cache = RotaryCache::new(8, 10000., p).unwrap();
        // Independent analytic frequencies: [1, .1, .01, .001] with axis
        // positions [1,2,3,3], hence angles [1,.2,.03,.003].
        let mut values = [1., 2., 3., 4., 5., 6., 7., 8.];
        cache.apply(&mut values).unwrap();
        for (i, angle) in [1f64, 0.2, 0.03, 0.003].into_iter().enumerate() {
            let a = (i + 1) as f64;
            let b = (i + 5) as f64;
            assert!((values[i] as f64 - (a * angle.cos() - b * angle.sin())).abs() < 1e-6);
            assert!((values[i + 4] as f64 - (a * angle.sin() + b * angle.cos())).abs() < 1e-6);
        }
        let mut repeated = [1.; 16];
        cache.apply(&mut repeated).unwrap();
        assert_eq!(repeated[..8], repeated[8..]);
        assert!(cache.apply(&mut [0.; 7]).is_err());
        for sections in [[0; 4], [usize::MAX, 1, 1, 1], [1, 1, 1, 0]] {
            assert!(RotaryCache::new(8, 10000., RotaryPosition { sections, ..p }).is_err());
        }
        assert!(RotaryCache::new(512, 10000., p).is_err());
        assert!(RotaryCache::new(8, f32::NAN, p).is_err());
    }
}

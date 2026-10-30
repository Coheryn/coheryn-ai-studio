//! Owned, admitted still-image input. Image-conditioned prefixes are never cached.
use super::*;
use crate::adapters::Execution;
use crate::inference::VisionMetadata;
use crate::multimodal::{
    image::{self, ImageLimits, ImagePlan, Processor},
    qwen2vl::{ChatContent, ChatMessage, ChatPart},
};

pub struct PreparedImages {
    prepared: PreparedGeneration,
    prompt: String,
    tokens: Vec<u32>,
    positions: Vec<[usize; 3]>,
    spans: Vec<(usize, usize)>,
    images: Vec<Vec<u8>>,
    plans: Vec<ImagePlan>,
    held: Reservation,
    next_position: usize,
    peak: u64,
}
pub(super) struct Conditioning {
    pub features: Vec<vision::VisionResult>,
    pub positions: Vec<[usize; 3]>,
    pub spans: Vec<(usize, usize)>,
    pub next_position: usize,
    pub metadata: VisionMetadata,
}
impl Conditioning {
    pub fn embedding(&self, position: usize, width: usize) -> Option<&[f32]> {
        self.spans.iter().enumerate().find_map(|(i, (begin, end))| {
            if position >= *begin && position < *end {
                let row = position - begin;
                Some(&self.features[i].values[row * width..(row + 1) * width])
            } else {
                None
            }
        })
    }
}
impl PreparedImages {
    pub fn prompt_token_count(&self) -> usize {
        self.tokens.len()
    }
    pub fn request_reserved_bytes(&self) -> u64 {
        self.prepared.request_reserved_bytes() + self.held.bytes()
    }
}
impl Model {
    pub fn vision_configuration(&self) -> Result<&vision::VisionConfig> {
        self.vision.as_ref().ok_or_else(|| {
            Error::new(
                ErrorCode::UnsupportedImageProcessor,
                "checkpoint has no qualified native still-image processor/component",
            )
        })
    }
    pub fn render_images(&self, messages: &[ChatMessage], images: usize) -> Result<String> {
        // Share role, delimiter, ordering and finite-message validation with existing image requests.
        crate::multimodal::qwen2vl::render(messages, images)?;
        let mut values = Vec::new();
        for message in messages {
            let content = match &message.content {
                ChatContent::Text(text) => serde_json::json!(text),
                ChatContent::Parts(parts) => serde_json::Value::Array(
                    parts
                        .iter()
                        .map(|part| match part {
                            ChatPart::Text { text } => {
                                serde_json::json!({"type":"text","text":text})
                            }
                            ChatPart::Image { .. } => serde_json::json!({"type":"image"}),
                        })
                        .collect(),
                ),
            };
            values.push(serde_json::json!({"role":message.role,"content":content,"reasoning_content":"","tool_calls":[]}));
        }
        for message in &values {
            let text = message["content"].to_string();
            if ["<|", "<tool_", "<think>", "</think>"]
                .iter()
                .any(|marker| text.contains(marker))
            {
                return Err(Error::request(
                    "image chat text contains reserved control delimiters",
                ));
            }
        }
        self.render_values(values)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_images(
        &self,
        messages: &[ChatMessage],
        images: Vec<Vec<u8>>,
        limits: ImageLimits,
        options: GenerationOptions,
        context: usize,
        execution: Execution<'_>,
        concurrency: usize,
        cancel: &Cancellation,
    ) -> Result<PreparedImages> {
        cancel.check()?;
        limits.validate()?;
        let config = self.vision_configuration()?;
        if images.len() > limits.max_images {
            return Err(Error::new(
                ErrorCode::ImageInputLimitExceeded,
                "image count exceeds finite limit",
            ));
        }
        let prompt = self.render_images(messages, images.len())?;
        let original = self.tokenize(&prompt)?;
        let mut plans = Vec::new();
        let mut total = 0u64;
        let mut features = 0usize;
        for image in &images {
            total = total
                .checked_add(image.len() as u64)
                .ok_or_else(|| Error::resource("image bytes overflow"))?;
            let p = image::plan(
                image::inspect(image, limits, cancel)?,
                Processor::Qwen35Still,
                limits,
            )?;
            features += p.image_feature_tokens;
            plans.push(p);
        }
        if total > limits.max_total_encoded_bytes || features > limits.max_total_image_tokens {
            return Err(Error::new(
                ErrorCode::ImageInputLimitExceeded,
                "aggregate image limit exceeded",
            ));
        }
        let expanded = original
            .len()
            .checked_add(features)
            .and_then(|n| n.checked_sub(images.len()))
            .ok_or_else(|| Error::resource("image token overflow"))?;
        if expanded
            .checked_add(options.max_tokens)
            .is_none_or(|n| n > context)
        {
            return Err(Error::new(
                ErrorCode::ContextLimitExceeded,
                "expanded image prompt plus output exceeds requested context",
            ));
        }
        let prepared = self.prepare_generation(options, context, execution, concurrency)?;
        let extra =
            super::image_plan::image_stages(config, &plans, prepared.admission.threads, limits)?
                .extra_reserved_bytes;
        let held = self.budget.reserve(extra)?;
        let mut tokens = Vec::with_capacity(expanded);
        let mut positions = Vec::with_capacity(expanded);
        let mut spans = Vec::new();
        let mut offset = 0;
        let mut cursor = 0;
        for token in original {
            if token == 248056 {
                let p = plans
                    .get(cursor)
                    .ok_or_else(|| Error::request("unbound image marker"))?;
                let start = tokens.len();
                let w = p.grid[2] / 2;
                for j in 0..p.image_feature_tokens {
                    tokens.push(token);
                    positions.push([offset, offset + j / w, offset + j % w]);
                }
                spans.push((start, tokens.len()));
                offset += (p.grid[1] / 2).max(w);
                cursor += 1;
            } else {
                tokens.push(token);
                positions.push([offset; 3]);
                offset += 1;
            }
        }
        if cursor != images.len() {
            return Err(Error::request("image marker count mismatch"));
        }
        let peak = extra + prepared.request_reserved_bytes();
        Ok(PreparedImages {
            prepared,
            prompt,
            tokens,
            positions,
            spans,
            images,
            plans,
            held,
            next_position: offset,
            peak,
        })
    }
    pub fn begin_images(
        &self,
        prepared: PreparedImages,
        cancel: &Cancellation,
        mut cooperate: Option<&mut crate::multimodal::WorkYield<'_>>,
        mut observer: Option<&mut vision::Observer<'_>>,
    ) -> Result<GenerationState> {
        let PreparedImages {
            prepared,
            prompt,
            tokens,
            positions,
            spans,
            images,
            plans,
            mut held,
            next_position,
            peak,
        } = prepared;
        let timed_cancel = cancel.with_timeout(prepared.options.timeout_seconds)?;
        let cancel = &timed_cancel;
        if !std::sync::Arc::ptr_eq(&self.owner, &prepared.owner) {
            return Err(Error::request("image admission belongs to another model"));
        }
        let config = self.vision_configuration()?;
        let threads = prepared.admission.threads;
        let mut features = Vec::new();
        let mut elapsed = 0.;
        for (bytes, p) in images.iter().zip(plans) {
            cancel.check()?;
            let image = image::prepare_admitted(bytes, p, &self.budget, &mut held, cancel)?;
            let stage = held.split(config.estimate(image.plan().patch_rows, threads)?)?;
            let result = vision::encode(
                self,
                &image,
                config,
                threads,
                cancel,
                cooperate.as_deref_mut(),
                Some(stage),
                observer.as_deref_mut(),
            )?;
            elapsed += result.encoding_ms;
            features.push(result);
        }
        let mut state = self.begin_generation(&prompt, prepared, cancel)?;
        let metadata = VisionMetadata {
            cooperative: None,
            profile: vision::PROFILE.into(),
            language_sha256: String::new(),
            vision_sha256: String::new(),
            images: features.len(),
            vision_ms: elapsed,
            effective_positions: next_position as u32,
            peak_request_reserved_bytes: peak,
        };
        state.set_images(
            tokens,
            Conditioning {
                features,
                positions,
                spans,
                next_position,
                metadata,
            },
        )?;
        Ok(state)
    }
    pub fn trace_images(
        &self,
        state: &GenerationState,
        teacher: &[u32],
        cancel: &Cancellation,
        callback: &mut TraceObserver<'_>,
    ) -> Result<()> {
        if teacher.len() > 256
            || state.tokens.len() + teacher.len() > self.context_limit()
            || teacher
                .iter()
                .any(|t| *t as usize >= self.config.vocabulary)
        {
            return Err(Error::request("invalid image trace tokens"));
        }
        let _held = self.budget.reserve(Session::bytes(
            &self.config,
            self.context_limit(),
            state.threads,
        )?)?;
        let mut session = Session::new(&self.config, self.context_limit(), state.threads)?;
        let mut accounting = self.budget.track_buffers(_held.bytes());
        accounting.grow(session.capacity_bytes())?;
        for (position, token) in state.tokens.iter().chain(teacher).enumerate() {
            let condition = state
                .condition
                .as_ref()
                .ok_or_else(|| Error::request("missing image conditioning"))?;
            session.rotary = Some(condition.positions.get(position).copied().unwrap_or(
                [condition.next_position + position.saturating_sub(state.tokens.len()); 3],
            ));
            self.graph.forward_input(
                &self.set,
                &self.config,
                *token,
                condition.embedding(position, self.config.hidden),
                &mut session,
                true,
                cancel,
                Some(&mut |name, data| callback(position, name, data)),
                None,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires separately acquired pinned Qwen3.5 checkpoint"]
    fn image_admission_and_encoder_cancellation_release_all_buffers() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let model = Model::load(
            root.join(".cache/qwen3.5-0.8b-source"),
            ResourceLimits {
                context: 256,
                max_memory: 1 << 30,
            },
            &Default::default(),
        )
        .unwrap();
        let baseline = model.memory().reserved;
        let hardware = crate::planner::HardwareSnapshot {
            os: "test".into(),
            architecture: "test".into(),
            logical_cpus: 1,
            cpu_features: vec![],
            ram_total_bytes: None,
            ram_available_bytes: None,
            swap_total_bytes: None,
            swap_used_bytes: None,
            observed_unix_ms: 0,
            gpu_available_bytes: None,
            gpu_backend_qualified: false,
        };
        let execution = || Execution {
            profile: crate::planner::Profile::Reference,
            threads: Some(1),
            hardware: &hardware,
        };
        let text_reference: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("fixtures/qwen35/reference-v1.json")).unwrap(),
        )
        .unwrap();
        for case in text_reference["cases"].as_array().unwrap() {
            let messages: Vec<crate::tokenizer::Message> =
                serde_json::from_value(case["messages"].clone()).unwrap();
            let prompt = model.render(&messages).unwrap();
            assert_eq!(prompt, case["prompt"]);
            assert_eq!(
                serde_json::json!(model.encode(&prompt).unwrap()),
                case["prompt_token_ids"]
            );
        }
        let image_reference: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("fixtures/qwen35/vision/reference-v1.json")).unwrap(),
        )
        .unwrap();
        for case in image_reference["cases"].as_array().unwrap() {
            let ids = case["images"].as_array().unwrap();
            let images = ids
                .iter()
                .map(|id| {
                    std::fs::read(root.join(format!(
                        "fixtures/qwen35/vision/{}.png",
                        id.as_str().unwrap()
                    )))
                    .unwrap()
                })
                .collect();
            let mut parts: Vec<_> = (0..ids.len())
                .map(|index| ChatPart::Image { index })
                .collect();
            parts.push(ChatPart::Text {
                text: case["text"].as_str().unwrap().into(),
            });
            let messages = vec![ChatMessage {
                role: "user".into(),
                content: ChatContent::Parts(parts),
            }];
            assert_eq!(
                model.render_images(&messages, ids.len()).unwrap(),
                case["prompt"]
            );
            let prepared = model
                .prepare_images(
                    &messages,
                    images,
                    ImageLimits::default(),
                    GenerationOptions {
                        max_tokens: 4,
                        ..Default::default()
                    },
                    256,
                    execution(),
                    1,
                    &Default::default(),
                )
                .unwrap();
            assert_eq!(serde_json::json!(prepared.tokens), case["prompt_token_ids"]);
        }
        let no_images = vec![ChatMessage {
            role: "user".into(),
            content: ChatContent::Text("Hello".into()),
        }];
        let empty = model
            .prepare_images(
                &no_images,
                vec![],
                ImageLimits::default(),
                GenerationOptions {
                    max_tokens: 1,
                    ..Default::default()
                },
                256,
                execution(),
                1,
                &Default::default(),
            )
            .unwrap();
        assert!(empty.images.is_empty());
        drop(empty);
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: ChatContent::Parts(vec![
                ChatPart::Image { index: 0 },
                ChatPart::Text {
                    text: "What color?".into(),
                },
            ]),
        }];
        let image = std::fs::read(root.join("fixtures/qwen35/vision/red.png")).unwrap();
        let options = || GenerationOptions {
            max_tokens: 4,
            ..Default::default()
        };
        let prepare = |bytes: Vec<u8>, context, cancel: &Cancellation| {
            model.prepare_images(
                &messages,
                vec![bytes],
                ImageLimits::default(),
                options(),
                context,
                execution(),
                1,
                cancel,
            )
        };
        assert!(prepare(vec![0; 8], 256, &Default::default()).is_err());
        assert!(matches!(
            prepare(image.clone(), 64, &Default::default()),
            Err(Error {
                code: ErrorCode::ContextLimitExceeded,
                ..
            })
        ));
        let cancelled = Cancellation::default();
        cancelled.cancel();
        assert!(matches!(
            prepare(image.clone(), 256, &cancelled),
            Err(Error {
                code: ErrorCode::Cancelled,
                ..
            })
        ));
        let reserve = model
            .budget
            .reserve(model.memory().limit - baseline - 1024)
            .unwrap();
        assert!(matches!(
            prepare(image.clone(), 256, &Default::default()),
            Err(Error {
                code: ErrorCode::ResourceExhausted,
                ..
            })
        ));
        drop(reserve);
        let cancel = Cancellation::default();
        let prepared = prepare(image.clone(), 256, &cancel).unwrap();
        assert!(matches!(
            model.begin_images(
                prepared,
                &cancel,
                None,
                Some(&mut |_| {
                    cancel.cancel();
                    Ok(())
                })
            ),
            Err(Error {
                code: ErrorCode::Cancelled,
                ..
            })
        ));
        let prepared = prepare(image, 256, &Default::default()).unwrap();
        assert!(
            model
                .begin_images(
                    prepared,
                    &Default::default(),
                    None,
                    Some(&mut |_| Err(Error::request("observer failure")))
                )
                .is_err()
        );
        assert_eq!(model.memory().reserved, baseline);
        assert_eq!(model.memory().session_buffer_capacity_bytes, 0);
    }
}

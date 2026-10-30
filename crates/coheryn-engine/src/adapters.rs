//! Native adapter selection. This registry never imports model-supplied code.
use crate::{
    Cancellation, Error, ErrorCode, GenerationOptions, GenerationResult, Model, ResourceLimits,
    Result, TokenEvent,
    planner::{self, HardwareSnapshot, Profile},
    tokenizer::Message,
};
use sha2::Digest;
use std::{collections::BTreeMap, path::Path};

#[derive(Clone, serde::Serialize)]
pub struct Capabilities {
    pub adapter: &'static str,
    pub architecture: &'static str,
    pub format: &'static str,
    pub text_generation: bool,
    pub images: bool,
    pub managed_api: bool,
    pub continuous_batching: bool,
    pub qualification: &'static str,
}
pub struct Execution<'a> {
    pub profile: Profile,
    pub threads: Option<usize>,
    pub hardware: &'a HardwareSnapshot,
}
pub trait TextModel: Send + Sync {
    fn capabilities(&self) -> Capabilities;
    fn render(&self, messages: &[Message]) -> Result<String>;
    fn generate(
        &self,
        prompt: &str,
        options: &GenerationOptions,
        execution: Execution<'_>,
        cancel: &Cancellation,
        emit: &mut dyn FnMut(TokenEvent) -> Result<()>,
    ) -> Result<GenerationResult>;
}
pub trait TextAdapter: Send + Sync {
    fn id(&self) -> &'static str;
    /// Metadata-only architecture binding; None means this adapter does not claim the source.
    fn describe(
        &self,
        _source: &mut crate::catalog::Source,
        _cancel: &Cancellation,
    ) -> Option<Result<crate::catalog::TextMetadata>> {
        None
    }

    /// A positive claim must identify format/architecture, never a filename alias.
    fn accepts(&self, path: &Path, limits: ResourceLimits) -> Result<bool>;
    fn load(
        &self,
        path: &Path,
        limits: ResourceLimits,
        cancel: &Cancellation,
    ) -> Result<Box<dyn TextModel>>;
}
pub struct Registry {
    adapters: BTreeMap<&'static str, Box<dyn TextAdapter>>,
}
impl Registry {
    pub(crate) fn metadata_adapters(&self) -> impl Iterator<Item = &dyn TextAdapter> {
        self.adapters.values().map(Box::as_ref)
    }
    pub fn new() -> Self {
        Self {
            adapters: BTreeMap::new(),
        }
    }
    pub fn register(&mut self, adapter: Box<dyn TextAdapter>) -> Result<()> {
        let id = adapter.id();
        if id.is_empty() || id.len() > 128 || self.adapters.contains_key(id) {
            return Err(Error::request("duplicate/invalid native adapter ID"));
        }
        self.adapters.insert(id, adapter);
        Ok(())
    }
    pub fn load(
        &self,
        path: impl AsRef<Path>,
        limits: ResourceLimits,
        cancel: &Cancellation,
    ) -> Result<Box<dyn TextModel>> {
        let path = path.as_ref();
        let mut selected = None;
        for adapter in self.adapters.values() {
            cancel.check()?;
            if adapter.accepts(path, limits)? {
                if selected.is_some() {
                    return Err(Error::request("ambiguous native adapter selection"));
                }
                selected = Some(adapter);
            }
        }
        selected
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::UnsupportedModelArchitecture,
                    "no native text adapter implements this model source",
                )
            })?
            .load(path, limits, cancel)
    }
}
impl Default for Registry {
    fn default() -> Self {
        let mut registry = Self::new();
        registry
            .register(Box::new(Qwen2))
            .expect("unique built-in adapter");
        registry
            .register(Box::new(Qwen35))
            .expect("unique built-in adapter");
        registry
    }
}
struct Qwen2;
impl TextAdapter for Qwen2 {
    fn id(&self) -> &'static str {
        "qwen2-gguf-cpu-v1"
    }
    fn accepts(&self, path: &Path, limits: ResourceLimits) -> Result<bool> {
        if path.is_dir() {
            return Ok(false);
        }
        let metadata = std::fs::symlink_metadata(path)?;
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            return Err(Error::request(
                "native adapter requires a regular model file",
            ));
        }
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;
        let mut magic = [0; 4];
        let n = file.read(&mut magic)?;
        if n != 4 || (magic != *b"GGUF" && magic != *b"FUGG") {
            return Ok(false);
        }
        let mut parser = crate::ParseLimits::default();
        parser.max_metadata_allocation = parser.max_metadata_allocation.min(limits.max_memory / 2);
        Ok(crate::inspect(path, parser)?.architecture.as_deref() == Some("qwen2"))
    }
    fn describe(
        &self,
        source: &mut crate::catalog::Source,
        _cancel: &Cancellation,
    ) -> Option<Result<crate::catalog::TextMetadata>> {
        if source.format() != crate::catalog::Format::Gguf || source.architecture() != Some("qwen2")
        {
            return None;
        }
        Some((|| {
            let (_, report) = source.gguf().expect("GGUF profile");
            if !report.executable {
                return Err(report
                    .diagnostics
                    .first()
                    .cloned()
                    .unwrap_or_else(|| Error::request("unsupported GGUF profile")));
            }
            let config = report
                .config
                .clone()
                .ok_or_else(|| Error::request("missing decoder configuration"))?;
            let json = serde_json::to_value(&config).map_err(|e| Error::request(e.to_string()))?;
            Ok(crate::catalog::TextMetadata::new(
                qwen2_capabilities(),
                json,
                config.context,
                config.context,
                move |source, request, hardware| {
                    let (descriptor, _) = source
                        .gguf()
                        .ok_or_else(|| Error::request("mismatched metadata source"))?;
                    planner::estimate(planner::PlanInput {
                        constraints: request.constraints,
                        descriptor,
                        config: &config,
                        model: "local",
                        profile: request.profile,
                        context: request.context,
                        concurrency: request.concurrency,
                        max_concurrency: request.concurrency,
                        loaded: false,
                        budget: crate::memory::MemoryStats {
                            limit: request.max_memory,
                            ..Default::default()
                        },
                        hardware: hardware.clone(),
                        images: None,
                        threads: request.threads,
                    })
                },
            ))
        })())
    }
    fn load(
        &self,
        path: &Path,
        limits: ResourceLimits,
        cancel: &Cancellation,
    ) -> Result<Box<dyn TextModel>> {
        cancel.check()?;
        Ok(Box::new(Model::load(path, limits)?))
    }
}
struct Qwen35;
impl TextAdapter for Qwen35 {
    fn id(&self) -> &'static str {
        "qwen35-safetensors-cpu-v1"
    }
    fn accepts(&self, path: &Path, _limits: ResourceLimits) -> Result<bool> {
        let selected = path.is_dir()
            || path.extension().is_some_and(|x| x == "safetensors")
            || path
                .file_name()
                .is_some_and(|x| x == "model.safetensors.index.json");
        if !selected {
            return Ok(false);
        }
        let root = if path.is_dir() {
            path
        } else {
            path.parent().unwrap_or(Path::new("."))
        };
        let config = root.join("config.json");
        if !config.exists() {
            return Ok(false);
        }
        let bytes = crate::safetensors::bounded_bytes(&config, 1 << 20)?;
        let json = crate::safetensors::configuration(&bytes)?;
        Ok(json["model_type"] == "qwen3_5")
    }
    fn describe(
        &self,
        source: &mut crate::catalog::Source,
        cancel: &Cancellation,
    ) -> Option<Result<crate::catalog::TextMetadata>> {
        if source.format() != crate::catalog::Format::Safetensors
            || source.architecture() != Some("qwen3_5")
        {
            return None;
        }
        Some((|| {
            let config = crate::qwen35::Config::parse(
                source.configuration().expect("source configuration"),
            )?;
            config.validate_tensors(source.tensors().expect("tensor source"))?;
            // These IDs and vocabulary size are properties of the exact tokenizer hash below.
            if config.eos != 248044 || config.vocabulary < 248070 {
                return Err(Error::new(
                    ErrorCode::UnsupportedTokenizer,
                    "Qwen3.5 vocabulary/special-token mismatch",
                ));
            }
            let tokenizer = source.read_asset("tokenizer.json", 16 << 20, cancel)?;
            if crate::storage::digest_hex(sha2::Sha256::digest(&tokenizer).into())
                != crate::qwen35::TOKENIZER_SHA256
            {
                return Err(Error::new(
                    ErrorCode::UnsupportedTokenizer,
                    "unqualified Qwen3.5 tokenizer profile",
                ));
            }
            drop(tokenizer);
            let template = source.read_asset("chat_template.jinja", 16 << 10, cancel)?;
            if crate::storage::digest_hex(sha2::Sha256::digest(&template).into())
                != crate::qwen35::TEMPLATE_SHA256
            {
                return Err(Error::new(
                    ErrorCode::UnsupportedChatTemplate,
                    "unqualified Qwen3.5 chat template",
                ));
            }
            let mut capabilities = qwen35_capabilities();
            let root = source.tensors().expect("tensor source").root().to_owned();
            if root.join("preprocessor_config.json").try_exists()? {
                let processor = source.read_asset("preprocessor_config.json", 1 << 20, cancel)?;
                capabilities.images = crate::qwen35::vision::VisionConfig::parse(
                    source.configuration().expect("configuration"),
                    &processor,
                    source.tensors().expect("tensor source"),
                )
                .is_ok();
            }
            let json = serde_json::to_value(&config).map_err(|e| Error::request(e.to_string()))?;
            Ok(crate::catalog::TextMetadata::new(
                capabilities,
                json,
                config.context,
                config.context.min(8192),
                move |source, request, hardware| {
                    crate::qwen35::estimate(
                        &config,
                        source.tensors().expect("tensor source").tensor_bytes()?,
                        request,
                        hardware,
                        crate::memory::MemoryStats {
                            limit: request.max_memory,
                            ..Default::default()
                        },
                        false,
                    )
                },
            ))
        })())
    }
    fn load(
        &self,
        path: &Path,
        limits: ResourceLimits,
        cancel: &Cancellation,
    ) -> Result<Box<dyn TextModel>> {
        Ok(Box::new(crate::qwen35::Model::load(path, limits, cancel)?))
    }
}
impl TextModel for Model {
    fn capabilities(&self) -> Capabilities {
        qwen2_capabilities()
    }
    fn render(&self, messages: &[Message]) -> Result<String> {
        Model::render(self, messages)
    }
    fn generate(
        &self,
        prompt: &str,
        options: &GenerationOptions,
        e: Execution<'_>,
        cancel: &Cancellation,
        emit: &mut dyn FnMut(TokenEvent) -> Result<()>,
    ) -> Result<GenerationResult> {
        let plan = planner::estimate(planner::PlanInput {
            constraints: options.response_format.enabled(),
            descriptor: &self.descriptor,
            config: &self.config,
            model: "standalone",
            profile: e.profile,
            context: self.context_limit(),
            concurrency: 1,
            max_concurrency: 1,
            loaded: true,
            budget: self.memory(),
            hardware: e.hardware.clone(),
            images: None,
            threads: e.threads,
        })?;
        plan.require_admission()?;
        let mut admission = self.reserve_session_with_execution(
            self.context_limit(),
            plan.additional_reserved_bytes - plan.session_reserved_bytes,
            plan.io_buffer_bytes,
            crate::inference::ExecutionSettings {
                threads: plan.threads_per_session,
                prefill_chunk_tokens: plan.prefill_chunk_tokens,
            },
        )?;
        if plan.constraint_reserved_bytes > 0 {
            admission.constraints = Some(
                admission
                    .reservation
                    .split(plan.constraint_reserved_bytes)?,
            );
        }
        self.generate_admitted(prompt, options, admission, cancel, emit)
    }
}
impl TextModel for crate::qwen35::Model {
    fn capabilities(&self) -> Capabilities {
        let mut result = qwen35_capabilities();
        result.images = self.vision_configuration().is_ok();
        result
    }
    fn render(&self, messages: &[Message]) -> Result<String> {
        Self::render(self, messages)
    }
    fn generate(
        &self,
        prompt: &str,
        options: &GenerationOptions,
        e: Execution<'_>,
        cancel: &Cancellation,
        emit: &mut dyn FnMut(TokenEvent) -> Result<()>,
    ) -> Result<GenerationResult> {
        Self::generate_profiled(self, prompt, options, e, cancel, emit)
    }
}

fn qwen2_capabilities() -> Capabilities {
    Capabilities {
        adapter: "qwen2-gguf-cpu-v1",
        architecture: "qwen2",
        format: "gguf",
        text_generation: true,
        images: false,
        managed_api: true,
        continuous_batching: true,
        qualification: "exact-file-evidence-required",
    }
}

fn qwen35_capabilities() -> Capabilities {
    Capabilities {
        adapter: "qwen35-safetensors-cpu-v1",
        architecture: "qwen3_5",
        format: "safetensors",
        text_generation: true,
        images: false,
        managed_api: true,
        continuous_batching: true,
        qualification: "experimental-cpu-text-and-still-image",
    }
}

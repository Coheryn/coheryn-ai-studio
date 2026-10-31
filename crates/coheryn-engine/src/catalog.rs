//! Immutable, metadata-only model inspection. No graph/session/tokenizer is constructed.
use crate::{
    Cancellation, Error, ErrorCode, Result,
    adapters::{Capabilities, Registry},
    compat::{CompatibilityReport, ModelTask},
    gguf,
    planner::{ExecutionPlan, HardwareSnapshot, Profile},
    safetensors::TensorSet,
};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Format {
    Gguf,
    Safetensors,
}
#[derive(Serialize)]
pub struct SourceFile {
    pub name: String,
    pub role: &'static str,
    pub bytes: u64,
    pub metadata_sha256: String,
    pub hash_scope: &'static str,
}
#[derive(Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum StructuralStatus {
    Compatible,
    Unsupported,
    Invalid,
}
#[derive(Serialize)]
pub struct ModelDescriptor {
    pub schema_version: u32,
    pub format: Format,
    pub layout: &'static str,
    pub architecture: Option<String>,
    pub container_valid: bool,
    pub structural_status: StructuralStatus,
    pub executable: bool,
    pub task: Option<ModelTask>,
    pub text_adapter: Option<Capabilities>,
    pub configuration: Option<Value>,
    pub declared_context: Option<usize>,
    pub runtime_context_limit: Option<usize>,
    pub tensor_count: usize,
    pub tensor_bytes: u64,
    pub encodings: BTreeMap<String, usize>,
    pub source_files: Vec<SourceFile>,
    pub metadata_identity: String,
    pub identity_scope: &'static str,
    pub weight_payloads: &'static str,
    pub qualification_status: &'static str,
    pub admission_status: &'static str,
    pub diagnostics: Vec<Error>,
}
/// Opaque source snapshot; callers cannot replace its validated tensor catalog/configuration.
pub struct Source {
    path: PathBuf,
    format: Format,
    gguf: Option<(gguf::Descriptor, CompatibilityReport)>,
    safetensors: Option<TensorSet>,
    config: Option<Value>,
    files: Vec<SourceFile>,
    problems: Vec<Error>,
}
fn hash(bytes: &[u8]) -> String {
    crate::storage::digest_hex(Sha256::digest(bytes).into())
}
fn bounded(mut error: Error) -> Error {
    if error.message.len() > 4096 {
        error.message = error.message.chars().take(1024).collect::<String>() + "...";
    }
    error
}
impl Source {
    fn open(path: &Path, cancel: &Cancellation) -> Result<Self> {
        cancel.check()?;
        let meta = std::fs::symlink_metadata(path)?;
        if meta.file_type().is_symlink() || (!meta.is_dir() && !meta.is_file()) {
            return Err(Error::request(
                "model source must be a regular file or controlled directory",
            ));
        }
        let mut magic = [0; 4];
        let mut n = 0;
        if meta.is_file() {
            n = std::fs::File::open(path)?.read(&mut magic)?;
        }
        let is_gguf = meta.is_file()
            && ((n == 4 && (magic == *b"GGUF" || magic == *b"FUGG"))
                || path.extension().is_some_and(|e| e == "gguf"));
        if is_gguf {
            let d = gguf::inspect(path, gguf::ParseLimits::default())?;
            let compatibility = crate::validate(&d);
            let file = SourceFile {
                name: path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned(),
                role: "weights",
                bytes: d.file_bytes,
                metadata_sha256: d.header_sha256.clone(),
                hash_scope: "gguf-header",
            };
            cancel.check()?;
            return Ok(Self {
                path: path.canonicalize()?,
                format: Format::Gguf,
                gguf: Some((d, compatibility)),
                safetensors: None,
                config: None,
                files: vec![file],
                problems: vec![],
            });
        }
        if !meta.is_dir()
            && !path.extension().is_some_and(|e| e == "safetensors")
            && !path
                .file_name()
                .is_some_and(|e| e == "model.safetensors.index.json")
        {
            return Err(Error::new(
                ErrorCode::UnsupportedFeature,
                "expected GGUF or an explicit Safetensors checkpoint entrypoint",
            ));
        }
        let set = TensorSet::open(path, crate::safetensors::Limits::default(), cancel)?;
        let mut files = Vec::new();
        for file in set.files() {
            files.push(SourceFile {
                name: file.file.clone(),
                role: "weights",
                bytes: file.bytes,
                metadata_sha256: file.header_sha256.clone(),
                hash_scope: "safetensors-header-prefix-and-json",
            });
        }
        if let Some((bytes, digest)) = set.index_identity() {
            files.push(SourceFile {
                name: set
                    .entrypoint()
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned(),
                role: "index",
                bytes: *bytes as u64,
                metadata_sha256: digest.clone(),
                hash_scope: "complete-file",
            });
        }
        let mut source = Self {
            path: path.into(),
            format: Format::Safetensors,
            gguf: None,
            safetensors: Some(set),
            config: None,
            files,
            problems: vec![],
        };
        match source
            .read_asset("config.json", 1 << 20, cancel)
            .and_then(|bytes| crate::safetensors::configuration(&bytes))
        {
            Ok(value) => source.config = Some(value),
            Err(error)
                if matches!(
                    error.code,
                    ErrorCode::ResourceExhausted
                        | ErrorCode::Cancelled
                        | ErrorCode::DeadlineExceeded
                ) =>
            {
                return Err(error);
            }
            Err(error) => source.problems.push(bounded(error)),
        }
        Ok(source)
    }
    pub(crate) fn prepare_qwen35_gguf(&mut self, cancel: &Cancellation) -> Result<()> {
        if self.safetensors.is_some() {
            return Ok(());
        }
        let root = self.path.parent().unwrap_or(Path::new("."));
        let bytes = crate::safetensors::bounded_bytes(&root.join("config.json"), 1 << 20)?;
        let json = crate::safetensors::configuration(&bytes)?;
        let descriptor = &self
            .gguf
            .as_ref()
            .ok_or_else(|| Error::request("missing GGUF descriptor"))?
            .0;
        let (_, set) = crate::qwen35::gguf::open(&self.path, descriptor, &json, cancel)?;
        for f in set.files() {
            if !self.files.iter().any(|p| p.name == f.file) {
                self.files.push(SourceFile {
                    name: f.file.clone(),
                    role: "vision-weights",
                    bytes: f.bytes,
                    metadata_sha256: f.header_sha256.clone(),
                    hash_scope: "gguf-header",
                });
            }
        }
        self.safetensors = Some(set);
        self.config = Some(json);
        self.files.push(SourceFile {
            name: "config.json".into(),
            role: "asset",
            bytes: bytes.len() as u64,
            metadata_sha256: hash(&bytes),
            hash_scope: "complete-file",
        });
        self.read_asset("config.json", 1 << 20, cancel)?;
        Ok(())
    }
    pub fn format(&self) -> Format {
        self.format
    }
    pub fn gguf(&self) -> Option<(&gguf::Descriptor, &CompatibilityReport)> {
        self.gguf.as_ref().map(|(d, c)| (d, c))
    }
    pub fn tensors(&self) -> Option<&TensorSet> {
        self.safetensors.as_ref()
    }
    pub fn configuration(&self) -> Option<&Value> {
        self.config.as_ref()
    }
    pub fn architecture(&self) -> Option<&str> {
        match &self.gguf {
            Some((d, _)) => d.architecture.as_deref(),
            None => self.config.as_ref()?.get("model_type")?.as_str(),
        }
    }
    pub fn read_asset(
        &mut self,
        name: &str,
        limit: usize,
        cancel: &Cancellation,
    ) -> Result<Vec<u8>> {
        cancel.check()?;
        if !matches!(
            name,
            "config.json" | "tokenizer.json" | "chat_template.jinja" | "preprocessor_config.json"
        ) {
            return Err(Error::request("unsupported source asset role"));
        }
        let root = self
            .safetensors
            .as_ref()
            .ok_or_else(|| Error::request("source has no external assets"))?
            .root();
        let bytes = crate::safetensors::bounded_bytes(&root.join(name), limit).map_err(|e| {
            Error::new(
                e.code,
                format!("required source asset {name}: {}", e.message),
            )
        })?;
        let digest = hash(&bytes);
        if let Some(previous) = self
            .files
            .iter()
            .find(|f| f.name == name && f.role == "asset")
        {
            if previous.metadata_sha256 != digest {
                return Err(Error::new(
                    ErrorCode::ModelChanged,
                    "source asset changed during inspection",
                ));
            }
        } else {
            self.files.push(SourceFile {
                name: name.into(),
                role: "asset",
                bytes: bytes.len() as u64,
                metadata_sha256: digest,
                hash_scope: "complete-file",
            });
        }
        cancel.check()?;
        Ok(bytes)
    }
}
#[derive(Clone)]
pub struct PlanRequest {
    pub context: usize,
    pub concurrency: usize,
    pub max_memory: u64,
    pub profile: Profile,
    pub threads: Option<usize>,
    pub constraints: bool,
}
type Estimator =
    dyn Fn(&Source, &PlanRequest, &HardwareSnapshot) -> Result<ExecutionPlan> + Send + Sync;
pub struct TextMetadata {
    pub capabilities: Capabilities,
    pub configuration: Value,
    pub declared_context: usize,
    pub runtime_context: usize,
    estimator: Box<Estimator>,
}
impl TextMetadata {
    pub fn new(
        capabilities: Capabilities,
        configuration: Value,
        declared_context: usize,
        runtime_context: usize,
        estimator: impl Fn(&Source, &PlanRequest, &HardwareSnapshot) -> Result<ExecutionPlan>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self {
            capabilities,
            configuration,
            declared_context,
            runtime_context,
            estimator: Box::new(estimator),
        }
    }
}
/// Inspection owns a fixed report and private planning configuration. It is not deserializable.
pub struct Inspection {
    source: Source,
    descriptor: ModelDescriptor,
    text: Option<TextMetadata>,
}
impl Registry {
    pub fn inspect(&self, path: impl AsRef<Path>, cancel: &Cancellation) -> Result<Inspection> {
        let mut source = Source::open(path.as_ref(), cancel)?;
        cancel.check()?;
        if source.architecture().is_some_and(|a| a.len() > 128) {
            return Err(Error::request("architecture identifier exceeds 128 bytes"));
        }
        let mut diagnostics = source.problems.clone();
        let mut selected = None;
        let mut claimed = false;
        for adapter in self.metadata_adapters() {
            cancel.check()?;
            if let Some(result) = adapter.describe(&mut source, cancel) {
                if claimed {
                    return Err(Error::request(
                        "ambiguous native metadata adapter selection",
                    ));
                }
                claimed = true;
                match result {
                    Ok(metadata) => {
                        let format = match source.format() {
                            Format::Gguf => "gguf",
                            Format::Safetensors => "safetensors",
                        };
                        if metadata.capabilities.adapter != adapter.id()
                            || Some(metadata.capabilities.architecture) != source.architecture()
                            || metadata.capabilities.format != format
                            || !metadata.capabilities.text_generation
                            || metadata.runtime_context == 0
                            || metadata.runtime_context > metadata.declared_context
                        {
                            return Err(Error::new(
                                ErrorCode::Internal,
                                "native metadata adapter returned an inconsistent capability contract",
                            ));
                        }
                        selected = Some(metadata)
                    }
                    Err(error)
                        if matches!(
                            error.code,
                            ErrorCode::ResourceExhausted
                                | ErrorCode::Cancelled
                                | ErrorCode::DeadlineExceeded
                        ) =>
                    {
                        return Err(error);
                    }
                    Err(error) => diagnostics.push(bounded(error)),
                }
            }
        }
        let mut task = selected
            .as_ref()
            .map(|_| ModelTask::TextGeneration)
            .or_else(|| source.gguf().and_then(|(_, legacy)| legacy.task));
        let mut executable = selected.is_some() && diagnostics.is_empty();
        let mut configuration = selected.as_ref().map(|m| m.configuration.clone());
        let mut declared_context = selected.as_ref().map(|m| m.declared_context);
        let mut runtime_context_limit = selected.as_ref().map(|m| m.runtime_context);
        if !claimed && let Some((_, legacy)) = source.gguf() {
            diagnostics.extend(legacy.diagnostics.iter().cloned().map(bounded));
            task = legacy.task;
            executable = legacy.executable && legacy.task == Some(ModelTask::Embedding);
            if legacy.task != Some(ModelTask::Embedding) && diagnostics.is_empty() {
                diagnostics.push(Error::new(
                    ErrorCode::UnsupportedModelArchitecture,
                    "no registered native text metadata adapter",
                ));
            }
            if let Some(c) = &legacy.embedding_config {
                configuration =
                    Some(serde_json::to_value(c).map_err(|e| Error::request(e.to_string()))?);
                declared_context = Some(c.model_context);
                runtime_context_limit = Some(c.qualified_context);
            }
        } else if !claimed && diagnostics.is_empty() {
            diagnostics.push(Error::new(
                ErrorCode::UnsupportedModelArchitecture,
                format!(
                    "no native metadata adapter for architecture {}",
                    source.architecture().unwrap_or("<missing>")
                ),
            ));
        }
        let (count, bytes, encodings) =
            if let Some((d, _)) = source.gguf().filter(|_| source.tensors().is_none()) {
                (d.tensors.len(), d.tensor_bytes, d.encodings.clone())
            } else {
                let set = source.tensors().expect("validated source");
                let mut total = 0u64;
                let mut encodings = BTreeMap::new();
                for f in set.files() {
                    for t in &f.tensors {
                        total = total
                            .checked_add(t.bytes)
                            .ok_or_else(|| Error::resource("tensor inventory overflow"))?;
                        *encodings.entry(t.dtype.clone()).or_insert(0) += 1;
                    }
                }
                (set.tensor_count(), total, encodings)
            };
        let status = if executable {
            StructuralStatus::Compatible
        } else if diagnostics.iter().any(|e| {
            !matches!(
                e.code,
                ErrorCode::UnsupportedFeature
                    | ErrorCode::UnsupportedModelArchitecture
                    | ErrorCode::UnsupportedTokenizer
                    | ErrorCode::UnsupportedChatTemplate
                    | ErrorCode::UnsupportedGgufTensorType
            )
        }) {
            StructuralStatus::Invalid
        } else {
            StructuralStatus::Unsupported
        };
        let layout = if source.format == Format::Gguf {
            if source.tensors().is_none() {
                "single-file"
            } else if source.tensors().is_some_and(|s| s.files().len() > 1) {
                "paired-with-assets"
            } else {
                "single-file-with-assets"
            }
        } else if source
            .tensors()
            .expect("tensor source")
            .index_identity()
            .is_some()
        {
            "indexed"
        } else {
            "single-file"
        };
        source
            .files
            .sort_by(|a, b| (&a.role, &a.name).cmp(&(&b.role, &b.name)));
        let identity = serde_json::to_vec(&(
            1,
            source.format,
            layout,
            source.architecture(),
            &source.files,
            selected.as_ref().map(|m| m.capabilities.adapter),
        ))
        .map_err(|e| Error::request(e.to_string()))?;
        cancel.check()?;
        let descriptor = ModelDescriptor {
            schema_version: 1,
            format: source.format,
            layout,
            architecture: source.architecture().map(str::to_owned),
            container_valid: true,
            structural_status: status,
            executable,
            task,
            text_adapter: selected.as_ref().map(|m| m.capabilities.clone()),
            configuration,
            declared_context,
            runtime_context_limit,
            tensor_count: count,
            tensor_bytes: bytes,
            encodings,
            source_files: std::mem::take(&mut source.files),
            metadata_identity: hash(&identity),
            identity_scope: "container headers, selected index/assets, architecture profile; excludes weight payload bytes; not a model/cache identity",
            weight_payloads: "not-scanned",
            qualification_status: "unqualified-payload-identities-not-verified",
            admission_status: "not-evaluated",
            diagnostics,
        };
        Ok(Inspection {
            source,
            descriptor,
            text: selected,
        })
    }
}
impl Inspection {
    /// Immutable original configuration from this inspected source, before adapter normalization.
    pub fn qwen35_vision_configuration(
        &mut self,
        cancel: &Cancellation,
    ) -> Result<crate::qwen35::vision::VisionConfig> {
        self.require_compatible()?;
        if !matches!(self.source.architecture(), Some("qwen3_5" | "qwen35")) {
            return Err(Error::new(
                crate::ErrorCode::UnsupportedModelArchitecture,
                "source has no Qwen3.5 image planner",
            ));
        }
        if self
            .descriptor
            .text_adapter
            .as_ref()
            .is_none_or(|a| !a.images)
        {
            return Err(Error::new(
                crate::ErrorCode::UnsupportedImageProcessor,
                "source has no validated native image processor/component",
            ));
        }
        let processor = self
            .source
            .read_asset("preprocessor_config.json", 1 << 20, cancel)?;
        crate::qwen35::vision::VisionConfig::parse(
            self.source.configuration().expect("configuration"),
            &processor,
            self.source.tensors().expect("tensor source"),
        )
    }
    pub fn plan_images(
        &mut self,
        request: &PlanRequest,
        hardware: &HardwareSnapshot,
        images: &crate::planner::ImageWorkload,
        limits: crate::multimodal::image::ImageLimits,
        cancel: &Cancellation,
    ) -> Result<crate::qwen35::image_plan::ImageExecutionPlan> {
        let vision = self.qwen35_vision_configuration(cancel)?;
        let config =
            crate::qwen35::Config::parse(self.source_configuration().expect("configuration"))?;
        crate::qwen35::image_plan::estimate_images(
            &config,
            &vision,
            self.descriptor.tensor_bytes,
            request,
            hardware,
            crate::memory::MemoryStats {
                limit: request.max_memory,
                ..Default::default()
            },
            false,
            request.concurrency,
            images,
            limits,
        )
    }
    pub fn source_configuration(&self) -> Option<&Value> {
        self.source.configuration()
    }
    pub fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
    pub fn gguf(&self) -> Option<(&gguf::Descriptor, &CompatibilityReport)> {
        self.source.gguf()
    }
    pub fn require_compatible(&self) -> Result<()> {
        if self.descriptor.executable {
            Ok(())
        } else {
            Err(self
                .descriptor
                .diagnostics
                .first()
                .cloned()
                .unwrap_or_else(|| {
                    Error::new(
                        ErrorCode::UnsupportedFeature,
                        "model source is not executable",
                    )
                }))
        }
    }
    pub fn plan(
        &self,
        request: &PlanRequest,
        hardware: &HardwareSnapshot,
        cancel: &Cancellation,
    ) -> Result<ExecutionPlan> {
        cancel.check()?;
        self.require_compatible()?;
        if let Some(set) = self.source.tensors() {
            set.unchanged()?;
        }
        let metadata=self.text.as_ref().ok_or_else(||Error::new(ErrorCode::UnsupportedFeature,"local generic planning currently supports text-generation adapters; embedding/image planning retains its existing managed API"))?;
        if request.context == 0 || request.context > metadata.runtime_context {
            return Err(Error::request(
                "context exceeds the selected native profile limit",
            ));
        }
        (metadata.estimator)(&self.source, request, hardware)
    }
    pub fn deep_validate(&self, cancel: &Cancellation) -> Result<()> {
        cancel.check()?;
        self.require_compatible()?;
        if let Some(set) = self.source.tensors() {
            set.validate_payloads(cancel)
        } else if let Some((d, _)) = self.source.gguf() {
            crate::validate_tokenizer(d)?;
            crate::deep_validate(&self.source.path, d, cancel)
        } else {
            self.source
                .tensors()
                .expect("tensor source")
                .validate_payloads(cancel)
        }
    }
}

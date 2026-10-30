use crate::{
    auth::{Auth, Credentials},
    config::ServerConfig,
    embeddings::EmbeddingSpace,
    error::{ApiError, Result},
    registry::{ModelRecord, Registry, validate_hash},
    resident::{Resident, ResidentEmbedding},
    scheduler::Scheduler,
    wire::EstimateRequest,
};
use coheryn_engine::{Error, ErrorCode, Model, ParseLimits, ResourceLimits, memory::Budget};
use futures_util::{Stream, StreamExt};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::io::AsyncWriteExt;
use tokio_util::sync::CancellationToken;

pub struct Service {
    pub config: ServerConfig,
    pub auth: Auth,
    pub registry: Arc<Registry>,
    pub budget: Budget,
    pub scheduler: Scheduler,
    pub sessions: crate::sessions::Sessions,
    pub shutdown: CancellationToken,
    pub admin: tokio::sync::Mutex<()>,
    residents: Mutex<BTreeMap<String, Resident>>,
    pub http_slots: Arc<tokio::sync::Semaphore>,
}
impl Service {
    pub fn open(config: ServerConfig, credentials: Credentials) -> Result<Arc<Self>> {
        config.validate()?;
        let registry = Arc::new(Registry::open(&config)?);
        let auth = Auth::new(&credentials)?;
        let budget = Budget::new(config.max_memory);
        let shutdown = CancellationToken::new();
        let sessions = crate::sessions::Sessions::open(
            config.sessions.clone(),
            config.cache.clone(),
            config.persistence.clone(),
            budget.clone(),
            &config.data_dir,
        )?;
        let scheduler = Scheduler::new(config.clone(), budget.clone(), shutdown.clone())
            .with_sessions(sessions.clone());
        Ok(Arc::new(Self {
            config,
            auth,
            registry,
            budget,
            scheduler,
            sessions,
            shutdown,
            admin: tokio::sync::Mutex::new(()),
            residents: Mutex::new(BTreeMap::new()),
            http_slots: Arc::new(tokio::sync::Semaphore::new(64)),
        }))
    }
    fn residents(&self) -> Result<std::sync::MutexGuard<'_, BTreeMap<String, Resident>>> {
        self.residents
            .lock()
            .map_err(|_| Error::new(ErrorCode::Internal, "resident model lock poisoned").into())
    }
    fn resident(&self, id: &str) -> Result<Resident> {
        crate::registry::validate_id(id)?;
        if let Some(model) = self.residents()?.get(id).cloned() {
            return Ok(model);
        }
        self.registry.get(id)?;
        Err(Error::new(
            ErrorCode::Unavailable,
            "model is not loaded; use models load first",
        )
        .into())
    }
    pub fn model(&self, id: &str) -> Result<Arc<Model>> {
        match self.resident(id)? {
            Resident::Generation(m) => Ok(m),
            Resident::Embedding(_) | Resident::Vision(_) | Resident::Qwen35(_) => Err(Error::new(
                ErrorCode::UnsupportedModelArchitecture,
                "selected model does not support managed text sessions; use its stateless generation interface",
            )
            .into()),
        }
    }
    pub fn text_model(&self, id: &str) -> Result<crate::resident::TextResident> {
        match self.resident(id)? {
            Resident::Generation(model) => Ok(crate::resident::TextResident::Gguf(model)),
            Resident::Qwen35(model) => Ok(crate::resident::TextResident::Qwen35(model)),
            _ => Err(Error::new(
                ErrorCode::UnsupportedModelArchitecture,
                "selected model cannot serve text conversations",
            )
            .into()),
        }
    }
    pub fn submit_generation(
        &self,
        request: crate::wire::GenerationRequest,
        ingress: &mut Option<coheryn_engine::memory::Reservation>,
    ) -> Result<crate::scheduler::GenerationHandle> {
        match self.resident(&request.model)? {
            Resident::Generation(model) => self.scheduler.submit(model, request),
            Resident::Qwen35(model) => self.scheduler.submit_qwen35(model, request, ingress.take()),
            _ => Err(Error::new(
                ErrorCode::UnsupportedModelArchitecture,
                "selected model cannot serve stateless text generation",
            )
            .into()),
        }
    }
    pub fn hybrid_vision(&self, id: &str) -> Result<Option<Arc<coheryn_engine::qwen35::Model>>> {
        match self.resident(id)? {
            Resident::Qwen35(m) => {
                m.vision_configuration()?;
                Ok(Some(m))
            }
            _ => Ok(None),
        }
    }
    pub fn vision(&self, id: &str) -> Result<Arc<coheryn_engine::multimodal::qwen2vl::Qwen2Vl>> {
        match self.resident(id)? {
            Resident::Vision(m) => Ok(m),
            _ => Err(Error::new(
                ErrorCode::UnsupportedModelArchitecture,
                "image chat requires a loaded compatible language/vision model pair",
            )
            .into()),
        }
    }
    pub fn is_vision(&self, id: &str) -> Result<bool> {
        Ok(matches!(self.resident(id)?, Resident::Vision(_)))
    }
    pub fn require_tool_model(
        &self,
        id: &str,
        config: &coheryn_engine::tools::ToolConfig,
        tool_history: bool,
    ) -> Result<()> {
        config.validate()?;
        if (config.enabled() || tool_history)
            && !coheryn_engine::tools::qualified_model_hash(&self.registry.get(id)?.sha256)
        {
            return Err(Error::new(ErrorCode::UnsupportedFeature,"model bytes are not qualified for qwen-tools-v1; inspect the exact-file compatibility matrix").into());
        }
        Ok(())
    }
    pub fn embedding(&self, id: &str) -> Result<Arc<ResidentEmbedding>> {
        match self.resident(id)? {
            Resident::Embedding(m) => Ok(m),
            Resident::Generation(_) | Resident::Vision(_) | Resident::Qwen35(_) => Err(Error::new(
                ErrorCode::UnsupportedModelArchitecture,
                "generation model cannot perform qualified embedding inference",
            )
            .into()),
        }
    }
    pub fn loaded(&self) -> Result<Value> {
        Ok(Value::Array(self.residents()?.iter().map(|(id,m)|json!({"id":id,"context":m.context(),"task":m.task(),"pinned":m.pinned(),"conversation":m.conversation_capabilities(),"embedding_space":match m {Resident::Embedding(m)=>Some(&m.space),_=>None}})).collect()))
    }
    pub fn status(&self) -> Result<Value> {
        Ok(
            json!({"stage":"experimental-G4-E2-S6","continuous_batching":true,"batching_profile":"coheryn-cpu-batch-v1","compute_policy":"fifo-text-token-batch-embedding-microbatch-vision-tile-layer","image_input":true,"vision_limits":self.config.vision,"vision_profile":"qwen2vl-still-f16-cpu-greedy","generation_resident_limit":self.scheduler.max_concurrency(),"placement":{"backend":"native-cpu","fallback":"disabled","reason":"CPU is the only qualified execution backend"},"session_image_assets":self.sessions.image_stats()?,"session_limits":self.config.sessions,"session_storage":if self.sessions.persistence_enabled(){"opt-in-embedded"}else{"ephemeral"},"persistence_limits":self.config.persistence,"session_kv_reuse":true,"tool_calling":true,"tool_profiles":["qwen-tools-v1"],"strict_tool_arguments":true,"schema_constraints":true,"constraint_profile":"coheryn-json-v1","strict_tool_mode":"single-required-or-named-no-parallel","sampling_profile":"coheryn-sampling-v2","cache":self.sessions.cache_stats()?,"embedding_limits":self.config.embeddings,"backend":"cpu-reference","scheduler":self.scheduler.counters.snapshot(),"memory":self.budget.stats(),"loaded_models":self.loaded()?,"queue_capacity":self.config.queue_capacity,"active_limit":2*self.scheduler.max_concurrency()+self.config.embeddings.concurrency,"vision_phase_limit":self.scheduler.max_concurrency(),"compute_unit_limit":1,"embedding_phase_limit":self.config.embeddings.concurrency,"configured_active_limit":self.config.max_concurrency,"shutting_down":self.shutdown.is_cancelled()}),
        )
    }
    pub fn unload(&self, id: &str) -> Result<()> {
        crate::registry::validate_id(id)?;
        let mut residents = self.residents()?;
        if let Some(model) = residents.get(id)
            && model.pinned()
        {
            return Err(ApiError::conflict(
                "model is pinned by an active or queued request",
            ));
        }
        self.sessions.evict_model(id)?;
        residents
            .remove(id)
            .ok_or_else(|| ApiError::missing("model is not loaded"))?;
        Ok(())
    }
    pub fn remove(&self, id: &str) -> Result<()> {
        if self.residents()?.contains_key(id) {
            return Err(ApiError::conflict("unload the model before removing it"));
        }
        self.registry.remove(id)
    }
    pub async fn register_pair(
        self: &Arc<Self>,
        id: String,
        language: String,
        vision: String,
    ) -> Result<ModelRecord> {
        let service = self.clone();
        tokio::spawn(async move {
            let _admin = service.admin.lock().await;
            let record = service.registry.pair_record(&id, &language, &vision)?;
            let validation_service = service.clone();
            tokio::task::spawn_blocking(move || -> Result<()> {
                let _inspection = validation_service.budget.reserve(128 << 20)?;
                crate::vision_planner::descriptors(&validation_service, &record)?;
                Ok(())
            })
            .await??;
            service.registry.register_pair(&id, &language, &vision)
        })
        .await?
    }
    pub async fn load(self: &Arc<Self>, id: String, context: usize) -> Result<Value> {
        // The owned task finishes the transaction even if the HTTP caller disconnects.
        let service = self.clone();
        tokio::spawn(async move {
            let _admin=service.admin.lock().await;
            if service.shutdown.is_cancelled(){return Err(Error::new(ErrorCode::Unavailable,"server shutting down").into());}
            if context==0||context>service.config.max_context {return Err(ApiError::bad("context exceeds configured server limit"));}
            {
                let residents=service.residents()?;
                if let Some(model)=residents.get(&id) {
                    if model.context()!=context {return Err(ApiError::conflict("model already loaded with a different context; unload explicitly first"));}
                    return Ok(json!({"model":id,"context":context,"loaded":true}));
                }
                if residents.len()>=service.config.max_loaded_models {return Err(Error::resource("resident model limit reached").into());}
            }
            let record=service.registry.get(&id)?;
            if let Some(pair) = record.components.clone() {
                let language = service.registry.model_path(&pair.language.model)?;
                let vision = service.registry.model_path(&pair.vision.model)?;
                let budget = service.budget.clone();
                let limits = ResourceLimits { context, max_memory: service.config.max_memory };
                let model = tokio::task::spawn_blocking(move || -> Result<_> {
                    let model = coheryn_engine::multimodal::qwen2vl::Qwen2Vl::load(language, vision, limits, budget, &Default::default())?;
                    if model.model_identities() != (pair.language.sha256.as_str(), pair.vision.sha256.as_str()) {
                        return Err(Error::new(ErrorCode::ModelChanged, "model pair SHA-256 mismatch").into());
                    }
                    Ok(Resident::Vision(Arc::new(model)))
                }).await??;
                service.residents()?.insert(id.clone(), model);
                return Ok(json!({"model":id,"context":context,"loaded":true,"task":"image-to-text"}));
            }
            if let Some(manifest) = record.checkpoint.clone() {
                let root = service.registry.checkpoint_root(&id)?;
                let budget = service.budget.clone();
                let limits = ResourceLimits {context,max_memory:service.config.max_memory};
                let model = tokio::task::spawn_blocking(move || -> Result<_> {
                    let validation = budget.reserve(coheryn_engine::qwen35::RESIDENT_METADATA_BYTES)?;
                    manifest.verify(&root,true,&Default::default())?;
                    let inspection = manifest.bind(&root,&Default::default())?;
                    if inspection.descriptor().architecture.as_deref() != Some("qwen3_5") {
                        return Err(Error::new(ErrorCode::UnsupportedModelArchitecture,"managed checkpoint graph is not implemented").into());
                    }
                    drop(inspection); drop(validation);
                    let model = coheryn_engine::qwen35::Model::load_with_budget(root.join(&manifest.entrypoint), limits, budget, &Default::default())?;
                    Ok(Resident::Qwen35(Arc::new(model)))
                }).await??;
                service.residents()?.insert(id.clone(), model);
                return Ok(json!({"model":id,"context":context,"loaded":true,"task":"text-generation","adapter":manifest_adapter(&record)}));
            }
            let path=service.registry.model_path(&id)?;let budget=service.budget.clone();let max_memory=service.config.max_memory;
            let operation=service.budget.reserve(service.parser_limits().max_metadata_allocation)?;
            let max_concurrency=service.scheduler.max_concurrency();
            let model=tokio::task::spawn_blocking(move||->Result<Resident> {
                let _operation=operation;
                if coheryn_engine::storage::sha256_file(&path)?!=record.sha256 {return Err(Error::new(ErrorCode::ModelChanged,"managed model SHA-256 mismatch").into());}
                let descriptor=coheryn_engine::inspect(&path,Default::default())?;
                let report=coheryn_engine::validate(&descriptor);
                if !report.executable { return Err(report.diagnostics[0].clone().into()); }
                if let Some(config)=report.embedding_config {
                    let space=EmbeddingSpace::new(&record.sha256,&descriptor,&config)?;
                    drop(descriptor);
                    let model=coheryn_engine::embedding::EmbeddingModel::load_shared(path,ResourceLimits{context,max_memory},budget,max_concurrency)?;
                    Ok(Resident::Embedding(Arc::new(ResidentEmbedding {model,space})))
                } else {
                    drop(descriptor);
                    Ok(Resident::Generation(Arc::new(Model::load_with_budget(path,ResourceLimits{context,max_memory},budget)?)))
                }
            }).await??;
            service.residents()?.insert(id.clone(),model);
            Ok(json!({"model":id,"context":context,"loaded":true}))
        }).await?
    }
    pub fn parser_limits(&self) -> ParseLimits {
        let mut limits = ParseLimits::default();
        limits.max_metadata_allocation = limits
            .max_metadata_allocation
            .min(self.config.max_memory / 4);
        limits
    }
    pub async fn inspect(self: &Arc<Self>, id: String, deep: bool) -> Result<Value> {
        let _admin = self.admin.lock().await;
        let record = self.registry.get(&id)?;
        if let Some(manifest) = record.checkpoint.clone() {
            let root = self.registry.checkpoint_root(&id)?;
            let reservation = self
                .budget
                .reserve(coheryn_engine::qwen35::RESIDENT_METADATA_BYTES)?;
            return tokio::task::spawn_blocking(move || -> Result<Value> {
                let _reservation = reservation;
                if deep { manifest.verify(&root,true,&Default::default())?; }
                let inspection = manifest.bind(&root,&Default::default())?;
                if deep { inspection.deep_validate(&Default::default())?; }
                Ok(json!({"descriptor":inspection.descriptor(),"compatibility":{"executable":true,"adapter":manifest.adapter,"managed_text":true,"images":inspection.descriptor().text_adapter.as_ref().is_some_and(|a|a.images),"image_sessions":inspection.descriptor().text_adapter.as_ref().is_some_and(|a|a.images),"image_planning":inspection.descriptor().text_adapter.as_ref().is_some_and(|a|a.images),"tools":false,"sessions":true,"prefix_kv_reuse":true,"recurrent_state_reuse":true,"continuous_batching":true},"source_set_sha256":record.sha256,"payload_integrity":if deep {"verified"}else{"not-rehashed"}}))
            }).await?;
        }
        if record.components.is_some() {
            let service = self.clone();
            return tokio::task::spawn_blocking(move || {
                crate::vision_planner::inspect(&service, &record, deep)
            })
            .await?;
        }
        let path = self.registry.model_path(&id)?;
        let limits = self.parser_limits();
        let reservation = self.budget.reserve(limits.max_metadata_allocation)?;
        let embedding_limits = self.config.embeddings.clone();
        tokio::task::spawn_blocking(move || -> Result<Value> {
            let _reservation = reservation;
            let d = coheryn_engine::inspect(&path, limits)?;
            let mut report = coheryn_engine::validate(&d);
            if deep && report.executable {
                let result = coheryn_engine::validate_tokenizer(&d)
                    .and_then(|_| coheryn_engine::deep_validate(&path, &d, &Default::default()));
                if let Err(error) = result {
                    report.executable = false;
                    report.diagnostics.push(error);
                }
            }
            let space = report
                .embedding_config
                .as_ref()
                .filter(|_| report.executable)
                .map(|c| EmbeddingSpace::new(&record.sha256, &d, c))
                .transpose()?;
            Ok(json!({"tool_model_qualified":report.executable&&report.tool_protocol.is_some()&&coheryn_engine::tools::qualified_model_hash(&record.sha256),"descriptor":d,"compatibility":report,"embedding_space":space,"embedding_limits":embedding_limits}))
        })
        .await?
    }
    pub async fn estimate(self: &Arc<Self>, request: EstimateRequest) -> Result<Value> {
        use coheryn_engine::planner::Profile;
        Profile::parse(&request.profile)?;
        if let Some(images) = &request.images {
            images.validate()?;
        }
        let task = request.task.as_deref().unwrap_or("text-generation");
        if !matches!(task, "text-generation" | "embedding" | "image-to-text") {
            return Err(ApiError::bad("unsupported planning task"));
        }
        let service = self.clone();
        let _admin = self.admin.lock().await;
        let record = self.registry.get(&request.model)?;
        let loaded = self.residents()?.get(&request.model).cloned();
        if request.context == 0
            || request.context > self.config.max_context
            || loaded
                .as_ref()
                .is_some_and(|m| request.context > m.context())
        {
            return Err(ApiError::bad(
                "context exceeds model/server/loaded limit; reload explicitly to change capacity",
            ));
        }
        tokio::task::spawn_blocking(move || service.estimate_sync(request, loaded, record)).await?
    }
    fn estimate_sync(
        &self,
        request: EstimateRequest,
        loaded: Option<Resident>,
        record: ModelRecord,
    ) -> Result<Value> {
        use coheryn_engine::{
            embedding,
            planner::{self, PlanInput, Profile},
        };
        if let Some(manifest) = &record.checkpoint {
            if request.task.as_deref() == Some("image-to-text") {
                return crate::vision_planner::estimate_qwen35(
                    self,
                    request,
                    &record,
                    loaded.as_ref(),
                );
            }
            if request
                .task
                .as_deref()
                .is_some_and(|s| s != "text-generation")
                || request.images.is_some()
                || request.input_bytes.is_some()
                || request.batch_size.is_some()
                || request.max_input_tokens.is_some()
                || request.max_batch_tokens.is_some()
            {
                return Err(ApiError::bad(
                    "checkpoint currently supports text-generation planning only",
                ));
            }
            let hardware = crate::hardware::probe();
            let profile = Profile::parse(&request.profile)?;
            let plan_request = coheryn_engine::catalog::PlanRequest {
                context: request.context,
                concurrency: request.concurrency,
                max_memory: self.config.max_memory,
                profile,
                threads: request.threads,
                constraints: request.constraints,
            };
            let budget = self.budget.stats();
            let plan = if let Some(Resident::Qwen35(model)) = loaded {
                coheryn_engine::qwen35::estimate_shared(
                    model.configuration(),
                    model.weight_bytes()?,
                    &plan_request,
                    &hardware,
                    budget,
                    true,
                    self.scheduler.max_concurrency(),
                )?
            } else {
                let inspection_memory = self
                    .budget
                    .reserve(coheryn_engine::qwen35::RESIDENT_METADATA_BYTES)?;
                let inspection = manifest.bind(
                    &self.registry.checkpoint_root(&record.id)?,
                    &Default::default(),
                )?;
                let config = coheryn_engine::qwen35::Config::parse(
                    inspection
                        .source_configuration()
                        .ok_or_else(|| ApiError::bad("missing checkpoint configuration"))?,
                )?;
                let plan = coheryn_engine::qwen35::estimate_shared(
                    &config,
                    inspection.descriptor().tensor_bytes,
                    &plan_request,
                    &hardware,
                    budget,
                    false,
                    self.scheduler.max_concurrency(),
                )?;
                drop(inspection);
                drop(inspection_memory);
                plan
            };
            // Scheduling holds an additional bounded delivery reservation per request.
            let transport = (8u64 << 20)
                .checked_mul(request.concurrency as u64)
                .ok_or_else(|| Error::resource("transport reservation overflow"))?;
            let mut plan = plan;
            plan.model = request.model;
            plan.additional_reserved_bytes = plan
                .additional_reserved_bytes
                .checked_add(transport)
                .ok_or_else(|| Error::resource("checkpoint plan overflow"))?;
            plan.required_reserved_bytes = plan
                .required_reserved_bytes
                .checked_add(transport)
                .ok_or_else(|| Error::resource("checkpoint plan overflow"))?;
            plan.conservative_host_working_set_additional_bytes = plan
                .conservative_host_working_set_additional_bytes
                .checked_add(transport)
                .ok_or_else(|| Error::resource("checkpoint host estimate overflow"))?;
            plan.host_allocation_feasible = hardware
                .ram_available_bytes
                .map(|n| plan.additional_reserved_bytes <= n);
            plan.host_memory_feasible = hardware
                .ram_available_bytes
                .map(|n| plan.conservative_host_working_set_additional_bytes <= n);
            if plan.required_reserved_bytes > budget.limit
                || plan.host_allocation_feasible == Some(false)
            {
                plan.admissible = false;
                plan.reasons.retain(|r| !r.starts_with("request fits"));
                plan.reasons.push(
                    "Complete execution and transport reservations exceed available resources"
                        .into(),
                );
            }
            let mut value =
                serde_json::to_value(plan).map_err(|e| Error::request(e.to_string()))?;
            value["transport_reserved_bytes"] = json!(transport);
            value["source_set_sha256"] = json!(record.sha256);
            return Ok(value);
        }
        if record.components.is_some() {
            return crate::vision_planner::estimate(self, request, record, loaded.is_some());
        }
        if request.task.as_deref() == Some("image-to-text") {
            return Err(ApiError::bad(
                "image-to-text planning requires a registered model pair",
            ));
        }
        let service = self;
        let profile = Profile::parse(&request.profile)?;
        let (descriptor, inspection_bytes);
        let d = if let Some(model) = &loaded {
            inspection_bytes = 0;
            model
                .descriptor()
                .ok_or_else(|| ApiError::bad("selected resident has no GGUF descriptor"))?
        } else {
            let limits = service.parser_limits();
            inspection_bytes = limits.max_metadata_allocation;
            let _inspection = service.budget.reserve(inspection_bytes)?;
            descriptor =
                coheryn_engine::inspect(service.registry.model_path(&request.model)?, limits)?;
            &descriptor
        };
        let report = coheryn_engine::validate(d);
        if !report.executable {
            let mut e: ApiError = report.diagnostics[0].clone().into();
            e.details = Some(json!(report));
            return Err(e);
        }
        let hardware = crate::hardware::probe();
        if request.task.as_deref() == Some("embedding") {
            if request.images.is_some() || request.constraints {
                return Err(ApiError::bad(
                    "embedding plans do not accept images or generation constraints",
                ));
            }
            let c = report.embedding_config.ok_or_else(|| {
                ApiError::bad("model does not support qualified embedding inference")
            })?;
            if request.concurrency == 0
                || request.concurrency
                    > service
                        .config
                        .embeddings
                        .concurrency
                        .min(profile.concurrency_limit(service.scheduler.max_concurrency()))
            {
                return Err(ApiError::bad(
                    "embedding concurrency exceeds scheduler capacity",
                ));
            }
            let lengths = request.input_bytes.as_ref().ok_or_else(|| {
                ApiError::bad("embedding plans require input_bytes length descriptors")
            })?;
            if lengths.len() > service.config.embeddings.items
                || lengths
                    .iter()
                    .any(|n| *n > service.config.embeddings.input_bytes)
                || lengths
                    .iter()
                    .try_fold(0usize, |a, b| a.checked_add(*b))
                    .is_none_or(|n| n > service.config.embeddings.request_bytes)
            {
                return Err(crate::embeddings::batch_limit(
                    "embedding planning workload exceeds operator input limits",
                ));
            }
            let options = crate::embeddings::EmbeddingRequest {
                model: request.model.clone(),
                input: crate::embeddings::EmbeddingInput::Many(vec![]),
                encoding_format: Default::default(),
                dimensions: None,
                user: None,
                batch_size: request.batch_size,
                max_input_tokens: request.max_input_tokens,
                max_batch_tokens: request.max_batch_tokens,
                truncate: false,
                profile,
                threads: request.threads,
                max_memory: None,
            }
            .options(request.context, &service.config.embeddings)?;
            let metadata = if loaded.is_some() {
                0
            } else {
                embedding::estimate_model_bytes(d, &c)?
            };
            let serialization = crate::embeddings::response_allowance(
                lengths.len(),
                c.hidden,
                service.config.embeddings.response_bytes,
            )?;
            let ingress = crate::embeddings::parser_allowance(service.config.embeddings.body_bytes);
            let transport = serialization
                .checked_add(ingress)
                .ok_or_else(|| Error::resource("embedding transport estimate overflow"))?;
            let mut budget = service.budget.stats();
            budget.reserved = budget
                .reserved
                .checked_add(metadata)
                .ok_or_else(|| Error::resource("embedding planning overflow"))?;
            let original_limit = budget.limit;
            let share = budget.limit.saturating_sub(budget.reserved) / (request.concurrency as u64);
            budget.limit = budget
                .reserved
                .saturating_add(share.saturating_sub(transport));
            let plan = embedding::estimate(embedding::EmbeddingPlanInput {
                descriptor: d,
                config: &c,
                context: request.context,
                input_bytes: lengths,
                options: &options,
                budget,
                hardware: &hardware,
                max_concurrency: service.scheduler.max_concurrency(),
            })?;
            let additional = plan
                .additional_reserved_bytes
                .checked_add(transport)
                .and_then(|n| n.checked_mul(request.concurrency as u64))
                .and_then(|n| n.checked_add(metadata))
                .ok_or_else(|| Error::resource("embedding concurrency estimate overflow"))?;
            let required = service
                .budget
                .stats()
                .reserved
                .checked_add(additional)
                .ok_or_else(|| Error::resource("embedding estimate overflow"))?;
            let admissible = plan.admissible
                && required <= original_limit
                && hardware.ram_available_bytes.is_none_or(|n| additional <= n);
            Ok(
                json!({"schema_version":1,"task":"embedding","model":request.model,"embedding_space":EmbeddingSpace::new(&record.sha256,d,&c)?,"dimension":c.hidden,"context":request.context,"concurrency":request.concurrency,"backend":"cpu-reference","execution":plan,"model_metadata_additional":metadata,"ingress_reserved_bytes_per_request":ingress,"serialization_reserved_bytes_per_request":serialization,"additional_reserved_bytes":additional,"required_reserved_bytes":required,"admissible":admissible,"temporary_inspection_reserved_bytes":inspection_bytes,"note":"Length bounds only; actual tokens and aggregate token limits are checked before inference"}),
            )
        } else {
            if request.input_bytes.is_some()
                || request.batch_size.is_some()
                || request.max_input_tokens.is_some()
                || request.max_batch_tokens.is_some()
            {
                return Err(ApiError::bad(
                    "embedding workload fields require task=embedding",
                ));
            }
            let c = report
                .config
                .ok_or_else(|| ApiError::bad("embedding model requires task=embedding"))?;
            let plan = planner::estimate(PlanInput {
                constraints: request.constraints,
                descriptor: d,
                config: &c,
                model: &request.model,
                profile,
                context: request.context,
                concurrency: request.concurrency,
                max_concurrency: service.scheduler.max_concurrency(),
                loaded: loaded.is_some(),
                budget: service.budget.stats(),
                hardware,
                images: request.images.as_ref(),
                threads: request.threads,
            })?;
            let mut value =
                serde_json::to_value(plan).map_err(|e| Error::request(e.to_string()))?;
            value["temporary_inspection_reserved_bytes"] = json!(inspection_bytes);
            Ok(value)
        }
    }
    pub async fn ingest<S, E>(
        self: &Arc<Self>,
        name: String,
        expected: Option<String>,
        source: Option<String>,
        stream: S,
    ) -> Result<ModelRecord>
    where
        S: Stream<Item = std::result::Result<axum::body::Bytes, E>> + Send,
        E: std::fmt::Display,
    {
        self.registry.check_new(&name)?;
        if let Some(hash) = &expected {
            validate_hash(hash)?;
        }
        let used = self.registry.disk_bytes()?;
        let available = self
            .registry
            .disk_quota
            .saturating_sub(used)
            .min(self.registry.max_model_bytes)
            .min(fs2::available_space(self.registry.root())?.saturating_sub(64 << 20));
        let mut temporary = self.registry.temporary()?;
        let mut file =
            tokio::fs::File::from_std(temporary.file.take().expect("new temporary file"));
        let mut stream = Box::pin(stream);
        let mut hash = Sha256::new();
        let mut bytes = 0u64;
        let write_result=async {
            loop {
                let next=tokio::select! {_=self.shutdown.cancelled()=>return Err(ApiError::from(Error::new(ErrorCode::Cancelled,"server shutting down"))),next=tokio::time::timeout(Duration::from_secs(30),stream.next())=>next.map_err(|_|ApiError::bad("model acquisition idle timeout"))?};
                let Some(chunk)=next else{break};let chunk=chunk.map_err(|_|ApiError::bad("model stream interrupted"))?;
                bytes=bytes.checked_add(chunk.len() as u64).filter(|n|*n<=available).ok_or_else(||Error::resource("model exceeds file limit or disk quota"))?;
                file.write_all(&chunk).await?;hash.update(&chunk);
            }
            file.flush().await?;file.sync_all().await?;Ok::<(),ApiError>(())
        }.await;
        // Finish buffered filesystem writes before removing a failed upload on Windows.
        let flush = file.flush().await;
        drop(file);
        write_result?;
        flush?;
        let sha256 = coheryn_engine::storage::digest_hex(hash.finalize().into());
        if expected.is_some_and(|e| !e.eq_ignore_ascii_case(&sha256)) {
            return Err(ApiError::bad("model SHA-256 mismatch"));
        }
        let path = temporary.path.clone();
        let limits = self.parser_limits();
        let reservation = self.budget.reserve(limits.max_metadata_allocation)?;
        tokio::task::spawn_blocking(move || -> Result<()> {
            let _reservation = reservation;
            let d = coheryn_engine::inspect(path, limits)?;
            let report = coheryn_engine::validate(&d);
            if !report.executable {
                let architecture = d
                    .metadata
                    .get("general.architecture")
                    .and_then(|v| v.string());
                if architecture == Some("qwen2vl") {
                    coheryn_engine::multimodal::qwen2vl::language_configuration(&d)?;
                    coheryn_engine::multimodal::qwen2vl::validate_language_tokenizer(&d)?;
                    return Ok(());
                }
                if architecture == Some("clip") {
                    coheryn_engine::multimodal::vision::VisionConfig::from_descriptor(&d)?;
                    return Ok(());
                }
                let mut e: ApiError = report.diagnostics[0].clone().into();
                e.details = Some(json!(report));
                return Err(e);
            }
            coheryn_engine::validate_tokenizer(&d)?;
            Ok(())
        })
        .await??;
        let record = ModelRecord {
            components: None,
            checkpoint: None,
            schema_version: 1,
            id: name,
            sha256,
            bytes,
            source,
        };
        self.registry.commit(&mut temporary, &record)?;
        Ok(record)
    }
}

fn manifest_adapter(record: &ModelRecord) -> Option<&str> {
    record.checkpoint.as_ref().map(|m| m.adapter.as_str())
}

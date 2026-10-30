//! Bounded, owner-scoped authoritative histories. Request-local KV remains disposable.
mod images;
use crate::{
    error::{ApiError, Result},
    scheduler::GenerationEvent,
    session_store::{ExportDocument, PersistenceLimits, SavedTurn, Store},
    wire::GenerationRequest,
};
use coheryn_engine::{
    Cancellation, Error, ErrorCode, GenerationOptions, GenerationResult, Model,
    inference::PrefixCache,
    memory::{Budget, Reservation},
    sampling::Sampling,
    tokenizer::Message,
};
pub use images::{AssetLimits, Attachment, ExportImage, ImageAsset};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Serialize)]
pub struct CacheLimits {
    pub global_bytes: u64,
    pub session_bytes: u64,
    pub idle_seconds: u64,
}
impl Default for CacheLimits {
    fn default() -> Self {
        Self {
            global_bytes: 256 << 20,
            session_bytes: 64 << 20,
            idle_seconds: 300,
        }
    }
}
impl CacheLimits {
    pub fn validate(&self) -> coheryn_engine::Result<()> {
        if self.global_bytes > 1 << 30
            || self.session_bytes > 1 << 30
            || self.idle_seconds == 0
            || self.idle_seconds > 86400
        {
            return Err(Error::request("invalid idle cache limits"));
        }
        Ok(())
    }
}

#[derive(Clone, Serialize)]
pub struct SessionLimits {
    pub images: AssetLimits,
    pub total: usize,
    pub per_owner: usize,
    pub history_bytes: usize,
    pub messages: usize,
    pub turn_keys: usize,
    pub idle_seconds: u64,
}
impl Default for SessionLimits {
    fn default() -> Self {
        Self {
            images: AssetLimits::default(),
            total: 128,
            per_owner: 64,
            history_bytes: 1 << 20,
            messages: 256,
            turn_keys: 128,
            idle_seconds: 86400,
        }
    }
}
impl SessionLimits {
    pub fn validate(&self) -> coheryn_engine::Result<()> {
        self.images.validate()?;
        if self.total == 0
            || self.total > 128
            || self.per_owner == 0
            || self.per_owner > self.total
            || self.history_bytes == 0
            || self.history_bytes > 1 << 20
            || !(2..=256).contains(&self.messages)
            || self.turn_keys == 0
            || self.turn_keys > 128
            || self.idle_seconds == 0
            || self.idle_seconds > 2_592_000
        {
            return Err(Error::request("invalid session resource limits"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum OverflowPolicy {
    #[default]
    Reject,
    OldestCompleteTurns,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreateSession {
    #[serde(default)]
    pub task: Option<SessionTask>,
    pub model: String,
    #[serde(default)]
    pub persist: bool,
    #[serde(default, alias = "name")]
    pub alias: Option<String>,
    pub context: usize,
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub overflow: OverflowPolicy,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TurnSettings {
    #[serde(default = "default_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub top_k: usize,
    #[serde(default)]
    pub seed: u64,
    #[serde(default = "default_top_p", skip_serializing_if = "one_penalty")]
    pub repetition_penalty: f32,
    #[serde(default, skip_serializing_if = "zero_penalty")]
    pub frequency_penalty: f32,
    #[serde(default, skip_serializing_if = "zero_penalty")]
    pub presence_penalty: f32,
    #[serde(
        default,
        skip_serializing_if = "coheryn_engine::constraints::ResponseFormat::is_text"
    )]
    pub response_format: coheryn_engine::constraints::ResponseFormat,
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        deserialize_with = "crate::bounded::sequence::<_,_,16>"
    )]
    pub stop_tokens: Vec<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_seconds: Option<u64>,
    #[serde(default, deserialize_with = "crate::bounded::sequence::<_, _, 16>")]
    pub stop: Vec<String>,
    #[serde(default = "default_profile")]
    pub profile: String,
    #[serde(default)]
    pub threads: Option<usize>,
    #[serde(default)]
    pub max_memory: Option<u64>,
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        deserialize_with = "coheryn_engine::tools::bounded::<_,_,16>"
    )]
    pub tools: Vec<coheryn_engine::tools::Tool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<coheryn_engine::tools::ToolChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
}
fn default_tokens() -> usize {
    128
}
fn default_top_p() -> f32 {
    1.0
}
fn one_penalty(value: &f32) -> bool {
    *value == 1.
}
fn zero_penalty(value: &f32) -> bool {
    *value == 0.
}
fn default_profile() -> String {
    "balanced".into()
}
impl Default for TurnSettings {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            temperature: 0.,
            top_p: 1.,
            top_k: 0,
            seed: 0,
            repetition_penalty: 1.,
            frequency_penalty: 0.,
            presence_penalty: 0.,
            response_format: coheryn_engine::constraints::ResponseFormat::Text,
            stop_tokens: vec![],
            timeout_seconds: None,
            stop: vec![],
            profile: default_profile(),
            threads: None,
            max_memory: None,
            tools: vec![],
            tool_choice: None,
            parallel_tool_calls: None,
        }
    }
}
impl TurnSettings {
    pub fn tool_config(&self) -> coheryn_engine::tools::ToolConfig {
        coheryn_engine::tools::ToolConfig {
            tools: self.tools.clone(),
            tool_choice: self.tool_choice.clone(),
            parallel_tool_calls: self.parallel_tool_calls,
        }
    }

    pub(crate) fn request(
        &self,
        model: String,
        context: usize,
        prompt: String,
    ) -> GenerationRequest {
        let options = GenerationOptions {
            timeout_seconds: self.timeout_seconds,
            strict_tool: false,
            penalties: coheryn_engine::sampling::Penalties {
                repetition_penalty: self.repetition_penalty,
                frequency_penalty: self.frequency_penalty,
                presence_penalty: self.presence_penalty,
            },
            response_format: self.response_format.clone(),
            stop_tokens: self.stop_tokens.clone(),
            tool_markers: self.tool_config().enabled(),
            max_tokens: self.max_tokens,
            sampling: Sampling {
                temperature: self.temperature,
                top_p: self.top_p,
                top_k: self.top_k,
                seed: self.seed,
            },
            stop: self.stop.clone(),
        };
        let mut request = GenerationRequest::completion(model, prompt, &options, context, false);
        request.profile = self.profile.clone();
        request.threads = self.threads;
        request.max_memory = self.max_memory;
        request.tools = (!self.tools.is_empty()).then(|| self.tools.clone());
        request.tool_choice = self.tool_choice.clone();
        request.parallel_tool_calls = self.parallel_tool_calls;
        request
    }
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SubmitTurn {
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        deserialize_with = "crate::bounded::sequence::<_,_,4>"
    )]
    pub images: Vec<String>,
    pub expected_revision: u64,
    pub idempotency_key: String,
    #[serde(default)]
    pub content: String,
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        deserialize_with = "coheryn_engine::tools::bounded::<_,_,8>"
    )]
    pub tool_results: Vec<ToolResult>,
    #[serde(default)]
    pub settings: TurnSettings,
    #[serde(default)]
    pub stream: bool,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub content: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RevisionRequest {
    pub expected_revision: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResetRequest {
    pub expected_revision: u64,
    #[serde(default)]
    pub clear_instructions: bool,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistoryMessage {
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        deserialize_with = "crate::bounded::sequence::<_,_,4>"
    )]
    pub images: Vec<String>,
    pub id: String,
    pub turn_id: Option<String>,
    pub role: String,
    pub content: String,
    pub status: TurnStatus,
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        deserialize_with = "coheryn_engine::tools::bounded::<_,_,8>"
    )]
    pub tool_calls: Vec<coheryn_engine::tools::ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    pub created: u64,
}
#[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TurnStatus {
    Pending,
    Completed,
    Failed,
    Cancelled,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextAccounting {
    pub policy: OverflowPolicy,
    pub history_revision: u64,
    pub context: usize,
    pub prompt_tokens: usize,
    pub reserved_output_tokens: usize,
    #[serde(deserialize_with = "crate::bounded::sequence::<_, _, 256>")]
    pub included_message_ids: Vec<String>,
    #[serde(deserialize_with = "crate::bounded::sequence::<_, _, 256>")]
    pub excluded_message_ids: Vec<String>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TurnReceipt {
    pub session_id: String,
    pub turn_id: String,
    pub revision: u64,
    pub epoch: u64,
    pub status: TurnStatus,
    pub context: Option<ContextAccounting>,
    pub result: Option<GenerationResult>,
    pub error: Option<Error>,
}
#[derive(Clone)]
pub struct TurnWatch(Arc<Mutex<WatchState>>);
struct WatchState {
    receipt: TurnReceipt,
    _retained: Option<Reservation>,
}
impl TurnWatch {
    pub fn snapshot(&self) -> Result<TurnReceipt> {
        Ok(self.0.lock().map_err(|_| internal())?.receipt.clone())
    }
}
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SessionTask {
    #[default]
    TextGeneration,
    ImageToText,
}
impl SessionTask {
    fn is_text(&self) -> bool {
        *self == Self::TextGeneration
    }
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionView {
    #[serde(default, skip_serializing_if = "SessionTask::is_text")]
    pub task: SessionTask,
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        deserialize_with = "crate::bounded::sequence::<_,_,16>"
    )]
    pub image_assets: Vec<ImageAsset>,
    #[serde(default)]
    pub message_offset: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_offset: Option<usize>,
    #[serde(default)]
    pub storage_error: Option<String>,
    #[serde(default)]
    pub recovery_pending: bool,
    pub id: String,
    pub alias: Option<String>,
    pub model: String,
    pub model_sha256: String,
    pub context: usize,
    pub overflow: OverflowPolicy,
    pub revision: u64,
    pub epoch: u64,
    pub created: u64,
    pub updated: u64,
    pub idle_timeout_seconds: u64,
    pub active_turn: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        deserialize_with = "coheryn_engine::tools::bounded::<_,_,8>"
    )]
    pub pending_tool_calls: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_profile: Option<String>,
    pub message_count: usize,
    pub history_bytes: usize,
    pub persistent: bool,
    #[serde(default)]
    pub cached_tokens: usize,
    #[serde(default)]
    pub cache_reserved_bytes: u64,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::bounded::optional_sequence::<_, _, 256>"
    )]
    pub messages: Option<Vec<HistoryMessage>>,
}
struct TurnRecord {
    settings: TurnSettings,
    fingerprint: [u8; 32],
    watch: TurnWatch,
}
pub(crate) enum TextCache {
    Gguf(PrefixCache),
    Qwen35(coheryn_engine::qwen35::HybridCache),
}
impl TextCache {
    fn is_live(&self) -> bool {
        match self {
            Self::Gguf(c) => c.is_live(),
            Self::Qwen35(c) => c.is_live(),
        }
    }
    fn reserved_bytes(&self) -> u64 {
        match self {
            Self::Gguf(c) => c.reserved_bytes(),
            Self::Qwen35(c) => c.reserved_bytes(),
        }
    }
    fn token_count(&self) -> usize {
        match self {
            Self::Gguf(c) => c.token_count(),
            Self::Qwen35(c) => c.token_count(),
        }
    }
}
struct Session {
    loaded: bool,
    _history_reservation: Option<Reservation>,
    view: SessionView,
    owner: String,
    history: Vec<HistoryMessage>,
    keys: BTreeMap<String, TurnRecord>,
    touched: Instant,
    _metadata: Reservation,
    cache: Option<(TextCache, Instant)>,
}
impl Session {
    fn view(&self, history: bool) -> SessionView {
        let mut view = self.view.clone();
        view.cached_tokens = self.cache.as_ref().map_or(0, |(c, _)| c.token_count());
        view.cache_reserved_bytes = self.cache.as_ref().map_or(0, |(c, _)| c.reserved_bytes());
        view.messages = history.then(|| self.history.clone());
        view
    }
}
struct Inner {
    assets: Option<images::Assets>,
    state: Mutex<BTreeMap<String, Session>>,
    limits: SessionLimits,
    cache_limits: CacheLimits,
    persistence: PersistenceLimits,
    store: Option<Store>,
    budget: Budget,
}
#[derive(Clone)]
pub struct Sessions(Arc<Inner>);
pub enum BeginTurn {
    Replay(TurnWatch),
    New(Box<TurnLease>),
}
/// The scheduler owns this lease through worker acknowledgement, including disconnects.
pub struct TurnLease {
    task: SessionTask,
    pub(crate) cache: Option<TextCache>,
    sessions: Sessions,
    id: String,
    key: String,
    turn_id: String,
    model: String,
    model_sha256: String,
    context: usize,
    policy: OverflowPolicy,
    messages: Vec<HistoryMessage>,
    settings: TurnSettings,
    pub watch: TurnWatch,
    reserve: Option<Reservation>,
    input_bytes: usize,
    output_limit: usize,
    finished: bool,
}
fn internal() -> ApiError {
    Error::new(ErrorCode::Internal, "session state lock unavailable").into()
}
fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
fn opaque(prefix: &str) -> String {
    format!("{prefix}{}", uuid::Uuid::new_v4().simple())
}
pub fn idempotency_key() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}
fn history_message(
    role: &str,
    content: String,
    turn: Option<String>,
    status: TurnStatus,
) -> HistoryMessage {
    HistoryMessage {
        images: vec![],
        id: opaque("msg-"),
        role: role.into(),
        content,
        turn_id: turn,
        status,
        tool_calls: vec![],
        tool_call_id: None,
        created: now(),
    }
}
fn conflict(revision: u64, message: &str) -> ApiError {
    let mut error = ApiError::conflict(message);
    error.details = Some(serde_json::json!({"current_revision": revision}));
    error
}
fn identifier(value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err(ApiError::bad(
            "session aliases and idempotency keys need 1..=64 ASCII letters, digits, hyphens or underscores",
        ));
    }
    Ok(())
}
impl Sessions {
    pub fn new(limits: SessionLimits, budget: Budget) -> Self {
        Self::with_cache(limits, CacheLimits::default(), budget)
    }
    pub fn with_cache(limits: SessionLimits, cache_limits: CacheLimits, budget: Budget) -> Self {
        Self(Arc::new(Inner {
            assets: None,
            cache_limits,
            persistence: PersistenceLimits::default(),
            store: None,
            state: Mutex::new(BTreeMap::new()),
            limits,
            budget,
        }))
    }
    pub fn open(
        limits: SessionLimits,
        cache_limits: CacheLimits,
        persistence: PersistenceLimits,
        budget: Budget,
        root: &std::path::Path,
    ) -> Result<Self> {
        if !persistence.enabled {
            let mut sessions = Self::with_cache(limits, cache_limits, budget);
            let assets = images::Assets::new(root);
            assets.recover(&Default::default())?;
            Arc::get_mut(&mut sessions.0).expect("new sessions").assets = Some(assets);
            return Ok(sessions);
        }
        let store = Store::open(root, persistence.clone(), &budget)?;
        let index = store.index()?;
        if index.len() > limits.total {
            return Err(Error::resource("stored session count exceeds configured limit").into());
        }
        let mut state = BTreeMap::new();
        let mut owners: BTreeMap<String, usize> = BTreeMap::new();
        for (owner, mut view) in index {
            crate::registry::validate_hash(&owner)?;
            let count = owners.entry(owner.clone()).or_default();
            *count += 1;
            if *count > limits.per_owner {
                return Err(
                    Error::resource("stored owner session count exceeds configured limit").into(),
                );
            }
            if !view.persistent || view.messages.is_some() {
                return Err(ApiError::bad("invalid persistent session index"));
            }
            if let Some(alias) = &view.alias {
                identifier(alias)?;
            }
            identifier(&view.id)?;
            crate::registry::validate_id(&view.model)?;
            crate::registry::validate_hash(&view.model_sha256)?;
            if view.history_bytes > limits.history_bytes
                || view.message_count > limits.messages
                || view.context == 0
                || view.context > 1_048_576
            {
                return Err(ApiError::bad("stored session exceeds configured limits"));
            }
            if view.image_assets.len() > limits.images.session_items {
                return Err(ApiError::bad("stored image count exceeds quota"));
            }
            for asset in &view.image_assets {
                asset.validate()?;
            }
            if view.image_assets.iter().map(|a| a.bytes).sum::<u64>() > limits.images.session_bytes
            {
                return Err(ApiError::bad("stored image bytes exceed quota"));
            }
            view.messages = None;
            view.cached_tokens = 0;
            view.cache_reserved_bytes = 0;
            view.recovery_pending = view.active_turn.is_some();
            view.active_turn = None;
            let metadata = budget.reserve(16384)?;
            state.insert(
                view.id.clone(),
                Session {
                    view,
                    owner,
                    history: vec![],
                    keys: BTreeMap::new(),
                    touched: Instant::now(),
                    _metadata: metadata,
                    cache: None,
                    loaded: false,
                    _history_reservation: None,
                },
            );
        }
        let assets = images::Assets::new(root);
        let live = state
            .iter()
            .map(|(id, s)| {
                (
                    id.clone(),
                    s.view
                        .image_assets
                        .iter()
                        .map(|a| format!("{}.bin", a.id))
                        .collect(),
                )
            })
            .collect();
        assets.recover(&live)?;
        Ok(Self(Arc::new(Inner {
            assets: Some(assets),
            state: Mutex::new(state),
            limits,
            cache_limits,
            persistence,
            store: Some(store),
            budget,
        })))
    }
    pub fn persistence_enabled(&self) -> bool {
        self.0.store.is_some()
    }
    fn document(
        session: &Session,
        replacement: Option<(&str, &TurnReceipt)>,
    ) -> Result<ExportDocument> {
        let mut turns = Vec::new();
        for (key, record) in &session.keys {
            let receipt = if let Some((selected, receipt)) = replacement
                && selected == key
            {
                receipt.clone()
            } else {
                record.watch.snapshot()?
            };
            turns.push(SavedTurn {
                key: key.clone(),
                fingerprint: record.fingerprint,
                settings: record.settings.clone(),
                receipt,
            });
        }
        turns.sort_by_key(|t| t.receipt.revision);
        let mut view = session.view(false);
        view.messages = None;
        view.message_offset = 0;
        view.next_offset = None;
        view.cached_tokens = 0;
        view.cache_reserved_bytes = 0;
        Ok(ExportDocument {
            assets: vec![],
            version: if session.view.task == SessionTask::ImageToText {
                3
            } else if session.view.tool_profile.is_some()
                || session
                    .history
                    .iter()
                    .any(|m| !m.tool_calls.is_empty() || m.role == "tool")
                || session.keys.values().any(|r| {
                    !r.settings.tools.is_empty()
                        || r.settings.tool_choice.is_some()
                        || r.settings.parallel_tool_calls.is_some()
                })
            {
                2
            } else {
                1
            },
            session: view,
            messages: session.history.clone(),
            turns,
        })
    }
    fn persist(&self, session: &Session, replacement: Option<(&str, &TurnReceipt)>) -> Result<()> {
        if session.view.persistent {
            self.0
                .store
                .as_ref()
                .ok_or_else(|| ApiError::bad("persistence is disabled"))?
                .save(&session.owner, &Self::document(session, replacement)?)?;
        }
        Ok(())
    }
    fn hydrate(&self, state: &mut BTreeMap<String, Session>, id: &str) -> Result<()> {
        if state.get(id).is_some_and(|s| s.loaded) {
            return Ok(());
        }
        let store = self.0.store.as_ref().ok_or_else(internal)?;
        let required = store
            .bytes(id)?
            .checked_mul(8)
            .and_then(|n| n.checked_add(65536))
            .ok_or_else(|| Error::resource("history reservation overflow"))?;
        if self
            .0
            .budget
            .stats()
            .limit
            .saturating_sub(self.0.budget.stats().reserved)
            < required
        {
            for (other, s) in state.iter_mut() {
                if other != id
                    && s.view.persistent
                    && s.view.active_turn.is_none()
                    && s.view.storage_error.is_none()
                {
                    s.history.clear();
                    s.keys.clear();
                    s.cache = None;
                    s._history_reservation = None;
                    s.loaded = false;
                }
            }
        }
        let reservation = self.0.budget.reserve(required)?;
        let mut doc = store.load(id)?;
        validate_document(&doc, &self.0.limits, true)?;
        if doc.session.active_turn.is_some() {
            for m in &mut doc.messages {
                if m.status == TurnStatus::Pending {
                    m.status = TurnStatus::Cancelled;
                }
            }
            doc.session.active_turn = None;
            doc.session.recovery_pending = false;
            doc.session.revision += 1;
            doc.session.updated = now();
            for turn in &mut doc.turns {
                if turn.receipt.status == TurnStatus::Pending {
                    turn.receipt.status = TurnStatus::Cancelled;
                    turn.receipt.revision = doc.session.revision;
                    turn.receipt.error = Some(Error::new(
                        ErrorCode::Cancelled,
                        "turn interrupted by a previous process shutdown",
                    ));
                    turn.receipt.result = None;
                }
            }
            store.save(&state.get(id).ok_or_else(internal)?.owner, &doc)?;
        }
        let session = state.get_mut(id).ok_or_else(internal)?;
        session.view = doc.session;
        session.view.messages = None;
        session.history = doc.messages;
        session.keys = doc
            .turns
            .into_iter()
            .map(|turn| {
                (
                    turn.key,
                    TurnRecord {
                        settings: turn.settings,
                        fingerprint: turn.fingerprint,
                        watch: TurnWatch(Arc::new(Mutex::new(WatchState {
                            receipt: turn.receipt,
                            _retained: None,
                        }))),
                    },
                )
            })
            .collect();
        session._history_reservation = Some(reservation);
        session.loaded = true;
        Ok(())
    }
    fn state(&self) -> Result<std::sync::MutexGuard<'_, BTreeMap<String, Session>>> {
        let mut state = self.0.state.lock().map_err(|_| internal())?;
        let timeout = Duration::from_secs(self.0.limits.idle_seconds);
        let mut expired = Vec::new();
        for (id, session) in state.iter_mut() {
            if session.view.active_turn.is_none() && session.touched.elapsed() >= timeout {
                if session.view.persistent {
                    if session.view.storage_error.is_none() {
                        session.history.clear();
                        session.keys.clear();
                        session.cache = None;
                        session._history_reservation = None;
                        session.loaded = false;
                    }
                } else {
                    expired.push(id.clone());
                }
            }
            if session.view.persistent
                && session.view.active_turn.is_none()
                && now().saturating_sub(session.view.updated)
                    >= self.0.persistence.retention_seconds
            {
                if let Some(store) = &self.0.store {
                    store.delete(id, &session.owner)?;
                }
                expired.push(id.clone());
            }
        }
        for id in expired {
            state.remove(&id);
            if let Some(assets) = &self.0.assets {
                assets.remove_session(&id)?;
            }
        }

        self.trim_caches(&mut state);
        Ok(state)
    }
    fn trim_caches(&self, state: &mut BTreeMap<String, Session>) {
        let cap = self
            .0
            .cache_limits
            .global_bytes
            .min(self.0.budget.stats().limit / 4);
        for session in state.values_mut() {
            if session.cache.as_ref().is_some_and(|(c, t)| {
                !c.is_live()
                    || c.reserved_bytes() > self.0.cache_limits.session_bytes
                    || t.elapsed().as_secs() >= self.0.cache_limits.idle_seconds
            }) {
                session.cache = None;
            }
        }
        while state
            .values()
            .filter_map(|s| s.cache.as_ref())
            .map(|(c, _)| c.reserved_bytes())
            .sum::<u64>()
            > cap
        {
            let oldest = state
                .iter()
                .filter_map(|(id, s)| s.cache.as_ref().map(|(_, t)| (id.clone(), *t)))
                .min_by_key(|(_, t)| *t)
                .map(|(id, _)| id);
            if let Some(id) = oldest {
                state.get_mut(&id).expect("cache owner").cache = None;
            } else {
                break;
            }
        }
    }
    fn cache_state(&self) -> Result<std::sync::MutexGuard<'_, BTreeMap<String, Session>>> {
        let mut state = self.0.state.lock().map_err(|_| internal())?;
        self.trim_caches(&mut state);
        Ok(state)
    }
    pub fn evict_idle(&self) -> Result<()> {
        for s in self.cache_state()?.values_mut() {
            s.cache = None;
            if s.view.persistent && s.view.active_turn.is_none() && s.view.storage_error.is_none() {
                s.history.clear();
                s.keys.clear();
                s._history_reservation = None;
                s.loaded = false;
            }
        }
        Ok(())
    }
    pub fn evict_model(&self, model: &str) -> Result<()> {
        for s in self
            .cache_state()?
            .values_mut()
            .filter(|s| s.view.model == model)
        {
            s.cache = None;
        }
        Ok(())
    }
    pub(crate) fn cache_limit(&self) -> u64 {
        self.0
            .cache_limits
            .session_bytes
            .min(self.0.cache_limits.global_bytes)
            .min(self.0.budget.stats().limit / 4)
    }
    pub fn cache_stats(&self) -> Result<serde_json::Value> {
        let state = self.cache_state()?;
        Ok(
            serde_json::json!({"idle_caches":state.values().filter(|s|s.cache.is_some()).count(), "reserved_bytes":state.values().filter_map(|s|s.cache.as_ref()).map(|(c,_)|c.reserved_bytes()).sum::<u64>(), "limits":self.0.cache_limits}),
        )
    }
    fn resolve(state: &BTreeMap<String, Session>, owner: &str, id: &str) -> Result<String> {
        identifier(id)?;
        state
            .iter()
            .find_map(|(key, s)| {
                (s.owner == owner && (key == id || s.view.alias.as_deref() == Some(id)))
                    .then(|| key.clone())
            })
            .ok_or_else(|| ApiError::missing("session not found or expired"))
    }
    pub fn create(
        &self,
        owner: String,
        request: CreateSession,
        hash: String,
    ) -> Result<SessionView> {
        let task = request.task.unwrap_or(SessionTask::TextGeneration);
        self.create_task(owner, request, hash, task)
    }
    pub fn create_task(
        &self,
        owner: String,
        request: CreateSession,
        hash: String,
        task: SessionTask,
    ) -> Result<SessionView> {
        if request.task.is_some_and(|selected| selected != task) {
            return Err(ApiError::bad(
                "session task differs from validated model capability",
            ));
        }
        if task == SessionTask::ImageToText && request.context > 4096 {
            return Err(ApiError::bad("managed image context exceeds 4096"));
        }
        if request.persist && self.0.store.is_none() {
            return Err(ApiError::bad(
                "persistent histories require serve --session-persistence",
            ));
        }
        if let Some(alias) = &request.alias {
            identifier(alias)?;
        }
        if request
            .system
            .as_ref()
            .is_some_and(|s| s.len() > self.0.limits.history_bytes)
        {
            return Err(Error::resource("session instructions exceed history byte limit").into());
        }
        let mut state = self.state()?;
        if state.len() >= self.0.limits.total
            || state.values().filter(|s| s.owner == owner).count() >= self.0.limits.per_owner
        {
            return Err(Error::resource("session count limit reached").into());
        }
        if let Some(alias) = &request.alias
            && state.values().any(|s| {
                s.owner == owner && (s.view.alias.as_ref() == Some(alias) || &s.view.id == alias)
            })
        {
            return Err(ApiError::conflict(
                "session alias already exists; resume explicitly",
            ));
        }
        let bytes = request.system.as_ref().map_or(0, String::len);
        let metadata = self.0.budget.reserve(16384)?;
        let history_reservation = self.0.budget.reserve(bytes as u64 * 8)?;
        let history: Vec<_> = request
            .system
            .into_iter()
            .map(|text| history_message("system", text, None, TurnStatus::Completed))
            .collect();
        let view = SessionView {
            task,
            image_assets: vec![],
            message_offset: 0,
            next_offset: None,
            id: opaque("ses-"),
            alias: request.alias,
            model: request.model,
            model_sha256: hash,
            context: request.context,
            overflow: request.overflow,
            revision: 0,
            epoch: 0,
            created: now(),
            updated: now(),
            idle_timeout_seconds: self.0.limits.idle_seconds,
            active_turn: None,
            pending_tool_calls: vec![],
            tool_profile: None,
            message_count: history.len(),
            history_bytes: bytes,
            persistent: request.persist,
            storage_error: None,
            recovery_pending: false,
            cached_tokens: 0,
            cache_reserved_bytes: 0,
            messages: None,
        };
        let result = SessionView {
            messages: Some(history.clone()),
            ..view.clone()
        };
        state.insert(
            view.id.clone(),
            Session {
                loaded: true,
                _history_reservation: Some(history_reservation),
                view,
                owner,
                history,
                keys: BTreeMap::new(),
                touched: Instant::now(),
                _metadata: metadata,
                cache: None,
            },
        );
        if let Err(error) = self.persist(state.get(&result.id).expect("created session"), None) {
            state.remove(&result.id);
            return Err(error);
        }
        Ok(result)
    }
    pub fn get(&self, owner: &str, id: &str) -> Result<SessionView> {
        let mut state = self.state()?;
        let id = Self::resolve(&state, owner, id)?;
        self.hydrate(&mut state, &id)?;
        let session = state.get_mut(&id).expect("resolved session");
        session.touched = Instant::now();
        Ok(session.view(true))
    }
    pub fn get_page(
        &self,
        owner: &str,
        id: &str,
        offset: usize,
        limit: usize,
    ) -> Result<SessionView> {
        if limit == 0 || limit > 100 {
            return Err(ApiError::bad("history page limit must be in 1..=100"));
        }
        let mut state = self.state()?;
        let id = Self::resolve(&state, owner, id)?;
        if state.get(&id).is_some_and(|s| s.view.recovery_pending) {
            self.hydrate(&mut state, &id)?;
        }
        let session = state.get_mut(&id).ok_or_else(internal)?;
        if offset > session.view.message_count {
            return Err(ApiError::bad("history offset exceeds message count"));
        }
        let mut view = session.view(false);
        view.message_offset = offset;
        let messages = if session.loaded {
            session
                .history
                .iter()
                .skip(offset)
                .take(limit)
                .cloned()
                .collect()
        } else {
            self.0
                .store
                .as_ref()
                .ok_or_else(internal)?
                .page(&id, offset, limit)?
        };
        let end = offset + messages.len();
        view.next_offset = (end < session.view.message_count).then_some(end);
        view.messages = Some(messages);
        session.touched = Instant::now();
        Ok(view)
    }
    pub fn list(&self, owner: &str) -> Result<Vec<SessionView>> {
        Ok(self
            .state()?
            .values()
            .filter(|s| s.owner == owner)
            .map(|s| s.view(false))
            .collect())
    }
    pub fn reset(&self, owner: &str, id: &str, revision: u64) -> Result<SessionView> {
        self.reset_with_instructions(owner, id, revision, false)
    }
    pub fn reset_with_instructions(
        &self,
        owner: &str,
        id: &str,
        revision: u64,
        clear_instructions: bool,
    ) -> Result<SessionView> {
        let mut state = self.state()?;
        let id = Self::resolve(&state, owner, id)?;
        self.hydrate(&mut state, &id)?;
        let session = state.get_mut(&id).expect("resolved session");
        Self::mutable(session, revision)?;
        let old_view = session.view.clone();
        let old_history = session.history.clone();
        session.view.image_assets.clear();
        let old_keys = std::mem::take(&mut session.keys);
        session
            .history
            .retain(|m| m.role == "system" && !clear_instructions);
        session.keys.clear();
        session.cache = None;
        session.view.revision += 1;
        session.view.epoch += 1;
        session.view.pending_tool_calls.clear();
        session.view.tool_profile = None;
        session.view.message_count = session.history.len();
        session.view.history_bytes = session.history.iter().map(|m| m.content.len()).sum();
        session.view.updated = now();
        session.touched = Instant::now();
        session.view.storage_error = None;
        let preserved = self.0.budget.reserve(session.view.history_bytes as u64 * 8);
        let preserved = match preserved {
            Ok(r) => r,
            Err(error) => {
                session.view = old_view;
                session.history = old_history;
                session.keys = old_keys;
                return Err(error.into());
            }
        };
        if let Err(error) = self.persist(session, None) {
            session.view = old_view;
            session.history = old_history;
            session.keys = old_keys;
            return Err(error);
        }
        session._history_reservation = Some(preserved);
        if let Some(assets) = &self.0.assets {
            assets.remove_session(&id)?;
        }
        Ok(session.view(true))
    }
    pub fn delete(&self, owner: &str, id: &str, revision: u64) -> Result<()> {
        let mut state = self.state()?;
        let id = Self::resolve(&state, owner, id)?;
        self.hydrate(&mut state, &id)?;
        Self::mutable(state.get(&id).expect("resolved session"), revision)?;
        if let Some(session) = state.get(&id)
            && session.view.persistent
        {
            self.0
                .store
                .as_ref()
                .ok_or_else(internal)?
                .delete(&id, owner)?;
        }
        state.remove(&id);
        if let Some(assets) = &self.0.assets {
            assets.remove_session(&id)?;
        }
        Ok(())
    }
    fn mutable(session: &Session, revision: u64) -> Result<()> {
        if session.view.active_turn.is_some() || session.view.revision != revision {
            return Err(conflict(
                session.view.revision,
                "session revision is stale or a turn is active",
            ));
        }
        Ok(())
    }
    pub fn save(&self, owner: &str, id: &str, revision: u64) -> Result<SessionView> {
        if self.0.store.is_none() {
            return Err(ApiError::bad("persistent histories are disabled"));
        }
        let mut state = self.state()?;
        let id = Self::resolve(&state, owner, id)?;
        self.hydrate(&mut state, &id)?;
        let session = state.get_mut(&id).ok_or_else(internal)?;
        Self::mutable(session, revision)?;
        let old = session.view.clone();
        session.view.persistent = true;
        session.view.revision += 1;
        session.view.updated = now();
        if let Err(error) = self.persist(session, None) {
            session.view = old;
            return Err(error);
        }
        Ok(session.view(true))
    }
    pub fn export(&self, owner: &str, id: &str) -> Result<ExportDocument> {
        let mut state = self.state()?;
        let id = Self::resolve(&state, owner, id)?;
        self.hydrate(&mut state, &id)?;
        let session = state.get(&id).ok_or_else(internal)?;
        if session.view.active_turn.is_some() {
            return Err(ApiError::conflict("cannot export an active turn"));
        }
        let mut doc = Self::document(session, None)?;
        let bytes = serde_json::to_vec(&doc).map_err(|_| internal())?.len();
        doc.assets = self.exported_assets(&session.view, bytes)?;
        Ok(doc)
    }
    pub fn import(
        &self,
        owner: String,
        mut doc: ExportDocument,
        alias: Option<String>,
        persist: bool,
    ) -> Result<SessionView> {
        validate_document(&doc, &self.0.limits, false)?;
        if persist && self.0.store.is_none() {
            return Err(ApiError::bad(
                "persistent imports require serve --session-persistence",
            ));
        }
        let mut state = self.state()?;
        if state.len() >= self.0.limits.total
            || state.values().filter(|s| s.owner == owner).count() >= self.0.limits.per_owner
        {
            return Err(Error::resource("session count limit reached").into());
        }
        let alias = alias.or(doc.session.alias.take());
        if let Some(alias) = &alias {
            identifier(alias)?;
        }
        if alias.as_ref().is_some_and(|a| {
            state
                .values()
                .any(|s| s.owner == owner && (s.view.alias.as_ref() == Some(a) || &s.view.id == a))
        }) {
            return Err(ApiError::conflict("import alias already exists"));
        }
        let bytes = serde_json::to_vec(&doc)
            .map_err(|_| ApiError::bad("invalid imported history"))?
            .len();
        if bytes > crate::session_store::DOCUMENT_BYTES {
            return Err(Error::resource("imported history exceeds byte quota").into());
        }
        // Portable Base64 is an ingress buffer. Only metadata/history/replay live in RAM.
        let retained = serde_json::to_vec(&(&doc.session, &doc.messages, &doc.turns))
            .map_err(|_| internal())?
            .len();
        let history_reservation = self.0.budget.reserve(retained as u64 * 8 + 65536)?;
        let metadata = self.0.budget.reserve(16384)?;
        let id = opaque("ses-");
        doc.session.id = id.clone();
        doc.session.alias = alias;
        doc.session.persistent = persist;
        doc.session.updated = now();
        doc.session.created = now();
        doc.session.messages = None;
        doc.session.cached_tokens = 0;
        doc.session.cache_reserved_bytes = 0;
        doc.session.storage_error = None;
        doc.session.recovery_pending = false;
        for turn in &mut doc.turns {
            turn.receipt.session_id = id.clone();
        }
        self.import_assets(&state, &owner, &doc)?;
        let keys = doc
            .turns
            .into_iter()
            .map(|t| {
                (
                    t.key,
                    TurnRecord {
                        settings: t.settings,
                        fingerprint: t.fingerprint,
                        watch: TurnWatch(Arc::new(Mutex::new(WatchState {
                            receipt: t.receipt,
                            _retained: None,
                        }))),
                    },
                )
            })
            .collect();
        let session = Session {
            view: doc.session,
            owner,
            history: doc.messages,
            keys,
            touched: Instant::now(),
            _metadata: metadata,
            cache: None,
            loaded: true,
            _history_reservation: Some(history_reservation),
        };
        if let Err(error) = self.persist(&session, None) {
            if let Some(assets) = &self.0.assets {
                let _ = assets.remove_session(&session.view.id);
            }
            return Err(error);
        }
        let view = session.view(true);
        state.insert(id, session);
        Ok(view)
    }
    pub fn begin(&self, owner: &str, id: &str, mut request: SubmitTurn) -> Result<BeginTurn> {
        identifier(&request.idempotency_key)?;
        if (request.content.is_empty() && request.images.is_empty())
            == request.tool_results.is_empty()
            || (!request.images.is_empty() && !request.tool_results.is_empty())
            || request.content.len() > self.0.limits.history_bytes
        {
            return Err(ApiError::bad(
                "turn content is empty or exceeds history byte limit",
            ));
        }
        if self
            .0
            .budget
            .stats()
            .limit
            .saturating_sub(self.0.budget.stats().reserved)
            < (16 << 20)
        {
            self.evict_idle()?;
        }
        let fingerprint: [u8; 32] =
            Sha256::digest(serde_json::to_vec(&request).map_err(|_| internal())?).into();
        let mut state = self.state()?;
        let id = Self::resolve(&state, owner, id)?;
        self.hydrate(&mut state, &id)?;
        let session = state.get_mut(&id).expect("resolved session");
        if let Some(record) = session.keys.get(&request.idempotency_key) {
            if fingerprint != record.fingerprint {
                return Err(conflict(
                    session.view.revision,
                    "idempotency key was used with different content or settings",
                ));
            }
            if record.watch.snapshot()?.status == TurnStatus::Pending {
                return Err(conflict(
                    session.view.revision,
                    "this idempotent turn is still in progress",
                ));
            }
            return Ok(BeginTurn::Replay(record.watch.clone()));
        }
        Self::mutable(session, request.expected_revision)?;
        if session.view.storage_error.is_some() {
            return Err(Error::new(
                ErrorCode::Unavailable,
                "persistent session needs recovery/reset before another turn",
            )
            .into());
        }
        if !request.images.is_empty() && session.view.task != SessionTask::ImageToText {
            return Err(ApiError::bad(
                "image attachments require an image conversation",
            ));
        }
        let mut unique = std::collections::BTreeSet::new();
        for id in &request.images {
            if !unique.insert(id) || !session.view.image_assets.iter().any(|a| &a.id == id) {
                return Err(ApiError::bad(
                    "image reference is duplicate or does not belong to this session",
                ));
            }
        }
        let old_view = session.view.clone();
        let mut inputs = Vec::new();
        if !session.view.pending_tool_calls.is_empty() {
            let mut ids = std::collections::BTreeSet::new();
            for r in &request.tool_results {
                coheryn_engine::tools::identifier(&r.tool_call_id)?;
                if !ids.insert(r.tool_call_id.clone())
                    || !session.view.pending_tool_calls.contains(&r.tool_call_id)
                {
                    return Err(ApiError::bad(
                        "tool results must match outstanding calls exactly",
                    ));
                }
                let mut m = history_message("tool", r.content.clone(), None, TurnStatus::Pending);
                m.tool_call_id = Some(r.tool_call_id.clone());
                inputs.push(m);
            }
            if ids.len() != session.view.pending_tool_calls.len() || !request.content.is_empty() {
                return Err(ApiError::bad(
                    "supply all pending tool results before a new user turn",
                ));
            }
            if request
                .tool_results
                .iter()
                .map(|r| &r.tool_call_id)
                .ne(session.view.pending_tool_calls.iter())
            {
                return Err(ApiError::bad(
                    "Qwen tool results must follow the outstanding call order",
                ));
            }
            if request.settings.tools.is_empty()
                && let Some(previous) = session
                    .keys
                    .values()
                    .filter(|r| {
                        r.watch.snapshot().ok().is_some_and(|r| {
                            r.result.as_ref().is_some_and(|r| !r.tool_calls.is_empty())
                        })
                    })
                    .max_by_key(|r| r.watch.snapshot().map_or(0, |r| r.revision))
            {
                request.settings.tools = previous.settings.tools.clone();
                request.settings.parallel_tool_calls = request
                    .settings
                    .parallel_tool_calls
                    .or(previous.settings.parallel_tool_calls);
            }
        } else {
            if !request.tool_results.is_empty() {
                return Err(ApiError::bad("session has no pending tool calls"));
            }
            inputs.push(history_message(
                "user",
                request.content.clone(),
                None,
                TurnStatus::Pending,
            ));
        }
        let mut generation = request.settings.request(
            session.view.model.clone(),
            session.view.context,
            "validation".into(),
        );
        generation.stream = false;
        // Request validation for tool conditioning requires message mode.
        generation.prompt = None;
        generation.messages = Some(vec![Message {
            role: "user".into(),
            content: "validation".into(),
            ..Default::default()
        }]);
        generation.validate()?;
        if request.settings.max_tokens > session.view.context {
            return Err(ApiError::bad("output reserve exceeds session context"));
        }
        let profile = if generation.tool_config().enabled() || !request.tool_results.is_empty() {
            Some(coheryn_engine::tools::profile_identity())
        } else {
            None
        };
        if let (Some(expected), Some(current)) = (&session.view.tool_profile, &profile)
            && expected != current
        {
            return Err(Error::new(
                ErrorCode::UnsupportedFeature,
                "stored tool-template identity is unsupported by this runtime",
            )
            .into());
        }
        if let Some(input) = inputs.last_mut() {
            input.images = request.images.clone();
        }
        if session.view.task == SessionTask::ImageToText {
            crate::vision::validate_controls(&generation)?;
        }
        let input_bytes = inputs.iter().map(message_bytes).sum::<usize>();
        if session.history.len() + inputs.len() + 1 > self.0.limits.messages
            || session.keys.len() >= self.0.limits.turn_keys
            || session.view.history_bytes.saturating_add(input_bytes) >= self.0.limits.history_bytes
        {
            return Err(Error::resource(
                "session history or idempotency quota reached; reset/delete explicitly",
            )
            .into());
        }
        // Holds all bounded history snapshots, output/token IDs and commit/replay overlap.
        let declaration_reserve = if request.settings.tools.is_empty() {
            0
        } else {
            coheryn_engine::tools::DECLARATION_RESERVATION
        };
        let held_bytes = declaration_reserve
            + 65536
            + 8 * (self.0.limits.history_bytes as u64
                + input_bytes as u64
                + request.settings.max_tokens as u64 * 4
                + serde_json::to_vec(&request.settings)
                    .map_err(|_| internal())?
                    .len() as u64);
        if request
            .settings
            .max_memory
            .is_some_and(|cap| held_bytes > cap)
        {
            return Err(
                Error::resource("managed turn working copies exceed request max_memory").into(),
            );
        }
        let reserve = self.0.budget.reserve(held_bytes)?;
        if profile.is_some() {
            session.view.tool_profile = profile;
        }
        let turn_id = opaque("turn-");
        for input in &mut inputs {
            input.turn_id = Some(turn_id.clone());
        }
        let mut messages = session.history.clone();
        messages.extend(inputs.iter().cloned());
        session.history.extend(inputs);
        session.view.history_bytes += input_bytes;
        session.view.message_count = session.history.len();
        session.view.active_turn = Some(turn_id.clone());
        session.view.revision += 1;
        session.view.updated = now();
        session.touched = Instant::now();
        let watch = TurnWatch(Arc::new(Mutex::new(WatchState {
            receipt: TurnReceipt {
                session_id: id.clone(),
                turn_id: turn_id.clone(),
                revision: session.view.revision,
                epoch: session.view.epoch,
                status: TurnStatus::Pending,
                context: None,
                result: None,
                error: None,
            },
            _retained: None,
        })));
        session.keys.insert(
            request.idempotency_key.clone(),
            TurnRecord {
                settings: request.settings.clone(),
                fingerprint,
                watch: watch.clone(),
            },
        );
        if let Err(error) = self.persist(session, None) {
            let count = old_view.message_count;
            session.view = old_view;
            session.history.truncate(count);
            session.keys.remove(&request.idempotency_key);
            return Err(error);
        }
        let cache = session.cache.take().map(|(cache, _)| cache);
        Ok(BeginTurn::New(Box::new(TurnLease {
            task: session.view.task,
            cache,
            sessions: self.clone(),
            id,
            key: request.idempotency_key,
            turn_id,
            model: session.view.model.clone(),
            model_sha256: session.view.model_sha256.clone(),
            context: session.view.context,
            policy: session.view.overflow,
            messages,
            settings: request.settings,
            watch,
            reserve: Some(reserve),
            input_bytes,
            output_limit: self.0.limits.history_bytes - session.view.history_bytes,
            finished: false,
        })))
    }
}
impl TurnLease {
    pub fn model_id(&self) -> &str {
        &self.model
    }
    pub fn model_hash(&self) -> &str {
        &self.model_sha256
    }
    pub fn output_limit(&self) -> usize {
        self.output_limit
    }
    pub fn reserved_bytes(&self) -> u64 {
        self.reserve.as_ref().map_or(0, Reservation::bytes)
    }
    pub fn prepare(&mut self, model: &Model, cancel: &Cancellation) -> Result<GenerationRequest> {
        self.prepare_conversation(model, cancel)
    }
    pub fn prepare_conversation(
        &mut self,
        model: &dyn crate::resident::TextConversation,
        cancel: &Cancellation,
    ) -> Result<GenerationRequest> {
        if self.task != SessionTask::TextGeneration
            || self.context > model.context_limit()
            || !model.supports_history()
        {
            return Err(ApiError::bad(
                "conversation task/context is incompatible with loaded model",
            ));
        }
        model.validate_options(
            &self
                .settings
                .request(self.model.clone(), self.context, String::new())
                .options(),
        )?;
        let mut messages: Vec<_> = self
            .messages
            .iter()
            .filter(|m| {
                m.status == TurnStatus::Completed || m.turn_id.as_deref() == Some(&self.turn_id)
            })
            .cloned()
            .collect();
        let mut excluded: Vec<_> = self
            .messages
            .iter()
            .filter(|m| {
                m.status != TurnStatus::Completed && m.turn_id.as_deref() != Some(&self.turn_id)
            })
            .map(|m| m.id.clone())
            .collect();
        loop {
            cancel.check()?;
            let rendered = model.render_history(
                &messages
                    .iter()
                    .map(|m| Message {
                        role: m.role.clone(),
                        content: m.content.clone(),
                        tool_calls: m.tool_calls.clone(),
                        tool_call_id: m.tool_call_id.clone(),
                    })
                    .collect::<Vec<_>>(),
                &self
                    .settings
                    .request(self.model.clone(), self.context, String::new())
                    .tool_config(),
            );
            let rendered = match rendered {
                Ok(prompt) => Some(prompt),
                Err(error)
                    if self.policy == OverflowPolicy::OldestCompleteTurns
                        && error.code == ErrorCode::ResourceExhausted =>
                {
                    None
                }
                Err(error) => return Err(error.into()),
            };
            if let Some(rendered) = rendered {
                let tokens = model.token_count(&rendered)?;
                if tokens
                    .checked_add(self.settings.max_tokens)
                    .is_some_and(|total| total <= self.context)
                {
                    let history_revision = self.watch.snapshot()?.revision;
                    self.watch.0.lock().map_err(|_| internal())?.receipt.context =
                        Some(ContextAccounting {
                            policy: self.policy,
                            history_revision,
                            context: self.context,
                            prompt_tokens: tokens,
                            reserved_output_tokens: self.settings.max_tokens,
                            included_message_ids: messages.iter().map(|m| m.id.clone()).collect(),
                            excluded_message_ids: excluded,
                        });
                    let mut request =
                        self.settings
                            .request(self.model.clone(), self.context, rendered);
                    // The prompt is already rendered and counted. Retain tool configuration for parsing.
                    request.messages = Some(messages.iter().map(as_message).collect());
                    request.prompt = None;
                    return Ok(request);
                }
            }
            if self.policy == OverflowPolicy::Reject {
                return Err(ApiError::bad(
                    "rendered history plus output reserve exceeds session context",
                ));
            }
            let Some(oldest) = messages
                .iter()
                .find(|m| m.role == "user" && m.turn_id.as_deref() != Some(&self.turn_id))
                .and_then(|m| m.turn_id.clone())
            else {
                return Err(ApiError::bad(
                    "protected instructions and current turn plus output reserve cannot fit context",
                ));
            };
            let mut group = std::collections::BTreeSet::from([oldest]);
            loop {
                let calls: std::collections::BTreeSet<_> = messages
                    .iter()
                    .filter(|m| m.turn_id.as_ref().is_some_and(|id| group.contains(id)))
                    .flat_map(|m| m.tool_calls.iter().map(|c| c.id.as_str()))
                    .collect();
                let linked: Vec<_> = messages
                    .iter()
                    .filter(|m| {
                        m.role == "tool"
                            && m.tool_call_id
                                .as_deref()
                                .is_some_and(|id| calls.contains(id))
                    })
                    .filter_map(|m| m.turn_id.clone())
                    .collect();
                let count = group.len();
                group.extend(linked);
                if group.len() == count {
                    break;
                }
            }
            if group.contains(&self.turn_id) {
                return Err(ApiError::bad(
                    "protected current tool-call/result group cannot fit context",
                ));
            }
            messages.retain(|m| {
                if m.turn_id.as_ref().is_some_and(|id| group.contains(id)) {
                    excluded.push(m.id.clone());
                    false
                } else {
                    true
                }
            });
        }
    }
    pub fn finish(&mut self, event: &GenerationEvent) -> Result<()> {
        if self.finished {
            return Err(internal());
        }
        let (status, result, error) = match event {
            GenerationEvent::Finished(result) => (TurnStatus::Completed, Some(result), None),
            GenerationEvent::Failed(error) => (
                if matches!(
                    error.error.code,
                    ErrorCode::Cancelled | ErrorCode::DeadlineExceeded
                ) {
                    TurnStatus::Cancelled
                } else {
                    TurnStatus::Failed
                },
                None,
                Some(error.error.clone()),
            ),
            GenerationEvent::Token(_) | GenerationEvent::ToolDelta(_) => return Err(internal()),
        };
        if result.is_some_and(|r| result_bytes(r) > self.output_limit) {
            return Err(Error::resource(
                "assistant output exceeds remaining session history bytes",
            )
            .into());
        }
        let mut state = self.sessions.state()?;
        let session = state.get_mut(&self.id).ok_or_else(internal)?;
        if session.view.active_turn.as_deref() != Some(&self.turn_id)
            || !session.keys.contains_key(&self.key)
        {
            return Err(internal());
        }
        let old_view = session.view.clone();
        let old_len = session.history.len();
        let mut receipt = self.watch.0.lock().map_err(|_| internal())?;
        for message in &mut session.history {
            if message.turn_id.as_deref() == Some(&self.turn_id) {
                message.status = status;
            }
        }
        let output_bytes = result.map_or(0, result_bytes);
        let token_bytes = result.map_or(0, |r| r.token_ids.len() * 4);
        if let Some(result) = result {
            let mut assistant = history_message(
                "assistant",
                result.text.clone(),
                Some(self.turn_id.clone()),
                status,
            );
            assistant.tool_calls = result.tool_calls.clone();
            session.history.push(assistant);
            session.view.history_bytes += output_bytes;
            session.view.message_count = session.history.len();
        }
        session.view.pending_tool_calls = pending_calls(&session.history)?;
        session.view.active_turn = None;
        session.view.revision += 1;
        session.view.updated = now();
        session.touched = Instant::now();
        let mut candidate = receipt.receipt.clone();
        candidate.revision = session.view.revision;
        candidate.status = status;
        candidate.result = result.cloned();
        candidate.error = error;
        let declaration_reserve = if self.settings.tools.is_empty() {
            0
        } else {
            coheryn_engine::tools::DECLARATION_RESERVATION
        };
        let retained = declaration_reserve
            + 16384
            + 8 * (self.input_bytes
                + output_bytes
                + token_bytes
                + serde_json::to_vec(&self.settings)
                    .map_err(|_| internal())?
                    .len()) as u64;
        if self.reserve.as_ref().ok_or_else(internal)?.bytes() < retained {
            return Err(Error::resource("retained turn exceeds admission").into());
        }
        if let Err(error) = self
            .sessions
            .persist(session, Some((&self.key, &candidate)))
        {
            session.view = old_view;
            session.history.truncate(old_len);
            for message in &mut session.history {
                if message.turn_id.as_deref() == Some(&self.turn_id) {
                    message.status = TurnStatus::Pending;
                }
            }
            self.cache = None;
            return Err(error);
        }
        receipt.receipt = candidate;
        receipt._retained = Some(
            self.reserve
                .as_mut()
                .ok_or_else(internal)?
                .split(retained)?,
        );
        if status == TurnStatus::Completed {
            session.cache = self
                .cache
                .take()
                .filter(|cache| cache.is_live())
                .map(|cache| (cache, Instant::now()));
        }
        self.finished = true;
        drop(receipt);
        self.sessions.trim_caches(&mut state);
        Ok(())
    }
}
impl Drop for TurnLease {
    fn drop(&mut self) {
        if !self.finished {
            let error = Error::new(
                ErrorCode::Cancelled,
                "managed turn interrupted before successful completion",
            )
            .into();
            if self.finish(&GenerationEvent::Failed(error)).is_err()
                && let Ok(mut state) = self.sessions.0.state.lock()
                && let Some(session) = state.get_mut(&self.id)
                && session.view.active_turn.as_deref() == Some(&self.turn_id)
            {
                session.view.active_turn = None;
                session.view.revision = session.view.revision.saturating_add(1);
                session.view.storage_error =
                    Some("history commit failed; reset or restore is required".into());
                session.cache = None;
                for m in &mut session.history {
                    if m.turn_id.as_deref() == Some(&self.turn_id) {
                        m.status = TurnStatus::Failed;
                    }
                }
                if let Ok(mut watch) = self.watch.0.lock() {
                    watch.receipt.status = TurnStatus::Failed;
                    watch.receipt.revision = session.view.revision;
                    watch.receipt.result = None;
                    watch.receipt.error = Some(Error::new(
                        ErrorCode::Io,
                        "history was not saved; recovery is required",
                    ));
                    if let Some(reserve) = &mut self.reserve {
                        watch._retained = reserve
                            .split(
                                16384
                                    + 8 * self.input_bytes as u64
                                    + if self.settings.tools.is_empty() {
                                        0
                                    } else {
                                        coheryn_engine::tools::DECLARATION_RESERVATION
                                    }
                                    + 8 * serde_json::to_vec(&self.settings)
                                        .map_or(0, |v| v.len() as u64),
                            )
                            .ok();
                    }
                }
            }
        }
    }
}

pub(crate) fn validate_document(
    doc: &ExportDocument,
    limits: &SessionLimits,
    pending: bool,
) -> Result<()> {
    use std::collections::BTreeSet;
    if doc.session.messages.is_some()
        || doc.session.cached_tokens != 0
        || doc.session.cache_reserved_bytes != 0
    {
        return Err(ApiError::bad(
            "history documents cannot contain a duplicate page or KV state",
        ));
    }
    if !matches!(doc.version, 1..=3)
        || doc.messages.len() > limits.messages
        || doc.turns.len() > limits.turn_keys
        || doc.session.message_count != doc.messages.len()
        || doc.session.context == 0
        || doc.session.context > 1_048_576
        || doc.session.revision > u64::MAX - 1_000_000
        || doc.session.epoch > u64::MAX - 1_000_000
        || (!pending && doc.session.active_turn.is_some())
    {
        return Err(ApiError::bad("unsupported or oversized history document"));
    }
    identifier(&doc.session.id)?;
    crate::registry::validate_id(&doc.session.model)?;
    crate::registry::validate_hash(&doc.session.model_sha256)?;
    if let Some(alias) = &doc.session.alias {
        identifier(alias)?;
    }
    let mut asset_ids = BTreeSet::new();
    if doc.session.image_assets.len() > limits.images.session_items
        || doc
            .session
            .image_assets
            .iter()
            .try_fold(0u64, |n, a| n.checked_add(a.bytes))
            .is_none_or(|n| n > limits.images.session_bytes)
    {
        return Err(ApiError::bad("saved image quotas exceeded"));
    }
    for asset in &doc.session.image_assets {
        asset.validate()?;
        if !asset_ids.insert(asset.id.as_str()) {
            return Err(ApiError::bad("duplicate saved image asset"));
        }
    }
    if doc.session.task == SessionTask::TextGeneration && !asset_ids.is_empty() {
        return Err(ApiError::bad("text history cannot contain image assets"));
    }
    if doc.session.task == SessionTask::ImageToText && doc.session.context > 4096 {
        return Err(ApiError::bad(
            "image history context exceeds the qualified 4096-position capacity",
        ));
    }
    if doc.session.task == SessionTask::ImageToText && doc.version != 3 {
        return Err(ApiError::bad("image histories require format version 3"));
    }
    let mut ids = BTreeSet::new();
    let mut size = 0usize;
    let mut turn_ids = BTreeSet::new();
    let mut keys = BTreeSet::new();
    for message in &doc.messages {
        identifier(&message.id)?;
        if !message.images.is_empty()
            && (doc.session.task != SessionTask::ImageToText
                || message.role != "user"
                || message
                    .images
                    .iter()
                    .any(|id| !asset_ids.contains(id.as_str())))
        {
            return Err(ApiError::bad("invalid saved image reference/role"));
        }
        if !ids.insert(message.id.as_str())
            || !matches!(
                message.role.as_str(),
                "system" | "user" | "assistant" | "tool"
            )
            || (!pending && message.status == TurnStatus::Pending)
        {
            return Err(ApiError::bad(
                "invalid history message identity, role or state",
            ));
        }
        if doc.session.task == SessionTask::ImageToText
            && (message.role == "tool"
                || !message.tool_calls.is_empty()
                || message.tool_call_id.is_some())
        {
            return Err(ApiError::bad("tools are not qualified in image histories"));
        }
        if (!message.tool_calls.is_empty()
            && (message.role != "assistant" || message.status != TurnStatus::Completed))
            || (message.role == "tool") != message.tool_call_id.is_some()
        {
            return Err(ApiError::bad("invalid saved tool role/state"));
        }
        if let Some(id) = &message.tool_call_id {
            coheryn_engine::tools::identifier(id)?;
        }
        size = size
            .checked_add(message_bytes(message))
            .filter(|n| *n <= limits.history_bytes)
            .ok_or_else(|| Error::resource("imported history exceeds text byte limit"))?;
        if let Some(turn) = &message.turn_id {
            identifier(turn)?;
        } else if message.role != "system" {
            return Err(ApiError::bad("conversation message lacks a turn ID"));
        }
    }
    if size != doc.session.history_bytes {
        return Err(ApiError::bad("history byte accounting mismatch"));
    }
    for turn in &doc.turns {
        identifier(&turn.key)?;
        identifier(&turn.receipt.turn_id)?;
        if !keys.insert(&turn.key)
            || !turn_ids.insert(&turn.receipt.turn_id)
            || turn.receipt.session_id != doc.session.id
            || turn.receipt.revision > doc.session.revision
            || (!pending && turn.receipt.status == TurnStatus::Pending)
        {
            return Err(ApiError::bad("invalid replay identity or revision"));
        }
        let mut check = turn.settings.request(
            doc.session.model.clone(),
            doc.session.context,
            "validation".into(),
        );
        check.prompt = None;
        check.messages = Some(vec![Message {
            role: "user".into(),
            content: "validation".into(),
            ..Default::default()
        }]);
        check.validate()?;
        if doc.session.task == SessionTask::ImageToText {
            crate::vision::validate_controls(&check)?;
        }
        if let Some(result) = &turn.receipt.result {
            check
                .tool_config()
                .validate_output(&result.tool_calls, result.finish_reason)?;
        }
        if let Some(result) = &turn.receipt.result {
            if turn.receipt.status != TurnStatus::Completed
                || result.finish_reason == coheryn_engine::FinishReason::Cancelled
                || result.token_ids.len() != result.completion_tokens
                || result.completion_tokens > doc.session.context
                || result
                    .prompt_tokens
                    .checked_add(result.completion_tokens)
                    .is_none_or(|n| n > doc.session.context)
                || result.reused_prompt_tokens > result.prompt_tokens
                || result.text.len() > limits.history_bytes
                || turn.receipt.error.is_some()
            {
                return Err(ApiError::bad("invalid completed history result"));
            }
        } else if turn.receipt.status == TurnStatus::Completed {
            return Err(ApiError::bad("completed history lacks its result"));
        }
        if let Some(context) = &turn.receipt.context
            && (context.context != doc.session.context
                || context.history_revision > doc.session.revision
                || context
                    .prompt_tokens
                    .checked_add(context.reserved_output_tokens)
                    .is_none_or(|n| n > context.context)
                || context
                    .included_message_ids
                    .iter()
                    .chain(&context.excluded_message_ids)
                    .any(|id| !ids.contains(id.as_str())))
        {
            return Err(ApiError::bad("invalid saved context references"));
        }
    }
    for turn in &doc.turns {
        let messages: Vec<_> = doc
            .messages
            .iter()
            .filter(|m| m.turn_id.as_deref() == Some(&turn.receipt.turn_id))
            .collect();
        let completed = turn.receipt.status == TurnStatus::Completed;
        let input_count = messages.len().saturating_sub(usize::from(completed));
        if input_count == 0
            || input_count > 8
            || messages.iter().any(|m| m.status != turn.receipt.status)
        {
            return Err(ApiError::bad("invalid saved turn message group"));
        }
        let inputs = &messages[..input_count];
        if !(inputs.len() == 1
            && inputs[0].role == "user"
            && (!inputs[0].content.is_empty() || !inputs[0].images.is_empty()))
            && !inputs.iter().all(|m| m.role == "tool")
        {
            return Err(ApiError::bad(
                "saved turn needs a user input or matching tool results",
            ));
        }
        if completed {
            let output = messages.last().unwrap();
            if output.role != "assistant"
                || turn
                    .receipt
                    .result
                    .as_ref()
                    .is_none_or(|r| r.text != output.content || r.tool_calls != output.tool_calls)
            {
                return Err(ApiError::bad("saved assistant and replay result differ"));
            }
        }
    }
    if doc.version == 1
        && doc
            .messages
            .iter()
            .any(|m| m.role == "tool" || !m.tool_calls.is_empty() || m.tool_call_id.is_some())
    {
        return Err(ApiError::bad(
            "tool histories require export format version 2",
        ));
    }
    let conditioned = doc.turns.iter().any(|t| t.settings.tool_config().enabled())
        || doc
            .messages
            .iter()
            .any(|m| m.role == "tool" || !m.tool_calls.is_empty());
    if (conditioned || doc.session.tool_profile.is_some())
        && doc.session.tool_profile.as_deref()
            != Some(coheryn_engine::tools::profile_identity().as_str())
    {
        return Err(Error::new(
            ErrorCode::UnsupportedFeature,
            "missing or unsupported saved tool-template identity",
        )
        .into());
    }
    if doc.version == 1
        && (doc.session.tool_profile.is_some()
            || doc.turns.iter().any(|t| {
                !t.settings.tools.is_empty()
                    || t.settings.tool_choice.is_some()
                    || t.settings.parallel_tool_calls.is_some()
            }))
    {
        return Err(ApiError::bad(
            "tool settings require export format version 2",
        ));
    }
    let computed = pending_calls(&doc.messages)?;
    if computed != doc.session.pending_tool_calls {
        return Err(ApiError::bad(
            "saved pending-tool metadata differs from history",
        ));
    }
    for message in &doc.messages {
        if let Some(turn) = &message.turn_id
            && !turn_ids.contains(turn)
        {
            return Err(ApiError::bad("orphaned history turn"));
        }
    }
    Ok(())
}

fn message_bytes(m: &HistoryMessage) -> usize {
    m.content.len()
        + m.images.iter().map(String::len).sum::<usize>()
        + m.tool_call_id.as_ref().map_or(0, String::len)
        + if m.tool_calls.is_empty() {
            0
        } else {
            serde_json::to_vec(&m.tool_calls)
                .expect("serializable calls")
                .len()
        }
}
fn result_bytes(r: &GenerationResult) -> usize {
    r.text.len()
        + if r.tool_calls.is_empty() {
            0
        } else {
            serde_json::to_vec(&r.tool_calls)
                .expect("serializable calls")
                .len()
        }
}
fn as_message(m: &HistoryMessage) -> Message {
    Message {
        role: m.role.clone(),
        content: m.content.clone(),
        tool_calls: m.tool_calls.clone(),
        tool_call_id: m.tool_call_id.clone(),
    }
}
fn pending_calls(history: &[HistoryMessage]) -> Result<Vec<String>> {
    let messages: Vec<_> = history
        .iter()
        .filter(|m| m.status == TurnStatus::Completed)
        .map(as_message)
        .collect();
    if messages.is_empty() {
        return Ok(vec![]);
    }
    Ok(coheryn_engine::tools::ToolConfig::default().validate_messages(&messages, true)?)
}

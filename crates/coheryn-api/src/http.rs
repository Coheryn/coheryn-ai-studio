use crate::{
    error::{ApiError, Result},
    scheduler::{GenerationEvent, GenerationHandle},
    service::Service,
    wire::*,
};
use axum::{
    Json, Router,
    body::{Body, to_bytes},
    extract::{Path, Query, Request, State},
    http::{StatusCode, header},
    middleware::{self, Next},
    response::{
        IntoResponse, Response, Sse,
        sse::{Event, KeepAlive},
    },
    routing::{delete, get, post},
};
use coheryn_engine::{Error, ErrorCode, FinishReason, GenerationResult};
use futures_util::{Stream, StreamExt, stream};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use std::{
    convert::Infallible,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

pub fn router(service: Arc<Service>) -> Router {
    Router::new()
        .route("/health", get(|| async { Json(json!({"status":"ok"})) }))
        .route("/v1/models", get(openai_models))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/embeddings", post(embeddings))
        .merge(crate::session_http::routes())
        .route("/api/v1/generate", post(native))
        .route("/api/v1/models", get(models))
        .route("/api/v1/models/validate", post(validate_model))
        .route("/api/v1/models/import", post(import_model))
        .route("/api/v1/models/pull", post(pull))
        .route("/api/v1/models/load", post(load))
        .route("/api/v1/models/register-pair", post(register_pair))
        .route("/api/v1/models/unload", post(unload))
        .route("/api/v1/models/{id}/inspect", get(inspect_model))
        .route("/api/v1/models/{id}", delete(remove))
        .route("/api/v1/plans/estimate", post(estimate))
        .route("/api/v1/system/resources", get(resources))
        .route("/api/v1/server/status", get(status))
        .route("/api/v1/server/metrics", get(status))
        .fallback(|| async { ApiError::missing("route not found") })
        .method_not_allowed_fallback(|| async {
            ApiError::new(
                StatusCode::METHOD_NOT_ALLOWED,
                ErrorCode::InvalidRequest,
                "method not allowed",
            )
        })
        .layer(middleware::from_fn_with_state(service.clone(), access))
        .with_state(service)
}
async fn access(
    State(service): State<Arc<Service>>,
    request: Request,
    next: Next,
) -> Result<Response> {
    let start = std::time::Instant::now();
    let path = request.uri().path().to_owned();
    let method = request.method().to_string();
    if request.headers().len() > 64
        || request
            .headers()
            .iter()
            .map(|(k, v)| k.as_str().len() + v.len())
            .sum::<usize>()
            > 8192
    {
        return Err(ApiError::new(
            StatusCode::REQUEST_HEADER_FIELDS_TOO_LARGE,
            ErrorCode::InvalidRequest,
            "headers exceed configured count/byte limits",
        ));
    }
    if path != "/health" {
        service.auth.authorize(
            request.headers(),
            path.starts_with("/api/") && path != "/api/v1/generate",
        )?;
    }
    let slot = service
        .http_slots
        .clone()
        .try_acquire_owned()
        .map_err(|_| {
            ApiError::new(
                StatusCode::TOO_MANY_REQUESTS,
                ErrorCode::QueueFull,
                "too many HTTP requests",
            )
        })?;
    let mut response = next.run(request).await;
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        header::HeaderValue::from_static("no-store"),
    );
    response.headers_mut().insert(
        "x-content-type-options",
        header::HeaderValue::from_static("nosniff"),
    );
    eprintln!(
        "{}",
        json!({"event":"http_request","method":method,"path":path.chars().take(1024).collect::<String>(),"status":response.status().as_u16(),"elapsed_ms":start.elapsed().as_secs_f64()*1000.})
    );
    // Keep the HTTP permit until the entire body is consumed or disconnected,
    // including slow SSE consumers after computation itself has ended.
    let (parts, body) = response.into_parts();
    let stream = stream::unfold(
        (body.into_data_stream(), slot),
        |(mut body, slot)| async move { body.next().await.map(|next| (next, (body, slot))) },
    );
    Ok(Response::from_parts(parts, Body::from_stream(stream)))
}
pub(crate) async fn json_body<T: DeserializeOwned>(request: Request) -> Result<T> {
    json_body_limit(request, 1 << 20).await
}
pub(crate) async fn json_body_limit<T: DeserializeOwned>(
    request: Request,
    limit: usize,
) -> Result<T> {
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
    let bytes = tokio::time::timeout(
        Duration::from_secs(30),
        to_bytes(request.into_body(), limit),
    )
    .await
    .map_err(|_| ApiError::bad("request body timeout"))?
    .map_err(|_| {
        ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            ErrorCode::InvalidRequest,
            "JSON body exceeds configured byte limit or was interrupted",
        )
    })?;
    serde_json::from_slice(&bytes).map_err(|e| ApiError::bad(format!("invalid request JSON: {e}")))
}

struct EmbeddingBodyOwner {
    bytes: Vec<u8>,
    _model: Arc<crate::resident::ResidentEmbedding>,
    _reservation: coheryn_engine::memory::Reservation,
}
impl AsRef<[u8]> for EmbeddingBodyOwner {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}
async fn embeddings(State(s): State<Arc<Service>>, request: Request) -> Result<Response> {
    use crate::embeddings::{self, EmbeddingRequest};
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
    let configured_cap = s.config.embeddings.body_bytes;
    let declared = request
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|n| n.parse::<usize>().ok());
    let cap = declared.unwrap_or(configured_cap).min(configured_cap);
    if request
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|n| n.parse::<usize>().ok())
        .is_some_and(|n| n > configured_cap)
    {
        return Err(embeddings::batch_limit(
            "embedding JSON body exceeds configured byte limit",
        ));
    }
    // Includes raw JSON, bounded Serde collections/strings and their parsing overlap.
    let payload = s.budget.reserve(embeddings::parser_allowance(cap))?;
    let bytes = tokio::time::timeout(Duration::from_secs(30), to_bytes(request.into_body(), cap))
        .await
        .map_err(|_| ApiError::bad("embedding request body timeout"))?
        .map_err(|_| {
            embeddings::batch_limit("embedding request body exceeded its limit or was interrupted")
        })?;
    let request:EmbeddingRequest=serde_json::from_slice(&bytes).map_err(|_|ApiError::bad("invalid embedding JSON, unsupported field, or input type; supply a string or array of strings"))?;
    drop(bytes);
    request.validate(&s.config.embeddings)?;
    let mut handle =
        s.scheduler
            .submit_embedding(s.embedding(&request.model)?, request, payload)?;
    let delivery = handle.receive().await?;
    let crate::scheduler::EmbeddingDelivery {
        bytes,
        space,
        token_counts,
        truncated,
        diagnostics,
        _model,
        _serialization,
    } = delivery;
    let length = bytes.len();
    let bytes = axum::body::Bytes::from_owner(EmbeddingBodyOwner {
        bytes,
        _model,
        _reservation: _serialization,
    });
    let deadline =
        tokio::time::Instant::now() + Duration::from_secs(s.config.consumer_timeout_seconds);
    let stream = stream::unfold((bytes, deadline), |(mut bytes, deadline)| async move {
        if bytes.is_empty() {
            return None;
        }
        if tokio::time::Instant::now() >= deadline {
            return Some((
                Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "embedding response delivery timeout",
                )),
                (axum::body::Bytes::new(), deadline),
            ));
        }
        let chunk = bytes.split_to(bytes.len().min(16 << 10));
        Some((Ok(chunk), (bytes, deadline)))
    });
    let mut response = Body::from_stream(stream).into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("application/json"),
    );
    for (key, value) in [
        ("content-length", length.to_string()),
        ("x-coheryn-embedding-space", space.id),
        ("x-coheryn-model-sha256", space.model_sha256),
        ("x-coheryn-embedding-dimension", space.dimension.to_string()),
        ("x-coheryn-token-counts", token_counts),
        ("x-coheryn-truncated-indices", truncated),
        (
            "x-coheryn-batch-size",
            diagnostics["plan"]["batch_size"].to_string(),
        ),
        (
            "x-coheryn-batch-tokens",
            diagnostics["plan"]["token_capacity"].to_string(),
        ),
    ] {
        response.headers_mut().insert(
            axum::http::HeaderName::from_static(key),
            header::HeaderValue::from_str(&value).map_err(|_| {
                Error::new(ErrorCode::Internal, "invalid embedding response metadata")
            })?,
        );
    }
    Ok(response)
}
async fn models(State(s): State<Arc<Service>>) -> Result<Json<Value>> {
    Ok(Json(
        json!({"models":s.registry.list()?,"loaded":s.loaded()?}),
    ))
}
async fn openai_models(State(s): State<Arc<Service>>) -> Result<Json<Value>> {
    let records = s.registry.list()?;
    Ok(Json(
        json!({"object":"list","data":records.into_iter().map(|r|json!({"id":r.id,"object":"model","created":0,"owned_by":"local"})).collect::<Vec<_>>()}),
    ))
}
async fn status(State(s): State<Arc<Service>>) -> Result<Json<Value>> {
    Ok(Json(s.status()?))
}
async fn resources(State(s): State<Arc<Service>>) -> Result<Json<Value>> {
    let memory = s.budget.stats();
    let hardware = tokio::task::spawn_blocking(crate::hardware::probe).await?;
    Ok(Json(
        json!({"ram_total_bytes":hardware.ram_total_bytes,"ram_available_bytes":hardware.ram_available_bytes,
        "swap_total_bytes":hardware.swap_total_bytes,"swap_used_bytes":hardware.swap_used_bytes,
        "logical_cpus":hardware.logical_cpus,"hardware": hardware, "engine_memory": memory,
        "mapped_weight_bytes":0,"gpu":null,
        "note":"OS observations, session buffer capacities and estimated reservations are distinct; GPU counters are unavailable"}),
    ))
}
async fn load(State(s): State<Arc<Service>>, request: Request) -> Result<Json<Value>> {
    let body: LoadRequest = json_body(request).await?;
    Ok(Json(s.load(body.model, body.context).await?))
}
async fn unload(State(s): State<Arc<Service>>, request: Request) -> Result<Json<Value>> {
    let body: ModelRequest = json_body(request).await?;
    let _admin = s.admin.lock().await;
    s.unload(&body.model)?;
    Ok(Json(json!({"model":body.model,"loaded":false})))
}
async fn remove(State(s): State<Arc<Service>>, Path(id): Path<String>) -> Result<Json<Value>> {
    let _admin = s.admin.lock().await;
    s.remove(&id)?;
    Ok(Json(json!({"id":id,"deleted":true})))
}
async fn inspect_model(
    State(s): State<Arc<Service>>,
    Path(id): Path<String>,
) -> Result<Json<Value>> {
    Ok(Json(s.inspect(id, false).await?))
}
async fn validate_model(State(s): State<Arc<Service>>, request: Request) -> Result<Response> {
    let body: ValidateRequest = json_body(request).await?;
    let result = s.inspect(body.model, body.deep).await?;
    let status = if result["compatibility"]["executable"] == true {
        StatusCode::OK
    } else {
        StatusCode::BAD_REQUEST
    };
    Ok((status, Json(result)).into_response())
}
async fn estimate(State(s): State<Arc<Service>>, request: Request) -> Result<Json<Value>> {
    let body: EstimateRequest = json_body(request).await?;
    Ok(Json(s.estimate(body).await?))
}
async fn import_model(State(s): State<Arc<Service>>, request: Request) -> Result<Json<Value>> {
    let query: Query<ImportQuery> =
        Query::try_from_uri(request.uri()).map_err(|e| ApiError::bad(e.to_string()))?;
    if request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        != Some("application/octet-stream")
    {
        return Err(ApiError::new(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            ErrorCode::InvalidRequest,
            "imports require application/octet-stream",
        ));
    }
    if request
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .is_some_and(|n| n > s.registry.max_model_bytes)
    {
        return Err(ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            ErrorCode::ResourceExhausted,
            "import exceeds maximum model size",
        ));
    }
    let _admin = s.admin.lock().await;
    let stream = request.into_body().into_data_stream();
    let result = tokio::time::timeout(
        Duration::from_secs(600),
        s.ingest(query.0.name, query.0.sha256, None, stream),
    )
    .await
    .map_err(|_| ApiError::bad("model import timeout"))??;
    Ok(Json(json!(result)))
}
async fn pull(State(s): State<Arc<Service>>, request: Request) -> Result<Json<Value>> {
    let body: PullRequest = json_body(request).await?;
    crate::registry::validate_hash(&body.sha256)?;
    s.registry.check_new(&body.name)?;
    let _admin = s.admin.lock().await;
    let response = crate::download::response(&body.url).await?;
    if response
        .content_length()
        .is_some_and(|v| v > s.registry.max_model_bytes)
    {
        return Err(Error::resource("download exceeds maximum model size").into());
    }
    // Do not retain URL query strings, which may contain temporary acquisition credentials.
    let mut source =
        reqwest::Url::parse(&body.url).map_err(|_| ApiError::bad("invalid download URL"))?;
    source.set_query(None);
    let result = tokio::time::timeout(
        Duration::from_secs(600),
        s.ingest(
            body.name,
            Some(body.sha256),
            Some(source.to_string()),
            response.bytes_stream(),
        ),
    )
    .await
    .map_err(|_| ApiError::bad("model pull timeout"))??;
    Ok(Json(json!(result)))
}

#[derive(Clone, Copy, PartialEq)]
enum Flavor {
    Completion,
    Chat,
    Native,
}
async fn completions(State(s): State<Arc<Service>>, request: Request) -> Result<Response> {
    let _ingress = s.budget.reserve(coheryn_engine::planner::REQUEST_BYTES)?;
    generation(s, json_body(request).await?, Flavor::Completion).await
}
async fn chat(State(s): State<Arc<Service>>, request: Request) -> Result<Response> {
    chat_or_native(s, request, Flavor::Chat).await
}
async fn native(State(s): State<Arc<Service>>, request: Request) -> Result<Response> {
    chat_or_native(s, request, Flavor::Native).await
}
async fn chat_or_native(s: Arc<Service>, request: Request, flavor: Flavor) -> Result<Response> {
    let (value, wire_bytes, mut ingress) =
        crate::vision::read_body(request, &s.budget, &s.config.vision).await?;
    let model = value["model"]
        .as_str()
        .ok_or_else(|| ApiError::bad("model must be a managed ID"))?
        .to_owned();
    let vision = s.is_vision(&model).unwrap_or(false);
    if vision || crate::vision::has_parts(&value) {
        let request = crate::vision::VisionRequest::from_value(value)?;
        let engine = s.vision(&model)?; // Reject unsupported models before any image acquisition.
        ingress.extend(s.config.vision.images.max_total_encoded_bytes)?;
        let streaming = request.controls.stream;
        let include_usage = request
            .controls
            .stream_options
            .as_ref()
            .is_some_and(|o| o.include_usage);
        let cancel = coheryn_engine::Cancellation::default()
            .with_timeout(request.controls.timeout_seconds)?;
        let (controls, messages, images) = tokio::time::timeout(
            Duration::from_secs(s.config.vision.acquisition_timeout_seconds),
            request.acquire(&s.config.vision, &cancel),
        )
        .await
        .map_err(|_| ApiError::bad("image request acquisition timeout"))??;
        let handle = s
            .scheduler
            .submit_vision(engine, controls, messages, images, ingress)
            .await?;
        generation_response(handle, model, streaming, include_usage, false, flavor).await
    } else {
        // Preserve the ordinary text request ceiling even though this endpoint accepts images.
        if wire_bytes > 1 << 20 {
            return Err(ApiError::bad("text request exceeds one MiB"));
        }
        let request = serde_json::from_value(value)
            .map_err(|e| ApiError::bad(format!("invalid request JSON: {e}")))?;
        generation(s, request, flavor).await
    }
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RegisterPair {
    model: String,
    language: String,
    vision: String,
}
async fn register_pair(State(s): State<Arc<Service>>, request: Request) -> Result<Response> {
    let request: RegisterPair = json_body(request).await?;
    Ok(Json(
        s.register_pair(request.model, request.language, request.vision)
            .await?,
    )
    .into_response())
}
fn reason(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::Cancelled => "stop",
        FinishReason::ToolCalls => "tool_calls",
    }
}
fn usage(result: &GenerationResult) -> Value {
    json!({"prompt_tokens":result.prompt_tokens,"completion_tokens":result.completion_tokens,"total_tokens":result.prompt_tokens+result.completion_tokens})
}
fn response_value(
    flavor: Flavor,
    id: &str,
    model: &str,
    created: u64,
    result: GenerationResult,
) -> Value {
    if flavor == Flavor::Native {
        return json!(result);
    }
    let choices = if flavor == Flavor::Chat {
        {
            let mut message = json!({"role":"assistant","content":result.text});
            if !result.tool_calls.is_empty() {
                message["tool_calls"] = json!(result.tool_calls);
                if result.text.is_empty() {
                    message["content"] = Value::Null;
                }
            }
            json!([{"index":0,"message":message,"logprobs":null,"finish_reason":reason(result.finish_reason)}])
        }
    } else {
        json!([{"index":0,"text":result.text,"logprobs":null,"finish_reason":reason(result.finish_reason)}])
    };
    let mut response = json!({"id":id,"object":if flavor==Flavor::Chat{"chat.completion"}else{"text_completion"},"created":created,"model":model,"choices":choices,"usage":usage(&result)});
    response["coheryn"] = json!({"vision":result.vision,"sampling_profile":if result.vision.is_some(){None}else{Some(coheryn_engine::sampling::PROFILE)},"sampling":result.sampling_metadata,"constraint_validation":result.constraint_validation,"batching":result.batching});
    response
}
fn chunk(flavor: Flavor, id: &str, model: &str, created: u64, choices: Value) -> Value {
    json!({"id":id,"object":if flavor==Flavor::Chat{"chat.completion.chunk"}else{"text_completion"},"created":created,"model":model,"choices":choices})
}
fn event(value: impl Serialize) -> Event {
    Event::default().data(serde_json::to_string(&value).expect("serializable wire event"))
}
async fn generation(
    s: Arc<Service>,
    request: GenerationRequest,
    flavor: Flavor,
) -> Result<Response> {
    request.validate()?;
    if (flavor == Flavor::Completion && request.messages.is_some())
        || (flavor == Flavor::Chat && request.prompt.is_some())
    {
        return Err(ApiError::bad(
            "wrong prompt/messages field for this endpoint",
        ));
    }
    s.require_tool_model(
        &request.model,
        &request.tool_config(),
        request.messages.as_ref().is_some_and(|m| {
            m.iter()
                .any(|m| m.role == "tool" || !m.tool_calls.is_empty())
        }),
    )?;
    let tool_mode = request.tool_config().enabled();
    let model = request.model.clone();
    let streaming = request.stream;
    let include_usage = request
        .stream_options
        .as_ref()
        .is_some_and(|v| v.include_usage);
    let handle = s.scheduler.submit(s.model(&model)?, request)?;
    generation_response(handle, model, streaming, include_usage, tool_mode, flavor).await
}
async fn generation_response(
    handle: GenerationHandle,
    model: String,
    streaming: bool,
    include_usage: bool,
    tool_mode: bool,
    flavor: Flavor,
) -> Result<Response> {
    let id = format!(
        "{}{}",
        if flavor == Flavor::Chat {
            "chatcmpl-"
        } else {
            "cmpl-"
        },
        uuid::Uuid::new_v4().simple()
    );
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    if !streaming {
        let mut handle = handle;
        while let Some(message) = handle.next().await {
            match message {
                GenerationEvent::Token(_) | GenerationEvent::ToolDelta(_) => (),
                GenerationEvent::Finished(result) => {
                    return Ok(
                        Json(response_value(flavor, &id, &model, created, result)).into_response()
                    );
                }
                GenerationEvent::Failed(error) => return Err(error),
            }
        }
        return Err(Error::new(ErrorCode::Internal, "generation ended without a result").into());
    }
    let state = StreamState {
        tool_mode,
        handle,
        flavor,
        id,
        model,
        created,
        include_usage,
        pending: std::collections::VecDeque::new(),
        started: false,
        terminal: false,
    };
    Ok(Sse::new(sse_stream(state))
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(10)))
        .into_response())
}
struct StreamState {
    tool_mode: bool,
    handle: GenerationHandle,
    flavor: Flavor,
    id: String,
    model: String,
    created: u64,
    include_usage: bool,
    pending: std::collections::VecDeque<Event>,
    started: bool,
    terminal: bool,
}
fn payload_event(mut value: Value, state: &StreamState) -> Event {
    if state.include_usage && state.flavor != Flavor::Native {
        value["usage"] = Value::Null;
    }
    event(value)
}
fn sse_stream(
    state: StreamState,
) -> impl Stream<Item = std::result::Result<Event, Infallible>> + Send {
    stream::unfold(state, |mut state| async move {
        {
            if let Some(event) = state.pending.pop_front() {
                return Some((Ok(event), state));
            }
            if state.terminal {
                return None;
            }
            if !state.started {
                state.started = true;
                if state.flavor == Flavor::Chat {
                    let value = chunk(
                        state.flavor,
                        &state.id,
                        &state.model,
                        state.created,
                        json!([{"index":0,"delta":if state.tool_mode{json!({"role":"assistant"})}else{json!({"role":"assistant","content":""})},"logprobs":null,"finish_reason":null}]),
                    );
                    return Some((Ok(payload_event(value, &state)), state));
                }
            }
            match state.handle.next().await {
                Some(GenerationEvent::Token(token)) => {
                    let value = match state.flavor {
                        Flavor::Native => json!({"token":token}),
                        Flavor::Chat => chunk(
                            state.flavor,
                            &state.id,
                            &state.model,
                            state.created,
                            json!([{"index":0,"delta":{"content":token.text},"logprobs":null,"finish_reason":null}]),
                        ),
                        Flavor::Completion => chunk(
                            state.flavor,
                            &state.id,
                            &state.model,
                            state.created,
                            json!([{"index":0,"text":token.text,"logprobs":null,"finish_reason":null}]),
                        ),
                    };
                    Some((Ok(payload_event(value, &state)), state))
                }
                Some(GenerationEvent::ToolDelta(delta)) => {
                    let value = if state.flavor == Flavor::Native {
                        json!({"tool_call":delta})
                    } else {
                        chunk(
                            state.flavor,
                            &state.id,
                            &state.model,
                            state.created,
                            json!([{"index":0,"delta":{"tool_calls":[delta]},"logprobs":null,"finish_reason":null}]),
                        )
                    };
                    Some((Ok(payload_event(value, &state)), state))
                }
                Some(GenerationEvent::Finished(result)) => {
                    let value = if state.flavor == Flavor::Native {
                        json!({"result":result})
                    } else {
                        let choices = if state.flavor == Flavor::Chat {
                            json!([{"index":0,"delta":{},"logprobs":null,"finish_reason":reason(result.finish_reason)}])
                        } else {
                            json!([{"index":0,"text":"","logprobs":null,"finish_reason":reason(result.finish_reason)}])
                        };
                        let mut finish = chunk(
                            state.flavor,
                            &state.id,
                            &state.model,
                            state.created,
                            choices,
                        );
                        finish["coheryn"] = json!({"vision":result.vision,"sampling_profile":if result.vision.is_some(){None}else{Some(coheryn_engine::sampling::PROFILE)},"sampling":result.sampling_metadata,"constraint_validation":result.constraint_validation,"batching":result.batching});
                        if state.include_usage {
                            let mut u = chunk(
                                state.flavor,
                                &state.id,
                                &state.model,
                                state.created,
                                json!([]),
                            );
                            u["usage"] = usage(&result);
                            state.pending.push_back(event(u));
                        }
                        finish
                    };
                    state.pending.push_back(Event::default().data("[DONE]"));
                    state.terminal = true;
                    Some((Ok(payload_event(value, &state)), state))
                }
                Some(GenerationEvent::Failed(error)) => {
                    state.pending.push_back(Event::default().data("[DONE]"));
                    state.terminal = true;
                    Some((Ok(event(error.body())), state))
                }
                None => {
                    state.pending.push_back(Event::default().data("[DONE]"));
                    state.terminal = true;
                    Some((
                        Ok(event(
                            ApiError::from(Error::new(
                                ErrorCode::Internal,
                                "generation ended without a result",
                            ))
                            .body(),
                        )),
                        state,
                    ))
                }
            }
        }
    })
}
pub async fn serve(listener: tokio::net::TcpListener, service: Arc<Service>) -> Result<()> {
    let shutdown = service.shutdown.clone();
    let server = axum::serve(listener, router(service.clone()))
        .with_graceful_shutdown(async move { shutdown.cancelled().await });
    let server = server.into_future();
    tokio::pin!(server);
    tokio::select! {
        result=&mut server=>result?,
        _=service.shutdown.cancelled()=>{let _=tokio::time::timeout(Duration::from_secs(35),&mut server).await;}
    }
    service.scheduler.drained().await;
    Ok(())
}

#[cfg(test)]
mod tool_protocol_tests {
    use super::*;
    fn fixture() -> (GenerationResult, Vec<GenerationEvent>) {
        let v: Value = serde_json::from_str(include_str!(
            "../../../fixtures/tools/qwen2.5-reference.json"
        ))
        .unwrap();
        let case = &v["cases"][3];
        let config = serde_json::from_value(json!({"tools":case["tools"]})).unwrap();
        let mut parser =
            coheryn_engine::tools::ToolParser::new(config, "call_fixture".into()).unwrap();
        let mut events = vec![];
        for ch in case["generated_text"].as_str().unwrap().chars() {
            for event in parser.push(&ch.to_string()).unwrap() {
                match event {
                    coheryn_engine::tools::ParsedEvent::Text(text) => {
                        events.push(GenerationEvent::Token(coheryn_engine::TokenEvent {
                            token_id: None,
                            text,
                        }))
                    }
                    coheryn_engine::tools::ParsedEvent::Delta(delta) => {
                        events.push(GenerationEvent::ToolDelta(delta))
                    }
                }
            }
        }
        let (text, calls, tail) = parser.finish(FinishReason::Stop).unwrap();
        if !tail.is_empty() {
            events.push(GenerationEvent::Token(coheryn_engine::TokenEvent {
                token_id: None,
                text: tail,
            }));
        }
        let result:GenerationResult=serde_json::from_value(json!({"text":text,"tool_calls":calls,"token_ids":[0],"prompt_tokens":12,"completion_tokens":1,"finish_reason":"tool_calls","prefill_ms":0,"tokenization_ms":0,"decode_ms":0,"output_delivery_ms":0,"first_token_ms":null,"memory":{"limit":1024,"reserved":0,"peak_reserved":0}})).unwrap();
        (result, events)
    }
    async fn body(events: Vec<GenerationEvent>) -> String {
        let state = StreamState {
            tool_mode: true,
            handle: crate::scheduler::protocol_fixture(events),
            flavor: Flavor::Chat,
            id: "chat-test".into(),
            model: "test".into(),
            created: 0,
            include_usage: true,
            pending: Default::default(),
            started: false,
            terminal: false,
        };
        let response = Sse::new(sse_stream(state)).into_response();
        String::from_utf8(
            to_bytes(response.into_body(), 1 << 20)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap()
    }
    #[tokio::test]
    async fn multiple_tool_calls_have_complete_streaming_parity_usage_and_done() {
        let (result, mut events) = fixture();
        assert_eq!(result.tool_calls.len(), 2);
        events.push(GenerationEvent::Finished(result.clone()));
        let text = body(events).await;
        assert!(text.ends_with("data: [DONE]\n\n"));
        let values: Vec<Value> = text
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .filter(|l| *l != "[DONE]")
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        let complete = response_value(Flavor::Chat, "chat-test", "test", 0, result.clone());
        assert_eq!(complete["choices"][0]["finish_reason"], "tool_calls");
        let mut calls =
            vec![json!({"id":"","type":"function","function":{"name":"","arguments":""}}); 2];
        let mut content = String::new();
        for value in &values {
            if let Some(s) = value["choices"][0]["delta"]["content"].as_str() {
                content.push_str(s);
            }
            if let Some(deltas) = value["choices"][0]["delta"]["tool_calls"].as_array() {
                for d in deltas {
                    let call = &mut calls[d["index"].as_u64().unwrap() as usize];
                    if d.get("id").is_some() {
                        call["id"] = d["id"].clone();
                        call["function"]["name"] = d["function"]["name"].clone();
                    }
                    let args = format!(
                        "{}{}",
                        call["function"]["arguments"].as_str().unwrap(),
                        d["function"]["arguments"].as_str().unwrap()
                    );
                    call["function"]["arguments"] = json!(args);
                }
            }
        }
        assert_eq!(
            json!(calls),
            complete["choices"][0]["message"]["tool_calls"]
        );
        assert_eq!(content, result.text);
        assert!(
            values
                .iter()
                .any(|v| v["choices"][0]["finish_reason"] == "tool_calls")
        );
        assert_eq!(values.last().unwrap()["usage"]["total_tokens"], 13);
    }
    #[tokio::test]
    async fn tool_only_stream_preserves_absent_content_and_matches_complete_null() {
        let (mut result, mut events) = fixture();
        result.text.clear();
        events.retain(|e| matches!(e, GenerationEvent::ToolDelta(_)));
        let complete = response_value(Flavor::Chat, "test", "test", 0, result.clone());
        assert!(complete["choices"][0]["message"]["content"].is_null());
        events.push(GenerationEvent::Finished(result));
        let text = body(events).await;
        for line in text
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .filter(|l| *l != "[DONE]")
        {
            let value: Value = serde_json::from_str(line).unwrap();
            assert!(value["choices"][0]["delta"].get("content").is_none());
        }
    }
    #[tokio::test]
    async fn partial_tool_stream_terminates_with_error_and_never_claims_success() {
        let (_, mut events) = fixture();
        events.truncate(1);
        events.push(GenerationEvent::Failed(
            Error::new(ErrorCode::Cancelled, "cancelled").into(),
        ));
        let text = body(events).await;
        assert!(text.contains("CANCELLED"));
        assert!(!text.contains("\"finish_reason\":\"tool_calls\""));
        assert!(text.ends_with("data: [DONE]\n\n"));
    }
}

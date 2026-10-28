use crate::{
    error::{ApiError, Result},
    http::{json_body, json_body_limit},
    scheduler::GenerationEvent,
    service::Service,
    session_store::{DOCUMENT_BYTES, ExportDocument, TRANSFER_BYTES},
    sessions::{BeginTurn, CreateSession, ResetRequest, RevisionRequest, SubmitTurn, TurnWatch},
};
use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::{Path, Query, Request, State},
    response::{
        IntoResponse, Response, Sse,
        sse::{Event, KeepAlive},
    },
    routing::{get, post},
};
use coheryn_engine::{Cancellation, Error, ErrorCode, memory::Reservation};
use futures_util::stream;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{convert::Infallible, sync::Arc, time::Duration};

pub(crate) fn routes() -> Router<Arc<Service>> {
    Router::new()
        .route("/coheryn/v1/sessions", get(list).post(create))
        .route("/coheryn/v1/sessions/import", post(import))
        .route("/coheryn/v1/sessions/{id}/export", get(export))
        .route("/coheryn/v1/sessions/{id}", get(show).delete(delete))
        .route("/coheryn/v1/sessions/{id}/reset", post(reset))
        .route("/coheryn/v1/sessions/{id}/save", post(save))
        .route("/coheryn/v1/sessions/{id}/messages", post(messages))
        .route("/coheryn/v1/sessions/{id}/images", post(attach_image))
}
async fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<T> {
    tokio::task::spawn_blocking(work).await?
}
async fn reserved<T: Send + 'static>(
    reservation: Reservation,
    work: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<(T, Reservation)> {
    // The detached worker must retain its allocation credit after a disconnected
    // handler is dropped. Return it with the value through response serialization.
    blocking(move || Ok((work()?, reservation))).await
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ImportRequest {
    document: ExportDocument,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    persist: bool,
}
async fn export(
    State(s): State<Arc<Service>>,
    Path(id): Path<String>,
    request: Request,
) -> Result<Response> {
    let owner = s.auth.principal(request.headers())?;
    let _operation = s.budget.reserve((DOCUMENT_BYTES * 4 + 65536) as u64)?;
    let sessions = s.sessions.clone();
    let (document, _operation) = reserved(_operation, move || sessions.export(&owner, &id)).await?;
    reply(&s, document)
}
async fn import(State(s): State<Arc<Service>>, request: Request) -> Result<Response> {
    let owner = s.auth.principal(request.headers())?;
    let _operation = s.budget.reserve((TRANSFER_BYTES * 4) as u64)?;
    let body: ImportRequest = json_body_limit(request, TRANSFER_BYTES).await?;
    let sessions = s.sessions.clone();
    let (view, _operation) = reserved(_operation, move || {
        sessions.import(owner, body.document, body.name, body.persist)
    })
    .await?;
    reply(&s, view)
}
struct ResponseOwner {
    bytes: Vec<u8>,
    _reservation: Reservation,
}
struct PrepareCancellation(Cancellation);
impl Drop for PrepareCancellation {
    fn drop(&mut self) {
        self.0.cancel();
    }
}
impl AsRef<[u8]> for ResponseOwner {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}
fn response_reservation(s: &Service) -> Result<Reservation> {
    // Covers worst-case escaped history, receipt token IDs and serialization overlap.
    Ok(s.budget.reserve(
        65536 + s.config.sessions.history_bytes as u64 * 8 + s.config.max_context as u64 * 16,
    )?)
}
fn reply(s: &Service, value: impl Serialize) -> Result<Response> {
    let reservation = response_reservation(s)?;
    let bytes = serde_json::to_vec(&value)
        .map_err(|_| Error::new(ErrorCode::Internal, "session response serialization failed"))?;
    let bytes = Bytes::from_owner(ResponseOwner {
        bytes,
        _reservation: reservation,
    });
    Ok((
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        Body::from(bytes),
    )
        .into_response())
}
async fn create(State(s): State<Arc<Service>>, request: Request) -> Result<Response> {
    let owner = s.auth.principal(request.headers())?;
    let _payload = s.budget.reserve(4 << 20)?;
    let body: CreateSession = json_body(request).await?;
    let task = if s.is_vision(&body.model)? {
        let model = s.vision(&body.model)?;
        if body.context == 0
            || body.context > model.context_limit()
            || body.context > s.config.max_context
        {
            return Err(ApiError::bad(
                "session context exceeds loaded model/server capacity",
            ));
        }
        let mut messages = Vec::new();
        if let Some(system) = &body.system {
            messages.push(coheryn_engine::multimodal::qwen2vl::ChatMessage {
                role: "system".into(),
                content: coheryn_engine::multimodal::qwen2vl::ChatContent::Text(system.clone()),
            });
        }
        messages.push(coheryn_engine::multimodal::qwen2vl::ChatMessage {
            role: "user".into(),
            content: coheryn_engine::multimodal::qwen2vl::ChatContent::Text(" ".into()),
        });
        coheryn_engine::multimodal::qwen2vl::render(&messages, 0)?;
        crate::sessions::SessionTask::ImageToText
    } else {
        let model = s.model(&body.model)?;
        if !model.tokenizer.supports_conversations() {
            return Err(Error::new(
                ErrorCode::UnsupportedChatTemplate,
                "template is not qualified for managed multi-turn histories",
            )
            .into());
        }
        if body.context == 0
            || body.context > model.context_limit()
            || body.context > s.config.max_context
        {
            return Err(ApiError::bad(
                "session context exceeds loaded model/server capacity",
            ));
        }
        if let Some(system) = &body.system {
            let prompt = model.render(&[coheryn_engine::tokenizer::Message {
                role: "system".into(),
                content: system.clone(),
                ..Default::default()
            }])?;
            if model.tokenizer.encode(&prompt)?.len() >= body.context {
                return Err(ApiError::bad(
                    "initial instructions cannot fit the session context",
                ));
            }
        }
        crate::sessions::SessionTask::TextGeneration
    };
    let hash = s.registry.get(&body.model)?.sha256;
    let sessions = s.sessions.clone();
    let (view, _payload) = reserved(_payload, move || {
        sessions.create_task(owner, body, hash, task)
    })
    .await?;
    reply(&s, view)
}
async fn list(State(s): State<Arc<Service>>, request: Request) -> Result<Response> {
    let owner = s.auth.principal(request.headers())?;
    let _payload = s.budget.reserve(1 << 20)?;
    let sessions = s.sessions.clone();
    let (list, _payload) = reserved(_payload, move || sessions.list(&owner)).await?;
    reply(
        &s,
        json!({"sessions":list,"persistence_enabled":s.sessions.persistence_enabled()}),
    )
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PageRequest {
    #[serde(default)]
    offset: usize,
    #[serde(default = "page_limit")]
    limit: usize,
}
fn page_limit() -> usize {
    100
}
async fn show(
    State(s): State<Arc<Service>>,
    Path(id): Path<String>,
    request: Request,
) -> Result<Response> {
    let page = Query::<PageRequest>::try_from_uri(request.uri())
        .map_err(|_| ApiError::bad("invalid history pagination"))?
        .0;
    let owner = s.auth.principal(request.headers())?;
    let _payload = response_reservation(&s)?;
    let sessions = s.sessions.clone();
    let (view, _payload) = reserved(_payload, move || {
        sessions.get_page(&owner, &id, page.offset, page.limit)
    })
    .await?;
    reply(&s, view)
}
async fn reset(
    State(s): State<Arc<Service>>,
    Path(id): Path<String>,
    request: Request,
) -> Result<Response> {
    let owner = s.auth.principal(request.headers())?;
    let _payload = s.budget.reserve(4 << 20)?;
    let body: ResetRequest = json_body(request).await?;
    let sessions = s.sessions.clone();
    let (view, _payload) = reserved(_payload, move || {
        sessions.reset_with_instructions(
            &owner,
            &id,
            body.expected_revision,
            body.clear_instructions,
        )
    })
    .await?;
    reply(&s, view)
}
async fn save(
    State(s): State<Arc<Service>>,
    Path(id): Path<String>,
    request: Request,
) -> Result<Response> {
    let owner = s.auth.principal(request.headers())?;
    let _operation = s.budget.reserve(4 << 20)?;
    let request: RevisionRequest = json_body_limit(request, 1024).await?;
    let sessions = s.sessions.clone();
    let (view, _operation) = reserved(_operation, move || {
        sessions.save(&owner, &id, request.expected_revision)
    })
    .await?;
    reply(&s, view)
}
async fn delete(
    State(s): State<Arc<Service>>,
    Path(id): Path<String>,
    request: Request,
) -> Result<Response> {
    let owner = s.auth.principal(request.headers())?;
    // A bounded control-plane teardown must work even when workload reservations
    // are full. Its tiny parser belongs to the fixed HTTP-slot overhead, like unload.
    let body: RevisionRequest = json_body_limit(request, 1024).await?;
    let sessions = s.sessions.clone();
    blocking(move || sessions.delete(&owner, &id, body.expected_revision)).await?;
    Ok(Json(json!({"deleted": true})).into_response())
}
fn wire_event(value: impl Serialize) -> Event {
    Event::default().data(serde_json::to_string(&value).expect("serializable managed event"))
}
fn receipt_reply(s: &Service, watch: &TurnWatch) -> Result<Response> {
    let receipt = watch.snapshot()?;
    if let Some(error) = receipt.error {
        let mut error: ApiError = error.into();
        error.details = Some(
            json!({"session_id":receipt.session_id,"turn_id":receipt.turn_id,"revision":receipt.revision,"status":receipt.status}),
        );
        return Err(error);
    }
    reply(s, receipt)
}
async fn messages(
    State(s): State<Arc<Service>>,
    Path(id): Path<String>,
    request: Request,
) -> Result<Response> {
    let owner = s.auth.principal(request.headers())?;
    let _payload = s.budget.reserve(coheryn_engine::planner::REQUEST_BYTES)?;
    let request: SubmitTurn = json_body(request).await?;
    let streaming = request.stream;
    let sessions = s.sessions.clone();
    let service = s.clone();
    let (begin, _payload) = reserved(_payload, move || {
        let view = sessions.get_page(&owner, &id, 0, 1)?;
        if !request.tool_results.is_empty() && view.pending_tool_calls.is_empty() {
            return Err(ApiError::bad("session has no pending tool calls"));
        }
        service.require_tool_model(
            &view.model,
            &request.settings.tool_config(),
            !request.tool_results.is_empty(),
        )?;
        sessions.begin(&owner, &id, request)
    })
    .await?;
    let turn = match begin {
        BeginTurn::Replay(watch) => return receipt_reply(&s, &watch),
        BeginTurn::New(turn) => turn,
    };
    let watch = turn.watch.clone();
    if turn.task() == crate::sessions::SessionTask::ImageToText {
        let handle = managed_vision(&s, turn, _payload).await?;
        return managed_response(s, handle, watch, streaming).await;
    }
    let model = match s.model(turn.model_id()).and_then(|model| {
        if s.registry.get(turn.model_id())?.sha256 != turn.model_hash() {
            return Err(ApiError::conflict(
                "session model identity changed; reset does not rebind weights",
            ));
        }
        Ok(model)
    }) {
        Ok(model) => model,
        Err(error) => {
            let mut turn = turn;
            let failure = error.clone();
            blocking(move || turn.finish(&GenerationEvent::Failed(failure))).await?;
            return Err(error);
        }
    };
    // Moving the guard into this task keeps the writer lease if the caller disconnects
    // during bounded template/tokenizer work. Dropping the completed task releases it.
    let shutdown = s.shutdown.clone();
    let preflight_cancel = Cancellation::default();
    let preflight_guard = PrepareCancellation(preflight_cancel.clone());
    let mut preflight = tokio::task::spawn_blocking(move || {
        let mut turn = turn;
        if shutdown.is_cancelled() {
            preflight_cancel.cancel();
        }
        let prepared = turn.prepare(&model, &preflight_cancel);
        (turn, prepared.map(|request| (model, request)))
    });
    let (turn, prepared) = tokio::select! {
        result = &mut preflight => result?,
        _ = s.shutdown.cancelled() => { preflight_guard.0.cancel(); preflight.await? },
    };
    drop(preflight_guard);
    let (model, request) = match prepared {
        Ok(pair) => pair,
        Err(error) => {
            let mut turn = turn;
            let failure = error.clone();
            blocking(move || turn.finish(&GenerationEvent::Failed(failure))).await?;
            return Err(error);
        }
    };
    let scheduler = s.scheduler.clone();
    let handle = blocking(move || scheduler.submit_managed(model, request, turn)).await?;
    managed_response(s, handle, watch, streaming).await
}
async fn managed_response(
    s: Arc<Service>,
    mut handle: crate::scheduler::GenerationHandle,
    watch: crate::sessions::TurnWatch,
    streaming: bool,
) -> Result<Response> {
    if !streaming {
        while let Some(event) = handle.next().await {
            match event {
                GenerationEvent::Token(_) | GenerationEvent::ToolDelta(_) => (),
                GenerationEvent::Finished(_) => return receipt_reply(&s, &watch),
                GenerationEvent::Failed(error) => return Err(error),
            }
        }
        return Err(Error::new(
            ErrorCode::Internal,
            "managed generation ended without a result",
        )
        .into());
    }
    let reservation = response_reservation(&s)?;
    let events = stream::unfold(
        (handle, watch, false, reservation),
        |(mut handle, watch, terminal, reservation)| async move {
            if terminal {
                return None;
            }
            let event = match handle.next().await {
                Some(GenerationEvent::Token(token)) => {
                    return Some((
                        Ok::<_, Infallible>(wire_event(json!({"token": token}))),
                        (handle, watch, false, reservation),
                    ));
                }
                Some(GenerationEvent::ToolDelta(delta)) => {
                    return Some((
                        Ok::<_, Infallible>(wire_event(json!({"tool_call":delta}))),
                        (handle, watch, false, reservation),
                    ));
                }
                Some(GenerationEvent::Finished(_)) => match watch.snapshot() {
                    Ok(receipt) => wire_event(receipt),
                    Err(error) => wire_event(error.body()),
                },
                Some(GenerationEvent::Failed(error)) => wire_event(error.body()),
                None => wire_event(
                    ApiError::from(Error::new(
                        ErrorCode::Internal,
                        "managed generation ended without a result",
                    ))
                    .body(),
                ),
            };
            Some((Ok(event), (handle, watch, true, reservation)))
        },
    );
    let done = stream::once(async { Ok::<_, Infallible>(Event::default().data("[DONE]")) });
    use futures_util::StreamExt;
    Ok(Sse::new(events.chain(done))
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(10)))
        .into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn disconnect_retains_store_worker_credit_until_work_is_done() {
        let budget = coheryn_engine::memory::Budget::new(4096);
        let reservation = budget.reserve(1024).unwrap();
        let (started, ready) = tokio::sync::oneshot::channel();
        let (release, wait) = std::sync::mpsc::channel();
        let request = tokio::spawn(reserved(reservation, move || {
            let _ = started.send(());
            wait.recv().unwrap();
            Ok(())
        }));
        ready.await.unwrap();
        request.abort();
        assert!(matches!(request.await, Err(error) if error.is_cancelled()));
        assert_eq!(budget.stats().reserved, 1024);
        release.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while budget.stats().reserved != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ImageRevision {
    expected_revision: u64,
}
async fn attach_image(
    State(s): State<Arc<Service>>,
    Path(id): Path<String>,
    request: Request,
) -> Result<Response> {
    let owner = s.auth.principal(request.headers())?;
    let axum::extract::Query(query) =
        axum::extract::Query::<ImageRevision>::try_from_uri(request.uri())
            .map_err(|_| ApiError::bad("image upload requires expected_revision"))?;
    if request
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_none_or(|v| !matches!(v, "application/octet-stream" | "image/png" | "image/jpeg"))
    {
        return Err(ApiError::bad(
            "image uploads require application/octet-stream, image/png or image/jpeg",
        ));
    }
    let limits = s.config.vision.images;
    let reservation = s.budget.reserve(limits.max_encoded_bytes * 3 + (1 << 20))?;
    let body = tokio::time::timeout(
        Duration::from_secs(s.config.vision.acquisition_timeout_seconds),
        axum::body::to_bytes(request.into_body(), limits.max_encoded_bytes as usize),
    )
    .await
    .map_err(|_| ApiError::bad("image upload timeout"))?
    .map_err(|_| ApiError::bad("image upload exceeds byte limit or was interrupted"))?;
    let sessions = s.sessions.clone();
    let (attachment, _reservation) = reserved(reservation, move || {
        sessions.attach_image(
            &owner,
            &id,
            query.expected_revision,
            &body,
            limits,
            &Cancellation::default(),
        )
    })
    .await?;
    reply(&s, attachment)
}
async fn managed_vision(
    s: &Arc<Service>,
    mut turn: Box<crate::sessions::TurnLease>,
    payload: Reservation,
) -> Result<crate::scheduler::GenerationHandle> {
    let preflight = (|| -> Result<_> {
        if s.registry.get(turn.model_id())?.sha256 != turn.model_hash() {
            return Err(ApiError::conflict("session model-pair identity changed"));
        }
        let model = s.vision(turn.model_id())?;
        let request = turn.vision_request();
        let options = s.scheduler.vision_options(&model, &request)?;
        Ok((model, request, options))
    })();
    let (model, request, options) = match preflight {
        Ok(v) => v,
        Err(e) => {
            let error = e.clone();
            blocking(move || turn.finish(&GenerationEvent::Failed(error))).await?;
            return Err(e);
        }
    };
    let cancel = Cancellation::default().with_timeout(request.timeout_seconds)?;
    let guard = PrepareCancellation(cancel.clone());
    let engine = model.clone();
    let limits = s.config.vision.images;
    let mut job = tokio::task::spawn_blocking(move || {
        let prepared = turn.prepare_vision(&engine, &options, limits, &cancel);
        (turn, prepared)
    });
    let (turn, prepared) =
        tokio::select! {r=&mut job=>r?,_=s.shutdown.cancelled()=>{guard.0.cancel();job.await?}};
    drop(guard);
    let prepared = match prepared {
        Ok(v) => v,
        Err(e) => {
            let mut turn = turn;
            let error = e.clone();
            blocking(move || turn.finish(&GenerationEvent::Failed(error))).await?;
            return Err(e);
        }
    };
    s.scheduler
        .submit_managed_vision(model, request, prepared, payload, turn)
}

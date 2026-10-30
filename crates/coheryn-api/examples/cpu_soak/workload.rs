use super::{Profile, Result, metrics::Log, models::Sources, require};
use axum::{
    Router,
    body::{Body, to_bytes},
    http::Request,
    response::Response,
};
use coheryn_api::{scheduler::GenerationEvent, service::Service, wire::GenerationRequest};
use coheryn_engine::{Cancellation, ErrorCode, GenerationOptions};
use serde_json::{Value, json};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tower::ServiceExt;

fn request(method: &str, path: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", "application/json")
        .header("authorization", format!("Bearer {}", "i".repeat(64)))
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap()
}
async fn response(
    app: &Router,
    method: &str,
    path: &str,
    body: Value,
    p: &Profile,
) -> Result<Response> {
    Ok(tokio::time::timeout(
        Duration::from_secs(p.request_timeout_seconds),
        app.clone().oneshot(request(method, path, body)),
    )
    .await??)
}
async fn call(
    app: &Router,
    method: &str,
    path: &str,
    body: Value,
    expected: u16,
    p: &Profile,
) -> Result<Value> {
    tokio::time::timeout(Duration::from_secs(p.request_timeout_seconds), async {
        let response = response(app, method, path, body, p).await?;
        let status = response.status().as_u16();
        let bytes = to_bytes(response.into_body(), 4 << 20).await?;
        let value: Value = serde_json::from_slice(&bytes)?;
        require(
            status == expected,
            &format!("{method} {path}: expected {expected}, received {status}"),
        )?;
        Ok(value)
    })
    .await?
}
async fn post(app: &Router, path: &str, body: Value, p: &Profile, log: &Log) -> Result<Value> {
    let task = if path == "/v1/embeddings" {
        "embedding"
    } else if body["model"] == "image" {
        "image"
    } else {
        "text"
    };
    let start = Instant::now();
    let result = call(app, "POST", path, body, 200, p).await;
    log.write(json!({"kind":"operation","task":task,"path":path,"elapsed_ms":start.elapsed().as_secs_f64()*1000.,"outcome":if result.is_ok(){"pass"}else{"fail"}}))?;
    result
}
async fn wait(p: &Profile, mut predicate: impl FnMut() -> bool) -> Result<()> {
    tokio::time::timeout(Duration::from_secs(p.cleanup_timeout_seconds), async {
        while !predicate() {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    })
    .await?;
    Ok(())
}
pub async fn quiescent(s: &Arc<Service>, baseline: u64, p: &Profile) -> Result<()> {
    let result = wait(p, || {
        let stats = s.scheduler.counters.snapshot();
        let memory = s.budget.stats();
        stats.active == 0
            && stats.queued == 0
            && stats.waiting_generation_units == 0
            && stats.vision_waiting_units == 0
            && memory.reserved == baseline
            && memory.session_buffer_capacity_bytes == 0
            && s.http_slots.available_permits() == 64
    })
    .await;
    result.map_err(|_| {
        std::io::Error::other(format!(
            "cleanup did not reach resident baseline {baseline}: {}",
            s.status().unwrap_or(Value::Null)
        ))
        .into()
    })
}
pub async fn idle(s: &Arc<Service>, p: &Profile) -> Result<()> {
    wait(p, || {
        let stats = s.scheduler.counters.snapshot();
        stats.active == 0
            && stats.queued == 0
            && stats.waiting_generation_units == 0
            && stats.vision_waiting_units == 0
            && s.budget.stats().session_buffer_capacity_bytes == 0
            && s.http_slots.available_permits() == 64
            && s.loaded()
                .ok()
                .and_then(|v| {
                    v.as_array()
                        .map(|rows| rows.iter().all(|r| r["pinned"] == false))
                })
                .unwrap_or(false)
    })
    .await
}

fn text(profile: &str) -> Value {
    json!({"model":"text","prompt":"Hello. বাংলা 中文 العربية 🙂","max_tokens":2,"context":512,"threads":1,"profile":profile})
}
fn embedding(profile: &str) -> Value {
    json!({"model":"embed","input":["hello","hello world","hello","hello world","hello","hello world"],"batch_size":2,"threads":1,"profile":profile})
}
fn vision(
    sources: &Sources,
    profile: &str,
    stream: bool,
    count: usize,
    tokens: usize,
) -> Result<Value> {
    let url = coheryn_api::vision::data_url(&sources.image()?, &Cancellation::default())?;
    let mut parts: Vec<_> = (0..count)
        .map(|_| json!({"type":"image_url","image_url":{"url":url}}))
        .collect();
    parts.push(json!({"type":"text","text":"Describe the picture in detail. বাংলা 中文 العربية 🙂"}));
    Ok(
        json!({"model":"image","messages":[{"role":"user","content":parts}],"max_tokens":tokens,"context":512,"threads":1,"profile":profile,"stream":stream}),
    )
}
fn generation_identity(v: &Value) -> Value {
    json!({"text":v["text"],"token_ids":v["token_ids"],"prompt_tokens":v["prompt_tokens"],"finish_reason":v["finish_reason"]})
}
fn embedding_identity(v: &Value) -> Result<Value> {
    let rows = v["data"]
        .as_array()
        .ok_or("embedding result has no vectors")?;
    require(rows.len() == 6, "embedding item count mismatch")?;
    let mut values = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        require(row["index"] == i, "embedding order mismatch")?;
        let vector: Vec<f32> = serde_json::from_value(row["embedding"].clone())?;
        require(
            !vector.is_empty() && vector.iter().all(|v| v.is_finite()),
            "invalid embedding vector",
        )?;
        let norm = vector
            .iter()
            .map(|v| f64::from(*v).powi(2))
            .sum::<f64>()
            .sqrt();
        require((norm - 1.).abs() <= 1e-5, "embedding normalization changed")?;
        values.push(row["embedding"].clone());
        if i >= 2 {
            require(
                values[i] == values[i % 2],
                "duplicate embeddings contaminated",
            )?;
        }
    }
    Ok(json!({"vectors":values,"usage":v["usage"]}))
}
pub async fn mixed(
    app: &Router,
    sources: &Sources,
    profile: &str,
    p: &Profile,
    log: &Log,
) -> Result<Value> {
    // All three requests are live concurrently against the same engine budget and
    // compute scheduler. In-process bodies deliberately exclude TCP buffering.
    let image = vision(sources, profile, false, 2, 2)?;
    let (text, embedded, pictured) = tokio::join!(
        post(app, "/api/v1/generate", text(profile), p, log),
        post(app, "/v1/embeddings", embedding(profile), p, log),
        post(app, "/api/v1/generate", image, p, log),
    );
    let pictured = pictured?;
    require(
        pictured["vision"]["images"] == 2,
        "multiple-image execution was not observed",
    )?;
    Ok(
        json!({"text":generation_identity(&text?),"embedding":embedding_identity(&embedded?)?,"image":generation_identity(&pictured)}),
    )
}
fn long_request() -> GenerationRequest {
    let mut request = GenerationRequest::completion(
        "text".into(),
        "List integers from one to one hundred, one per line:\n1.".into(),
        &GenerationOptions {
            max_tokens: 128,
            ..Default::default()
        },
        512,
        true,
    );
    request.profile = "low-memory".into();
    request.threads = Some(1);
    request
}

async fn saturation(s: &Arc<Service>, app: &Router, p: &Profile, log: &Log) -> Result<()> {
    let start = Instant::now();
    let mut handles = vec![s.submit_generation(long_request(), &mut None)?];
    wait(p, || s.scheduler.counters.snapshot().active == 1).await?;
    for _ in 0..s.config.queue_capacity {
        handles.push(s.submit_generation(long_request(), &mut None)?);
    }
    require(
        s.scheduler.counters.snapshot().queued == s.config.queue_capacity as u64,
        "queue was not saturated",
    )?;
    let rejected = call(app, "POST", "/v1/completions", text("balanced"), 429, p).await?;
    require(
        rejected["error"]["code"] == "QUEUE_FULL",
        "saturation returned the wrong error",
    )?;
    require(
        s.unload("text")
            .is_err_and(|e| e.error.code == ErrorCode::Conflict),
        "active/queued model was unloadable",
    )?;
    drop(handles);
    log.write(json!({"kind":"fault","scenario":"queue_saturation_and_disconnect","held_requests":s.config.queue_capacity+1,"elapsed_ms":start.elapsed().as_secs_f64()*1000.}))
}

async fn slow_consumer(s: &Arc<Service>, p: &Profile, log: &Log) -> Result<()> {
    let start = Instant::now();
    let before = s.scheduler.counters.snapshot();
    let mut handle = s.submit_generation(long_request(), &mut None)?;
    tokio::time::timeout(
        Duration::from_secs(p.request_timeout_seconds),
        s.scheduler.drained(),
    )
    .await?;
    require(
        s.scheduler.counters.snapshot().cancelled > before.cancelled,
        "slow consumer did not trigger cancellation",
    )?;
    let (mut tokens, mut terminal) = (0, false);
    while let Some(event) = tokio::time::timeout(
        Duration::from_secs(p.cleanup_timeout_seconds),
        handle.next(),
    )
    .await?
    {
        match event {
            GenerationEvent::Token(_) | GenerationEvent::ToolDelta(_) => tokens += 1,
            GenerationEvent::Failed(error) => {
                require(
                    error.error.code == ErrorCode::Cancelled,
                    "slow consumer terminal error changed",
                )?;
                terminal = true;
            }
            GenerationEvent::Finished(_) => {
                return Err("model terminated before exercising slow-consumer backpressure".into());
            }
        }
    }
    require(
        tokens == 16 && terminal,
        "bounded delivery/terminal slot contract changed",
    )?;
    log.write(json!({"kind":"fault","scenario":"slow_consumer_timeout","buffered_token_events":tokens,"terminal_error_observed":terminal,"elapsed_ms":start.elapsed().as_secs_f64()*1000.}))
}

async fn image_stall(
    s: &Arc<Service>,
    app: &Router,
    sources: &Sources,
    profile: &str,
    p: &Profile,
    log: &Log,
) -> Result<()> {
    let before = s.scheduler.counters.snapshot();
    let response = response(
        app,
        "POST",
        "/v1/chat/completions",
        vision(sources, profile, true, 1, 128)?,
        p,
    )
    .await?;
    require(
        response.status() == 200,
        "image streaming submission failed",
    )?;
    // Do not read the body: this fills the actual bounded application event queue.
    tokio::time::timeout(Duration::from_secs(p.request_timeout_seconds), async {
        while s.scheduler.counters.snapshot().vision_backpressure_waits
            == before.vision_backpressure_waits
        {
            let stats = s.scheduler.counters.snapshot();
            require(
                stats.completed == before.completed
                    && stats.cancelled == before.cancelled
                    && stats.failed == before.failed,
                "image terminated before exercising backpressure; choose a compatible workload",
            )?;
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    })
    .await??;
    require(
        s.unload("image").is_err(),
        "undelivered image response lost its residency pin",
    )?;
    let start = Instant::now();
    let (text, embedding) = tokio::join!(
        post(app, "/api/v1/generate", text(profile), p, log),
        post(app, "/v1/embeddings", embedding(profile), p, log)
    );
    text?;
    embedding_identity(&embedding?)?;
    require(
        s.scheduler.counters.snapshot().cancelled == before.cancelled,
        "image timed out before peers made progress",
    )?;
    require(
        s.scheduler.counters.snapshot().vision_yields > before.vision_yields,
        "image compute did not yield",
    )?;
    drop(response);
    log.write(json!({"kind":"fault","scenario":"image_backpressure_peer_progress_and_drop","peer_completion_ms":start.elapsed().as_secs_f64()*1000.,"vision_backpressure_observed":true}))
}

pub struct Cycle<'a> {
    pub sources: &'a Sources,
    pub execution_profile: &'a str,
    pub profile: &'a Profile,
    pub expected: &'a Value,
    pub resident_baseline: u64,
    pub history_baseline: u64,
}
pub async fn cycle(
    s: &Arc<Service>,
    app: &Router,
    log: &Log,
    context: &Cycle<'_>,
    faults: bool,
) -> Result<()> {
    let Cycle {
        sources,
        execution_profile: profile,
        profile: p,
        expected,
        resident_baseline: baseline,
        history_baseline,
    } = *context;
    let actual = mixed(app, sources, profile, p, log).await?;
    require(
        &actual == expected,
        "mixed workload output/usage changed after reuse or restart",
    )?;
    quiescent(s, baseline, p).await?;
    if faults {
        saturation(s, app, p, log).await?;
        quiescent(s, baseline, p).await?;
        slow_consumer(s, p, log).await?;
        quiescent(s, baseline, p).await?;
        image_stall(s, app, sources, profile, p, log).await?;
        quiescent(s, baseline, p).await?;
        let mut memory = text(profile);
        memory["max_memory"] = json!(1);
        let error = call(app, "POST", "/api/v1/generate", memory, 503, p).await?;
        require(
            error["error"]["code"] == "RESOURCE_EXHAUSTED",
            "resource failure was not explicit",
        )?;
        call(
            app,
            "POST",
            "/v1/embeddings",
            json!({"model":"embed","input":"hello world","max_input_tokens":2}),
            413,
            p,
        )
        .await?;
        quiescent(s, baseline, p).await?;
        for id in sources.ids() {
            s.unload(id)?;
        }
        quiescent(s, history_baseline, p).await?;
        for id in sources.ids() {
            s.load(id.into(), 512).await?;
        }
        require(
            s.budget.stats().reserved == baseline,
            "load/unload cycle changed resident reservation baseline",
        )?;
        log.write(json!({"kind":"fault","scenario":"resource_rejection_and_model_reload","resident_baseline_bytes":baseline}))?;
    }
    quiescent(s, baseline, p).await
}

fn seed_request() -> Value {
    json!({"expected_revision":0,"idempotency_key":"durable-soak-seed","content":"hello","settings":{"max_tokens":1,"threads":1}})
}
pub async fn seed_history(app: &Router, p: &Profile) -> Result<Value> {
    call(
        app,
        "POST",
        "/coheryn/v1/sessions",
        json!({"model":"text","alias":"soak-history","persist":true,"context":512}),
        200,
        p,
    )
    .await?;
    call(
        app,
        "POST",
        "/coheryn/v1/sessions/soak-history/messages",
        seed_request(),
        200,
        p,
    )
    .await
}
pub async fn replay(app: &Router, expected: &Value, p: &Profile) -> Result<()> {
    let actual = call(
        app,
        "POST",
        "/coheryn/v1/sessions/soak-history/messages",
        seed_request(),
        200,
        p,
    )
    .await?;
    require(
        &actual == expected,
        "unloaded restart did not replay the exact durable receipt",
    )
}

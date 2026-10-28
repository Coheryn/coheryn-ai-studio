#[path = "../../coheryn-engine/tests/common/mod.rs"]
mod common;
#[path = "../../coheryn-engine/tests/common/vision.rs"]
mod vision_fixture;
use axum::{
    Router,
    body::{Body, Bytes, to_bytes},
    http::{Request, StatusCode},
};
use coheryn_api::{auth::Credentials, config::ServerConfig, http::router, service::Service};
use futures_util::stream;
use serde_json::{Value, json};
use std::{sync::Arc, time::Duration};
use tower::ServiceExt;
const PNG: &[u8] = include_bytes!("../../../fixtures/vlm/images/minimum.png");
fn config(root: &std::path::Path, persistent: bool) -> ServerConfig {
    ServerConfig {
        data_dir: root.join("service"),
        max_memory: 768 << 20,
        max_context: 1024,
        max_model_bytes: 8 << 20,
        disk_quota: 32 << 20,
        queue_capacity: 3,
        consumer_timeout_seconds: 1,
        generation_timeout_seconds: 15,
        persistence: coheryn_api::session_store::PersistenceLimits {
            enabled: persistent,
            ..Default::default()
        },
        ..Default::default()
    }
}
fn open(c: ServerConfig) -> Arc<Service> {
    Service::open(
        c,
        Credentials {
            inference_token: "i".repeat(64),
            admin_token: "a".repeat(64),
        },
    )
    .unwrap()
}
async fn pair(s: &Arc<Service>) {
    let vision = vision_fixture::component(256);
    for (id, bytes) in [
        ("language", common::tiny_vlm_model()),
        ("vision", std::fs::read(vision.path()).unwrap()),
    ] {
        s.ingest(
            id.into(),
            None,
            None,
            stream::once(async move { Ok::<_, std::io::Error>(Bytes::from(bytes)) }),
        )
        .await
        .unwrap();
    }
    s.register_pair("pair".into(), "language".into(), "vision".into())
        .await
        .unwrap();
    s.load("pair".into(), 512).await.unwrap();
}
fn request(method: &str, path: &str, value: Value) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", "application/json")
        .header("authorization", format!("Bearer {}", "i".repeat(64)))
        .body(Body::from(serde_json::to_vec(&value).unwrap()))
        .unwrap()
}
async fn call(app: &Router, method: &str, path: &str, value: Value) -> (StatusCode, Value) {
    let response = app
        .clone()
        .oneshot(request(method, path, value))
        .await
        .unwrap();
    let status = response.status();
    let b = to_bytes(response.into_body(), 16 << 20).await.unwrap();
    (status, serde_json::from_slice(&b).unwrap())
}
async fn create(app: &Router, persist: bool, overflow: &str) -> Value {
    let (status,v)=call(app,"POST","/coheryn/v1/sessions",json!({"model":"pair","context":512,"persist":persist,"alias":"images","overflow":overflow})).await;
    assert_eq!(status, 200, "{v}");
    assert_eq!(v["task"], "image-to-text");
    v
}
async fn attach(app: &Router, id: &str, revision: u64, bytes: &[u8]) -> Value {
    let r = Request::builder()
        .method("POST")
        .uri(format!(
            "/coheryn/v1/sessions/{id}/images?expected_revision={revision}"
        ))
        .header("authorization", format!("Bearer {}", "i".repeat(64)))
        .header("content-type", "application/octet-stream")
        .body(Body::from(bytes.to_vec()))
        .unwrap();
    let r = app.clone().oneshot(r).await.unwrap();
    let status = r.status();
    let b = to_bytes(r.into_body(), 1 << 20).await.unwrap();
    let v: Value = serde_json::from_slice(&b).unwrap();
    assert_eq!(status, 200, "{v}");
    v
}
fn turn(revision: u64, key: &str, images: Vec<Value>) -> Value {
    json!({"expected_revision":revision,"idempotency_key":key,"content":"Describe. বাংলা 中文 العربية 🙂","images":images,"settings":{"max_tokens":2,"threads":1}})
}
#[tokio::test]
async fn attached_once_followups_persist_restart_and_portable_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let c = config(dir.path(), true);
    let s = open(c.clone());
    pair(&s).await;
    let app = router(s.clone());
    let created = create(&app, true, "reject").await;
    let id = created["id"].as_str().unwrap();
    let a = attach(&app, id, 0, PNG).await;
    let image = a["image"]["id"].clone();
    let duplicate = attach(&app, id, 1, PNG).await;
    assert_eq!(duplicate["revision"], 1);
    assert_eq!(duplicate["image"]["id"], image);
    let request = turn(1, "first", vec![image.clone()]);
    let (status, first) = call(
        &app,
        "POST",
        &format!("/coheryn/v1/sessions/{id}/messages"),
        request.clone(),
    )
    .await;
    assert_eq!(status, 200, "{first}");
    assert_eq!(first["result"]["vision"]["images"], 1);
    let execution = &first["result"]["vision"]["cooperative"];
    assert!(execution["yields"].as_u64().unwrap() > 0);
    assert!(execution["units"].as_u64().unwrap() > execution["yields"].as_u64().unwrap());
    assert!(execution["compute_lease_ms"].as_f64().unwrap() > 0.);
    assert!(execution["waiting_for_compute_ms"].as_f64().unwrap() >= 0.);
    assert_eq!(
        call(
            &app,
            "POST",
            &format!("/coheryn/v1/sessions/{id}/messages"),
            request
        )
        .await
        .1,
        first
    );
    let (status, followup) = call(
        &app,
        "POST",
        &format!("/coheryn/v1/sessions/{id}/messages"),
        turn(3, "followup", vec![]),
    )
    .await;
    assert_eq!(status, 200, "{followup}");
    assert_eq!(followup["result"]["vision"]["images"], 1);
    let (_, before) = call(
        &app,
        "GET",
        &format!("/coheryn/v1/sessions/{id}"),
        Value::Null,
    )
    .await;
    assert_eq!(before["messages"][0]["images"], json!([image]));
    assert_eq!(before["message_count"], 4);
    assert_eq!(before["cached_tokens"], 0);
    let (_, export) = call(
        &app,
        "GET",
        &format!("/coheryn/v1/sessions/{id}/export"),
        Value::Null,
    )
    .await;
    assert_eq!(export["version"], 3);
    assert_eq!(export["assets"].as_array().unwrap().len(), 1);
    s.shutdown.cancel();
    s.scheduler.drained().await;
    drop(app);
    drop(s);
    let s = open(c);
    s.load("pair".into(), 512).await.unwrap();
    let app = router(s.clone());
    let (_, after) = call(
        &app,
        "GET",
        &format!("/coheryn/v1/sessions/{id}"),
        Value::Null,
    )
    .await;
    assert_eq!(after["messages"], before["messages"]);
    let (status, resumed) = call(
        &app,
        "POST",
        &format!("/coheryn/v1/sessions/{id}/messages"),
        turn(5, "resumed", vec![]),
    )
    .await;
    assert_eq!(status, 200, "{resumed}");
    assert_eq!(resumed["result"]["vision"]["images"], 1);
    let (status, imported) = call(
        &app,
        "POST",
        "/coheryn/v1/sessions/import",
        json!({"document":export,"name":"copy","persist":true}),
    )
    .await;
    assert_eq!(status, 200, "{imported}");
    assert_ne!(imported["id"], json!(id));
    let imported_id = imported["id"].as_str().unwrap();
    let (status, result) = call(
        &app,
        "POST",
        &format!("/coheryn/v1/sessions/{imported_id}/messages"),
        turn(5, "resumed", vec![]),
    )
    .await;
    assert_eq!(status, 200, "{result}");
    assert_eq!(
        result["result"]["token_ids"],
        resumed["result"]["token_ids"]
    );
    let (status, _) = call(
        &app,
        "DELETE",
        &format!("/coheryn/v1/sessions/{imported_id}"),
        json!({"expected_revision":7}),
    )
    .await;
    assert_eq!(status, 200);
    let (status, reset) = call(
        &app,
        "POST",
        &format!("/coheryn/v1/sessions/{id}/reset"),
        json!({"expected_revision":7}),
    )
    .await;
    assert_eq!(status, 200, "{reset}");
    assert!(reset.get("image_assets").is_none());
    assert_eq!(
        std::fs::read_dir(dir.path().join("service/image-assets"))
            .unwrap()
            .count(),
        0
    );
}
#[tokio::test]
async fn image_identity_ownership_quotas_context_exclusion_and_mutation_are_enforced() {
    let dir = tempfile::tempdir().unwrap();
    let mut c = config(dir.path(), false);
    c.sessions.images.session_bytes = PNG.len() as u64;
    let s = open(c);
    pair(&s).await;
    let app = router(s.clone());
    let view = create(&app, false, "oldest-complete-turns").await;
    let id = view["id"].as_str().unwrap();
    let a = attach(&app, id, 0, PNG).await;
    let image = a["image"]["id"].clone();
    let bad = turn(
        1,
        "foreign",
        vec![json!("img-00000000000000000000000000000000")],
    );
    assert_eq!(
        call(
            &app,
            "POST",
            &format!("/coheryn/v1/sessions/{id}/messages"),
            bad
        )
        .await
        .0,
        400
    );
    assert!(s.sessions.get(&"x".repeat(64), id).is_err());
    let mut body = turn(1, "first", vec![image.clone()]);
    body["settings"]["max_tokens"] = json!(1);
    assert_eq!(
        call(
            &app,
            "POST",
            &format!("/coheryn/v1/sessions/{id}/messages"),
            body
        )
        .await
        .0,
        200
    );
    // A near-capacity later prompt explicitly evicts the entire first turn and its image.
    let mut body = turn(3, "overflow", vec![]);
    body["content"] = json!("x".repeat(430));
    body["settings"]["max_tokens"] = json!(1);
    let (status, receipt) = call(
        &app,
        "POST",
        &format!("/coheryn/v1/sessions/{id}/messages"),
        body,
    )
    .await;
    assert_eq!(status, 200, "{receipt}");
    assert!(
        receipt["context"]["excluded_message_ids"]
            .as_array()
            .unwrap()
            .len()
            >= 2
    );
    assert_eq!(receipt["result"]["vision"]["images"], 0);
    // Corruption is rejected before executing image-conditioned inference.
    let file = dir
        .path()
        .join("service/image-assets")
        .join(id)
        .join(format!("{}.bin", image.as_str().unwrap()));
    let mut bytes = std::fs::read(&file).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    std::fs::write(file, bytes).unwrap();
    let body = turn(5, "changed", vec![image]);
    let (status, error) = call(
        &app,
        "POST",
        &format!("/coheryn/v1/sessions/{id}/messages"),
        body,
    )
    .await;
    assert_eq!(status, 409, "{error}");
    assert_eq!(error["error"]["code"], "MODEL_CHANGED");
    s.scheduler.drained().await;
    assert_eq!(
        s.sessions
            .get(
                &s.auth
                    .principal(request("GET", "/", Value::Null).headers())
                    .unwrap(),
                id
            )
            .unwrap()
            .active_turn,
        None
    );
}
#[tokio::test]
async fn image_delivery_backpressure_does_not_hold_compute_from_text_or_embeddings() {
    let dir = tempfile::tempdir().unwrap();
    let s = open(config(dir.path(), false));
    pair(&s).await;
    let app = router(s.clone());
    for (id, bytes) in [
        ("text", common::tiny_model()),
        ("embed", common::embedding_model(0)),
    ] {
        s.ingest(
            id.into(),
            None,
            None,
            stream::once(async move { Ok::<_, std::io::Error>(Bytes::from(bytes)) }),
        )
        .await
        .unwrap();
        s.load(id.into(), 128).await.unwrap();
    }
    let view = create(&app, false, "reject").await;
    let id = view["id"].as_str().unwrap();
    let a = attach(&app, id, 0, PNG).await;
    let mut body = turn(1, "blocked", vec![a["image"]["id"].clone()]);
    body["stream"] = json!(true);
    body["settings"]["max_tokens"] = json!(128);
    let response = app
        .clone()
        .oneshot(request(
            "POST",
            &format!("/coheryn/v1/sessions/{id}/messages"),
            body,
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    tokio::time::timeout(Duration::from_secs(10), async {
        while s.scheduler.counters.snapshot().vision_backpressure_waits == 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    let (status, text) = call(
        &app,
        "POST",
        "/v1/completions",
        json!({"model":"text","prompt":"hello","max_tokens":1}),
    )
    .await;
    assert_eq!(status, 200, "{text}");
    let (status, embedding) = call(
        &app,
        "POST",
        "/v1/embeddings",
        json!({"model":"embed","input":"hello"}),
    )
    .await;
    assert_eq!(status, 200, "{embedding}");
    assert!(s.scheduler.counters.snapshot().vision_yields > 0);
    drop(response);
    tokio::time::timeout(Duration::from_secs(5), s.scheduler.drained())
        .await
        .unwrap();
    let (_, view) = call(
        &app,
        "GET",
        &format!("/coheryn/v1/sessions/{id}"),
        Value::Null,
    )
    .await;
    assert!(view["active_turn"].is_null());
    assert_eq!(view["messages"][0]["status"], "cancelled");
}

#[tokio::test]
async fn text_and_embedding_work_progress_during_image_prefill_with_one_generation_slot() {
    let dir = tempfile::tempdir().unwrap();
    let s = open(config(dir.path(), false));
    pair(&s).await;
    for (id, bytes) in [
        ("text", common::tiny_model()),
        ("embed", common::embedding_model(0)),
    ] {
        s.ingest(
            id.into(),
            None,
            None,
            stream::once(async move { Ok::<_, std::io::Error>(Bytes::from(bytes)) }),
        )
        .await
        .unwrap();
        s.load(id.into(), 128).await.unwrap();
    }
    let app = router(s.clone());
    let view = create(&app, false, "reject").await;
    let id = view["id"].as_str().unwrap();
    let a = attach(&app, id, 0, PNG).await;
    let mut body = turn(1, "prefill", vec![a["image"]["id"].clone()]);
    body["content"] = json!("x".repeat(380));
    body["stream"] = json!(true);
    let response = app
        .clone()
        .oneshot(request(
            "POST",
            &format!("/coheryn/v1/sessions/{id}/messages"),
            body,
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    tokio::time::timeout(Duration::from_secs(5), async {
        while s.scheduler.counters.snapshot().vision_units < 20 {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
    let results = tokio::join!(
        call(
            &app,
            "POST",
            "/v1/completions",
            json!({"model":"text","prompt":"hello","max_tokens":1})
        ),
        call(
            &app,
            "POST",
            "/v1/embeddings",
            json!({"model":"embed","input":"hello"})
        )
    );
    assert_eq!(results.0.0, 200, "{}", results.0.1);
    assert_eq!(results.1.0, 200, "{}", results.1.1);
    let stats = s.scheduler.counters.snapshot();
    assert_eq!(
        stats.completed, 2,
        "image prefill must still be live when both other tasks finish"
    );
    assert!(stats.vision_yields > 0);
    drop(response);
    tokio::time::timeout(Duration::from_secs(5), s.scheduler.drained())
        .await
        .unwrap();
    assert_eq!(s.scheduler.counters.snapshot().vision_waiting_units, 0);
}

#[tokio::test]
async fn asset_quotas_revision_checks_and_orphan_recovery_are_bounded() {
    let dir = tempfile::tempdir().unwrap();
    let mut c = config(dir.path(), false);
    c.sessions.images.session_items = 1;
    c.sessions.images.session_bytes = PNG.len() as u64;
    let s = open(c.clone());
    pair(&s).await;
    let app = router(s.clone());
    let view = create(&app, false, "reject").await;
    let id = view["id"].as_str().unwrap();
    let a = attach(&app, id, 0, PNG).await;
    let owner = s
        .auth
        .principal(request("GET", "/", Value::Null).headers())
        .unwrap();
    let err = s
        .sessions
        .attach_image(
            &owner,
            id,
            0,
            PNG,
            Default::default(),
            &coheryn_engine::Cancellation::default(),
        )
        .err()
        .unwrap();
    assert_eq!(err.error.code, coheryn_engine::ErrorCode::Conflict);
    let other = include_bytes!("../../../fixtures/vlm/images/landscape.png");
    let e = s
        .sessions
        .attach_image(
            &owner,
            id,
            1,
            other,
            Default::default(),
            &coheryn_engine::Cancellation::default(),
        )
        .err()
        .unwrap();
    assert_eq!(e.error.code, coheryn_engine::ErrorCode::ResourceExhausted);
    assert_eq!(s.sessions.image_stats().unwrap()["assets"], 1);
    let (_, doc) = call(
        &app,
        "GET",
        &format!("/coheryn/v1/sessions/{id}/export"),
        Value::Null,
    )
    .await;
    let mut forged = doc.clone();
    forged["session"]["image_assets"][0]["info"]["width"] = json!(56);
    // Change a descriptor to a different valid geometry while preserving original-byte identity.
    forged["session"]["image_assets"][0]["info"]["height"] = json!(84);
    let (status, _) = call(
        &app,
        "POST",
        "/coheryn/v1/sessions/import",
        json!({"document":forged,"name":"forged"}),
    )
    .await;
    assert_eq!(status, 400);
    assert_eq!(s.sessions.image_stats().unwrap()["assets"], 1);
    assert!(a["image"]["sha256"].as_str().is_some());
    s.shutdown.cancel();
    s.scheduler.drained().await;
    drop(app);
    drop(s);
    let orphan = dir.path().join("service/image-assets/abandoned");
    coheryn_api::auth::private_directory(&orphan).unwrap();
    let mut f = coheryn_api::auth::private_file(&orphan.join("unfinished.part")).unwrap();
    std::io::Write::write_all(&mut f, PNG).unwrap();
    drop(f);
    let s = open(c);
    assert_eq!(
        s.sessions.image_stats().unwrap()["disk_bytes"],
        0,
        "ephemeral assets are swept after restart"
    );
}

#[tokio::test]
async fn image_only_turns_keep_valid_portable_histories_and_failed_imports_release_capacity() {
    let dir = tempfile::tempdir().unwrap();
    let s = open(config(dir.path(), true));
    pair(&s).await;
    let app = router(s.clone());
    let view = create(&app, true, "reject").await;
    let id = view["id"].as_str().unwrap();
    let a = attach(&app, id, 0, PNG).await;
    let mut body = turn(1, "image-only", vec![a["image"]["id"].clone()]);
    body["content"] = json!("");
    let (status, result) = call(
        &app,
        "POST",
        &format!("/coheryn/v1/sessions/{id}/messages"),
        body,
    )
    .await;
    assert_eq!(status, 200, "{result}");
    let (_, doc) = call(
        &app,
        "GET",
        &format!("/coheryn/v1/sessions/{id}/export"),
        Value::Null,
    )
    .await;
    let baseline = s.budget.stats().reserved;
    let mut bad = doc.clone();
    bad["assets"][0]["base64"] = json!("AAAA");
    assert_eq!(
        call(
            &app,
            "POST",
            "/coheryn/v1/sessions/import",
            json!({"document":bad,"name":"bad"})
        )
        .await
        .0,
        400
    );
    assert_eq!(s.budget.stats().reserved, baseline);
    let (status, imported) = call(
        &app,
        "POST",
        "/coheryn/v1/sessions/import",
        json!({"document":doc,"name":"image-only-copy"}),
    )
    .await;
    assert_eq!(status, 200, "{imported}");
    let copied = imported["id"].as_str().unwrap();
    let (status, reuse) = call(
        &app,
        "POST",
        &format!("/coheryn/v1/sessions/{copied}/messages"),
        turn(3, "followup", vec![]),
    )
    .await;
    assert_eq!(status, 200, "{reuse}");
    assert_eq!(reuse["result"]["vision"]["images"], 1);
}

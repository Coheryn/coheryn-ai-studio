#[path = "../../coheryn-engine/tests/common/mod.rs"]
mod common;
use axum::{
    Router,
    body::{Body, Bytes, to_bytes},
    http::{HeaderMap, Request, StatusCode},
};
use coheryn_api::{
    auth::Credentials, config::ServerConfig, embeddings::EmbeddingRequest, http::router,
    service::Service, wire::GenerationRequest,
};
use coheryn_engine::{
    Cancellation, ErrorCode, GenerationOptions, ResourceLimits,
    embedding::{EmbeddingModel, EmbeddingOptions},
};
use serde_json::{Value, json};
use std::{sync::Arc, time::Duration};
use tower::ServiceExt;

fn credentials() -> Credentials {
    Credentials {
        inference_token: "i".repeat(64),
        admin_token: "a".repeat(64),
    }
}
fn setup(change: impl FnOnce(&mut ServerConfig)) -> (tempfile::TempDir, Arc<Service>, Router) {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = ServerConfig {
        data_dir: dir.path().join("service"),
        max_memory: 256 << 20,
        max_model_bytes: 1 << 20,
        disk_quota: 4 << 20,
        queue_timeout_seconds: 2,
        consumer_timeout_seconds: 2,
        embedding_timeout_seconds: 10,
        ..Default::default()
    };
    change(&mut cfg);
    let s = Service::open(cfg, credentials()).unwrap();
    let app = router(s.clone());
    (dir, s, app)
}
async fn encoder(s: &Arc<Service>, dtype: u32) {
    s.ingest(
        "encoder".into(),
        None,
        None,
        futures_util::stream::iter([Ok::<_, std::io::Error>(Bytes::from(
            common::embedding_model(dtype),
        ))]),
    )
    .await
    .unwrap();
    s.load("encoder".into(), 512).await.unwrap();
}
fn request(path: &str, value: Value, token: Option<&str>) -> Request<Body> {
    let bytes = serde_json::to_vec(&value).unwrap();
    let mut req = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .header("content-length", bytes.len());
    if let Some(t) = token {
        req = req.header("authorization", format!("Bearer {}", t.repeat(64)));
    }
    req.body(Body::from(bytes)).unwrap()
}
async fn call(app: &Router, path: &str, value: Value) -> (StatusCode, HeaderMap, Value) {
    let r = app
        .clone()
        .oneshot(request(path, value, Some("i")))
        .await
        .unwrap();
    let status = r.status();
    let headers = r.headers().clone();
    let body = to_bytes(r.into_body(), 16 << 20).await.unwrap();
    let value = serde_json::from_slice(&body).unwrap();
    (status, headers, value)
}
async fn wait_for(mut f: impl FnMut() -> bool) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while !f() {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn native_float_base64_dimensions_order_usage_and_independent_embedding_only_residency() {
    for dtype in [0, 1, 8] {
        let (_dir, s, app) = setup(|_| {});
        encoder(&s, dtype).await;
        assert_eq!(s.loaded().unwrap()[0]["task"], "embedding");
        assert!(s.model("encoder").is_err());
        let native = EmbeddingModel::load(
            s.registry.model_path("encoder").unwrap(),
            ResourceLimits {
                context: 512,
                max_memory: 128 << 20,
            },
        )
        .unwrap();
        let expected = native
            .embed(
                &["hello", "hello world", "hello"],
                &EmbeddingOptions::default(),
                &Cancellation::default(),
            )
            .unwrap();
        let baseline = s.budget.stats().reserved;
        let (status, headers, value) = call(
            &app,
            "/v1/embeddings",
            json!({"model":"encoder","input":["hello","hello world","hello"],"dimensions":32}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(value["object"], "list");
        assert_eq!(value["usage"]["total_tokens"], 10);
        assert_eq!(headers["x-coheryn-embedding-dimension"], "32");
        let identity = headers["x-coheryn-embedding-space"].to_str().unwrap();
        for (index, item) in value["data"].as_array().unwrap().iter().enumerate() {
            assert_eq!(item["index"], index);
            assert_eq!(item["object"], "embedding");
            let actual: Vec<f32> = serde_json::from_value(item["embedding"].clone()).unwrap();
            assert_eq!(actual, expected.data[index].embedding);
        }
        let (status, encoded_headers, encoded) = call(
            &app,
            "/v1/embeddings",
            json!({"model":"encoder","input":"hello","encoding_format":"base64"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(encoded_headers["x-coheryn-embedding-space"], identity);
        let vector = coheryn_api::embeddings::Vector::Base64(
            encoded["data"][0]["embedding"].as_str().unwrap().into(),
        )
        .floats()
        .unwrap();
        assert_eq!(vector, expected.data[0].embedding);
        s.scheduler.drained().await;
        assert_eq!(s.budget.stats().reserved, baseline);
        assert_eq!(s.budget.stats().session_buffer_capacity_bytes, 0);
    }
}

#[tokio::test]
async fn response_bytes_own_the_reservation_and_model_pin_through_last_chunk() {
    let (_dir, s, app) = setup(|_| {});
    encoder(&s, 1).await;
    let baseline = s.budget.stats().reserved;
    let r = app
        .oneshot(request(
            "/v1/embeddings",
            json!({"model":"encoder","input":"hello"}),
            Some("i"),
        ))
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::OK);
    s.scheduler.drained().await;
    assert!(s.budget.stats().reserved > baseline);
    assert_eq!(
        s.unload("encoder").unwrap_err().status,
        StatusCode::CONFLICT
    );
    let bytes = to_bytes(r.into_body(), 1 << 20).await.unwrap();
    assert!(s.budget.stats().reserved > baseline);
    assert!(s.unload("encoder").is_err());
    drop(bytes);
    assert_eq!(s.budget.stats().reserved, baseline);
    s.unload("encoder").unwrap();
    assert_eq!(s.budget.stats().reserved, 0);
}

#[tokio::test]
async fn request_types_fields_empty_input_limits_and_unsupported_dimensions_are_explicit() {
    let (_dir, s, app) = setup(|cfg| {
        cfg.embeddings.items = 2;
        cfg.embeddings.input_bytes = 32;
    });
    encoder(&s, 1).await;
    let baseline = s.budget.stats().reserved;
    for input in [
        json!(""),
        json!([]),
        json!(["hello", 1]),
        json!([101, 102]),
        json!([[101, 102]]),
        Value::Null,
    ] {
        assert_eq!(
            call(
                &app,
                "/v1/embeddings",
                json!({"model":"encoder","input":input})
            )
            .await
            .0,
            StatusCode::BAD_REQUEST
        );
    }
    for input in [json!(["hello", "world", "a"]), json!("a".repeat(33))] {
        assert_eq!(
            call(
                &app,
                "/v1/embeddings",
                json!({"model":"encoder","input":input})
            )
            .await
            .0,
            StatusCode::PAYLOAD_TOO_LARGE
        );
    }
    for extra in [
        json!({"stream":true}),
        json!({"user":"u".repeat(257)}),
        json!({"dimensions":0}),
        json!({"dimensions":16}),
        json!({"threads":17}),
    ] {
        let mut body = json!({"model":"encoder","input":"hello"});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        assert_eq!(
            call(&app, "/v1/embeddings", body).await.0,
            StatusCode::BAD_REQUEST
        );
    }
    assert_eq!(
        call(
            &app,
            "/v1/embeddings",
            json!({"model":"encoder","input":"hello world","max_input_tokens":3})
        )
        .await
        .0,
        StatusCode::PAYLOAD_TOO_LARGE
    );
    let (status, headers, value) = call(
        &app,
        "/v1/embeddings",
        json!({"model":"encoder","input":"hello world","max_input_tokens":3,"truncate":true}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers["x-coheryn-truncated-indices"], "0");
    assert_eq!(value["usage"]["prompt_tokens"], 3);
    s.scheduler.drained().await;
    assert_eq!(s.budget.stats().reserved, baseline);
}

#[tokio::test]
async fn body_limits_authentication_missing_residency_and_wrong_task_are_preserved() {
    let (_dir, s, app) = setup(|cfg| cfg.embeddings.body_bytes = 1024);
    encoder(&s, 1).await;
    let req = json!({"model":"encoder","input":"hello"});
    assert_eq!(
        app.clone()
            .oneshot(request("/v1/embeddings", req.clone(), None))
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        app.clone()
            .oneshot(request("/v1/embeddings", req, Some("a")))
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );
    assert_eq!(
        call(
            &app,
            "/v1/embeddings",
            json!({"model":"encoder","input":"h".repeat(1024)})
        )
        .await
        .0,
        StatusCode::PAYLOAD_TOO_LARGE
    );
    assert_eq!(
        call(
            &app,
            "/v1/embeddings",
            json!({"model":"missing","input":"hello"})
        )
        .await
        .0,
        StatusCode::NOT_FOUND
    );
    s.ingest(
        "generation".into(),
        None,
        None,
        futures_util::stream::iter([Ok::<_, std::io::Error>(Bytes::from(common::tiny_model()))]),
    )
    .await
    .unwrap();
    s.load("generation".into(), 128).await.unwrap();
    assert_eq!(
        call(
            &app,
            "/v1/embeddings",
            json!({"model":"generation","input":"hello"})
        )
        .await
        .0,
        StatusCode::BAD_REQUEST
    );
    s.scheduler.drained().await;
    s.unload("encoder").unwrap();
    assert_eq!(
        call(
            &app,
            "/v1/embeddings",
            json!({"model":"encoder","input":"hello"})
        )
        .await
        .0,
        StatusCode::SERVICE_UNAVAILABLE
    );
}

#[tokio::test]
async fn total_tokens_response_policy_and_explicit_memory_caps_fail_without_partial_results() {
    let (_dir, s, app) = setup(|cfg| cfg.embeddings.total_tokens = 6);
    encoder(&s, 1).await;
    let baseline = s.budget.stats().reserved;
    let (status, _, body) = call(
        &app,
        "/v1/embeddings",
        json!({"model":"encoder","input":["hello world","hello world"]}),
    )
    .await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert!(body.get("data").is_none());
    assert_eq!(
        call(
            &app,
            "/v1/embeddings",
            json!({"model":"encoder","input":"hello","max_memory":1})
        )
        .await
        .0,
        StatusCode::SERVICE_UNAVAILABLE
    );
    s.scheduler.drained().await;
    assert_eq!(s.budget.stats().reserved, baseline);
    assert_eq!(s.budget.stats().session_buffer_capacity_bytes, 0);
    let (_dir, s, app) = setup(|cfg| cfg.embeddings.response_bytes = 1024);
    encoder(&s, 1).await;
    assert_eq!(
        call(
            &app,
            "/v1/embeddings",
            json!({"model":"encoder","input":"hello"})
        )
        .await
        .0,
        StatusCode::PAYLOAD_TOO_LARGE
    );
}

#[tokio::test]
async fn metadata_plans_do_not_load_weights_and_expose_stable_space_identity() {
    let (_dir, s, app) = setup(|_| {});
    s.ingest(
        "encoder".into(),
        None,
        None,
        futures_util::stream::iter([Ok::<_, std::io::Error>(Bytes::from(
            common::embedding_model(8),
        ))]),
    )
    .await
    .unwrap();
    let response = app
        .oneshot(request(
            "/api/v1/plans/estimate",
            json!({"model":"encoder","task":"embedding","context":512,"input_bytes":[5,20]}),
            Some("a"),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    let plan: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(plan["task"], "embedding");
    assert_eq!(plan["dimension"], 32);
    assert_eq!(plan["admissible"], true);
    assert!(plan["model_metadata_additional"].as_u64().unwrap() > 0);
    assert!(s.loaded().unwrap().as_array().unwrap().is_empty());
    let inspection = s.inspect("encoder".into(), true).await.unwrap();
    assert_eq!(inspection["embedding_space"], plan["embedding_space"]);
}

#[tokio::test]
async fn yielding_microbatches_allow_queued_generation_to_execute_and_cancellation_reclaims_both() {
    let (_dir, s, _app) = setup(|_| {});
    encoder(&s, 1).await;
    s.ingest(
        "generation".into(),
        None,
        None,
        futures_util::stream::iter([Ok::<_, std::io::Error>(Bytes::from(common::tiny_model()))]),
    )
    .await
    .unwrap();
    s.load("generation".into(), 128).await.unwrap();
    let baseline = s.budget.stats().reserved;
    let options = EmbeddingOptions {
        batch_size: 1,
        max_input_tokens: 128,
        ..Default::default()
    };
    let request = EmbeddingRequest::new("encoder".into(), vec!["a".repeat(100); 32], &options);
    let mut handle = s
        .scheduler
        .submit_embedding(
            s.embedding("encoder").unwrap(),
            request,
            s.budget.reserve(0).unwrap(),
        )
        .unwrap();
    wait_for(|| s.scheduler.counters.snapshot().active == 1).await;
    let generation = GenerationRequest::completion(
        "generation".into(),
        "Hi".into(),
        &GenerationOptions {
            max_tokens: 64,
            ..Default::default()
        },
        128,
        true,
    );
    let mut text = s
        .scheduler
        .submit(s.model("generation").unwrap(), generation)
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), text.next())
        .await
        .unwrap()
        .unwrap();
    // Generation is running while the embedding request still owns a queued pin/workspace.
    assert!(s.unload("encoder").is_err());
    assert!(
        tokio::time::timeout(Duration::from_millis(30), handle.receive())
            .await
            .is_err()
    );
    drop(handle);
    drop(text);
    s.scheduler.drained().await;
    assert_eq!(s.budget.stats().reserved, baseline);
    assert_eq!(s.budget.stats().session_buffer_capacity_bytes, 0);
}

#[tokio::test]
async fn embedding_queue_ceiling_and_shutdown_release_queued_and_running_jobs() {
    let (_dir, s, _app) = setup(|cfg| cfg.embeddings.requests = 1);
    encoder(&s, 1).await;
    let baseline = s.budget.stats().reserved;
    let options = EmbeddingOptions {
        batch_size: 1,
        max_input_tokens: 128,
        ..Default::default()
    };
    let req = || EmbeddingRequest::new("encoder".into(), vec!["a".repeat(100); 64], &options);
    let mut first = s
        .scheduler
        .submit_embedding(
            s.embedding("encoder").unwrap(),
            req(),
            s.budget.reserve(0).unwrap(),
        )
        .unwrap();
    let error = match s.scheduler.submit_embedding(
        s.embedding("encoder").unwrap(),
        req(),
        s.budget.reserve(0).unwrap(),
    ) {
        Err(e) => e,
        Ok(_) => panic!("queue ceiling ignored"),
    };
    assert_eq!(error.error.code, ErrorCode::QueueFull);
    s.shutdown.cancel();
    assert!(first.receive().await.is_err());
    drop(first);
    s.scheduler.drained().await;
    assert_eq!(s.budget.stats().reserved, baseline);
}

#[path = "../../coheryn-engine/tests/common/mod.rs"]
mod common;
use axum::{
    Router,
    body::{Body, Bytes, to_bytes},
    http::{Request, StatusCode},
};
use coheryn_api::{auth::Credentials, config::ServerConfig, http::router, service::Service};
use futures_util::{StreamExt, stream};
use serde_json::{Value, json};
use std::{sync::Arc, time::Duration};
use tower::ServiceExt;

fn credentials() -> Credentials {
    Credentials {
        inference_token: "i".repeat(64),
        admin_token: "a".repeat(64),
    }
}
async fn setup(context: usize) -> (tempfile::TempDir, Arc<Service>, Router) {
    let directory = tempfile::tempdir().unwrap();
    let config = ServerConfig {
        data_dir: directory.path().join("service"),
        max_context: 1024,
        max_memory: 256 << 20,
        consumer_timeout_seconds: 1,
        generation_timeout_seconds: 10,
        ..Default::default()
    };
    let service = Service::open(config, credentials()).unwrap();
    let data = common::conversation_model();
    service
        .ingest(
            "chat".into(),
            None,
            None,
            stream::iter([Ok::<_, std::io::Error>(Bytes::from(data))]),
        )
        .await
        .unwrap();
    service.load("chat".into(), context).await.unwrap();
    let app = router(service.clone());
    (directory, service, app)
}
fn request(method: &str, path: &str, body: Value, admin: bool) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", "application/json")
        .header(
            "authorization",
            format!(
                "Bearer {}",
                if admin {
                    credentials().admin_token
                } else {
                    credentials().inference_token
                }
            ),
        )
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap()
}
async fn call_as(
    app: &Router,
    method: &str,
    path: &str,
    body: Value,
    admin: bool,
) -> (StatusCode, Value) {
    let response = app
        .clone()
        .oneshot(request(method, path, body, admin))
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 16 << 20).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}
async fn call(app: &Router, method: &str, path: &str, body: Value) -> (StatusCode, Value) {
    call_as(app, method, path, body, false).await
}
async fn create(app: &Router, context: usize, overflow: &str) -> Value {
    let (status, view) = call(app, "POST", "/coheryn/v1/sessions", json!({"model":"chat","alias":"project","system":"S","context":context,"overflow":overflow})).await;
    assert_eq!(status, StatusCode::OK, "{view}");
    view
}
fn turn(revision: u64, key: &str, content: &str) -> Value {
    json!({"expected_revision":revision,"idempotency_key":key,"content":content,"settings":{"max_tokens":2}})
}

#[tokio::test]
async fn ownership_aliases_idempotency_and_revision_conflicts() {
    let (_directory, service, app) = setup(512).await;
    let before = service.budget.stats().reserved;
    let created = create(&app, 512, "reject").await;
    assert_eq!(created["persistent"], false);
    assert_eq!(created["revision"], 0);
    assert_eq!(
        call(
            &app,
            "POST",
            "/coheryn/v1/sessions",
            json!({"model":"chat","alias":"project","context":512})
        )
        .await
        .0,
        StatusCode::CONFLICT
    );
    assert_eq!(
        call_as(
            &app,
            "GET",
            "/coheryn/v1/sessions/project",
            Value::Null,
            true
        )
        .await
        .0,
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        call_as(&app, "GET", "/coheryn/v1/sessions", Value::Null, true)
            .await
            .1["sessions"],
        json!([])
    );
    let body = turn(0, "first", "বাংলা 日本語 العربية 🙂");
    let (status, first) = call(
        &app,
        "POST",
        "/coheryn/v1/sessions/project/messages",
        body.clone(),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{first}");
    assert_eq!(first["result"]["text"], "!!");
    assert_eq!(first["revision"], 2);
    let completed = service.scheduler.counters.snapshot().completed;
    let (_, replay) = call(&app, "POST", "/coheryn/v1/sessions/project/messages", body).await;
    assert_eq!(first, replay);
    assert_eq!(service.scheduler.counters.snapshot().completed, completed);
    assert_eq!(
        call(
            &app,
            "POST",
            "/coheryn/v1/sessions/project/messages",
            turn(0, "first", "changed")
        )
        .await
        .0,
        StatusCode::CONFLICT
    );
    assert_eq!(
        call(
            &app,
            "POST",
            "/coheryn/v1/sessions/project/messages",
            turn(0, "other", "stale")
        )
        .await
        .0,
        StatusCode::CONFLICT
    );
    let (_, view) = call(&app, "GET", "/coheryn/v1/sessions/project", Value::Null).await;
    assert_eq!(view["messages"].as_array().unwrap().len(), 3);
    assert_eq!(view["messages"][1]["content"], "বাংলা 日本語 العربية 🙂");
    let (status, reset) = call(
        &app,
        "POST",
        "/coheryn/v1/sessions/project/reset",
        json!({"expected_revision":2}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(reset["epoch"], 1);
    assert_eq!(reset["revision"], 3);
    assert_eq!(reset["messages"].as_array().unwrap().len(), 1);
    assert_eq!(reset["messages"][0]["content"], "S");
    let (status, cleared) = call(
        &app,
        "POST",
        "/coheryn/v1/sessions/project/reset",
        json!({"expected_revision":3,"clear_instructions":true}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(cleared["revision"], 4);
    assert_eq!(cleared["messages"], json!([]));
    assert_eq!(
        call(
            &app,
            "DELETE",
            "/coheryn/v1/sessions/project",
            json!({"expected_revision":4})
        )
        .await
        .0,
        StatusCode::OK
    );
    assert_eq!(
        call(&app, "GET", "/coheryn/v1/sessions/project", Value::Null)
            .await
            .0,
        StatusCode::NOT_FOUND
    );
    service.scheduler.drained().await;
    assert_eq!(service.budget.stats().reserved, before);
}

#[tokio::test]
async fn admission_failure_is_idempotent_and_does_not_poison_the_next_turn() {
    let (_directory, service, app) = setup(512).await;
    create(&app, 512, "reject").await;
    let mut body = turn(0, "budget", "hello");
    body["settings"]["max_memory"] = json!(16 << 20);
    let (status, first) = call(
        &app,
        "POST",
        "/coheryn/v1/sessions/project/messages",
        body.clone(),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{first}");
    let rejected = service.scheduler.counters.snapshot().rejected;
    let (status, replay) = call(&app, "POST", "/coheryn/v1/sessions/project/messages", body).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{replay}");
    assert_eq!(replay["error"]["code"], first["error"]["code"]);
    assert_eq!(service.scheduler.counters.snapshot().rejected, rejected);
    let (status, recovered) = call(
        &app,
        "POST",
        "/coheryn/v1/sessions/project/messages",
        turn(2, "recover", "again"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{recovered}");
    assert_eq!(
        recovered["context"]["included_message_ids"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(recovered["revision"], 4);
    assert_eq!(
        recovered["context"]["excluded_message_ids"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
}

#[tokio::test]
async fn idle_session_delete_remains_available_at_the_workload_budget_limit() {
    let (_directory, service, app) = setup(512).await;
    create(&app, 512, "reject").await;
    let stats = service.budget.stats();
    let held = service
        .budget
        .reserve(stats.limit - stats.reserved)
        .unwrap();
    assert_eq!(service.budget.stats().reserved, stats.limit);
    let (status, result) = call(
        &app,
        "DELETE",
        "/coheryn/v1/sessions/project",
        json!({"expected_revision":0}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{result}");
    assert_eq!(result["deleted"], true);
    assert!(service.budget.stats().reserved < stats.limit);
    drop(held);
    assert!(
        call(&app, "GET", "/coheryn/v1/sessions", Value::Null)
            .await
            .1["sessions"]
            .as_array()
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn exact_context_rejection_and_whole_turn_exclusion_preserve_authoritative_history() {
    for policy in ["reject", "oldest-complete-turns"] {
        let (_directory, service, app) = setup(128).await;
        create(&app, 128, policy).await;
        let (status, first) = call(
            &app,
            "POST",
            "/coheryn/v1/sessions/project/messages",
            turn(0, "one", &"a".repeat(48)),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{first}");
        let first_context = first["context"].clone();
        assert_eq!(
            first_context["prompt_tokens"],
            first["result"]["prompt_tokens"]
        );
        assert!(first_context["prompt_tokens"].as_u64().unwrap() + 2 <= 128);
        let (status, second) = call(
            &app,
            "POST",
            "/coheryn/v1/sessions/project/messages",
            turn(2, "two", &"b".repeat(48)),
        )
        .await;
        let (_, view) = call(&app, "GET", "/coheryn/v1/sessions/project", Value::Null).await;
        assert!(view["active_turn"].is_null());
        if policy == "reject" {
            assert_eq!(status, StatusCode::BAD_REQUEST, "{second}");
            assert_eq!(view["messages"][3]["status"], "failed");
            assert_eq!(service.scheduler.counters.snapshot().completed, 1);
        } else {
            assert_eq!(status, StatusCode::OK, "{second}");
            assert_eq!(
                second["context"]["excluded_message_ids"]
                    .as_array()
                    .unwrap()
                    .len(),
                2
            );
            assert_eq!(
                second["context"]["included_message_ids"]
                    .as_array()
                    .unwrap()
                    .len(),
                2
            );
            assert_eq!(view["messages"].as_array().unwrap().len(), 5);
            assert_eq!(view["messages"][0]["content"], "S");
            assert_eq!(view["messages"][1]["content"], "a".repeat(48));
            assert_eq!(
                second["context"]["prompt_tokens"],
                second["result"]["prompt_tokens"]
            );
        }
    }
}

#[tokio::test]
async fn concurrent_turns_cannot_mix_histories() {
    let (_directory, _service, app) = setup(512).await;
    create(&app, 512, "reject").await;
    let (one, two) = tokio::join!(
        call(
            &app,
            "POST",
            "/coheryn/v1/sessions/project/messages",
            turn(0, "one", "first")
        ),
        call(
            &app,
            "POST",
            "/coheryn/v1/sessions/project/messages",
            turn(0, "two", "second")
        )
    );
    assert!(
        (one.0 == StatusCode::OK && two.0 == StatusCode::CONFLICT)
            || (two.0 == StatusCode::OK && one.0 == StatusCode::CONFLICT)
    );
    let (_, view) = call(&app, "GET", "/coheryn/v1/sessions/project", Value::Null).await;
    assert_eq!(view["revision"], 2);
    assert_eq!(view["messages"].as_array().unwrap().len(), 3);
}

#[tokio::test]
async fn disconnect_cancels_worker_before_releasing_writer_and_resources() {
    let (_directory, service, app) = setup(1024).await;
    create(&app, 1024, "reject").await;
    let baseline = service.budget.stats().reserved;
    let mut body = turn(0, "cancel", "hello");
    body["stream"] = json!(true);
    body["settings"]["max_tokens"] = json!(700);
    let response = app
        .clone()
        .oneshot(request(
            "POST",
            "/coheryn/v1/sessions/project/messages",
            body,
            false,
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let mut stream = response.into_body().into_data_stream();
    let frame = stream.next().await.unwrap().unwrap();
    assert!(std::str::from_utf8(&frame).unwrap().contains("token"));
    assert_eq!(
        call(
            &app,
            "POST",
            "/coheryn/v1/sessions/project/reset",
            json!({"expected_revision":1})
        )
        .await
        .0,
        StatusCode::CONFLICT
    );
    assert_eq!(
        call(
            &app,
            "DELETE",
            "/coheryn/v1/sessions/project",
            json!({"expected_revision":1})
        )
        .await
        .0,
        StatusCode::CONFLICT
    );
    drop(stream);
    tokio::time::timeout(Duration::from_secs(3), service.scheduler.drained())
        .await
        .unwrap();
    let (_, view) = call(&app, "GET", "/coheryn/v1/sessions/project", Value::Null).await;
    assert!(view["active_turn"].is_null());
    assert_eq!(view["revision"], 2);
    assert_eq!(view["messages"].as_array().unwrap().len(), 2);
    assert_eq!(view["messages"][1]["status"], "cancelled");
    assert!(service.budget.stats().reserved <= baseline + (64 << 10));
}

#[tokio::test]
async fn successful_stream_commits_before_terminal_receipt_and_keeps_stateless_routes_independent()
{
    let (_directory, _service, app) = setup(512).await;
    create(&app, 512, "reject").await;
    let mut body = turn(0, "stream", "hello");
    body["stream"] = json!(true);
    let response = app
        .clone()
        .oneshot(request(
            "POST",
            "/coheryn/v1/sessions/project/messages",
            body,
            false,
        ))
        .await
        .unwrap();
    let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(text.ends_with("data: [DONE]\n\n"));
    let events: Vec<Value> = text
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|l| *l != "[DONE]")
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let receipt = events.last().unwrap();
    assert_eq!(receipt["revision"], 2);
    assert_eq!(receipt["status"], "completed");
    let (_, before) = call(&app, "GET", "/coheryn/v1/sessions/project", Value::Null).await;
    assert_eq!(call(&app,"POST","/v1/chat/completions",json!({"model":"chat","messages":[{"role":"user","content":"stateless"}],"max_tokens":2})).await.0,StatusCode::OK);
    let (_, after) = call(&app, "GET", "/coheryn/v1/sessions/project", Value::Null).await;
    assert_eq!(before, after);
}

#[tokio::test]
async fn unsupported_fields_missing_models_and_identity_changes_fail_without_silent_rebinding() {
    let (_directory, service, app) = setup(512).await;
    create(&app, 512, "reject").await;
    assert_eq!(
        call(
            &app,
            "POST",
            "/coheryn/v1/sessions",
            json!({"model":"chat","context":512,"persist":true})
        )
        .await
        .0,
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        call(
            &app,
            "POST",
            "/coheryn/v1/sessions/project/messages",
            json!({"expected_revision":0,"idempotency_key":"bad","messages":[]})
        )
        .await
        .0,
        StatusCode::BAD_REQUEST
    );
    service.unload("chat").unwrap();
    assert_eq!(
        call(
            &app,
            "POST",
            "/coheryn/v1/sessions/project/messages",
            turn(0, "unloaded", "hello")
        )
        .await
        .0,
        StatusCode::SERVICE_UNAVAILABLE
    );
    service.remove("chat").unwrap();
    let mut data = common::conversation_model();
    let last = data.len() - 1;
    data[last] ^= 1;
    service
        .ingest(
            "chat".into(),
            None,
            None,
            stream::iter([Ok::<_, std::io::Error>(Bytes::from(data))]),
        )
        .await
        .unwrap();
    service.load("chat".into(), 512).await.unwrap();
    assert_eq!(
        call(
            &app,
            "POST",
            "/coheryn/v1/sessions/project/messages",
            turn(2, "changed", "hello")
        )
        .await
        .0,
        StatusCode::CONFLICT
    );
    let (_, view) = call(&app, "GET", "/coheryn/v1/sessions/project", Value::Null).await;
    assert_eq!(view["revision"], 4);
    assert!(view["active_turn"].is_null());
    assert!(
        view["messages"]
            .as_array()
            .unwrap()
            .iter()
            .skip(1)
            .all(|m| m["status"] == "failed")
    );
}

#[tokio::test]
async fn finite_session_and_history_quotas_reclaim_budget_after_delete_and_expiry() {
    use coheryn_api::sessions::{
        BeginTurn, CreateSession, OverflowPolicy, SessionLimits, Sessions, SubmitTurn, TurnSettings,
    };
    use coheryn_engine::memory::Budget;
    let budget = Budget::new(32 << 20);
    let sessions = Sessions::new(
        SessionLimits {
            total: 1,
            per_owner: 1,
            history_bytes: 64,
            messages: 4,
            turn_keys: 1,
            idle_seconds: 1,
        },
        budget.clone(),
    );
    let create = || CreateSession {
        persist: false,
        model: "chat".into(),
        alias: Some("owned".into()),
        context: 128,
        system: None,
        overflow: OverflowPolicy::Reject,
    };
    sessions
        .create("one".into(), create(), "a".repeat(64))
        .unwrap();
    assert!(
        sessions
            .create("two".into(), create(), "a".repeat(64))
            .is_err()
    );
    assert!(sessions.get("two", "owned").is_err());
    let body = SubmitTurn {
        tool_results: vec![],
        expected_revision: 0,
        idempotency_key: "one".into(),
        content: "hello".into(),
        settings: TurnSettings {
            max_tokens: 2,
            ..Default::default()
        },
        stream: false,
    };
    let BeginTurn::New(lease) = sessions.begin("one", "owned", body).unwrap() else {
        panic!("new turn expected")
    };
    assert!(sessions.reset("one", "owned", 1).is_err());
    drop(lease);
    assert_eq!(sessions.get("one", "owned").unwrap().revision, 2);
    let body = SubmitTurn {
        tool_results: vec![],
        expected_revision: 2,
        idempotency_key: "two".into(),
        content: "again".into(),
        settings: TurnSettings {
            max_tokens: 2,
            ..Default::default()
        },
        stream: false,
    };
    assert!(sessions.begin("one", "owned", body).is_err());
    sessions.delete("one", "owned", 2).unwrap();
    assert_eq!(budget.stats().reserved, 0);
    sessions
        .create("one".into(), create(), "a".repeat(64))
        .unwrap();
    tokio::time::sleep(Duration::from_millis(1100)).await;
    assert!(sessions.list("one").unwrap().is_empty());
    assert_eq!(budget.stats().reserved, 0);
}

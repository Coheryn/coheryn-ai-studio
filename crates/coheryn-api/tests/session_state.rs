#[path = "../../coheryn-engine/tests/common/mod.rs"]
mod common;
use axum::{
    Router,
    body::{Body, Bytes, to_bytes},
    http::{Request, StatusCode},
};
use coheryn_api::{
    auth::Credentials, config::ServerConfig, http::router, service::Service,
    session_store::PersistenceLimits,
};
use futures_util::stream;
use serde_json::{Value, json};
use std::{path::Path, sync::Arc, time::Duration};
use tower::ServiceExt;
fn credentials() -> Credentials {
    Credentials {
        inference_token: "i".repeat(64),
        admin_token: "a".repeat(64),
    }
}
fn open(root: &Path, enabled: bool, cache: u64) -> (Arc<Service>, Router) {
    let mut config = ServerConfig {
        data_dir: root.to_owned(),
        max_memory: 256 << 20,
        max_context: 1024,
        ..Default::default()
    };
    config.persistence.enabled = enabled;
    config.cache.global_bytes = cache;
    config.cache.session_bytes = cache;
    let service = Service::open(config, credentials()).unwrap();
    let app = router(service.clone());
    (service, app)
}
async fn install(service: &Arc<Service>) {
    service
        .ingest(
            "chat".into(),
            None,
            None,
            stream::iter([Ok::<_, std::io::Error>(Bytes::from(
                common::conversation_model(),
            ))]),
        )
        .await
        .unwrap();
    service.load("chat".into(), 512).await.unwrap();
}
async fn call_as(
    app: &Router,
    method: &str,
    path: &str,
    body: Value,
    admin: bool,
) -> (StatusCode, Value) {
    let req = Request::builder()
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
        .unwrap();
    let response = app.clone().oneshot(req).await.unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 16 << 20).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}
async fn call(app: &Router, method: &str, path: &str, body: Value) -> (StatusCode, Value) {
    call_as(app, method, path, body, false).await
}
async fn create(app: &Router, name: &str, persist: bool) -> Value {
    let (status, view) = call(
        app,
        "POST",
        "/coheryn/v1/sessions",
        json!({"model":"chat","alias":name,"context":512,"system":"S","persist":persist}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{view}");
    view
}
fn turn(revision: u64, key: &str, content: &str) -> Value {
    json!({"expected_revision":revision,"idempotency_key":key,"content":content,"settings":{"max_tokens":2}})
}

#[tokio::test]
async fn kv_reuse_matches_stateless_and_is_evicted_on_reset_unload_and_global_pressure() {
    let directory = tempfile::tempdir().unwrap();
    let (service, app) = open(&directory.path().join("server"), false, 10_000);
    install(&service).await;
    create(&app, "one", false).await;
    create(&app, "two", false).await;
    assert_eq!(
        call(
            &app,
            "POST",
            "/coheryn/v1/sessions/one/messages",
            turn(0, "first", "hello")
        )
        .await
        .0,
        StatusCode::OK
    );
    let (_, first) = call(&app, "GET", "/coheryn/v1/sessions/one", Value::Null).await;
    assert!(first["cache_reserved_bytes"].as_u64().unwrap() > 0);
    let (status, second) = call(
        &app,
        "POST",
        "/coheryn/v1/sessions/one/messages",
        turn(2, "second", "world"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{second}");
    assert!(second["result"]["reused_prompt_tokens"].as_u64().unwrap() > 0);
    let (_, view) = call(&app, "GET", "/coheryn/v1/sessions/one", Value::Null).await;
    let messages: Vec<Value> = view["messages"]
        .as_array()
        .unwrap()
        .iter()
        .take(4)
        .map(|m| json!({"role":m["role"],"content":m["content"]}))
        .collect();
    let (_, cold) = call(
        &app,
        "POST",
        "/api/v1/generate",
        json!({"model":"chat","messages":messages,"context":512,"max_tokens":2}),
    )
    .await;
    assert_eq!(cold["token_ids"], second["result"]["token_ids"]);
    assert_eq!(cold["prompt_tokens"], second["result"]["prompt_tokens"]);
    call(
        &app,
        "POST",
        "/coheryn/v1/sessions/two/messages",
        turn(0, "other", "hello"),
    )
    .await;
    assert_eq!(
        call(&app, "GET", "/coheryn/v1/sessions/one", Value::Null)
            .await
            .1["cached_tokens"],
        0
    );
    assert!(
        call(&app, "GET", "/coheryn/v1/sessions/two", Value::Null)
            .await
            .1["cached_tokens"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert_eq!(
        call(
            &app,
            "POST",
            "/coheryn/v1/sessions/two/reset",
            json!({"expected_revision":2})
        )
        .await
        .0,
        StatusCode::OK
    );
    assert_eq!(
        call(&app, "GET", "/coheryn/v1/sessions/two", Value::Null)
            .await
            .1["cached_tokens"],
        0
    );
    call(
        &app,
        "POST",
        "/coheryn/v1/sessions/one/messages",
        turn(4, "third", "again"),
    )
    .await;
    service.scheduler.drained().await;
    service.unload("chat").unwrap();
    let (_, view) = call(&app, "GET", "/coheryn/v1/sessions/one", Value::Null).await;
    assert_eq!(view["cached_tokens"], 0);
    assert_eq!(view["message_count"], 7);
}

#[tokio::test]
async fn persistent_restart_restores_histories_and_idempotency_without_loading_weights() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("server");
    let (service, app) = open(&root, true, 64 << 20);
    install(&service).await;
    let created = create(&app, "durable", true).await;
    create(&app, "temporary", false).await;
    let body = turn(0, "first", "বাংলা العربية 日本語 🙂");
    let (status, receipt) = call(
        &app,
        "POST",
        "/coheryn/v1/sessions/durable/messages",
        body.clone(),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{receipt}");
    service.scheduler.drained().await;
    drop(app);
    drop(service);
    let (service, app) = open(&root, true, 64 << 20);
    let (_, list) = call(&app, "GET", "/coheryn/v1/sessions", Value::Null).await;
    assert_eq!(list["sessions"].as_array().unwrap().len(), 1);
    assert_eq!(list["sessions"][0]["id"], created["id"]);
    assert_eq!(list["sessions"][0]["cached_tokens"], 0);
    assert!(service.loaded().unwrap().as_array().unwrap().is_empty());
    let (status, replay) = call(&app, "POST", "/coheryn/v1/sessions/durable/messages", body).await;
    assert_eq!(status, StatusCode::OK, "{replay}");
    assert_eq!(replay, receipt);
    let (_, view) = call(&app, "GET", "/coheryn/v1/sessions/durable", Value::Null).await;
    assert_eq!(view["messages"][1]["content"], "বাংলা العربية 日本語 🙂");
    assert_eq!(view["revision"], 2);
    assert_eq!(
        call_as(
            &app,
            "GET",
            "/coheryn/v1/sessions/durable",
            Value::Null,
            true
        )
        .await
        .0,
        StatusCode::NOT_FOUND
    );
    service.load("chat".into(), 512).await.unwrap();
    let (status, next) = call(
        &app,
        "POST",
        "/coheryn/v1/sessions/durable/messages",
        turn(2, "next", "hello"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{next}");
    assert_eq!(next["result"]["reused_prompt_tokens"], 0);
}

#[tokio::test]
async fn disabled_store_never_touches_database_and_save_import_export_enforce_ownership_and_format()
{
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("server");
    let (service, app) = open(&root, false, 64 << 20);
    install(&service).await;
    assert!(!root.join("conversations").exists());
    let (status, _) = call(
        &app,
        "POST",
        "/coheryn/v1/sessions",
        json!({"model":"chat","context":512,"persist":true}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    create(&app, "source", false).await;
    call(
        &app,
        "POST",
        "/coheryn/v1/sessions/source/messages",
        turn(0, "one", "hello"),
    )
    .await;
    let (status, document) = call(
        &app,
        "GET",
        "/coheryn/v1/sessions/source/export",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{document}");
    assert_eq!(document["version"], 1);
    assert!(document.get("owner").is_none());
    assert_eq!(document["session"]["cached_tokens"], 0);
    assert_eq!(
        call_as(
            &app,
            "GET",
            "/coheryn/v1/sessions/source/export",
            Value::Null,
            true
        )
        .await
        .0,
        StatusCode::NOT_FOUND
    );
    let (status, copy) = call(
        &app,
        "POST",
        "/coheryn/v1/sessions/import",
        json!({"document":document,"name":"copy"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{copy}");
    assert_ne!(copy["id"], document["session"]["id"]);
    assert_eq!(copy["messages"], document["messages"]);
    for array in ["messages", "turns", "context", "stops"] {
        let mut oversized = document.clone();
        match array {
            "messages" => oversized["messages"] = json!(vec![document["messages"][0].clone(); 257]),
            "turns" => oversized["turns"] = json!(vec![document["turns"][0].clone(); 129]),
            "context" => {
                oversized["turns"][0]["receipt"]["context"]["included_message_ids"] =
                    json!(vec!["id"; 257]);
            }
            _ => oversized["turns"][0]["settings"]["stop"] = json!(vec!["stop"; 17]),
        }
        assert_eq!(
            call(
                &app,
                "POST",
                "/coheryn/v1/sessions/import",
                json!({"document":oversized})
            )
            .await
            .0,
            StatusCode::BAD_REQUEST
        );
    }
    let mut invalid = document.clone();
    invalid["version"] = json!(99);
    assert_eq!(
        call(
            &app,
            "POST",
            "/coheryn/v1/sessions/import",
            json!({"document":invalid,"name":"invalid"})
        )
        .await
        .0,
        StatusCode::BAD_REQUEST
    );
    let mut invalid = document.clone();
    invalid["messages"][1]["role"] = json!("tool");
    assert_eq!(
        call(
            &app,
            "POST",
            "/coheryn/v1/sessions/import",
            json!({"document":invalid})
        )
        .await
        .0,
        StatusCode::BAD_REQUEST
    );
    let mut invalid = document.clone();
    invalid["messages"][1]["id"] = invalid["messages"][0]["id"].clone();
    assert_eq!(
        call(
            &app,
            "POST",
            "/coheryn/v1/sessions/import",
            json!({"document":invalid})
        )
        .await
        .0,
        StatusCode::BAD_REQUEST
    );
    assert!(!root.join("conversations").exists());
    service.scheduler.drained().await;
    drop(app);
    drop(service);
    let (service, app) = open(&root, true, 64 << 20);
    install_registered(&service).await;
    create(&app, "promote", false).await;
    let (status, saved) = call(
        &app,
        "POST",
        "/coheryn/v1/sessions/promote/save",
        json!({"expected_revision":0}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{saved}");
    assert_eq!(saved["persistent"], true);
    drop(app);
    drop(service);
    let (_service, app) = open(&root, true, 64 << 20);
    assert_eq!(
        call(&app, "GET", "/coheryn/v1/sessions/promote", Value::Null)
            .await
            .0,
        StatusCode::OK
    );
}
async fn install_registered(service: &Arc<Service>) {
    service.load("chat".into(), 512).await.unwrap();
}

#[tokio::test]
async fn quota_rejection_leaves_no_half_created_session_and_corruption_is_not_erased() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("server");
    let mut config = ServerConfig {
        data_dir: root.clone(),
        max_memory: 256 << 20,
        max_context: 1024,
        ..Default::default()
    };
    config.persistence = PersistenceLimits {
        enabled: true,
        owner_bytes: 128,
        ..Default::default()
    };
    let service = Service::open(config.clone(), credentials()).unwrap();
    let app = router(service.clone());
    install(&service).await;
    let (status, result) = call(
        &app,
        "POST",
        "/coheryn/v1/sessions",
        json!({"model":"chat","alias":"too-large","context":512,"persist":true}),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{result}");
    assert!(
        call(&app, "GET", "/coheryn/v1/sessions", Value::Null)
            .await
            .1["sessions"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    drop(app);
    drop(service);
    let path = root.join("conversations/history.sqlite");
    std::fs::write(&path, b"broken database bytes").unwrap();
    let before = std::fs::read(&path).unwrap();
    assert!(Service::open(config.clone(), credentials()).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), before);
    config.persistence.enabled = false;
    let service = Service::open(config, credentials()).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert!(service.status().is_ok());
}

#[tokio::test]
async fn failed_commits_release_writers_and_restart_recovers_pending_without_reexecuting() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("server");
    let (service, app) = open(&root, true, 64 << 20);
    install(&service).await;
    create(&app, "recover", true).await;
    let database = rusqlite::Connection::open(root.join("conversations/history.sqlite")).unwrap();
    database.execute_batch("CREATE TRIGGER reject_final BEFORE UPDATE ON sessions WHEN json_extract(NEW.meta,'$.active_turn') IS NULL AND json_extract(OLD.meta,'$.active_turn') IS NOT NULL BEGIN SELECT RAISE(ABORT,'injected final commit failure'); END;").unwrap();
    let (status, response) = call(
        &app,
        "POST",
        "/coheryn/v1/sessions/recover/messages",
        turn(0, "interrupted", "hello"),
    )
    .await;
    assert!(status.is_server_error(), "{response}");
    service.scheduler.drained().await;
    let (_, view) = call(&app, "GET", "/coheryn/v1/sessions/recover", Value::Null).await;
    assert!(view["active_turn"].is_null());
    assert!(view["storage_error"].is_string());
    assert_eq!(view["messages"].as_array().unwrap().len(), 2);
    database
        .execute_batch("DROP TRIGGER reject_final;")
        .unwrap();
    drop(database);
    drop(app);
    drop(service);
    let (service, app) = open(&root, true, 64 << 20);
    let (_, view) = call(&app, "GET", "/coheryn/v1/sessions/recover", Value::Null).await;
    assert_eq!(view["revision"], 2);
    assert_eq!(view["messages"][1]["status"], "cancelled");
    assert_eq!(service.scheduler.counters.snapshot().completed, 0);
    assert!(view["active_turn"].is_null());
    let (status, replay) = call(
        &app,
        "POST",
        "/coheryn/v1/sessions/recover/messages",
        turn(0, "interrupted", "hello"),
    )
    .await;
    assert_eq!(status, StatusCode::REQUEST_TIMEOUT, "{replay}");
    assert_eq!(service.scheduler.counters.snapshot().completed, 0);
}

#[tokio::test]
async fn future_database_version_is_rejected_without_mutation_and_cache_expiry_keeps_history() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("server");
    let (service, app) = open(&root, true, 64 << 20);
    install(&service).await;
    create(&app, "kept", true).await;
    drop(app);
    drop(service);
    let path = root.join("conversations/history.sqlite");
    let database = rusqlite::Connection::open(&path).unwrap();
    database.pragma_update(None, "user_version", 99i64).unwrap();
    drop(database);
    let before = std::fs::read(&path).unwrap();
    let mut config = ServerConfig {
        data_dir: root,
        max_memory: 256 << 20,
        ..Default::default()
    };
    config.persistence.enabled = true;
    assert!(Service::open(config, credentials()).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), before);
    let root = directory.path().join("expiry");
    let mut config = ServerConfig {
        data_dir: root,
        max_memory: 256 << 20,
        max_context: 1024,
        ..Default::default()
    };
    config.cache.idle_seconds = 1;
    let service = Service::open(config, credentials()).unwrap();
    let app = router(service.clone());
    install(&service).await;
    create(&app, "cached", false).await;
    call(
        &app,
        "POST",
        "/coheryn/v1/sessions/cached/messages",
        turn(0, "one", "hello"),
    )
    .await;
    tokio::time::sleep(Duration::from_millis(1100)).await;
    let (_, view) = call(&app, "GET", "/coheryn/v1/sessions/cached", Value::Null).await;
    assert_eq!(view["cached_tokens"], 0);
    assert_eq!(view["message_count"], 3);
}

#[tokio::test]
async fn restored_history_pages_do_not_hydrate_other_turns_or_execution_buffers() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("server");
    let (service, app) = open(&root, true, 64 << 20);
    install(&service).await;
    create(&app, "source", false).await;
    call(
        &app,
        "POST",
        "/coheryn/v1/sessions/source/messages",
        turn(0, "first", "hello"),
    )
    .await;
    let (_, mut document) = call(
        &app,
        "GET",
        "/coheryn/v1/sessions/source/export",
        Value::Null,
    )
    .await;
    let user = document["messages"][1].clone();
    let assistant = document["messages"][2].clone();
    let record = document["turns"][0].clone();
    let mut messages = vec![document["messages"][0].clone()];
    let mut turns = Vec::new();
    for index in 0..60 {
        let mut user = user.clone();
        let mut assistant = assistant.clone();
        let mut record = record.clone();
        let turn_id = format!("turn-{index}");
        user["id"] = json!(format!("user-{index}"));
        user["turn_id"] = json!(turn_id);
        assistant["id"] = json!(format!("assistant-{index}"));
        assistant["turn_id"] = json!(turn_id);
        record["key"] = json!(format!("key-{index}"));
        record["receipt"]["turn_id"] = json!(turn_id);
        record["receipt"]["revision"] = json!(index * 2 + 2);
        record["receipt"]["context"] = Value::Null;
        messages.push(user);
        messages.push(assistant);
        turns.push(record);
    }
    let bytes: usize = messages
        .iter()
        .map(|m| m["content"].as_str().unwrap().len())
        .sum();
    document["messages"] = json!(messages);
    document["turns"] = json!(turns);
    document["session"]["message_count"] = json!(121);
    document["session"]["history_bytes"] = json!(bytes);
    document["session"]["revision"] = json!(120);
    let (status, result) = call(
        &app,
        "POST",
        "/coheryn/v1/sessions/import",
        json!({"document":document,"name":"paged","persist":true}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{result}");
    service.scheduler.drained().await;
    drop(app);
    drop(service);
    let (service, app) = open(&root, true, 64 << 20);
    let baseline = service.budget.stats().reserved;
    let (status, page) = call(
        &app,
        "GET",
        "/coheryn/v1/sessions/paged?offset=0&limit=100",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{page}");
    assert_eq!(page["messages"].as_array().unwrap().len(), 100);
    assert_eq!(page["next_offset"], 100);
    let (_, tail) = call(
        &app,
        "GET",
        "/coheryn/v1/sessions/paged?offset=100&limit=100",
        Value::Null,
    )
    .await;
    assert_eq!(tail["messages"].as_array().unwrap().len(), 21);
    assert!(tail.get("next_offset").is_none());
    assert_eq!(service.budget.stats().reserved, baseline);
    assert_eq!(
        call(
            &app,
            "GET",
            "/coheryn/v1/sessions/paged?limit=101",
            Value::Null
        )
        .await
        .0,
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        call(
            &app,
            "GET",
            "/coheryn/v1/sessions/paged?offset=122",
            Value::Null
        )
        .await
        .0,
        StatusCode::BAD_REQUEST
    );
}

#[tokio::test]
async fn retention_expires_durable_data_without_loading_weights() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("server");
    let (service, app) = open(&root, true, 0);
    install(&service).await;
    let created = create(&app, "expired", true).await;
    drop(app);
    drop(service);
    let db = rusqlite::Connection::open(root.join("conversations/history.sqlite")).unwrap();
    let mut metadata = created;
    metadata["messages"] = Value::Null;
    metadata["updated"] = json!(0);
    db.execute(
        "UPDATE sessions SET meta=?1",
        [serde_json::to_string(&metadata).unwrap()],
    )
    .unwrap();
    drop(db);
    let (service, app) = open(&root, true, 0);
    let (_, sessions) = call(&app, "GET", "/coheryn/v1/sessions", Value::Null).await;
    assert_eq!(sessions["sessions"], json!([]));
    assert_eq!(service.budget.stats().reserved, 16 << 20);
    drop(app);
    drop(service);
    let db = rusqlite::Connection::open(root.join("conversations/history.sqlite")).unwrap();
    assert_eq!(
        db.query_row("SELECT count(*) FROM sessions", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
    assert_eq!(
        db.query_row("SELECT count(*) FROM records", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[cfg(unix)]
#[test]
fn history_store_refuses_symlinks_extra_links_and_public_permissions() {
    use coheryn_api::session_store::Store;
    use coheryn_engine::memory::Budget;
    use std::os::unix::fs::{PermissionsExt, symlink};
    for case in ["symlink", "hardlink", "permissions"] {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("server");
        let limits = PersistenceLimits::default();
        drop(Store::open(&root, limits.clone(), &Budget::new(64 << 20)).unwrap());
        let path = root.join("conversations/history.sqlite");
        let before = std::fs::read(&path).unwrap();
        let original = directory.path().join("original.sqlite");
        match case {
            "symlink" => {
                std::fs::rename(&path, &original).unwrap();
                symlink(&original, &path).unwrap();
            }
            "hardlink" => std::fs::hard_link(&path, &original).unwrap(),
            _ => std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap(),
        }
        assert!(Store::open(&root, limits, &Budget::new(64 << 20)).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }
}

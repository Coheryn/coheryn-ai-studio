//! Coordinator fixtures inject decoded proposals; numerical/model parity is a separate check.
#[path = "../../coheryn-engine/tests/common/mod.rs"]
mod common;
use coheryn_api::{scheduler::GenerationEvent, session_store::PersistenceLimits, sessions::*};
use coheryn_engine::{
    Cancellation, FinishReason, GenerationResult, Model, ResourceLimits, memory::Budget, tools::*,
};
use serde_json::{Value, json};
use std::path::Path;
fn settings() -> TurnSettings {
    serde_json::from_value(json!({"max_tokens":1,"tools":[{"type":"function","function":{"name":"add","parameters":{"type":"object"}}}]})).unwrap()
}
fn owner() -> String {
    "a".repeat(64)
}
fn open(root: &Path) -> (Sessions, Budget) {
    let budget = Budget::new(256 << 20);
    let sessions = Sessions::open(
        SessionLimits::default(),
        CacheLimits::default(),
        PersistenceLimits {
            enabled: true,
            ..Default::default()
        },
        budget.clone(),
        root,
    )
    .unwrap();
    (sessions, budget)
}
fn create(s: &Sessions, name: &str, context: usize) {
    s.create(
        owner(),
        CreateSession {
            model: "chat".into(),
            persist: true,
            alias: Some(name.into()),
            context,
            system: Some("Be brief.".into()),
            overflow: OverflowPolicy::OldestCompleteTurns,
        },
        "b".repeat(64),
    )
    .unwrap();
}
fn proposal(budget: &Budget, text: &str, calls: Vec<ToolCall>) -> GenerationResult {
    GenerationResult {
        batching: None,
        sampling_metadata: None,
        constraint_validation: None,
        text: text.into(),
        tool_calls: calls.clone(),
        token_ids: vec![0],
        prompt_tokens: 1,
        reused_prompt_tokens: 0,
        completion_tokens: 1,
        finish_reason: if calls.is_empty() {
            FinishReason::Stop
        } else {
            FinishReason::ToolCalls
        },
        prefill_ms: 0.,
        tokenization_ms: 0.,
        decode_ms: 0.,
        output_delivery_ms: 0.,
        first_token_ms: None,
        memory: budget.stats(),
    }
}
fn calls() -> Vec<ToolCall> {
    serde_json::from_value(json!([{"id":"call-z","type":"function","function":{"name":"add","arguments":"{\"a\":2,\"b\":2}"}},{"id":"call-a","type":"function","function":{"name":"add","arguments":"{\"a\":3,\"b\":4}"}}])).unwrap()
}
fn body(revision: u64, key: &str) -> SubmitTurn {
    SubmitTurn {
        expected_revision: revision,
        idempotency_key: key.into(),
        content: "Use add.".into(),
        tool_results: vec![],
        settings: settings(),
        stream: false,
    }
}
fn finish(s: &Sessions, budget: &Budget, request: SubmitTurn, calls: Vec<ToolCall>) {
    let BeginTurn::New(mut turn) = s.begin(&owner(), "test", request).unwrap() else {
        panic!("new writer expected")
    };
    turn.finish(&GenerationEvent::Finished(proposal(budget, "", calls)))
        .unwrap();
}
#[test]
fn pending_calls_survive_restart_replay_export_import_without_execution() {
    let temp = tempfile::tempdir().unwrap();
    let (s, budget) = open(temp.path());
    create(&s, "test", 1024);
    let request = body(0, "first");
    finish(&s, &budget, request.clone(), calls());
    let view = s.get(&owner(), "test").unwrap();
    assert_eq!(view.pending_tool_calls, vec!["call-z", "call-a"]);
    assert!(s.get(&"c".repeat(64), "test").is_err());
    assert!(s.begin(&owner(), "test", body(2, "new-user")).is_err());
    let exported = s.export(&owner(), "test").unwrap();
    assert_eq!(exported.version, 2);
    let mut invalid = serde_json::to_value(&exported).unwrap();
    invalid["messages"][2]["tool_calls"][0]["function"]["name"] = json!("undeclared");
    assert!(
        s.import(
            owner(),
            serde_json::from_value(invalid).unwrap(),
            Some("bad".into()),
            false
        )
        .is_err()
    );
    drop(s);
    assert_eq!(budget.stats().reserved, 0);
    let (s, budget) = open(temp.path());
    assert_eq!(
        s.get(&owner(), "test").unwrap().pending_tool_calls,
        view.pending_tool_calls
    );
    let BeginTurn::Replay(replay) = s.begin(&owner(), "test", request).unwrap() else {
        panic!("replay expected")
    };
    assert_eq!(
        replay.snapshot().unwrap().result.unwrap().tool_calls,
        calls()
    );
    let imported = s
        .import(owner(), exported, Some("copy".into()), true)
        .unwrap();
    assert_eq!(imported.pending_tool_calls, view.pending_tool_calls);
    assert_ne!(imported.id, view.id);
    let mut results = body(2, "results");
    results.content.clear();
    results.settings = TurnSettings {
        max_tokens: 1,
        ..Default::default()
    };
    results.tool_results = vec![
        ToolResult {
            tool_call_id: "call-z".into(),
            content: "4 বাংলা 🌍".into(),
        },
        ToolResult {
            tool_call_id: "call-a".into(),
            content: "7 日本語".into(),
        },
    ];
    for kind in ["reversed", "missing", "unknown", "duplicate"] {
        let mut bad = results.clone();
        match kind {
            "reversed" => bad.tool_results.reverse(),
            "missing" => {
                bad.tool_results.pop();
            }
            "unknown" => bad.tool_results[0].tool_call_id = "unknown".into(),
            _ => bad.tool_results[1].tool_call_id = "call-z".into(),
        };
        assert!(s.begin(&owner(), "test", bad).is_err());
    }
    finish(&s, &budget, results, vec![]);
    let view = s.get(&owner(), "test").unwrap();
    assert!(view.pending_tool_calls.is_empty());
    assert_eq!(view.message_count, 6);
    assert_eq!(view.revision, 4);
    let doc = s.export(&owner(), "test").unwrap();
    s.import(owner(), doc, Some("completed-copy".into()), false)
        .unwrap();
    s.reset(&owner(), "test", 4).unwrap();
    assert!(
        s.get(&owner(), "test")
            .unwrap()
            .pending_tool_calls
            .is_empty()
    );
}
#[test]
fn tool_groups_are_truncated_atomically_and_current_result_groups_are_protected() {
    let temp = tempfile::tempdir().unwrap();
    let model_path = temp.path().join("model.gguf");
    std::fs::write(&model_path, common::conversation_model()).unwrap();
    let model = Model::load(
        &model_path,
        ResourceLimits {
            max_memory: 64 << 20,
            context: 1024,
        },
    )
    .unwrap();
    let (s, budget) = open(&temp.path().join("store"));
    create(&s, "test", 1024);
    let mut first = body(0, "first");
    first.content = "x".repeat(150);
    finish(&s, &budget, first, calls());
    let mut results = body(2, "results");
    results.content.clear();
    results.tool_results = vec![
        ToolResult {
            tool_call_id: "call-z".into(),
            content: "4".into(),
        },
        ToolResult {
            tool_call_id: "call-a".into(),
            content: "7".into(),
        },
    ];
    finish(&s, &budget, results, vec![]);
    let mut next = body(4, "next");
    next.content = "z".repeat(200);
    let BeginTurn::New(mut turn) = s.begin(&owner(), "test", next).unwrap() else {
        panic!()
    };
    let prepared = turn.prepare(&model, &Cancellation::default()).unwrap();
    let accounting = turn.watch.snapshot().unwrap().context.unwrap();
    assert!(!accounting.excluded_message_ids.is_empty());
    let selected = prepared.messages.unwrap();
    assert!(
        selected
            .iter()
            .all(|m| m.role != "tool" && m.tool_calls.is_empty())
    );
    drop(turn);
    // A fresh waiting call group must never lose its originating assistant calls.
    let (s, budget) = open(&temp.path().join("protected"));
    create(&s, "test", 256);
    finish(&s, &budget, body(0, "first"), calls());
    let mut results = body(2, "results");
    results.content.clear();
    results.tool_results = vec![
        ToolResult {
            tool_call_id: "call-z".into(),
            content: "4".into(),
        },
        ToolResult {
            tool_call_id: "call-a".into(),
            content: "7".into(),
        },
    ];
    let BeginTurn::New(mut turn) = s.begin(&owner(), "test", results).unwrap() else {
        panic!()
    };
    assert!(turn.prepare(&model, &Cancellation::default()).is_err());
}
#[test]
fn failed_pending_write_rolls_back_all_results_and_schema_upgrade_preserves_legacy_data() {
    let temp = tempfile::tempdir().unwrap();
    let (s, budget) = open(temp.path());
    create(&s, "test", 1024);
    finish(&s, &budget, body(0, "first"), calls());
    let db = rusqlite::Connection::open(temp.path().join("conversations/history.sqlite")).unwrap();
    db.execute_batch("CREATE TRIGGER reject_change BEFORE UPDATE ON sessions BEGIN SELECT RAISE(FAIL,'injected pending failure'); END;").unwrap();
    let mut request = body(2, "results");
    request.content.clear();
    request.tool_results = vec![
        ToolResult {
            tool_call_id: "call-z".into(),
            content: "4".into(),
        },
        ToolResult {
            tool_call_id: "call-a".into(),
            content: "7".into(),
        },
    ];
    assert!(s.begin(&owner(), "test", request).is_err());
    let view = s.get(&owner(), "test").unwrap();
    assert_eq!(view.message_count, 3);
    assert_eq!(view.revision, 2);
    assert!(view.active_turn.is_none());
    db.execute_batch("DROP TRIGGER reject_change;").unwrap();
    drop(db);
    drop(s);
    // Build data with the exact legacy text-only payload shape, then mark its v1 table layout.
    let legacy_root = temp.path().join("legacy");
    let (legacy, legacy_budget) = open(&legacy_root);
    create(&legacy, "test", 1024);
    let mut original = body(0, "legacy");
    original.settings = TurnSettings {
        max_tokens: 1,
        ..Default::default()
    };
    finish(&legacy, &legacy_budget, original.clone(), vec![]);
    let before = serde_json::to_value(legacy.export(&owner(), "test").unwrap()).unwrap();
    drop(legacy);
    let db = rusqlite::Connection::open(legacy_root.join("conversations/history.sqlite")).unwrap();
    db.pragma_update(None, "user_version", 1i64).unwrap();
    drop(db);
    let (legacy, _) = open(&legacy_root);
    assert_eq!(
        before,
        serde_json::to_value(legacy.export(&owner(), "test").unwrap()).unwrap()
    );
    assert!(matches!(
        legacy.begin(&owner(), "test", original).unwrap(),
        BeginTurn::Replay(_)
    ));
    let db = rusqlite::Connection::open(legacy_root.join("conversations/history.sqlite")).unwrap();
    assert_eq!(
        db.query_row("PRAGMA user_version", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        2
    );
}
#[tokio::test]
async fn unsupported_tool_models_and_inputs_fail_before_queueing() {
    use axum::{
        body::{Body, Bytes, to_bytes},
        http::Request,
    };
    use coheryn_api::{auth::Credentials, config::ServerConfig, http::router, service::Service};
    use tower::ServiceExt;
    let temp = tempfile::tempdir().unwrap();
    let service = Service::open(
        ServerConfig {
            data_dir: temp.path().join("service"),
            max_memory: 256 << 20,
            max_context: 1024,
            ..Default::default()
        },
        Credentials {
            inference_token: "i".repeat(64),
            admin_token: "a".repeat(64),
        },
    )
    .unwrap();
    service
        .ingest(
            "chat".into(),
            None,
            None,
            futures_util::stream::iter([Ok::<_, std::io::Error>(Bytes::from(
                common::conversation_model(),
            ))]),
        )
        .await
        .unwrap();
    service.load("chat".into(), 1024).await.unwrap();
    let app = router(service.clone());
    for (body, expected) in [
        (
            json!({"model":"chat","messages":[{"role":"user","content":"test"}],"tools":[{"type":"function","function":{"name":"add","strict":true,"parameters":{"type":"object"}}}]}),
            400,
        ),
        (
            json!({"model":"chat","messages":[{"role":"tool","tool_call_id":"unknown","content":"4"}]}),
            400,
        ),
        (
            json!({"model":"chat","messages":[{"role":"user","content":"test"}],"tools":settings().tools,"tool_choice":"required","max_tokens":1}),
            400,
        ),
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("authorization", format!("Bearer {}", "i".repeat(64)))
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status().as_u16(), expected);
        let value: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 1 << 20).await.unwrap())
                .unwrap();
        assert!(value["error"]["message"].is_string());
    }
    service.scheduler.drained().await;
    assert_eq!(service.status().unwrap()["scheduler"]["active"], 0);
}

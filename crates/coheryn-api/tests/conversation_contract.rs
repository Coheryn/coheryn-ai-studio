//! Exercise authoritative context selection through a non-GGUF history interface.
use coheryn_api::{
    resident::TextConversation,
    scheduler::GenerationEvent,
    sessions::{BeginTurn, CreateSession, OverflowPolicy, Sessions, SubmitTurn},
};
use coheryn_engine::{
    Cancellation, Error, ErrorCode, GenerationOptions, GenerationResult, memory::Budget,
    tokenizer::Message, tools::ToolConfig,
};
use serde_json::json;

struct Conversation;
impl TextConversation for Conversation {
    fn context_limit(&self) -> usize {
        64
    }
    fn supports_history(&self) -> bool {
        true
    }
    fn render_history(
        &self,
        messages: &[Message],
        tools: &ToolConfig,
    ) -> coheryn_engine::Result<String> {
        assert!(!tools.enabled());
        Ok(messages
            .iter()
            .map(|m| format!("{}:{}\n", m.role, m.content))
            .collect())
    }
    fn token_count(&self, prompt: &str) -> coheryn_engine::Result<usize> {
        Ok(prompt.chars().count())
    }
    fn validate_options(&self, options: &GenerationOptions) -> coheryn_engine::Result<()> {
        coheryn_engine::qwen35::Model::check_options(options)
    }
}
fn setup(policy: OverflowPolicy) -> (Sessions, Budget) {
    let budget = Budget::new(32 << 20);
    let sessions = Sessions::new(Default::default(), budget.clone());
    sessions
        .create(
            "owner".into(),
            CreateSession {
                task: None,
                model: "checkpoint".into(),
                alias: Some("history".into()),
                context: 64,
                system: Some("S".into()),
                overflow: policy,
                persist: false,
            },
            "a".repeat(64),
        )
        .unwrap();
    (sessions, budget)
}
fn submit(revision: u64, key: &str, text: &str) -> SubmitTurn {
    serde_json::from_value(json!({"expected_revision":revision,"idempotency_key":key,"content":text,"settings":{"max_tokens":4}})).unwrap()
}
fn result(prompt_tokens: usize) -> GenerationResult {
    serde_json::from_value(json!({"text":"old answer old answer","token_ids":[1,2,3,4],"prompt_tokens":prompt_tokens,"completion_tokens":4,"reused_prompt_tokens":0,"finish_reason":"length","tokenization_ms":0.,"prefill_ms":0.,"decode_ms":0.,"output_delivery_ms":0.,"first_token_ms":null,"memory":{"limit":0,"reserved":0,"peak_reserved":0,"session_buffer_capacity_bytes":0,"peak_session_buffer_capacity_bytes":0}})).unwrap()
}
fn first(sessions: &Sessions) {
    let BeginTurn::New(mut turn) = sessions
        .begin("owner", "history", submit(0, "first", "first first first"))
        .unwrap()
    else {
        panic!("new turn expected")
    };
    let request = turn
        .prepare_conversation(&Conversation, &Default::default())
        .unwrap();
    assert_eq!(request.messages.as_ref().unwrap().len(), 2);
    let count = turn
        .watch
        .snapshot()
        .unwrap()
        .context
        .unwrap()
        .prompt_tokens;
    turn.finish(&GenerationEvent::Finished(result(count)))
        .unwrap();
}
#[test]
fn generic_history_excludes_whole_old_turns_and_preserves_system_and_unicode() {
    let (sessions, budget) = setup(OverflowPolicy::OldestCompleteTurns);
    first(&sessions);
    let previous = sessions.get("owner", "history").unwrap();
    let old_ids: Vec<_> = previous
        .messages
        .as_ref()
        .unwrap()
        .iter()
        .filter(|m| m.role != "system")
        .map(|m| m.id.clone())
        .collect();
    let BeginTurn::New(mut turn) = sessions
        .begin(
            "owner",
            "history",
            submit(2, "second", "中文 বাংলা 🌍 current current"),
        )
        .unwrap()
    else {
        panic!("new turn expected")
    };
    let request = turn
        .prepare_conversation(&Conversation, &Default::default())
        .unwrap();
    let messages = request.messages.unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].content, "S");
    assert_eq!(messages[1].content, "中文 বাংলা 🌍 current current");
    let context = turn.watch.snapshot().unwrap().context.unwrap();
    assert_eq!(context.excluded_message_ids, old_ids);
    assert_eq!(
        context.prompt_tokens,
        "system:S\nuser:中文 বাংলা 🌍 current current\n"
            .chars()
            .count()
    );
    assert_eq!(context.reserved_output_tokens, 4);
    turn.finish(&GenerationEvent::Failed(
        Error::new(ErrorCode::Cancelled, "test cancellation").into(),
    ))
    .unwrap();
    let view = sessions.get("owner", "history").unwrap();
    // Exclusion affects execution only, never the authoritative old messages.
    assert_eq!(view.message_count, 4);
    assert_eq!(view.cached_tokens, 0);
    assert!(view.active_turn.is_none());
    assert!(matches!(
        sessions
            .begin(
                "owner",
                "history",
                submit(2, "second", "中文 বাংলা 🌍 current current")
            )
            .unwrap(),
        BeginTurn::Replay(_)
    ));
    drop(turn);
    sessions.delete("owner", "history", 4).unwrap();
    drop(sessions);
    assert_eq!(budget.stats().reserved, 0);
}
#[test]
fn generic_history_rejects_overflow_cancel_and_unsupported_controls_without_losing_writer() {
    let (sessions, _) = setup(OverflowPolicy::Reject);
    first(&sessions);
    let BeginTurn::New(mut turn) = sessions
        .begin(
            "owner",
            "history",
            submit(2, "second", "current current current"),
        )
        .unwrap()
    else {
        panic!("new turn expected")
    };
    assert!(
        turn.prepare_conversation(&Conversation, &Default::default())
            .is_err()
    );
    drop(turn);
    assert!(
        sessions
            .get("owner", "history")
            .unwrap()
            .active_turn
            .is_none()
    );
    let mut request = submit(4, "third", "hi");
    request.settings.temperature = 0.5;
    let BeginTurn::New(mut turn) = sessions.begin("owner", "history", request).unwrap() else {
        panic!("new turn expected")
    };
    assert_eq!(
        turn.prepare_conversation(&Conversation, &Default::default())
            .err()
            .unwrap()
            .error
            .code,
        ErrorCode::UnsupportedFeature
    );
    drop(turn);
    let BeginTurn::New(mut turn) = sessions
        .begin("owner", "history", submit(6, "fourth", "hi"))
        .unwrap()
    else {
        panic!("new turn expected")
    };
    let cancel = Cancellation::default();
    cancel.cancel();
    assert_eq!(
        turn.prepare_conversation(&Conversation, &cancel)
            .err()
            .unwrap()
            .error
            .code,
        ErrorCode::Cancelled
    );
    drop(turn);
    assert!(
        sessions
            .get("owner", "history")
            .unwrap()
            .active_turn
            .is_none()
    );
}

#[test]
fn durable_history_replay_preserves_float_bits_and_text_without_residency() {
    let directory = tempfile::tempdir().unwrap();
    let open = || {
        Sessions::open(
            Default::default(),
            Default::default(),
            coheryn_api::session_store::PersistenceLimits {
                enabled: true,
                ..Default::default()
            },
            Budget::new(64 << 20),
            directory.path(),
        )
        .unwrap()
    };
    let owner = "b".repeat(64);
    let sessions = open();
    sessions
        .create(
            owner.clone(),
            CreateSession {
                task: None,
                model: "checkpoint".into(),
                alias: Some("history".into()),
                context: 64,
                system: Some("S".into()),
                overflow: OverflowPolicy::Reject,
                persist: true,
            },
            "a".repeat(64),
        )
        .unwrap();
    let request = submit(0, "first", "first first first");
    let BeginTurn::New(mut turn) = sessions.begin(&owner, "history", request.clone()).unwrap()
    else {
        panic!("new turn expected")
    };
    turn.prepare_conversation(&Conversation, &Default::default())
        .unwrap();
    let mut output = result(
        turn.watch
            .snapshot()
            .unwrap()
            .context
            .unwrap()
            .prompt_tokens,
    );
    output.first_token_ms = Some(14591.165042999999);
    output.decode_ms = 1.2345678901234567;
    output.prefill_ms = f64::from_bits(0x3fd5555555555555);
    let expected = [
        output.first_token_ms.unwrap().to_bits(),
        output.decode_ms.to_bits(),
        output.prefill_ms.to_bits(),
    ];
    turn.finish(&GenerationEvent::Finished(output)).unwrap();
    drop(turn);
    drop(sessions);
    let sessions = open();
    let BeginTurn::Replay(watch) = sessions.begin(&owner, "history", request).unwrap() else {
        panic!("stored turn must replay without a model")
    };
    let replay = watch.snapshot().unwrap().result.unwrap();
    assert_eq!(
        [
            replay.first_token_ms.unwrap().to_bits(),
            replay.decode_ms.to_bits(),
            replay.prefill_ms.to_bits()
        ],
        expected
    );
    assert_eq!(replay.text, "old answer old answer");
    assert_eq!(replay.token_ids, [1, 2, 3, 4]);
    assert_eq!(replay.reused_prompt_tokens, 0);
}

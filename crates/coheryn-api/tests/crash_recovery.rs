//! Actual process termination at durable receipt boundaries; no destructor-based recovery.
use coheryn_api::{
    scheduler::GenerationEvent,
    session_store::PersistenceLimits,
    sessions::{BeginTurn, CreateSession, Sessions, SubmitTurn, TurnStatus},
};
use coheryn_engine::{GenerationResult, memory::Budget};
use serde_json::json;
use std::{
    path::Path,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};
fn open(path: &Path) -> Sessions {
    Sessions::open(
        Default::default(),
        Default::default(),
        PersistenceLimits {
            enabled: true,
            ..Default::default()
        },
        Budget::new(64 << 20),
        path,
    )
    .unwrap()
}
fn turn(revision: u64, key: &str) -> SubmitTurn {
    serde_json::from_value(json!({"expected_revision":revision,"idempotency_key":key,"content":"saved বাংলা 日本語 input","settings":{"max_tokens":2}})).unwrap()
}
fn result() -> GenerationResult {
    serde_json::from_value(json!({"text":"saved","token_ids":[1],"prompt_tokens":3,"completion_tokens":1,"reused_prompt_tokens":0,"finish_reason":"length","tokenization_ms":0.,"prefill_ms":0.,"decode_ms":0.,"output_delivery_ms":0.,"first_token_ms":null,"memory":{"limit":0,"reserved":0,"peak_reserved":0,"session_buffer_capacity_bytes":0,"peak_session_buffer_capacity_bytes":0}})).unwrap()
}
#[test]
#[ignore = "subprocess entrypoint used only by the crash-recovery test"]
fn crash_child_probe() {
    let Some(root) = std::env::var_os("COHERYN_CRASH_FIXTURE") else {
        return;
    };
    let root = Path::new(&root);
    let sessions = open(root);
    sessions
        .create(
            "b".repeat(64),
            CreateSession {
                task: None,
                model: "fixture".into(),
                persist: true,
                alias: Some("durable".into()),
                context: 128,
                system: None,
                overflow: Default::default(),
            },
            "a".repeat(64),
        )
        .unwrap();
    let BeginTurn::New(mut first) = sessions
        .begin(&"b".repeat(64), "durable", turn(0, "committed"))
        .unwrap()
    else {
        panic!("new fixture turn")
    };
    first.finish(&GenerationEvent::Finished(result())).unwrap();
    drop(first);
    let pending = if std::env::var("COHERYN_CRASH_PHASE").unwrap() == "pending" {
        let BeginTurn::New(turn) = sessions
            .begin(&"b".repeat(64), "durable", turn(2, "pending"))
            .unwrap()
        else {
            panic!("new pending turn")
        };
        Some(turn)
    } else {
        None
    };
    std::fs::write(root.join("ready"), b"durable boundary reached").unwrap();
    // The parent kills this process while a live writer may exist. Neither this turn
    // nor the store gets a normal Drop/close path during the qualification.
    std::hint::black_box(&pending);
    loop {
        std::thread::park();
    }
}
struct KillOnDrop(Child);
impl Drop for KillOnDrop {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
#[test]
fn committed_receipts_survive_kill_and_pending_writers_recover_without_model_residency() {
    for phase in ["committed", "pending"] {
        let dir = tempfile::tempdir().unwrap();
        let mut child = KillOnDrop(
            Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "crash_child_probe", "--ignored", "--nocapture"])
                .env("COHERYN_CRASH_FIXTURE", dir.path())
                .env("COHERYN_CRASH_PHASE", phase)
                .stdout(Stdio::null())
                .stderr(Stdio::inherit())
                .spawn()
                .unwrap(),
        );
        let start = Instant::now();
        while !dir.path().join("ready").is_file() {
            assert!(
                start.elapsed() < Duration::from_secs(30),
                "child did not reach durable boundary"
            );
            assert!(
                child.0.try_wait().unwrap().is_none(),
                "fixture process exited early"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        child.0.kill().unwrap();
        assert!(!child.0.wait().unwrap().success());
        let sessions = open(dir.path());
        let view = sessions.get(&"b".repeat(64), "durable").unwrap();
        assert!(view.active_turn.is_none());
        assert!(!view.recovery_pending);
        let exported = sessions.export(&"b".repeat(64), "durable").unwrap();
        let committed = exported
            .turns
            .iter()
            .find(|t| t.key == "committed")
            .unwrap();
        assert!(committed.receipt.status == TurnStatus::Completed);
        assert_eq!(committed.receipt.result.as_ref().unwrap().text, "saved");
        let BeginTurn::Replay(watch) = sessions
            .begin(&"b".repeat(64), "durable", turn(0, "committed"))
            .unwrap()
        else {
            panic!("committed replay must not execute")
        };
        assert!(watch.snapshot().unwrap().status == TurnStatus::Completed);
        if phase == "pending" {
            let interrupted = exported.turns.iter().find(|t| t.key == "pending").unwrap();
            assert!(interrupted.receipt.status == TurnStatus::Cancelled);
            assert!(interrupted.receipt.result.is_none());
            let BeginTurn::Replay(watch) = sessions
                .begin(&"b".repeat(64), "durable", turn(2, "pending"))
                .unwrap()
            else {
                panic!("interrupted retry must replay cancellation")
            };
            assert!(watch.snapshot().unwrap().status == TurnStatus::Cancelled);
        }
        drop(sessions);
        let restarted = open(dir.path());
        assert!(
            restarted
                .get(&"b".repeat(64), "durable")
                .unwrap()
                .active_turn
                .is_none()
        );
    }
}

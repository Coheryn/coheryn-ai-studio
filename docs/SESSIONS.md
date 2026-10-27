# Managed text conversations — S1–S4 CPU preview

The native daemon supports owned conversations, session-safe prefix KV reuse and opt-in embedded history storage. These capabilities are qualified initially for the pinned Qwen text template on macOS ARM64 CPU. Qualified tool-call/result turns and pending-call persistence are documented in [TOOL_CALLING.md](TOOL_CALLING.md). Developer roles, images, constrained output and continuous batching remain later gates. Functions are never executed automatically. Stateless completion/chat/embedding endpoints and ordinary standalone chat retain their contracts.

After importing a model as `qwen` and loading it with context 512:

```sh
coheryn sessions create qwen --name project-alpha --context 512 --system 'Be brief.'
coheryn sessions send project-alpha --input 'Hello' --expected-revision 0 --idempotency-key first --max-tokens 16
coheryn sessions show project-alpha --offset 0 --limit 100
coheryn chat qwen --session project-alpha --resume --max-tokens 16
coheryn sessions list
```

`chat qwen --session new-project --context 512` creates an ephemeral conversation. An existing name requires `--resume`. Resume uses stored context/instructions; supplying `--context`, `--system` or `--overflow` is an error. Names and IDs allow 1–64 ASCII letters, digits, hyphens and underscores; message content is UTF-8. Managed sessions use the explicitly configured daemon, without silent startup or standalone fallback.

Every mutation checks a revision. Accepting a turn advances it once for the pending input and again for completion/failure/cancellation; an initial successful turn normally ends at revision 2. Read returned revisions rather than hard-coding them. One writer is permitted per session. Stale/concurrent mutations and reset/delete during execution conflict. Use opaque session IDs for automated retries because deleted aliases can be reassigned.

`sessions send` returns a JSON receipt with turn/session IDs, revision, exact context accounting and generation results. An identical idempotency key/request replays its saved receipt without inference, even after a durable restart. Changing the request conflicts; an in-progress duplicate conflicts. Keys last for the bounded conversation epoch and are cleared on reset. Replayed timing/memory fields describe the original execution.

```sh
coheryn sessions reset project-alpha --expected-revision CURRENT_REVISION
coheryn sessions reset project-alpha --expected-revision CURRENT_REVISION --clear-instructions
coheryn sessions delete project-alpha --expected-revision CURRENT_REVISION
```

Reset preserves system instructions by default and starts a new epoch. Clearing stored instructions still permits the model's qualified default-system behavior. Delete releases history, replay and cache state. Its bounded control-plane request remains usable under workload budget exhaustion.

## Prefix KV reuse

Each session owns an isolated cache. Reuse requires the same loaded model instance, validated model/header and tokenizer identity, context capacity, execution settings and an exact token prefix. The final prompt token is recomputed to obtain logits. Generation results expose `reused_prompt_tokens`; session views expose `cached_tokens` and `cache_reserved_bytes`. These fields report cache state, not a performance guarantee.

The cache retains KV rows and a bounded token ledger; it holds neither model weights nor execution scratch/workers. Its reservation transfers atomically between idle and active execution, avoiding duplicate admission charges. A weak identity prevents cached state from keeping unloaded weights alive. Model unload/reload, incompatible settings, reset/delete, cancellation, errors, idle expiry and memory pressure invalidate or evict it. Prefix edits reuse only the identical leading tokens. Eviction changes compute cost, not authoritative history. No cache is shared between conversations or persisted to disk.

Cache defaults: `serve --idle-cache-bytes 268435456 --session-cache-bytes 67108864 --cache-idle-timeout 300`. The global ceiling also intersects one quarter of the engine budget. Zero disables retention. Oldest idle caches are evicted first; active worker reservations remain owned until acknowledgement.

## Opt-in durable histories

Persistence defaults off. A disabled daemon neither opens nor repairs an existing conversation database. Ephemeral sessions remain available regardless of the database's contents.

```sh
coheryn serve --session-persistence
# In another shell, using the same configured data directory and endpoint:
coheryn sessions create qwen --name durable-project --context 512 --persist
coheryn chat qwen --session another-project --context 512 --persist
coheryn sessions save project-alpha --expected-revision CURRENT_REVISION
coheryn sessions export durable-project --output conversation.json
coheryn sessions import conversation.json --name restored-copy --persist
```

`--persist` is creation-only; resume follows stored persistence. `sessions save` explicitly promotes an owned ephemeral conversation and advances its revision. Without `--persist`, import creates an ephemeral copy. Saving while persistence is disabled fails explicitly.

The version-2 SQLite store (with an additive transactional migration from version 1) is fixed at `DATA_DIR/conversations/history.sqlite`. It stores text, tool calls/results, companion-template identity, model digest, context policy, revisions, pending-turn records and bounded replay receipts. It stores no weights or KV tensors. Metadata loads at startup; history is read in bounded pages and hydrated only for operations that need it. Idle clean durable histories can leave RAM while remaining on disk. Restart requires explicit model loading with the same digest before new inference; inspection/export/replay need no loaded weights.

Transactions commit pending input before scheduling and completed history before publishing a successful receipt. Interrupted pending turns recover as cancelled; they are never automatically executed. Failed commits do not report saved success. If cancellation cleanup also cannot commit, the writer is released with `storage_error` and recoverable durable pending state. Corrupt, unrelated or newer-format databases are preserved and rejected when persistence is enabled; back up existing data before manual recovery. No destructive automatic reset is performed; only the validated version-1-to-2 migration is supported.

Defaults: `--session-store-bytes 1073741824`, `--session-owner-bytes 536870912`, `--session-retention-seconds 2592000`. Retention uses the last durable update, is enforced lazily on session operations and never expires a live writer. Half the global physical quota is reserved for the database and half for its rollback journal; usable logical payload can therefore be smaller. Per-owner logical quotas also apply. A 16 MiB engine reservation covers the store's bounded baseline; session hydration, responses and transfer buffers have additional reservations. Disk quotas and engine-controlled allocation limits are separate from RSS and physical residency.

On Unix, the store directory is private (0700), database private (0600), and symlink/nonregular/extra-hardlink database paths are rejected. Windows ACL and platform qualification remain release work. File exports use private temporary files, synchronization and same-directory atomic publication; existing output requires `--overwrite`. Imports are bounded, validated, inert JSON assigned a new session ID and the caller's owner. They cannot transfer another credential's identity or load models. Histories/exports contain plaintext prompts and replies; manage local filesystem access accordingly.

## API contracts

All routes require an inference or administrative Bearer credential. Its fingerprint defines ownership. Sharing a credential shares sessions; administrative credentials do not expose another credential's histories. Restart with the same credentials preserves access. Credential rotation revokes old access without migrating ownership; principal migration is not implemented.

| Method and path | Body / result |
|---|---|
| `POST /coheryn/v1/sessions` | `{ "model": "qwen", "alias": "project-alpha", "context": 512, "system": "Be brief.", "overflow": "reject", "persist": false }` |
| `GET /coheryn/v1/sessions` | Owned summaries without message bodies |
| `GET /coheryn/v1/sessions/{id-or-alias}?offset=0&limit=100` | One page, IDs/statuses, digest, revision, epoch, limits and cache state; `next_offset` when more messages exist |
| `POST /coheryn/v1/sessions/{id-or-alias}/messages` | Revision-checked idempotent user turn; JSON receipt or native SSE |
| `POST /coheryn/v1/sessions/{id-or-alias}/reset` | `{ "expected_revision": 2, "clear_instructions": false }` |
| `DELETE /coheryn/v1/sessions/{id-or-alias}` | `{ "expected_revision": 3 }`; maximum 1 KiB request body |
| `POST /coheryn/v1/sessions/{id-or-alias}/save` | `{ "expected_revision": 2 }`; promote ephemeral to durable |
| `GET /coheryn/v1/sessions/{id-or-alias}/export` | Complete bounded version-1 text or version-2 tool document; refuses active writers |
| `POST /coheryn/v1/sessions/import` | `{ "document": { ... }, "name": "new-copy", "persist": false }`; validates and assigns new ownership/ID |

Example turn:

```json
{
  "expected_revision": 0,
  "idempotency_key": "first",
  "content": "Explain ownership in Rust.",
  "settings": {
    "max_tokens": 64,
    "temperature": 0,
    "top_p": 1,
    "top_k": 0,
    "seed": 0,
    "stop": [],
    "profile": "balanced"
  },
  "stream": true
}
```

Settings also accept qualified `threads` and request `max_memory`. Context/model/instructions come from the session. Unsupported fields, roles and configurations fail explicitly. Message pages default to 100 and accept 1–100; `message_count` describes the complete history. The export route returns the complete bounded document rather than a page. Remote routes accept data, never arbitrary server filesystem paths.

Native SSE emits `data: {"token": ...}`, a terminal committed receipt with `result` and context metadata, then `data: [DONE]`. Error envelopes also terminate with `[DONE]`. Replay returns its cached JSON receipt even if the original turn streamed. These routes do not change OpenAI stateless SSE contracts. Disconnect/deadline/shutdown triggers cooperative cancellation; leases and reservations survive until worker acknowledgement. Partial output is not saved as a completed assistant turn. SQLite operations run on blocking workers rather than the async network reactor.

Before execution, the qualified template is rendered/tokenized exactly. Input plus requested output must fit the bound context. Default overflow rejects. Explicit `oldest-complete-turns` excludes whole completed turns from active context while preserving instructions/current input. Receipts identify included/excluded messages and exact counts. Failed/cancelled inputs remain inspectable but are excluded from later inference. No implicit summarization or context/output reduction occurs.

## Limits and evidence

Session defaults remain `--max-sessions 128`, `--sessions-per-owner 64`, `--session-history-bytes 1048576`, `--session-messages 256`, `--session-turn-keys 128`, `--session-idle-timeout 86400`. Counts include failed inputs/replay data. Idle ephemeral sessions expire; idle durable sessions release RAM but obey separate disk retention. Inspection refreshes idle RAM lifetime; listing does not. Store documents are capped at 8 MiB; transfer requests/files at 16 MiB. Array counts for histories, replay records, context IDs and stops are checked during JSON parsing before full materialization. Oversized histories are rejected rather than silently retained without bounds.

Reservations cover retained histories/results, bounded working copies, serialization, KV and execution state. They are not RSS measurements or OS caps. Request `max_memory` includes the managed working copy plus full model/execution estimates; cached allocations are credited only in the shared atomic reservation. Retained histories also count against the global budget. Fixed network/control-plane overhead remains separate.

Tests cover nonconstant-graph KV/logit parity, edits/settings/model invalidation, cache pressure/TTL/reset/unload, Unicode histories, isolation, restart/replay, quota failures, failed commits, corruption/newer formats, private paths, retention, bounded lazy pages and CLI export/import/resume. `tools/check_session_state.py` checks the actual pinned FP16 file, preserved independent greedy prefix, matched native fresh/reused continuation, exact durable restart/replay without weights and explicit cold-KV rebuild. Evidence: `fixtures/qualification/s2-s3-service.json` and `s2-s3-verification.json`; previous S1 evidence is retained. The qualification harness explicitly allows 600 seconds per generation. One final-binary run reached the default 300-second deadline on the longer restored FP16 prompt and cancelled with HTTP 408; this CPU path remains slow. Broader crash/fuzz/security campaigns, models/platforms, strict constraints, additional tool combinations, vision and Metal remain open release gates.

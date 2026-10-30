# Qwen3.5 managed conversations

The experimental native CPU text profile supports owner-scoped managed histories through the existing CLI and session API. Qwen2 GGUF histories and image-pair histories retain their existing behavior. [MANAGED_CHECKPOINTS.md](MANAGED_CHECKPOINTS.md) describes source acquisition, import, integrity and explicit model loading; [SESSIONS.md](SESSIONS.md) defines the shared ownership, revision, replay, persistence and quota contract.

```sh
./target/release/coheryn serve --session-persistence --max-memory 805306368
# In another terminal, with the same data directory and managed q35 source:
./target/release/coheryn models load q35 --context 256
./target/release/coheryn sessions create q35 --name notes --context 256 --persist
./target/release/coheryn sessions send notes --input 'Say hello.' \
  --expected-revision 0 --idempotency-key first --max-tokens 4 --json
./target/release/coheryn sessions show notes --json
```

Use the returned revision for each new turn. Repeating the same idempotency key requires identical content, settings, revision and stream flag; it returns the stored terminal receipt. Interactive `chat q35 --session notes --resume` uses this history interface. Explicit load is still required for new inference after restart.

For worker/profile controls and automation, save a bounded request JSON and use `sessions submit notes --request turn.json` (or add `--stream`):

```json
{
  "expected_revision": 2,
  "idempotency_key": "followup",
  "content": "你好. বাংলা again.",
  "settings": {
    "max_tokens": 4,
    "profile": "balanced",
    "threads": 2
  }
}
```

## Execution and resource ownership

Shared `TextConversation` preparation validates the selected native profile, renders the bounded history, counts its tokens and records exact included/excluded message IDs. Counting can exceed the loaded execution context so `oldest-complete-turns` can remove whole older turns before admission; actual execution still rejects excess context. Protected instructions and the current turn cannot be silently discarded. Exclusion changes the execution context, preserving the authoritative stored messages.

Qwen3.5 can retain an exact-prefix snapshot of every full-attention KV buffer, recurrent matrix and convolution cursor/history before the last assistant-start marker. The next compatible turn restores it only when the complete saved token ledger matches the new prompt prefix and model instance, context, worker count and budget agree. Partial rewind is unsupported. `reused_prompt_tokens`, `cached_tokens` and `cache_reserved_bytes` report actual reuse and retained capacity. The snapshot has its own reservation and weak model identity; it cannot pin unloaded weights. Per-session/global caps, idle expiry, pressure, reset, deletion and unload evict it. Insufficient optional cache capacity falls back to ordinary full prefill with identical semantics; it never changes requested context.

Text lanes share decoded weight tiles through a bounded continuous coordinator. Convolution, recurrent state, KV positions, sampling, cancellation and output channels remain independent. Matrix-tile checkpoints release the shared FIFO CPU lease after workers join; slow output delivery holds no compute permit. Checkpoint inspection advertises these capabilities independently from exact-file qualification. Image requests do not use the text prefix cache.

Before queueing, the scheduler reserves complete session/request/profile headroom and accounts for the writer's history/receipt reservation plus transport in request memory limits. The same FIFO CPU gate and joined-worker checkpoints serve stateless and managed generation. No compute permit is held during output backpressure. The writer lease remains with scheduling until cancellation acknowledgement and durable finalization; a successful receipt is emitted only after history commit. Failed admission, preparation, generation or persistence cannot publish a successful turn.

Unsupported sampling, tools, schema constraints and custom stops fail explicitly. Profile control validation occurs before declaring a new pending turn. Existing resource ceilings, output reserve, template limits and exact tokenizer/template checks still apply. Qwen3.5 also supports explicit image sessions through [IMAGE_SESSIONS.md](IMAGE_SESSIONS.md). Existing text sessions retain their task and reject attachments; new image sessions use their own bounded history preparation.

## Persistence and identity

Persist/save/export/import/reset/delete use the existing bounded history store and portable text document formats. There is no new database schema, copied weight payload or serialized KV/recurrent state. JSON parsing preserves receipt floats exactly across durable serialization. Complete managed source-set SHA-256 binds the session to every source file and its native profile. Replacing a model under the same ID cannot silently rebind an existing history; mismatched identity fails and releases the writer.

Stored receipts and history can be inspected/replayed after unload or restart while the source remains registered, before weights load. New inference requires a compatible resident with the identical source-set identity and sufficient context. Original failed/cancelled inputs retain their status and are excluded from later generation. Portable histories are inert user data; import does not execute code, load models or attest to the truth of earlier conversation content.

## Verification

Ordinary tests exercise the generic history contract with an independent bounded renderer/token counter: whole-turn exclusion, Unicode preservation, protected instructions, exact accounting, overflow rejection, unsupported options, cancellation, replay and writer/resource cleanup. Existing GGUF, embedding, image and persistence regressions remain required.

After acquiring the pinned source and building the release workspace:

```sh
python3 tools/check_managed_checkpoint.py --sessions \
  --report fixtures/qualification/NEW-managed-sessions-runtime.json
```

This historical full-prefill regression explicitly disables idle caches. The opt-in uses a temporary private managed copy and local daemon. It checks frozen first-turn tokens/text and SSE, Unicode CLI follow-up versus identical stateless history, durable restart and unloaded replay, portable CLI export/import, full identity mismatch rejection and cancellation/writer cleanup. It also reruns the managed source-set/stateless integrity checks. Existing qualification files are never overwritten. The first turn has independent frozen evidence; follow-up parity tests integration against the same native executor. This is not new independent full-model numerical or language-proficiency qualification.

CPU greedy text remains experimental. Broader image qualification, model/layout/backend/platform support and sustained fairness/backpressure/crash/security campaigns remain open.

# Native tool calling — S4 CPU preview

Coheryn supports function-call proposals through `/v1/chat/completions`, native CLI requests and managed sessions. Functions are data: Coheryn never runs a suggested function, shell command, plugin or network/database operation. The calling application validates arguments and authorizes execution before returning results.

Initial model qualification is the exact published `qwen2.5-0.5b-instruct-q4_k_m.gguf`, SHA-256 `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`, on macOS ARM64 CPU with F32 KV. See `fixtures/tools/manifest.json` and `fixtures/qualification/s4-service.json`. Other files, including other precisions/sizes in the same family, require separate tool qualification. A compatible template alone does not establish model behavior. Managed requests and standalone tool requests check the exact file identity; unknown combinations fail explicitly.

The `qwen-tools-v1` companion template is an explicit, pinned inert asset. The publisher's original GGUF template has a double-braced schematic JSON example; an independent run copied it into invalid JSON. The original template and negative output remain preserved. The companion corrects that example and renders already validated argument strings without numeric/whitespace normalization. Ordinary text requests retain their original GGUF template. Native decoding exposes only the qualified opening/closing tool delimiters to the parser; ordinary special-token suppression and EOS behavior are preserved.

## CLI

The companion now includes a non-rendering license/modification comment. Its current file/profile identity is recorded separately from the frozen reference body in the manifest. Exact native prompt tests cover all five frozen reference cases. Historical service/binary hashes remain historical; the notice-only change does not rewrite independent oracle outputs. See [licensing records](../resources/licensing/README.md).

With the daemon running, import and load the qualified file using the usual local model-management commands, for example:

```sh
coheryn models import ./qwen2.5-0.5b-instruct-q4_k_m.gguf --name tools-chat \
  --sha256 74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db
coheryn models load tools-chat --context 512
coheryn chat tools-chat --request request.json --json
coheryn chat tools-chat --request request.json --stream --json
coheryn chat ./qwen2.5-0.5b-instruct-q4_k_m.gguf --standalone --request request.json --json
```

A request file is a bounded chat JSON object. Omit `model` when using it across CLI modes; the positional model selects the target. Managed requests reject conflicting file/CLI model IDs. Generation settings belong in the request file; interactive context/profile/thread/token flags conflict with `--request`. `--max-memory` still limits the execution budget. The new request mode is headless and supports JSON automation; ordinary interactive chat remains available.

```json
{
  "messages": [{"role": "user", "content": "Use add to calculate 2 + 2."}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "add",
      "description": "Add two integers.",
      "parameters": {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
        "additionalProperties": false
      }
    }
  }],
  "tool_choice": "required",
  "parallel_tool_calls": false,
  "context": 512,
  "max_tokens": 96,
  "temperature": 0
}
```

Without streaming, CLI returns a native generation result, including `tool_calls`, `finish_reason`, raw generated token IDs and usage/timings. With streaming it emits JSON Lines: `token`, `tool_call` delta and a terminal `result`. Failure returns a structured error and nonzero exit status. A partial delta is not a successful proposal; wait for successful completion before executing anything. Unsupported configurations retain exit status 4, malformed requests/output use 2, and resource exhaustion uses 5.

## OpenAI-compatible chat

Send the same body, adding `model`, to `POST /v1/chat/completions` with a Bearer credential. Function declarations support names, optional descriptions, object parameter schemas and `strict: false`/omitted. Choices are `none`, `auto`, `required`, or a named selection:

```json
{"type": "function", "function": {"name": "add"}}
```

`none` disables outbound tool interpretation and conditioning. `auto` permits text or calls. `required` and named selections validate the completed response and return an error if the model does not comply. Named selection conditions the model on the selected declaration. This preview does not force decoder-level constraints; S5 remains mandatory work. `parallel_tool_calls: false` rejects multiple generated calls; qualified multiple-call framing supports up to eight calls.

A completed proposal has `finish_reason: "tool_calls"` and assistant `tool_calls` entries with unique IDs, `type: "function"`, exact declared names and JSON argument strings. Pure-call content can be null. Sampling usage includes generated control/EOS tokens. Arguments retain their decoded JSON value bytes, including large integers, escaping and internal whitespace. Validation checks JSON object syntax, depth, declaration membership, selected function, call count and identifiers. Parameter schemas condition the model; their semantic constraints are **not guaranteed**. `strict: true` is rejected until S5 supplies a qualified constraint decoder.

With `stream: true`, deltas contain call indices, initial ID/type/name and ordered argument fragments, followed by a successful finish event, optional usage and `[DONE]`. Calls are buffered until their closing marker and validation, then arguments are delivered in UTF-8 fragments of at most 256 bytes. This is incremental delivery of validated calls, not per-token speculative argument delivery. A later error/cancellation terminates explicitly and must not authorize earlier fragments. Pure-call streams omit text content, matching null complete content. Completion/native text endpoints retain their contracts; tool declarations require chat messages.

For stateless continuation, return the assistant proposal and matching tool messages in `messages`. Tool-result content is a string. This Qwen profile requires every pending result **in original call order**, because its template renders consecutive results without IDs in model text. Missing, duplicate, orphaned, reversed or interleaved results fail explicitly. IDs remain present in the API/history. Developer roles, images, strict schemas, undeclared selections and custom stops combined with tool generation are unqualified.

## Managed conversations and restart

Use `sessions submit` for structured user or tool-result turns:

```sh
coheryn serve --session-persistence
coheryn sessions create tools-chat --name work --context 512 --persist
coheryn sessions submit work --request first-turn.json --stream
coheryn sessions show work
coheryn sessions submit work --request tool-results.json
```

An initial turn file uses the existing managed envelope:

```json
{
  "expected_revision": 0,
  "idempotency_key": "first",
  "content": "Use add to calculate 2 + 2.",
  "settings": {
    "max_tokens": 96,
    "tools": [{"type": "function", "function": {
      "name": "add", "description": "Add two integers.",
      "parameters": {"type": "object", "properties": {
        "a": {"type": "integer"}, "b": {"type": "integer"}
      }, "required": ["a", "b"], "additionalProperties": false}
    }}],
    "tool_choice": "required"
  }
}
```

The returned generation turn is completed while `pending_tool_calls` identifies results still awaited. An empty pending list may be omitted in JSON. Supply all matching results before another user turn:

```json
{
  "expected_revision": 2,
  "idempotency_key": "results",
  "tool_results": [{"tool_call_id": "RETURNED_CALL_ID", "content": "4"}],
  "settings": {"max_tokens": 32}
}
```

Continuation inherits the previous declarations when omitted; tool choice defaults to auto. Explicit settings still undergo validation. Revisions advance at pending-input and terminal-commit boundaries. Identical retries replay saved results with the same IDs without inference. Completed tool-call/result chains are excluded from context as whole groups; a required current group is never split to fit.

Persistent histories retain calls, results, replay receipts and the exact companion-template profile/hash. Pending proposals survive restart without tool execution or automatic model loading. KV remains disposable and cold after restart. Import assigns new session ownership/ID and validates call/result relationships; imported proposals remain untrusted data, not execution authorization. Unknown template identities are rejected rather than silently reframed.

The embedded table format advances to version 2 through an additive, transactional version-1 layout migration. Legacy text records and replay data remain readable. Text-only exports retain version 1; tool-bearing/settings-bearing exports use version 2. Unsupported future formats remain preserved and rejected. Version-1 binaries reject version-2 databases; retain a backup if you need to roll back the runtime.

## Limits and evidence

Limits are 16 declarations, 64-byte ASCII function names/IDs, 4 KiB descriptions, 64 KiB total declarations, 256 aggregate schema nodes at depth 8, eight calls, 64 KiB arguments per call at depth 16 and one MiB decoded output. Schema references are rejected and never fetched. Session history/keys/pages, store quotas, deadlines, queue limits and admission remain applicable. Generation ingress and planner payloads have bounded reservations; managed declaration retention receives a conservative 2 MiB allowance per declaration-bearing record. Reservations are distinct from RSS and full allocator instrumentation.

`fixtures/tools/qwen2.5-reference.json` freezes five independent prompt/token cases, single/multiple greedy calls and a result continuation. `tools/check_tool_calling.py` consumes those inert outputs through the native binary, the optional standard SDK and CLI; it never captures expected results from Coheryn or invokes an external model executor. The independent capture sources remain separate research under `resources/tool-assessment/`.

Tests cover UTF-8 fragmentation, raw numeric/escaped arguments, incomplete/malformed/oversized calls, schema/choice limits, unsupported files, special-token policy, complete/SSE parity, pending-ID ordering, owner isolation, context groups, quota rollback, migration and restart/replay. Two fuzz targets compile; deterministic mutations supplement unit fixtures. Sustained fuzz/security/crash campaigns, strict schema guarantees, other tool model/backend/platform combinations, vision and Metal remain release work.

# G2 CLI and HTTP service

The service uses the same independent Rust engine as standalone execution. This is an experimental text-generation and embedding OpenAI-compatible subset. The G4 CPU planner, packed quantized kernels and controlled row parallelism are implemented. E2 adds native embeddings; GPU and vision remain later gates. See [planner contracts and M0 status](PLANNER.md).

The mandatory release scope includes image attachments/multimodal chat and independent embeddings via `/v1/embeddings` and `coheryn embed`. Native CPU embeddings and the embeddings route are implemented; image execution remains pending. This guide documents current behavior; M0–M2 and E0–E3 govern the additional qualified contracts. The E2 guide identifies the implemented embedding/RAG interfaces and remaining qualification limits.

## Run the daemon

```sh
target/release/coheryn serve
```

The default address is `127.0.0.1:11450`. The default data directory is `~/Library/Application Support/CoherynAIStudio` on macOS, `$XDG_DATA_HOME/coheryn` or `~/.local/share/coheryn` on Linux, and `%LOCALAPPDATA%/CoherynAIStudio` on Windows. Use global `--data-dir PATH` to choose another directory. On Unix, an existing storage directory must be private (mode 0700); the daemon creates new storage directories with that mode. One daemon owns a directory through an exclusive registry lock.

The daemon creates distinct inference and administrative credentials in `credentials.json`, with mode 0600 on Unix. It logs the credential-file location, never the credentials. `COHERYN_API_TOKEN`, `COHERYN_ADMIN_TOKEN`, `--api-token-file` and `--admin-token-file` can override the stored values without changing the vault. Credentials must contain 32–4096 non-whitespace ASCII bytes and the server roles must have different values. Clients using overrides need the same overrides.

Local CLI clients read the vault automatically. A remote `--endpoint https://HOST` requires explicitly supplied environment credentials or token files; the CLI does not send its local vault to a remote host. Non-loopback server binding is explicit through `serve --host`. Use TLS termination at a reverse proxy for remote access. No CORS access is enabled.

## Manage and execute a model

In a second terminal, using the same `--data-dir` when customized:

```sh
target/release/coheryn models import .cache/models/qwen2.5-0.5b-instruct-fp16.gguf --name qwen05
target/release/coheryn models load qwen05 --context 512
target/release/coheryn run qwen05 --prompt Hello --context 512 --max-tokens 8 --stream
target/release/coheryn chat qwen05 --context 512
target/release/coheryn status --json
target/release/coheryn models unload qwen05
target/release/coheryn models remove qwen05
```

`models register` aliases `models import`: it copies a local file into managed storage through an authenticated upload. Imports can include `--sha256 HASH`. `models pull URL --name ID --sha256 HASH` downloads a public HTTPS file with a mandatory expected checksum. IDs contain at most 64 lowercase letters, digits, dots, underscores or hyphens, beginning with a letter/digit.

`run` and `chat` use the daemon by default. They never silently start a separate engine. `--standalone` explicitly loads a separate process-local model from a GGUF path. Local existing paths and explicit paths remain valid for `models inspect` and `models validate`; a managed ID uses the daemon. Interactive chat uses `/exit`, `/quit` or EOF to exit. For machine-readable chat, use `run --messages messages.json --json`.

`bench --model ID --context 512 --profile balanced` benchmarks through the daemon; `--standalone` uses a local path. `plan ID --context 512 --profile balanced --concurrency 2` estimates without loading weights. `reference`, `performance`, `balanced` and `low-memory` are supported by plan/run/chat/bench; balanced is the generation/planning default. The portable CPU backend, selected worker count, prefill chunk size and F32 cache remain explicit in each plan. CLI plan/run/chat/bench accept `--threads`; estimate and generation JSON accept optional `threads` (1–16, bounded by the configured CPU share). Reference/low-memory require one.

## Routes and credentials

All protected routes require `Authorization: Bearer TOKEN`. Administrative credentials can also perform inference. Inference credentials cannot perform administrative operations.

| Method and route | Role | Input/result |
|---|---|---|
| `GET /health` | Public | Minimal liveness result |
| `GET /v1/models` | Inference | Registered local model IDs; loading is explicit |
| `POST /v1/completions` | Inference | Text prompt, JSON result or SSE |
| `POST /v1/chat/completions` | Inference | Text messages, JSON result or SSE |
| `POST /v1/embeddings` | Inference | Single/batch strings; ordered float/Base64 vectors, usage and identity headers |
| `POST /api/v1/generate` | Inference | Native generation result/events for CLI parity |
| `GET /api/v1/models` | Admin | Registry records and resident state |
| `GET /api/v1/models/{id}/inspect` | Admin | Descriptor and compatibility report |
| `POST /api/v1/models/validate` | Admin | `{"model":"ID","deep":true}`; no remote filesystem paths |
| `POST /api/v1/models/import?name=ID&sha256=HASH` | Admin | Bounded `application/octet-stream` upload; hash optional |
| `POST /api/v1/models/pull` | Admin | `{"name":"ID","url":"https://...","sha256":"HASH"}` |
| `POST /api/v1/models/load` | Admin | `{"model":"ID","context":4096}` |
| `POST /api/v1/models/unload` | Admin | `{"model":"ID"}` |
| `DELETE /api/v1/models/{id}` | Admin | Remove an unloaded model, including an orphaned managed file |
| `POST /api/v1/plans/estimate` | Admin | `{"model":"ID","context":4096,"profile":"balanced","concurrency":1}` |
| `GET /api/v1/system/resources` | Admin | Observed RAM/swap and separate engine reservations |
| `GET /api/v1/server/status` | Admin | Residents, scheduler counters and engine reservations |
| `GET /api/v1/server/metrics` | Admin | Same initial structured metrics snapshot |

JSON requests require `Content-Type: application/json`. Unsupported fields fail explicitly. Compatibility/import failures include structured diagnostics when parsing reaches compatibility validation. Deep validation scans all executable float/packed weights using bounded workspace.

## Supported completion fields

- `model`: registered, explicitly loaded ID.
- `prompt`: nonempty string on completions, or `messages`: 1–256 objects with string `role`/`content` on chat. Supported roles are `system`, `user`, `assistant`.
- `max_tokens`: positive integer, default 128. `max_completion_tokens` is an alias; supplying both is rejected.
- `temperature`: numeric 0–2, default 0 for reproducible greedy execution; `top_p`: numeric `(0,1]`, default 1; `top_k`: nonnegative integer, default 0; `seed`: unsigned integer, default 0.
- `stop`: string or array of at most 16 nonempty strings of at most 4096 bytes.
- `stream`: boolean, default false; `n`: only 1.
- `stream_options.include_usage`: boolean, only with streaming.
- Coheryn extensions: `context` cannot exceed loaded capacity; `max_memory` adds a conservative per-request cap over model metadata, session, transport allowance and profile headroom. `profile` accepts reference/performance/balanced/low-memory (default balanced). `threads` optionally selects a bounded compute worker count; each worker stack is included in admission. The daemon's global atomic budget also applies.

Tools, function calls, multimodal content, multiple choices, log-probability requests and structured-output modes are unsupported. Sampling and required numeric fields reject JSON null; optional `context`, `max_memory` and `threads` treat null as omitted. No request changes a specified context or sampling setting to make it fit.

OpenAI-shaped responses include IDs, object type, creation epoch, model, one choice, finish reason and usage. Chat streaming uses assistant-role/content deltas and a finish chunk. When requested, usage is null on preceding chunks and reported in a separate chunk with empty choices before `data: [DONE]`, following the [official streaming schema](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events) and [include-usage contract](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create). The legacy completion shape follows the [official completion reference](https://developers.openai.com/api/reference/resources/completions/methods/create).

The native streaming endpoint sends `{"token":{"token_id":...,"text":...}}`, then `{"result":...}` and `[DONE]`. The native result preserves engine token IDs, timings, finish reason and reservation statistics. Engine timings exclude HTTP queue/transport time. A streaming failure sends an error object and `[DONE]`; it does not fabricate a successful finish. Disconnected consumers cancel their request.

## Resource and lifecycle behavior

Defaults are one active generation, eight queued requests, four loaded models, a 1 GiB global reservation budget, a maximum requested context of 8192, a 4 GiB model-file ceiling, and a 20 GiB managed-weight quota. Flags configure memory/context/queue/model/disk limits; `serve --max-concurrency N` permits 1–16 active sessions, bounded by logical CPUs; the default remains one. Low-memory requests acquire exclusive execution; balanced limits its own concurrent requests to two. Admission reserves the full session/payload/profile headroom in one transaction before queueing, and transfers session ownership into execution without a release/reacquire gap. HTTP serialization/body buffers remain bounded separately. Engine session vector capacities are observed separately from reservations; this is not an OS-wide RAM cap or complete allocator instrumentation.

There are at most 64 in-flight authenticated HTTP responses, including their streaming-body lifetime. Ordinary generation JSON input is limited to one MiB; session imports have a separate 16 MiB limit; headers to 64 entries/8192 bytes. Token delivery holds at most sixteen events plus a reserved terminal event. JSON-body/upload idle timeout is 30 seconds, queue timeout 30 seconds, consumer backpressure timeout 30 seconds, generation timeout 300 seconds, and model acquisition timeout 600 seconds. Health is public but minimal. JSON client responses/SSE frames are bounded to 16 MiB.

Plan estimates accept optional bounded image dimensions/profile descriptors, perform no URL fetch and return inadmissible/null vision costs until image execution is qualified. No generation endpoint accepts image parts yet.

Loaded model descriptors, tokenizers and the read-only weight source are shared through `Arc<Model>`. KV, scratch and I/O buffers belong to each session. Weight cursor operations are synchronized. Queued and active requests pin residency; unload returns 409 until pins release. Remove requires explicit unload. A conflicting context reload returns 409 instead of silently changing running settings. SIGINT/SIGTERM cancel queued/active work and begin graceful shutdown; stalled HTTP connections have a bounded drain interval.

Downloads require HTTPS on port 443, validate every redirect and resolved address, reject local/private/reserved destinations, pin validated DNS answers and ignore environment proxies. Uploads/downloads use private temporary files, bounded writes, SHA-256 verification and atomic installation. The daemon rejects corrupt hashes on load. Interrupted transfers are removed; stale owned temporary files are cleaned after acquiring the registry lock. A crash between weight and record installation can leave an orphan charged to quota; an administrator can remove its ID explicitly.

## Errors and verification

Application errors use `{"error":{"message":...,"type":...,"code":...,"param":null,"details":...}}`. Statuses include 400 malformed/unsupported, 401 missing/invalid credential, 403 wrong role, 404 missing model, 405 unsupported method, 409 pinned/conflicting state, 413/431 input limits, 429 queue/HTTP capacity, and 503 resource/unloaded/shutdown conditions. A cancelled or timed-out in-progress request returns 408 when a response can still be delivered.

CLI exit statuses add `6` credential/role failure, `7` conflict/full queue and `8` unavailable daemon to the standalone statuses documented in the README. Client connection failure never falls back to standalone.

`cargo test --workspace --locked` includes synthetic model tests over an actual loopback listener. `python3 tools/check_service.py` runs a Unix pinned-model CLI/API/SSE smoke test and writes `fixtures/qualification/g2-service.json`. It uses cached weights, temporary private storage and temporary credentials, then removes its model and shuts down its daemon. The optional ignored HTTPS test reads only the publisher file's GGUF magic. Sustained fuzzing, full platform packaging, optimized kernels and the full hardware-aware planner remain later work.

Integrated benchmark JSON reports `load_ms` for standalone metadata/tokenizer/storage initialization; daemon-client mode reports null because loading is a separate administration operation. Buffered tensor reads occur during execution. Per-run tokenization, prefill, decode and output-sink timing remain separate; cache conditions are labeled uncontrolled.

## E2 embeddings

`POST /v1/embeddings` now supports strings/string arrays, native dimensions, float/Base64 vectors, usage and bounded execution extensions. The daemon loads embedding models independently, shares scheduling/admission with generation and checks task compatibility. Local/managed inspect and validation expose the task and embedding configuration; managed inspection also exposes immutable embedding-space identity and operator limits. `coheryn embed`, `plan --task embedding` and `bench --task embedding` are available. See [EMBEDDING_API.md](EMBEDDING_API.md) for the exact field subset, defaults, headers, CLI atomic export and tested SDK/RAG interfaces.

## Managed sessions

The separate `/coheryn/v1/sessions` namespace provides owned ephemeral/opt-in durable histories, revision-checked/idempotent turns, isolated bounded prefix-KV reuse and native SSE. Save/export/import and paged inspection use this namespace; persistence requires `serve --session-persistence`. See [SESSIONS.md](SESSIONS.md) for exact supported fields, context policies, CLI commands and resource limits. Existing stateless routes remain independent from managed histories.

## S4 function-call proposals

`/v1/chat/completions` accepts bounded function declarations, `none`/`auto`/`required`/named choices, qualified multiple calls, assistant tool calls and ordered matching tool messages. Complete and SSE responses expose call IDs/indices/names/argument strings and `tool_calls` finish reasons. Exact-file qualification, the companion template, unsupported strict schemas/stops, complete-result validation and SDK evidence are described in [TOOL_CALLING.md](TOOL_CALLING.md). `tools` is a chat field; completion requests reject it. Ordinary text contracts and embedding interfaces remain available.

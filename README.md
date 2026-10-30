# Coheryn AI Studio

[Website](https://coheryntech.com/products/coheryn-ai-studio) · [Repository](https://github.com/Coheryn/coheryn-ai-studio)

Coheryn-authored code is licensed under [Apache 2.0](LICENSE). Included third-party assets retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md).

An independent Rust-native inference engine with GGUF execution and an experimental Safetensors text adapter. This repository provides an **experimental CPU CLI/API preview** with text generation, independent embeddings, managed text conversations, session-safe KV reuse and opt-in persistent histories. The [release checklist](docs/RELEASE_CHECKLIST.md) tracks the full mandatory scope. S0 provides automated source/dependency/artifact checks; S1–S6 have initial macOS ARM64 CPU evidence, including an exact-file tool profile, native JSON/schema constraints and expanded sampling controls. Native compatible CPU continuous batching has an initial checkpoint. Native PNG/JPEG preprocessing and the selected Qwen2-VL CPU vision encoder/merger have bounded, reference-tested foundations; see [image-stage status](docs/IMAGES.md) and [native vision scope](docs/VISION.md). The selected Qwen2-VL pair now has an [offline native image-to-text development runner](docs/MULTIMODAL.md). Product image attachments now use registered component pairs, bounded CLI/Base64/opt-in HTTPS acquisition and shared admission/cancellation; see [image CLI/API](docs/IMAGE_API.md). Managed image histories, opt-in save/resume and cooperative CPU scheduling now have an [image conversation checkpoint](docs/IMAGE_SESSIONS.md). Metal and broader batching/platform/release qualification remain open. Full RAG and vector storage remain optional. See [session commands and persistence](docs/SESSIONS.md) and [planner evidence and limits](docs/PLANNER.md).

Coheryn owns the Qwen2 and Qwen3.5 generation graphs, Nomic embedding graph, packed kernels and controlled CPU parallelism. Native build/test/runtime paths contain no external model executor. Tests consume frozen independent outputs; original capture source and provenance are retained as inert resource research.

The [native adapter registry](docs/ADAPTERS.md) now selects standalone text implementations. [Qwen3.5-0.8B development support](docs/QWEN35.md) uses the pinned official Safetensors checkpoint on CPU; its managed stateless text/chat API now shares checkpoint integrity, admission and cooperative scheduling; see [managed checkpoints](docs/MANAGED_CHECKPOINTS.md). Qwen3.5 [managed text histories](docs/QWEN35_SESSIONS.md) support persistence, replay and bounded hybrid prefix reuse. Qwen3.5 also has continuous text batching, metadata-only image plans and explicit persistent image conversations through the CLI/API. Local model inspection/validation and standalone text memory estimates now share a [versioned descriptor](docs/MODEL_METADATA.md).

## Build and test

Install Rust using the pinned `rust-toolchain.toml`, then:

```sh
cargo build --release --workspace --locked
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo fmt --all -- --check
```

Package maintenance follows [AGENT.md](AGENT.md); [dependency upgrade notes](docs/DEPENDENCIES.md) record verified versions, API migrations and upstream transitive constraints.

`./tools/cargo` runs the same commands using a workspace-local toolchain when `.cache/cargo` exists. CPU CI covers macOS ARM64, Linux x86-64, and Windows x86-64; configured CI is not evidence that those remote jobs have run.

## First native inference

For the prioritized Qwen3.5-0.8B standalone development profile:

```sh
python3 tools/fetch_checkpoint.py
target/release/coheryn run .cache/qwen3.5-0.8b-source \
  --standalone --messages fixtures/qwen35/smoke-messages.json \
  --context 256 --max-tokens 8 --threads 2 --stream
```

This CPU text profile is experimental; [supported options and remaining work](docs/QWEN35.md) are explicit. It does not yet use the managed daemon.

## GGUF regression and service model

Download the pinned publisher model into the ignored cache, checking its SHA-256 before atomic installation:

```sh
python3 tools/fetch_model.py
target/release/coheryn models inspect .cache/models/qwen2.5-0.5b-instruct-fp16.gguf --json
target/release/coheryn models validate .cache/models/qwen2.5-0.5b-instruct-fp16.gguf --deep --json
target/release/coheryn run .cache/models/qwen2.5-0.5b-instruct-fp16.gguf \
  --prompt 'Hello' --max-tokens 16 --context 512 --standalone --stream
```

For chat, provide `--messages conversation.json` instead of `--prompt`. The file is an array such as `[{"role":"user","content":"Hi"}]`. The pinned GGUF chat template supplies its own default system prompt. For qualified tool requests use `chat --request`; for the supported vision pair use `chat --image` as documented in [IMAGE_API.md](docs/IMAGE_API.md).

`--standalone` is explicit: it creates a separate process-local model instance. Default `run`/`chat` mode uses the daemon with a managed model ID. It never falls back to loading a second model. After acquisition, all inference works offline. There is no telemetry.

## Local service

Start `target/release/coheryn serve`, then in a second terminal:

```sh
target/release/coheryn models import .cache/models/qwen2.5-0.5b-instruct-fp16.gguf --name qwen05
target/release/coheryn models load qwen05 --context 512
target/release/coheryn run qwen05 --prompt Hello --context 512 --max-tokens 8 --stream
target/release/coheryn chat qwen05 --context 512
target/release/coheryn status --json
```

`serve --max-concurrency 2` enables up to two budget-admitted sessions sharing one model. `plan MODEL_ID --context 512 --profile low-memory --json` shows the selected CPU placement, buffers, headroom and rejection reasons without loading weights. `run`/`chat`/`bench` also accept `--profile` and bounded `--threads`; low-memory uses one thread, reduces I/O tiles and acquires exclusive execution. Thread defaults respect the configured CPU share.

The listener defaults to `127.0.0.1:11450`. Distinct inference/admin credentials are created in a private local vault and loaded automatically by local CLI clients. Use the same global `--data-dir PATH` for a custom daemon directory and its clients. Remote clients must supply environment credentials or token files explicitly. See [CLI/API guide](docs/API.md) for routes, supported OpenAI fields, authentication, limits and lifecycle behavior.

## Current execution contract

- GGUF v3 metadata and aligned tensor inspection; numeric execution supports little-endian F32/F16/Q8_0/Q4_0/Q5_0/Q4_K/Q6_K Qwen2. Big-endian containers can be inspected but are rejected for execution.
- Actual tensor layouts, dimensions, tokenizer pretokenizer, and approved template fingerprints determine compatibility. Q4_K_M is a recipe, not a tensor encoding. Only the exact files and configurations recorded in the compatibility matrix have model-level qualification; other encodings and graphs fail explicitly.
- Read-only buffered weight access, with one bounded I/O chunk, avoids a persistent full-weight/dequantized copy and avoids mutable-file mapping faults. Do not modify model files during use. Header and metadata-change checks detect common mutations; they are not a cryptographic guarantee against a concurrent malicious writer.
- F32 KV cache grows in bounded chunks within a pre-reserved context. Attention reuses a single linear score row and accumulator without a quadratic score matrix. Loaded models share immutable metadata/tokenizers/weight access; sessions own KV, scratch and I/O buffers. SSE has sixteen token slots plus a reserved terminal slot; slow/disconnected consumers cancel without unbounded growth.
- `--max-memory` is a conservative **reservation/admission budget**, not an OS memory cap or a measurement of every allocation made by third-party tokenizer/template libraries. Reservations and OS measurements are reported separately. CPU hardware-aware planning and profile-specific atomic admission are implemented. Engine session buffer capacities are observed separately; complete allocator accounting remains outstanding.
- G1 limits: one MiB input/rendered-template/output, 16 KiB template source, 256 chat messages, 16 stop strings of at most 4096 bytes, token text at most 4096 bytes. Requested context and generation limits are never silently reduced. SIGINT cancels between bounded compute units.

CLI errors have stable JSON codes. Exit statuses: `0` success, `1` internal, `2` invalid request/arguments, `3` I/O/missing/changed model, `4` compatibility/format, `5` resource exhaustion, `6` credentials, `7` conflict/full queue, `8` unavailable daemon, `130` cancellation. `--json` writes results to stdout and failures to stderr. `--deep` additionally validates the task-specific tokenizer (including a chat probe for generation models) and scans all supported weight payloads for non-finite values using bounded workspace.

Quantized CPU qualification and embedding reference preparation are documented in [QUANTIZATION.md](docs/QUANTIZATION.md) and [the E0 fixture guide](fixtures/embedding/README.md). Native embedding CLI/API and external RAG interfaces are documented in [EMBEDDING_API.md](docs/EMBEDDING_API.md). The selected Qwen2-VL image path has separate [CLI/API evidence](docs/IMAGE_API.md); The pinned Qwen3.5 source has separately qualified [still-image support](docs/QWEN35.md#still-image-input-and-scheduling).

## Native verification and reference evidence

Native tests consume frozen expected outputs without building or invoking another model executor. Original capture source, manifests and comparator provenance are preserved as inert research records under `resources/reference-archive/`; see [the independence policy](docs/INDEPENDENCE.md). New external assessments must run separately and exchange verified data only. Do not regenerate expected outputs from Coheryn to hide a regression.

```sh
python3 tools/check_repository_policy.py
python3 tools/check_embedding_fixtures.py
python3 tools/check_independence.py
python3 tools/check_service.py
```

The real-model service check requires previously acquired pinned weights and a release binary. Its cache conditions are uncontrolled. Unit and fixture checks work without model downloads. Reference-precision settings and frozen tolerances remain in the fixture manifests. Development F32 weights can be derived with `python3 tools/derive_f32.py`; this promotes existing binary16 values and is not original pretrained F32 precision.

See [implementation plan](docs/IMPLEMENTATION_PLAN.md), [validation evidence](docs/VALIDATION.md), [threat model](docs/THREAT_MODEL.md), and [release checklist](docs/RELEASE_CHECKLIST.md).

Development now prioritizes [qualifying the existing CPU capabilities](docs/CPU_RELEASE.md) before additional architectures or GPU execution. Reproducible candidate archives and native startup checks are distinct from production qualification. See [operation and recovery procedures](docs/OPERATIONS.md); the production label stays blocked until the required model, platform, security and operational evidence is reviewed.

The [pinned real-model CPU runner](docs/CPU_QUALIFICATION.md) combines independent numerical fixtures, extracted CLI and authenticated TCP/SSE checks. A manual workflow targets macOS ARM64, Linux x86-64 and Windows x86-64; successful execution evidence is required separately on each platform.

## Managed text sessions

After loading a model as `qwen`, create or resume an owned ephemeral conversation:

```sh
target/release/coheryn chat qwen --session project-alpha --context 512 --max-tokens 16
target/release/coheryn chat qwen --session project-alpha --resume --max-tokens 16
target/release/coheryn sessions list
target/release/coheryn sessions show project-alpha
```

CLI/API histories have explicit revisions, idempotent turns, context accounting and cancellation-safe writer ownership. KV prefixes are reused within isolated bounded session caches. Histories default to memory-only; start `serve --session-persistence` and explicitly create with `--persist` or use `sessions save` to retain them across restart. `sessions export` and `sessions import` provide bounded portable JSON copies. See [session contracts and limits](docs/SESSIONS.md).

## Native CPU embeddings

The engine independently executes the pinned Nomic v1.5 F16/Q8_0 GGUF encoders through the Rust library, with ordered 768-dimensional vectors, bounded packed batches, explicit truncation, memory admission and cancellation. The multilingual corpus checks 50 language/Unicode cases against the independent tokenizer and encoder; language quality still depends on the selected model. See [the embedding guide](docs/EMBEDDINGS.md) for the model-specific normalization contract, Rust usage, limits and qualification commands. `coheryn embed` and `/v1/embeddings` are available; see [CLI/API and RAG guide](docs/EMBEDDING_API.md).

## Native function-call proposals

The S4 CPU preview supports a pinned Qwen tool model and companion template, complete/streamed function proposals, matching tool results and persistent pending calls. Use `chat MODEL --request request.json` or `sessions submit SESSION --request turn.json`; the calling application executes authorized tools. See [tool contracts, examples and limits](docs/TOOL_CALLING.md). Single-function strict tool arguments and structured responses are documented in [the constraint and sampling guide](docs/CONSTRAINTS.md). Broader model/backend/platform qualification remains open.

## Structured output and sampling

The native CPU decoder supports bounded JSON-object/schema masks, qualified single strict tool calls, repetition/frequency/presence penalties, token stops and per-request deadlines. CLI, API and managed sessions share these controls. See [CONSTRAINTS.md](docs/CONSTRAINTS.md) for the exact schema subset, limits, streaming/error contracts, reproducibility scope and qualification commands.

## Continuous CPU batching

The daemon shares matrix weight tiles across compatible live requests and schedules generation token units fairly with embedding microbatches. Each request retains its own KV, positions, RNG, grammar, output and cancellation. Default generation concurrency remains one; use `serve --max-concurrency 2` to permit two resident generation requests. See [BATCHING.md](docs/BATCHING.md) for eligibility, memory, metrics and qualification limits.

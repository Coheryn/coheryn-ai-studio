# Coheryn AI Studio

An independent Rust-native GGUF inference engine. This repository includes the **experimental G4 text CPU CLI/API service and E2 native CPU embedding CLI/API**, not the complete v1.0 product. The [v1.7 engineering plan](resources/Coheryn_AI_Studio_Final_Engineering_Plan_v1.7.md) preserves mandatory text, image understanding, independent embeddings, CLI/API, adaptive memory and CPU/GPU execution. It adds managed sessions, session-safe cache reuse, opt-in persistent-history capability, tool calling, constrained output, continuous batching and separate independence/naming audits. These additions are specification requirements awaiting implementation and qualification. Native image/VLM execution, Metal and final platform/release qualification remain open. Full RAG and vector storage remain optional. CPU profiles/admission and M0 provenance contracts are implemented; see [planner evidence and limits](docs/PLANNER.md).

Coheryn owns the Qwen2 generation graph, Nomic bidirectional embedding graph, packed quantized kernels and controlled CPU parallelism. Production code does not link llama.cpp, Ollama, Candle, PyTorch, or another inference engine. A separately built reference executable is used only for differential tests.

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

Download the pinned publisher model into the ignored cache, checking its SHA-256 before atomic installation:

```sh
python3 tools/fetch_model.py
target/release/coheryn models inspect .cache/models/qwen2.5-0.5b-instruct-fp16.gguf --json
target/release/coheryn models validate .cache/models/qwen2.5-0.5b-instruct-fp16.gguf --deep --json
target/release/coheryn run .cache/models/qwen2.5-0.5b-instruct-fp16.gguf \
  --prompt 'Hello' --max-tokens 16 --context 512 --standalone --stream
```

For chat, provide `--messages conversation.json` instead of `--prompt`. The file is an array such as `[{"role":"user","content":"Hi"}]`. The pinned GGUF chat template supplies its own default system prompt. Tools and multimodal messages are unsupported.

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

Quantized CPU qualification and embedding reference preparation are documented in [QUANTIZATION.md](docs/QUANTIZATION.md) and [the E0 fixture guide](fixtures/embedding/README.md). Native embedding CLI/API and external RAG interfaces are documented in [EMBEDDING_API.md](docs/EMBEDDING_API.md). Image inference remains unimplemented.

## Reproduce the independent oracle

These development dependencies are separate from Cargo and production execution:

```sh
python3 -m pip install --target .cache/devtools -r tools/requirements-oracle.txt
python3 tools/setup_reference.py
python3 tools/check_oracle.py --trace
```

The oracle uses the pinned llama.cpp revision, CPU only, one thread, F32 KV and disabled flash attention/native CPU tuning/llamafile. On macOS ARM, `armv8-a` and disabled NEON select scalar reference operations rather than half-precision accumulators and vector approximations. The initial CPU baseline uses F32 products/activations/output with a wide scalar reduction; optimized F32 accumulators remain subject to parity qualification. FP16 matrices round the activation operand to FP16. Python Jinja independently renders the pinned template. `--capture` intentionally replaces the committed small golden artifact after all comparisons pass. Full layer/logit arrays stay under `.cache/oracle`; the fixture manifest freezes numerical tolerances. Never update goldens or loosen thresholds solely to hide a regression.

```sh
python3 tools/benchmark.py -- --model .cache/models/qwen2.5-0.5b-instruct-fp16.gguf --context 512 --max-tokens 8
```

This Unix wrapper adds observed peak RSS, page faults and I/O block counters to portable benchmark output. It labels cache conditions uncontrolled and unavailable counters null. Model hashing warms the file cache; no result from this command is labeled a cold-cache measurement.

`mapped_bytes` in the benchmark refers to model weight mappings, which are zero for the buffered backend; it does not describe all process mappings. Run the opt-in pinned-model tokenizer regression with `cargo test --release -p coheryn-engine --test reference -- --ignored` after acquiring the model.

To reproduce the development F32 artifact and test its independently executed graph:

```sh
python3 tools/derive_f32.py
python3 tools/check_derived_f32.py
```

The conversion promotes existing binary16 values exactly and preserves F32 tensors. It is not original pretrained F32 weights. Its pinned output hash, source hash and conversion recipe are recorded; the additional 2.53 GB file stays ignored.

See [implementation plan](docs/IMPLEMENTATION_PLAN.md), [validation evidence](docs/VALIDATION.md), [threat model](docs/THREAT_MODEL.md), and [release checklist](docs/RELEASE_CHECKLIST.md). The consolidated v1.7 plan, source amendment and archived v1.4–v1.6 plans remain in `resources/`.

## Native CPU embeddings

The engine independently executes the pinned Nomic v1.5 F16/Q8_0 GGUF encoders through the Rust library, with ordered 768-dimensional vectors, bounded packed batches, explicit truncation, memory admission and cancellation. The multilingual corpus checks 50 language/Unicode cases against the independent tokenizer and encoder; language quality still depends on the selected model. See [the embedding guide](docs/EMBEDDINGS.md) for the model-specific normalization contract, Rust usage, limits and qualification commands. `coheryn embed` and `/v1/embeddings` are available; see [CLI/API and RAG guide](docs/EMBEDDING_API.md).

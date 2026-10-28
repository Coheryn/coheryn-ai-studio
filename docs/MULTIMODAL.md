# Offline native image-to-text

The development runner connects the selected Qwen2-VL F16 vision encoder/merger to its native Rust language decoder. It accepts local image files and ordered text/image chat, generates greedy text, reports physical and rotary context counts, and supports bounded execution with cooperative cancellation. This is an experimental macOS ARM64 CPU checkpoint. Product `coheryn chat --image` and image HTTP messages now use the same engine through [IMAGE_API.md](IMAGE_API.md). Managed multimodal sessions, LLaVA execution, Metal and other-platform qualification remain open.

The component pair is pinned in [the model manifest](../fixtures/vlm/models.json):

| Component | SHA-256 |
|---|---|
| `Qwen2-VL-2B-Instruct-f16.gguf` | `ec8b6dc5fd686f12b7d74850561f74b779db67424e477403fc043b98a0f40752` |
| `mmproj-Qwen2-VL-2B-Instruct-f16.gguf` | `ecb20cabcdd8dbc277de06bd6eb980aeb2adfaaba9f199a434e328d205675d03` |

Both files are fully hashed during loading. The adapter checks the graph, tensor shapes/encodings, tokenizer/template identity, image markers and merger/language widths. Shape compatibility for another file is separate from numerical or model-pair qualification; no universal VLM or GGUF support is claimed. Existing production text and embedding loaders retain their earlier supported architectures.

## Run a local image request

Acquire the pinned components with the existing [image provenance workflow](IMAGES.md), then build and run from the repository root:

```sh
CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=2 \
  ./tools/cargo build --release --locked -p coheryn-engine --example vlm_chat

target/release/examples/vlm_chat \
  .cache/vlm/full/qwen2-vl-2b/Qwen2-VL-2B-Instruct-f16.gguf \
  .cache/vlm/full/qwen2-vl-2b/mmproj-Qwen2-VL-2B-Instruct-f16.gguf \
  examples/vision-chat.json
```

The sample asks which color is on the left of the procedural fixture; the qualified response is `Red`. With `stream: true`, token events go to stderr and the final JSON goes to stdout. Omit `stream` for final JSON only. This example is a separate development executable, not the product CLI or a network service. On Windows, an executable suffix is required; model execution there remains unqualified.

Change the `images` array to local files and use `{"type":"image","index":0}` in the corresponding user-message content. Repeat indexed parts for ordered attachments. Each supplied attachment must appear exactly once; two entries may refer to the same file. Text-only user, assistant and initial system messages are accepted in the same conversation. The qualified template supplies its default system prompt when none is provided. Tools, video, remote URLs, Base64 ingress and literal reserved media/chat delimiters in text fail explicitly.

`context` is physical KV capacity, including expanded image features and the requested output allowance. The default is 256 and the development ceiling is 4,096. The requested prompt/output must fit; no automatic context reduction or resolution fallback occurs. `options.max_tokens` is 1–256, greedy only; `threads` and `vision.threads` are independently bounded to 1–16. The sample uses two workers and a 64-patch ceiling to keep the workload small. Larger photos may need a larger admitted context and patch ceiling; the processor's independent limits still apply. An engine ceiling is not a numerical qualification of every resolution, image count or thread setting.

`timeout_seconds` optionally supplies a cooperative deadline. `trace_directory` optionally writes borrowed image embeddings, final-prompt decoder layers and per-step logits as little-endian F32 files. Traces contain user/model data and remain a deliberate local development output. Acquiring files and serializing final JSON in the runner are separately bounded and are not fully instrumented by engine memory counters.

## Runtime and memory ownership

The unstable `multimodal::qwen2vl` interface exposes `Qwen2Vl`, `ChatMessage`, indexed `ChatPart`s, `Options`, generation/trace methods and a reservation-owning response. It uses the existing Rust decoder's checked buffered weights, F16 operands, F32 activations, wide reductions, biased GQA projections, causal attention, RMSNorm and SwiGLU. Language RMSNorm reads the actual GGUF epsilon. The output matrix follows the file's tied-embedding representation.

Text token IDs and projected image rows enter one physical KV sequence. Text advances all active rotary axes equally; image rows use temporal/vertical/horizontal positions. Decoder frequencies continue across sections, while encoder spatial frequencies restart per axis. The adapter caches decoder trigonometry once per input row and reuses it for Q/K, all heads and all layers. It skips full-vocabulary output projection on non-output prompt rows and skips the last block's attention/FFN when only its K/V is needed.

Before inference, header-only image plans and tokenization determine physical rows, rotary spans, context/output feasibility, full decoder reservation and the maximum sequential image-stage peak. One shared budget atomically admits that peak with decoder scratch/KV. Prepared pixels and projected embeddings transfer reservations out of this admission and recycle them after consumption; the next image reuses that capacity without releasing and reacquiring it. Images are processed one at a time. No complete weight/dequantization copy or quadratic attention matrix is retained. Request/output storage has a separate retained reservation that survives until the response drops.

Cancellation, deadlines, observer errors, delivery errors, unsupported inputs, file changes and allocation/admission failures unwind request state and owned workers. A response cannot be mistaken for a completed successful stream after those errors. Reported `vision_ms` covers encoder/merger computation. `prefill_ms` covers post-admission work excluding that vision time, including preprocessing and optional observers; `decode_ms` includes greedy decoding and optional observer/delivery callbacks. These are development observations, not isolated throughput benchmarks. Counters distinguish conservative reservations from tracked execution-vector capacities; metadata/tokenizer/allocator overhead, runner buffers, OS file cache, RSS and driver allocations are separate quantities. The product adapter now transfers an owned prepared request with held admission into the shared daemon scheduler; the development runner remains a separate entry point.

## Reference correction and evidence

The pinned assessment runtime omitted Qwen2-VL's RMSNorm epsilon during metadata loading: its original diagnostic runs report zero, while the GGUF and source configuration declare approximately `1e-6`. An isolated library adds only that metadata read; original source/builds, inputs and captures are preserved. [The correction record](../resources/tool-assessment/qwen-language-source-record.json) hashes the source change, reused objects, replacement object and resulting library. [Independent first-layer calculations](../fixtures/vlm/qwen2-language-independent-v2.json) reproduce both epsilon-zero and corrected checkpoints bit-for-bit using separate array arithmetic and identical teacher-forced inputs.

The [corrected image-conditioned reference](../fixtures/vlm/qwen2-image-conditioned-reference-v2.json) and [decoder-layer reference](../fixtures/vlm/qwen2-language-reference-v2.json) supersede the Qwen language numerics in the original diagnostic profile. Prompt bytes, token/image order, image embeddings, positions, weights, cache settings and frozen absolute/relative/L2 criteria remain unchanged. The native vision v2 profile additionally matches the assessment's FP16-input/output lookup-GELU contract without retaining a lookup table; earlier v1 component evidence is kept as historical evidence.

[Native qualification](../fixtures/qualification/m2-offline-vlm.json) compares actual PNG-to-text execution over six cases: red/blue image sensitivity, explicit system/EOS, Unicode multi-turn chat and two-image order reversal. It checks complete projected embeddings, all final-prompt decoder-layer arrays, per-step logits, exact greedy IDs/text, context counts and cleanup. Another run varies workers and physical context. All 231 complete arrays match bit-for-bit on the qualified host, alongside 192 ordinary workspace tests. Executables run with an empty executable search path. These small synthetic scenes establish executor correctness for the tested pair; they are not a broad visual reasoning or multilingual proficiency benchmark.

Reproduce with `python3 tools/check_native_vlm.py` once independent raw reference arrays are cached. Long runs can separate `--capture-only` and `--verify-captured`; the latter verifies executable, request, result and raw-trace SHA-256 receipts before comparing without rerunning inference. Capture alone is not acceptance. Independent research capture stays under the resource assessment boundary and outside native builds/tests/runtime. Ordinary CI checks frozen-reference integrity and small deterministic native graphs without downloading weights or executing an external runtime.

The subsequent [image CLI/API checkpoint](IMAGE_API.md) supplies secure attachments, managed pair residency, metadata-only estimates, queue-held complete admission, backpressure and disconnect cancellation. Persistent multimodal history/asset lifecycle and the other required model/backend/platform and release gates remain open.

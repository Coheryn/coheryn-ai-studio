# Qwen3.5 development CPU support

The native `qwen35-safetensors-cpu-v1` adapter executes the text decoder of the pinned official **Qwen/Qwen3.5-0.8B** checkpoint. This is an experimental CPU profile with standalone and managed CLI/API execution, owner-scoped text histories, bounded hybrid prefix reuse, continuous text batching and still-image input. Existing GGUF models remain regression inputs. The still-image component requires its validated processor asset and actual tensor inventory. Multi-token prediction weights remain inactive.

The source inventory and complete per-file hashes are in `fixtures/qwen35/model.json`. The loader accepts an explicit single Safetensors file, the canonical index, or a directory with an unambiguous canonical entrypoint. Whole-tensor shards retain separate checked file handles; weights are read in bounded tiles without a permanent dequantized/full-weight copy. F32/F16/BF16 container reading is implemented; this architecture's initial executable weight profile permits F32/BF16 only. Packed source quantization schemes are rejected.

## Local use

Build the current release binary and acquire the immutable checkpoint, including its license:

```sh
./tools/cargo build --release --workspace --locked
python3 tools/fetch_checkpoint.py
./target/release/coheryn run .cache/qwen3.5-0.8b-source \
  --standalone --messages fixtures/qwen35/smoke-messages.json \
  --context 256 --max-tokens 8 --threads 2 --max-memory 805306368 --stream
```

`fetch_checkpoint.py --verify-only` checks already cached bytes offline. New acquisitions verify every pinned length/hash in private staging before renaming the complete directory. No generation request downloads assets. Use `chat CHECKPOINT --standalone` for interactive text. For daemon import/load/run and its stateless API subset, see [MANAGED_CHECKPOINTS.md](MANAGED_CHECKPOINTS.md). There is no silent fallback between daemon and standalone execution.

The supported development subset is greedy text, at most 256 output tokens, an explicitly reserved context of 1–8192 tokens, bounded user/assistant/system messages and the exact qualified tokenizer/template with thinking disabled. Source dimensions, layer types, partial rotary settings, tied output representation and tensor shapes are validated rather than taken from the filename. Unsupported sampling, tools, schema constraints, custom stops and control delimiters fail explicitly. Unicode strings use the source tokenizer and incremental UTF-8 output; language quality remains dependent on the model.

The graph implements zero-centered normalization, causal depthwise convolution, gated delta-rule recurrent state, gated full GQA attention, partial rotary positions, gated FFNs and tied output projection. CPU workers join before their reservation is released. Standalone profile admission now shares the metadata estimator and atomically reserves session/request capacity plus profile headroom before tokenization. Session KV/recurrent/convolution/scratch/output buffers are reserved before construction, reused within the request, observed by capacity accounting and reclaimed on errors/cancellation. The fixed 256 MiB resident metadata/tokenizer reservation is a conservative estimate, not a measurement of library allocations. Physical RAM, OS file cache and allocator overhead are separate from `--max-memory` admission.

## Evidence and remaining work

`fixtures/qwen35/reference-v1.json` freezes independent array-equation outputs for English and mixed Chinese/Bengali/emoji input, prompt bytes/IDs, 24 final-prompt layer checkpoints and teacher-forced logits/greedy continuations. It declares BF16 source weights promoted to F32 activations, F32 products and wide reductions. This is independent numerical evidence, not parity with the publisher's complete framework or lower-precision activation profile. Frozen thresholds must not be relaxed to accept an implementation.

The separate assessment recipe is maintained under the resource research boundary. Its full arrays stay in ignored `.cache/q35-reference/`. Native verification reads those inert arrays and never invokes an external inference engine:

```sh
./tools/cargo build --release -p coheryn-engine --example qwen35_probe --locked
python3 tools/check_qwen35.py --report fixtures/qualification/NEW-native-text.json
```

The official source index contains one physical weight shard; the single/indexed model checks select that same file through different entrypoints. Tiny parser fixtures cover multiple files, while real multi-file model qualification remains open.

The native probe checks prompt/token parity, full arrays, exact greedy outputs, stream reconstruction, worker counts, single/indexed entrypoints and failure/cancellation cleanup. `fixtures/qualification/q35-native-text.json` records accepted runs and the exact binary/reference/model identities. Ordinary CI runs bounded parser, registry and recurrent-operation tests without downloading this checkpoint.

Local metadata inspection/validation and standalone CAEP estimates are now available through the [shared descriptor](MODEL_METADATA.md), without constructing a generation session.

Full publisher-framework comparison, broader prompts/EOS/long-context campaigns, checkpoint URL acquisition, broader multi-file qualification, broader image semantics, quantized source weights, GPU/Metal and other-platform qualification remain open. This checkpoint does not complete A0–A5, F0–F5 or the production release matrix. See [adapter boundaries](ADAPTERS.md), [format gates](SAFETENSORS.md) and the [release checklist](RELEASE_CHECKLIST.md).

Managed source sets now bind every file hash and native profile, use the shared daemon budget, and reach stateless text/chat/SSE through cooperative CPU scheduling. Managed histories use optional exact-prefix hybrid snapshots through [QWEN35_SESSIONS.md](QWEN35_SESSIONS.md). Continuous text batches share decoded weight tiles. Additional sampler features remain unsupported. See [MANAGED_CHECKPOINTS.md](MANAGED_CHECKPOINTS.md) for the upload and lifecycle contract.

## Still-image input and scheduling

The pinned source's `preprocessor_config.json`, vision tensor inventory and interleaved rotary sections enable `qwen35-still-cpu-v1`. The native path decodes PNG/JPEG, applies the model's resize/normalization and 16-pixel temporal patches, executes the vision encoder and merger, inserts expanded features, and preserves physical KV positions separately from three-axis rotary coordinates. Up to four ordered images are accepted within configured byte/pixel/feature/context limits. Quality or context is never silently reduced. This is a validated profile, not universal Qwen-VL compatibility.

```sh
./target/release/coheryn chat .cache/qwen3.5-0.8b-source --standalone \
  --image fixtures/qwen35/vision/red.png --prompt 'What color is the picture?' \
  --context 256 --max-tokens 16 --threads 2 --profile balanced --stream
```

No separate vision-model argument is needed for this complete source package. Managed import includes the inert processor file in its full source-set identity. Earlier managed packages lacking that asset remain text-only; explicitly import a new complete package. Stateless `/v1/chat/completions` and `/api/v1/generate` accept the same bounded text/image parts described in [IMAGE_API.md](IMAGE_API.md). URL policy remains disabled by default. Managed image sessions are now supported with explicit `task: image-to-text`; text remains the default for existing Qwen3.5 sessions. See [IMAGE_SESSIONS.md](IMAGE_SESSIONS.md).

Admission reserves the language session, encoded images, processor stages, encoder scratch/workers and retained features before queueing. Images encode sequentially. Encoder tiles yield the shared CPU lease; decoding uses the same bounded coordinator as text. Response delivery owns no compute permit. Image-conditioned snapshots are never reused. Feature/output reservations retire on success, cancellation and observer/consumer failure. For image requests, `vision_ms` measures encoding separately; `first_token_ms` starts with decoder generation and is not end-to-end request latency. Vision metadata uses empty component hash strings when the engine has no verified full-payload identity; managed model source-set identity remains separately bound by the registry.

`fixtures/qwen35/vision/contract-v1.json` freezes authored square/portrait constant-color PNGs, ordered two-image integration and numerical tolerances. `reference-v1.json` records the independent F32/wide-reduction equation oracle; full arrays remain in ignored cache. `b3-native-images.json` passes 140 full encoder/merger/decoder/logit comparisons and exact four-token continuations. Those short continuations establish executor parity, not broad image understanding or language proficiency. GPU, publisher-framework precision parity, textured-image campaigns and other platforms remain open.

```sh
./tools/cargo build --release -p coheryn-engine --example qwen35_image_probe --locked
python3 tools/check_qwen35_images.py --report fixtures/qualification/NEW-native-images.json
```

The optional real-checkpoint tests cover hybrid cache equality against fresh inference, independent batch cancellation, image admission and encoder failure cleanup. The in-process HTTP qualification target exercises real handlers and scheduling without sockets. Reports refuse overwrite; use a new absolute output path:

```sh
COHERYN_Q35_REPORT="$PWD/fixtures/qualification/NEW-http-runtime.json" \
  ./tools/cargo test --release -p coheryn-api --test qwen35_runtime -- --ignored
```

## Image plans and persistent conversations

Image planning now reads validated headers, tensor inventory and processor metadata without constructing a model or numeric session. Standalone and managed estimates share the runtime's stage arithmetic. Results distinguish image preprocessing, encoder scratch, retained features, decoder/session capacity, weights, ingress/delivery and profile headroom. Exact text/output/context and atomic reservations are still checked at submission. Dimension-only PNG/RGBA and encoded-byte ceilings are conservative upper bounds.

```sh
coheryn plan .cache/qwen3.5-0.8b-source --standalone --task image-to-text \
  --image fixtures/qwen35/vision/red.png --context 256 --profile reference --threads 1
coheryn plan q35 --task image-to-text --image photo.png --context 256
coheryn chat q35 --session photos --persist --image photo.png \
  --prompt 'Describe this image.' --context 256 --max-tokens 32
coheryn chat q35 --session photos --resume
```

The first image-chat command creates an image session; a later resume reconstructs the same original attachments without resending them. Explicit creation also supports `sessions create q35 --task image-to-text --name photos --context 256 --persist`. Managed image contexts currently support at most 4096 tokens and must fit the loaded/server limits. Selected history can contain at most four images under the configured finite limits. Original assets, history ownership, revisions, idempotent receipts, persistence, portable format 3, whole-turn overflow policy, cancellation and cleanup share the existing image-session implementation. KV/recurrent state and projected image features remain request-local for image sessions.

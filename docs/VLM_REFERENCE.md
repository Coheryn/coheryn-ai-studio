# Image-conditioned reference qualification

This checkpoint establishes exact-file inputs for native vision development. It does not enable image-to-text in the product. Reference execution is isolated under the resource assessment boundary; native build, test and runtime paths consume only frozen data.

## Accepted reference policies

The selected legacy LLaVA vision component contains blocks 0–22, already excluding the final block of its 24-block source encoder. The original pinned reference subtracted another block and appended CLS where the position/patch inputs expected CLS first. A separate corrected source copy executes all 23 stored blocks for this MLP profile and places CLS first. It excludes CLS before projection. Original source files, weights, processor goldens and the earlier diagnostic vision capture remain unchanged.

The independently written NumPy calculation reads the identical pinned GGUF weights and compares the complete pre-layernorm result, all 23 vision-layer checkpoints and projected image embeddings against the corrected reference. F16 matrix operands, F32 elementary-operation order and wider reductions are explicit; its standard-libm bridge contains no graph or external inference code. The independent forward pass never uses reference activations as inputs. The original absolute/relative/L2 thresholds remain unchanged. These comparisons qualify the specified processor/weights/math policy, not general model-family behavior.

The selected processor remains `llava15-pad336-v1`. The original publisher configuration records an aspect-padding policy; the separately preserved HF processor conversion records center cropping. The exact GGUF profile uses the frozen pad336 pixels. No equivalence between arbitrary pad/crop pipelines or universal HF processor parity is claimed.

Qwen2-VL retains its unchanged reference graph with explicit source pixel limits. Actual vision FFN tensors use width 5,120; the misleading 1,536 metadata field is not an executable FFN shape. Both profiles derive projector output sizes from actual tensor inventories. [Exact-file graph contracts](../fixtures/vlm/execution-profiles-v1.json) record these choices while keeping native inference support false.

## Captured inputs and outputs

[The request corpus](../fixtures/vlm/requests-v1.json) contains two LLaVA single-image cases and six Qwen2-VL cases: image-sensitive pairs, ordered/reversed two-image requests, Unicode multi-turn content, and an explicit-system EOS case. [The image-conditioned reference](../fixtures/vlm/image-conditioned-reference-v1.json) records source-template prompt bytes, marker-expanded token chunks, ordered image embeddings/positions, complete-vector hashes/summaries, teacher-forced logits along greedy prefixes and at most eight generated token IDs. Large raw F32 tensors remain ignored; the isolated capture can reproduce them.

The same image must produce the same vision features independently of prompt text. Reordering images must change the conditioned language logits. The paired single-image cases produce Red/Blue and EOS. Both two-image continuations share the same generic eight-token prefix; reversing the images changes the underlying logits (maximum absolute change about 0.535). This is conditioning evidence rather than a complete semantic-answer benchmark. Greedy outputs and EOS flags describe the pinned reference; they are acceptance inputs for the future native executor, not broad language or visual-quality benchmarks.

[Independent position spans](../fixtures/vlm/positions-v1.json) feed ordinary native Rust tests. `multimodal::positions::ImagePositions` exposes checked positions without allocating a coordinate matrix. A 112×56 Qwen image contributes eight physical feature rows but a four-position rotary span. Its decoder axes are temporal, vertical, horizontal and reserved. LLaVA contributes 576 sequential feature rows/positions. KV/admission must count physical entries independently of rotary coordinates; this helper does not admit a full vision request.

## Reproduction and provenance

Only the independent resource assessment guide contains external build/capture commands. Native offline checks are:

```sh
python3 tools/check_image_fixtures.py
python3 tools/check_vlm_references.py
CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_INCREMENTAL=0 \
  ./tools/cargo test -p coheryn-engine --test image_positions --locked
```

The capture first loads language metadata with tensor allocation disabled, runs vision and saves bounded features/chunks, then exits that process. A separate process maps language weights and evaluates staged inputs with F32 KV. This avoids simultaneous vision/language weight residency. Per-stage peak RSS is an OS observation; allocator overhead, compression, file cache and uncontrolled host/cache conditions remain separate from admission estimates. No throughput advantage or cold-cache claim is made.

Capture receipts pin input files, component hashes, processor/template bytes, binary/source identity and outputs before resume. The pipeline verifies full weights before execution. It runs components/cases sequentially, bounds image/chunk/context/output counts and keeps large data ignored. Isolated NumPy and helper tools are pinned to current verified releases and do not enter either Cargo graph.

The copied original publisher configuration is mapped to its source revision/hash and retained custom terms in the legal inventory. No model weights are included in the repository or distribution notices. Full model-chain and platform/package review remain release work.

## Remaining native work

The native vision encoders, projectors, language adapters, image-expanded KV/scheduling and secure CLI/API ingress remain pending. Native feature/logit/greedy parity, memory/cancellation qualification and CPU/Metal/platform evidence are still required. LLaVA reference coverage is one image per request; Qwen reference coverage is at most two. These limits do not qualify native requests or arbitrary resolutions/models/backends.

The subsequent [native Qwen vision checkpoint](VISION.md) compares complete encoder/merger arrays against these frozen references and three additional independently captured geometries. It preserves this reference-only record and its scope. Native image-conditioned language execution remains the next integration step.

The subsequent [offline language checkpoint](MULTIMODAL.md) identifies an omitted Qwen RMSNorm epsilon read in the pinned assessment loader. Original Qwen language outputs remain epsilon-zero diagnostics; versioned v2 references read the declared GGUF epsilon. Independent first-layer calculations reproduce both profiles exactly. This correction is isolated from original source/builds and leaves the existing vision inputs, positions, component bytes and numerical thresholds unchanged.

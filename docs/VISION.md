# Native Qwen2-VL vision encoder

The native Rust library executes the selected Qwen2-VL F16 vision encoder and spatial merger on CPU. This component produces projected image embeddings. The [offline Qwen language adapter](MULTIMODAL.md) now consumes them to generate text with complete local request admission. Product image CLI/API attachments, managed sessions, LLaVA execution, Metal and other-platform qualification remain open. This is a partial M2 implementation; image understanding remains a mandatory release gate.

The exact component is `mmproj-Qwen2-VL-2B-Instruct-f16.gguf`, SHA-256 `ecb20cabcdd8dbc277de06bd6eb980aeb2adfaaba9f199a434e328d205675d03`. [The component manifest](../fixtures/vlm/models.json) pins its source and revision; [full-byte verification](../fixtures/qualification/m0-components.json) supersedes the historical prefix-only selection status. Compatible-looking names and metadata are insufficient for qualification.

## Graph and interfaces

`multimodal::vision` is an unstable workspace interface. `VisionConfig::from_descriptor` separates shape compatibility from exact-file qualification. `VisionModel::load` parses with finite metadata limits, validates all tensors, opens checked buffered storage, hashes the complete file and retains a metadata reservation. The image must be prepared with `qwen2vl-still-v1` under the same `Budget` as the model.

`plan` computes checked workspace/output reservations without executing weights. `encode` returns token-major F32 embeddings with their own retained reservation; output can outlive the model. `trace` borrows patch, per-layer, post-normalization and projected checkpoints through a synchronous fallible observer. Observers own any copies or file output they create. These interfaces do not admit a complete image-to-text workload or integrate with the daemon scheduler.

The selected file has 32 blocks, hidden width 1,280, 16 attention heads, patch size 14, two temporal patch kernels and a spatial merge of two on each axis. The actual expansion tensors have width 5,120; the declared FFN metadata value of 1,536 is retained in diagnostics and never used to size those matrices. Legacy expansion/contraction names are validated against their actual shapes. Each block executes affine LayerNorm, biased Q/K/V projections, spatial rotary positions, noncausal attention, output/residual and QuickGELU FFN/residual. The merger applies post-LayerNorm, groups four already-adjacent patch rows, then executes two projections with the versioned tanh-GELU precision contract between them. Vision profile `coheryn-qwen2vl-vision-v2` matches the frozen scalar FP16 lookup behavior by rounding activation input/output to binary16 without retaining a lookup table; values outside ±10 follow the reference saturation rule. Earlier v1 full-F32 GELU component evidence is historical and unchanged. The output width is 1,536 for this file.

Only this graph's F16 matrix and F32 bias/norm representations are implemented. Unknown vision features, different activation declarations, shards, extra/missing/duplicate tensors, unsupported encodings and inconsistent shapes fail explicitly. Shape acceptance for a development fixture does not qualify another publisher model.

## Bounded execution

Weights are read in bounded tiles without mapping or retaining a complete dequantized copy. Matrix input tiles reuse weight reads and existing owned CPU workers; F16 operands use F32 products and wide scalar reductions. The spatial trigonometric cache is computed once per image and reused across blocks and heads. Paired scalar trigonometry is kept across optimization boundaries because a few F32 rounding differences otherwise propagate through later binary16 operands.

Attention uses one score row and one head accumulator per worker rather than a quadratic attention matrix. States, projections and FFN scratch are reused across all blocks. Still frames are read once from the processor's virtual temporal layout; grouping for the merger needs no extra shuffle buffer. Model, image, scratch, worker-stack allowances and retained output reservations share one budget. Capacity counters measure owned vectors separately from conservative reservations; neither is a hard limit on allocator overhead, process RSS or OS file cache.

Options bound threads to 1–16, matrix tiles to 1–32 token rows and patch rows to 4–4,096 in groups of four. Processor limits still apply. Defaults are one worker and 32-row tiles; choose a smaller patch ceiling for local development. Cancellation is checked during hashing, patch gathering, matrix tiles, normalization, spatial rotation, attention rows/key groups, activation chunks and block boundaries. On error, cancellation or observer failure, worker pools join and temporary reservations/buffers drop. Buffered storage checks file identity/size/modification state before and throughout use. Callers must not modify loaded files; these checks are not a filesystem snapshot against hostile concurrent writers.

## Reproduce the component check

After acquiring the pinned component using the existing [image provenance guide](IMAGES.md), build the diagnostic example:

```sh
CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=2 \
  ./tools/cargo build --release --locked -p coheryn-engine --example vision_probe
```

Create a local request file, for example `.cache/vision-request.json`:

```json
{
  "image": "fixtures/vlm/images/landscape.png",
  "options": {"threads": 1, "tile_tokens": 32, "max_patches": 64},
  "max_memory": 134217728,
  "trace_directory": ".cache/vision-trace"
}
```

Run the native component:

```sh
target/release/examples/vision_probe \
  .cache/vlm/full/qwen2-vl-2b/mmproj-Qwen2-VL-2B-Instruct-f16.gguf \
  .cache/vision-request.json
```

Omit `trace_directory` to avoid writing raw checkpoints. The example reports component identity, dimensions, output fingerprint, elapsed encoding time, reservations/capacities and post-drop cleanup. File acquisition in this development helper is bounded separately from engine accounting.

[Native qualification](../fixtures/qualification/m2-vision-v2.json) compares complete arrays for all 32 blocks and the projected output, under unchanged absolute/relative/L2 criteria. Five images cover color order, portrait, minimum resolution and odd source dimensions; an additional run checks two workers and a smaller tile. `python3 tools/check_native_vision.py` reproduces those comparisons when the pinned component and independently captured raw tensors are cached. It executes only the native example and verifies frozen raw hashes before comparison. [The reference guide](VLM_REFERENCE.md) documents separate reference capture. Ordinary CI uses small nonzero synthetic graphs, bounded/error/lifecycle tests and frozen-reference integrity checks without external execution or model downloads.

The [offline language integration](MULTIMODAL.md) now covers projected input embeddings, multidimensional rotary positions, physical/rotary context accounting and local request admission. Product CLI/API acquisition and daemon planner/scheduler integration are now documented in [IMAGE_API.md](IMAGE_API.md). Persistent image histories, finer scheduling and the remaining backend/platform qualifications stay open.

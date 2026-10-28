# Native image preprocessing

The engine has a bounded PNG/JPEG preprocessing foundation for the selected LLaVA and Qwen-VL model sets. **Offline image-to-text is now available for the selected Qwen2-VL CPU pair through the [development runner](MULTIMODAL.md). Product image CLI/API attachments are documented in [IMAGE_API.md](IMAGE_API.md).** The selected Qwen2-VL CPU vision encoder and merger now have a [native component checkpoint](VISION.md). Native language integration and image-expanded positions/KV now have an offline checkpoint. Managed model pairs now connect the qualified graph to bounded CLI/API acquisition and shared scheduling/admission. Managed image histories and broader release qualification remain open. Existing text and embedding commands keep their current behavior.

[Processor profiles](../fixtures/vlm/preprocessing/profiles.json) define the exact file-specific rules. [Independent processor cases](../fixtures/vlm/preprocessing/reference.json) freeze 20 cases: ten images through two processors, including odd dimensions, downscaling, grayscale, alpha, baseline JPEG and EXIF orientation. Full resized RGB, normalized F32 and streamed patch fingerprints are checked. JPEG reconstruction is compared separately with a maximum difference of two channel values; processor comparisons use identical canonical decoded pixels. These are processor checks, not model understanding or universal format support.

## Use the image stage locally

This development example inspects a local file, prepares pixels and prints JSON with the plan, output fingerprint, elapsed time and memory counters. It does not load model weights or produce a model response.

```sh
CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 ./tools/cargo run --locked \
  -p coheryn-engine --example prepare_image -- \
  llava15-pad336-v1 fixtures/vlm/images/landscape.png
CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 ./tools/cargo run --locked \
  -p coheryn-engine --example prepare_image -- \
  qwen2vl-still-v1 fixtures/vlm/images/landscape.png
python3 tools/check_image_fixtures.py
```

The unstable Rust module `multimodal::image` exposes `inspect`, `plan`, `decode`, `prepare_decoded`, `prepare` and ordered `prepare_many`. Decoded pixels and prepared outputs retain their reservations until dropped. The separate decode/prepare interfaces allow pixel reuse without a second codec pass. `patch_values` yields a logical patch row without allocating a patch matrix; Qwen still frames are repeated virtually.

## Input and processor rules

- PNG: 8-bit grayscale, grayscale-alpha, RGB or RGBA. All chunks have checked ranges and CRCs. Palette/color-key transparency, animated PNG, ICC/cICP and unknown critical chunks fail explicitly.
- JPEG: baseline 8-bit grayscale/RGB with one interleaved scan. Progressive, CMYK/YCCK, ICC and multi-picture containers fail explicitly. Trailing bytes and incomplete containers are rejected.
- Apply primary EXIF orientation 1–8 before resizing. Reject duplicate or invalid orientation metadata. Grayscale replicates into RGB; alpha is discarded while retaining RGB samples. Channel samples are used without gamma/chromaticity conversion; no general color management is claimed.
- `llava15-pad336-v1`: preserve aspect ratio, resize with F32 scale/ceil and pad a centered 336×336 canvas with `[122,116,104]`. This follows the pinned GGUF assessment processor. The preserved HF processor uses center cropping; the selected-GGUF profile remains explicit. The corrected graph policy and independent comparisons are documented in [VLM_REFERENCE.md](VLM_REFERENCE.md).
- `qwen2vl-still-v1`: dimensions are multiples of 28, source pixel limits are 3,136–12,845,056 and aspect ratio is at most 200. Explicit source limits override the assessment runtime's different defaults. Operator limits can reject this canonical output; they never silently lower its resolution.
- Both profiles use horizontal-then-vertical Keys bicubic filtering with 22-bit coefficients and U8 rounding/clamping after each pass, then CLIP RGB normalization. Two explicit Pillow passes reproduce this order; a convenience resize call can choose another order and differ.

## Memory and cancellation

Defaults are 16 MiB encoded per image, 32 MiB per request, 8,192 source edge, 16 Mi source pixels, 1 Mi processed pixels, 1,024 feature tokens per image, 4,096 total feature tokens and four ordered images. Finite hard ceilings prevent unbounded configuration. These are frontend limits; they do not establish a model's permitted image count or context capacity.

Batch preparation checks aggregate bytes/features and atomically reserves the complete processor peak before pixel decoding. It processes one image at a time, retains only completed normalized outputs, and transfers their reservations without releasing/reacquiring capacity. Resizing uses coefficient tables and a vertical-support row ring rather than a full intermediate horizontal image. EXIF coordinate mapping avoids a second oriented source image. Qwen's duplicated temporal frame and grouped patch layout are views over the normalized pixels.

Plans distinguish decoder, resampling, normalization and retained-output allowances. Engine-owned buffer capacities are observed separately from reservations. Codec allocations use conservative bounded-input allowances and best-effort codec allocation limits; library/allocator overhead and OS residency are not fully instrumented or hard capped. Borrowed encoded bytes remain caller-owned; their configured byte count is included in request-stage estimates. These counters do not include vision weights/workspace, projected embeddings, language KV or GPU memory and cannot admit a complete image-conditioned generation request.

Cancellation is checked during container scanning/CRC, coefficient construction, row filtering, hashing and normalization. One bounded baseline-image codec decode is a cancellation unit: checks occur before/after it, and admission is retained until the codec returns. Failure or cancellation drops buffers and reservations; a failed later image also releases earlier outputs. The standalone example's file acquisition is a bounded development helper, separate from the bounded product ingress documented in [IMAGE_API.md](IMAGE_API.md).

## Model provenance and remaining gates

[Component verification](../fixtures/qualification/m0-components.json) records complete byte counts and SHA-256 verification of all four selected language/vision GGUF files. Large files remain in the ignored `.cache/vlm/full/` cache. Reproduce acquisition with `python3 tools/fetch_vlm.py`; it downloads sequentially with bounded buffers, exact identities, atomic writes and a default 4 GiB disk floor. It executes no model. The original selection manifest retains its historical prefix-only status; this newer record supplies full-byte evidence.

[Independent vision/projector captures](../fixtures/vlm/vision-reference.json) cover two color-reordered inputs for both selected components: layer checkpoints, projected features, full-vector summaries and raw-tensor hashes. [Alignment findings](../fixtures/vlm/alignment.json) record the observed LLaVA layer mismatch and tensor/metadata dimension differences. The captures follow the pinned reference behavior and do not resolve those discrepancies.

Independent processor and vision research executes separately under the resource assessment boundary. Native CI reads frozen data and never builds, links or invokes an external executor. Independent vision/projector captures are preparatory evidence; Native image-conditioned logit comparisons are still required; the Qwen vision component has separate native feature comparisons. The new [reference qualification checkpoint](VLM_REFERENCE.md) resolves the selected legacy LLaVA layer/CLS policy and captures image-conditioned logits, greedy/EOS outputs and Qwen two-image order cases. Original diagnostic records remain unchanged. Native language integration, model-chain/release review and complete M2 qualification remain open. Secure Base64/URL ingress, scheduling integration and actual product attachments remain M1 work. CPU/Metal model qualification and other-platform execution remain open. Full v1.0 still requires image understanding.

The subsequent [offline Qwen image-to-text checkpoint](MULTIMODAL.md) adds the language decoder and atomic local request admission. Its metadata-correct reference profile preserves the earlier Qwen epsilon-zero language captures as diagnostics. The image stages and original input/reference bytes remain unchanged; full product/Metal/platform/release acceptance is still open.

The subsequent [product image ingress checkpoint](IMAGE_API.md) adds managed pairs, metadata-only vision estimates, strict Base64/opt-in public HTTPS acquisition and request-wide queue/response lifetimes. Earlier component-only limitations above describe their historical checkpoints; the compatibility matrix records the current end-to-end scope.

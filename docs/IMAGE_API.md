# Native image CLI and API

The product accepts image-to-text requests through `coheryn chat`, `/v1/chat/completions` and the native `/api/v1/generate` transport. The established GGUF execution profile is `qwen2vl-still-f16-cpu-greedy`: the exact Qwen2-VL 2B language/vision pair in [MULTIMODAL.md](MULTIMODAL.md), F32 KV, native CPU, greedy text output. PNG/JPEG decoding follows the bounded subset in [IMAGES.md](IMAGES.md). Numerical qualification covers the canonical PNG cases on macOS ARM64; decoder support alone does not qualify another model, format, resolution or platform.

## Register and run a pair

Build the CLI with `cargo build --release --workspace --locked`. Start the daemon with a budget appropriate to the selected context and image workload; the following uses a 768 MiB engine reservation limit for the small qualification images:

```sh
coheryn serve --max-memory 805306368
```

In another terminal, import the two already acquired, hash-verified components and register their roles:

```sh
coheryn models import Qwen2-VL-2B-Instruct-f16.gguf --name qwen-vl-language
coheryn models import mmproj-Qwen2-VL-2B-Instruct-f16.gguf --name qwen-vl-vision
coheryn models register-pair --name qwen-vl \
  --language qwen-vl-language --vision qwen-vl-vision
coheryn plan qwen-vl --task image-to-text --context 256 --image scene.png
coheryn models load qwen-vl --context 256
coheryn chat qwen-vl --image scene.png --prompt 'Describe this image.' \
  --context 256 --max-tokens 8 --stream
```

Repeat `--image` to attach ordered images. Omit `--prompt` to read one bounded UTF-8 prompt from stdin. `--json` returns the native result; combine it with `--stream` for JSON Lines token events and a final result. The daemon is the default; missing models or daemon failures never cause standalone fallback.

For an explicit process-local instance:

```sh
coheryn chat Qwen2-VL-2B-Instruct-f16.gguf --standalone \
  --vision-model mmproj-Qwen2-VL-2B-Instruct-f16.gguf \
  --image scene.png --prompt 'Describe this image.' \
  --context 256 --max-tokens 8 --json
```

Only the local CLI reads local attachment paths. Remote calls carry image bytes or validated HTTPS URLs. A model pair is one registry/resident entry referring to two immutable component identities; registration validates graph roles, widths and template metadata. Loading verifies both complete SHA-256 hashes, builds the tokenizer and validates the executable graph. Pair aliases copy no weights, add no duplicate disk-quota bytes and cannot nest. Referenced components cannot be removed before their pairs. An active, queued or undrained response pins the loaded pair against unloading. Ordinary imports remain bounded atomic uploads, with a per-component file ceiling.

## OpenAI-shaped messages

Send an inference bearer credential and `Content-Type: application/json`:

```json
{
  "model": "qwen-vl",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64_BYTES>", "detail": "auto"}}
      ]
    }
  ],
  "max_tokens": 8,
  "context": 256,
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

`messages` accepts text strings and ordered `text`/`image_url` parts. Image data URLs must use canonical Base64 with `image/png` or `image/jpeg`, and the bytes must match that media type. Images belong only to user messages. The initial system prompt is optional; the qualified template supplies its default. Prior text user/assistant turns are accepted, and the last turn must be a user turn. Limits are 64 messages, 128 parts per message and 64 KiB aggregate text, plus the configured image limits. Literal reserved chat/media delimiters fail explicitly. `detail` accepts `auto` or omission; other detail modes are unsupported.

The supported generation fields are `model`, `messages`, `max_tokens`/`max_completion_tokens` (1–256), `stream`, `stream_options.include_usage`, `n=1`, `context`, `max_memory`, `timeout_seconds`, `profile`, `threads`, and default greedy control values (`temperature=0`, `top_p=1`, `top_k=0`, unmodified penalties). A seed has no effect on greedy decoding. Tools, constrained output, custom stops and nondefault sampling fail explicitly. The native endpoint accepts the same image request shape and returns token IDs and engine diagnostics. `coheryn.vision` on OpenAI-shaped responses records the component identities, image count, effective positions and vision timing. Usage counts physical prompt rows, including expanded image features; completion usage includes the model's terminal EOS token when generated.

Streaming uses assistant-role/content deltas, one finish chunk, optional separate usage and `[DONE]`. Failure produces an error and `[DONE]` without a successful finish. Native streaming retains its `token`/`result` events. An undrained successful or failed stream retains bounded output capacity and model ownership until its body drops.

## Acquisition and execution limits

Remote image URLs default to disabled. `serve --allow-image-urls` permits public HTTPS on port 443, without URL credentials or fragments. Every redirect repeats URL validation and bounded DNS resolution; all returned addresses must be public, and the connection pins those addresses. Environment proxies and automatic redirects are disabled. Media types, declared lengths, streamed byte totals and the total acquisition deadline are checked. Request logs and errors omit image bodies and URL query strings.

Operator flags include `--image-max-body-bytes` (default 48 MiB), `--image-max-bytes` (16 MiB), `--image-max-total-bytes` (32 MiB), `--image-max-count` (4), `--image-acquisition-timeout` (30 seconds), and `--image-redirects` (3). Configuration can tighten the finite processor dimensions/pixels/features as well. These are resource and request limits, not development deadlines. No local path, arbitrary HTTP scheme, private address, proxy redirect or implicit format conversion is accepted.

Ingress grows an atomic reservation as bytes arrive. JSON structure, body bytes and acquisition time are bounded independently. Unsupported models and controls fail before image fetching. Header inspection establishes exact geometry before decoding. The queued owned job holds full decoder capacity plus the maximum sequential image-stage peak; retired JSON/Base64 ingress is released before queueing. Execution transfers existing guards rather than releasing/reacquiring admission. Requested context includes text, expanded image rows and output; insufficient physical or rotary capacity produces an error without silently reducing context or image quality.

`plan --task image-to-text --image FILE` sends oriented dimension descriptors, without uploading images or loading weights. `/api/v1/plans/estimate` accepts `images: {"processor_profile":"qwen2vl-still-v1","images":[{"width":112,"height":56}]}` with the registered pair, context, profile and concurrency. It reports processor/encoder/decoder costs, metadata, shared file storage and explicit rejection reasons. Dimension-only costs conservatively use image byte ceilings and four channels; exact text, attachments and atomic admission are checked on submission. File bytes, engine reservations, observed vector capacity and OS residency are separate quantities.

Images share the CPU FIFO gate with text token batches and embedding microbatches. The cooperative adapter now yields at completed vision tiles/query groups and decoder checkpoints, and releases compute during delivery. Image sessions do not participate in a shared batched forward pass; see [IMAGE_SESSIONS.md](IMAGE_SESSIONS.md). Consumer backpressure and request deadlines cancel cooperatively, and the supervisor awaits blocking-worker acknowledgement before releasing execution resources. One event-channel slot is reserved for the terminal error/result. Scheduler metrics expose `vision_backpressure_waits` without recording user contents.

## Evidence and remaining scope

Workspace tests cover small independent model pairs, header-only estimates, component mutation/deletion, native/OpenAI/CLI/standalone parity, multiple images, Unicode, malformed and oversized inputs, authentication, URL restrictions, memory admission, queue saturation, disconnection and slow-consumer cleanup. `python3 tools/check_image_service.py` reproduces the pinned-pair product check from cached components and the release CLI, runs native execution with an empty executable search path, and writes `fixtures/qualification/m3-image-service.json`. It reads frozen independent outputs and never runs an external executor.

Managed persistent image histories, private original assets and cooperative CPU scheduling are now documented in [IMAGE_SESSIONS.md](IMAGE_SESSIONS.md). Broad successful-public-HTTPS/failure campaigns, sustained fuzzing, LLaVA execution, Metal and other-platform release qualification remain open. The complete mandatory release remains subject to [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).

## Qwen3.5 source checkpoints

`qwen35-still-cpu-v1` uses the complete pinned Safetensors source, its validated processor and native encoder/merger/decoder. It accepts the same stateless message parts, transports, acquisition limits and streaming text output. Import the complete source directory under a new managed identity, then load it; a separate vision pair is unnecessary. Standalone uses `chat CHECKPOINT --standalone --image FILE --prompt TEXT`. Repeat `--image` for ordered attachments, within image/context/resource limits.

The current qualification is square/portrait constant-color PNG input and ordered two-image integration, independently frozen F32/wide-reduction arrays, native CPU on macOS ARM64. It does not establish universal image/model compatibility or publisher-framework precision parity. Qwen3.5 session-owned image attachments and metadata-only image planning are now supported through [IMAGE_SESSIONS.md](IMAGE_SESSIONS.md). GPU qualification remains open. Missing/unsupported processor assets preserve compatible text execution but explicitly reject image input. See [QWEN35.md](QWEN35.md).

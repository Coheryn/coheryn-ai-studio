# Managed image conversations and cooperative CPU scheduling

The qualified Qwen2-VL pair supports session-owned images, follow-up questions, opt-in save/restart/resume, portable image documents and shared CPU scheduling. This is an experimental macOS ARM64 CPU checkpoint. The supported model files, processor and greedy controls remain those in [IMAGE_API.md](IMAGE_API.md). Text sessions retain their existing tool, constraint, sampling and prefix-cache contracts.

## CLI workflow

Start a daemon with persistence enabled when conversations should survive restart. Register/load the compatible language/vision pair as described in the image API guide:

```sh
coheryn serve --session-persistence --max-memory 1073741824
# In another terminal, after registering qwen-vl:
coheryn models load qwen-vl --context 256
coheryn chat qwen-vl --session scene --image scene.png \
  --prompt 'Which color is on the left?' --context 256 --max-tokens 8 --json
coheryn sessions show scene
coheryn sessions send scene --input 'Describe the other side.' \
  --expected-revision 3 --idempotency-key followup --max-tokens 8
coheryn sessions save scene --expected-revision 5
```

A new conversation starts at revision 0. Uploading a new image advances it once; a turn advances once for pending input and again for its terminal state. The example's revision values assume successful operations without intervening mutations. Read the actual returned revision for automation. Exact duplicate bytes reuse the same session-owned asset and do not advance the upload revision.

After restarting with the same data directory, credentials and persistence setting, explicitly load the same model pair, then resume:

```sh
coheryn models load qwen-vl --context 256
coheryn chat qwen-vl --session scene --resume --max-tokens 8
```

The follow-up sends text; the daemon reconstructs the selected image history from its stored originals. `chat ... --session NAME --persist --image FILE --prompt TEXT` creates an immediately durable conversation. Repeat `--image` for attachments. `sessions send NAME --image FILE --input TEXT --expected-revision N` also uploads and submits images. `chat ... --session NAME --resume --image FILE --prompt TEXT` submits a single resumed image turn; `--stream --json` emits native JSON Lines. Standalone image chat remains process-local and uses its existing explicit component paths.

## Session API

The session API uses the credential fingerprint for ownership. Create a session using a loaded managed pair; the response reports `task: "image-to-text"` and binds the pair's exact immutable identity:

```http
POST /coheryn/v1/sessions
Content-Type: application/json
Authorization: Bearer TOKEN

{"model":"qwen-vl","alias":"scene","context":256,"persist":false}
```

Upload original PNG/JPEG bytes to the owned session:

```http
POST /coheryn/v1/sessions/scene/images?expected_revision=0
Content-Type: application/octet-stream
Authorization: Bearer TOKEN

<original image bytes>
```

`image/png` and `image/jpeg` content types are also accepted; actual bytes pass native container/processor validation. The response contains `session_id`, the resulting `revision`, and `image: {"id":"img-...","sha256":"...","bytes":123,"info":{...}}`. Uploads accept bounded bytes. Managed turns reference returned asset IDs; remote filesystem paths are never accepted. Stateless OpenAI-shaped messages retain their separate bounded Base64/opt-in HTTPS contract.

```json
{
  "expected_revision": 1,
  "idempotency_key": "first-image-question",
  "content": "Describe the image.",
  "images": ["img-RETURNED_ID"],
  "settings": {"max_tokens": 8, "threads": 2},
  "stream": true
}
```

Send that body to `POST /coheryn/v1/sessions/scene/messages`. Image IDs retain array order ahead of the turn's text content. Omit `images` on subsequent questions. Image-only turns may omit text. References belong to one session; duplicate references within a turn, foreign IDs, text-only model sessions and unsupported image controls fail explicitly. Initial system instructions, Unicode text, revision checks, idempotent replay, native streaming, error receipts and single-writer protection retain their session contracts. Keep the exact submitted JSON and uploaded IDs for automated idempotent retries. Upload and turn submission are separate revision-checked operations.

The current image profile supports greedy decoding with 1–256 output tokens. Tools, constrained output, custom stops and nondefault sampling are rejected. The maximum physical context is the loaded capacity, within the adapter's 4,096-position ceiling. The pair identity must still match before inference after restart/reload; no model substitution is automatic.

## Original assets, persistence and cleanup

Original encoded images live in private files under `DATA_DIR/image-assets/SESSION_ID/IMAGE_ID.bin`. Histories retain only bounded IDs and verified byte/geometry identities. On Unix, directories are 0700, files 0600, and nonregular/symlink/extra-hardlink asset files are rejected. Names are server-generated and validated. Hash and descriptor checks run before reusing stored bytes. Equal originals deduplicate within their session; ownership remains isolated across sessions.

Defaults are 16 stored assets and 64 MiB per session, 128 MiB per owner, and 512 MiB total asset disk storage. Operator flags are:

```sh
coheryn serve --image-store-bytes 536870912 \
  --owner-image-bytes 134217728 --session-image-bytes 67108864 \
  --session-max-images 16
```

The existing per-image and per-request processor limits also apply. Four images may be selected in one request by the qualified adapter; retaining 16 originals does not qualify simultaneous 16-image inference. Disk quota admission is serialized with session mutations and counts abandoned writes conservatively. The engine budget separately covers metadata, encoded reads, prepared work, KV/scratch, workers and delivery.

With `overflow: "reject"`, oversized histories fail. Explicit `oldest-complete-turns` can exclude prior complete user/assistant groups until physical context and image/message/text limits fit. Receipts identify included/excluded message IDs. Protected instructions and the current turn remain intact; image resolution and requested context are preserved. Exclusion affects model context; the authoritative history and owned originals remain inspectable until reset, deletion or expiry.

Image files are synchronized before publishing their references in a durable history transaction. Save/restart restores metadata and original assets without restoring KV or decoded pixels. Failed/cancelled turns remain inspectable and are excluded from later model context. Reset/deletion/expiry reclaim owned originals. The last session-store owner removes ephemeral files on clean shutdown; startup recovery sweeps abandoned temporary writes and ephemeral assets while preserving references from durable histories. Cleanup errors preserve data for bounded recovery and never justify following a link or deleting outside the private store.

Durable histories and original images are plaintext local data protected by ownership and filesystem access. Preserve the data directory and credentials together for recovery; model files are still explicitly loaded. Windows ACL, power-loss and hostile-operator snapshot campaigns remain release qualification work.

## Portable documents

Existing `sessions export/import` adds version-3 image documents containing original image bytes as canonical Base64, asset identities and complete bounded history/receipts. Imports verify bytes and geometry, enforce quotas, assign a new session ID/owner and publish references after file creation. Malformed imports release reservations and remove their uncommitted asset files. Text version-1 and tool version-2 documents retain their existing supported forms; the SQLite history schema stays at version 2.

Portable documents have an **8 MiB serialized limit**, including Base64 and history. Larger image conversations remain usable and save/resume normally; oversized portable export fails explicitly. Import response/runtime reservations retain only metadata/history/replay rather than the retired Base64 payload. Native KV, pixels, embeddings and execution scratch never enter a saved/exported document.

## Shared compute and measurements

`coheryn-vlm-cooperative-v1` parks the existing image worker after completed tensor units. Vision matrix operations yield after bounded weight tiles; noncausal attention yields after at most 32 complete query rows; language prefill/decode yields at decoder block/row checkpoints. All owned numeric workers join before the shared CPU permit is returned. Original-image codec work remains one bounded decode unit, with checks and yields around it.

Text token batches, embedding microbatches and image units use the same FIFO compute gate. Image output delivery releases this gate while waiting for the bounded consumer channel. One event slot stays available for the terminal result/error. Cancellation, disconnects, deadlines and callback failures retain resources until worker acknowledgement. Full request reservations and private buffers remain owned while parked, so yielding creates no release/reacquire admission race and copies no KV state.

Generation, embedding and vision have separate bounded phase residency. With `--max-concurrency 1`, an admitted text phase and an image phase can alternate compute units while the shared compute-unit limit remains one. The shared request-slot and atomic memory limits still apply. This is cooperative interleaving; compatible image requests do not share a batched forward pass.

Admin status includes original-asset counts/disk bytes, phase limits and `vision_units`, `vision_yields`, `vision_waiting_units`, `vision_backpressure_waits` and `vision_max_unit_micros`. Results include `vision.cooperative` with unit/yield counts, `compute_lease_ms` and `waiting_for_compute_ms`. These measure held-permit and intervening wait wall time, excluding initial queue/preflight from the latter. Vision/prefill/decode stage wall times can include interleaving waits; these fields are not CPU-time or whole-allocator measurements. Reservations, observed vectors, physical residency and file cache remain separate quantities.

## Qualification

Workspace coverage includes no-resend follow-ups, CLI save/restart/resume, image-only turns, byte/owner quotas, stale revisions, duplicate/foreign refs, explicit whole-turn context exclusion, malformed portable imports, mutation, original-file cleanup, orphan recovery, and text/embedding progress during image prefill and backpressured delivery. A separate 67-row scalar attention fixture crosses two complete query tiles and a short tail, with cancellation after a joined unit. Cooperative and uninterrupted native execution agree on the small independent graphs.

`python3 tools/check_image_sessions.py` qualifies the cached pinned pair through the release CLI/API, including a frozen image response, text progress with one generation slot, durable restart, a CLI follow-up without image bytes, equivalent stateless history, portable originals and cleanup. `python3 tools/check_cooperative_vlm.py` compares actual cooperative image embeddings, decoder layers and logits against the unchanged frozen full-array reference. Both execute native binaries with an empty executable search path. Their reports are `fixtures/qualification/m4-image-sessions.json` and `m4-cooperative-numerics.json`.

The current scope remains the exact Qwen2-VL FP16 CPU pair and tested settings on macOS ARM64. Broader image counts/resolutions/model configurations, LLaVA, Metal, other CPU platforms, sustained fairness/tail-latency/security/fuzz/crash campaigns and complete mandatory release qualification remain open.

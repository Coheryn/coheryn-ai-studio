# Managed native checkpoints

The experimental Qwen3.5 CPU text profile supports managed Safetensors source sets, shared daemon admission, stateless CLI/API generation and [managed text histories](QWEN35_SESSIONS.md). GGUF, embedding and image-pair interfaces retain their existing contracts. Model packages contain inert assets; compiled native adapters determine execution support.

```sh
# Acquire the pinned development source separately, then import a private copy.
python3 tools/fetch_checkpoint.py
./target/release/coheryn serve --max-memory 805306368
# In another terminal, using the same data directory:
./target/release/coheryn models import .cache/qwen3.5-0.8b-source --name q35
./target/release/coheryn models load q35 --context 256
./target/release/coheryn run q35 --messages fixtures/qwen35/smoke-messages.json \
  --context 256 --max-tokens 4 --threads 2 --json
./target/release/coheryn models unload q35
./target/release/coheryn models remove q35
```

Explicit single-file and canonical indexed entrypoints are accepted with their required neighboring assets. `models register` remains an import alias. Default run/chat commands use the daemon; `--standalone` creates an independent local instance. Interactive text without a managed session uses stateless chat messages. Qwen3.5 managed text histories can reuse bounded exact-prefix KV/recurrent/convolution snapshots. Text lanes share weight tiles through continuous batching. Complete source packages with a validated processor asset also accept stateless image inputs. Tools, schema constraints, custom stops and non-greedy sampling are unsupported. Its existing context/output limits and exact tokenizer/template checks still apply; see [QWEN35.md](QWEN35.md).

## Identity and storage

A version-2 registry record contains a version-1 checkpoint manifest: `schema_version`, `entrypoint`, compiled `adapter` profile ID and sorted `files` containing `name`, `bytes`, and lowercase complete-file `sha256`. File names are unique across case folding. An indexed set includes its index and every referenced whole-tensor shard. The native descriptor must account for every execution input; extra weight/index files are rejected. Optional recognized tokenizer configuration and LICENSE/NOTICE files are retained. Unrelated source-tree files are not copied.

The record's `sha256` is the SHA-256 of the ASCII domain prefix `coheryn-checkpoint-v1` followed by a zero byte and the canonical compact manifest JSON. Field order is `schema_version`, `entrypoint`, `adapter`, `files`; each file uses `name`, `bytes`, `sha256`. File order is ascending ASCII name. No whitespace, alternate Unicode escapes or key reordering are part of that canonical encoding. Manifest names and adapter IDs are ASCII. This binds complete weight bytes, assets, layout and native profile. It differs from a bare weight hash and from the metadata-only fingerprint. For a checkpoint, `models import --sha256` checks this full source-set identity; for GGUF it continues to check the file hash.

Imports stream into a private staging directory using bounded tiles. Each file is length/hash checked and synced; the complete set is inspected through its native adapter before publication. The complete directory is renamed into place before the atomic record pointer is installed. Unix directory synchronization follows publication. Caller timeout/disconnect cannot remove a validation worker's inputs until that worker retires. Startup under the registry lock reclaims recognized flat staging sets and checkpoint directories without a published record. Valid committed records survive restart. No archive extraction or repository code runs.

Listing checks exact filenames, regular-file types and lengths. Load rehashes every payload before constructing a resident; explicit deep validation also rehashes and checks finite tensor payloads. Inspection/planning do not rehash multi-gigabyte weights or establish numerical qualification. Their response exposes the persisted full identity separately from the descriptor's unverified metadata snapshot.

The private managed root must remain under operator control. This is not qualification against a hostile process that can concurrently replace files/directories inside that root. Handle-relative containment across all target platforms, real multi-shard model qualification and power-loss/crash campaigns remain release work. Directory synchronization is currently Unix-specific.

## Upload protocol

`POST /api/v1/models/import-checkpoint?name=ID&sha256=HASH` requires administrative authentication and `Content-Type: application/vnd.coheryn.checkpoint.v1`. The optional hash is the source-set identity above. The local CLI assembles this stream; remote callers send bytes and a bounded manifest, never a server filesystem path.

1. Twelve-byte ASCII magic `COHERYNCP1\r\n`.
2. Four-byte little-endian unsigned manifest length.
3. That many UTF-8 JSON bytes describing the manifest.
4. Exact file bytes concatenated in manifest order, followed by EOF.

The manifest is at most 48 KiB and 132 files. Portable basenames cannot contain paths, device names or case collisions. Only Safetensors weights, the canonical index, recognized configuration/tokenizer/template files and LICENSE/NOTICE are accepted. Configuration/ancillary files are capped at 1 MiB, tokenizer JSON at 16 MiB, and the text template at 16 KiB. Model byte, registry-count, disk quota and free-space limits apply before payload writes; trailing bytes, truncation, hash mismatches and unsupported source sets roll back. Uploads retain the existing 30-second idle and 600-second total limits. `models pull` remains a single-file GGUF HTTPS download; checkpoint URL discovery/pull is unavailable.

## Scheduling and public responses

Managed Qwen3.5 models charge the daemon's shared `Budget`. Metadata-only loaded/unloaded plans use the same session/recurrent/scratch arithmetic and CPU/profile policy as execution, adding an explicit 8 MiB transport allowance per request. Admission holds session/request/profile headroom before queueing. Queue slots and generation phase limits are shared with existing text work. Numeric workers join before matrix-tile/layer checkpoints release the FIFO CPU gate; no compute permit is held during output backpressure.

The model remains pinned by queued/active jobs and undrained response handles. Transport reservations stay with the handle, execution reservations stay with the worker, and cancellation waits for worker retirement. Load/unload/remove require existing admin credentials and reject conflicting residency/lifecycle operations. A removed source cannot invalidate an active request.

`/v1/completions`, `/v1/chat/completions` and `/api/v1/generate` reuse existing bounded JSON/SSE framing, usage, errors and `[DONE]` behavior. Unsupported controls fail explicitly. The greedy-only profile supports continuous text batching while preserving independent request state. Managed text histories extend this CPU profile through [QWEN35_SESSIONS.md](QWEN35_SESSIONS.md). Managed Qwen3.5 image histories, broader multimodal/backend/platform and production qualification remain open.

## Reproduction

Ordinary workspace tests cover bounded manifests, paths/symlinks, authentication, failed/partial imports, registry identity and restart reclamation without downloading models. Use the opt-in proof after acquiring the pinned source and building the release workspace:

```sh
python3 tools/check_managed_checkpoint.py \
  --report fixtures/qualification/NEW-managed-runtime.json
```

It refuses to overwrite an existing report, checks original model hashes, imports one private copy, tests exact native/CLI/chat/SSE outputs, planning/admission, unsupported controls, live cancellation, unload/restart and same-length payload corruption, then removes that copy. It invokes no external inference runtime. The report records binary/model/reference identities and scoped results; uncontrolled file-cache conditions are not called cold.

`preprocessor_config.json` is an optional bounded inert checkpoint asset. Its complete bytes participate in source-set identity when present. Missing/unsupported processors retain compatible text behavior and reject image requests before acquiring attachments. See [QWEN35.md](QWEN35.md) for still-image scope and the socket-free HTTP qualification target.

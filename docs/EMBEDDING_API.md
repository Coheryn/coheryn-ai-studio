# Embedding CLI, API and external RAG — E2

E2 exposes the native Nomic F16/Q8_0 encoder through standalone CLI, daemon client and authenticated `POST /v1/embeddings`. The model graph, tokenizer, pooling, dimensions and precision criteria remain those qualified in [E1](EMBEDDINGS.md). No generation model is required. Exact files/backend/platform evidence lives in [the compatibility matrix](../fixtures/compatibility.json). This is an experimental macOS ARM64 CPU preview; image understanding, Metal and other-platform qualification remain separate gates.

## Run locally

With the pinned files already in `.cache/models`, a local path creates one separate process-local embedding instance:

```sh
target/release/coheryn embed .cache/models/nomic-embed-text-v1.5.f16.gguf \
  --input 'search_query: What is local inference?'

target/release/coheryn embed .cache/models/nomic-embed-text-v1.5.Q8_0.gguf \
  --file documents.txt --output embeddings.json --batch-size 8
```

The explicit local path selects standalone operation; `--standalone` is also accepted. A bounded diagnostic on stderr identifies the separate instance. Data on stdout is always JSON, with or without `--json`.

To share models through the daemon:

```sh
target/release/coheryn serve --max-concurrency 2 --embedding-concurrency 2
# In another terminal, with the same --data-dir if customized:
target/release/coheryn models import .cache/models/nomic-embed-text-v1.5.f16.gguf --name nomic
target/release/coheryn models load nomic --context 512
target/release/coheryn embed nomic --input 'search_query: What is local inference?'
target/release/coheryn plan nomic --task embedding --context 512 --input-bytes 64,128
target/release/coheryn bench --task embedding --model nomic --context 512 --batch-items 8
```

Model residency remains explicit. Managed IDs use the daemon and never fall back to standalone. `models load` must use a context within the embedding model's qualified 512-token limit; the existing generation-oriented load default is 4096. Generation endpoints reject embedding models, and the embedding endpoint rejects generation models. Imports, inspect/deep validation, load/unload/remove and status use the same managed registry and role credentials. Active, queued and undelivered response buffers pin the resident model.

## CLI file contract

Supply exactly one of `--input TEXT` and `--file PATH`. File input is UTF-8 with one prepared text record per line. Only LF or CRLF delimiters are removed; leading/trailing spaces, tabs and a final record without LF are preserved. Empty records, invalid UTF-8 and oversized records fail with a record number without echoing the document. An empty file is invalid. Embedded newlines belong in single `--input`/API text, rather than in this line-oriented file format. Coheryn does not parse documents or create semantic chunks.

| Option | Default / contract |
|---|---|
| `--batch-size` | 8; positive compute microbatch ceiling, at most 64 in the engine |
| `--max-input-tokens` | 512 including CLS/SEP; explicit values are never silently reduced |
| `--max-batch-tokens` | 2048; at most 4096 packed tokens |
| `--request-size` | 32 file records per bounded window; 1–256, independent of compute batch size |
| `--truncate` | Off; explicit model-qualified truncation records each affected input |
| `--profile`, `--threads` | Existing CPU profiles and bounded worker policy |
| `--max-memory` | 1 GiB; local engine/client admission policy, separate from physical RAM |
| `--encoding-format` | `float` or `base64` |
| `--output` | Atomic JSON file export; otherwise stdout is buffered and limited to 8 MiB |
| `--overwrite` | Required to replace an existing destination |
| `--max-output-bytes` | 1 GiB file quota; cannot enlarge the 8 MiB stdout ceiling |
| `--max-file-bytes` | 1 GiB input-file quota; also at most one million records |

File windows also have finite byte bounds: up to 8 MiB for standalone inference, and 128 KiB of raw text for the daemon client so worst-case JSON escaping fits the default request-body limit. A requested record ceiling may therefore produce smaller windows. Operator limits can be stricter and produce explicit rejection. A standalone model stays loaded across all windows.

File output streams indexed vectors into a private temporary file in the destination directory, accumulates checked usage and verifies model SHA-256, dimension and embedding-space identity across windows. Flush/sync and atomic publication occur only after complete success. Cancellation, input errors, quota/disk errors, daemon failures and identity switches remove the unfinished temporary file. Existing destinations survive failed exports. Stdout is published only after the entire bounded JSON result succeeds; oversized results fail with guidance to use `--output`.

CLI JSON retains the standard `object`, `data`, `model` and `usage` structure, adds per-row `prompt_tokens`/`truncated`, and puts identity and effective limits under `coheryn`. Indices span the entire file. Exit codes remain: invalid input/record/limit `2`, I/O or model change `3`, unsupported dimensions/compatibility `4`, memory/output quota `5`, credentials `6`, queue/conflict `7`, unavailable daemon `8`, cancellation `130`.

## OpenAI-compatible API

The standard wire subset follows the [official Create embeddings reference](https://developers.openai.com/api/reference/resources/embeddings/methods/create). Coheryn uses its local model/operator limits and supports **strings and arrays of strings**. It rejects pretokenized integer arrays, unknown fields, generation options and unimplemented modes.

```sh
curl http://127.0.0.1:11450/v1/embeddings \
  -H "Authorization: Bearer $COHERYN_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"model":"nomic","input":["search_query: local inference","search_document: Models run on your computer."]}'
```

`COHERYN_API_TOKEN` is a Coheryn inference credential. Administrative credentials also allow inference; inference credentials cannot load/import/unload models. Ordinary local CLI clients read the private Coheryn credential vault automatically. No OpenAI account or hosted API key is needed for this local endpoint.

| Field | Supported behavior |
|---|---|
| `model` | Required managed model ID, explicitly loaded with embedding capability |
| `input` | Required nonempty string or nonempty array of nonempty strings; order and duplicates preserved |
| `encoding_format` | `float` default, or `base64` containing ordered little-endian float32 components |
| `dimensions` | Omitted or the native dimension (768 for the pinned models); other values rejected |
| `user` | Optional opaque string up to 256 UTF-8 bytes; no logging or telemetry requirement |
| `batch_size`, `max_batch_tokens` | Optional compute ceilings within operator policy |
| `max_input_tokens` | Optional explicit input limit; omission uses the smaller of loaded context and operator maximum |
| `truncate` | False by default; explicit qualified truncation only |
| `profile`, `threads`, `max_memory` | Coheryn execution/admission extensions |

No implicit query/document prefix is added. Callers supply qualified prefixes literally. `input_type`, automatic document chunking, reduced dimensions and generation hidden-state extraction are unsupported.

The response contains `object: "list"`, `model`, ordered `data` entries with `object: "embedding"`, `index`, `embedding`, and exact `usage.prompt_tokens`/`total_tokens`. Counts include consumed qualified special tokens and exclude padding/generation. Every vector must be finite and have the declared dimension. Any failed item fails the whole request; there are no success-shaped partial batches. Embeddings use bounded JSON, while text/chat retain SSE.

Response headers expose identity and diagnostics without adding fields to the standard response body:

- `X-Coheryn-Embedding-Space`, `X-Coheryn-Model-Sha256`, `X-Coheryn-Embedding-Dimension`.
- `X-Coheryn-Token-Counts`, `X-Coheryn-Truncated-Indices` (comma-separated original indices).
- `X-Coheryn-Batch-Size`, `X-Coheryn-Batch-Tokens` (selected compute ceilings).

JSON shortest decimals and Base64-decoded values can differ when widened to binary64 by a client. They round-trip to the **same float32 bits**. The SDK qualification checks those bits and the frozen numerical tolerances.

## Operator limits, planning and lifecycle

Defaults are 256 items, 64 KiB per item, 8 MiB aggregate decoded text, 1 MiB JSON body, 512 tokens per input, 32768 actual tokens per request, a compute ceiling of 8 items/2048 tokens, and a 16 MiB response ceiling. The effective limits intersect model, operator and admission constraints. `serve --help` lists the finite `--embedding-max-*` flags. `--embedding-requests` limits outstanding embedding requests (default 4), `--embedding-concurrency` limits active embedding compute (default 1, within shared concurrency), and `--embedding-timeout` supplies the total request deadline (default 300 seconds). Queue and consumer bounds remain shared with generation.

Ingress reserves bounded raw JSON/Serde storage before reading. An honest bounded `Content-Length` permits a smaller reservation; the same declared read ceiling is enforced, so understated lengths cannot bypass it. Chunked/unknown-length input reserves the configured worst case. Compute reserves tokenizer/token storage, reusable encoder buffers, worker stacks and all output vectors. Serialization has a separate checked upper bound reserved before execution. Parsed text storage is released after preparation. Engine vectors are released after serialization; reference-counted HTTP byte buffers retain the serialization reservation and model pin through delivery, including the last chunk. Delivery checks the consumer deadline when polled; stalled clients remain bounded by response/HTTP-slot/budget limits and cannot create unbounded buffers. Reservations are not an OS-wide memory cap.

Embedding preparation is followed by resumable microbatches. Between microbatches, the request returns its active permits to the shared fair scheduler while keeping its reserved workspace and tokenized inputs. Queued text generation can then run. Low-memory execution uses the existing exclusive global permits. Rejections, queue deadlines, disconnects, shutdown and cancellation wait for compute acknowledgement before releasing live resources. No generation KV cache is allocated for an encoder.

`plan --task embedding` and the versioned estimate route accept bounded `input_bytes` length descriptors plus `context`, `concurrency`, profile and optional embedding ceilings. They inspect metadata without loading weights or tokenizing text. Plans expose immutable space identity, model/serialization/ingress reservations, dimension, stage costs and rejection/admission status. Actual token counts and aggregate token limits are rechecked before inference. `bench --task embedding` reports identity, workload, selected ceilings, vectors/s, input-tokens/s and available memory counters. It labels cache conditions uncontrolled and makes no comparative performance claim.

Error conventions: malformed/unsupported input or dimensions `400`; body/item/byte/token/response-policy limits `413`; unknown model `404`; authentication `401/403`; queue saturation `429`; unavailable residency or admission exhaustion `503`; cancellation/deadline `408`. The existing structured error envelope carries descriptive codes including `UNSUPPORTED_EMBEDDING_DIMENSIONS`, `EMBEDDING_INPUT_LIMIT_EXCEEDED` and `EMBEDDING_BATCH_LIMIT_EXCEEDED`. Messages report bounded indices/limits, never document text or vectors.

## Optional external RAG integration

The [LangChain/Qdrant example](../examples/rag_client.py) calls Coheryn for embeddings and stores supplied vectors in Qdrant's local in-memory mode. It checks the response embedding-space header across document/query requests and stores that identity with each point. Persistent applications must retain the identity with their index and explicitly rebuild or segregate vectors when it changes.

```sh
python3 -m pip install -r tools/requirements-rag.txt
COHERYN_API_TOKEN='your-coheryn-inference-token' \
  python3 examples/rag_client.py --model nomic
```

The pinned optional clients are OpenAI Python 3.39.0, LangChain OpenAI 1.7.3 and Qdrant Client 1.20.0, verified as latest stable on 2026-10-27. For LangChain, `check_embedding_ctx_length=False` is required to send raw strings and let Coheryn use its own tokenizer; pretokenized arrays are rejected. The example adds query/document prefixes explicitly and disables environment proxies for its HTTP client. Full RAG orchestration, ingestion, vector storage and retrieval remain optional, outside Cargo and the core inference dependency graph.

## Qualification

```sh
./tools/cargo test --workspace --locked
./tools/cargo build --release --workspace --locked
python3 tools/check_embedding_fixtures.py
python3 tools/check_embedding_service.py
```

The last command needs the cached pinned models and optional packages installed in an isolated environment or `.cache/rag-tools`. It starts an ephemeral loopback daemon under the ignored cache, verifies API vectors against independent fixtures, exercises 50 multilingual/Unicode cases per file, SDK default/float/Base64 forms, CLI/client file parity, external RAG/vector-store interfaces and unload cleanup. Temporary model copies and credentials are removed. Raw content/vectors are not logged. The resulting `fixtures/qualification/e2-service.json` records model hashes, executable identity and client versions. Frozen references/tolerances are unchanged. Unicode/numerical parity does not establish semantic proficiency in every language.

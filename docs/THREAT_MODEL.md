# G0/G1 threat model

## Trust boundaries

Local model paths, GGUF metadata, tensor dimensions/offsets, tokenizer tables, prompt text and chat messages are untrusted. The HTTP service separates inference/admin credentials and accepts managed model IDs or bounded binary uploads, never remote filesystem paths. Reference tools remain development-only and separately executed.

## Controls implemented

| Threat | Control | Evidence |
|---|---|---|
| Header/count/string allocation attacks | Checked arithmetic, explicit count/byte/depth/allocation limits, minimum encoded extents before array allocation | Parser tests and fuzz target |
| Tensor overlap, overflow, truncation | Checked dimensions, encoding block sizes, alignment, file ranges and non-overlap | Valid/invalid and mutated fixtures |
| Unsupported execution semantics | Architecture/shape/type checks before session allocation; reject shards, scaling and unknown graph variants | Compatibility validation |
| Template computation/output attacks | Exact known template fingerprints; bounded messages/output, recursion and fuel; no external functions/files | Tokenizer/template tests |
| Mutable model files | Buffered reads, checked row ranges, header revalidation, length/mtime checks, recoverable read errors | Truncation and mutation tests |
| Request memory exhaustion | Context reservation before compute, demand-grown KV bounded by context, reusable scratch/I/O chunks, bounded output | Budget/context/cleanup tests |
| Consumer disconnect or cancellation | Synchronous fallible token sink, shared cancellation, bounded compute checks, RAII session release | Sink failure/cancellation tests |
| Partial/corrupt model acquisition | Pinned HTTPS revision and SHA-256, byte ceiling, disk-space check, temporary file plus fsync/atomic rename | Acquisition utility and verified reference artifact |

## Remaining limitations

Reservations do not instrument allocator internals inside general-purpose libraries. A malicious local writer can evade timestamp checks; files must remain immutable while executing. Buffered reads avoid memory-mapping crashes but cannot guarantee coherent contents under hostile concurrent mutation. Filesystem-backed model identity is qualified by offline SHA-256 verification.

The parser bounds accepted allocations but cannot make host allocation failure impossible. OS memory pressure, file cache residency, drivers, and other processes are outside its budget. Parser and numeric operations contain no unsafe Rust. Transitive dependencies have their own implementation and safety boundaries.

G2 implements loopback defaults, distinct inference/admin credentials, bounded bodies/headers/queues/SSE, managed model identifiers, path/symlink checks, HTTPS redirect/DNS restrictions, disk quotas and pin-aware unload. Credentials live in private local storage and are never returned or logged. The CLI does not forward its local vault to remote endpoints. Remote binding requires explicit configuration; use TLS termination for remote deployments. Sustained API fuzzing, platform credential review and release hardening remain G6 work.

## Fuzzing

Run `cargo +nightly fuzz run gguf -- -max_len=1048576` from `fuzz/` after installing cargo-fuzz. The target uses stricter allocation limits than ordinary model inspection. Fast deterministic mutation coverage runs in ordinary `cargo test`. A committed fuzz target is not evidence of a completed long fuzz campaign; sustained fuzzing and sanitizers remain G6 release gates.

## G3 admission and M0 boundaries

Queued requests now hold full session capacity, payload and profile headroom atomically until cancellation/completion; ownership transfers into the engine without releasing the reservation. Engine vector-capacity counters are separate from estimates and OS measurements. Available-RAM probes may be unavailable, and OS pressure is not atomically reservable. Low-memory tiles are validated against tensor rows before execution.

M0 source/config/header acquisition is a bounded development tool, separate from production managed loading. Declared full-file hashes do not prove downloaded tensor integrity. No VLM graph, remote image acquisition or image decoder is enabled in production; M1 must implement the v1.5 decoder/SSRF/stage-reservation controls before acceptance. Ordered content and component-manifest structures reject excess bounds, unsafe identities and false provenance status; structural validity is not execution compatibility.

## G4 packed execution and E0 references

Packed row extents are checked against encoding blocks, tensor shapes and the admitted I/O tile. Deep validation scans finite scales using bounded encoded and decoded buffers; dot products reject non-finite results. Activation quantization checks finite inputs, scaling and reserved capacity. Hostile block/shape/scale fixtures and independent reference bytes cover these boundaries. No unchecked pointer arithmetic or new unsafe blocks are introduced.

Explicit threads must fit the configured CPU share and the 1–16 ceiling. Worker stacks and overhead enter the session reservation; request-owned pools join before that reservation is released. Cancellation is checked between bounded I/O/row groups and prefill token forwards. This is cooperative cancellation, not an OS scheduling latency guarantee. Full-file page-cache estimates are advisory for buffered reads; additional engine allocations still require budget and observed-RAM admission.

E0 downloads and executes separately pinned reference tooling for byte-verified Nomic GGUF assets. Publisher remote-code declarations are never executed. E1 enables the validated native Nomic encoder through the Rust library; E2 embedding request routes now enforce bounded parsing, task-aware residency and shared admission. external reference engine is only a pinned comparator artifact; no daemon, cloud request or user-home identity was created by reference preparation.

## E1 embedding controls

Embedding inputs are limited to 256 items, 64 KiB per item and 8 MiB total; tokens include CLS/SEP and are limited to 512. Oversized inputs require explicit truncation or fail before execution. Strict graph shapes/metadata/dtypes and a task-specific tokenizer prevent generation/encoder confusion. Normalization follows the qualified GGUF contract and must not clean arbitrary application text. Atomic budget/concurrency guards, immutable admissions, bounded matrix tiles, reusable per-worker attention rows and result-owned output reservations prevent unbounded batch/workspace growth. Cancellation, observer failures and changed storage release session capacity; test fixtures cover admission races and failed handoffs. Third-party tokenizer internals still use conservative allowances rather than complete allocator instrumentation. Multilingual parity is distinct from model language/retrieval quality.

## E2 transport/export controls

Authenticated embedding JSON has separate body/item/UTF-8/token/response ceilings. Bounded ingress reservations precede reads; an advertised smaller Content-Length is enforced as the actual read limit. Unknown fields, pretokenized arrays and wrong model tasks fail explicitly. Serialization is reserved before execution and response byte ownership retains its budget/model pin through the last chunk. Finite queues, per-task active/outstanding caps, cooperative deadlines, microbatch yielding and disconnect/shutdown acknowledgement bound work and protect generation scheduling. No raw documents, queries, vectors or opaque user identifiers are logged by default.

CLI records are read in finite windows with byte/record quotas; outputs use private temporary files and atomic publication. Failed/cancelled exports preserve existing destinations and remove temporary data. Client file windows pin model hash/dimension/embedding-space identity; changes abort publication. The optional external RAG example checks the same identity before mixing document/query vectors and stores it with indexed points. A literal URL in embedding text is never fetched. Full allocator instrumentation, sustained fuzz/security campaigns and broader backend/platform fault coverage remain release work.

## S0–S3 boundaries

Archived capture code is inert research, outside native build/test/runtime paths. Source/path, all-feature workspace/fuzz graph and controlled executable checks are distinct evidence. Original model/tokenizer assets and numerical outputs retain their digests. New assessments execute separately and provide verified inert data.

Managed histories default to ephemeral and are owner-scoped by authenticated credential fingerprints. Opaque IDs are not credentials; canonical IDs should be used for automated retries. One writer and exact revision/idempotency checks prevent mixed histories and duplicate inference. Reset/delete refuse active turns. Expiry never reclaims a live lease. Cancellation and shutdown retain ownership until worker acknowledgement; interrupted inputs are marked incomplete and excluded from future model context. Request/session/history/replay/output quotas bound growth. The small delete control plane remains usable under a fully reserved workload budget.

No suggested tool or saved history executes host side effects. Persistent-history/import/export controls are described below; tool/schema and vision extensions require their own security qualification. Raw histories are not default logs. Session responses hold bounded serialization reservations independently of deleted/expired state. General-purpose library/HTTP overhead and OS residency are distinct from conservative workload reservations.

S2 caches retain only isolated KV and a bounded token ledger, keyed by exact prefix, model/tokenizer/context/execution identity. Atomic credit transfer avoids admission gaps/double charging. Cache ownership cannot pin unloaded weights. Reset, edits, reload, changed settings, cancellation, errors, TTL and pressure force safe truncation/invalidation/eviction. No cross-principal/conversation reuse is permitted.

S3 storage is disabled by default and never consulted in disabled mode. Enabled storage uses private fixed paths, refuses unsafe database file types/links/Unix permissions, validates application/schema version and integrity, and applies logical owner/global quotas plus bounded database/journal allocation. Imported inert versioned documents have finite transfer/text/record limits, coherent IDs/roles/replay state and new caller ownership. No attachment URL, tool or model executes on import/recovery. Exports are plaintext local files using private atomic publication; existing output requires explicit overwrite.

Pending-input and terminal-history commits are transactional; successful receipts publish only after commit. Failed commits roll back RAM changes, release the writer and expose unsaved/recoverable state. Restored pending turns become cancelled without execution. Metadata-only startup, paged history reads, conservative hydration/serialization reservations, retention and idle RAM eviction bound resource use. Detached storage workers retain their operation reservations through completion even after HTTP cancellation. Unknown/corrupt stores are preserved and rejected rather than silently reset. No encrypted-at-rest store, credential migration, schema migration or complete power-loss/security campaign is claimed by this CPU preview.

## S4 tool protocol

Function proposals are inert. Exact qualified file hashes, known template assets, bounded schema trees/names/arguments, typed role/result-ID ordering and output validation precede successful tool responses. Strict schemas, external schema references, unknown tool choices/models and conflicting custom stops fail explicitly. Raw argument JSON is preserved without numeric or whitespace repair; control delimiters are decoded only under the qualified output policy. Multiple tool deltas remain provisional until terminal success; cancellation/length/malformed output cannot claim a completed call.

Pending calls, results and the companion profile/hash persist as data. Restore/import/replay cannot execute a tool or load weights automatically. Transactions roll back every result in a failed batch. Context selection retains/removes whole linked call/result groups; schema/text changes alter exact rendered token prefixes. Ingress, schema retention and worker ownership remain bounded through cancellation. No default history/argument logging is added. Broad crash, security and platform campaigns remain release gates.

## M1 image frontend foundation

Native container preflight bounds encoded bytes, source dimensions/pixels, marker/chunk counts, EXIF directories and orientation before pixel buffers. PNG ranges and every chunk CRC are checked; unsupported animation, palettes, profiles and JPEG variants fail explicitly. Canonical processed pixels/features have separate finite ceilings. Ordered batches admit aggregate bytes and stage peaks atomically, process sequentially, retain guarded outputs and clean up the whole request on error. The image fuzz harness has strict small input/pixel/memory limits. See [IMAGES.md](IMAGES.md) for the qualified subset and best-effort third-party codec allocation/cancellation boundaries. Sustained fuzzing and network acquisition security remain open. No image URL is fetched and no image-conditioned model is executed by the current product routes.

## Native vision component boundary

The Qwen2-VL component loader bounds metadata entries, tensor counts, strings, arrays, depth and aggregate header allocation before execution. It rejects unrecognized vision options, shards and incompatible/missing/extra tensors, derives FFN sizes from actual tensors and checks every buffered extent. Complete-file hashing and repeated file-state checks identify ordinary changes; callers must preserve immutability, and no filesystem snapshot against hostile concurrent mutation is claimed. Model, prepared image, reusable scratch, worker allowances and retained projected output share atomic budget reservations. Cancellation and observer failures release temporary state after workers finish. [VISION.md](VISION.md) records finite workload ceilings and measurement limits. Current HTTP/CLI generation still accepts no image input; network acquisition, complete model-chain admission and multimodal session security remain open.

## Offline multimodal decoder

The local Qwen adapter bounds decoder metadata/counts, accepted dimensions/encodings, exact template/marker behavior, message/part/text/image totals, physical context and rotary sections. Reserved chat/media delimiters in text, unused/duplicate/out-of-range attachment indices, unqualified roles/tools/video, incompatible component widths and unsupported graphs fail explicitly. Before execution, one shared-budget reservation covers full decoder KV/scratch and maximum sequential processor/vision peak. Stage guards transfer and recycle capacity; failures, cancellation, deadlines and callback errors release request state after workers finish. Original model-file immutability rules continue to apply. The development runner accepts bounded local files and optionally writes explicit traces; it performs no URL acquisition. These checks do not qualify network ingress, filesystem containment for remote users, server scheduling or managed multimodal histories.

## Product image ingress checkpoint

The current CLI/API accepts a strict text/image schema for qualified managed pairs. Incremental body admission, a lexical JSON complexity ceiling, bounded canonical Base64, declared-versus-sniffed media types and header geometry checks precede decoding. Unsupported model/role/control combinations fail before acquisition. Optional public HTTPS checks all bounded DNS answers, pins connections, rejects credentials/fragments/private/reserved targets and revalidates each redirect; proxy and implicit redirect behavior are disabled. Aggregate bytes, stage peaks and acquisition/execution deadlines remain finite. Pair records pin component identities, refuse nesting and prevent referenced-component deletion. Queue and response ownership pin both loaded components. Complete prepared admission transfers into shared compute; blocked delivery retains a terminal slot, cancels on timeout/disconnect and awaits worker acknowledgement. Retired ingress capacity is released before waiting. Image bodies, query credentials and prompts do not enter ordinary request logs. Earlier paragraphs describe historical component checkpoints. Persistent image assets/histories, hostile-operator file snapshots, full allocator instrumentation and sustained public-network/fuzz campaigns remain open.

## Managed image assets and cooperative compute

Authenticated session uploads retain private immutable original bytes behind bounded opaque IDs. Session mutation locking makes disk-quota/revision publication atomic; byte/geometry hashes, directory containment and Unix private/no-extra-link file checks precede reuse. Save publishes synchronized files before durable references; failed imports remove uncommitted assets, reset/delete/expiry reclaim originals, and bounded startup recovery preserves durable references while sweeping abandoned data. Export/import format 3 carries bounded canonical original bytes and assigns new ownership/IDs. Retired Base64 stays outside retained-history reservations.

Yield checkpoints occur after numeric workers join. Shared CPU ownership is released while waiting for another compute unit or a slow client; complete admitted buffers and the writer lease survive those waits. Supervisors await worker acknowledgement before committing cancellation or releasing resources. Phase residency, shared slots and atomic budgets remain bounded. See [IMAGE_SESSIONS.md](IMAGE_SESSIONS.md). Cross-platform ACLs, power-loss/hostile filesystem snapshots and sustained fairness/security/fuzz campaigns remain open.

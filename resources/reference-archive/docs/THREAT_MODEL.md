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

E0 downloads and executes separately pinned reference tooling for byte-verified Nomic GGUF assets. Publisher remote-code declarations are never executed. E1 enables the validated native Nomic encoder through the Rust library; E2 embedding request routes now enforce bounded parsing, task-aware residency and shared admission. Ollama is only a pinned comparator artifact; no daemon, cloud request or user-home identity was created by reference preparation.

## E1 embedding controls

Embedding inputs are limited to 256 items, 64 KiB per item and 8 MiB total; tokens include CLS/SEP and are limited to 512. Oversized inputs require explicit truncation or fail before execution. Strict graph shapes/metadata/dtypes and a task-specific tokenizer prevent generation/encoder confusion. Normalization follows the qualified GGUF contract and must not clean arbitrary application text. Atomic budget/concurrency guards, immutable admissions, bounded matrix tiles, reusable per-worker attention rows and result-owned output reservations prevent unbounded batch/workspace growth. Cancellation, observer failures and changed storage release session capacity; test fixtures cover admission races and failed handoffs. Third-party tokenizer internals still use conservative allowances rather than complete allocator instrumentation. Multilingual parity is distinct from model language/retrieval quality.

## E2 transport/export controls

Authenticated embedding JSON has separate body/item/UTF-8/token/response ceilings. Bounded ingress reservations precede reads; an advertised smaller Content-Length is enforced as the actual read limit. Unknown fields, pretokenized arrays and wrong model tasks fail explicitly. Serialization is reserved before execution and response byte ownership retains its budget/model pin through the last chunk. Finite queues, per-task active/outstanding caps, cooperative deadlines, microbatch yielding and disconnect/shutdown acknowledgement bound work and protect generation scheduling. No raw documents, queries, vectors or opaque user identifiers are logged by default.

CLI records are read in finite windows with byte/record quotas; outputs use private temporary files and atomic publication. Failed/cancelled exports preserve existing destinations and remove temporary data. Client file windows pin model hash/dimension/embedding-space identity; changes abort publication. The optional external RAG example checks the same identity before mixing document/query vectors and stores it with indexed points. A literal URL in embedding text is never fetched. Full allocator instrumentation, sustained fuzz/security campaigns and broader backend/platform fault coverage remain release work.

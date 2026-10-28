# CPU adaptive execution planner and M0 provenance

The experimental G4 CPU planner shares one atomic reservation budget with model loading and request scheduling. The pure planner accepts validated metadata, an explicit hardware snapshot and current reservations; identical inputs produce identical plans. It never loads tensor payloads to estimate a model. The daemon now supports explicit embedding workload plans and uses the E1 estimator with E2 transport-stage reservations over the shared budget.

```sh
target/release/coheryn serve --max-concurrency 2
target/release/coheryn plan MODEL_ID --context 4096 --profile balanced --concurrency 2 --json
target/release/coheryn run MODEL_ID --prompt Hello --context 512 --max-tokens 8 --profile low-memory
```

The configured concurrency is the resident-generation ceiling, bounded by available logical CPUs; default one remains unchanged. Embedding phase limits are separate, with a shared exclusive compute gate in S6. `status` distinguishes configured and effective limits. Two requests share one resident model but own their KV, scratch, decode/I/O and sampling buffers. Explicit context, generation settings and model compatibility are never changed to make a request fit.

| Profile | I/O tile | Execution policy | Reserved request headroom |
|---|---|---|---|
| `reference` | 256 KiB | Configured effective concurrency | 0% beyond the conservative baseline estimate |
| `performance` | 256 KiB | Configured effective concurrency | 5% |
| `balanced` (default) | 256 KiB | At most two balanced requests, also subject to the global limit | 10% |
| `low-memory` | Smallest power of two covering the largest validated tensor row, at least 16 KiB | Acquires all generation-residency slots; embedding units may alternate through exclusive shared compute | 5% |

Each session uses one I/O tile plus a decode buffer twice its size. For the pinned FP16 text model, low-memory selects 16 KiB rather than 256 KiB and saves 737,280 bytes of these buffers per session. Longer contexts still require their full F32 KV capacity. More disk reads may reduce throughput; no speed advantage is claimed. Every profile retains fixed per-row arithmetic and F32 KV. Reference/low-memory use one compute thread; balanced defaults to at most two and performance to at most four, bounded by logical CPUs divided by configured concurrency. Explicit `threads` must fit that share and the 1–16 limit; reference/low-memory require one. Dedicated request-owned worker pools reserve 512 KiB stacks plus 64 KiB overhead per worker and join before releasing their reservation. Packed rows operate directly on quantized weights with one reusable quantized input vector per matrix. Configured prefill chunk ceilings are 1/8/16/32 tokens by profile. S6 uses a smaller effective unit of one model token per live sequence, sharing matrix weight tiles across compatible prefill/decode vectors. Within-sequence multi-token prefill GEMM and ISA-specific SIMD remain future optimization work. Metal is not implemented.

`run`, `chat` and integrated `bench` accept `--profile` and `--threads`; standalone execution applies the same profile and admission logic. `plan` uses the daemon and managed model ID, without loading weights. `bench` labels cache conditions uncontrolled. Plans and generation accept only implemented profile names.

## Reservations and observations

A plan reports shared weight bytes once per model, model metadata needed only when unloaded, exact architecture-based KV capacity, conservative workspace/payload allowances, profile headroom, requested context/concurrency, device placement and structured reasons. Buffered weights have no full-model heap copy or weight mapping; GPU allocations/transfers are zero for this CPU backend. Available GPU capacity is unknown, represented as null.

The scheduler atomically reserves payload + complete session capacity + headroom before queue admission completes. It retains this reservation while queued, then transfers the session portion directly to engine execution. There is no release/reacquire gap. A separate waiting semaphore preserves the configured queue bound even when a profile uses fewer workers than the global ceiling; zero queue capacity permits only immediately runnable work. Queue capacity is an upper bound, not a promise that every queued session can fit in memory. Timeout, rejection, disconnect and cancellation release ownership only after compute finishes. Mixed profiles respect the shared budget and global slots; low-memory is exclusive, while balanced limits its own concurrent requests to two.

`memory.reserved` and `peak_reserved` are estimates held by RAII guards. `session_buffer_capacity_bytes` and its peak observe actual vector capacities for engine KV, scratch and I/O buffers. They exclude reserved worker stacks/runtime overhead, tokenizer/template libraries, strings, transport allocations, allocator overhead, fragmentation and OS page cache. Full allocator instrumentation is not implemented. HTTP transport buffers have their separate existing bounds.

OS RAM/swap observations are separate from both counters. Zero/unavailable available-RAM probes become null because some sandbox/platform probes return zero without a usable observation. With an available-RAM observation, `host_allocation_feasible` checks additional engine reservations. `host_memory_feasible` separately reports the conservative full-weight working-set bound; for buffered storage its failure produces an advisory warning, since loading a file does not allocate a full-weight heap copy. The estimate can overcount already cached pages. Host pressure is a snapshot, not an atomic OS reservation. Only the engine budget provides atomic admission; `--max-memory` cannot impose an OS-wide physical RAM cap. Unified-memory sharing is not double-counted as an extra GPU allocation.

`/api/v1/plans/estimate` accepts `model`, `context`, `profile`, `concurrency`, optional `threads` and an optional image descriptor workload:

```json
{
  "model": "MODEL_ID",
  "context": 512,
  "profile": "balanced",
  "concurrency": 1,
  "images": {
    "processor_profile": "unqualified-candidate",
    "images": [{"width": 336, "height": 336}]
  }
}
```

The service planner performs no image URL fetch or decoding. Registered qualified Qwen2-VL pairs now accept `task=image-to-text` and dimension descriptors, reporting native processor/encoder/decoder costs, transport, metadata and explicit admission reasons; see [IMAGE_API.md](IMAGE_API.md). A text-only model still returns inadmissible/null vision stages, because text estimates cannot establish a vision graph. Exact prompt tokens, physical/rotary positions and attachment limits are checked before the owned request receives full queue-held admission.

## M0 evidence and remaining acceptance

[The VLM selection manifest](../fixtures/vlm/models.json) pins LLaVA 1.5 7B Q4_K_M plus an F16 vision/projector component, and Qwen2-VL 2B F16 plus its F16 vision/projector component. Pins include repository revisions, declared whole-file hashes/lengths, native GGUF header inventories and source processor/tokenizer asset hashes. Small configs/licenses are committed; tokenizer binaries/large JSON and header prefixes stay ignored. Declared licenses are provenance fields, not a completed license or redistribution review.

The component inventories reveal a Llama language graph with actual Q4_K/Q6_K/F32 tensors for the selected LLaVA file; its CLIP component contains F16/F32 tensors and a model-specific feature-layer selection. The selected Qwen-VL uses a distinct `qwen2vl` graph and merger. Neither is the current Qwen2 text adapter. LLaVA M2 qualification requires G4 packed kernels in addition to the new language/vision/integration adapters.

The Qwen2.5-VL 3B candidate was excluded after its conversion metadata and pinned upstream LICENSE disagreed. The exact observations and source revision are recorded in [rejected candidates](../fixtures/vlm/rejected-candidates.json); no production compatibility or licensing conclusion is inferred from a repository card.

Reproduce selected provenance with `python3 tools/prepare_vlm.py`. This fetches bounded pinned source assets and GGUF prefixes, validates their checksums and runs the development-only native header inventory example. Full GGUF payloads are not downloaded or hash-verified by this tool. `--pin` explicitly refreshes revision selection; it is not part of normal reproduction. The tool never executes an external runtime or unverified weights.

[Procedural PNG/JPEG inputs](../fixtures/vlm/images/manifest.json), ordered/image-sensitive cases and [comparison settings](../fixtures/vlm/reference-contract.json) are committed. Numeric criteria are frozen before native vision implementation, but the newer [image-stage checkpoint](IMAGES.md) now adds all four complete file hashes and 20 independent processor cases. The [reference checkpoint](VLM_REFERENCE.md) now resolves the selected reference graph policy and adds single/two-image logits/greedy, Unicode and EOS inputs. Native execution and model-chain release qualification remain open. M1 has native processor-only staged admission; secure ingress and product attachments remain open. M2 has a native Qwen2-VL vision/merger component checkpoint; language integration and complete workload admission remain open. These records do not expand the executable entries in `fixtures/compatibility.json`.

## Verification

`cargo test --workspace --locked` covers deterministic planning, context/byte overflow, sharing arithmetic, unknown counters/vision costs, atomic reservation races/handoff, low-memory output/buffer parity, concurrent sessions, exclusive low-memory execution, queued-memory rejection and cleanup. M0 tests reject unsafe/duplicate component identities and false verification status, and preserve ordered bounded multimodal content without enabling it on the text API.

`python3 tools/check_planner.py` uses the existing pinned FP16 cache and a real loopback daemon. It checks contexts 128/512/4096, concurrency one/two and all profiles, exact independent golden tokens, observed buffers, real two-session admission and disconnect cleanup. Evidence goes to `fixtures/qualification/g3-planner.json`. Short prompts with a reserved capacity are not long-context quality or full-capacity memory qualification. Sustained soak/fuzzing and Linux/Windows runtime evidence remain release work.

G4 repeats the real planner/lifecycle smoke using the pinned 0.5B Q4_K_M file in `fixtures/qualification/g4-planner.json`; context/profile/concurrency plans, real two-session sharing, exact profile outputs and cleanup pass. `g4-benchmarks.json` separately records serial local measurements. At context 512, two-worker balanced samples outperformed four-worker performance samples on this M1; the default stays balanced and the profile name does not promise higher throughput. Cache conditions are uncontrolled.

## E1 embedding admission

The Rust embedding estimator reserves packed encoder activations, per-worker linear attention scratch, matrix tiles, tokenizer/input/output allowances and headroom. It reduces microbatch token capacity under pressure while preserving input limits and output dimensions. Read-only admission objects bind a reservation and concurrency slot to one model and input-size workload; execution consumes the held reservation and keeps only output memory after completion. Output memory is released with the result. E2 exposes this through `/api/v1/plans/estimate` with `task: embedding` and bounded `input_bytes` descriptors; unloaded metadata, ingress and serialization costs are included. Details and limits: [EMBEDDINGS.md](EMBEDDINGS.md).

E2 embedding requests reserve serialization before planning compute, so selected packed capacity sees every live stage. Explicit per-request `max_memory` supplies a capped additional reservation on the same global budget; packing can shrink while input limits/dimensions stay fixed. Embeddings release the shared FIFO compute permit between microbatches, retain their workspace/reservation and return through the fair scheduler. Generation can run between those microbatches. Per-task outstanding/active ceilings supplement global queue/CPU limits. See [EMBEDDING_API.md](EMBEDDING_API.md).

## Structured-output admission

`plan --constraints` and the estimate endpoint with `constraints:true` add a separate 32 MiB `constraint_reserved_bytes` per request, including profile headroom in the total. Concurrency multiplies private grammar cost while shared weights and KV formulas retain their established meaning. Embedding plans reject this generation-only option. The scheduler reserves grammar, payload, session and headroom atomically before queueing, then transfers grammar/session guards directly to execution. Standalone execution uses the same planner and splits the grammar guard from its held reservation.

Observed session-buffer counters also include actual private trie/mask/alias/schema and sampler workspace capacities. They remain distinct from reservations, allocator overhead, worker stacks, transport/library allocations and physical residency. See [CONSTRAINTS.md](CONSTRAINTS.md) for the bounded compiler/tokenizer profile and timing scope.

## S6 live CPU packing

Resumable generation and embedding microbatches share a FIFO compute gate. Each compatible token group reads matrix tiles once using existing admitted per-request workspaces; changing membership requires no batch-sized activation allocation or reservation gap. Independent sequence capacities and grammar reservations remain unchanged. No memory discount is applied merely because requests may share a batch. CPU remains the only qualified placement, with no automatic backend/model fallback. [BATCHING.md](BATCHING.md) documents generation residency versus compute/embedding limits and the scope of observed counters.

The native `multimodal::positions::ImagePositions` helper is now checked against ten independent image spans. Physical KV entries and rotary-position counts are separate: the 112×56 Qwen fixture contributes eight feature rows and advances four rotary positions. Full image workload estimates remain inadmissible until component costs, language KV and scheduling are integrated and qualified; this helper alone does not authorize execution.

The native vision plan reserves reusable encoder/merger scratch, row-wise attention, owned worker allowances and retained projected embeddings under the same budget as prepared pixels. It does not modify the HTTP planner or enable image request admission. Complete weights are read through buffered storage; reservations are distinct from OS file-cache residency. See [VISION.md](VISION.md).

## Offline Qwen request admission

The [offline image-to-text adapter](MULTIMODAL.md) computes actual processor/vision, decoder, physical KV, rotary span and output costs for its supported graph. It atomically reserves a complete request peak, splits reservations into stages and recycles image capacity between attachments. This is separate from the daemon planner contract: the HTTP image estimate remains inadmissible and its unimplemented costs remain unknown. Connecting the qualified graph, model-pair residency and image inputs to CAEP/scheduling is the next product integration gate.

## Product image requests

Registered Qwen2-VL pairs accept `task=image-to-text` and bounded oriented image dimensions through the planner endpoint. The estimate validates both component graphs from headers and reports conservative processor/encoder/decoder, transport and model metadata costs without loading weights. Exact token/image geometry and full capacity are atomically reserved before scheduling. Profile worker/concurrency policy is shared with text planning; the initial image request holds the CPU FIFO gate for one bounded request. [IMAGE_API.md](IMAGE_API.md) documents fields, conservative byte ceilings, cancellation and the remaining image-history/batching/backend limits.

## Managed image history admission

Image sessions reconstruct selected original-byte references under session/owner/global disk quotas. Exact prompt/feature geometry, full decoder capacity and the sequential image-stage peak are admitted before scheduling; metadata/history and transport reservations are counted separately. Explicit oldest-complete-turn exclusion can remove older image turns and records their IDs. Cooperative compute checkpoints retain the existing reservation and buffers while parked; output backpressure releases CPU ownership. Original assets remain on disk and KV/pixels/embeddings remain request-local. See [IMAGE_SESSIONS.md](IMAGE_SESSIONS.md) for finite limits, phase residency and persistence semantics.

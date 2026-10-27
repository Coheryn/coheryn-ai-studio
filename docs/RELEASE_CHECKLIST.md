# Release evidence and outstanding work

Source licensing records and distribution staging are in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). CI checks the locked all-target dependency notice inventory and file-level mirrored asset mapping. Every source/binary package must retain the complete legal payload; `python3 tools/license_notices.py --bundle target/distribution` stages it for packaging. Custom model terms and final platform/package/legal review remain applicable.

Status: experimental G4 text CPU CLI/API and E2 native embedding CPU CLI/API; M0 provenance groundwork, automated S0 checks and initial S1–S4 managed text sessions and tool proposals, isolated KV reuse and opt-in histories. Full v1.0 is not complete.

The mandatory release scope includes C01–C28 and C29–C38: sessions, persistence, tools, constraints, sampling, continuous batching and independence/presentation proof. S0 has scoped automated checks and S1 has initial text CPU implementation/evidence. S2/S3 now have initial current-host CPU implementation/evidence; S4 now has initial exact-file CPU implementation/evidence; S5–S6 and complete release qualification remain open. Native Nomic F16/Q8_0 CPU embedding execution is qualified on macOS ARM64; embedding CLI, `/v1/embeddings` and external RAG interfaces have initial E2 evidence. Multimodal extensions to C02/C03/C06/C08/C09/C12–C18/C22/C23 require M0–M2 and G5/G6 evidence, including component manifests, image-aware admission and secure acquisition; existing text-only checks do not satisfy those additions.

The G0/G1 engine, G2/G3 service/planner and G4 packed CPU execution have passed the macOS ARM64 checks recorded in `VALIDATION.md`, including real-model CLI/API/SSE parity. Full allocator accounting and broader platform/backend qualification remain outstanding.

| Requirement | Current evidence | Remaining release work |
|---|---|---|
| C01 native execution | Coheryn scalar Qwen2 graph and CPU kernels | Complete backend/format qualification |
| C02 CLI | Standalone/client run and chat; registry commands, serve, inspect/validate, plan preview, status, metrics, bench and doctor | Versioned configuration evolution and final CLI release qualification |
| C03 HTTP API | Text completion/chat, SSE, native generation and versioned admin routes | Broader compatibility/release security qualification |
| C04 streaming | Bounded CLI/SSE delivery, reserved terminal slot, consumer timeout, disconnect cancellation | Cross-platform release soak tests |
| C05 headless | Same Rust engine in standalone CLI and Axum daemon | Packaged offline release tests |
| C06 GGUF | Bounded v3 parser; F32/F16 and packed Qwen2 execution | Broader backend/platform qualification |
| C07 loading | Shared buffered weight source, checked ranges/header/mutation checks and SHA-256 on managed load | Broader hostile corpus and reviewed mappings if beneficial |
| C08 management | Managed registry/import/register/pull/inspect/validate/load/unload/remove; hash and quota checks | Packaging/migration/recovery hardening |
| C09 tokenizer/template | GGUF Qwen BPE and approved text-template fingerprints | Release corpus and documented supported variants |
| C10 quantized computation | Native Q8_0/Q4_0/Q5_0/Q4_K/Q6_K; independent block/activation/dot fixtures and pinned-model parity | ISA-specific optimization and complete platform/backend matrix |
| C11 CPU | Controlled 1–16 worker row parallelism; bounded chunks, reusable activations and fixed per-row reductions | Cross-platform qualification, batched prefill and calibrated performance work |
| C12 GPU | Not implemented | Mandatory Metal FP16/Q4_K_M qualification |
| C13 residency | Shared immutable loaded model; atomic held session reservations and observed engine I/O/KV/scratch capacities | Complete allocator/residency/transfer instrumentation |
| C14 CAEP | Deterministic CPU profiles, hardware probes, metadata-only CLI/API planning and atomic queue-to-execution admission | Vision stage qualification and calibrated complete allocation accounting |
| C15 KV/workspace | Bounded per-session F32 KV, scratch/I/O and linear-row attention with shared admission | Pooling/paged-KV optimization and calibrated accounting |
| C16 monitoring | Engine timings/reservations, scheduler counters, OS RAM/swap API, Unix benchmark counters | Private/GPU observability and calibrated planner feedback |
| C17 compatibility | Structured early graph/type/tokenizer/template failures | Full release support/negative matrix |
| C18 security | Role credentials, loopback default, body/header/queue quotas, managed IDs, atomic hash-checked acquisition, redirect/DNS restrictions | Sustained API fuzz/security and platform credential review |
| C19 platforms | Three-platform CPU CI configured | Actual Linux/Windows test evidence and release binaries |
| C20 offline | Standalone engine and local daemon need no cloud service or telemetry | Packaged offline smoke tests |
| C21 diagnostics | Structured HTTP logs/errors, explicit exits, private persistent registry/credentials, reproducible flags | Versioned configuration files and final contract qualification |
| C22 scheduling | Shared residency, configurable budget-admitted sessions (default one), exclusive low-memory execution, strict waiting queue, pin-aware unload and cancellation | Broader concurrent-serving optimization and soak tests |
| C23 verification | Engine/parser tests, independent oracle, actual HTTP/SSE integration, pinned-model service smoke, fuzz target and benchmarks | Sustained fuzz/sanitizer campaigns and complete backend/platform matrix |
| C24 image understanding (P0) | M0 pinned components, header inventories, processor/content contracts and image inputs; no image execution qualification | Pinned LLaVA and Qwen-VL model sets; native JPEG/PNG preprocessing, vision/projector/integration; CLI/API/SSE, multi-image, reference parity, CPU/Metal, memory and security evidence |
| C25 native embeddings (P0) | E1 independent Nomic F16/Q8_0 graph, 768-D mean/L2 vectors, exact tokenizer, single/batch/profile parity, 50 multilingual cases, bounded admission and cleanup; see EMBEDDINGS.md | Broader pretrained architecture/encoding qualification, Metal, actual Linux/Windows evidence and comparable memory/throughput benchmarks |
| C26 embeddings API (P0) | E2 strings/arrays, float/Base64, exact usage, identity headers, task-aware residency, bounded shared scheduling and SDK checks | Broader platform/backend, sustained security and release contract qualification |
| C27 embedding CLI (P0) | E2 headless text/file windows, native/client parity, quotas, atomic export/overwrite protection, identity-switch abort and cancellation | Cross-platform packaging and extended disk/transport fault campaigns |
| C28 RAG interoperability (P0) | E2 versioned space identity, standard SDK, LangChain raw-text embeddings and Qdrant local vector-store example; clients remain outside Cargo | Broader client/version/platform release matrix |

| C29 isolated managed sessions (P0) | S1 owned ephemeral histories, CLI/API lifecycle, revision/retry isolation, streaming and cancellation on macOS ARM64 | Broader model/platform and qualified role/image/tool combinations; persistent-store work remains C32 |
| C30 context management (P0) | S1 exact qualified text-template token counts, output reserve, protected instructions, reported whole-turn/incomplete-history exclusions | Qualified tool/image accounting and broader model/platform evidence |
| C31 session-safe KV reuse (P0) | S2 isolated prefix cache, nonconstant KV/logit parity, actual FP16 fresh/reused parity, identity/invalidation, TTL/LRU and reservation transfer | Broader backend/model/platform and fault/pressure campaigns |
| C32 persistent histories (P0 capability) | S3 opt-in SQLite store, bounded lazy pages, transactional history/replay, restart recovery, quotas, private paths, retention and CLI/API save/import/export | Broader crash/fuzz/security/platform campaigns and future schema migrations |
| C33 tool/function calling (P0) | S4 exact-file Qwen companion profile, native marker/parser path, choices, raw arguments, multiple calls, SDK/CLI/API/SSE, ordered results and pending restart/replay | Broader model/backend/platform and security qualification; strict schemas remain C34/S5 |
| C34 structured/constrained output (P0) | No qualified token-level schema decoder | S5 JSON mode, documented schema subset, bounded token masks and final/partial validation semantics |
| C35 full sampling controls (P0) | Existing greedy/seeded temperature/top-k/top-p and stops | S5 complete qualified penalties/context/batch controls, validation and versioned reproducibility scope |
| C36 continuous batching (P0) | Shared bounded scheduling and embedding microbatches | S6 compatible dynamic generation batches, fairness, per-request state and failure/cancellation isolation |
| C37 verified independence (P0) | S0 all-feature workspace/fuzz graphs, inert archived captures, source audit, empty-search-path native test and current-host executable checks | Full release provenance, legal/security review and clean cross-platform package qualification |
| C38 repository presentation (P0) | Zero findings in controlled source/path and current-host executable scans; resource catalog and CI gate; intact relocated assets resolve identified vocabulary/provenance collisions | Actual remote CI, final generated/shipped package/notice review and continued enforcement |

G0/G1 evidence lives in `fixtures/models.json`, `fixtures/golden/`, `fixtures/qualification/`, and the ordinary Rust tests. Large raw traces and models remain ignored. Qualification records must identify their exact hardware, inputs, settings and tolerances. Do not mark G0/G1 accepted solely because synthetic tests pass.

G3 CPU planner behavior and remaining M0 acceptance are described in [PLANNER.md](PLANNER.md). Actual VLM tensor payloads remain unverified; header inventories are not inference qualification. Existing text compatibility entries retain their exact-file evidence.

Full v1.0 under v1.7 requires C01–C38, G0–G6/M0–M2/E0–E3/S0–S6 evidence, including embedded persistent-history capability with opt-in runtime use, tool/schema and independent audit acceptance. Existing feature rows above describe their historical scope and do not qualify the newly extended contracts. Optional full RAG/storage/retrieval/reranking features do not gate the core release.

G4 exact-file qualification covers four 0.5B quantized publisher files and the second-size 1.5B FP16/Q4_K_M pair. Actual tensor inventories, frozen primary/second-size contracts, layer/logit and greedy results, quantized service/concurrency cleanup and serial local memory/throughput observations are linked from `fixtures/compatibility.json`. This closes the initial macOS ARM64 packed CPU implementation/qualification work; it does not satisfy GPU, native vision, embedding GPU or other-platform release requirements. E1 CPU library evidence is recorded separately in `e1-embeddings.json`, `e1-multilingual.json` and `e1-boundaries.json`.

E2 CPU transport/integration evidence is recorded in `fixtures/qualification/e2-service.json`; [EMBEDDING_API.md](EMBEDDING_API.md) documents the exact tested subset. No image, Metal, other-platform or comparative-performance requirement is marked complete by this gate.

S0–S3 evidence and exact supported contracts are documented in [INDEPENDENCE.md](INDEPENDENCE.md), [SESSIONS.md](SESSIONS.md) and `fixtures/qualification/s2-s3-verification.json`. Persistence is disabled by default; only owned explicitly saved conversations survive restart. KV is isolated, bounded and never persisted.

S4 current-host evidence and exact supported fields are in [TOOL_CALLING.md](TOOL_CALLING.md), `fixtures/tools/manifest.json` and `fixtures/qualification/s4-verification.json`. This qualifies a specific file/profile, preserves the original malformed-output rejection fixture and leaves broader release obligations open.

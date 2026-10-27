# Validation evidence

Current status is experimental G4 text CPU service and E2 native embedding CPU CLI/API and initial S1–S4 managed text-session, prefix-cache, persistent-history and exact-file tool qualification on macOS ARM64. The workspace passes 143 ordinary tests, with two opt-ins kept separate. Exact-file/model, service, planner and benchmark evidence is linked below; native vision, Metal, other-platform execution and release qualification remain open.

The [release checklist](RELEASE_CHECKLIST.md) tracks the complete target. S0 now has source/path, locked workspace/fuzz graph, intact archived-data and current-host executable-string evidence. S1 adds managed histories and exact context controls. S2/S3 add bounded isolated KV reuse and opt-in transactional storage, restart/replay, lazy pages, retention and import/export. S4 has initial exact-file CPU protocol/model evidence; S5–S6 and complete platform/security/license/package qualification remain open. Earlier sections retain their historical scope; archived capture source is inert and excluded from the native chain.

## S4 checkpoint

`fixtures/qualification/s4-verification.json` identifies the current source/build checks; `s4-service.json` records native exact-file prompt tokens, multiple-call greedy output, tool-result continuation, standard SDK streaming/usage, pending export/import, restart/replay without weights and cleanup. Independent expected data are in `fixtures/tools/qwen2.5-reference.json`. The original publisher example induced malformed double-braced JSON; its independent negative capture is preserved and rejected without repair. The explicit companion template corrects that example and preserves raw argument strings. All five frozen prompt byte/token sequences remain exact.

New regressions cover lossless large integers/escapes, bounded schema/arguments, special delimiters, multiple-call complete/SSE parity, nullable content, malformed/cancelled outputs, model checksum qualification, owner/replay isolation, whole tool groups, declaration retention, pending-write rollback, legacy migration and headless CLI workflows. The new tool fuzz target compiles alongside GGUF; 2,000 deterministic protocol mutations run in ordinary tests. The original eight-case tokenizer/template opt-in also passes. Runtime qualification uses an explicit 1,200-second generation deadline and uncontrolled OS cache conditions, with no comparative speed claim.

## S2/S3 checkpoint

`fixtures/qualification/s2-s3-verification.json` identifies the current sources, binary and checks. `s2-s3-service.json` records the pinned FP16 native-model run: unchanged independent greedy prefix, exact cached/stateless continuation, durable restart and replay without loaded weights, explicit cold-KV rebuild and reclamation to the store baseline. One-token model smoke supplements the preserved full S1 continuations; it does not replace earlier numeric/model qualification. Prefill observations have uncontrolled OS cache conditions and are not benchmark superiority claims. The final-binary default-deadline attempt correctly cancelled the longer restored FP16 turn after 300 seconds (HTTP 408); successful qualification uses an explicitly configured 600-second generation deadline.

New synthetic coverage includes nonzero attention/head KV/logit parity, settings/model/edit invalidation, cancellation, LRU/TTL, SQLite quota/failed-commit recovery, future/corrupt-store preservation, Unix path/privacy refusal, retention cleanup, history pages without full hydration and separate-process CLI export/import/restart/resume. Persistent stores keep no weights or KV; runtime opt-in is required. Current-host checks do not complete cross-platform, GPU, sustained crash/fuzz/security or packaging/license gates.

## S0/S1 checkpoint

`fixtures/qualification/s1-verification.json` records current checks and their scope. `s0-independence.json` inventories both locked graphs and 30 byte-verified archived originals; `s0-artifacts.json` records controlled UTF-8/UTF-16 strings in the current-host executables; `s0-dependency-versions.json` records upstream version checks. Source and executable scans have zero findings in their declared scopes. CI includes these gates; configured workflows do not prove remote jobs ran.

Native tests add multilingual session histories, credential isolation, revisions/idempotency, writer races, exact context policies, model changes, SSE/disconnect cleanup, quotas/expiry and deletion at a fully reserved workload budget. CLI tests exercise separate-process resume, explicit context conflicts, JSON receipts and reset/delete. A standalone native process runs with an empty executable search path and no service requirement. These are current-host checks, not a claim that the development machine has no cached external software.

`s1-service-full-greedy.json` retains the full eight-token pinned-model verification. `s1-service.json` identifies the additional S1-build smoke and its token count. Evidence includes original independent greedy prefixes, exact managed/stateless native token parity, streamed CLI resume, retry identity, zero reservations after deletion/unload and graceful shutdown. This is correctness/lifecycle evidence with uncontrolled cache/host conditions, not a comparative throughput claim. That historical S1 evidence does not qualify persistence or cross-turn KV reuse; those have separate S2/S3 evidence above. Image execution, Metal and other-platform model runs remain unqualified.


## G0/G1 checkpoint

The initial native CPU implementation has reference-parity evidence on an Apple M1 MacBook Air with 8 GB RAM. This is experimental G0/G1 evidence, not completion of v1.0 or qualification of other platforms/models/backends.

## Executed checks

- Thirty ordinary Rust tests: bounded GGUF parsing, malformed/truncated/count/offset/shape fixtures, 2000 deterministic parser mutations, scalar operations, sampling, budget arithmetic, synthetic whole-model execution, EOS, stop sequences, UTF-8, cancellation/sink cleanup, hostile templates and non-finite payload scans.
- Opt-in real-model tokenizer/template regression: passed against the committed independent goldens.
- Published FP16 file: all eight independent prompt cases passed exact token IDs, byte-exact chat templates and greedy continuations. The multi-turn case generates EOS and stops with `2+2 equals 4.`.
- Published FP16 file: 24 layer outputs and all 151936 logits after `Hello` passed unchanged tolerances. Final-logit maximum absolute difference was approximately `5e-8` against the separately executed scalar reference.
- Reproducibly derived F32 file: 25 layer/logit comparisons and four greedy tokens passed against the identical-file reference. This promotes FP16 values; it does not restore original pretrained F32 precision.
- CLI smoke checks: insufficient budget exits 5; context above GGUF's declared 8192 exits 2; streaming SIGINT exits 130 and releases the process/session. The recorded cancellation observation is one run, not a latency guarantee.
- Release workspace build, formatting and Clippy with warnings denied passed. The test-only libFuzzer target compiled. A sustained instrumented fuzz campaign has not run.
- CPU CI is configured for macOS ARM64, Linux x86-64 and Windows x86-64. Linux/Windows jobs and release binaries have not been executed or qualified in this session.

## Arithmetic and oracle controls

The reference source revision, archive checksum, tool settings and numerical tolerances are in `fixtures/models.json`. At the original checkpoint, the reference executable was built separately and never linked into the Rust product; that capture source is now an inert resource archive. Its build provenance records source/settings/executable hashes. On macOS ARM, native tuning, NEON and external reference engine are disabled so scalar functions avoid hardware-specific half accumulation and approximate vector nonlinearities.

The G1 scalar baseline uses F32 products, activations and output with wider scalar reductions. FP16 matrix inputs round to binary16. F32 accumulation optimizations must qualify against this baseline. Initial native ARM comparisons exposed precision-contract differences; numerical thresholds were not loosened to make them pass.

## Memory interpretation

The benchmark records cover context reservations of 128, 512 and 4096 with a short prompt. These prove admission/reservation behavior and demand allocation, not full 4096-token prefill/quality qualification. The longest independent corpus prompt is 49 tokens followed by eight generated tokens.

`peak_rss_bytes` is an observed process counter. It excludes OS-owned file-cache memory; buffered reads still use the OS cache. `reserved` and `peak_reserved` are conservative admission estimates, not allocator instrumentation. Zero `mapped_bytes` describes model-weight mappings only. Private memory, GPU and unsupported counters are null. Hashing warms the model cache, so all recorded runs label cache conditions uncontrolled. No total-memory advantage or optimized throughput claim is made.

## Artifacts

- `fixtures/compatibility.json`: exact-file/backend qualification boundaries.
- `fixtures/golden/`: independent prompt/token/greedy cases and numeric comparison summaries.
- `fixtures/qualification/`: tensor inventory, derived provenance, OS benchmark records and CLI smoke results.
- `.cache/oracle/` and `.cache/f32-oracle/`: reproducible large raw traces, intentionally ignored.

At this checkpoint, hardware-aware CAEP, quantized CPU, Metal and release work remained. Later sections record the completed CPU planner/quantization work; `RELEASE_CHECKLIST.md` tracks outstanding gates.

## G2 service validation

The G2 suite adds real loopback transport and in-process router checks to the existing engine coverage. At the G2 checkpoint, the ordinary workspace suite passed 49 tests on macOS; two opt-in tests cover the pinned model and outbound HTTPS separately. Service tests exercise authentication/roles, managed import/load/validate/unload/remove, exact standalone/native/OpenAI text parity, chat SSE deltas/finish/usage/DONE, body/header limits, hostile paths/fields, hashes/quotas/interrupted acquisition, bounded admission, model pins, disconnect/shutdown cleanup and slow consumers. A concurrent library test verifies shared immutable weights with independent sessions and combined memory reservations. CLI subprocess tests exercise interactive chat, integrated benchmarks and cancellation while waiting for input; the input-signal test is Unix-specific.

`python3 tools/check_service.py` passed with the pinned publisher FP16 model over an actual TCP daemon. CLI JSON, CLI SSE and OpenAI completions match independent greedy goldens; chat SSE matches the explicit-system golden and reports correct usage. Active unload returns 409, inference credentials cannot access administration, an explicit client memory cap is enforced, disconnected generation is cancelled, unload releases all reservations, and SIGINT shutdown exits successfully. A subsequent CLI request fails explicitly rather than starting standalone inference. Evidence is in `fixtures/qualification/g2-service.json`.

The optional HTTPS probe passed against the pinned publisher URL, following validated redirects and reading only the four-byte GGUF magic. Hash validation and atomic commit are exercised separately by the streaming acquisition tests. The private-directory and actual-socket tests require filesystem permissions and loopback capability; sandboxed runs need those capabilities enabled. The service's OpenAI compatibility is the documented text subset in `API.md`, not every field or modality in the upstream API.

## G3 CPU planner and M0 groundwork

At the G3 checkpoint, the ordinary workspace suite passed **63 tests** on macOS ARM64. Formatting, strict Clippy (`--all-targets -- -D warnings`) and the release workspace build pass. New coverage includes deterministic plans, checked byte/context arithmetic, model sharing, host pressure/unknown counters, unsupported vision stage costs, atomic budget contention and reservation transfer, smaller I/O tile output/capacity parity, configurable concurrent sessions, exclusive low-memory execution and queued-session overcommit rejection and strict waiting-queue bounds under exclusive profiles. Component/content tests check bounded ordered image descriptors, unsafe paths, duplicate roles and false verification claims. The two opt-in reference/network tests remain separate from this ordinary count.

`tools/check_planner.py` passed against the actual pinned Qwen2.5-0.5B FP16 model and a real loopback daemon. `fixtures/qualification/g3-planner.json` records 24 metadata-only plans across contexts 128/512/4096, concurrency one/two and four profiles. All four profiles reproduce the independent English eight-token greedy continuation exactly. Two live real-model requests share residency, remain inside reservations, and cancel/release buffers on disconnect; unload returns reservations to zero. The generation requests use a short prompt and reserve capacity; this does not qualify long-context behavior. Session buffer observations exclude allocator/library/transport overhead, and no general speed or OS-memory-cap claim is made.

M0 selected LLaVA 1.5 7B Q4_K_M with its F16 vision/projector and Qwen2-VL 2B F16 with its F16 vision/projector. The native parser inspected bounded prefixes and committed complete tensor inventories plus source processor/config/tokenizer identities in `fixtures/vlm/`. Whole-file sizes/hashes remain publisher-declared; full tensor payloads were not acquired/verified, and no VLM inference ran. Procedural PNG/JPEG inputs and a frozen comparison contract exist, but independent vision outputs, component/license review and multi-image qualification remain open. `fixtures/compatibility.json` adds provenance links without adding executable VLM entries. See `PLANNER.md` for remaining M0/M1/M2 work and the selected LLaVA dependency on G4.

Platform qualification remains macOS ARM64 only. Configured Linux/Windows CI, checked-in fuzz targets and proposed reference comparisons do not constitute completed platform, fuzzing or VLM qualification.

## G4 packed CPU and E0 reference preparation

At the G4 implementation checkpoint, the workspace suite passed **70 ordinary tests** on macOS ARM64; the two model/network opt-ins remain separate. Formatting and strict Clippy pass. New coverage includes 100 independently captured packed decoding/activation/dot cases across Q8_0/Q4_0/Q5_0/Q4_K/Q6_K, 12 RoPE dimension/position fixtures, malformed scales/extents, synthetic complete packed models with 1/2/4 workers, cancellation cleanup, linear-row attention and explicit API thread admission/stack reservations. The available-RAM check distinguishes additional engine allocations from advisory full-file page-cache residency.

The four pinned 0.5B publisher quantized variants pass all eight prompt cases and two-token teacher forcing (50 layer/logit comparisons per file). English and multi-turn outputs are also exact under low-memory/performance profiles. These comparisons use identical weights and the existing frozen model tolerances. Actual mixed-type inventories are committed; notably the 0.5B Q4_K_M file contains Q5_0 tensors. `g4-quantization-quality.json` separately reports FP16 versus quantized reference differences and does not change executor thresholds. See [QUANTIZATION.md](QUANTIZATION.md) for reproduction and exact-file evidence.

E0 pins and verifies complete Nomic Embed Text v1.5 F16 and Q8_0 files, source/tokenizer/pooling assets and the independent scalar encoder. Twelve single-input 768-dimensional unit vectors, encoder/token/pooling checkpoints and eight duplicate/mixed-length batch vectors are committed. Pooling reconstructs from real token rows, and single/batch vectors pass the frozen comparison. The offline evidence checker runs without downloads or a native embedding runtime. These remain independent reference inputs; E1/E2 now implement native CPU execution and CLI/API transport. Long-context/reduced-dimension behavior, GPU/other-platform qualification and external reference engine execution/benchmarks remain pending. [The E0 guide](../fixtures/embedding/README.md) distinguishes captured evidence from those remaining gates.

The final G4 service build passed actual quantized CLI/native/OpenAI/SSE checks in `fixtures/qualification/g4-service.json`. The quantized planner smoke in `g4-planner.json` passed 24 context/profile/concurrency estimates, all-profile exact outputs, real concurrent sessions and disconnect/unload cleanup. `g4-benchmarks.json` records fresh-process memory/timing observations for two iterations at each of 12 context/profile combinations; `QUANTIZATION.md` gives the local throughput observations and measurement limits. At that checkpoint, the release build, strict Clippy and all 70 tests passed after session teardown was ordered to clear buffer observations before releasing admission.

For second-size qualification, the manifest explicitly requires exact primary tokenizer/template fingerprints and all eight tokenizer/render cases, English/Unicode greedy comparisons and every layer/logit at two teacher-forced positions (58 comparisons per 28-layer model). This tests the changed graph dimensions while retaining the primary files' full eight-case greedy corpus. Existing numeric thresholds are unchanged. Completed independent cases now have hash-checked resumable captures; incomplete/partial runs cannot promote compatibility.

Both pinned 1.5B files now pass the second-size contract, including all eight tokenizer/render cases, two greedy cases, three-profile English parity and 58 layer/logit comparisons each. Their actual inventories are F16/F32 for FP16, and Q4_K/Q6_K/F32 for Q4_K_M. Session buffer observations remain below reservations and return to zero after generation. The compatibility matrix links these completed records rather than inferring support from family names.

`g4-float-regression.json` records the final-build FP16 regression: all eight committed prompt/token/greedy goldens remain unchanged, and 25 layer/logit comparisons pass. The current derived F32 file also passes 25 layer/logit comparisons and four greedy tokens. `g4-validation.json` summarizes the 70-test result, build/Clippy commands, tested executable, locked production dependency audit and evidence pointers; no external inference runtime occurs in the normal Cargo graph.

## Latest stable dependency refresh

The 2026-10-26 refresh upgrades the Rust toolchain to 1.99.1 and verifies the latest stable releases of all 21 directly managed Rust packages, development Python packages and CI actions. [DEPENDENCIES.md](DEPENDENCIES.md) documents the breaking API adaptations and unavoidable upstream transitive version constraints; [the qualification record](../fixtures/qualification/dependency-upgrade.json) records versions, lockfile hashes and validation.

All **71 ordinary tests**, strict Clippy, formatting, release compilation and fuzz compilation checks pass. Both opt-ins also pass: exact pinned tokenizer/template goldens and the public HTTPS header probe using the upgraded TLS backend. All six quantized/second-size contracts pass again against the retained, hash-verified independent references. Real CLI/API/SSE and concurrent planner lifecycle checks pass on the upgraded executable. The original FP16 eight-case goldens remain unchanged; FP16 and derived F32 each pass 25 layer/logit comparisons, with the derived F32 four-token continuation also exact. Embedding reference integrity checks pass without enabling native embeddings. No numerical threshold or expected output was relaxed for these updates.

The existing G4 throughput samples retain their recorded Rust 1.90.0 compiler and executable identity as historical measurements; no new performance advantage is claimed for the dependency refresh. Other CPU platforms, native vision/embedding execution, Metal and sustained fuzzing remain separate qualification work.

## E1 native CPU embeddings

The native `nomic-bert` graph, model-specific WordPiece tokenizer, mean pooling, L2 normalization and ordered packed batches execute independently of generation models. Shared matrix primitives tile input rows, reuse F16/Q8 operands and buffered weight tiles, and preserve the existing scalar reductions. Dedicated workers, reusable linear attention scratch, bounded input/token/batch limits, atomic immutable admissions and result-owned output reservations provide CPU execution and cleanup. [EMBEDDINGS.md](EMBEDDINGS.md) specifies the complete contract.

- `e1-embeddings.json`: both pinned F16/Q8_0 model hashes; six exact-tokenizer and encoder/checkpoint/pooling/vector cases each; five mixed/duplicate batch/profile/worker configurations per file; usage, dimensions, output reservations and cleanup.
- `e1-multilingual.json`: 50 independently captured multilingual/Unicode tokenizer and vector cases for each file. Original goldens and precision thresholds are unchanged. The model-specific first-decomposition/all-mark-stripping normalizer is explicitly documented; parity does not imply proficiency in every language or equivalence to the source Hugging Face tokenizer.
- `e1-boundaries.json`: actual 512-token reference vectors, two maximum-length sequences with isolated attention, explicit truncation, oversized token/byte/batch rejection and memory-exhaustion errors for both files.
- Ordinary Rust tests now total **84 passing**: new checks include independent LayerNorm/uniform-attention fixtures; F32/F16/Q8 single/batch/profile parity; 56 frozen tokenization cases using committed vocabulary without model downloads; reservation retention, failed handoffs, admission races, cancellation/observer cleanup and task-aware CLI/API deep validation.
- Formatting, strict workspace Clippy, release build and fuzz compilation pass. The GGUF fuzz target now also exercises decoder/encoder compatibility checks. The real primary FP16 text graph retains exact eight-token English continuation and all 25 layer/logit comparisons after sharing matrix operations.

Numerical probe timings have uncontrolled cache/host conditions and are not comparative throughput benchmarks. Engine vector capacities and reservation peaks are observed; RSS, private memory, page faults and I/O are not captured by these probes. E3 retains calibrated workload and external reference engine comparison benchmarks. Metal, Linux/Windows runtime evidence, image understanding, public embedding CLI/API and RAG interoperability remain open. Historical E0/upgrade records keep their capture-time status. New Unicode dependency verification and production-runtime separation are recorded in `e1-dependencies.json` and `e1-production-dependencies.json`.

## E2 CPU CLI/API and RAG interfaces

The workspace now passes **100 ordinary tests**, strict Clippy, formatting, release compilation and locked fuzz compilation. New coverage includes queue-safe resumable/capped engine admissions; API float/Base64 parity, dimensions/order/usage, validation and role/resource limits; shared-generation fairness and cancellation; response-byte reservation/pin ownership; and CLI bounded records, atomic publication/overwrite protection, quotas, truncation, Base64, benchmarking and model-switch aborts.

`fixtures/qualification/e2-service.json` records real execution of both pinned Nomic F16/Q8_0 files through standalone CLI, daemon client and HTTP. It compares six reference cases and all 50 multilingual/Unicode vectors per file, verifies exact usage/identity, exercises OpenAI Python 3.39.0 default/float/Base64 behavior and tests LangChain OpenAI 1.7.3 with Qdrant Client 1.20.0. Base64 and JSON are compared after float32 conversion, avoiding false binary64 equality assumptions. Optional clients remain outside Cargo and model execution remains independent Rust. Unload returns the daemon reservation/buffer counters to zero.

`e2-native-regression.json` retains independent encoder/checkpoint/pooling/profile comparisons after resumable execution changes. `e2-verification.json` links checks, executable identities and the current production dependency scan. `e2-dependencies.json` records latest stable release verification, including MiniJinja 3.0.4 and unchanged Rust 1.99.1. Expected vectors, tokenizer fixtures and precision thresholds remain unchanged. The primary text tokenizer/template fixtures and targeted FP16 layer/logit checks guard the dependency update.

This closes the initial macOS ARM64 CPU E2 implementation. It does not claim image understanding, Metal, actual Linux/Windows execution, sustained fuzzing, calibrated comparative throughput/physical-memory benchmarks, or proficiency in every language. See [EMBEDDING_API.md](EMBEDDING_API.md) for the exact supported contract and reproduction commands.

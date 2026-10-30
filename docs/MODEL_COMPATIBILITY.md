# Architecture-based model compatibility plan

Status: A0/A1 text registry and an initial managed checkpoint/stateless API slice implemented; broader gates remain planned. This document defines the A0–A5 acceptance gates; mandatory Safetensors single/indexed layouts add the [F0–F5 format gates](SAFETENSORS.md). The [initial implementation](ADAPTERS.md) selects existing Qwen2 GGUF and experimental Qwen3.5 Safetensors CPU text; it does not complete these gates. Existing exact-file evidence remains in `fixtures/compatibility.json`. Delivery follows dependencies and acceptance, without development duration estimates or deadlines.

The goal is automatic loading of GGUF and Safetensors models whose complete requirements are implemented, including compatible sizes and fine-tunes, without a filename or qualification-checksum whitelist for basic inference. Universal support for every current or future model is not a release promise. Container data and associated configuration do not implement the architecture's computation, tokenizer, task semantics or numeric encodings.

This work strengthens C06/C09/C17/C23 and related planner, lifecycle and independence requirements. It preserves C01–C38, all required GGUF text/vision/embedding combinations and the CPU/Metal release matrices. Single-file and indexed-sharded Safetensors are now mandatory through F0–F5, with their own exact minimum matrix; parsing either format does not implement additional architectures. General-purpose platform targets drive architecture decisions; the current development host is one qualification environment, not a product capacity limit.

## Resolution and execution boundary

```text
bounded GGUF OR single/indexed Safetensors + inert asset inspection
    -> architecture and task registry
    -> validated configuration + tensor bindings + tokenizer/processor
    -> required operations and per-tensor encodings
    -> backend/kernel capability validation
    -> immutable execution descriptor + separate qualification lookup
    -> operator/request policy + CAEP admission
    -> shared native execution, cancellation and bounded delivery
```

Start with engine modules. Do not introduce new crates, a graph interpreter, a compiler or a plugin ABI merely to move the current architecture dispatch. A registry selects Coheryn-owned implementations; registration does not execute model-supplied code.

| Module boundary | Required responsibility |
|---|---|
| Architecture registry | Stable architecture/profile versions, task capabilities, strict configuration parser, required tensor schema and native graph builder |
| Tensor encodings | Block layout, checked storage access, supported shapes, CPU/device kernels and bounded scalar fallback only where implemented and validated |
| Operators | Explicit semantics for normalization, activations, positions, attention, projections and task outputs; shared implementations where semantics match |
| Tokenizer/template registry | Qualified algorithms, bounded native rendering and explicit special-token/role/tool behavior; no arbitrary template execution |
| Processor/component registry | Ordered component roles, feature widths, processor rules, positions and asset identities for vision and embedding tasks |
| Format loader and storage | Independent bounded GGUF/Safetensors/index readers, explicit source assets and one owned tensor catalog with per-file ranges and integrity; no shard merge |
| Compatibility resolver | Source-specific metadata/tensor-name bindings into the same complete requirement check, bounded diagnostics and immutable descriptor used by planning and loading |
| Qualification index | Inert evidence keyed to exact files, semantics and execution configuration; never a substitute for structural validation |

Configuration derives supported dimensions from validated metadata. Required tensors, biases, tied weights, normalization order, attention masks/windows, positional rules and task heads remain explicit architecture semantics. Unknown graph-affecting metadata or tensors fail clearly. Declared inert provenance may be retained under parser limits; classification is versioned, never a blanket ignore of unknown fields.

Tensor names/shapes do not establish semantics by themselves. Do not infer an unknown graph, processor, tokenizer, missing projector, pooling policy or trained head from resemblance, a filename or a marketing family name. Metadata errors must not trigger expensive trial execution.

## Compatibility, evidence and admission are separate

| Report field | Target values and meaning |
|---|---|
| `container_valid` | Existing bounded format validity result |
| `structural_status` | `compatible`, `unsupported`, or `invalid`; compatible means the selected architecture/task/backend's full requirements pass validation |
| `qualification_status` | `qualified`, `unqualified`, or `not-applicable`; exact evidence is independent of structural acceptance |
| `compatibility_status` | User-facing summary: `qualified`, `structurally-compatible`, `unsupported`, or `invalid`; qualified requires both structural compatibility and matching accepted evidence |
| `admission_status` | `not-evaluated`, `admitted`, or `rejected`; resource or operator-policy rejection must not relabel a supported architecture as unsupported |

Bounded CLI/admin reports include a schema version, source format/layout and per-file/shard identity, configuration/binding profile, task, architecture/profile version, model-set and tokenizer/template/processor identities, actual tensor encodings, backend/kernel contract, limits, missing capabilities, evidence references and policy/admission reasons. A successful inspect does not imply executable compatibility. Qualification may be pending, stale or failed within `unqualified`; a known numerical failure blocks execution of the affected configuration rather than being treated as ordinary missing evidence.

Resolve task/backend capabilities individually. For example, CPU text support does not grant Metal, image, embedding, tool, schema, session-prefix reuse or continuous-batch support. Missing advanced capabilities must not remove otherwise compatible basic generation, but requests for those features fail before execution. Existing exact-model tool/template/processor restrictions stay intact until their own generalization is qualified.

Basic inference defaults to allowing structurally compatible, unqualified configurations with visible status. Operators may require `qualified-only`; a request may tighten policy but cannot loosen server policy. Such a denial uses a policy diagnostic, not an unsupported-architecture error. No recurring approval prompt or hidden compatibility fallback is part of this design. Invalid, unsupported and known-failing configurations remain blocked.

## Identity, provenance and qualification

Removing a qualification whitelist does not remove identity or integrity checks. Hash complete relevant model/component bytes for managed load, verify imported provenance and byte counts, retain mutation checks and bind immutable component sets. Local standalone identity uses the existing checked-storage contract; no qualification digest is required merely to be structurally compatible.

An execution identity includes selected format/file-set/layout and source-configuration/binding identities, model/component hashes, architecture/descriptor semantics, tokenizer/template/processor identities, positions, encodings, backend/kernel/precision/cache contracts and relevant runtime version. Batching, session reuse, restored histories and embedding-space identity continue using their full appropriate keys. Changed aliases, bindings or profiles cannot reuse old KV, vectors or evidence accidentally. Losslessly repacked layouts still have distinct artifact identities; equivalence does not automatically grant cache, batching or embedding-space identity reuse.

Exact qualification records identify model bytes, supported task/features, runtime/profile versions, backend/platform/kernel/precision, tested shape/context/settings and linked acceptance results. Lookup validates all applicable scope fields. Changing weights, tokenizer, graph semantics or backend does not transfer qualification; unchanged records remain historical. Validated dimensions and hard ceilings are published separately from the smaller set of numerically tested configurations.

## CLI/API target contracts

Extend existing `models inspect`, `models validate --json`, `plan`, import/load, `doctor` and admin inspection/validation routes. Preserve current routes and stable errors/exits; version the richer report and publish migration rules before changing existing fields. `executable` must remain a documented structural execution-support result rather than silently becoming a qualification or request-admission flag.

Target examples, not implemented commands:

```sh
coheryn models validate ./new-finetune.gguf --json
coheryn models import ./new-finetune.gguf --name local-finetune
coheryn plan local-finetune --context 4096 --json
coheryn run local-finetune --prompt 'Hello' --json
coheryn serve --compatibility-policy qualified-only
```

Standalone mode gains the same optional `--compatibility-policy qualified-only`; the default is `compatible`. A policy flag never bypasses parser, backend, integrity, resource or task-feature checks. Failed daemon requests never become standalone execution. An administrator configures the daemon policy; ordinary inference inputs cannot change it.

Keep OpenAI-compatible response shapes unchanged. Put complete reports in versioned native/admin interfaces and opt-in CLI JSON metadata; expose a bounded summary through model discovery without making standard clients understand internal operators. Remote validation/import still accepts managed references or bounded uploads, never arbitrary server paths. Inspection/validation cannot cause an unrequested asset fetch or execute repository code.

## A0–A5 implementation and acceptance

| Gate | Implementation work | Exit evidence |
|---|---|---|
| A0: contracts and inventory | Audit current hard-coded branches and exact-hash restrictions, separating graph validation, integrity and feature qualification. Freeze report/policy/schema contracts, operation/encoding capability inventory and migration fixtures. Select one concrete Gemma text variant and pin files, revisions, licenses, tensor/tokenizer/template inventory and independent references before A3 acceptance. | Reviewable missing-capability reports; no inference-accuracy claim from metadata; frozen numerical criteria and an explicit CPU target/encoding set for the selected Gemma variant. |
| A1: registry without behavior change | Introduce typed architecture/task/profile registration and immutable validated descriptors. Move Qwen2, Nomic and paired Qwen2-VL resolution behind it, preserving current feature restrictions and rejection behavior. Reuse current storage, scheduler and planner. | Existing supported/unsupported, tokenizer, layer/logit/vector, lifecycle and real product fixtures pass unchanged; no new file becomes qualified or executable merely through refactoring. |
| A2: parameterized compatible loading | Derive supported dimensions/bindings from metadata; separate per-tensor kernel support, tokenizer/processors and qualification lookup. Implement report statuses and operator policy. All inference entrypoints consume the validated descriptor. | A supported checkpoint and an independently pinned size/fine-tune variation load without registering a qualification hash; renamed identical bytes behave identically. Mutated graph metadata and unsupported encodings still fail before execution allocation. Missing evidence stays visibly unqualified. |
| A3: first new text-family adapter | Implement the selected Gemma variant's actual tokenizer/framing, attention, normalization, position, weight/output and task semantics using native operations. Select the variant in A0; do not treat every Gemma generation as one graph. Keep the required LLaVA work and its reference contracts. | Selected Gemma CPU text profile passes frozen prompt/token/layer/logit/greedy tests, required reference-precision and selected existing packed-encoding comparisons, CLI/API/SSE, memory/cancellation and negative configuration checks. A second size/fine-tune variation tests parameterization without an exact-file execution whitelist. |
| A4: shared integration and isolation | Integrate policy/status with CLI/API/discovery, planner, managed registry, sessions/cache/persistence, batching and embedding-space contracts. Version reports and configuration; handle stale/failed evidence. | Standalone/client/API status and policy parity; no authorization bypass, stale-evidence promotion, component mismatch, identity switch or cross-request cache/batch leakage. Infeasible compatible requests report resource rejection. |
| A5: qualification and release audit | Publish versioned capability/evidence matrices and tested limits; fuzz resolver/metadata/template/descriptor boundaries; qualify mandatory CPU/platform and existing Metal combinations; measure dispatch/planning/execution overhead. | C01–C38 retain their original obligations plus A0–A5 evidence. New Gemma CPU support is tested on all three required CPU platforms. Gemma GPU support is a separate explicitly qualified extension; existing mandatory text/VLM/embedding Metal targets are unchanged. No universal GGUF or performance claim. |

A0 precedes A1. A2 depends on A1 and existing kernel/tokenizer checks. A3 depends on A0's frozen oracle, A2 and relevant native operations; implement the existing mandatory LLaVA adapter after the registry baseline and before the new Gemma adapter. Its work can exercise registry/component reuse without delaying A0/A1. A4 follows A2 and is rerun for A3. A5 depends on A0–A4 and the retained backend/platform gates. F0 uses A0 contracts; F1/F2 develop bounded format/storage support and F3 binds it to A1/A2 (and A3 for Gemma). G6 requires both A5 and F5. Existing M0–M4 and S/E/G evidence keeps its historical scope and is not rewritten by this plan.

Gemma's initial commitment is one explicitly selected CPU text architecture/profile and bounded variants, not universal Gemma, vision/audio, every quantization or GPU support. Additional architectures enter the same registry through reviewed implementations and qualification. No requirement is satisfied by adding an architecture name alone.

## Validation and resource acceptance

- Freeze expected outputs and tolerances before accepting a new adapter or changed kernel. Compare identical weights/precision/settings; measure quantization quality separately. Keep original references intact.
- Test renamed files, compatible size/fine-tune variants, missing/extra tensors, tied/untied weights, required biases, contradictory dimensions, positional/mask changes, unknown graph fields, mixed encodings and missing tokenizer/processor/component assets.
- Test qualification lookup against changed digests/profile/runtime/backend/context ranges, stale records, known failures and forged declarations. Test stricter operator policy independently of compatibility and memory admission.
- Validate graph/descriptor construction under finite tensor/node/edge/depth/shape/expert/tokenizer/template/diagnostic limits and checked arithmetic. Limits are versioned and intersect operator/model/backend ceilings; no unlimited mode.
- Preserve metadata-only planning before bulk weight load, then full identity checks and immutable descriptor verification at load. Reserve descriptor/compiler scratch, model storage, workers, KV/encoder workspaces and outputs before their allocation; never invent an estimate for an unsupported operation.
- Preserve bounded cancellation units, acknowledged cleanup and phase reservations across all adapters. Batch only complete compatible execution identities; a shared architecture label is insufficient.
- Run frozen text/embedding/vision and CLI/API/session regressions at A1/A2/A4, plus reference/profile/platform tests for the new adapter. Property/fuzz, allocation failure, changed-file, hostile-template and cache/persistence tests precede release claims.
- Benchmark metadata resolution and steady-state execution separately, with device/settings/cache conditions recorded. Resolve descriptors once per immutable load and avoid repeated graph discovery in token loops. Use direct native dispatch first; specialized execution must pass the same numerical contracts.

## Deferred extensions

Declarative graph recipes may later compose already implemented, versioned operations after static validation and bounded shape/resource planning. They cannot create a new operation or prove correctness from tensor names. Untrusted weights and metadata never supply executable code, native libraries, shell commands, network-loaded processors or arbitrary model hooks. Runtime code plugins, automatic third-party executor fallback, a general graph compiler and universal unknown-architecture execution are not A0–A5 requirements.

Single-file and indexed-sharded Safetensors feed the same validated descriptor and runtime through the mandatory F0–F5 bounded loaders and source-asset bindings. Those requirements are in [SAFETENSORS.md](SAFETENSORS.md); they do not implement unknown architectures or replace mandatory GGUF support. Arbitrary distributed training layouts, packed schemes and a general graph executor remain separate qualification work.

## Shared local metadata implementation

The registry now provides immutable source/configuration snapshots, a versioned metadata/capability report and local text cost estimation through [MODEL_METADATA.md](MODEL_METADATA.md). This extends the initial standalone execution registry without completing A0–A5. Full weight identity, managed multi-file lifecycle, source-aware execution admission, API/session integration and complete architecture/backend/platform qualification remain open.

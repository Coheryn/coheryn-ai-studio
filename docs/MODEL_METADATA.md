# Shared model metadata and local planning

The engine's `catalog::Inspection` owns a read-only source snapshot, a versioned model descriptor and a private validated planning configuration. Registered text adapters provide metadata bindings and cost estimators through `TextAdapter::describe`. Inspection does not construct a model graph, tokenizer or generation session. GGUF headers and selected Safetensors headers/index/configuration/tokenizer/template assets are inspected; tensor payloads are not scanned or decoded during ordinary inspection or planning.

```sh
target/release/coheryn models inspect .cache/qwen3.5-0.8b-source --json
target/release/coheryn models validate .cache/qwen3.5-0.8b-source --json
target/release/coheryn plan .cache/qwen3.5-0.8b-source --standalone \
  --context 256 --profile reference --threads 1 --max-memory 805306368 --json
```

These local commands need no daemon, credentials, network or inference execution. Directory selection follows the bounded source reader's canonical entrypoint rules; explicit single files and canonical indexes are also accepted. Managed IDs keep their existing daemon behavior. Local planning requires explicit `--standalone`, and never falls back after a daemon failure.

## Descriptor contract

| Field | Meaning |
|---|---|
| `schema_version` | Native report version, initially 1 |
| `format`, `layout`, `architecture` | Inspected container/layout and declared architecture |
| `container_valid`, `structural_status`, `executable` | Container validity and basic execution compatibility for the selected native profile; unsupported sources remain inspectable |
| `task`, `text_adapter`, `configuration` | Known task, compiled text profile/capabilities and validated configuration summary |
| `declared_context`, `runtime_context_limit` | Source maximum versus implemented profile limit; requests are never silently reduced |
| `tensor_count`, `tensor_bytes`, `encodings` | Whole selected-container inventory, including any explicitly inactive components |
| `source_files` | Bounded file roles, byte lengths, metadata digests and each digest's scope |
| `metadata_identity`, `identity_scope` | Deterministic header/index/selected-asset/profile fingerprint, independent of the root directory |
| `weight_payloads`, `qualification_status` | Payloads are unscanned and full weight identities unverified during metadata-only inspection |
| `admission_status`, `diagnostics` | Inspection itself holds no execution reservation; compatibility failures remain separate from later admission |

The metadata fingerprint excludes tensor payload bytes. It is **not a complete model identity or a cache/session/batch reuse key**. Different weights can have identical headers. A structurally compatible report does not establish exact-file numerical qualification or authorize features beyond its adapter profile. Existing provenance, full integrity and feature-specific restrictions remain required for managed execution.

For GGUF, local `models inspect` preserves existing top-level fields and adds `model_descriptor`. Safetensors inspection returns the native descriptor directly. Validation retains the relevant report fields and adds `validation_depth` and `payload_scan_completed`; failures use the existing structured errors and CLI exit codes. A failed deep check sets the validation result's `executable` to false. Descriptor fields such as `weight_payloads` describe the metadata snapshot; `payload_scan_completed` separately records a successful explicit deep scan.

`models validate SOURCE --deep --json` explicitly scans tensor values with bounded buffers and cancellation. GGUF retains its existing task-specific tokenizer/template checks; the initial Qwen3.5 source profile verifies the exact tokenizer/template hashes during metadata inspection. Deep finiteness validation does not compute full model hashes or establish executor correctness.

## Planning and scope

Local text plans return `mode: standalone-metadata`, the descriptor and a standard execution plan. Existing GGUF cost calculations are preserved. Qwen3.5 uses the same checked session arithmetic as generation, including full-attention KV, fixed recurrent/convolution state, scratch and worker allowances. Its resident metadata/tokenizer estimate remains 256 MiB; observed library allocation/RSS is separate. Its development profile uses a fixed 256 KiB I/O tile and sequential prefill, including under low-memory planning.

Plans report engine reservations separately from advisory file-cache working sets. Inactive source components remain included in the conservative whole-file working-set upper bound. Concurrency projects independent session costs subject to the selected profile and CPU share; it does not enable batching or a scheduler. `--max-memory` is a candidate execution reservation budget for this estimate, not an allocation cap for the inspection command or an OS memory limit. Reader metadata limits remain finite and separate.

An inadmissible estimate is printed with rejection reasons and exits with resource status 5. Context/thread/concurrency errors fail explicitly. The planner holds no reservation and cannot guarantee that a later execution will still fit; actual admission must remain atomic in the runtime.

Qwen3.5 standalone run/chat now use the same estimator as metadata planning. Execution atomically reserves the complete additional request, including profile headroom, before tokenization and session allocation. The guard remains held until session buffers and CPU workers retire; cancellation and consumer failures release it. Concurrent admission cannot oversubscribe the process-local engine budget. The explicit-thread library diagnostic generation entrypoint retains nominal reference-budget behavior. Shared managed-model budgets and stateless Qwen3.5 scheduling are documented in [MANAGED_CHECKPOINTS.md](MANAGED_CHECKPOINTS.md).

Generic local planning currently covers Qwen2 GGUF and the Qwen3.5 Safetensors CPU text profile. Nomic GGUF inspection/validation preserves its independent embedding task and context bounds; managed embedding/image planning retains the existing APIs. Managed Safetensors source-set import/publication, shared budgeting and stateless text APIs now extend this checkpoint. Unified task residents, managed checkpoint histories, complete source/backend qualification and broader architecture/task descriptors remain subsequent work. See [adapter boundaries](ADAPTERS.md) and [format gates](SAFETENSORS.md).

The real-model admission opt-in runs against separately acquired inputs using optimized test code:

```sh
./tools/cargo test --release -p coheryn-engine --lib --locked \
  profiled_admission_cleans_up -- --ignored
python3 tools/check_model_metadata.py --admission \
  --report fixtures/qualification/a2-admission-runtime.json
```

These checks compare reference/balanced plan reservations with runtime peaks, reject execution when profile headroom is missing, and verify reclamation after consumer failure, in-compute cancellation and model drop. Ordinary CI exercises pure costs, competing admissions and budget ownership without downloading weights.

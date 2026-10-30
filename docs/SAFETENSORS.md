# Native Safetensors and sharded checkpoint plan

Status: bounded reader and Qwen3.5 standalone/managed stateless CPU text implementation; complete F0–F5 qualification remains open and mandatory alongside GGUF for the amended production release. Existing qualification records retain their original scope. The architecture-based resolver is specified in [MODEL_COMPATIBILITY.md](MODEL_COMPATIBILITY.md).

Single-file and sharded Safetensors are two storage layouts for one tensor format. Sharding does not quantize weights, lower total parameter storage, select GPUs or prove reduced execution memory. The implementation must accept both qualified layouts as one logical model, without requiring a full-file merge or a GGUF conversion. Native GGUF and its complete encoding/model/backend obligations remain mandatory.

## Initial format and numeric scope

| Input | Mandatory target | Boundary |
|---|---|---|
| Single Safetensors file | Checked header, tensor directory and bounded read-only data access | A bare weight file still needs an explicit bounded asset/configuration set before inference |
| Indexed Safetensors checkpoint | `model.safetensors.index.json` with `weight_map` and referenced whole-tensor shard files | Supported inventory is selected explicitly; do not load every file found in a directory |
| F32, F16 and BF16 weight tensors | CPU reads/operations and independent numeric fixtures | BF16 support is explicit, not reinterpretation as F16; supported device precision is separately qualified |
| Config/tokenizer/template/processor assets | Bounded inert data interpreted by implemented native profiles | Weight format does not supply the architecture graph or permit repository code execution |
| Packed/FP8/FP4/custom quantization | Explicitly unsupported until its complete scheme and kernels are implemented and qualified | A standard tensor dtype can store a packed scheme requiring extra scales, groups and operations; the suffix alone proves nothing |
| Tensor/expert/pipeline-parallel training shards | Unsupported unless an explicit reconstruction profile is implemented and qualified | Initial indexed support maps whole named tensors to files; it does not concatenate arbitrary fragments or infer distributed topology |

F32/F16/BF16 are initial dense-weight storage types. Integer/bool tensors needed for a selected architecture require explicit non-weight semantics and fixtures; the parser may describe bounded valid types that execution rejects. Valid scalar/empty tensors are distinguished from unsupported graph shapes. A configuration requesting an unimplemented operation, architecture, tokenizer, dtype or processor receives an explicit compatibility error before execution allocation.

## Shared native loading boundary

```text
explicit local checkpoint root / managed immutable model set
    -> GGUF loader OR single/indexed Safetensors loader
    -> bounded tensor catalog + source configuration/assets
    -> native architecture profile + versioned tensor-name/layout bindings
    -> one immutable validated execution descriptor
    -> qualification / operator policy / CAEP admission
    -> shared CPU/Metal runtime and task-specific outputs
```

Implement engine modules first: format detection, Safetensors header parsing, indexed-set resolution, source configuration/assets, tensor bindings and model-set storage. A bounded Rust format/serialization library may be reused after dependency/license review; no general-purpose inference engine is introduced. Parsing and storage are separate from model execution. Keep source tensor names and bytes intact; adapters map them into semantic slots without rewriting artifacts or pretending they have GGUF metadata.

Each tensor view identifies its owner, file/component, checked byte range, shape, dtype and permitted layout. Read bounded tiles directly from the owning file or reviewed immutable mapping. Shards remain separate files behind one catalog; no persistent full-model copy, full dequantization or mandatory shard-sized buffer. Bound open handles, headers, metadata, catalog entries, tokenizer/processor state and I/O tiles. A handle limit must not reopen an unchecked mutable pathname: retain verified ownership/identity or fail safely on change. Mapping safety, buffered-access mutation checks and storage lifetimes apply to every shard.

## Header and tensor validation

Validate the little-endian header length before allocation; bound UTF-8 JSON bytes/depth/key counts and reject duplicate keys. Validate tensor names, dtype, rank/dimensions, checked element/byte arithmetic, offsets relative to the tensor data region, nonoverlap and complete buffer coverage under the pinned format contract. Reject truncation, invalid ranges, holes and unsupported sub-byte alignment; never round a malformed range into a valid one. Treat `__metadata__` as bounded string-to-string data, not arbitrary executable/configuration objects. Tensor payloads can contain non-finite values despite a valid container; apply the selected operation/model numerical policy separately.

Bound each header and aggregate header/index/configuration work before reading the full set. Format validity and executable compatibility are distinct. Do not call an otherwise valid file executable merely because every range parses.

## Shard-set resolution and filesystem safety

The initial supported package shape is:

```text
checkpoint/
  config.json
  tokenizer.json
  tokenizer_config.json
  model.safetensors.index.json
  model-00001-of-00002.safetensors
  model-00002-of-00002.safetensors
```

Exact required tokenizer/template/processor files vary by native profile. A versioned component manifest can select a supported index or single file with its associated assets. Multiple competing indexes/weight sets require an explicit selection; lexicographic order, filename counts or directory size never decide the model. Do not guess a missing shard from its numeric suffix or concatenate files blindly.

Parse `weight_map` with finite tensor/shard/path/index/aggregate-byte limits and duplicate-key rejection. Every mapped tensor must exist in the named shard with the expected shape/dtype/range; no cross-shard duplicate tensor, missing reference or unexplained payload is silently accepted. Extra tensors within selected shards require an explicit versioned architecture classification. Unselected directory files are ignored, never loaded or used to fill gaps. `metadata.total_size`, when present, is checked against the declared layout's validated tensor bytes; it is not trusted for admission. Freeze exact schema rules before F2 acceptance.

Initial shard references are portable basename-only relative names, contained in the selected root. Reject absolute/drive/UNC paths, separators, `.`/`..`, NUL, URLs, reserved platform names, case/normalization collisions, symlinks, nonregular files and extra hardlinks under the private managed-store policy. Bind validation to owned file handles and identity checks, preventing replacement and check/open races; use platform-specific safe path primitives and fail explicitly where the contract cannot be maintained. Nested layouts need a separately reviewed containment profile rather than an implicit path-policy relaxation.

Managed acquisition uses an explicit file inventory with per-file bytes/hashes, revision, licenses and asset roles. Download into private quota-controlled staging, verify the complete inventory, synchronize files/manifest, then atomically publish one registry identity. Missing/corrupt shards, cancellation, disk exhaustion or interrupted promotion leave no loadable partial model. Startup recovery is bounded; remove/unload respects shared references and queued/active pins for all shards/assets. No inference-time network fetch, glob-selected substitute or external processor code is permitted. Local imports keep the same path, integrity and mutation guarantees.

## Identity, architecture and CAEP integration

Record source-format/layout, full per-file hashes/bytes, index/configuration/tokenizer/template/processor identities, source-to-semantic tensor mapping version and native architecture/backend contract. Deterministic model-set identity covers the exact selected file inventory and semantic bindings; arbitrary directory enumeration cannot change it. Different shard layouts have different artifact identities, even when every logical tensor byte is identical. Cross-layout or cross-format numeric equivalence requires explicit evidence and does not automatically authorize cache, batching or embedding-space identity reuse.

The loader feeds the A1/A2 registry rather than introducing another model executor. Validate tied/untied embeddings, required/absent biases, matrix orientation, fused/split QKV, normalization, masks/positions, pooling/projection and output semantics explicitly for each selected profile. Use bounded view/stride or tiled transforms where supported; any retained transformed weight copy is planned/accounted and justified. Do not silently transpose, fill missing tensors, discard graph-affecting configuration or synthesize a processor from a family name.

Shared CAEP admission covers header/index/configuration/catalog bytes, handle/mapping ownership, tokenizer/processor buffers, numeric transforms, workers, KV/recurrent/encoder scratch and outputs. Metadata-only plans precede bulk weight load; full hashing is reported separately as load/verification I/O. Unsupported graph costs remain unknown and inadmissible. Preserve bounded compute/cancellation and worker acknowledgement during hashing, load, execution, unload and failed-later-shard cleanup. Report estimated reservations, observed allocations, mappings and physical residency separately; sharding alone is not a memory optimization claim.

## CLI/API and model management targets

Existing inspect/validate/import/plan/run/chat/embed and managed API routes accept qualified checkpoint roots, explicit index files or single files with an explicit associated asset set. Example target commands:

```sh
coheryn models inspect ./checkpoint/model.safetensors.index.json --json
coheryn models validate ./checkpoint --json
coheryn models import ./checkpoint --name publisher-model
coheryn run publisher-model --prompt 'Hello' --json
coheryn run ./checkpoint --standalone --prompt 'Hello' --json
coheryn embed ./embedding-checkpoint --standalone --input 'Example text' --json
```

These commands describe future contracts, not current support. F4 freezes unambiguous input selection and any asset-manifest option without breaking existing GGUF automation. A bare file missing required configuration produces an actionable missing-asset error. Managed IDs use the daemon; explicit standalone mode uses the same loader/runtime without starting HTTP or falling back after a failed daemon request.

Remote validation/import accepts bounded managed references or explicit staged multi-file uploads/manifests, never arbitrary server filesystem paths, inferred remote directory listings or unrestricted archive extraction. Each part and the aggregate set have finite byte/count/disk limits, authorization and integrity checks; atomic finalize publishes only a complete set. A privileged `pull` operation pins repository/revision and explicit inventory under existing network policy; inference never fetches missing assets. F4 publishes the multipart/staging/finalize schema and cleanup semantics before acceptance and preserves existing single-GGUF requests.

Report container format, layout, selected file/shard count, actual numeric/quantization scheme, architecture/task, missing components, bounded limits, integrity status and separate structural/qualification/policy/admission states through versioned native/admin interfaces. Keep standard generation/embedding response shapes unchanged. File extensions, format tags and `trust_remote_code`/`auto_map` declarations never authorize executable code. Preserve stable CLI exits and structured API diagnostics for malformed containers/indexes, unsafe references, missing/changed shards, unsupported types/architecture/assets and resource exhaustion.

## F0–F5 tasks and acceptance

| Gate | Implementation | Acceptance evidence |
|---|---|---|
| F0: contracts and provenance | Freeze format/index/asset schemas, finite profiles, error/migration rules, dtype/quantization scope and qualification manifests. Pin publisher checkpoints from already selected text, embedding and vision architectures plus the selected Gemma text profile. | Exact files/revisions/licenses/hashes/tensor/asset inventories and independent tokenizer/numerical references. Separate original artifacts from lossless repacks and any numeric conversion. No provider-wide compatibility claim. |
| F1: single-file reader and numeric types | Bounded Safetensors header/range/catalog parsing, owned read-only views and explicit F32/F16/BF16 reads/operations. | Independent scalar/empty/boundary/dtype fixtures; malformed/duplicate/truncated/overflow/nonoverlap/coverage tests and allocation-failure/cancellation cleanup. BF16 accuracy passes frozen criteria. |
| F2: indexed shard sets | Checked index-to-tensor resolution, contained file access, deterministic set identity and bounded handle/I/O management without merging weights. | Single-file and losslessly sharded copies expose identical named tensor bytes; missing/duplicate/extra/changed shards, path/link/race/case collisions and interrupted import fail safely. Metadata-only inspection remains bounded. |
| F3: shared architecture bindings | Native source configuration/tokenizer/template/processor interpretation and tensor mappings into A1/A2 descriptors for selected text/vision/embedding profiles. | Independent prompt/token/layer/logit/vector/vision-output evidence for the actual source weights; same-weight single/shard execution parity. Existing GGUF regressions and feature-specific restrictions remain unchanged. |
| F4: transport and lifecycle | CLI/API/registry import/pull/validate/load/unload/plan, private atomic multi-file staging, policy/status reports and session/cache/batch identities. | Standalone/client/API/SSE or embedding parity, offline operation, authorization, interruption/restart, cancellation, pin-aware deletion and memory/disk cleanup. No arbitrary remote paths or inference-time downloads. |
| F5: release qualification | Required CPU platforms, selected Metal dense-precision combinations, source/numeric equivalence checks, fuzz/fault/security tests, packaging and memory/throughput observations. | Minimum matrix below passes; A/G/M/E/S obligations retained. Publish exact unsupported architectures, layouts, quantizations and device combinations. |

F0 follows A0's contract boundary; provenance/fixtures can be prepared without an executor. F1/F2 require existing parser/storage safety and G3 budgeting. F3 requires F1/F2 and A1/A2; Gemma bindings additionally require A3. Retained LLaVA execution work remains required and may share these interfaces. F4 follows F3 plus existing CLI/API/registry boundaries. F5 requires F0–F4 and the corresponding CPU/Metal/platform gates. G6 requires F5 as well as all earlier gates. No duration estimates or development dates are introduced.

## Minimum qualification matrix

| Target | Mandatory coverage |
|---|---|
| Supported Qwen2.5 text checkpoint and the selected Gemma text profile | Pin native publisher assets/precision; execute both single-file and indexed-sharded layouts on macOS ARM64, Linux x86-64 and Windows x86-64 CPU. Lossless repacks may provide a missing layout; do not call them publisher originals. |
| Supported independent Nomic embedding checkpoint | Both layouts on all three CPU platforms; exact tokenizer/mask/pooling/normalization/dimensions, single/batch vectors and task independence. No generation weights required. |
| Qualified Qwen2-VL image profile | Complete Safetensors language/vision/projector/processor binding; both layouts on all three CPU platforms; image-conditioned reference/CLI/API/SSE, multi-image where permitted and cleanup. Preserve the separate LLaVA GGUF/Metal requirements. |
| Dense numeric operations | Independent F32/F16/BF16 CPU fixtures plus full-model source-precision checks from inspected inventories. Do not claim BF16-model execution solely from scalar fixtures. |
| Initial Safetensors Metal subset | Selected supported Qwen2.5 text, Qwen2-VL vision and Nomic embedding profiles in F16, both layouts. If the publisher weights differ in dtype, use an explicitly derived, versioned F16 artifact with its own accuracy evidence; preserve original files and identity. BF16 device execution is unqualified until separately implemented/tested. |
| Retained GGUF matrix | Every existing mandatory encoding/model/backend/platform combination, independent of Safetensors acceptance. GGUF packed quantization is not automatically supported inside Safetensors. |

Freeze exact model, dtype, layout/transform, source-to-tensor mapping, reference precision, backend/kernel/platform, context/batch/image limits and tolerances before acceptance. Compare logical tensor inventories and bytes for lossless repacks. For cross-format comparisons, require proof of identical tensor values and graph/tokenizer/processor semantics; separate conversion or quantization quality from executor correctness. Different GGUF conversions, tokenizer recipes or cache policies cannot stand in for an identical-weight comparison.

Extend native CI with small authored format/shard fixtures and resolver tests. Independent expected outputs remain inert inputs outside production dependencies. Add fuzzing for header/index/configuration/tokenizer/template/component selection; cover oversized counts, arbitrary names, non-finite tensors, unsafe paths, missing shards, alias changes, resource exhaustion, slow clients and cancelled/failed publication. Test nonstandard distributed checkpoints as explicit unsupported inputs. Report actual load/hash/open-file/I/O costs, engine allocation peaks, RSS/private/mapped/device memory where observable, and cold/warm conditions only when controlled.

## Non-goals and references

This format amendment does not automatically support newer Qwen, every Gemma generation, Kimi, MiMo, DeepSeek or other unimplemented architectures. Safetensors is a tensor container, not an executable graph. Universal provider compatibility, arbitrary distributed-checkpoint reconstruction, a generic ONNX executor, runtime repository code and new packed schemes remain separate work. Native GGUF stays first-class.

Normative format behavior and shard-index conventions are documented in the [Safetensors format specification](https://github.com/safetensors/safetensors/blob/main/README.md#format) and [checkpoint sharding documentation](https://huggingface.co/docs/transformers/main/en/big_models#sharded-checkpoints). F0 pins the precise specification/library/reference revisions used for implementation; documentation links alone do not constitute qualification.

## Initial development implementation

The bounded `safetensors::TensorSet` reader and [Qwen3.5 CPU text adapter](QWEN35.md) implement the first concrete single/indexed path with F32/F16/BF16 storage. The selected Qwen3.5 execution profile permits F32/BF16 weights. Per-shard handles/ranges, duplicate-key and coverage validation, portable ASCII shard names, owned bounded reads and mutation checks are implemented. Full platform-safe handle-relative containment, managed publication/import, shared format-independent descriptors/plans and the entire F5 matrix remain open. This implementation is not a completed F0–F5 claim.

The [managed checkpoint implementation](MANAGED_CHECKPOINTS.md) adds bounded manifest-first streaming import, complete payload identities, private atomic source-set publication, shared daemon budgeting/planning and stateless CPU text/chat/SSE. This advances source-set lifecycle work without completing real multi-shard, remaining architecture/dtype/backend/platform or F0–F5 release qualification.

# Pinned real-model CPU qualification

`tools/run_cpu_qualification.py` executes independent numerical fixtures, extracted-candidate CLI inference and authenticated loopback TCP/SSE checks on the actual native host. Current expansion targets [modern Qwen3.5 model sets](MODERN_MODELS.md); earlier text and embedding fixtures remain regression coverage. CPU remains experimental until the complete [release gates](CPU_RELEASE.md) pass review.

## Groups and exact coverage

| Group | Suites and evidence |
|---|---|
| `safetensors` | Publisher Qwen3.5 BF16/F32 single/indexed source and a losslessly derived three-physical-shard model; both frozen text cases with one/two workers, all square/portrait/ordered-two-image cases, full captured arrays and packaged CLI/API/SSE |
| `gguf-quantized` | Qwen3.5 F16, Q8_0, Q4_0, Q4_K_M and Q6_K; separate independent references for each actual mixed tensor inventory; two text cases, full layer/teacher-logit arrays, one/two workers and packaged CLI/API/SSE |
| `gguf-vision` | Modern F16 and Q4_K_M language files paired with the F16 vision component; all three frozen image cases, encoder/merger/language/logit arrays and packaged CLI/API/SSE |
| `legacy-regression` | Eight original Qwen2.5-0.5B FP16 tokenizer/template/greedy cases and 25 Hello arrays; Nomic F16/Q8 six-case tokenizer/vector/pooling/checkpoint-row tests and five ordered batch/profile/thread configurations, with CLI/API checks |

The default runs all groups; `--group` selects one and repeated `--suite` selects exact cases. `cpu-qualification.yml` configures each group separately on macOS ARM64, Linux x86-64 and Windows x86-64. The workflow is manual; a push or configured matrix is not real-model execution evidence. Reports identify the executed source, physical layout, component identities, encoding, backend, context, threads and gaps. No configuration automatically promotes production status.

The separate [modern CPU soak workflow](CPU_SOAK.md#github-actions-execution) runs guarded Rust checks and synthetic/pinned mixed workloads on the same three native target classes. It needs no independent data bundle because it checks repeatability and lifecycle behavior against its first native capture. Its results cannot replace the numerical comparisons in this document. This is the selected route for remaining soak execution after the local Mac repeatedly reached its host cutoff.

Every value in a selected full array is checked for shape, finiteness, absolute/relative error and normalized L2 against its frozen precision contract. Nomic's original checkpoint contract contains first/last rows; that coverage remains row-specific. The image probe captures encoder outputs rather than raw preprocessing patches. Quantization quality versus F16 is reported separately from executor correctness against the identical packed file; different encodings need not produce the same text. Qwen3.5 references are independent array equations, with publisher-framework parity still pending.

The derived layout exercises real model tensors across three physical Safetensors files with unchanged named tensor bytes. It does not qualify a publisher-supplied multi-file checkpoint or other dimensions. Larger configurations, broader image semantics/codecs, all contexts, retained untested model/backend combinations, GPU, durable histories and sustained stress/security remain separate release requirements. Cache conditions are uncontrolled. Internal reservations and retired buffer capacities are checked; RSS/private memory, faults and I/O are unavailable in this runner.

TCP checks use disposable managed copies, explicit load/unload and separate administrator/inference credentials. They verify HTTP without TLS or network-isolation claims. Standalone commands have an empty executable search path, no inherited credentials and an unreachable daemon endpoint. Reports state actual process termination semantics, including Windows termination. No independent reference executor or assessment package is invoked by native jobs.

## Frozen data bundle

Large independently captured arrays remain outside Git. The current inventory is `fixtures/cpu-qualification/modern-references-v1.json`, including the unchanged original references and separately captured modern GGUF arrays. The archive descriptor is `fixtures/cpu-qualification/modern-reference-bundle-v1.json`. Earlier inventory/archive descriptors remain immutable historical records. The packaging tool copies verified bytes and the complete original legal payload; it never generates expected values from native output.

On the reference maintainer's verified cache:

```sh
python3 tools/package_cpu_references.py \
  --output target/modern-cpu-reference-data-v1.tar.gz \
  --report target/modern-cpu-reference-data-v1.json
```

The result must match the committed current descriptor. Host this exact archive at a reviewed HTTPS location before dispatching remote jobs. No approved location is currently configured, so the remote numerical campaign remains pending. The separately verified three-platform operational smoke does not use or replace this archive. Local packaging performs no publication or remote execution. The workflow accepts a location, with lengths and hashes fixed by the descriptor. Extraction checks every member against the committed inventory and reviewed legal payload, rejecting unknown/missing files, duplicates, links, unsafe/Windows-reserved paths, altered data and oversized expansion. Compressed/expanded limits are 192/256 MiB with at most 4096 files. Legal changes require a newly reviewed bundle; numerical reference identities remain immutable.

Model acquisition uses revision-pinned publisher/converter URLs, exact lengths/SHA-256, HTTPS redirects, deadlines, disk checks and exclusive atomic publication. Source sidecars and complete paired components are verified. Derived shard files must match their frozen layout hashes. Existing corrupt cache entries fail visibly. `--fetch` explicitly permits acquisition; the default requires cached assets.

## Execution and evidence

Check the host without loading a model or compiling:

```sh
python3 tools/check_cpu_host.py --report target/cpu-host-preflight.json
```

The extended runner samples host health before execution and approximately every two seconds while children run. It requires at least 1 GiB of estimated available memory and an 8 GiB disk reserve; a pending managed copy must fit above that disk reserve. On macOS it also requires the normal memory-pressure indicator, no more than 35% of physical RAM occupied by the compressor, and a nominal/fair Foundation thermal state. Available-memory estimates differ by OS and remain separate from engine reservations. Unix one-minute load must remain at or below twice the reported logical CPU count. These are conservative qualification-tool cutoffs, not model compatibility or production memory guarantees.

Unsafe or unavailable required measurements stop execution. Owned commands and servers run in dedicated process groups; a host stop terminates those groups, including build descendants, and records the failed observation. HTTP calls are interrupted when their owned server stops. Other applications are never stopped. Linux/Windows memory and disk observations are supported; portable thermal/compressor sensors are unavailable there and are reported explicitly. Actual platform validation remains required. Polling and OS-provided thermal states cannot prevent every OS or hardware fault.

The protected runner reserves disk headroom before downloading new inputs or deriving shards. Downloads and shard copies check the observer between bounded chunks; a stop removes unpublished staging data. The macOS thermal policy uses [Foundation's process thermal state](https://developer.apple.com/documentation/foundation/processinfo/thermalstate-swift.enum), not a temperature inferred from log messages.

`host-health.jsonl` retains bounded observations. Reports are checkpointed atomically after commands, numerical suites and completed HTTP suites. An interrupted record remains `running` or `fail`; completed sub-results do not turn an interrupted campaign into a pass. A new report/artifact path is required for a retry. Earlier source-bound results stay historical evidence and are not silently relabeled for changed tooling.

Build and package the same source using [CPU_RELEASE.md](CPU_RELEASE.md), then supply the actual archive basename/SHA-256 and current reference-bundle SHA-256:

```sh
python3 tools/test_cpu_qualification.py
python3 tools/test_cpu_shards.py
python3 tools/run_cpu_qualification.py --group gguf-vision \
  --archive target/cpu-release/ARCHIVE.tar.gz --archive-sha256 ARCHIVE_SHA256 \
  --native-checks target/cpu-native-checks.json \
  --reference-bundle target/modern-cpu-reference-data-v1.tar.gz \
  --reference-sha256 REFERENCE_BUNDLE_SHA256 \
  --report target/cpu-modern-vision.json
```

The runner builds locked native release probes and verifies the extracted archive's native target, legal payload and current source/build record before inference. The image probe buffers F32 trace output in 64 KiB chunks and explicitly flushes errors; serialization keeps the original little-endian F32 bytes without a full-array copy. Trace I/O is diagnostic overhead, not an inference benchmark. Keep source files unchanged throughout a run. Every run requires fresh report/artifact paths. Complete numerical results remain recorded even if a subsequent transport check fails.

Reports preserve pass/fail status, complete source snapshot, both lockfile hashes, compiler, available hardware, model revisions/hashes, layout provenance, reference inventory/archive, packaged/probe executable identities, exact commands, bounded stdout/stderr digests, numerical errors and declared gaps. Failures retain diagnostics and exit unsuccessfully. Credentials and private managed weight copies live outside uploaded evidence and are deleted afterward. Grouping bounds per-job storage and isolates failures; each platform still requires independently reviewed execution results.

The local campaign before the host restart completed seven selected suites and 896 full-array comparisons. Its preserved records predate the host guard; see [the interruption assessment](../fixtures/qualification/cpu-modern-host-interruption-v1.json). The [verification index](../fixtures/qualification/cpu-modern-verification.json) distinguishes historical execution, simulated guard failures, live observations and unfinished work. A passing host observation does not qualify a platform or establish protection against every restart. No release gate is promoted by this checkpoint.

# Pinned real-model CPU qualification

`tools/run_cpu_qualification.py` runs frozen numerical fixtures, extracted-candidate CLI inference and an authenticated loopback TCP daemon on the actual native host. It supports macOS ARM64, Linux x86-64 and Windows x86-64. The manual `cpu-qualification.yml` workflow configures all three; configuration is not execution evidence. Production status remains experimental until the complete [CPU release gates](CPU_RELEASE.md) pass review.

## Initial coverage

| Suite | Numerical and packaged CLI checks | TCP checks |
|---|---|---|
| `gguf-text` | Pinned Qwen2.5-0.5B FP16: eight exact prompt/token/greedy cases; all 24 Hello layer arrays and final logits | First frozen raw completion, native token IDs, CLI daemon client, OpenAI JSON/SSE and usage |
| `embedding-f16` / `embedding-q8` | Pinned Nomic v1.5: six tokenizer/vector/pooling cases per encoding, checkpoint first/last rows, five ordered batch/profile/thread configurations, individual/batch consistency and cleanup | Six-input embeddings API batch, ordering, vectors and token usage |
| `qwen35-text` | Pinned Qwen3.5-0.8B BF16/F32: both text cases, full 24-layer/teacher-logit arrays, exact greedy/events, indexed/single-file with one/two workers, cleanup | First frozen chat, native token IDs, CLI daemon client, OpenAI JSON/SSE and usage |
| `qwen35-image` | All three authored square/portrait/ordered-two-image cases: captured encoder/merger/language/teacher-logit arrays, exact prompts/tokens/greedy/events and cleanup | First red-image chat through Base64 input, OpenAI JSON/SSE and usage |

All comparisons use the existing frozen precision contracts. Every value in a selected full array is checked for shape, finiteness, absolute/relative error and normalized L2. Nomic's original checkpoint contract contains first/last rows; that coverage remains row-specific. The Qwen3.5 image probe captures encoder outputs rather than raw preprocessing patches. Its reference uses independent array equations; publisher-framework parity remains pending. Cache conditions are uncontrolled. Internal reservations and retired buffer capacities are checked; RSS/private memory, faults and I/O are unavailable in this runner.

Other GGUF quantizations, second model sizes, the paired Qwen2-VL GGUF path, real multi-shard checkpoints, broad image semantics/codecs, GPU, durable histories and sustained stress/security campaigns remain separate release requirements. TCP runs use disposable managed copies, explicit load/unload and separate administrator/inference credentials. They test HTTP without TLS or network-isolation claims. Standalone commands use an empty executable search path, no inherited credentials and an unreachable daemon endpoint. The report states actual process termination semantics, including Windows termination. No external inference executor is invoked.

## Frozen reference bundle

Large independent arrays remain outside Git. `fixtures/cpu-qualification/references-v1.json` pins their bytes and original manifests. The GGUF raw trace hash was already frozen in the quantization baseline; Qwen3.5 hashes come from its original independent text/image manifests. The packaging tool only copies verified bytes and never generates expected values from Coheryn output.

On the maintainer's existing verified reference cache:

```sh
python3 tools/package_cpu_references.py \
  --output target/cpu-reference-data-v1.tar.gz \
  --report target/cpu-reference-data-v1.json
```

The result must match `fixtures/cpu-qualification/reference-bundle-v1.json`. The deterministic data-only archive includes the complete original legal payload. A maintainer must host this exact archive at a reviewed HTTPS location before dispatching remote jobs. Local packaging performs no hosting or remote execution. The workflow accepts its location, with archive identity fixed by the committed descriptor. Extraction verifies every member against the reference inventory and reviewed legal payload, rejecting unknown/missing files, duplicates, links, unsafe/Windows-reserved paths, altered bytes and oversized expansion. Legal inventory changes require a newly reviewed bundle descriptor; numerical identities remain immutable.

Model downloads use revision-pinned publisher URLs, exact lengths/SHA-256, HTTPS redirects, deadlines, disk checks and exclusive atomic publication. Existing corrupt cache entries fail visibly. No model code is executed. `--fetch` explicitly permits acquisition; the default requires cached assets.

## Local execution and evidence

Build and package the same source using [CPU_RELEASE.md](CPU_RELEASE.md), then use the actual archive basename and SHA-256 from its package report:

```sh
python3 tools/test_cpu_qualification.py
python3 tools/run_cpu_qualification.py \
  --archive target/cpu-release/ARCHIVE.tar.gz --archive-sha256 ARCHIVE_SHA256 \
  --native-checks target/cpu-native-checks.json \
  --reference-bundle target/cpu-reference-data-v1.tar.gz \
  --reference-sha256 REFERENCE_BUNDLE_SHA256 \
  --report target/cpu-model-qualification.json
```

Repeat `--suite` to select a subset; add `--fetch` for missing assets. Embedding-only runs need no bundle because their expected vectors are committed. No new Python or Rust dependency is introduced. The runner builds locked native release probes and verifies the extracted archive's native target, legal payload and current source/build record before inference. Keep source files unchanged during execution.

Each run requires new report/artifact paths. Pass/fail reports preserve the full source snapshot, both lockfile hashes, compiler, OS/architecture/CPU count, model revisions/hashes, reference inventory/archive, packaged/probe executable identities, exact commands, bounded complete stdout/stderr hashes, numerical errors and declared gaps. Failed runs retain diagnostics and exit unsuccessfully. Private credentials and managed weight copies live outside uploaded evidence and are deleted afterward. The workflow uploads native checks, candidate and qualification evidence without publishing a release or marking compatibility/production gates complete. Review actual successful artifacts independently for each platform.

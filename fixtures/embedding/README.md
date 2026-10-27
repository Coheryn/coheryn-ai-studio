# E0 embedding reference fixtures

These records prepare the independent oracle for native embedding development. **E1 now executes the exact F16/Q8_0 files through the Rust CPU library.** E2 now exposes `embed` and `/v1/embeddings`; no generation model is used to produce these vectors.

`models.json` pins the publisher's Nomic Embed Text v1.5 F16 and Q8_0 GGUF files, source revisions, exact lengths, SHA-256 values and tokenizer/config/pooling assets. `source/` and the intact resource archive contain upstream metadata, vocabulary and license/provenance records; `models.json` identifies relocated asset paths without changing their bytes. Remote-code declarations are data only; preparation does not execute publisher Python code. The inspected architecture is `nomic-bert`, a distinct bidirectional encoder, rather than the Qwen2 generation graph.

`reference-contract.json` freezes 768-dimensional mean pooling over actual tokens (including qualified special tokens), L2 normalization, literal caller-supplied query/document prefixes, packed sequence masks, special-token handling and numeric tolerances before E1. The GGUF metadata declares 2,048 positions (the source also advertises 8,192); `*.metadata.json` records the actual graph and special-token parameters. The initial reference context is 512 and truncation is disabled. Reduced Matryoshka dimensions and long-context scaling are not qualified. A model's advertised maximum is not evidence of execution at that length.

Each reference file contains six English/Bengali/Unicode/whitespace/special-token cases, exact token IDs, masks/positions, selected first/last encoder checkpoint rows, raw pooled vectors, normalized vectors and duplicate/mixed-length batch comparisons. Pooling is independently reconstructed from the encoder's complete token output in ignored raw captures. The reference uses CPU, one thread, no flash attention and the pinned external reference engine build **outside Cargo**. Production dependencies remain independent.

Check the frozen inputs offline with:

```sh
python3 tools/check_embedding_fixtures.py
```

Original capture/build source is retained as inert research under `resources/reference-archive/`; it is not part of the native execution/build/test chain. New captures run in separately managed assessments and exchange verified data only. Never change the frozen contract or regenerate expected outputs from the native executor to hide errors. Full encoder arrays and weights stay in ignored caches. `quantization-observations.json` compares reference models separately from executor correctness; these vectors do not establish retrieval quality.

The archived comparator record under `resources/reference-archive/` pins a verified standalone comparator distribution. Its binary help was checked, but its daemon and embedding benchmarks have **not** run. The pinned daemon initializes identity keys in the real user home; preparation does not create those keys or override the user's home. E1 native parity and explicit truncation/boundary evidence are separate from these historical E0 captures. external reference engine comparison, broader API/client/platform qualification and calibrated throughput/peak-memory workloads, GPU and other-platform qualification remain acceptance work. The comparator is never a production dependency.

Current runtime contracts and reproduction commands are in [the E1 embedding guide](../../docs/EMBEDDINGS.md). `*.multilingual.json` adds 50 independently captured language/Unicode cases per file; `*.boundary.json` records actual 512-token inputs. Frozen E0 flags describe capture-time status and are not the current runtime compatibility matrix.

E2 transport, shared scheduling, atomic export, SDK and external RAG/vector-store contracts are documented in [EMBEDDING_API.md](../../docs/EMBEDDING_API.md), with current evidence separate from immutable reference captures.

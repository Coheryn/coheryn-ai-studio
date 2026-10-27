# E0 embedding reference fixtures

These records prepare the independent oracle for native embedding development. **E1 now executes the exact F16/Q8_0 files through the Rust CPU library.** E2 now exposes `embed` and `/v1/embeddings`; no generation model is used to produce these vectors.

`models.json` pins the publisher's Nomic Embed Text v1.5 F16 and Q8_0 GGUF files, source revisions, exact lengths, SHA-256 values and tokenizer/config/pooling assets. `source/` contains inert upstream metadata, vocabulary and license/provenance records. Remote-code declarations are data only; preparation does not execute publisher Python code. The inspected architecture is `nomic-bert`, a distinct bidirectional encoder, rather than the Qwen2 generation graph.

`reference-contract.json` freezes 768-dimensional mean pooling over actual tokens (including qualified special tokens), L2 normalization, literal caller-supplied query/document prefixes, packed sequence masks, special-token handling and numeric tolerances before E1. The GGUF metadata declares 2,048 positions (the source also advertises 8,192); `*.metadata.json` records the actual graph and special-token parameters. The initial reference context is 512 and truncation is disabled. Reduced Matryoshka dimensions and long-context scaling are not qualified. A model's advertised maximum is not evidence of execution at that length.

Each reference file contains six English/Bengali/Unicode/whitespace/special-token cases, exact token IDs, masks/positions, selected first/last encoder checkpoint rows, raw pooled vectors, normalized vectors and duplicate/mixed-length batch comparisons. Pooling is independently reconstructed from the encoder's complete token output in ignored raw captures. The reference uses CPU, one thread, no flash attention and the pinned llama.cpp build **outside Cargo**. Production dependencies remain independent.

Reproduce after building the workspace and installing the separately pinned reference:

```sh
python3 tools/setup_reference.py
python3 tools/prepare_embedding.py
python3 tools/check_embedding_fixtures.py
```

Preparation verifies committed source assets, fetches bounded SHA-256-checked model bytes if absent, compiles the standalone reference helper and regenerates reference records. Review regenerated fixtures; never change the frozen contract to hide executor errors. Full encoder arrays and weights stay under ignored `.cache/`. The offline checker runs in CI without model downloads or an external runtime. `quantization-observations.json` compares the two reference models separately from executor correctness; these six vectors do not establish retrieval quality.

`ollama-reference.json` pins a verified standalone comparator distribution. Its binary help was checked, but its daemon and embedding benchmarks have **not** run. The pinned daemon initializes identity keys in the real user home; preparation does not create those keys or override the user's home. E1 native parity and explicit truncation/boundary evidence are separate from these historical E0 captures. Ollama comparison, broader API/client/platform qualification and calibrated throughput/peak-memory workloads, GPU and other-platform qualification remain acceptance work. The comparator is never a production dependency.

Current runtime contracts and reproduction commands are in [the E1 embedding guide](../../docs/EMBEDDINGS.md). `*.multilingual.json` adds 50 independently captured language/Unicode cases per file; `*.boundary.json` records actual 512-token inputs. Frozen E0 flags describe capture-time status and are not the current runtime compatibility matrix.

E2 transport, shared scheduling, atomic export, SDK and external RAG/vector-store contracts are documented in [EMBEDDING_API.md](../../docs/EMBEDDING_API.md), with current evidence separate from immutable reference captures.

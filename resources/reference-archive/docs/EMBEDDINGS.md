# Native CPU embeddings — E1

The Rust engine independently executes the qualified `nomic-bert` encoder in the pinned Nomic Embed Text v1.5 F16 and Q8_0 GGUF files. It returns ordered, 768-dimensional, L2-normalized vectors without loading a generation model. Exact hashes, tensor inventories and evidence are linked in [the compatibility matrix](../fixtures/compatibility.json). This is experimental macOS ARM64 CPU qualification. E2 now exposes `coheryn embed`, shared daemon scheduling and `POST /v1/embeddings`; see [the transport/RAG guide](EMBEDDING_API.md).

## Rust interface

```rust,ignore
use coheryn_engine::{Cancellation, ResourceLimits};
use coheryn_engine::embedding::{EmbeddingModel, EmbeddingOptions};

let model = EmbeddingModel::load(
    ".cache/models/nomic-embed-text-v1.5.f16.gguf",
    ResourceLimits { context: 512, max_memory: 256 << 20 },
)?;
let vectors = model.embed(
    &["search_query: What is local inference?",
      "search_document: Local inference runs on your own computer."],
    &EmbeddingOptions::default(),
    &Cancellation::default(),
)?;
assert_eq!(vectors.dimension, 768);
```

These interfaces remain internal to the workspace's development contract. `embed_with_hardware` accepts an explicit hardware snapshot. The convenience `embed` entrypoint probes the CPU count but reports RAM/GPU capacity as unavailable; callers needing host-pressure admission must supply a usable snapshot. `plan` estimates before tokenization or execution. `reserve_request` atomically holds a concurrency slot and the shared budget; `embed_admitted` consumes that reservation without a release/reacquire gap. Model descriptors/configuration and admissions expose read-only accessors. Admissions cannot be transferred between models. `load_shared` allows multiple models to use one `Budget`; its limit must not exceed the supplied memory policy.

## Model and tokenizer contract

The loader validates bidirectional attention, mean pooling, QKV layout, token-type embeddings, LayerNorm affine tensors, RoPE settings, gated feed-forward tensors and all dimensions before encoder allocation. Required matrices support native F32/F16/Q8_0 reads; norm vectors require F32. Extra tensors, unimplemented architecture metadata, other pooling/tokenizer variants, shards and unsupported encodings fail explicitly. F32 uses synthetic operation/model fixtures; the actual publisher qualification covers F16 and Q8_0, including their F32 tensors.

The adapter uses the GGUF WordPiece vocabulary, CLS/SEP identities, literal special-token-looking input, zero token types, and caller-supplied query/document prefixes. No prefix is invented. Each packed sequence starts its own positions at zero and attends only to its own real tokens, including CLS/SEP. Mean pooling excludes other sequences and L2 normalization always returns the full qualified dimension. Zero/nonfinite output norms are errors. Matryoshka slicing and long-context scaling are unsupported.

Unicode processing is model-specific. The pinned GGUF/reference WordPiece normalizer maps each original code point to its **first canonical decomposition code point**, strips all Unicode marks (Mn/Mc/Me), cleans controls, lowercases, and splits punctuation/Chinese characters. This is lossy and differs from standard Hugging Face BERT normalization, notably for Bengali marks and composed Korean syllables. It must not be applied to the Qwen byte tokenizer or general application text. The adapter preserves the qualified GGUF behavior; it does not claim equivalence to the source model's Hugging Face tokenizer. Composed and decomposed spellings can therefore differ under this contract. Any future normalizer change must introduce a separately qualified embedding-space identity rather than silently changing existing vectors.

The multilingual corpus covers 50 representative language/Unicode cases: Indic scripts, Arabic/Hebrew/Persian/Urdu, East Asian scripts, Southeast Asian scripts, Cyrillic, Greek, Latin accents, Armenian, Georgian, Amharic, Cherokee, mixed scripts, emoji, rare characters, controls and combining marks. Ordinary CI tests all frozen token IDs using the committed publisher vocabulary, without model downloads. Full native F16/Q8_0 comparisons also check all 100 multilingual vectors against the independent encoder. **This is tokenizer/numerical compatibility evidence, not semantic proficiency in every language.** Unsupported vocabulary pieces become the model's UNK token. Choose a separately qualified multilingual model when the selected model's language or retrieval quality is insufficient.

## Limits and memory

| Limit | Current contract |
|---|---|
| Input text | 1–65,536 UTF-8 bytes per item; whitespace-only text is valid |
| Request | 1–256 items, at most 8 MiB total UTF-8 bytes |
| Input tokens | 2–512, including CLS/SEP; must fit the configured model context |
| Truncation | Off by default; explicit opt-in preserves CLS/SEP and reports each shortened item |
| Requested microbatch | 1–64 items and input-token-limit–4,096 packed tokens |
| Workers | 1–16, bounded by CPU share; reference/low-memory use one |
| Output | Fixed model dimension; ordered indices, per-item token counts, total usage |

Batch size is an upper bound. CAEP can reduce packed microbatch capacity to fit the engine budget and observed available RAM. It never reduces the per-input token limit or vector dimensions. Low-memory uses one input's token capacity and small I/O tiles. Profiles use 4/16/32-token matrix tiles, read each weight tile once for multiple input rows, and reuse rounded/quantized activations and scratch. Dedicated pools keep scalar reduction order unchanged across worker counts.

Memory scales linearly with admitted packed tokens and model dimensions. Attention reuses one score row per worker; it never stores a quadratic attention matrix. Weights remain in bounded buffered storage without persistent full-model dequantization, a full-weight heap copy, or a generation KV cache. Tokenization, retained token IDs, output vectors, worker stacks, scratch, I/O and headroom have separate admission allowances. Completed vectors retain their output reservation until the result is dropped; compute capacity and slots are released on completion. Cancellation, input rejection, storage changes and observer errors release resources through RAII.

Reservations are conservative estimates, not an OS memory cap. Observed engine vector capacities exclude allocator overhead, tokenizer-library internals, page cache and other process allocations. RSS/private memory, faults, GPU memory and I/O require separate measurement; these numerical probes do not measure them. Benchmark timings include uncontrolled cache effects, and trace runs also include checkpoint serialization. No performance superiority is claimed.

## Reproduce

Acquire the two byte-verified models using `tools/fetch_model.py --manifest fixtures/embedding/models.json --id MODEL_ID`, then:

```sh
./tools/cargo build --release -p coheryn-engine --example embedding_probe --locked
python3 tools/check_embedding_fixtures.py
python3 tools/check_native_embeddings.py
python3 tools/check_embedding_languages.py
python3 tools/check_embedding_boundaries.py
```

`embedding_probe` is a development-only JSONL diagnostic example. It is not the public `embed` CLI contract. Its request file contains `inputs`, optional `options`, `max_memory`, `trace` and `tokenize_only`; unknown fields are rejected. Raw arrays and model bytes remain in ignored `.cache/`. Independent reference capture is separate: `prepare_embedding.py`, `check_embedding_languages.py --capture` and `check_embedding_boundaries.py --capture` use the pinned external oracle. Never regenerate expected outputs from Coheryn or relax thresholds to hide regressions. Historical E0 manifests retain their original native-unsupported status; current runtime qualification is recorded separately.

E2 implements bounded CLI/API transport, task-aware daemon residency/queueing, float/Base64 output and SDK interoperability. E3/G5/G6 retain Ollama/comparable benchmark runs, calibrated peak-memory/throughput workloads, Metal and actual Linux/Windows qualification. Full RAG, document ingestion, vector databases and retrieval stay outside the core engine.

E2 adds queue-safe capped reservations and resumable embedding jobs. Preparation tokenizes once; each `step` executes one bounded microbatch over a reusable workspace, and `finish` transfers only output reservations. A failed step poisons the job; dropped/incomplete jobs reclaim their held buffers. Existing single-call and observed interfaces use the same sequence internally.

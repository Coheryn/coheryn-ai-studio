# G4 packed CPU execution

Reference capture source is now an inert research archive. Native verification consumes frozen independent outputs; new captures run in separately managed assessments. See [INDEPENDENCE.md](INDEPENDENCE.md).


The engine implements native Q8_0, Q4_0, Q4_K and Q6_K block decoding and direct packed matrix/vector products. Q5_0 is also required by the actual publisher 0.5B Q4_K_M inventory. A quantization recipe name never substitutes for inspecting tensor types. F32/F16 tensors remain supported and mixed-type files bind through the same validated Qwen2 graph.

One bounded input vector is quantized to Q8_0/Q8_K once per matrix. Integer dot products read packed rows directly; only embedding lookups and bounded deep-validation chunks materialize floats. There is no persistent full-model dequantization, full-weight heap copy or weight mapping. Workers operate on disjoint output rows, with fixed per-row reduction order and cooperative cancellation between row groups. All execution is safe Rust. Rayon supplies general-purpose scheduling, not numeric execution.

Session admission covers KV capacity, reusable I/O/decode/activation/attention buffers and bounded worker stacks. Attention reuses one linear score row and a per-head accumulator. This retains linear attention workspace rather than allocating a quadratic matrix. Prefill chunks are bounded groups of sequential token forwards; they are not a batched matrix multiplication optimization. See [the planner contract](PLANNER.md) for profiles, thread shares, host observations and accounting limits.

## Reproduce

```sh
python3 tools/fetch_model.py --manifest fixtures/quantized/models.json --id qwen2.5-0.5b-instruct-q4_k_m
cargo test -p coheryn-engine --test quant --locked
python3 tools/check_service.py --manifest fixtures/quantized/models.json --id qwen2.5-0.5b-instruct-q4_k_m --golden fixtures/qualification/g4-qwen2.5-0.5b-instruct-q4_k_m.json --evidence-out fixtures/qualification/g4-service.json
python3 tools/benchmark_quantized.py
```

Fetch each selected manifest entry before running the full model check. Large weights and raw traces stay ignored. Kernel and RoPE capture tools link only independently built development helpers. The fixtures cover 100 packed block/activation/dot combinations, plus 12 positional operation cases with nonzero positions. Activation bytes and decoded block values match their fixtures exactly. RoPE operation tolerance bounds platform trigonometric rounding; existing whole-model tolerances remain unchanged.

`fixtures/quantized/models.json` pins four publisher 0.5B quantized files and the second-size 1.5B FP16/Q4_K_M configuration. The qualifier verifies the complete bytes, scans supported payloads, records actual tensor inventories, checks exact prompt/token/greedy behavior and compares teacher-forced layer/logit checkpoints against identical-file references. The primary 0.5B files use all eight greedy cases. The second-size contract requires exact primary tokenizer/template fingerprints, all eight tokenizer/render cases, English/Unicode greedy outputs and 58 layer/logit comparisons across two teacher-forced positions, with English profile parity. These explicit contracts are frozen in the manifest before second-size acceptance; numeric tolerances are unchanged. Alternative execution profiles must preserve their qualified outputs. Completed independent reference cases are checkpointed immediately. `--reuse-reference-cache` reuses them only when model, prompt, oracle/build and output hashes match; native comparisons run again. Partial `--case` runs remain ignored development results and do not promote compatibility. Numerical tolerances are executor criteria; weight quantization quality is a separate comparison.

Qualification records identify the tested executable hash. They are evidence from specific builds and hardware, not a claim that arbitrary binaries or GGUF files work. Context reservations of 128/512/4096 with short prompts do not qualify long-context inference. The benchmark runs fresh child processes and records per-child peak RSS, faults, I/O counters, timings and engine reservations/capacities. Cache conditions are uncontrolled; worker stacks and library/allocator overhead are excluded from observed vector capacities. No general speed or total-memory advantage is implied.

Current model/platform qualification is limited to the exact entries in [the compatibility matrix](../fixtures/compatibility.json). ISA-specific SIMD, batched prefill GEMM, alternate KV representations, native vision, embedding GPU/platform/benchmark qualification, Metal and other CPU-platform qualification remain separate work. Unsupported graphs, encodings, templates and backend combinations fail explicitly.

## Local measurement evidence

`fixtures/qualification/g4-benchmarks.json` records two iterations per profile at context reservations 128/512/4096 on an 8 GB Apple M1, with a one-token prompt and eight generated tokens. No competing validation job ran during capture; OS cache residency, CPU placement and background applications remain uncontrolled. At context 512, the median decode samples were:

| Profile | Explicit workers | Decode tokens/sec |
|---|---:|---:|
| reference | 1 | 4.34 |
| balanced | 2 | 6.60 |
| performance | 4 | 5.38 |
| low-memory | 1 | 3.98 |

These measurements use the recorded Rust 1.90.0 build and remain a historical baseline after the dependency refresh; current upgrade validation is described in [DEPENDENCIES.md](DEPENDENCIES.md). The balanced two-worker default is retained. More workers do not guarantee higher throughput; explicit `--threads` allows local calibration. Low-memory saved 737,280 bytes of observed vector capacity per session. These observations do not measure a total-system memory advantage: worker stacks, allocator/libraries and OS file cache are separate. The JSON includes each child's peak RSS/fault/I/O observations and conservative reservations. Load timing measures metadata/tokenizer/storage initialization; buffered weights are read during execution.

`g4-service.json` records real quantized CLI/native/OpenAI/SSE parity, role enforcement, active-unload rejection, disconnect/shutdown cleanup and explicit daemon failure. `g4-planner.json` records 24 plans across contexts/concurrency/profiles, exact profile outputs, live two-session admission, cancellation and return to zero reservations after unload. Both identify the tested executable hash.

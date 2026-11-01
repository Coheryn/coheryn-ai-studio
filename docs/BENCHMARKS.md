# Native text benchmark observations

The integrated `coheryn bench` command uses the same native text adapter registry as `run` and `chat`. It supports the implemented Qwen2 GGUF and Qwen3.5 GGUF/Safetensors adapters. Unsupported architectures and configurations fail through the same validation and admission path; embedding benchmarks retain their separate task interface. No external inference executor is involved. This normal benchmark command is native Rust and needs no Python. The separate `coheryn-bench` executable retains its initial Qwen2 GGUF scope.

```sh
target/release/coheryn bench \
  --model .cache/qwen35-gguf/Qwen3.5-0.8B-q4_k_m.gguf \
  --standalone --prompt Hello --context 512 --max-tokens 8 \
  --profile balanced --threads 2 --iterations 3
```

This benchmarks raw text completion. Interactive chat additionally renders a template and accumulates conversation history. The command prints JSON after completion; engine first-token timings are distinct from terminal display latency. Run the same prompt, model bytes, backend, context and output settings when comparing worker counts. More workers are not automatically faster.

The response retains `backend`, `mode`, `profile`, `requested_threads`, `load_ms`, `cache_condition` and `runs`. `adapter` identifies the selected native architecture, container and supported capabilities in standalone mode; daemon client mode reports `null` because it does not load an adapter locally. Every iteration creates fresh generation state while reusing the loaded model. `prefill_ms`, `decode_ms`, `tokenization_ms`, `first_token_ms`, delivery timing, token counts and tracked reservations remain separate measurements. Engine reservations are not process RSS, and these observations do not establish a service SLA or a CPU/GPU speed advantage.

## Pinned integration checks

`tools/check_text_bench.py` runs the explicit `native_text_bench` Rust integration test under the existing host guard. The [benchmark workflow](../.github/workflows/cpu-text-bench.yml) selects macOS ARM64, Linux x86-64 and Windows x86-64 separately for:

- Qwen2.5-0.5B FP16 GGUF.
- Qwen3.5-0.8B Q4_K_M GGUF with its pinned source assets.
- Publisher Qwen3.5 BF16/F32 Safetensors through directory, index and direct-file entrypoints. The publisher index in this check names one physical weight shard; this is not a new multi-shard qualification.

Each source uses its first two frozen prompt/token/greedy fixtures. Required workers are one and two, plus four where the host admits four workers. Every case runs two fresh generations per benchmark invocation. The checks require exact frozen token IDs and text, exact prompt/output token counts, zero prefix reuse, complete native routing, finite timing observations and the declared engine budget. Model input hashes are checked before and after execution. Source, compiler, executable, input and diagnostic identities are retained.

```sh
python3 tools/check_text_bench.py --suite qwen35-gguf-q4_k_m-text \
  --fetch --report target/text-benchmark-checks.json
```

Use fresh report paths. Acquisition is explicit, revision/hash pinned and protected by the same host limits. GGUF acquisition selects the requested language encodings and adds the vision component when a GGUF image suite is selected. Publisher Safetensors acquisition retains the complete pinned source checkpoint. Existing cached files remain intact. All compiler/test children are owned and terminated on a host rejection or deadline. Empty, failed, filtered, incomplete or source-mutated runs cannot pass. The guard does not guarantee protection from every OS/hardware fault.

This check needs the committed small greedy fixtures rather than the large numerical-array archive. It does not replace the full numerical matrix in [CPU_QUALIFICATION.md](CPU_QUALIFICATION.md). Input verification warms caches; OS file residency and shared-runner load remain uncontrolled. Results are observations to guide later kernel optimization. Cross-platform execution and artifact review are required before claiming a passing benchmark checkpoint.

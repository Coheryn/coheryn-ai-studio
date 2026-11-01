# Optional Metal backend

Metal development now precedes additional architecture work. The initial implementation is an optional `coheryn-metal` workspace crate plus the engine's `metal` feature. It contains Coheryn-owned dense F32/F16/BF16 and packed Q4_0/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K matrix kernels and a checked `Tensor`/`Storage` bridge shared by GGUF and Safetensors bindings. CPU-only builds do not activate or link the platform bindings. Architecture adapters still execute on CPU; no CLI/API GPU execution profile is available yet.

The first kernel establishes a numerical baseline before optimization. [Frozen reference cases](../fixtures/gpu/metal-dense-v1.json) use an independent Python standard-library oracle: F32 products, sequential F64 accumulation and final F32 rounding. The GPU uses compensated F32 accumulation with fast math and contraction disabled. Acceptance checks every finite result against `2e-5 + 2e-5 * abs(expected)`. Cases cover cancellation-sensitive sums, rectangular shapes, batch lanes, tails, long reductions and F16 subnormals. This primitive precision contract does not replace architecture-level logit, activation, embedding or image comparisons.

[Packed comparisons](../fixtures/gpu/metal-packed-v1.json) add 120 cases and 720 scalar outputs derived from the original independent block-decoding fixtures, with mixed rows, multiple blocks and two F32 activation lanes. Their expected results were frozen before implementing packed GPU reads. Weights stay encoded in GPU storage and decode into scalar registers. This path preserves F32 activations; it does not implement the legacy CPU Q8 activation-dot contract or establish model correctness. Block alignment and byte counts are checked before allocation.

GPU buffers hold reservations from the same engine `Budget` as CPU allocations. Weights upload directly in chunks no larger than 256 KiB, without a full CPU heap copy. Each request owns reusable input/output buffers. Shared Apple Silicon memory is reserved once for each requested allocation; driver-reported allocation sizes remain separate observations. Driver/compiler overhead, physical residency and OS pressure remain outside the engine-controlled byte limit and require host monitoring. The initial backend explicitly requires unified memory.

Only the private platform module allows documented unsafe operations; the engine continues to forbid unsafe code. Buffer pointers never escape. Kernel dimensions, index ranges, device identity and buffer sizes are checked before dispatch. Submitted work retires before output access or resource release, including cancellation and Rust unwinding. Cancellation checks occur between bounded dispatches; a stuck driver does not have a guaranteed cooperative cancellation deadline. Model packages cannot supply shader code.

The standalone native checks require an admitted Apple Silicon Mac:

```bash
cargo fetch --locked
python3 tools/check_metal_fixtures.py
python3 tools/check_metal_packed_fixtures.py
python3 tools/check_cpu_host.py --report target/metal-host-unique.json
python3 tools/run_metal_checks.py --gpu --report target/metal-checks-unique.json
```

Use fresh report paths. The runner monitors the existing host limits during compilation and execution, retains logs and binds evidence to source, kernels and fixture bytes. `--gpu` requires actual GPU execution; unavailable devices fail. Without it, passing checks establish compilation/library tests only. The [workflow](../.github/workflows/metal.yml) uses the standard macOS runner and requests no paid GPU runner. Hosted GPU availability and capacity must pass the same checks.

A separate Linux job uses `--cross-check` to compile the macOS bindings and tests for `aarch64-apple-darwin` under the host guard. It never executes foreign binaries, emits a distinct evidence kind and cannot satisfy native or GPU numerical acceptance. This provides compiler feedback when a macOS runner is not admitted.

Remaining GPU work includes real-device acceptance of the dense and packed reference kernels, qualified accelerated architecture dispatch, planner placement and CLI/API selection, legacy activation precision where required, attention/vision/embedding coverage, buffer reuse and throughput tuning, and complete numerical/lifecycle/platform qualification. No model/backend combination or production release is promoted by this initial boundary.

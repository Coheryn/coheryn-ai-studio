# Optional Metal backend

Metal development now precedes additional architecture work. The optional `coheryn-metal` workspace crate and engine `metal` feature contain Coheryn-owned dense F32/F16/BF16 and packed Q4_0/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K matrix kernels. CPU-only builds do not activate or link the platform bindings. Default architecture selection and CLI/API execution remain CPU. An explicit experimental Qwen3.5 library binding now routes language matrix operations through Metal; pretrained-model qualification is pending.

On an admitted native Mac, `qwen35::Model::enable_metal` consumes a loaded model and binds its validated graph matrices. Exact tied aliases share one encoded GPU allocation; CPU embedding lookup, attention, recurrent state, norms and sampling remain in the shared native graph. GPU residency is reserved atomically before upload, and a complete minimum request is checked before device compilation. Each request reserves CPU and private GPU buffers together, transferring the GPU portion to buffer-owned leases without double charging. A foreign or insufficient reservation fails before allocation. Switching execution invalidates previous cache/state ownership. The profile is text-only and rejects image execution explicitly.

Model info reports device identity, encoded and driver-observed weight allocation, private workspace bytes and cumulative completed GPU dispatches separately. The planner includes shared GPU residency through the loaded budget and adds private GPU workspace once per admitted request. Unified allocations are not counted again as separate VRAM. Dynamic activation transfers and compiler/driver overhead are not inferred from logical reservations. Model inference never silently falls back after a GPU binding or execution failure.

The first kernel establishes a numerical baseline before optimization. [Frozen reference cases](../fixtures/gpu/metal-dense-v1.json) use an independent Python standard-library oracle: F32 products, sequential F64 accumulation and final F32 rounding. The GPU uses compensated F32 accumulation with fast math and contraction disabled. Acceptance checks every finite result against `2e-5 + 2e-5 * abs(expected)`. Cases cover cancellation-sensitive sums, rectangular shapes, batch lanes, tails, long reductions and F16 subnormals. This primitive precision contract does not replace architecture-level logit, activation, embedding or image comparisons.

[Packed comparisons](../fixtures/gpu/metal-packed-v1.json) add 120 cases and 720 scalar outputs derived from the original independent block-decoding fixtures, with mixed rows, multiple blocks and two F32 activation lanes. Their expected results were frozen before implementing packed GPU reads. Weights stay encoded in GPU storage and decode into scalar registers. This path preserves F32 activations; it does not implement the legacy CPU Q8 activation-dot contract or establish model correctness. Block alignment and byte counts are checked before allocation.

GPU buffers hold reservations from the same engine `Budget` as CPU allocations. Weights upload directly in chunks no larger than 256 KiB, without a full CPU heap copy. Each request owns reusable input/output buffers. Shared Apple Silicon memory is reserved once for each requested allocation; driver-reported allocation sizes remain separate observations. Driver/compiler overhead, physical residency and OS pressure remain outside the engine-controlled byte limit and require host monitoring. The initial backend explicitly requires unified memory.

`WorkspaceLimits` admits maximum input width, output rows and batch independently of weight encoding. One workspace can serve smaller validated layer shapes without reallocating buffers. Each dispatch uses the actual matrix strides; capacities are checked before copying input or submitting work. The packed test reuses buffers across all cases, checks smaller output strides and rejects insufficient capacity without publishing output.

Only the private platform module allows documented unsafe operations; the engine continues to forbid unsafe code. Buffer pointers never escape. Kernel dimensions, index ranges, device identity and buffer sizes are checked before dispatch. Submitted work retires before output access or resource release, including cancellation and Rust unwinding. Cancellation checks occur between bounded dispatches; a stuck driver does not have a guaranteed cooperative cancellation deadline. Model packages cannot supply shader code.

Immutable matrices can be shared across workers; an idle workspace can move to its owning worker. The private boundary documents these thread-safety invariants explicitly. Workspaces require exclusive access and do not expose shared mutable activation storage. A real-device test submits concurrent reads of one matrix with distinct inputs, moves private workspaces into both workers and checks numerical isolation and reservation reclamation after they join.

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

The [reviewed dense checkpoint](../fixtures/qualification/metal-dense-native-v1.json) passed six independent dense cases and the native engine storage boundary at source `e6ff90b`, with API/GPU validation enabled and 125 healthy host observations. Its device identifies as **Apple Paravirtual device**. This is evidence for that hosted Metal execution, not physical M1 performance, packed kernels, later source changes or model acceleration. Original reports and artifact digests remain attached to their executed source.

Remaining GPU work includes real-device acceptance of the dense and packed reference kernels, qualified accelerated architecture dispatch, planner placement and CLI/API selection, legacy activation precision where required, attention/vision/embedding coverage, buffer reuse and throughput tuning, and complete numerical/lifecycle/platform qualification. No model/backend combination or production release is promoted by this initial boundary.

The guarded runner can additionally acquire and test the pinned modern Q4_K_M model with `--gpu --model-suite qwen35-gguf-q4_k_m-text --fetch`; `qwen35-text` selects the published Safetensors source. These modes compare complete native CPU/GPU layer and logit arrays at two positions, then the original independent sampled layer/teacher-forced checkpoints and greedy answers for both frozen prompts. Existing model tolerances remain unchanged. Cancellation, consumer failure, unsupported images and unload must reclaim request allocations. This is selected native library qualification, separate from CLI/API, all-reference-array, physical-device and production acceptance.

# Mixed CPU workload soak

The development-only driver exercises the real native engine, authenticated HTTP handlers, scheduler, admission, persistent history and model lifecycle in one process. It is separate from the shipped CLI and engine interfaces and introduces no dependencies. It supports deterministic small GGUF fixtures and separately pinned pretrained inputs.

```sh
python3 tools/test_cpu_soak.py
python3 tools/run_cpu_soak.py \
  --report fixtures/qualification/cpu-soak-synthetic-v1.json
```

The default [frozen operational profile](../fixtures/soak/cpu-synthetic-v1.json) runs 100 measured cycles for each combination of concurrency one/two and balanced/low-memory execution: 400 measured cycles in total. Each cycle submits text generation, a six-item embedding batch and two-image generation concurrently. Outputs and usage must match the first native capture under the same configuration; repeated embeddings must agree and remain finite/unit-normalized. This detects cross-request state contamination. Independent numerical correctness remains the responsibility of the frozen model reference tests.

Use a fresh report filename for every run. Existing reports/run directories are never overwritten. The wrapper builds the optimized driver, freezes the exact configuration, binds source/reference/lockfile/executable identities, streams observations and records a failed outcome on errors. Source changes during a run invalidate its evidence. `--timeout-seconds` applies a hard deadline to the owned driver process; exceeding it is a failure, never a shortened passing run.

For a small harness regression run:

```sh
python3 tools/run_cpu_soak.py --profile fixtures/soak/cpu-smoke-v1.json \
  --report target/cpu-soak-smoke.json
```

## Workload and fault coverage

Warmup executes the fault paths as well as normal requests. Measured cycles inject faults at the profile's explicit cadence. Every configured workload must demonstrate:

- A fully occupied waiting queue and a structured `QUEUE_FULL` response. Active/queued work pins the model; dropping all request handles releases it.
- A consumer that does not read token events. The bounded queue retains exactly sixteen token events and a cancellation error. Completion before backpressure is exercised is a failed workload, not a passing cancellation test.
- An unread image SSE body that reaches the actual application backpressure counter. Text and embedding peers must finish before the stalled image consumer times out. Dropping the body must release request reservations, buffers and residency pins.
- Explicit memory and oversized-token rejection, followed by successful model unload/reload without reservation drift.
- Durable receipt replay after reopening the service with no models resident. Replay must return the exact original receipt without executing inference.

Every drain requires zero active/queued/compute-waiting work, zero request buffer capacities, all HTTP permits returned and the exact established resident reservation baseline. Persistent history legitimately retains its own reservation after weights unload. The runner seeds and closes a bootstrap service, then restores the history before establishing separate history-only and loaded-model baselines. This compares the same hydrated-page state across all measured restarts. Closing each service must release all its reservations and tracked buffers.

These are service restarts within one process, which retains allocator effects for growth detection. Actual process-kill recovery is covered separately by `crash_recovery`. HTTP bodies run through the real router in process; this gives deterministic application backpressure without relying on OS socket-buffer sizes. It does not qualify TCP/TLS/proxy behavior or network disconnects. Existing loopback transport tests remain separate evidence.

Prefix KV reuse is disabled in this profile so idle cache ownership cannot mask request leaks; persistent history remains enabled. Cache-enabled workload calibration and multi-principal fairness require separate profiles. Vision delivery backpressure uses the same observable counter for the paired GGUF and Qwen3.5 checkpoint paths. A blocked checkpoint image delivery must not disappear from this metric.

Qwen3.5 image requests use the bounded vision phase, independently of text execution/profile permits. A queued image can acquire that phase while a text response remains pinned, and an unread image body cannot occupy the text phase. Both paths retain shared admission slots, global memory reservations and the single FIFO compute gate. All generation paths use one bounded channel size: sixteen token events plus a terminal slot.

## Memory and latency acceptance

The profile is fixed before execution. The default limits are independent quantities:

| Quantity | Default operational acceptance |
|---|---|
| Engine reservation/admission ceiling | 768 MiB |
| Sampled process RSS ceiling | 768 MiB |
| Drained RSS growth from the warmed baseline | At most 64 MiB per configuration |
| Request completion | Within the configured 15-second operational timeout |
| Cleanup | Within the configured 5-second timeout |
| Slow-consumer timeout | 1 second; peers must progress before that cancellation |
| RSS sampling | Every 50 ms; missing required measurements fail the run |

These limits are a versioned synthetic workload contract, not a general performance SLA or a production memory calibration. A different pretrained model/hardware workload needs its own reviewed profile before execution. The runner never increases limits, reduces context/concurrency, skips a fault, retries to obtain a passing result or silently substitutes a model/backend. Preserve failed attempts and diagnose them before revising a contract.

Internal reservation and engine-buffer high-water counters are recorded separately from sampled RSS. RSS includes the driver, tokenizer, store, libraries and observer; sampling can miss brief peaks. Virtual memory is address space, not physical residency, and is not used as a RAM acceptance metric. Portable private-memory measurement is unavailable and reported as `null` with a reason. OS enforcement, complete allocation instrumentation, page-fault/I/O accounting and independent security/performance review remain separate work.

`observations.jsonl` contains per-operation latencies, epoch/cycle/phase identity, scheduler progress and memory samples. Summaries include counts, mean/max operation latency, sampled peak RSS, warmed drained growth, and internal high-water marks. These are observations under uncontrolled host/cache conditions, not controlled cold-cache or comparative benchmarks.

The observer uses a fixed write buffer and streams records; the controller summarizes them in constant space rather than retaining every sample. Individual records are capped at 2 MiB and the observation log at 256 MiB. Disposable model stores are removed between configurations and after failed driver processes exit; contracts, logs and evidence remain under `.cache/cpu-soak/REPORT_STEM/`. Test credentials are synthetic and never sent outside the in-process router.

## Pinned pretrained inputs

After acquiring the existing verified fixtures, run the separately scoped pretrained harness smoke:

```sh
python3 tools/run_cpu_soak.py \
  --profile fixtures/soak/cpu-pinned-smoke-v1.json \
  --models fixtures/soak/qwen35-nomic-models-v1.json \
  --report fixtures/qualification/cpu-soak-pinned-smoke-v1.json
```

This input set selects the pinned Qwen2.5-0.5B Q4_K_M text file, Nomic Q8_0 embedding file and Qwen3.5-0.8B image checkpoint with an authored PNG. All files must already exist; the runner performs no downloads. GGUF imports verify complete file hashes. Checkpoint import requires the exact pinned consumed-file inventory and verifies every file while streaming into private managed storage. The source assets remain unchanged.

Custom input sets use schema 1 with `kind: "pinned"`, `text` and `embedding` objects containing a local `path` and expected `sha256`. `vision` accepts either `kind: "checkpoint"` with `path`, the complete consumed `pinned_files` name/hash map and an `image` path/hash object, or `kind: "pair"` with `language`, `projector` and `image` path/hash objects. Architecture support is still determined by native validation, not this input file. Text-role file imports currently require compatible GGUF; the image checkpoint option supports the existing native Qwen3.5 profile.

Pretrained generation must actually reach the bounded event queue for backpressure tests. If a selected model stops early, the scenario fails explicitly; there is no hidden EOS override or synthetic replacement. The two-cycle pinned smoke verifies the harness path, not sustained pretrained qualification, all encodings/models or other hardware. A full release still requires reviewed representative workloads on every mandatory native platform. The [CPU release checklist](CPU_RELEASE.md) retains those open requirements; these reports do not automatically accept its resource-soak gate.

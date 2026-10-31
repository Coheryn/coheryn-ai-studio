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

## Shared modern checkpoint workload

The schema-2 [modern input set](../fixtures/soak/qwen35-shared-nomic-models-v2.json) assigns both text and image requests to one managed Qwen3.5-0.8B Q4_K_M/F16-vision checkpoint. Nomic Q8_0 remains an independent embedding model. The checkpoint inventory pins every consumed language, vision, tokenizer, template, configuration and license file. Its adapter still determines execution compatibility; role sharing does not make unsupported checkpoints executable. Resident baselines record the unique model IDs and image role, and the driver checks that exactly two models are loaded. The existing schema-1 inputs keep their three-resident behavior.

```sh
python3 tools/run_cpu_soak.py \
  --profile fixtures/soak/cpu-modern-shared-smoke-v1.json \
  --models fixtures/soak/qwen35-shared-nomic-models-v2.json \
  --report fixtures/qualification/cpu-modern-soak-pinned-v1.json \
  --timeout-seconds 7200
```

Schema 2 accepts exactly one `text` file or `text_checkpoint` source. `vision.kind: "shared_text"` requires the checkpoint source and supplies a separately hashed image. Optional `workload` fields freeze the text, backpressure and image prompts before execution; each is bounded to 4096 UTF-8 bytes. Image requests are classified by their content as well as their model role, so sharing an ID cannot hide image-operation coverage. Image bytes are hashed after the bounded read, and managed file imports verify the expected checksum rather than merely attaching it as provenance.

The modern smoke fixes two measured cycles, one fault-bearing warmup, a restart after each cycle, concurrency one and balanced execution. Engine reservations and sampled RSS each have a separate 1 GiB ceiling; drained RSS growth is limited to 128 MiB. Requests have a 1200-second bound, cleanup 30 seconds and consumer cancellation 60 seconds. Those values define this smoke workload, not a general service SLA. Prompts request sufficiently long output naturally; EOS behavior is unchanged and early completion still fails backpressure coverage. A passing smoke does not satisfy a sustained-soak or production-release gate.

The controller protects both its single-job optimized build and native driver with the [qualification host guard](CPU_QUALIFICATION.md). It checks available-memory, disk and load reserves, plus macOS pressure, compressed memory and thermal state, before starting and while running. Managed-copy disk headroom counts consumed physical files once for shared text/vision and retains an additional staging margin. Guard cutoffs, timeout and interruption stop only owned child process groups. Disposable stores are removed only after confirming all owned children have exited; cleanup failure retains the private stores and fails the run. Progress reports are written atomically; failed attempts and host observations remain available. The guard observes host conditions and cannot guarantee prevention of every OS/hardware fault. It does not change engine admission limits or make unobserved platform sensors available.

On 2026-10-31 the modern extension's controller checks passed, but two optimized test-build attempts and a metadata-only compiler attempt crossed the host compressed-memory cutoff. No new synthetic or pretrained native smoke was executed. The implementation is pending Rust compilation/tests and the two source-bound smoke runs above; previous native evidence remains attached to its original snapshot. Keep the cutoff intact and resume on a host with sufficient memory headroom.

## GitHub Actions execution

The [Opt-in modern CPU soak workflow](../.github/workflows/cpu-soak.yml) prepares independent native jobs on macOS ARM64, Linux x86-64 and Windows x86-64. It is separate from the numerical qualification workflow and needs no reference-data archive. Native runs remain pending until the workflow actually executes and its reports pass review. Configuration does not register release evidence.

Once the workflow is committed and available on the repository's default branch, select **Actions → Opt-in modern CPU soak → Run workflow**, choose the source branch and keep `modern_models` enabled for pretrained coverage. Disabling that input runs only the Rust/controller checks and synthetic operational smoke. Each platform uses its own runner; independent jobs can run concurrently. No model weights, tokens or private managed stores are uploaded as evidence.

For qualification before merging, pushing a branch under `qualification/cpu-soak/` also starts the workflow, with modern models enabled. This opt-in branch trigger works through ordinary Git SSH access without a separate API credential. Other branch pushes do not start this workflow. The existing general CI still runs under its own triggers. Every report remains bound to the submitted source; publishing a branch or starting a run does not establish a passing result.

The job verifies native runner identity, checks host health, runs controller regressions and then executes:

```sh
python3 tools/check_cpu_soak.py --fetch-only --report .cache/cpu-soak-dependencies.json
python3 tools/license_notices.py --tracked
python3 tools/check_cpu_soak.py --report .cache/cpu-soak-checks.json
python3 tools/run_cpu_soak.py --profile fixtures/soak/cpu-smoke-v1.json \
  --report .cache/cpu-soak-synthetic.json --timeout-seconds 300
python3 tools/prepare_cpu_soak.py --fetch --download-attempts 2 --report .cache/cpu-soak-inputs.json
python3 tools/run_cpu_soak.py --profile fixtures/soak/cpu-modern-shared-smoke-v1.json \
  --models fixtures/soak/qwen35-shared-nomic-models-v2.json \
  --report .cache/cpu-soak-modern.json --timeout-seconds 7200
```

These commands also work on another supported development machine. Omit `--fetch` when all pinned inputs are already cached. Preparation acquires only the Q4_K_M language file, F16 vision component, required source sidecars and Nomic Q8_0 file; it does not download other encodings or the full source-weight checkpoint. It checks the exact frozen soak inventory with bounded streaming reads and verifies pending managed-copy disk headroom. Preparation is byte validation, not inference qualification.

The hosted preparation explicitly permits at most two attempts for a transient timeout, connection failure or selected retryable HTTP status. Every attempt and the active pinned input are retained in the report. Backoff defaults to one second; a numeric server `Retry-After` is honored up to the five-second bound, and unsupported/longer delays fail rather than being ignored. Completed cached files are reverified, unpublished partial files are removed, and the existing download size, socket timeout, per-file deadline, atomic publication and host limits remain enforced. Integrity failures, unsafe paths, certificate failures and host rejections do not retry. The default remains one attempt, and native inference or failed qualification scenarios are never retried automatically.

The guarded Rust checker records actual Cargo/compiler versions, the native target, source/lockfile/input identities, complete diagnostic logs and test totals. Empty, filtered or failed test runs cannot pass. Both tests and strict Clippy use the workspace dependency feature set, retaining cache reuse with workspace builds. All builds use one Cargo job. A failed preflight starts no compiler; cutoff, interruption and incomplete shutdown preserve failed reports. Concurrent shutdown calls cannot signal a successfully retired owned group twice. Permission failures and Windows tree-termination failures remain explicit; they never imply confirmed descendant cleanup.

Fresh runners first use `--fetch-only` to acquire both locked workspace and fuzz dependency graphs under the same guard. The offline license checker needs those registry manifests even before any compilation. Dependency preparation has its own report kind and cannot count as passing Rust or model checks. It does not modify lockfiles or regenerate legal records.

Legal checks also require every indexed source asset and notice in Git. A locally ignored provenance file cannot conceal a missing clean-checkout input. The small original LLaVA configuration and its provenance are explicitly tracked with unchanged publisher bytes; broad resource exclusions still protect private specifications and model caches.

The first hosted Mac preflight recorded load 30.64 on three CPUs while memory, disk and thermal observations were within limits. A subsequent 180-second idle observation recorded load declining from 41.26 to 7.69, still above the unchanged cutoff of 6. That failed record is preserved. The workflow now uses the tool's bounded maximum of 300 seconds for startup load to settle. Every rejected sample is retained; other violations or malformed measurements fail immediately. No compiler or model starts during that window. This does not retry failed inference, relax a cutoff or change the continuously enforced guard after admission. Failure to reach a healthy startup state still fails the job.

Every job uploads reports and observer/driver logs even after failure. Artifact names include the target and run attempt. Failed-run stores are excluded from upload, including when cleanup cannot be confirmed. Source changes, missing fault coverage, early EOS or exceeded limits fail the run; profiles are never relaxed automatically. A passing two-cycle smoke remains separate from sustained resource, independent numerical and production-release acceptance.

A later workspace-feature test build on the local Mac also hit the unchanged compressed-memory cutoff. The cache-only preparation verified all ten selected file hashes before its final host check failed. Those observations do not establish successful compilation or native smoke execution; the user selected GitHub Actions for the remaining native work.

## Verified hosted smoke checkpoint

[The recorded campaign](../fixtures/qualification/cpu-github-modern-smoke-v1.json) completed on macOS ARM64, Linux x86-64 and Windows x86-64 in [GitHub Actions](https://github.com/Coheryn/coheryn-ai-studio/actions/runs/41155745636). It binds source commit `665d564960d184bad81d3abbf3db094cab94c04c`; later tooling changes do not relabel those reports as current-source qualification. Original check, synthetic and modern report bytes are preserved beside the index. Artifact ZIP hashes were verified against official metadata, every declared report artifact was checked, and observer summaries were recomputed before recording these results.

| Native target | Driver tests / strict Clippy | Synthetic smoke | Modern shared smoke | Sampled peak RSS / warmed drained growth |
|---|---|---|---|---|
| macOS ARM64 | Pass | 2 measured cycles | 2 measured cycles | 305.98 / 55.30 MiB |
| Linux x86-64 | Pass | 2 measured cycles | 2 measured cycles | 396.45 / 0.00 MiB |
| Windows x86-64 | Pass | 2 measured cycles | 2 measured cycles | 150.97 / 1.62 MiB |

Each modern run completed two service epochs, six text requests, six embedding operations and four image operations. All four required fault scenarios occurred twice during measured execution; cleanup and unchanged source were confirmed. Each run used the frozen balanced, concurrency-one profile with shared Qwen3.5 text/vision and independent Nomic embeddings. These are separate host observations under uncontrolled cache conditions, not cross-platform performance comparisons. RSS sampling is not a complete physical-allocation measurement.

This closes the selected operational smoke execution checkpoint. It does not close the 400-cycle synthetic campaign, sustained pretrained workloads, numerical/platform matrix, TCP/security/package review or production approval. The independent numerical workflow still needs the exact frozen reference archive at an approved HTTPS location. CPU support remains experimental.

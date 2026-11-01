# CPU release readiness

The immediate priority is qualifying the existing native CPU implementation before expanding architectures or GPU execution. It remains experimental. A CPU candidate is a narrower distribution than the full mandatory release described in [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md); all full-release requirements remain in force.

## Candidate scope

[The machine-readable scope](../fixtures/cpu-release/scope.json) identifies the required native targets and existing text, embedding and vision profiles. The detailed file identities, encodings, contexts, templates and numerical evidence remain in [the compatibility matrix](../fixtures/compatibility.json). Container validity and architecture compatibility do not establish model quality or production qualification. No universal model, language, image or GGUF claim is made.

Qualification must cover macOS ARM64, Linux x86-64 and Windows x86-64. CI configuration alone does not count as execution evidence. Passing synthetic tests and native executable startup do not establish model-level numerical parity. Current independent numerical model evidence is scoped to macOS ARM64. The [selected modern operational smoke](CPU_SOAK.md#verified-hosted-smoke-checkpoint) has passed on all three native targets; it checks repeatability and lifecycle behavior, not independent numerical parity. Reference and numerical limitations in the individual task guides still apply.

[Verified native candidate evidence](../fixtures/qualification/cpu-github-native-candidates-v1.json) now records all three targets through ordinary tests, release compilation, legal/policy checks, package integrity and extracted startup. [Repeated modern smoke evidence](../fixtures/qualification/cpu-github-modern-smoke-v2.json) separately records selected text/image/embedding lifecycle checks. Both retain their original source revision and explicit limits; neither accepts the sustained, numerical, security or release-review gates.

## Required evidence

1. Warning-free formatting, Clippy, ordinary tests, crash recovery and release compilation on each native target. Record the exact source, lockfiles, compiler, OS, CPU and executable identities.
2. Frozen-tokenizer, intermediate-array, final-vector/logit and greedy-output comparisons for each advertised exact-file profile and encoding. Preserve independent references and their precision thresholds. Exercise CLI, API, streaming, embeddings and managed image/text histories, including multiple images and supported resolutions/formats.
3. Mixed-task sustained load, cancellation/disconnect, slow consumers, queue saturation, unload/restart and repeated failure tests. Capture allocation reservations, measured buffer capacities and observable RSS/private memory separately. Establish and review explicit memory/latency acceptance thresholds before approving results.
4. Instrumented parser/decoder/tool/constraint fuzzing, resource-exhaustion and fault tests, dependency advisories, authentication/path/download review and complete legal notices. Bounded fuzz campaigns provide useful evidence but do not close sustained security qualification.
5. Verified offline archives, actual model inference from extracted binaries, target-specific signing/provenance, supported OS baselines and upgrade/rollback review. A checksum distributed beside a package detects corruption; authenticity requires an independently trusted signature or provenance channel.
6. A clean, stable release source revision and human review of numerical, security, operational and licensing exceptions. A development version or dirty source tree cannot be promoted by renaming an archive.

The readiness command reports missing evidence and exits unsuccessfully until the required records are accepted:

```sh
python3 tools/check_cpu_release.py --report target/cpu-readiness.json
```

It checks the scope/evidence file bindings and current source identity. It does not certify externally supplied evidence or replace independent review. Requirements with no implemented evidence validator remain blocked; an arbitrary `pass` field never qualifies them.

## Local hardening checks

The [pinned real-model qualification runner](CPU_QUALIFICATION.md) combines frozen numerical comparisons, extracted-candidate CLI and real authenticated TCP checks. Its opt-in workflow targets all three native CPU platforms. It covers an explicit initial subset and requires the exact hosted reference-data bundle; remote jobs and the complete model matrix remain pending until actually executed and reviewed.

The [mixed CPU workload soak](CPU_SOAK.md) adds fixed-limit text/image/embedding stress, application backpressure, queue/failure cleanup, model reload and durable service-restart replay. Synthetic operational coverage and pinned pretrained smoke are recorded separately. They do not automatically close sustained resource or platform qualification.

```sh
./tools/cargo fmt --all -- --check
./tools/cargo clippy --release --workspace --all-targets --locked -- -D warnings
./tools/cargo test --release --workspace --locked
./tools/cargo build --release --workspace --locked
python3 tools/test_cpu_release.py
python3 tools/license_notices.py
python3 tools/check_independence.py
```

`crash_recovery` kills a separate process after a committed receipt or while a durable writer is pending. Reopening must preserve completed output/idempotency and recover the interrupted writer as cancelled, without loading a model. This tests process termination, not power-loss durability or filesystem/hardware failure.

`tools/run_cpu_checks.py --report FRESH_PATH` executes the complete check inventory under the live host guard with one offline Cargo job and one test-harness thread. Explicit concurrency workloads inside individual tests remain unchanged. Release-profile lint/tests reuse the production dependency graph and exercise deployed arithmetic without building a second debug executable graph. Compiler versions, logs, test totals, execution settings, owned cleanup and host observations are retained; rejected or interrupted checks cannot pass.

Known desktop metadata (`.DS_Store`, `Thumbs.db`, `desktop.ini`) is excluded from source fingerprints. Other hidden and unknown fixture files remain bound, and model/reference inputs still have their explicit hash checks. Earlier snapshots retain their original scope and bytes.

Source fingerprints also bind the root `NOTICE`, Git attributes, local Cargo build configuration when present, and every file in the mirrored legal payload. Adding, removing or changing a release notice invalidates earlier source-bound evidence. This extends the fingerprint contract; earlier reports remain attached to their recorded scope and are not relabeled for the stronger contract.

Run real-model opt-ins with `cargo test --release` and keep their reports separate from ordinary-test totals.

Install the latest verified `cargo-fuzz` development tool and pin a sanitizer-capable nightly independently of the production stable toolchain. Record both versions. The bounded campaign runner retains corpus identities, actual coverage, iteration counts, commands, source hashes and log digests:

```sh
python3 tools/run_cpu_fuzz.py --toolchain nightly-2026-10-30 \
  --runs 100000 --report target/cpu-fuzz-FRESH.json
```

Use a fresh report path for each run. Reproducing a campaign requires the recorded seed corpus; mutations and coverage may vary across compilers/platforms. The Safetensors target uses the file loader's shared header validator. The image target covers all three implemented preprocessing profiles. Inputs, output tensors, libFuzzer allocations and per-input time are bounded. Reference files remain immutable.

The runner now applies the unchanged host guard to compiler/tool probes, optional preparation and every campaign. It uses one Cargo job, dedicated owned process groups, per-command deadlines, bounded diagnostic logs and atomic progress checkpoints. `--prepare` explicitly permits guarded installation of the workspace-pinned stable compiler, the pinned nightly and verified stable `cargo-fuzz` 0.13.2, plus locked dependency acquisition; the default requires installed tools and cached dependencies. An unsafe initial host starts no installation, compiler or seed work. Interruptions, failed coverage, source changes or unconfirmed cleanup remain failures with partial evidence preserved. Corpus inventories reject links and excessive bytes/counts; authored seeds are never overwritten.

The [sanitizer workflow](../.github/workflows/cpu-fuzz.yml) executes this path on Linux when its tooling/targets change or on manual dispatch. It retains original diagnostics, corpora and the complete legal payload. Controller regressions simulate admission, interruption and evidence failures without a compiler; only an executed sanitizer report establishes a campaign result. Earlier `cpu-hardening-fuzz-v*` records predate this guard and remain historical evidence.

The [reviewed initial guarded campaign](../fixtures/qualification/cpu-guarded-fuzz-v1.json) completed 500,000 inputs across GGUF, Safetensors, images, tools and constraints at revision `4b5717b`. Official artifact/source identities, all preparation/campaign logs, coverage counts, initial/final corpora, complete legal payload and every healthy host observation were verified. The later legal/build-policy fingerprint extension requires its own source-bound rerun. These bounded Linux campaigns do not close sustained security, API/authentication/path/download, other-platform or independent release review requirements.

Run `cargo audit` against both workspace and fuzz lockfiles with the current advisory database. Capture the database revision and every warning. The current upstream tokenizer uses the build-time `paste` macro dependency, which has unmaintained advisory `RUSTSEC-2024-0436`; retaining the latest tokenizer does not resolve this warning. It requires an upstream fix or a reviewed replacement/exception before release approval. Do not globally ignore advisories or silently patch numeric/tokenizer behavior.

## Candidate packaging

Run the source-bound checks on the declared native target first, then:

```sh
python3 tools/run_cpu_checks.py --report target/cpu-native-checks.json
python3 tools/package_cpu_release.py --target aarch64-apple-darwin \
  --build-record target/cpu-native-checks.json \
  --report target/cpu-package.json
python3 tools/verify_cpu_package.py target/cpu-release/ARCHIVE.tar.gz \
  --sha256 EXPECTED_SHA256 --report target/cpu-package-smoke.json
python3 tools/check_cpu_package_models.py --package-report target/cpu-package.json \
  --report target/cpu-package-models.json
```

Select the matching Linux or Windows target when applicable. The package contains both executables, public Markdown documentation, the scope/compatibility matrices and the complete original legal payload. It excludes model weights, caches, private specifications and reference executors. Archives have normalized ordering, ownership, permissions and timestamps; identical inputs must serialize identically. This is archive reproducibility, not proof of reproducible compiler output.

Verification rejects unsafe paths/links, duplicate entries, oversized archives, undeclared or tampered files, missing original legal bytes and mismatched executable targets before running anything. Native startup checks run with an empty executable search path and no credentials. Model downloads, model-level offline inference, clean-runner results, signing and release approval are separate requirements. The CI artifact workflow produces candidates only and does not publish a production release.

See [OPERATIONS.md](OPERATIONS.md) for local deployment and recovery procedures.

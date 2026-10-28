# Dependency maintenance

The policy in [AGENT.md](../AGENT.md) requires the latest stable, non-yanked project packages. Versions were checked against crates.io, PyPI and the relevant official tool/action releases on 2026-10-26. All 21 directly managed external Rust packages now specify their latest stable release, and both the workspace and separate fuzz lockfiles have been refreshed.

| Component | Previous resolved/pinned version | Updated version |
|---|---|---|
| Rust toolchain | 1.90.0 | 1.99.1 |
| Reqwest | 0.12.28 | 0.13.5 |
| SHA-2 | 0.10.9 | 0.11.1 |
| Tokenizers | 0.22.2 | 0.23.2 |
| MiniJinja | 2.24.0 | 3.0.3 |
| Sysinfo | 0.37.2 | 0.39.6 |
| CMake development package | 3.31.6 | 4.4.4 |
| actions/checkout | v4 | v7.0.1 |
| actions/setup-python | v5 | v7.1.0 |

The remaining direct Rust packages also have their latest full version in the manifests, including packages whose lockfile was already current. Jinja2 3.1.6 and MarkupSafe 3.0.4 were already current and remain pinned. CI asks setup-python for the latest available stable Python 3 release with `check-latest: true`; hosted interpreter distributions can follow an upstream security release after its publication.

Reqwest now enables its `rustls` and `query` features explicitly. SHA-2 output uses a shared fixed-size lowercase hex encoder, preserving existing file/header/template fingerprints. Tokenizers' added-token registration consumes owned tokens and propagates registration failures. MiniJinja enables Serde support and wraps the message context explicitly, preserving prompt bytes, fuel, recursion and output limits.

## Transitive constraints

Latest direct packages do not imply that every indirect package can use its newest major version. Cargo selects the newest graph permitted by upstream dependency requirements. For example, the latest Tokenizers still depends on `spm_precompiled`, which requires Base64 0.13; its derive helpers also require Syn 2. The graph contains current Base64/Syn versions alongside those older required majors. Replacing those versions directly would violate upstream requirements and can break compilation or public types.

[The machine-readable dependency qualification record](../fixtures/qualification/dependency-upgrade.json) lists all older transitive versions found in the refreshed registry index, their newer stable releases and the affected lockfile. These remain upstream constraints rather than outdated direct project pins. Inspect a particular path with:

```sh
./tools/cargo tree --locked --invert base64@0.13.1
./tools/cargo tree --locked --invert syn@2.0.119
```

Models, reference snapshots and historical qualification records are immutable test inputs, separate from the application's package dependency graph. This update preserves their bytes and numerical criteria. Existing independent captures are hash-checked before being reused to test the upgraded native binary; expected outputs are not regenerated from that binary. Reference inference engines remain outside production dependencies.

## Repeat locally

Verify current upstream stable releases before editing dependency requirements; `cargo update` alone will not cross incompatible version ranges. Then refresh both graphs and run the checks:

```sh
./tools/cargo update
./tools/cargo update --manifest-path fuzz/Cargo.toml
./tools/cargo fmt --all -- --check
./tools/cargo clippy --workspace --all-targets --locked -- -D warnings
./tools/cargo test --workspace --locked
./tools/cargo check --manifest-path fuzz/Cargo.toml --locked
./tools/cargo build --release --workspace --locked
python3 tools/check_embedding_fixtures.py
```

All installations and verification in this refresh are local to the workspace/cache. No release was published or pushed. New packages have not expanded the engine's supported architectures or backend/platform qualification.

## E1 Unicode normalization dependency

E1 adds `unicode-normalization` **0.1.26**, the latest stable non-yanked registry release verified on 2026-10-26 (published 2026-10-16). Verification used the refreshed Cargo sparse registry index and resolution. It supplies general-purpose canonical decomposition for the qualified GGUF WordPiece adapter; it is not an inference runtime. Both workspace and fuzz lockfiles include it. Existing direct versions remain unchanged.

## E2 dependency verification — 2026-10-27

All direct Rust packages, the stable Rust channel, optional Python clients, existing Python development tools and CI actions were checked against their official release registries. Rust remains 1.99.1. MiniJinja updates to **3.0.4** in workspace/fuzz manifests and lockfiles. Base64 **0.23.1** is now a direct API dependency; Tempfile **3.27.0** is also a CLI production dependency for private atomic exports. Existing direct packages and CI/devtool pins otherwise remain current. `fixtures/qualification/e2-dependencies.json` records the verification.

OpenAI Python **3.39.0**, LangChain OpenAI **1.7.3** and Qdrant Client **1.20.0** are optional interoperability clients in `tools/requirements-rag.txt`; they are installed in an isolated environment/cache and are never linked into the Rust engine. No external inference runtime entered the production graph. Historical oracle/model snapshots and their expected outputs stay pinned.

## S0/S1 dependency verification

Official release registries were rechecked for all 23 directly managed Rust packages and the stable toolchain. Existing pins remain current; this implementation adds no dependency. `fixtures/qualification/s0-dependency-versions.json` records the upstream results. Source/path policy and both locked dependency graphs are checked separately by `tools/check_independence.py`. Original external capture tooling is retained as inert research material under the resource archive and is excluded from native build/test/runtime chains.

## S2/S3 dependency verification — 2026-10-27

All 24 direct Rust packages and stable Rust were rechecked against their official registries. Rust remains **1.99.1** and existing pins remain current. S3 adds **Rusqlite 0.41.0**, with bundled SQLite and the limits feature, exclusively to the API crate. It provides transactions and bounded local history storage, not model execution. The Rust engine has no database dependency. The bundled C database requires the normal native C build toolchain; the package/feature/license inventory includes its transitive dependencies.

`fixtures/qualification/s2-s3-dependency-versions.json` records current direct-package/toolchain verification. Locked all-feature graphs, archive integrity and executable strings are checked separately; inventory is not a completed license audit. Immutable reference/model inputs remain unchanged.

## S4 dependency verification — 2026-10-27

Official registries were rechecked: all 24 directly managed Rust packages and Rust 1.99.1 remain current. No new production package is introduced. The existing Serde JSON pin enables `raw_value` to preserve tool argument numbers, whitespace and escaping. The separate fuzz harness reuses that same current package. OpenAI Python 3.39.0 remains current and is used only as an optional local protocol client, outside the Rust graph. See `fixtures/qualification/s4-dependency-versions.json`. Research captures stay pinned and separate from native checks.

## S5 constraints and sampling

Official registries were rechecked for all 24 directly managed Rust packages and the stable Rust toolchain; existing Rust pins remain current. S5 introduces no engine dependency. The optional standard Python protocol client is updated to OpenAI 3.40.0 after the official registry check; HTTPX 0.28.1 remains current. These clients stay outside the production Rust graph. See `fixtures/qualification/s5-dependency-versions.json`; source/model/reference licenses and immutable records retain their original bytes.

## S6 dependency verification

The official registry refresh again confirms Rust 1.99.1 and all 24 directly managed Rust packages are current; S6 adds no Rust dependency. `fixtures/qualification/s6-dependency-versions.json` preserves the check. Model/reference snapshots and their original legal bytes remain unchanged. Local debug/test verification disables debug symbols and incremental artifacts through Cargo environment overrides to limit rebuild storage after cleanup; production release settings are unchanged.

## M1 image codec dependency — 2026-10-28

`image` **0.25.10** is the latest stable non-yanked release verified from the official registry. Only its PNG/JPEG codec features are enabled; resize, normalization, patch layout, admission and all model execution remain Coheryn-owned. Both lockfiles and the complete legal inventory are refreshed. The 25 direct Rust packages and Rust 1.99.1 are current in `fixtures/qualification/m1-dependency-versions.json`.

Pillow **13.0.0** is used only in the isolated resource assessment environment for independent decoding and resize comparisons. It is not a native dependency or fallback. Original model/reference/license bytes remain unchanged. Development checks continue to disable debug symbols and incremental artifacts locally to limit disk use; release settings are unchanged.

## M0 reference and position qualification

No new Rust dependency is added. Official registries again confirm all 25 direct Rust packages and Rust 1.99.1 are current in `fixtures/qualification/m0-reference-dependency-versions.json`. Isolated assessment-only requirements pin NumPy 2.5.4, Pillow 13.0.0, Jinja2 3.1.6 and MarkupSafe 3.0.4; the independent tool record also verifies CMake 4.4.4 and pip 26.2.2. These packages and corrected reference execution remain outside both Cargo graphs and native CI/runtime. Original numerical inputs are retained; corrected outputs are separate versioned records under unchanged precision criteria.

## M2 native vision component

The native Qwen encoder/merger adds no Rust dependency. `fixtures/qualification/m2-vision-dependency-versions.json` again verifies Rust 1.99.1 and all 25 direct Rust packages against their official stable/non-yanked releases. Existing model/reference snapshots remain pinned. Native math, buffered weights, worker pools and resource accounting reuse Coheryn modules; independent capture tools remain outside production and CI execution.

The offline Qwen language integration adds no Rust dependency. `fixtures/qualification/m2-offline-dependency-versions.json` verifies the same 25 direct stable/non-yanked packages and Rust 1.99.1. The isolated reference metadata correction and independent first-layer array assessment remain outside both Cargo graphs and native CI execution; their original inputs/licenses and prior captures are preserved.

## Product image CLI/API integration

The runtime reuses existing current Rust libraries for HTTP, canonical Base64 and serialization; image execution remains Coheryn-owned. API/CLI development fixtures now also declare the existing `half` 2.7.1 package directly to construct small independently authored F16 components. No new external package enters either lockfile. `fixtures/qualification/m3-service-dependency-versions.json` verifies Rust 1.99.1 and all 25 direct stable/non-yanked Rust package releases against official registries on 2026-10-28. The legal inventory is regenerated for the updated direct development edges while preserving all license and original reference bytes.

## Managed image histories and cooperative scheduling

This checkpoint adds no dependency. Private original assets, canonical portable encoding, history persistence and FIFO checkpoints reuse current workspace libraries and Coheryn numeric/storage code. `fixtures/qualification/m4-session-dependency-versions.json` verifies Rust 1.99.1 and all 25 direct Rust package releases against the official registries. Existing immutable model/reference/license bytes and thresholds are preserved.

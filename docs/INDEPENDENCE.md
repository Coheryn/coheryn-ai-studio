# Native execution and repository policy

Distribution notices are documented in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). Artifact scans permit only exact-path, byte-identical files from that source-controlled legal payload; they do not exempt arbitrary resource files. The license checker verifies locked dependency coverage, mirrored asset attribution and original archive digests. Required notices remain intact.

Coheryn executes supported models through its own Rust graphs and kernels. General-purpose codecs, tokenizers, templates, HTTP, cryptography and hardware APIs may be used. External model executors cannot serve requests through a dependency, FFI, subprocess, service or optional fallback.

The resource policy catalog controls names and package patterns. `tools/check_repository_policy.py` scans authored source, hidden configuration, paths and ignored documentation. Only repository-root `resources/` descendants are a textual exception. Cache/build exclusions are explicitly classified in its report; they are never permitted distribution inputs. Symlinks, unreadable text and unclassified binary content fail rather than silently disappearing from the inventory. Release staging must be supplied separately with `--artifact DIRECTORY` to inspect controlled embedded strings as well as files.

Executable scans inspect UTF-8 and UTF-16 in data and metadata. For bounded [ELF64](https://gabi.xinuos.com/elf/03-sheader.html), [Mach-O64](https://github.com/apple-oss-distributions/xnu/blob/main/EXTERNAL_HEADERS/mach-o/loader.h) and [PE](https://learn.microsoft.com/en-us/windows/win32/debug/pe-format) section tables, text inspection excludes declared instruction sections: operands can coincide with short ASCII labels or normalize into unrelated Unicode labels. Unknown formats receive full scanning; malformed recognized section tables fail the check. Reports record the excluded instruction byte ranges. Labels in data and metadata still fail; inline instruction bytes are outside this text-inspection scope. This is a string inspection with a declared scope, not disassembly or proof of dependency absence; the locked dependency and runtime checks provide separate evidence. Original executable bytes, the pattern catalog and required notices remain unchanged.

Original external reference builders/captures are retained as inert `.txt` research records in `resources/reference-archive/`. They are removed from the native tool/build/test chain. Historical manifests and reports are retained byte-for-byte, including notices and provenance. Active manifests retain model identities, thresholds and frozen outputs, with neutral references to original research. Mirrored vocabulary/configuration assets containing policy collisions are relocated intact; token IDs and source digests remain unchanged. New comparative assessments run separately and supply bounded verified expected data; the native engine never regenerates its own oracle to pass a check.

```sh
python3 tools/check_repository_policy.py --report resources/policy/source-findings.json
python3 tools/test_repository_policy.py
python3 tools/check_independence.py
python3 tools/check_embedding_fixtures.py
./tools/cargo test --workspace --locked --offline
```

The dependency check inventories the locked all-feature graph and declared licenses. Native CLI regression tests run with an empty executable search path and no service endpoint, using small complete model fixtures. Actual model tests use Coheryn plus previously acquired weights and inert expected outputs. These checks are separate evidence: a zero-match source scan does not prove runtime independence, performance, legal compliance, remote deployment provenance or complete release qualification. Final package strings, platform binaries, security campaigns, complete license review and signed/checksummed releases remain G6 obligations. Required protocol identifiers and legal notices must never be hidden, rewritten or stripped to obtain a passing result.

S2/S3 add only a general-purpose embedded database utility in the API layer. The bundled SQLite implementation never executes model graphs and is absent from the engine dependency graph. `s2-s3-independence.json` records the updated locked inventory; `s2-s3-artifacts.json` identifies current native executables. Prefix/persistence model checks use the same native binary and preserved expected data, without an external inference service or fallback.

S4 tool execution consists of Coheryn's template/marker/JSON protocol and existing native decoder graph. The optional standard SDK is a local test client; it executes no model. `tools/check_tool_calling.py` reads inert expected outputs and runs only the native binary/local daemon. Independently executed research capture sources remain under `resources/tool-assessment/`, outside native build/test/runtime. `s4-independence.json` and `s4-artifacts.json` record current graph/archive and executable checks.

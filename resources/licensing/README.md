# Licensing and distribution records

Coheryn-authored code is Apache-2.0. This directory records separate third-party terms; it does not replace them or assert that every included item is Apache-licensed.

## Source assets

`assets.json` maps each mirrored source asset to its publisher, revision, SHA-256 and accompanying license records. Original reference/archive files remain byte-for-byte unchanged. Copied templates, tokenizer/vocabulary and model documentation retain their upstream terms. Model weights are acquired separately and are not included in the repository or notice bundle.

The Qwen companion template carries a non-rendering in-file modification notice. Its annotated file identity differs from the frozen reference body; `fixtures/tools/manifest.json` records both. The frozen independent prompt/token/output records and historical qualification reports are unchanged. Existing native prompt-parity tests verify the legal comment does not change model input; old binary hashes identify the original binaries, not a newly rebuilt executable.

Nomic's publisher declares Apache-2.0. Its model card, exact source assets and upstream BERT license record are retained. The copied Qwen-VL license includes Alibaba Cloud's copyright notice. New upstream NOTICE files or additional copied components must be preserved and added to the inventory before redistribution.

## LLaVA research assets

The selected publisher declares `llama2` for LLaVA 1.5, not Apache-2.0. The original publisher model card, official Llama 2 Community License and acceptable-use policy are retained under `upstream/`. The required attribution is in `fixtures/vlm/llava-1.5-7b/Notice` and in this legal payload's `Notice`.

The custom agreement includes use restrictions and additional commercial conditions. It must not be advertised as an unrestricted Apache-licensed model. Included material is limited to configuration/processor/template records and derived tensor inventories. Weights, tokenizer model and full vision-projector payloads are not shipped. Future execution or redistribution of those components requires reviewing their full chain of terms, including base models and vision assets. No source notice grants an exception to those terms.

## Reference tools

The references to llama.cpp and Ollama identify independent research/comparison tools. Their original MIT terms are preserved under `upstream/` for provenance; no engine source, headers, library or executable from those projects is shipped or linked into the native runtime. Project-authored capture wrappers remain inert archived research. Merely referring to a comparator does not relicense it or imply endorsement.

## Dependencies and binary packaging

`dependencies.json` inventories the complete locked all-feature/all-target workspace and fuzz graphs, including build/test packages. Collection verifies each downloaded `.crate` archive against its Cargo.lock SHA-256, then retains all separately named license/copyright/notice/author files, including nested vendor material. Identical legal texts are stored once under content-addressed `texts/` names, with original paths and package associations retained. Original authors and SPDX expressions, including AND/OR terms, are preserved.

Some crate archives omit license texts. `upstream/sources.json` records supplementary documents from exact source commits. The Objective-C binding projects provide `LICENSE.md` with links to standard MIT terms; both are retained along with their authors verbatim. Those upstream documents also discuss Apple SDK-derived bindings and their licensing caveat. This inventory preserves that disclosure; it cannot resolve upstream ownership questions or replace the SDK's terms. Parent-project MIT notices are retained for Windows import-library crates whose archives omit them.

The notice set conservatively includes all inventoried targets/features rather than guessing what a particular binary links. It includes bundled cryptography, Unicode and certificate-data notices. It is a distribution input, not an SPDX compatibility solver, legal opinion or assertion that every transitive native header has received manual review. Existing final platform/package/license release gates remain applicable.

`toolchain.json` also retains the installed pinned Rust distribution's complete standard-library copyright inventory and its referenced license texts. This broad inventory includes material that may not be linked into a particular application. It does not redistribute the compiler or establish rights to separate system/SDK redistributables. Changing `rust-toolchain.toml` requires regenerating these records along with dependency notices.

```sh
# Check committed legal evidence offline, after Cargo dependencies are fetched.
python3 tools/license_notices.py
# After dependency changes, regenerate from the acquired locked registry archives.
python3 tools/license_notices.py --collect
# Copy the complete legal payload into the actual release staging directory.
python3 tools/license_notices.py --bundle target/distribution
```

Ship `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md` and the complete `resources/licensing/` directory. Source archives also retain the asset-local LLaVA Notice. The copied legal payload includes the same attribution for binary archives. Installers must install these records rather than merely leaving them in a build directory. Optional Python clients and model/comparator downloads are separate installations, not vendored dependencies.

Required notices are never stripped to pass a presentation check. Artifact scanning permits only byte-identical files from the source-controlled legal payload; modified or additional files remain subject to ordinary scanning.

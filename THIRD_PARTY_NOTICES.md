# Third-party notices

Coheryn-authored code uses [Apache License 2.0](LICENSE). Third-party material retains its original terms; this file does not relicense model assets or dependencies.

The [legal records](resources/licensing/README.md) include:

- A [file-level asset inventory](resources/licensing/assets.json) for mirrored documentation, configuration, tokenizer/vocabulary and chat templates, including preserved reference assets.
- The [locked dependency inventory](resources/licensing/dependencies.json), with original license, copyright and notice texts, including nested vendor records. Its scope includes all targets/features and development/fuzz packages; inclusion does not imply every package is linked into every binary.
- [Supplemental upstream records](resources/licensing/upstream/sources.json) pinned to source revisions or explicitly identified publisher snapshots.
- The [pinned Rust standard-library notices](resources/licensing/toolchain.json), including the original copyright inventory and associated license texts.

The LLaVA research configuration is covered separately by its publisher-declared custom model terms. It is not represented as Apache-licensed Coheryn code. No model weights or external comparator binaries are distributed here. Small procedural image inputs are original project fixtures; their provenance is in `fixtures/vlm/images/manifest.json`.

Retain `LICENSE`, `NOTICE`, this file and the complete `resources/licensing/` directory when distributing the source or native binaries. Run `python3 tools/license_notices.py` after dependency/asset changes. Stage these notices for a native release with `python3 tools/license_notices.py --bundle target/distribution`; include the staged files in the actual archive or installer.

This inventory records license evidence and packaging obligations. It does not grant rights to third-party trademarks or certify ownership of contributed code. Optional Python clients and externally acquired models/tools are separate installations with their own terms.

# Native model adapters

Standalone text `run` and interactive `chat` now load through the engine's `adapters::Registry`. The registry selects a compiled native factory using bounded format/configuration inspection. Callers use `TextModel` for rendering, generation, cancellation and token events. Existing Qwen2 GGUF execution and the initial [Qwen3.5 CPU text adapter](QWEN35.md) use that interface.

A compatible new checkpoint can reuse an implemented adapter when its configuration, tensor inventory, tokenizer/template and requested features pass validation. A new architecture needs its own graph/bindings and any missing operations. Adding that implementation and registering its factory avoids rewriting these CLI flows. Arbitrary future architectures cannot become executable from weight metadata alone.

```text
checkpoint -> bounded format/configuration inspection
           -> registered TextAdapter -> validated native TextModel
           -> resource admission -> shared tensor operations -> token events
```

To extend the workspace:

1. Implement `TextAdapter` with a stable profile ID, bounded architecture recognition and strict source validation. Its metadata-only `describe` hook supplies validated configuration, capabilities and a pure cost estimator; it must not load an execution graph.
2. Implement `TextModel`, reusing checked storage, native operations and memory/cancellation mechanisms. Declare per-task/backend capabilities accurately.
3. Register the factory once in `Registry::default()` or construct a registry explicitly with `Registry::new()` and `register()`.
4. Freeze independent reference inputs and numerical limits, then test model semantics, cleanup and unsupported configurations.

Duplicate registration, ambiguous claims, unknown architectures and selected-adapter failures are explicit errors. Selection never retries another executor after a compatibility or numerical failure. Model packages cannot register code, download plugins or execute scripts. New compiled adapters require a rebuild; configuration changes within implemented semantics can load at runtime. The Rust interfaces remain workspace development contracts.

This is the first A0/A1 integration slice. Local `models inspect/validate` and explicit standalone text planning now share an immutable model descriptor; see [MODEL_METADATA.md](MODEL_METADATA.md). The daemon's managed residents, embedding execution interface and multimodal scheduler retain their existing typed interfaces. Migrating those callers, common immutable descriptors, a complete per-feature qualification resolver and all F0–F5 format gates remain work. Registering a text factory does not grant image, embedding, tools, persistence, batching or GPU capability.

The current source loader owns regular-file handles and bounds each read. Local source directories must remain under operator control during loading and inference. Full hostile-directory, platform-specific handle-relative containment and full multi-file/platform qualification remain open; managed publication is documented in [MANAGED_CHECKPOINTS.md](MANAGED_CHECKPOINTS.md).

## Distribution model

Official architecture adapters stay in the main Coheryn repository and compile into the engine and application binaries. The existing Qwen2 and Qwen3.5 factories are registered in `Registry::default()`. This distribution mechanism is implemented locally; it does not imply that a new public release has been published.

| Component | Distribution | Current status |
|---|---|---|
| Official native architecture code | Main repository, compiled into Coheryn binaries | Implemented for standalone selection and managed checkpoint text binding |
| Model weights, configuration and tokenizer/template assets | Separately acquired, versioned model packages with explicit integrity/provenance | Pinned acquisition and managed checkpoint import/stateless text execution implemented; complete F gates remain open |
| Optional adapter crates | Separate Rust crates within the same repository, selected at build time | Future work after shared interfaces stabilize |
| Community adapter packages | Potentially separate repositories using a versioned extension contract | Future work; no supported external plugin contract |

A new compatible checkpoint can use an existing compiled adapter when its complete source and feature requirements pass validation. Adding a new architecture implementation requires rebuilding Coheryn; users receive that support through an application update. No separate GitHub repository or adapter installation step is required for bundled official support.

Keep modules inside the engine initially. Extract an optional crate only when its dependency or maintenance boundary justifies it. A separate crate does not require a separate repository, and publishing source as a crate would not itself enable runtime plugin installation.

Separately installable executable adapters require their own design and acceptance gate: versioned interfaces, engine/backend compatibility checks, resource and cancellation contracts, package integrity and publisher trust, lifecycle handling and qualification evidence. That gate remains future work. Model packages continue to be inert assets and cannot install or execute architecture code. Managed checkpoints use the same registry for source validation and profile selection. Daemon execution retains typed residents; a common executable interface across managed text, embedding and vision remains future work.

Qwen3.5 standalone generation now shares the metadata estimator and atomically holds session/request/profile-headroom reservations through worker retirement. [MODEL_METADATA.md](MODEL_METADATA.md) documents this process-local admission checkpoint; managed source sets, shared budgets and stateless API scheduling now extend it through [MANAGED_CHECKPOINTS.md](MANAGED_CHECKPOINTS.md). Managed text histories now share a bounded `TextConversation` preparation contract across Qwen2 and Qwen3.5; see [QWEN35_SESSIONS.md](QWEN35_SESSIONS.md). Qwen3.5 state reuse and the full task/execution-adapter migration remain open.

## Qwen3.5 hybrid continuation

The bundled Qwen3.5 adapter now exposes validated still-image capability when the actual processor/tensor/rotary profile matches. Typed managed dispatch connects native image inputs to the existing bounded transport and compute coordinator. Text generation states support independent cancellation and shared weight-tile batching. Optional history snapshots contain full KV, recurrent matrices and convolution history/cursors under separate budget reservations; they are not portable adapter packages or serialized model state. Broader executable interface unification and external adapter distribution remain separate gates. [QWEN35.md](QWEN35.md) and [QWEN35_SESSIONS.md](QWEN35_SESSIONS.md) define current scope.

# Independent assessment sources

These scripts are research capture sources, excluded from Coheryn's native build, test and runtime chain. They execute an independently built assessment program and are never inference fallbacks. Native checks consume only the frozen files in `fixtures/tools/`.

Inputs are the exact model pin, reference revision/executable digest and settings recorded in `fixtures/tools/qwen2.5-reference.json`, Jinja2 3.1.6, and the pinned companion template. Use a separate assessment checkout/cache. The existing scalar reference source is preserved at `resources/reference-archive/tools/reference_oracle.cpp.txt`; its independently built executable is expected at `.cache/reference-oracle`, with pinned reference sources/version marker at `.cache/reference/`. Original build provenance/settings remain in `fixtures/models.json` and the archive. This directory does not bootstrap or install an external runtime into the application.

From the repository root, in that independently prepared environment:

```sh
python3 resources/tool-assessment/capture.py
python3 resources/tool-assessment/decode.py
python3 resources/tool-assessment/capture_parallel.py
python3 resources/tool-assessment/decode.py
python3 resources/tool-assessment/capture_results.py
python3 resources/tool-assessment/decode.py
```

These deliberately write candidate research outputs. Compare them against the frozen acceptance criteria before adopting any change; native outputs must never replace their own oracle. The stages preserve previously captured cases when appending additional cases. `decode.py` verifies the model SHA before reading the vocabulary. Expected prompt strings/token IDs are independently captured; `tools/check_tool_fixtures.py` only checks their identities/bounds and never executes these scripts.

The original double-braced publisher-example negative capture is retained separately. It can be replayed by passing its frozen prompt to the pinned independent program in `generate` mode with count 96, then decoding with that same pinned vocabulary. It is not repaired or substituted with the companion result.

## Separate image processor and vision captures

`image_oracle.cpp` runs only the pinned external processor against canonical RGB; `capture_images.py` uses Pillow 13.0.0 in `.cache/image-reference` and verifies an explicit horizontal-then-vertical bicubic reconstruction. Frozen outputs go to `fixtures/vlm/preprocessing/`; native checks consume those bytes without invoking these programs. Keep the pinned source-revision marker and archive identity from `fixtures/models.json`. Source file hashes are recorded in the processor capture. Full GGUF inputs must match `fixtures/qualification/m0-components.json` before use.

Example compilation in the independently prepared assessment environment (macOS):

```sh
mkdir -p .cache/image-oracle
clang++ -std=c++17 -O2 -ffp-contract=off \
  -I.cache/reference/tools/mtmd -I.cache/reference/include -I.cache/reference/ggml/include \
  resources/tool-assessment/image_oracle.cpp .cache/reference/tools/mtmd/mtmd-image.cpp \
  -L.cache/reference/build/bin -lggml-base \
  -Wl,-rpath,"$PWD/.cache/reference/build/bin" -o .cache/image-oracle/processor
python3 resources/tool-assessment/capture_images.py
```

`vision_oracle.cpp` loads the exact selected vision/projector GGUF on CPU, runs one bounded image and records layer/projected tensors. Raw F32 files stay ignored under `.cache/vision-oracle/`; the compact committed reference records their hashes, full-vector summaries and deterministic samples. The capture is reference-only and does not qualify the native graph or image-conditioned language outputs.

Use a separate scalar assessment build, leaving the original reference build intact:

```sh
PYTHONPATH=.cache/devtools .cache/devtools/bin/cmake \
  -S .cache/reference -B .cache/vision-assessment-build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS='-U__ARM_NEON -U__ARM_NEON__ -ffp-contract=off' \
  -DCMAKE_CXX_FLAGS='-U__ARM_NEON -U__ARM_NEON__ -ffp-contract=off' \
  -DGGML_CPU_ARM_ARCH=armv8-a -DGGML_NATIVE=OFF -DGGML_METAL=OFF \
  -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF -DGGML_LLAMAFILE=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DMTMD_VIDEO=OFF
PYTHONPATH=.cache/devtools .cache/devtools/bin/cmake --build .cache/vision-assessment-build --target mtmd -j2
clang++ -std=c++17 -O2 -ffp-contract=off \
  -I.cache/reference/tools/mtmd -I.cache/reference/include -I.cache/reference/ggml/include \
  resources/tool-assessment/vision_oracle.cpp \
  -L.cache/vision-assessment-build/bin -lmtmd -lggml -lggml-base \
  -Wl,-rpath,"$PWD/.cache/vision-assessment-build/bin" -o .cache/image-oracle/vision
python3 resources/tool-assessment/capture_vision.py
```

The vision process is single-threaded and sequential across images/components. Selected Qwen limits explicitly use 4–16,384 feature tokens to match source processor pixels, instead of assessment defaults. The legacy LLaVA file stores 23 blocks, while this reference's default selects layer 21 (22 executed blocks); the source configuration specifies 24 blocks with feature layer −2. Together with padding versus source center crop, this is an unresolved alignment issue, not accepted pretrained-source parity. Preserve the observed result and resolve the profile/model/reference choice before native graph acceptance. Do not modify original artifacts or relax numeric criteria to hide the difference.

## Corrected, staged vision-language reference

The original pinned source and original `vision-reference.json` are retained as diagnostic inputs. `prepare_vision_assessment.py` copies that source into `.cache/vision-corrected-source` without its build directory and applies two narrowly recorded legacy-MLP graph corrections. The source record preserves before/after hashes and links to the two upstream reports; report status alone is not numerical acceptance. The independent `llava_numpy.py` calculation provides the actual cross-check using identical GGUF weights. It uses NumPy 2.5.4, Pillow 13.0.0 and a small own standard-libm bridge, outside native dependencies.

Prepare the source with `python3 resources/tool-assessment/prepare_vision_assessment.py`, then use the scalar CMake command above with source/build paths changed to `.cache/vision-corrected-source` and `.cache/vision-corrected-build`. Build target `mtmd`. Compile the own helpers:

```sh
clang -std=c11 -O2 -ffp-contract=off -fno-vectorize -fno-slp-vectorize -dynamiclib \
  resources/tool-assessment/scalar_math.c -o .cache/image-oracle/scalar-math.dylib
clang++ -std=c++17 -O2 -ffp-contract=off \
  -I.cache/vision-corrected-source/tools/mtmd -I.cache/vision-corrected-source/include \
  -I.cache/vision-corrected-source/ggml/include resources/tool-assessment/vlm_oracle.cpp \
  -L.cache/vision-corrected-build/bin -lmtmd -lllama -lggml -lggml-base \
  -Wl,-rpath,"$PWD/.cache/vision-corrected-build/bin" -o .cache/image-oracle/vlm-corrected
python3 resources/tool-assessment/capture_vlm.py --stage encode
python3 resources/tool-assessment/freeze_vlm_positions.py
python3 resources/tool-assessment/capture_vlm.py --stage decode
```

Install the pinned assessment-only Python requirements in an ignored target, using the current isolated installer (pip 26.2.2); `requirements-vision.txt` never enters native CI/runtime. The capture uses the existing `.cache/vision-numpy`, `.cache/image-reference` and `.cache/devtools` paths. After numerical stages complete, `python3 resources/tool-assessment/capture_vlm.py --stage collect` verifies receipts and collects complete metadata without executing a model.

The second-stage process maps language weights only after the first process exits. Metadata-only loading uses `no_alloc=true` with `LLAMA_LOAD_MODE_NONE`; this pinned API does not provide usable model dimensions in `vocab_only` mode and asserts on `no_alloc` combined with mmap. These are reference API constraints, not native production behavior. Ordinary native checks never invoke this pipeline.

The default request corpus contains exactly eight cases. Each completed stage writes identity/output-hash receipts; resuming verifies every cached output before reuse. Unknown identities are rejected rather than silently overwriting prior evidence. Source weights remain unchanged. Capture settings stay one CPU thread, no GPU/flash attention, F32 KV and greedy output capped at eight tokens. Per-stage OS peak RSS is recorded separately from native reservations. Other configurations require independent capture/qualification.

## Additional Qwen vision geometries

With the original `vision` helper compiled as above and the pinned complete component cached, run `python3 resources/tool-assessment/capture_qwen_vision_extra.py`. The helper captures portrait, minimum and odd-dimension canonical PNGs independently into ignored `.cache/vision-extra/` raw arrays and `fixtures/vlm/qwen2-vision-extra-v1.json`. It records component, wrapper, capture and executable hashes. This is a separate frozen reference record; previous captures and tolerance criteria are unchanged. Native CI reads these records without launching this helper. `tools/check_native_vision.py` subsequently runs only the native Rust example against frozen arrays.

## Qwen language epsilon alignment

The pinned `src/models/qwen2vl.cpp` reads rotary sections but omits `LLM_KV_ATTENTION_LAYERNORM_RMS_EPS`; original logs report zero although the selected GGUF/source configuration declares approximately `1e-6`. Preserve these diagnostic captures. `vlm_language_oracle.cpp` adds bounded last-row decoder checkpoint instrumentation around the immutable staged wrapper. Build it against the original corrected-vision library and run `capture_vlm_language.py` to reproduce epsilon-zero decoder layers without changing the original image-conditioned fixtures.

`prepare_qwen_language_assessment.py` pins the original Qwen source hash and creates a separate `.cache/qwen-language-qualified/` capsule. It adds just the epsilon metadata read, recompiles only the affected unity object and relinks a private library with hashed unchanged object inputs. Original source/builds remain unchanged. `qwen-language-source-record.json` records before/after source and unity hashes, original object inventory, replacement object, compile/link arguments and resulting library. The object inventory includes the replaced original unity object; `link_argv` selects the separate replacement. The metadata-read change is the only model-source correction for this capsule.

```sh
python3 resources/tool-assessment/prepare_qwen_language_assessment.py
clang++ -std=c++17 resources/tool-assessment/vlm_language_oracle.cpp \
  -I.cache/vision-corrected-source/tools/mtmd -I.cache/vision-corrected-source/include \
  -I.cache/vision-corrected-source/ggml/include -L.cache/qwen-language-qualified/bin \
  -lmtmd -lllama -lggml -lggml-base \
  -Wl,-rpath,"$PWD/.cache/qwen-language-qualified/bin" \
  -o .cache/qwen-language-qualified/bin/vlm-language
python3 resources/tool-assessment/capture_qwen_corrected_language.py
clang -O3 -ffp-contract=off -fno-vectorize -fno-slp-vectorize -dynamiclib \
  resources/tool-assessment/qwen_scalar_math.c -o .cache/image-oracle/qwen-scalar-math.dylib
python3 resources/tool-assessment/qwen_layer_numpy.py
```

The corrected capture explicitly selects the isolated library and checks the logged epsilon. It retains identical staged embeddings, prompts, token/position inputs, F32 cache and unchanged tolerances. V2 manifests are separate from all original captures. The independent NumPy first-layer calculation reads identical GGUF tensors, uses half operands/wide matrix reductions, explicit causal GQA/positions and scalar libm, and reproduces both zero and metadata epsilon profiles bit-for-bit. It uses the frozen teacher-forced token/image inputs; no reference decoder activations are fed into its computation, and it uses no native Coheryn math. The helper is assessment-only; no inference dependencies enter Cargo or native CI/runtime. The current capsule recipe and binary records are macOS ARM64 evidence, not portable/release qualification.

The native vision v2 contract also reproduces the pinned scalar reference's FP16 input/output GELU lookup precision without retaining a table. This is a declared native precision profile; original numerical criteria and prior v1 component evidence remain unchanged. See the native offline guide for supported execution and remaining gates.

## Qwen3.5 independent array equations

`qwen35_operations.py` evaluates authored rational inputs using explicit binary32 scalar arithmetic. `qwen35_numpy.py` independently implements the published text recurrence/attention equations with NumPy 2.5.4, Python Tokenizers 0.23.3 and Jinja2 3.1.6. The pinned read-only architecture source hash/revision and all source checkpoint identities are recorded in the resulting inert manifest. This is a declared BF16-weight/F32-activation wide-reduction profile, not a capture from the publisher's complete inference framework. Broader framework/EOS/context/vision qualification remains open.

Prepare the isolated packages from `requirements-qwen35.txt`, using the current local installer, in the cache paths declared by the capture. Acquire/verify the checkpoint separately with the native development asset acquisition script. Copy the committed read-only `qwen35-source.json` record to `.cache/qwen3.5-0.8b-source/architecture-source.json`; it pins the official source used to audit the equations and is not executable code. Then run `python3 resources/tool-assessment/qwen35_numpy.py` in the assessment environment. Candidate raw arrays go to ignored `.cache/q35-reference/`; compact deterministic samples, complete raw hashes, prompts/IDs and greedy outputs go to `fixtures/qwen35/reference-v1.json`. Review candidates before adoption; never replace them with native output. Existing source weights, references and tolerances remain unchanged.

The native `tools/check_qwen35.py` reads frozen inert data and launches only the Rust probe. It never invokes this assessment script or imports NumPy/an external inference framework. Ordinary native tests consume the scalar fixture without loading Python. The first Qwen3.5 milestone is standalone CPU text; source vision/MTP tensors remain inactive.

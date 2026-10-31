# Modern Qwen3.5 formats and qualification

Current development targets the pinned publisher Qwen3.5-0.8B checkpoint across Safetensors and community GGUF conversions. The native Qwen3.5 graph is shared: the GGUF adapter binds checked physical tensors into the existing hybrid language and vision implementation. Original Qwen2 fixtures remain regression coverage. CPU support remains experimental; configured CI jobs do not establish qualification on a platform.

## Selected source sets

| Group | Sources | Independent reference coverage |
|---|---|---|
| `safetensors` | Publisher BF16/F32 single/indexed source and a losslessly derived three-physical-shard layout | Existing frozen text and image arrays, prompt/token IDs and greedy continuations |
| `gguf-quantized` | Qwen3.5 F16, Q8_0, Q4_0, Q4_K_M and Q6_K language files with source sidecars | Per-file independent text layers, teacher-forced logits, exact tokenization and greedy text; both one/two workers |
| `gguf-vision` | F16 and Q4_K_M language files paired with the F16 vision component | Independent square, portrait and ordered-two-image encoder/merger/language/logit arrays, prompt/token IDs and greedy text |
| `legacy-regression` | Earlier FP16 Qwen2 GGUF and F16/Q8 Nomic embeddings | Existing immutable regression contracts |

`fixtures/qwen35/gguf/models-v1.json` records the exact conversion supplier, repository revision, lengths and checksums. These are community conversions of the separately pinned official checkpoint, with distinct provenance. No universal Qwen, VLM, GGUF or quantization claim is made. Inspect actual tensor inventories; a filename label is insufficient. Exact executed combinations and limits belong to their qualification reports.

The three-shard fixture uses every original model tensor, with no numeric conversion or tensor splitting. `fixtures/qwen35/sharded-layout-v1.json` records source identity, deterministic recipe, physical file hashes and per-tensor byte hashes. This is real model execution across three physical files, not a publisher-supplied multi-shard release. Larger/different publisher configurations remain separate qualification. Layouts have distinct managed source-set identities.

## Native GGUF source assets

Select the language file explicitly. Place the qualified publisher `config.json`, `tokenizer.json`, `chat_template.jinja` and, for images, `preprocessor_config.json` beside it. The paired vision file is named `vision.gguf`. The preparation tool acquires and verifies the complete selected set, including attribution. `--vision-model` retains its earlier paired-model semantics; the modern Qwen3.5 model set uses its associated `vision.gguf`.

The adapter validates graph dimensions, hybrid layer policy, rotary parameters, tokenizer inventory/template identity, required tensor shapes and actual encodings. Unknown graph/tokenizer/vision options and unsupported GGUF shard declarations fail explicitly. Missing or mismatched assets cannot silently choose another backend/model. The native checkpoint transport binds every selected execution asset and rejects extra unselected weight files, unsafe names, links, incomplete uploads and changed payloads.

GGUF norm and recurrent coefficients are consumed in their stored folded representation. The two temporal patch kernels are exposed through a checked byte view in source channel/time order. Language and vision matrices retain F32 operands/products with wider scalar reductions, decoding packed weights only into bounded reusable tiles. No persistent full-model dequantization or full-model heap copy is introduced. The earlier GGUF precision/activation-quantization behavior remains covered by its original regression fixtures.

## Local preparation and inference

```sh
./tools/cargo build --release --workspace --locked
python3 tools/prepare_modern_models.py --group gguf-vision --fetch \
  --report target/modern-inputs.json

target/release/coheryn run .cache/qwen35-gguf/Qwen3.5-0.8B-q4_k_m.gguf \
  --standalone --messages fixtures/qwen35/smoke-messages.json \
  --context 256 --max-tokens 4 --profile reference --threads 1 --json

target/release/coheryn chat .cache/qwen35-gguf/Qwen3.5-0.8B-q4_k_m.gguf \
  --standalone --image fixtures/qwen35/vision/red.png \
  --prompt 'What color is the picture?' --context 256 --max-tokens 4 \
  --profile reference --threads 1 --json
```

Use `--group safetensors` to prepare the original and derived three-file layout, or `--group gguf-quantized` for all selected encodings. Without `--fetch`, preparation requires cached publisher inputs. Files stay in ignored caches; existing corrupt data and occupied derivation destinations fail visibly. The derivation streams bounded chunks, stages complete files atomically and preserves original source bytes. New report paths are required for every run.

Managed import/load, CLI daemon generation, OpenAI chat/completions and SSE reuse the established engine and source-set protocol. Import the selected language file or the Safetensors index with `models import`; remote clients upload bounded inert model-set bytes rather than filesystem paths or model code. Image source-set identity includes both GGUF components and source processor assets. Compatible managed text/image lifecycle capabilities retain their own qualification obligations.

## Numerical evidence and release limits

Each quantized file has independently captured references for the identical stored weights. Quantized-versus-F16 differences are measured separately as quality observations; they cannot loosen executor correctness thresholds. Different quantizations may produce different valid continuations. `fixtures/qwen35/gguf/quantization-quality-v1.json` records first-logit differences, top-token overlap and greedy differences on the two short text cases; it does not establish broad semantic quality. The independent assessment code and its separately pinned reader/dequantizer remain under `resources/tool-assessment/`, excluded from production and native CI. Native jobs consume only checksum-verified frozen data and execute Coheryn code.

The current combined reference inventory is `fixtures/cpu-qualification/modern-references-v1.json`; its archive descriptor is `modern-reference-bundle-v1.json`. Earlier references and bundle descriptors remain immutable historical records. Package the current data with `tools/package_cpu_references.py`, host the exact verified archive through reviewed HTTPS, then dispatch the manual workflow. Its four groups run independently on each native CPU target to bound individual job storage and isolate evidence. Select a local group with `tools/run_cpu_qualification.py --group GROUP` using the archive/native-check arguments in [CPU_QUALIFICATION.md](CPU_QUALIFICATION.md).

Reports distinguish exact source/layout/component/encoding/backend/context/thread combinations. Local array or byte-layout parity does not establish publisher-framework parity, broad image semantics, larger dimensions, all contexts, CPU/GPU parity, sustained fairness/memory behavior, signing or production readiness. Physical residency and mappings remain separate from engine-controlled reservations. All retained release requirements remain in force.

The pinned Q4_K_M file mixes Q4_K, Q5_K and Q6_K tensor blocks. Native kernels cover all three; the filename alone does not establish compatibility. Q5_K block decoding, activation packing and dot products have separate independent fixtures.

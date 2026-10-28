# Native structured output and sampling

The CPU preview implements `coheryn-json-v1` token constraints and `coheryn-sampling-v2`. These are native decoder modes over the actual tokenizer bytes. A requested constraint is never replaced by prompt-only formatting, JSON repair or an unconstrained retry. The exact-model evidence and supported combinations are recorded in [compatibility.json](../fixtures/compatibility.json); broader model/backend/platform qualification remains separate.

## CLI and API

`run`, interactive `chat`, and `sessions send` accept `--response-format text|json_object|json_schema`. Schema mode additionally requires `--schema FILE`, a regular JSON file of at most 16 KiB. `--json` controls the CLI result envelope independently of the model's response format. Headless `chat --request FILE` and `sessions submit --request FILE` accept the equivalent request/settings fields; flags that would conflict with a headless request are rejected.

For example, save this as `schema.json`:

```json
{
  "type": "object",
  "properties": {"answer": {"type": "string", "const": "ok"}},
  "required": ["answer"],
  "additionalProperties": false
}
```

```sh
coheryn --json run .cache/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --standalone --prompt 'Return exactly the JSON object {"answer":"ok"}.' \
  --context 512 --max-tokens 64 --profile reference --threads 1 \
  --response-format json_schema --schema schema.json
```

Both `/v1/completions` and `/v1/chat/completions` support `response_format`; `/api/v1/generate` returns the native result. For chat, a body may contain:

```json
{
  "model": "managed-model",
  "messages": [{"role": "user", "content": "Return answer as the string ok in a JSON object."}],
  "max_tokens": 64,
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "result",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {"answer": {"type": "string", "const": "ok"}},
        "required": ["answer"],
        "additionalProperties": false
      }
    }
  }
}
```

`{"type":"text"}` preserves ordinary generation. `{"type":"json_object"}` enforces bounded JSON-object syntax without a user schema. Schema names contain 1–64 ASCII letters, digits, `_` or `-`; `strict` must be true. Unknown response-format fields fail. Supply instructions describing the intended content: constraints enforce the qualified structure and values, not factual correctness.

## Schema profile and finite limits

The [machine-readable profile](../fixtures/constraints/profile-v1.json) defines the contract. Every strict node requires `type`. Supported types are object, array, string, number, integer, boolean and null. Nullable properties use exactly `[type,"null"]` or `["null",type]`; the root must be a non-nullable object.

- Objects require `properties` and `additionalProperties:false`. `required` names must be unique and declared. Properties may appear in any order, once each; optional properties may be omitted. Generated property-name spelling is canonical JSON, including its escaping.
- Arrays require one homogeneous `items` schema. `maxItems` defaults to 32 and accepts 0–128.
- Strings count Unicode scalar values. `maxLength` defaults to 256 and accepts 0–4096. Raw UTF-8, JSON escapes and properly paired surrogate escapes are supported across token boundaries.
- Primitive `enum` and `const` are supported; they cannot be combined. Enum values must be unique, match the declared type, number at most 32, and serialize to at most 4096 bytes each. The generator uses their canonical JSON spelling.
- Numeric spellings are bounded to 32 bytes and two exponent digits; ordinary integer spellings use at most 18 digits. Numeric enum/const spellings are their finite canonical JSON representations. No numeric range/multiple constraints are implemented.
- `title` and `description` are string annotations. Optional `$schema` accepts only `https://json-schema.org/draft/2020-12/schema`; it triggers no network access.

Other keywords and forms—including boolean schemas, `$ref`, `$id`, recursion, general unions, composition, patterns, formats, defaults and numeric constraints—are explicit unsupported-schema errors. Constraints are not silently treated as annotations.

| Bound | Value |
|---|---|
| Serialized schema source | 16 KiB; counted without an unbounded serialization buffer |
| Source JSON depth / nodes | 32 / 4096 |
| Typed schema depth / nodes | 8 / 256 |
| Generated container depth | 8 |
| Properties per object / property-name bytes | 64 / 256 |
| Constrained output | 64 KiB, including strict tool framing |
| Tokenizer vocabulary | 262,144 tokens |
| Eligible token piece | 1–128 decoded bytes; longer pieces are excluded |
| Tokenizer trie | 1,048,576 nodes |
| Transition/mask cache | No persistent cache or schema-state × vocabulary table |
| Private grammar/trie reservation | 32 MiB per constrained request |

JSON-object mode uses the same finite container, property, string, array, output and tokenizer bounds. Its generic object names need not be unique; strict typed schemas reject duplicate properties. Actual model files and tokenizers require qualification; a family name alone is not evidence.

The request-owned byte trie prunes impossible token prefixes without reparsing the entire generated output for every token. It handles incomplete UTF-8 and quoted/escaped JSON states. Special/EOS tokens are masked; completion stops automatically after the complete object. A strict tool profile permits only its qualified framing markers. Final syntax and profile-specific semantic validation run before conformant success is reported.

## Strict tool arguments

The qualified mode supports exactly **one** function with `function.strict:true`, a `required` or matching named choice, and explicit `parallel_tool_calls:false`. Its parameter schema must satisfy this profile. The native grammar constrains the function name, argument object and opening/closing tool frame; the envelope consumes the same depth/source/output limits. The function is proposed as data and is never executed by Coheryn.

Strict automatic/optional selection, parallel calls and multiple declarations are unqualified and rejected. Non-strict tool calling retains its existing behavior and limits. A separate `response_format` cannot be combined with enabled tools. Custom string/token stops cannot be combined with either constrained output or enabled tools.

## Sampling, stops and deadlines

The processing order is versioned:

1. Apply signed repetition adjustment, then presence and frequency penalties over the **exact selected prompt and previously generated token IDs**, including control tokens. Reused KV prefixes do not remove history from these counts.
2. Enforce the mandatory token/grammar mask. No subsequent filter can restore a disallowed token.
3. For temperature zero, choose the highest allowed adjusted logit, breaking ties by lowest token ID. This path uses a linear scan and no sorting/probability buffers.
4. Otherwise rank allowed candidates, retain top-k when enabled, compute stable temperature-scaled F64 probabilities, apply top-p, and draw from the retained mass using the per-request SplitMix64 stream. Candidate/probability buffers are reused; top-k uses partial selection before sorting its retained candidates.

| Setting | Range / default |
|---|---|
| `temperature` / `--temperature` | Finite 0–2 / 0 |
| `top_p` / `--top-p` | `(0,1]` / 1 |
| `top_k` / `--top-k` | 0 disables; otherwise at most the loaded vocabulary / 0 |
| `seed` / `--seed` | Unsigned 64-bit integer / 0 |
| `repetition_penalty` / `--repetition-penalty` | Finite 0.01–10 / 1 |
| `frequency_penalty` / `--frequency-penalty` | Finite −2–2 / 0 |
| `presence_penalty` / `--presence-penalty` | Finite −2–2 / 0 |
| `stop` / repeatable `--stop` | At most 16 nonempty strings, 4096 bytes each |
| `stop_tokens` / repeatable `--stop-token` | At most 16 valid vocabulary IDs |
| `timeout_seconds` / `--timeout-seconds` | Optional integer 1–86400; daemon policy may impose an earlier deadline |

Stops are checked after selection. A selected stop token contributes to completion usage but emits no text. String stops retain their existing cross-token buffering. `n=1` remains the per-request generation ceiling; daemon concurrency and queue limits bound independent requests. The S6 CPU coordinator can batch compatible requests while keeping RNG and constraint state private; see [BATCHING.md](BATCHING.md). Context and maximum output tokens remain explicit; requests are rejected when the requested capacity does not fit. Conflicting aliases and unsupported combinations fail instead of changing effective settings.

Native deadlines are checked cooperatively between bounded compute/trie units. Daemon deadlines cover admitted queue and execution time; parsing and model acquisition are separate operations. Clients apply their request deadline while sending/receiving; standalone generation checks it in the engine. Ctrl-C, disconnect, shutdown, queue limits and backpressure retain their existing cancellation boundaries. Resources remain held until compute acknowledges termination.

Identical seeds are scoped to the pinned model, tokenizer/template, options, selected history, engine/build, backend/precision and execution strategy. Queue order and another request do not advance a request's RNG. There is no cross-device/kernel bitwise guarantee. Native results and the OpenAI `coheryn` extension report the sampling profile, effective sampling/penalty settings, tokenizer fingerprint, backend/cache dtype and worker/chunk settings, without raw private history.

## Completion and failure contracts

A constrained success exposes `constraint_validation` with profile, schema fingerprint, `valid:true`, and target `response` or `tool_arguments`. OpenAI complete responses and terminal SSE chunks carry it under `coheryn`. Schema compilation/trie construction and mask timings are separate; mask and sampling timings are subsets of decode time. Native tokenization time measures tokenization rather than including constraint compilation.

Streams may contain incomplete JSON prefixes or tool argument fragments. Length, dead-end, invalid-output, cancellation, deadline or resource failures never produce a successful conformant terminal response. Streaming failures emit a structured error and `[DONE]`; complete HTTP failures return an error instead of a completed result. Constraint failures include validation status with `valid:false`.

| Outcome | HTTP | CLI exit |
|---|---|---|
| Invalid field/range/conflict | 400 | 2 |
| Unsupported schema/combination | 400 | 4 |
| Constraint incomplete/dead-end/violation | 422 | 9 |
| Engine resource exhaustion | 503 | 5 |
| Explicit cancellation | 408 | 130 |
| Deadline exceeded | 408 | 124 |

Other existing exit meanings remain unchanged. CLI raw streaming can leave an incomplete prefix on stdout; consumers must inspect exit status. Headless request streaming supplies native JSON Lines with a final result only on success.

## Admission and verification

`plan --constraints` and `/api/v1/plans/estimate` with `constraints:true` include the 32 MiB per-request reservation and profile headroom without loading weights. Embedding plans reject this generation-only option. Concurrent estimates multiply private grammar cost, not shared weights. Queued constrained requests hold their reservation atomically with session/payload capacity. Trie/mask/alias/schema and reusable sampler vector capacities contribute to observed engine buffer counters; allocator, library and OS residency remain separate quantities.

Fast checks cover the schema compiler, byte parser, UTF-8/surrogates, bounds, masks, actual native generation, strict framing, final validation, memory cleanup, API/SSE errors, CLI flags, planner arithmetic, history/RNG and 96 frozen independently computed sampling decisions. A dedicated constraint fuzz target supplements deterministic tests; sustained fuzz/security qualification remains open.

```sh
python3 tools/check_constraint_fixtures.py
./tools/cargo test --workspace --locked --offline
./tools/cargo check --manifest-path fuzz/Cargo.toml --locked --offline
# Opt-in: cached pinned model and rebuilt native release binaries required.
python3 tools/check_constraints.py
```

The native qualification runner reads independently specified contracts and frozen arithmetic fixtures; it does not regenerate expected model outputs from Coheryn. Its observations belong in `fixtures/qualification/s5-service.json`. Historical reference captures and earlier binary hashes remain unchanged.

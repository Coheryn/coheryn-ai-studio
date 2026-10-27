# Separate S4 assessment sources

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

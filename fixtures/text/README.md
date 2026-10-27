# Qualified text-template fixture

`qwen-chat-template.json` stores the exact JSON-encoded `tokenizer.chat_template` extracted from the initial pinned publisher FP16 GGUF identified in `fixtures/models.json`. Its SHA-256 is `71672240b1a3b068067b6a332bc38d3fc05aaade9d41b4b27d7510ff4335504b`, matching the existing native allowlist. It is a small model asset used by synthetic conversation tests, not model execution code. Publisher model licensing and source revision remain in the manifest and archived provenance. Existing independent token/logit goldens remain unchanged.

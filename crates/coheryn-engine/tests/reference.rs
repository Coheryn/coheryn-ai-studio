//! Opt-in real-model regression test; ordinary CI never downloads model weights.
use coheryn_engine::{
    inspect,
    storage::sha256_file,
    tokenizer::{Message, TextTokenizer},
};
use std::path::PathBuf;

#[test]
#[ignore = "requires the pinned model; run tools/fetch_model.py first"]
fn pinned_tokenizer_and_chat_templates_match_independent_goldens() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model = std::env::var_os("COHERYN_REFERENCE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join(".cache/models/qwen2.5-0.5b-instruct-fp16.gguf"));
    let golden: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("fixtures/golden/qwen2.5-0.5b-instruct-fp16.json"))
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        sha256_file(&model).unwrap(),
        golden["model_sha256"].as_str().unwrap()
    );
    let descriptor = inspect(&model, Default::default()).unwrap();
    let tokenizer = TextTokenizer::from_descriptor(&descriptor).unwrap();
    let inputs: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(root.join("fixtures/prompts.json")).unwrap())
            .unwrap();
    let cases = golden["cases"].as_array().unwrap();
    assert_eq!(cases.len(), inputs.len());
    for (input, case) in inputs.iter().zip(cases) {
        assert_eq!(input["id"], case["id"]);
        let prompt = if let Some(prompt) = input["prompt"].as_str() {
            prompt.to_owned()
        } else {
            let messages: Vec<Message> = serde_json::from_value(input["messages"].clone()).unwrap();
            tokenizer.render(&messages).unwrap()
        };
        assert_eq!(prompt, case["prompt"].as_str().unwrap());
        let expected: Vec<u32> = serde_json::from_value(case["prompt_token_ids"].clone()).unwrap();
        assert_eq!(tokenizer.encode(&prompt).unwrap(), expected);
    }
    assert!(tokenizer.is_eog(151643));
    assert!(tokenizer.is_eog(151645));
}

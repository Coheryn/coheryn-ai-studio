#![no_main]
use libfuzzer_sys::fuzz_target;
fuzz_target!(|data: &[u8]| {
    if data.len() > 32 << 10 {
        return;
    }
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(data) else {
        return;
    };
    if let Some(schema) = value.get("schema")
        && let Ok(grammar) = coheryn_engine::constraints::Grammar::compile(schema)
        && let Some(output) = value.get("output").and_then(serde_json::Value::as_str)
    {
        let _ = grammar.validate_output(output);
    }
});

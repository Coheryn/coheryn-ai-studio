#![no_main]
use coheryn_engine::{FinishReason, tools::*};
use libfuzzer_sys::fuzz_target;
fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_ARGUMENT_BYTES {
        return;
    }
    if let Ok(config) = serde_json::from_slice::<ToolConfig>(data) {
        let _ = config.validate();
    }
    if let Ok(text) = std::str::from_utf8(data) {
        let config:ToolConfig=serde_json::from_value(serde_json::json!({"tools":[{"type":"function","function":{"name":"add","parameters":{"type":"object"}}}]})).unwrap();
        let mut parser = ToolParser::new(config, "fuzz".into()).unwrap();
        if parser.push(text).is_ok() {
            let _ = parser.finish(FinishReason::Stop);
        }
    }
});

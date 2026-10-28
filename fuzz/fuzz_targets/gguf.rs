#![no_main]
use libfuzzer_sys::fuzz_target;
fuzz_target!(|data: &[u8]| {
    let limits = coheryn_engine::ParseLimits {
        max_metadata_allocation: 1 << 20,
        max_header_bytes: 1 << 20,
        max_array_items: 4096,
        max_tensors: 4096,
        max_metadata_entries: 4096,
        max_string_bytes: 65536,
        max_depth: 3,
    };
    if let Ok(descriptor) =
        coheryn_engine::gguf::parse(std::io::Cursor::new(data), data.len() as u64, limits)
    {
        // Exercise both decoder and encoder compatibility before execution allocation.
        let _ = coheryn_engine::validate(&descriptor);
        if descriptor.architecture.as_deref() == Some("qwen2vl") {
            let _ = coheryn_engine::multimodal::qwen2vl::language_configuration(&descriptor);
        }
    }
});

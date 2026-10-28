#![no_main]
use coheryn_engine::multimodal::positions::ImagePositions;
use coheryn_engine::{Cancellation, memory::Budget, multimodal::image::*};
use libfuzzer_sys::fuzz_target;
fuzz_target!(|bytes: &[u8]| {
    let limits = ImageLimits {
        max_encoded_bytes: 64 << 10,
        max_total_encoded_bytes: 64 << 10,
        max_source_pixels: 4096,
        max_edge: 128,
        max_images: 1,
        ..Default::default()
    };
    let budget = Budget::new(8 << 20);
    for processor in [Processor::Llava15Pad336, Processor::Qwen2VlStill] {
        if let Ok(prepared) = prepare(bytes, processor, limits, &budget, &Cancellation::default()) {
            for start in [0, i32::MAX as u32 - 32, u32::MAX] {
                if let Ok(span) = ImagePositions::new(prepared.plan(), start) {
                    assert_eq!(span.iter().len(), span.physical_tokens());
                    assert!(span.at(span.physical_tokens()).is_err());
                    for axis in span.at(span.physical_tokens() - 1).unwrap() {
                        assert!(axis <= i32::MAX as u32);
                    }
                }
            }
        }
        assert_eq!(budget.stats().reserved, 0);
        assert_eq!(budget.stats().session_buffer_capacity_bytes, 0);
    }
});

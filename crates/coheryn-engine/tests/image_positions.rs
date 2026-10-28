use coheryn_engine::{
    ErrorCode,
    multimodal::{
        image::{ImageFormat, ImageInfo, ImageLimits, Processor, plan},
        positions::ImagePositions,
    },
};
use serde_json::Value;
use std::path::Path;

fn fixture_plan(processor: Processor) -> coheryn_engine::multimodal::image::ImagePlan {
    plan(
        ImageInfo {
            format: ImageFormat::Png,
            width: 112,
            height: 56,
            channels: 3,
            orientation: 1,
            encoded_bytes: 100,
        },
        processor,
        ImageLimits::default(),
    )
    .unwrap()
}

#[test]
fn native_spans_match_independent_multimodal_positions_and_ordered_offsets() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/vlm/positions-v1.json");
    let reference: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let cases = reference["spans"].as_array().unwrap();
    assert_eq!(cases.len(), 10);
    for case in cases {
        let processor = Processor::parse(case["processor"].as_str().unwrap()).unwrap();
        let p = fixture_plan(processor);
        let span = ImagePositions::new(&p, case["start"].as_u64().unwrap() as u32).unwrap();
        assert_eq!(
            span.position_count() as u64,
            case["position_count"].as_u64().unwrap()
        );
        assert_eq!(
            span.physical_tokens() as u64,
            case["physical_tokens"].as_u64().unwrap()
        );
        assert_eq!(span.next_position() as u64, case["next"].as_u64().unwrap());
        let expected: Vec<[u32; 4]> = serde_json::from_value(case["decoder_axes"].clone()).unwrap();
        assert_eq!(span.iter().collect::<Vec<_>>(), expected);
        for (index, value) in expected.iter().enumerate() {
            assert_eq!(&span.at(index).unwrap(), value);
        }
        assert_eq!(
            span.at(span.physical_tokens()).unwrap_err().code,
            ErrorCode::InvalidRequest
        );
    }
}

#[test]
fn positional_reservations_keep_physical_features_separate_and_check_signed_overflow() {
    let p = fixture_plan(Processor::Qwen2VlStill);
    let span = ImagePositions::new(&p, 17).unwrap();
    assert_eq!(span.physical_tokens(), 8);
    assert_eq!(span.position_count(), 4);
    assert_eq!(span.next_position(), 21);
    assert_eq!(span.at(0).unwrap(), [17, 17, 17, 0]);
    assert_eq!(span.at(7).unwrap(), [17, 18, 20, 0]);
    assert_eq!(
        ImagePositions::new(&p, i32::MAX as u32 - 3)
            .unwrap_err()
            .code,
        ErrorCode::ResourceExhausted
    );
    assert!(ImagePositions::new(&p, i32::MAX as u32 - 4).is_ok());
    let linear = ImagePositions::new(&fixture_plan(Processor::Llava15Pad336), 5).unwrap();
    assert_eq!(linear.position_count(), 576);
    assert_eq!(linear.physical_tokens(), 576);
    assert_eq!(linear.at(575).unwrap(), [580; 4]);
}

#[test]
fn forged_image_geometry_and_feature_counts_are_rejected_before_iteration() {
    let p = fixture_plan(Processor::Qwen2VlStill);
    let mut bad = p.clone();
    bad.image_feature_tokens += 1;
    assert!(ImagePositions::new(&bad, 0).is_err());
    let mut bad = p.clone();
    bad.grid[0] = 2;
    assert!(ImagePositions::new(&bad, 0).is_err());
    let mut bad = p.clone();
    bad.width = 111;
    assert!(ImagePositions::new(&bad, 0).is_err());
    let mut bad = p;
    bad.image_feature_tokens = usize::MAX;
    assert!(ImagePositions::new(&bad, 0).is_err());
}

use coheryn_engine::{Cancellation, ErrorCode, memory::Budget, multimodal::image::*};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn hex(bytes: impl AsRef<[u8]>) -> String {
    Sha256::digest(bytes.as_ref())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn png(width: u32, height: u32, kind: u8) -> Vec<u8> {
    let mut bytes = include_bytes!("../../../fixtures/vlm/images/minimum.png").to_vec();
    bytes[16..20].copy_from_slice(&width.to_be_bytes());
    bytes[20..24].copy_from_slice(&height.to_be_bytes());
    bytes[25] = kind;
    let mut crc = u32::MAX;
    for byte in &bytes[12..29] {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = if crc & 1 == 0 {
                crc >> 1
            } else {
                (crc >> 1) ^ 0xedb88320
            };
        }
    }
    bytes[29..33].copy_from_slice(&(crc ^ u32::MAX).to_be_bytes());
    bytes
}
#[test]
fn independent_processors_match_canonical_pixels_normalization_and_virtual_patch_layout() {
    let reference: Value = serde_json::from_str(include_str!(
        "../../../fixtures/vlm/preprocessing/reference.json"
    ))
    .unwrap();
    let budget = Budget::new(256 << 20);
    for case in reference["cases"].as_array().unwrap() {
        let encoded = std::fs::read(root().join(case["canonical"].as_str().unwrap())).unwrap();
        let profile = Processor::parse(case["processor"].as_str().unwrap()).unwrap();
        let result = prepare(
            &encoded,
            profile,
            Default::default(),
            &budget,
            &Cancellation::default(),
        )
        .unwrap();
        assert_eq!(
            result.processed_rgb_sha256(),
            case["processed_rgb_sha256"].as_str().unwrap(),
            "{}",
            case["id"]
        );
        assert_eq!(
            result.plan().width as u64,
            case["processed_size"][0].as_u64().unwrap()
        );
        assert_eq!(
            result.plan().height as u64,
            case["processed_size"][1].as_u64().unwrap()
        );
        assert_eq!(
            result.plan().patch_rows as u64,
            case["patch_rows"].as_u64().unwrap()
        );
        assert_eq!(
            result.plan().patch_features as u64,
            case["patch_features"].as_u64().unwrap()
        );
        assert_eq!(
            result.plan().image_feature_tokens as u64,
            case["image_feature_tokens"].as_u64().unwrap()
        );
        for sample in case["normalized_samples"].as_array().unwrap() {
            let index = sample["index"].as_u64().unwrap() as usize;
            let expected = sample["value"].as_f64().unwrap() as f32;
            assert!(
                (result.normalized_hwc()[index] - expected).abs() <= 1e-5 + 1e-5 * expected.abs()
            );
        }
        let mut full = Sha256::new();
        for value in result.normalized_hwc() {
            full.update(value.to_le_bytes());
        }
        let actual = full
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        assert_eq!(
            actual,
            case["normalized_sha256"].as_str().unwrap(),
            "strict f32 reference {}",
            case["id"]
        );
        let mut patches = Sha256::new();
        for row in 0..result.plan().patch_rows {
            for value in result.patch_values(row).unwrap() {
                patches.update(value.to_le_bytes());
            }
        }
        let actual = patches
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        assert_eq!(
            actual,
            case["patches_sha256"].as_str().unwrap(),
            "{}",
            case["id"]
        );
        assert!(result.patch_values(result.plan().patch_rows).is_err());
        assert!(budget.stats().session_buffer_capacity_bytes <= budget.stats().reserved);
        drop(result);
        assert_eq!(budget.stats().reserved, 0);
        assert_eq!(budget.stats().session_buffer_capacity_bytes, 0);
    }
}
#[test]
fn codecs_orientation_gray_alpha_and_decoded_reuse_match_independent_reconstruction() {
    let reference: Value = serde_json::from_str(include_str!(
        "../../../fixtures/vlm/preprocessing/reference.json"
    ))
    .unwrap();
    for case in reference["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["processor"] == "llava15-pad336-v1")
    {
        let bytes = std::fs::read(root().join(case["source"].as_str().unwrap())).unwrap();
        let canonical = std::fs::read(root().join(case["canonical"].as_str().unwrap())).unwrap();
        let budget = Budget::new(256 << 20);
        let cancel = Cancellation::default();
        let actual = decode(&bytes, Default::default(), &budget, &cancel).unwrap();
        let expected = decode(&canonical, Default::default(), &budget, &cancel).unwrap();
        let tolerance = if actual.info().format == ImageFormat::Jpeg {
            2
        } else {
            0
        };
        let a: Vec<_> = actual.pixels_rgb().collect();
        let b: Vec<_> = expected.pixels_rgb().collect();
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(&b) {
            assert!(a.abs_diff(*b) <= tolerance, "{}", case["id"]);
        }
        if tolerance == 0 {
            assert_eq!(hex(a), case["decoded_rgb_sha256"].as_str().unwrap());
        }
        let prepared = prepare_decoded(
            &actual,
            Processor::Llava15Pad336,
            Default::default(),
            &cancel,
        )
        .unwrap();
        assert!(prepared.normalized_hwc().iter().all(|v| v.is_finite()));
        drop(prepared);
        drop(actual);
        drop(expected);
        assert_eq!(budget.stats().reserved, 0);
        assert_eq!(budget.stats().session_buffer_capacity_bytes, 0);
    }
}
#[test]
fn hostile_headers_limits_and_unsupported_images_fail_before_pixel_allocations() {
    let cancel = Cancellation::default();
    for bytes in [png(0, 10, 2), png(u32::MAX, 1, 2), png(8192, 8192, 2)] {
        assert_eq!(
            inspect(&bytes, Default::default(), &cancel)
                .unwrap_err()
                .code,
            ErrorCode::ImageInputLimitExceeded
        );
    }
    for bytes in [png(1, 1, 3), b"GIF89a".to_vec(), b"RIFF0000WEBP".to_vec()] {
        assert_eq!(
            inspect(&bytes, Default::default(), &cancel)
                .unwrap_err()
                .code,
            ErrorCode::UnsupportedImageFormat
        );
    }
    let mut bytes = png(1, 1, 2);
    bytes[8..12].copy_from_slice(&u32::MAX.to_be_bytes());
    assert!(inspect(&bytes, Default::default(), &cancel).is_err());
    let png = include_bytes!("../../../fixtures/vlm/images/landscape.png");
    let budget = Budget::new(64 << 20);
    for end in 0..png.len() {
        assert!(
            prepare(
                &png[..end],
                Processor::Llava15Pad336,
                Default::default(),
                &budget,
                &cancel
            )
            .is_err()
        );
        assert_eq!(budget.stats().reserved, 0);
    }
    let mut corrupted = png.to_vec();
    let n = corrupted.len();
    corrupted[n - 1] ^= 1;
    assert!(
        prepare(
            &corrupted,
            Processor::Llava15Pad336,
            Default::default(),
            &budget,
            &cancel
        )
        .is_err()
    );
    assert_eq!(budget.stats().reserved, 0);
    assert_eq!(budget.stats().session_buffer_capacity_bytes, 0);
}
#[test]
fn admission_is_atomic_context_quality_is_preserved_and_cancellation_releases_all_buffers() {
    let bytes = include_bytes!("../../../fixtures/vlm/images/landscape.png");
    let cancel = Cancellation::default();
    let budget = Budget::new(8 << 20);
    let info = inspect(bytes, Default::default(), &cancel).unwrap();
    let p = plan(info, Processor::Llava15Pad336, Default::default()).unwrap();
    let tiny = Budget::new(p.peak_reserved_bytes - 1);
    assert_eq!(
        prepare(
            bytes,
            Processor::Llava15Pad336,
            Default::default(),
            &tiny,
            &cancel
        )
        .unwrap_err()
        .code,
        ErrorCode::ResourceExhausted
    );
    assert_eq!(tiny.stats().reserved, 0);
    let unavailable = ImageLimits {
        max_processed_pixels: 100,
        ..Default::default()
    };
    assert_eq!(
        prepare(
            bytes,
            Processor::Llava15Pad336,
            unavailable,
            &budget,
            &cancel
        )
        .unwrap_err()
        .code,
        ErrorCode::ImageInputLimitExceeded
    );
    let low_features = ImageLimits {
        max_image_tokens: 100,
        ..Default::default()
    };
    assert!(
        prepare(
            bytes,
            Processor::Llava15Pad336,
            low_features,
            &budget,
            &cancel
        )
        .is_err()
    );
    let images = prepare_many(
        &[bytes, bytes],
        Processor::Llava15Pad336,
        Default::default(),
        &budget,
        &cancel,
    )
    .unwrap();
    assert_eq!(
        images[0].processed_rgb_sha256(),
        images[1].processed_rgb_sha256()
    );
    assert_eq!(budget.stats().reserved, p.output_reserved_bytes * 2);
    assert!(budget.stats().session_buffer_capacity_bytes <= budget.stats().reserved);
    drop(images);
    assert_eq!(budget.stats().reserved, 0);
    let limits = ImageLimits {
        max_images: 1,
        ..Default::default()
    };
    assert!(
        prepare_many(
            &[bytes, bytes],
            Processor::Llava15Pad336,
            limits,
            &budget,
            &cancel
        )
        .is_err()
    );
    cancel.cancel();
    assert_eq!(
        prepare(
            bytes,
            Processor::Qwen2VlStill,
            Default::default(),
            &budget,
            &cancel
        )
        .unwrap_err()
        .code,
        ErrorCode::Cancelled
    );
    assert_eq!(budget.stats().reserved, 0);
    assert_eq!(budget.stats().session_buffer_capacity_bytes, 0);
}
#[test]
fn dimension_planning_respects_model_rounding_and_explicit_operator_limits() {
    for (w, h) in [
        (1, 1),
        (14, 28),
        (42, 70),
        (37, 19),
        (1025, 769),
        (4096, 2160),
    ] {
        let info = ImageInfo {
            format: ImageFormat::Png,
            width: w,
            height: h,
            channels: 3,
            orientation: 1,
            encoded_bytes: 100,
        };
        match plan(info, Processor::Qwen2VlStill, Default::default()) {
            Ok(plan) => {
                assert_eq!(plan.width % 28, 0);
                assert_eq!(plan.height % 28, 0);
                assert!(u64::from(plan.width) * u64::from(plan.height) >= 3136);
            }
            Err(error) => assert_eq!(error.code, ErrorCode::ImageInputLimitExceeded),
        }
    }
    let info = ImageInfo {
        format: ImageFormat::Png,
        width: 8192,
        height: 1,
        channels: 3,
        orientation: 1,
        encoded_bytes: 100,
    };
    assert!(plan(info, Processor::Qwen2VlStill, Default::default()).is_err());
    assert!(Processor::parse("unqualified-model-family").is_err());
    assert!(
        ImageLimits {
            max_images: 0,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
}

#[test]
fn malformed_codec_streams_release_admission_and_jpeg_variants_are_explicit() {
    let png = include_bytes!("../../../fixtures/vlm/images/landscape.png");
    let jpeg = include_bytes!("../../../fixtures/vlm/images/landscape.jpg");
    let budget = Budget::new(16 << 20);
    let cancel = Cancellation::default();
    // Every incomplete JPEG prefix is rejected, including segments and entropy bytes.
    for end in 0..jpeg.len() {
        assert!(inspect(&jpeg[..end], Default::default(), &cancel).is_err());
    }
    let mut progressive = jpeg.to_vec();
    let frame = progressive
        .windows(2)
        .position(|p| p == [0xff, 0xc0])
        .unwrap();
    progressive[frame + 1] = 0xc2;
    assert_eq!(
        inspect(&progressive, Default::default(), &cancel)
            .unwrap_err()
            .code,
        ErrorCode::UnsupportedImageFormat
    );
    let mut trailing = jpeg.to_vec();
    trailing.push(0);
    assert_eq!(
        inspect(&trailing, Default::default(), &cancel)
            .unwrap_err()
            .code,
        ErrorCode::InvalidImage
    );
    // Mutate compressed pixels while preserving a structurally valid PNG checksum.
    let mut corrupt = png.to_vec();
    let kind = corrupt.windows(4).position(|p| p == b"IDAT").unwrap();
    let count = u32::from_be_bytes(corrupt[kind - 4..kind].try_into().unwrap()) as usize;
    corrupt[kind + 4] = 0;
    let mut crc = u32::MAX;
    for byte in &corrupt[kind..kind + 4 + count] {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = if crc & 1 == 0 {
                crc >> 1
            } else {
                (crc >> 1) ^ 0xedb88320
            };
        }
    }
    corrupt[kind + 4 + count..kind + 8 + count].copy_from_slice(&(crc ^ u32::MAX).to_be_bytes());
    assert!(inspect(&corrupt, Default::default(), &cancel).is_ok());
    assert_eq!(
        prepare_many(
            &[png, &corrupt],
            Processor::Llava15Pad336,
            Default::default(),
            &budget,
            &cancel
        )
        .unwrap_err()
        .code,
        ErrorCode::InvalidImage
    );
    assert_eq!(budget.stats().reserved, 0);
    assert_eq!(budget.stats().session_buffer_capacity_bytes, 0);
    let aggregate = ImageLimits {
        max_total_encoded_bytes: png.len() as u64,
        ..Default::default()
    };
    assert_eq!(
        prepare_many(
            &[png, png],
            Processor::Llava15Pad336,
            aggregate,
            &budget,
            &cancel
        )
        .unwrap_err()
        .code,
        ErrorCode::ImageInputLimitExceeded
    );
    let features = ImageLimits {
        max_total_image_tokens: 600,
        ..Default::default()
    };
    assert_eq!(
        prepare_many(
            &[png, png],
            Processor::Llava15Pad336,
            features,
            &budget,
            &cancel
        )
        .unwrap_err()
        .code,
        ErrorCode::ImageInputLimitExceeded
    );
    assert_eq!(budget.stats().reserved, 0);
}

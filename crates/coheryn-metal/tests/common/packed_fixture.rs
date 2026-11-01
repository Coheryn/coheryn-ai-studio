use crate::common::Budget;
use coheryn_metal::{Device, KernelProfile, MatrixShape, WeightType, WorkspaceLimits};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

pub fn packed_profile(kernel: KernelProfile) {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../fixtures/gpu/metal-packed-v1.json"
    ))
    .unwrap();
    let sources: Vec<serde_json::Value> = [
        include_str!("../../../../fixtures/quantized/kernels.json"),
        include_str!("../../../../fixtures/quantized/q5_k-v1.json"),
    ]
    .iter()
    .map(|raw| serde_json::from_str(raw).unwrap())
    .collect();
    let absolute = fixture["precision"]["absolute_tolerance"].as_f64().unwrap();
    let relative = fixture["precision"]["relative_tolerance"].as_f64().unwrap();
    let used = Arc::new(AtomicU64::new(0));
    let device = Device::with_kernel_profile(Arc::new(Budget(Arc::clone(&used))), kernel).unwrap();
    assert_eq!(device.kernel_profile(), kernel);
    println!("Metal packed reference device: {}", device.name());
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 120);
    let limits = WorkspaceLimits {
        width: cases
            .iter()
            .map(|case| case["width"].as_u64().unwrap() as usize)
            .max()
            .unwrap(),
        rows: 3,
        batch: 2,
    };
    let mut workspace = device.workspace_with_limits(limits).unwrap();
    let workspace_bytes = limits.bytes().unwrap() as u64;
    for case in cases {
        let source = &sources[case["source"].as_u64().unwrap() as usize]["cases"];
        let width = case["width"].as_u64().unwrap() as usize;
        let dtype = case["dtype"].as_u64().unwrap() as u32;
        let row_ids = case["rows"].as_array().unwrap();
        let shape = MatrixShape {
            width,
            rows: row_ids.len(),
            dtype: WeightType::try_from(dtype).unwrap(),
        };
        let mut bytes = Vec::new();
        for row in row_ids {
            let block = &source[row.as_u64().unwrap() as usize];
            assert_eq!(block["dtype"].as_u64().unwrap(), u64::from(dtype));
            assert_eq!(block["decoded"].as_array().unwrap().len(), width);
            bytes.extend(
                block["bytes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u8),
            );
        }
        assert_eq!(bytes.len(), shape.weight_bytes().unwrap());
        let original = source[case["input_case"].as_u64().unwrap() as usize]["input"]
            .as_array()
            .unwrap();
        assert_eq!(original.len(), width);
        let mut input: Vec<_> = original
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        for (i, value) in original.iter().enumerate() {
            input.push((value.as_f64().unwrap() * if i % 3 == 0 { 0.5 } else { -1. }) as f32);
        }
        let matrix = device
            .upload(
                shape,
                |offset, chunk| {
                    chunk.copy_from_slice(&bytes[offset as usize..offset as usize + chunk.len()]);
                    Ok(())
                },
                || false,
            )
            .unwrap();
        assert_eq!(
            used.load(Ordering::SeqCst),
            shape.weight_bytes().unwrap() as u64 + workspace_bytes
        );
        let mut actual = vec![0.; shape.rows * 2];
        if case["id"] == "source-0-case-0" {
            let mut too_small = device
                .workspace_with_limits(WorkspaceLimits { rows: 2, ..limits })
                .unwrap();
            actual.fill(999.);
            assert_eq!(
                too_small
                    .multiply(&matrix, &input, &mut actual, || false)
                    .unwrap_err()
                    .kind,
                coheryn_metal::ErrorKind::InvalidInput
            );
            assert!(actual.iter().all(|v| *v == 999.));
            drop(too_small);
            assert_eq!(
                used.load(Ordering::SeqCst),
                shape.weight_bytes().unwrap() as u64 + workspace_bytes
            );
        }
        workspace
            .multiply(&matrix, &input, &mut actual, || false)
            .unwrap();
        let expected = case["expected"].as_array().unwrap();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            let expected = expected.as_f64().unwrap();
            assert!(
                actual.is_finite()
                    && (f64::from(*actual) - expected).abs()
                        <= absolute + relative * expected.abs(),
                "{}: {actual} != {expected}",
                case["id"]
            );
        }
        drop(matrix);
        assert_eq!(used.load(Ordering::SeqCst), workspace_bytes);
        // Reuse the same output allocation with a different row stride; stale
        // values from the larger matrix must never become the second lane.
        let narrow = MatrixShape { rows: 1, ..shape };
        let matrix = device
            .upload(
                narrow,
                |offset, chunk| {
                    chunk.copy_from_slice(&bytes[offset as usize..offset as usize + chunk.len()]);
                    Ok(())
                },
                || false,
            )
            .unwrap();
        let mut actual = [0.; 2];
        workspace
            .multiply(&matrix, &input, &mut actual, || false)
            .unwrap();
        for (actual, index) in actual.into_iter().zip([0, 3]) {
            let expected = expected[index].as_f64().unwrap();
            assert!(
                actual.is_finite()
                    && (f64::from(actual) - expected).abs() <= absolute + relative * expected.abs()
            );
        }
        drop(matrix);
        assert_eq!(used.load(Ordering::SeqCst), workspace_bytes);
    }
    drop(workspace);
    assert_eq!(used.load(Ordering::SeqCst), 0);
}

#![cfg(all(feature = "metal", target_os = "macos"))]
mod common;
use coheryn_metal::{Device, Error, ErrorKind, MatrixShape, WeightType};
use common::Budget;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

#[test]
#[ignore = "requires an admitted real unified-memory Metal device"]
fn independent_dense_references_admission_cancellation_and_cleanup() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../../fixtures/gpu/metal-dense-v1.json")).unwrap();
    let absolute = fixture["precision"]["absolute_tolerance"].as_f64().unwrap();
    let relative = fixture["precision"]["relative_tolerance"].as_f64().unwrap();
    let used = Arc::new(AtomicU64::new(0));
    // No skip-to-pass when a GPU is absent: explicit qualification must fail.
    let device = Device::new(Arc::new(Budget(Arc::clone(&used)))).unwrap();
    println!("Metal device: {}", device.name());
    for case in fixture["cases"].as_array().unwrap() {
        let dtype = match case["dtype"].as_str().unwrap() {
            "f32" => WeightType::F32,
            "f16" => WeightType::F16,
            "bf16" => WeightType::Bf16,
            _ => panic!("unrecognized fixture dtype"),
        };
        let shape = MatrixShape {
            width: case["width"].as_u64().unwrap() as usize,
            rows: case["rows"].as_u64().unwrap() as usize,
            dtype,
        };
        let batch = case["batch"].as_u64().unwrap() as usize;
        let hex = case["weights_le_hex"].as_str().unwrap();
        let bytes: Vec<_> = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect();
        let input: Vec<_> = case["input"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let expected: Vec<_> = case["expected"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
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
        let mut work = device.workspace(shape, batch).unwrap();
        assert_eq!(
            used.load(Ordering::SeqCst),
            (shape.weight_bytes().unwrap() + shape.workspace_bytes(batch).unwrap()) as u64
        );
        let mut actual = vec![999.; expected.len()];
        assert_eq!(
            work.multiply(&matrix, &input, &mut actual, || true)
                .unwrap_err()
                .kind,
            ErrorKind::Cancelled
        );
        assert!(actual.iter().all(|v| *v == 999.));
        let checks = std::cell::Cell::new(0);
        assert_eq!(
            work.multiply(&matrix, &input, &mut actual, || {
                checks.set(checks.get() + 1);
                checks.get() >= 3
            })
            .unwrap_err()
            .kind,
            ErrorKind::Cancelled
        );
        // This cancellation is observed after a committed dispatch retires.
        assert!(actual.iter().all(|v| *v == 999.));
        for _ in 0..3 {
            work.multiply(&matrix, &input, &mut actual, || false)
                .unwrap();
            for (actual, expected) in actual.iter().zip(&expected) {
                assert!(
                    actual.is_finite()
                        && (f64::from(*actual) - expected).abs()
                            <= absolute + relative * expected.abs(),
                    "{}: {actual} != {expected}",
                    case["id"]
                );
            }
        }
        if case["id"] == "f32-cancellation" {
            let mut left = device.workspace(shape, batch).unwrap();
            let mut right = device.workspace(shape, batch).unwrap();
            let scaled: Vec<_> = input.iter().map(|value| value * 0.5).collect();
            let barrier = std::sync::Barrier::new(2);
            let (left_result, right_result) = std::thread::scope(|scope| {
                let matrix = &matrix;
                let barrier = &barrier;
                let input = &input;
                let left = scope.spawn(move || {
                    barrier.wait();
                    let mut result = vec![0.; shape.rows * batch];
                    left.multiply(matrix, input, &mut result, || false).unwrap();
                    result
                });
                let right = scope.spawn(move || {
                    barrier.wait();
                    let mut result = vec![0.; shape.rows * batch];
                    right
                        .multiply(matrix, &scaled, &mut result, || false)
                        .unwrap();
                    result
                });
                (left.join().unwrap(), right.join().unwrap())
            });
            assert_eq!(left_result, vec![1.]);
            assert_eq!(right_result, vec![0.5]);
            assert_eq!(
                used.load(Ordering::SeqCst),
                (shape.weight_bytes().unwrap() + shape.workspace_bytes(batch).unwrap()) as u64
            );
        }
        assert_eq!(
            work.multiply(&matrix, &[], &mut actual, || false)
                .unwrap_err()
                .kind,
            ErrorKind::InvalidInput
        );
        drop(work);
        drop(matrix);
        assert_eq!(used.load(Ordering::SeqCst), 0);
    }
    let large = MatrixShape {
        width: 1024,
        rows: 1024,
        dtype: WeightType::F32,
    };
    assert!(matches!(
        device.upload(
            large,
            |_, _| panic!("reader called before admission"),
            || false
        ),
        Err(Error {
            kind: ErrorKind::ResourceExhausted,
            ..
        })
    ));
    assert_eq!(used.load(Ordering::SeqCst), 0);
    let small = MatrixShape {
        width: 8,
        rows: 8,
        dtype: WeightType::F16,
    };
    assert!(
        device
            .upload(
                small,
                |_, _| Err(Error::new(ErrorKind::InvalidInput, "reader failed")),
                || false
            )
            .is_err()
    );
    assert_eq!(used.load(Ordering::SeqCst), 0);
    let workspace_too_large = MatrixShape {
        width: 32768,
        rows: 1_048_576,
        dtype: WeightType::F16,
    };
    // Input allocation succeeds; failure to admit output must release it.
    assert!(device.workspace(workspace_too_large, 1).is_err());
    assert_eq!(used.load(Ordering::SeqCst), 0);
    let reads = std::cell::Cell::new(0);
    let upload_shape = MatrixShape {
        width: 257,
        rows: 300,
        dtype: WeightType::F32,
    };
    let upload = device.upload(
        upload_shape,
        |_, chunk| {
            chunk.fill(0);
            reads.set(reads.get() + 1);
            Ok(())
        },
        || reads.get() > 0,
    );
    assert!(matches!(
        upload,
        Err(Error {
            kind: ErrorKind::Cancelled,
            ..
        })
    ));
    assert_eq!(reads.get(), 1);
    assert_eq!(used.load(Ordering::SeqCst), 0);
}

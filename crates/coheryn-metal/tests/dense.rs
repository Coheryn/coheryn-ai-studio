#![cfg(all(feature = "metal", target_os = "macos"))]
use coheryn_metal::{Device, Error, ErrorKind, MatrixShape, MemoryBudget, MemoryLease, Result, WeightType};
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};

struct Budget(Arc<AtomicU64>);
struct Lease { used: Arc<AtomicU64>, bytes: u64 }
impl Drop for Lease {
    fn drop(&mut self) { self.used.fetch_sub(self.bytes, Ordering::SeqCst); }
}
impl MemoryBudget for Budget {
    fn reserve(&self, bytes: u64) -> Result<Box<dyn MemoryLease>> {
        self.0.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |used| used.checked_add(bytes).filter(|v| *v <= 1 << 20))
            .map_err(|_| Error::new(ErrorKind::ResourceExhausted, "test GPU budget exhausted"))?;
        Ok(Box::new(Lease { used: Arc::clone(&self.0), bytes }))
    }
}

#[test]
#[ignore = "requires an admitted real unified-memory Metal device"]
fn independent_dense_references_admission_cancellation_and_cleanup() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!("../../../fixtures/gpu/metal-dense-v1.json")).unwrap();
    let absolute = fixture["precision"]["absolute_tolerance"].as_f64().unwrap();
    let relative = fixture["precision"]["relative_tolerance"].as_f64().unwrap();
    let used = Arc::new(AtomicU64::new(0));
    // No skip-to-pass when a GPU is absent: explicit qualification must fail.
    let device = Device::new(Arc::new(Budget(Arc::clone(&used)))).unwrap();
    println!("Metal device: {}", device.name());
    for case in fixture["cases"].as_array().unwrap() {
        let dtype = match case["dtype"].as_str().unwrap() {
            "f32" => WeightType::F32, "f16" => WeightType::F16, "bf16" => WeightType::Bf16,
            _ => panic!("unrecognized fixture dtype"),
        };
        let shape = MatrixShape { width: case["width"].as_u64().unwrap() as usize,
            rows: case["rows"].as_u64().unwrap() as usize, dtype };
        let batch = case["batch"].as_u64().unwrap() as usize;
        let hex = case["weights_le_hex"].as_str().unwrap();
        let bytes: Vec<_> = (0..hex.len()).step_by(2).map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap()).collect();
        let input: Vec<_> = case["input"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let expected: Vec<_> = case["expected"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let matrix = device.upload(shape, |offset, chunk| {
            chunk.copy_from_slice(&bytes[offset as usize..offset as usize + chunk.len()]); Ok(())
        }, || false).unwrap();
        let mut work = device.workspace(shape, batch).unwrap();
        assert_eq!(used.load(Ordering::SeqCst), (shape.weight_bytes().unwrap() + shape.workspace_bytes(batch).unwrap()) as u64);
        let mut actual = vec![999.; expected.len()];
        assert_eq!(work.multiply(&matrix, &input, &mut actual, || true).unwrap_err().kind, ErrorKind::Cancelled);
        assert!(actual.iter().all(|v| *v == 999.));
        let checks = std::cell::Cell::new(0);
        assert_eq!(work.multiply(&matrix, &input, &mut actual, || {
            checks.set(checks.get() + 1);
            checks.get() >= 3
        }).unwrap_err().kind, ErrorKind::Cancelled);
        // This cancellation is observed after a committed dispatch retires.
        assert!(actual.iter().all(|v| *v == 999.));
        for _ in 0..3 {
            work.multiply(&matrix, &input, &mut actual, || false).unwrap();
            for (actual, expected) in actual.iter().zip(&expected) {
                assert!(actual.is_finite() && (f64::from(*actual) - expected).abs() <= absolute + relative * expected.abs(), "{}: {actual} != {expected}", case["id"]);
            }
        }
        assert_eq!(work.multiply(&matrix, &[], &mut actual, || false).unwrap_err().kind, ErrorKind::InvalidInput);
        drop(work); drop(matrix);
        assert_eq!(used.load(Ordering::SeqCst), 0);
    }
    let large = MatrixShape { width: 1024, rows: 1024, dtype: WeightType::F32 };
    assert!(matches!(device.upload(large, |_, _| panic!("reader called before admission"), || false), Err(Error { kind: ErrorKind::ResourceExhausted, .. })));
    assert_eq!(used.load(Ordering::SeqCst), 0);
    let small = MatrixShape { width: 8, rows: 8, dtype: WeightType::F16 };
    assert!(device.upload(small, |_, _| Err(Error::new(ErrorKind::InvalidInput, "reader failed")), || false).is_err());
    assert_eq!(used.load(Ordering::SeqCst), 0);
}

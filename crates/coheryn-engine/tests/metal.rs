#![cfg(all(feature = "metal", target_os = "macos"))]
mod common;

use coheryn_engine::{
    Cancellation, ErrorCode, ParseLimits, cpu, inspect, memory::Budget, metal::MetalCompute,
    storage::Storage,
};
use std::{io::Write, sync::Arc};

#[test]
#[ignore = "requires an admitted real unified-memory Metal device"]
fn checked_tensor_storage_gpu_budget_and_mutation_boundary() {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    let bytes = common::tiny_model();
    file.write_all(&bytes).unwrap();
    file.flush().unwrap();
    let descriptor = inspect(file.path(), ParseLimits::default()).unwrap();
    let tensor = descriptor
        .tensors
        .iter()
        .find(|tensor| tensor.name == "token_embd.weight")
        .unwrap();
    let storage = Arc::new(Storage::open(file.path(), &descriptor).unwrap());
    let budget = Budget::new(8 << 20);
    let cpu_reservation = budget.reserve(1024).unwrap();
    let gpu = MetalCompute::new(budget.clone()).unwrap();
    let cancel = Cancellation::default();
    let matrix = gpu
        .upload(
            Arc::clone(&storage),
            descriptor.data_offset,
            tensor,
            &cancel,
        )
        .unwrap();
    let mut workspace = gpu.workspace(&matrix, 1).unwrap();
    let width = tensor.dimensions[0] as usize;
    let rows = tensor.dimensions[1] as usize;
    let held = 1024 + tensor.bytes + ((width + rows) * 4) as u64;
    assert_eq!(budget.stats().reserved, held);
    let input: Vec<_> = (0..width).map(|i| (i as f32 - 2.) / 8.).collect();
    let mut actual = vec![0.; rows];
    workspace
        .multiply(&matrix, &input, &mut actual, &cancel)
        .unwrap();
    // This checks storage binding against the CPU row contract. Independent
    // GPU numerical acceptance lives in the separately frozen kernel fixture.
    let start = (descriptor.data_offset + tensor.offset) as usize;
    let mut decoded = vec![0.; width * rows];
    cpu::decode(
        &bytes[start..start + tensor.bytes as usize],
        tensor.dtype,
        &mut decoded,
    )
    .unwrap();
    for (row, actual) in decoded.chunks_exact(width).zip(&actual) {
        let expected = cpu::dot(row, &input);
        assert!((*actual - expected).abs() <= 2e-5 + 2e-5 * expected.abs());
    }
    let mut invalid = tensor.clone();
    invalid.bytes -= 1;
    assert!(
        gpu.upload(
            Arc::clone(&storage),
            descriptor.data_offset,
            &invalid,
            &cancel
        )
        .is_err()
    );
    assert_eq!(budget.stats().reserved, held);
    cancel.cancel();
    assert_eq!(
        workspace
            .multiply(&matrix, &input, &mut actual, &cancel)
            .unwrap_err()
            .code,
        ErrorCode::Cancelled
    );
    // An already-uploaded buffer cannot bypass source mutation detection.
    file.write_all(&[0]).unwrap();
    file.flush().unwrap();
    assert_eq!(
        workspace
            .multiply(&matrix, &input, &mut actual, &Cancellation::default())
            .unwrap_err()
            .code,
        ErrorCode::ModelChanged
    );
    drop(workspace);
    drop(matrix);
    drop(gpu);
    drop(cpu_reservation);
    assert_eq!(budget.stats().reserved, 0);
}

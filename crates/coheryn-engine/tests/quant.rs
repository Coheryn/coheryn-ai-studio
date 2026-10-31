mod common;
use coheryn_engine::{
    cpu,
    quant::{self, QuantizedInput},
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    tolerances: Tolerances,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Tolerances {
    decode_absolute: f32,
    dot_absolute: f32,
    dot_relative: f32,
}
#[derive(Deserialize)]
struct Case {
    dtype: u32,
    bytes: Vec<u8>,
    activation_bytes: Vec<u8>,
    input: Vec<f32>,
    decoded: Vec<f32>,
    dot: f32,
}
#[test]
fn independent_packed_block_activation_and_dot_fixtures() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("../../../fixtures/quantized/kernels.json")).unwrap();
    assert_eq!(fixture.cases.len(), 100);
    verify_cases(fixture);
}
#[test]
fn independent_q5_k_blocks_activation_and_dot_match_reference() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("../../../fixtures/quantized/q5_k-v1.json")).unwrap();
    assert_eq!(fixture.cases.len(), 20);
    verify_cases(fixture);
}
fn verify_cases(fixture: Fixture) {
    for (index, case) in fixture.cases.into_iter().enumerate() {
        let mut decoded = vec![0.; case.decoded.len()];
        cpu::decode(&case.bytes, case.dtype, &mut decoded).unwrap();
        assert!(
            decoded
                .iter()
                .zip(&case.decoded)
                .all(|(a, b)| (a - b).abs() <= fixture.tolerances.decode_absolute),
            "decode {index}"
        );
        let mut activation = QuantizedInput::new(case.input.len()).unwrap();
        activation.prepare(&case.input, case.dtype).unwrap();
        assert_eq!(
            activation.encoded(),
            case.activation_bytes,
            "activation {index}"
        );
        let dot = quant::dot(&case.bytes, case.dtype, &activation).unwrap();
        assert!(
            (dot - case.dot).abs()
                <= fixture.tolerances.dot_absolute
                    + fixture.tolerances.dot_relative * case.dot.abs(),
            "dot {index}: {dot} != {}",
            case.dot
        );
    }
}
#[test]
fn hostile_shapes_scales_and_activation_reuse_fail_without_panics() {
    for dtype in [2, 6, 8, 12, 13, 14] {
        let width = if dtype < 12 { 32 } else { 256 };
        let bytes = quant::row_bytes(dtype, width).unwrap();
        for length in 0..bytes {
            assert!(cpu::decode(&vec![0; length], dtype, &mut vec![0.; width]).is_err());
        }
        let mut bad = vec![0; bytes];
        let offset = if dtype == 14 { 208 } else { 0 };
        bad[offset] = 0;
        bad[offset + 1] = 0x7c;
        assert!(cpu::decode(&bad, dtype, &mut vec![0.; width]).is_err());
        let mut input = QuantizedInput::new(width).unwrap();
        input.prepare(&vec![1.; width], dtype).unwrap();
        input.prepare(&vec![0.; width], dtype).unwrap();
        assert_eq!(quant::dot(&vec![0; bytes], dtype, &input).unwrap(), 0.);
        assert!(input.prepare(&vec![f32::NAN; width], dtype).is_err());
        assert!(quant::dot(&vec![0; bytes], dtype, &input).is_err());
    }
    assert!(quant::row_bytes(2, 31).is_err());
    assert!(quant::row_bytes(12, 32).is_err());
    assert!(quant::row_bytes(99, 32).is_err());
    assert!(quant::row_bytes(0, usize::MAX).is_err());
}

#[test]
fn packed_models_execute_with_bounded_tiles_workers_and_cancellation_cleanup() {
    use coheryn_engine::{
        Cancellation, FinishReason, GenerationOptions, Model, ResourceLimits,
        inference::ExecutionSettings,
    };
    for dtype in [2, 6, 8, 12, 13, 14] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("packed.gguf");
        std::fs::write(&path, common::packed_model(dtype)).unwrap();
        let model = Model::load(
            &path,
            ResourceLimits {
                context: 64,
                max_memory: 64 << 20,
            },
        )
        .unwrap();
        coheryn_engine::deep_validate(&path, &model.descriptor, &Cancellation::default()).unwrap();
        let baseline = model.memory().reserved;
        for threads in [1, 2, 4] {
            let execution = ExecutionSettings {
                threads,
                prefill_chunk_tokens: 8,
            };
            let admission = model
                .reserve_session_with_execution(64, 0, 256 << 10, execution)
                .unwrap();
            let result = model
                .generate_admitted(
                    "Hello",
                    &GenerationOptions {
                        max_tokens: 4,
                        ..Default::default()
                    },
                    admission,
                    &Cancellation::default(),
                    |_| Ok(()),
                )
                .unwrap();
            assert_eq!(result.text, "!!!!");
            assert_eq!(model.memory().reserved, baseline);
            assert_eq!(model.memory().session_buffer_capacity_bytes, 0);
        }
        let cancel = Cancellation::default();
        let c = cancel.clone();
        let admission = model
            .reserve_session_with_execution(
                64,
                0,
                16 << 10,
                ExecutionSettings {
                    threads: 2,
                    prefill_chunk_tokens: 16,
                },
            )
            .unwrap();
        let result = model
            .generate_admitted(
                "Hello",
                &GenerationOptions {
                    max_tokens: 32,
                    ..Default::default()
                },
                admission,
                &cancel,
                |_| {
                    c.cancel();
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(result.finish_reason, FinishReason::Cancelled);
        assert_eq!(model.memory().reserved, baseline);
        assert_eq!(model.memory().session_buffer_capacity_bytes, 0);
    }
}

#[test]
fn rope_matches_independent_nonzero_position_graph_fixtures() {
    #[derive(serde::Deserialize)]
    struct RopeCase {
        head_dim: usize,
        position: usize,
        base: f32,
        input: Vec<f32>,
        output: Vec<f32>,
    }
    #[derive(serde::Deserialize)]
    struct RopeCases {
        cases: Vec<RopeCase>,
    }
    let fixtures: RopeCases =
        serde_json::from_str(include_str!("../../../fixtures/quantized/rope.json")).unwrap();
    for case in fixtures.cases {
        let mut actual = case.input.clone();
        coheryn_engine::cpu::rope(&mut actual, case.head_dim, case.position, case.base);
        for (a, b) in actual.iter().zip(&case.output) {
            assert!(
                (a - b).abs() <= 1e-6 + 1e-6 * b.abs(),
                "RoPE dim={}, position={}: {a} vs {b}",
                case.head_dim,
                case.position
            );
        }
    }
}

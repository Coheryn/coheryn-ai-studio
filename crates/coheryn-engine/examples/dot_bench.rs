//! Explicit, bounded synthetic comparison against Coheryn's original reduction.
use coheryn_engine::cpu;
use serde_json::json;
use std::{hint::black_box, time::Instant};

const ROWS: usize = 64;
const REPEATS: usize = 128;
const SAMPLES: usize = 7;

fn reference(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (*x * *y) as f64).sum::<f64>() as f32
}
fn value(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    let bits = *state as u32;
    f32::from_bits((bits & 0x807f_ffff) | 126 << 23)
}
fn measure(kernel: fn(&[f32], &[f32]) -> f32, matrix: &[f32], input: &[f32]) -> f64 {
    let kernel = black_box(kernel);
    let start = Instant::now();
    let mut checksum = 0f64;
    for _ in 0..REPEATS {
        for row in matrix.chunks_exact(input.len()) {
            checksum += f64::from(kernel(black_box(row), black_box(input)));
        }
    }
    black_box(checksum);
    start.elapsed().as_secs_f64() * 1000.
}
fn main() {
    let mut state = 0x5702_4f00_1e5b_90e3;
    let mut cases = Vec::new();
    for (width, cancellation) in [
        (256, false),
        (896, false),
        (1024, false),
        (1536, false),
        (3584, false),
        (4864, false),
        (8960, false),
        (1024, true),
    ] {
        let mut matrix: Vec<_> = (0..ROWS * width).map(|_| value(&mut state)).collect();
        let mut input: Vec<_> = (0..width).map(|_| value(&mut state)).collect();
        if cancellation {
            input.fill(1.);
            matrix.fill(0.);
            for row in matrix.chunks_exact_mut(width) {
                row[..4].copy_from_slice(&[1e30, 1., -1e30, 1.]);
            }
        }
        for row in matrix.chunks_exact(width) {
            assert_eq!(
                cpu::dot(row, &input).to_bits(),
                reference(row, &input).to_bits()
            );
        }
        // Warm both paths before alternating their measurement order.
        black_box(measure(reference, &matrix, &input));
        black_box(measure(cpu::dot, &matrix, &input));
        let mut original_ms = Vec::new();
        let mut candidate_ms = Vec::new();
        for sample in 0..SAMPLES {
            let (original, candidate) = if sample % 2 == 0 {
                (
                    measure(reference, &matrix, &input),
                    measure(cpu::dot, &matrix, &input),
                )
            } else {
                let candidate = measure(cpu::dot, &matrix, &input);
                (measure(reference, &matrix, &input), candidate)
            };
            assert!(
                original.is_finite() && original > 0. && candidate.is_finite() && candidate > 0.
            );
            original_ms.push(original);
            candidate_ms.push(candidate);
        }
        cases.push(
            json!({"width":width,"cancellation":cancellation,"rows":ROWS,"repeats":REPEATS,
                          "samples":SAMPLES,"reference_ms":original_ms,"candidate_ms":candidate_ms,
                          "bitwise_parity":true}),
        );
    }
    println!(
        "{}",
        json!({"schema_version":1,"kind":"cpu-dot-observations","outcome":"pass",
        "architecture":std::env::consts::ARCH,"os":std::env::consts::OS,"threads":1,"seed":"57024f001e5b90e3",
        "cases":cases,"cache_condition":"warm synthetic inputs; hardware cache pressure uncontrolled",
        "scope":"Same-process alternating synthetic kernel measurements with reference bit parity. No model-throughput, latency-SLA or production qualification claim."})
    );
}

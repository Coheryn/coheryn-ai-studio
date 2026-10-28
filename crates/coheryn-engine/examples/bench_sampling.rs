//! Same-input warm microbenchmark against the project's previous greedy sorting path.
use coheryn_engine::sampling::{Sampler, Sampling};
use std::{hint::black_box, time::Instant};
fn main() {
    let vocabulary = 151936;
    let iterations = 384;
    let logits: Vec<f32> = (0..vocabulary)
        .map(|i| {
            let mut n = (i as u64).wrapping_mul(0x9e3779b97f4a7c15);
            n = (n ^ (n >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            ((n >> 32) & 65535) as f32 / 4096. - 8.
        })
        .collect();
    let old = |values: &[f32]| {
        let mut ids: Vec<usize> = (0..values.len()).collect();
        ids.sort_unstable_by(|a, b| values[*b].total_cmp(&values[*a]).then(a.cmp(b)));
        ids[0] as u32
    };
    let mut sampler = Sampler::new(Sampling::default()).unwrap();
    let expected = old(&logits);
    for _ in 0..8 {
        assert_eq!(sampler.sample(black_box(&logits)).unwrap(), expected);
    }
    let start = Instant::now();
    for _ in 0..iterations {
        assert_eq!(black_box(old(black_box(&logits))), expected);
    }
    let sorting_ms = start.elapsed().as_secs_f64() * 1000.;
    let start = Instant::now();
    for _ in 0..iterations {
        assert_eq!(
            black_box(sampler.sample(black_box(&logits)).unwrap()),
            expected
        );
    }
    let scan_ms = start.elapsed().as_secs_f64() * 1000.;
    println!("{}",serde_json::to_string_pretty(&serde_json::json!({
        "schema_version":1,"scope":"Unconstrained greedy sampler only; same fixed input and host, warmed CPU/file caches, uncontrolled host frequency. Not inference throughput or cross-platform evidence.",
        "os":std::env::consts::OS,"architecture":std::env::consts::ARCH,"profile":coheryn_engine::sampling::PROFILE,
        "vocabulary":vocabulary,"iterations":iterations,"reference":"71ab977 native greedy full-vocabulary sort",
        "reference_sort_ms":sorting_ms,"native_linear_scan_ms":scan_ms,"observed_ratio":sorting_ms/scan_ms,"selected_token_id":expected
    })).unwrap());
}

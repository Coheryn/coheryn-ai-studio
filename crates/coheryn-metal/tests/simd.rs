#![cfg(all(feature = "metal", target_os = "macos"))]
mod common;
#[path = "common/dense_fixture.rs"]
mod dense;
#[path = "common/packed_fixture.rs"]
mod packed;

#[test]
#[ignore = "requires an admitted real unified-memory Metal device"]
fn cooperative_simd_matches_independent_fixtures_and_lifecycle_contracts() {
    let kernel = coheryn_metal::KernelProfile::SimdCompensated;
    dense::dense_profile(kernel);
    packed::packed_profile(kernel);
    println!(
        "PASS explicit SimdCompensated: 6 dense and 120 packed cases; no model/performance qualification"
    );
}

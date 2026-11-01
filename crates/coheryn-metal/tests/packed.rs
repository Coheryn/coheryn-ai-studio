#![cfg(all(feature = "metal", target_os = "macos"))]
mod common;
#[path = "common/packed_fixture.rs"]
mod fixture;

#[test]
#[ignore = "requires an admitted real unified-memory Metal device"]
fn independent_packed_weights_preserve_f32_activations_and_release_reservations() {
    fixture::packed_profile(coheryn_metal::KernelProfile::Reference);
}

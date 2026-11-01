#![cfg(all(feature = "metal", target_os = "macos"))]
mod common;
#[path = "common/dense_fixture.rs"]
mod fixture;

#[test]
#[ignore = "requires an admitted real unified-memory Metal device"]
fn independent_dense_references_admission_cancellation_and_cleanup() {
    fixture::dense_profile(coheryn_metal::KernelProfile::Reference);
}

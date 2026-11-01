#![cfg(all(feature = "metal", target_os = "macos"))]
#[path = "common/metal_model_fixture.rs"]
mod fixture;

#[test]
#[ignore = "requires pinned pretrained assets, native Metal and the owned host guard"]
fn native_qwen35_gpu_graph_matches_frozen_answers_and_cpu_arrays() {
    fixture::exercise(coheryn_engine::metal::KernelProfile::Reference);
}

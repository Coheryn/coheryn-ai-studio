// Coheryn-owned experimental cooperative kernel. Compiled only when selected.
// The runtime prepends the owned dense.metal decoder definitions.
void compensated_add(thread float& total, thread float& correction, float value) {
    float next = total + value;
    correction += abs(total) >= abs(value)
        ? (total - next) + value : (value - next) + total;
    total = next;
}

// One SIMD group owns one row/lane output. Adjacent threads read adjacent
// columns, retaining packed weights and F32 products. No shared scratch or
// atomics are needed; both components of each compensated partial sum are
// merged in a deterministic tree rather than discarded by a plain simd_sum.
kernel void dense_simd_compensated(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant Shape& shape [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint lanes [[threads_per_simdgroup]]) {
    if (group.x >= shape.tile_rows || group.y >= shape.batch) return;
    uint row = shape.first_row + group.x;
    if (row >= shape.rows) return;
    ulong activation = ulong(group.y) * shape.width;
    float total = 0.0f;
    float correction = 0.0f;
    for (uint column = lane; column < shape.width; column += lanes) {
        float product = weight_value(weights, shape, row, column) * input[activation + column];
        compensated_add(total, correction, product);
    }
    for (uint stride = lanes / 2; stride != 0; stride /= 2) {
        float upper = simd_shuffle_down(total, stride);
        float lower = simd_shuffle_down(correction, stride);
        if (lane < stride) {
            compensated_add(total, correction, upper);
            compensated_add(total, correction, lower);
        }
    }
    if (lane == 0) output[ulong(group.y) * shape.rows + row] = total + correction;
}

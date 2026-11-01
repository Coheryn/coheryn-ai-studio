// Coheryn-owned reference kernels. Keep precision policy frozen before tuning.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

struct Shape {
    uint width;
    uint rows;
    uint batch;
    uint dtype;
    uint first_row;
    uint tile_rows;
};

// Decode half subnormals without relying on native half flush-to-zero behavior.
float half_value(ushort bits) {
    uint sign = uint(bits & 0x8000) << 16;
    uint exponent = (bits >> 10) & 31;
    uint mantissa = bits & 1023;
    if (exponent == 0) {
        float value = float(mantissa) * 0x1p-24f;
        return (sign == 0) ? value : -value;
    }
    uint result = sign | (mantissa << 13);
    result |= (exponent == 31) ? 0x7f800000 : ((exponent + 112) << 23);
    return as_type<float>(result);
}

kernel void dense_reference(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant Shape& shape [[buffer(3)]],
    uint2 index [[thread_position_in_grid]]) {
    if (index.x >= shape.tile_rows || index.y >= shape.batch) return;
    uint row = shape.first_row + index.x;
    if (row >= shape.rows) return;
    // Use 64-bit offsets: a valid shape may exceed the 32-bit byte range.
    ulong base = ulong(row) * shape.width;
    ulong activation = ulong(index.y) * shape.width;
    float total = 0.0f;
    float correction = 0.0f;
    for (uint column = 0; column < shape.width; ++column) {
        ulong offset = base + column;
        float weight;
        if (shape.dtype == 0) {
            weight = reinterpret_cast<device const float*>(weights)[offset];
        } else {
            ushort bits = reinterpret_cast<device const ushort*>(weights)[offset];
            weight = shape.dtype == 1 ? half_value(bits) : as_type<float>(uint(bits) << 16);
        }
        // No contraction/fast math: round products to F32, then compensate the
        // accumulation. GPU F32 precision is qualified, not called CPU-bitwise.
        float product = weight * input[activation + column];
        float next = total + product;
        correction += abs(total) >= abs(product)
            ? (total - next) + product : (product - next) + total;
        total = next;
    }
    output[ulong(index.y) * shape.rows + row] = total + correction;
}

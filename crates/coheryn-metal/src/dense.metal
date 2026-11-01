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
    uint row_bytes;
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

ushort read_u16(device const uchar* p) {
    return ushort(p[0]) | (ushort(p[1]) << 8);
}

// Native packed reads follow Coheryn's checked block layout contract. Decode
// one scalar in registers; encoded weights remain packed for their lifetime.
float weight_value(device const uchar* weights, constant Shape& shape, uint row, uint column) {
    device const uchar* start = weights + ulong(row) * shape.row_bytes;
    if (shape.dtype == 0) return reinterpret_cast<device const float*>(start)[column];
    if (shape.dtype == 1) return half_value(read_u16(start + ulong(column) * 2));
    if (shape.dtype == 30) return as_type<float>(uint(read_u16(start + ulong(column) * 2)) << 16);
    if (shape.dtype == 2 || shape.dtype == 6 || shape.dtype == 8) {
        uint bytes = shape.dtype == 2 ? 18 : (shape.dtype == 6 ? 22 : 34);
        device const uchar* block = start + ulong(column / 32) * bytes;
        uint i = column % 32;
        float d = half_value(read_u16(block));
        if (shape.dtype == 8) return d * float(as_type<char>(block[2 + i]));
        uint packed = block[(shape.dtype == 2 ? 2 : 6) + i % 16];
        uint q = i < 16 ? packed & 15 : packed >> 4;
        if (shape.dtype == 2) return d * float(int(q) - 8);
        uint high = uint(block[2]) | (uint(block[3]) << 8)
            | (uint(block[4]) << 16) | (uint(block[5]) << 24);
        q |= ((high >> i) & 1) << 4;
        return d * float(int(q) - 16);
    }
    uint bytes = shape.dtype == 12 ? 144 : (shape.dtype == 13 ? 176 : 210);
    device const uchar* block = start + ulong(column / 256) * bytes;
    uint i = column % 256;
    if (shape.dtype == 14) {
        uint segment = i / 128;
        uint group = (i % 128) / 32;
        uint lane = i % 32;
        uint low = block[segment * 64 + (group % 2) * 32 + lane];
        low = group < 2 ? low & 15 : low >> 4;
        uint high = (block[128 + segment * 32 + lane] >> (2 * group)) & 3;
        float d = half_value(read_u16(block + 208));
        float scale = float(as_type<char>(block[192 + i / 16]));
        return (d * scale) * float(int(low | (high << 4)) - 32);
    }
    uint group = i / 32;
    uint lane = i % 32;
    device const uchar* scales = block + 4;
    uint scale;
    uint minimum;
    if (group < 4) {
        scale = scales[group] & 63;
        minimum = scales[group + 4] & 63;
    } else {
        scale = (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4);
        minimum = (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
    }
    uint packed = block[(shape.dtype == 13 ? 48 : 16) + (group / 2) * 32 + lane];
    uint q = group % 2 == 0 ? packed & 15 : packed >> 4;
    if (shape.dtype == 13) q |= ((block[16 + lane] >> group) & 1) << 4;
    float ds = half_value(read_u16(block)) * float(scale);
    float dm = half_value(read_u16(block + 2)) * float(minimum);
    return ds * float(q) - dm;
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
    ulong activation = ulong(index.y) * shape.width;
    float total = 0.0f;
    float correction = 0.0f;
    for (uint column = 0; column < shape.width; ++column) {
        float weight = weight_value(weights, shape, row, column);
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

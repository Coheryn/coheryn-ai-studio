#!/usr/bin/env python3
"""Freeze/check independent small dense GPU references using only Python stdlib.

The oracle rounds each product to binary32, accumulates sequentially in binary64,
then rounds the result to binary32. It never imports or executes Coheryn kernels.
"""
import argparse
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'fixtures/gpu/metal-dense-v1.json'


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def expected_fixture():
    cases = []
    for name, dtype, width, rows, batch in [
        ('f32-cancellation', 'f32', 3, 1, 1),
        ('f32-rectangular-tail', 'f32', 33, 5, 3),
        ('f16-subnormal-signs', 'f16', 7, 3, 2),
        ('f16-wide-tail', 'f16', 257, 2, 2),
        ('bf16-rectangular', 'bf16', 35, 3, 3),
        ('f32-wide-reduction', 'f32', 2049, 2, 1),
    ]:
        weights = [f32(((i * 13 + 7) % 101 - 50) / 32) for i in range(width * rows)]
        inputs = [f32(((i * 7 + 3) % 67 - 33) / 16) for i in range(width * batch)]
        if name == 'f32-cancellation':
            weights, inputs = [1e8, 1., -1e8], [1., 1., 1.]
        if name == 'f16-subnormal-signs':
            weights[:7] = [2 ** -24, -(2 ** -24), 2 ** -14, -(2 ** -14), 0., -0., 1.]
        if dtype == 'f32':
            encoded = b''.join(struct.pack('<f', w) for w in weights)
            decoded = list(struct.unpack('<' + 'f' * len(weights), encoded))
        elif dtype == 'f16':
            encoded = b''.join(struct.pack('<e', w) for w in weights)
            decoded = list(struct.unpack('<' + 'e' * len(weights), encoded))
        else:
            encoded = b''.join(struct.pack('<f', w)[2:] for w in weights)
            decoded = [struct.unpack('<f', b'\0\0' + encoded[i:i + 2])[0]
                       for i in range(0, len(encoded), 2)]
        outputs = []
        for lane in range(batch):
            for row in range(rows):
                # Explicit loop: do not depend on Python's compensated sum().
                total = 0.
                for column in range(width):
                    total += f32(decoded[row * width + column] * inputs[lane * width + column])
                outputs.append(f32(total))
        cases.append({'id': name, 'dtype': dtype, 'width': width, 'rows': rows,
                      'batch': batch, 'weights_le_hex': encoded.hex(), 'input': inputs,
                      'expected': outputs})
    return {'schema_version': 1, 'kind': 'metal-dense-reference',
            'oracle': 'Python stdlib: F32 products, sequential F64 accumulation, final F32 rounding',
            'precision': {'absolute_tolerance': 2e-5, 'relative_tolerance': 2e-5,
                          'comparison': 'abs(actual-expected) <= absolute + relative*abs(expected); every element finite'},
            'scope': 'Dense F32/F16/BF16 primitives only; no packed kernels, model or throughput qualification.',
            'cases': cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='explicitly create new frozen fixture; never overwrite')
    args = parser.parse_args()
    expected = expected_fixture()
    if args.write:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        with FIXTURE.open('x', encoding='utf-8') as stream:
            json.dump(expected, stream, indent=2, allow_nan=False)
            stream.write('\n')
    elif json.loads(FIXTURE.read_text()) != expected:
        raise SystemExit('Metal reference fixture changed; review the precision contract and independent oracle')
    print(f'PASS {len(expected["cases"])} independent dense reference cases; no GPU/model execution claimed')


if __name__ == '__main__':
    main()

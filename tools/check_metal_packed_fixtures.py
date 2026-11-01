#!/usr/bin/env python3
"""Freeze/check GPU packed-weight references from independently decoded blocks.

The immutable source fixtures supply decoded weights. This tool only performs
F32 products and sequential F64 accumulation; it never decodes a quantized block
or imports Coheryn execution code. F32 activations are intentionally preserved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'fixtures/gpu/metal-packed-v1.json'
SOURCES = ('fixtures/quantized/kernels.json', 'fixtures/quantized/q5_k-v1.json')


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def expected_fixture():
    sources, cases = [], []
    for source_id, name in enumerate(SOURCES):
        raw = (ROOT / name).read_bytes()
        source = json.loads(raw)
        sources.append({'path': name, 'sha256': hashlib.sha256(raw).hexdigest()})
        original = source['cases']
        for index, case in enumerate(original):
            siblings = [i for i, sibling in enumerate(original)
                        if sibling['dtype'] == case['dtype'] and sibling['blocks'] == case['blocks']]
            position = siblings.index(index)
            rows = [siblings[(position + step) % len(siblings)] for step in range(3)]
            inputs = [[f32(value) for value in case['input']],
                      [f32(value * (-1. if i % 3 else .5)) for i, value in enumerate(case['input'])]]
            outputs = []
            for values in inputs:
                for row in rows:
                    total = 0.
                    for weight, value in zip(original[row]['decoded'], values, strict=True):
                        total += f32(f32(weight) * value)
                    outputs.append(f32(total))
            cases.append({'id': f'source-{source_id}-case-{index}', 'source': source_id,
                          'rows': rows, 'width': len(case['decoded']), 'dtype': case['dtype'],
                          'input_case': index,
                          'expected': outputs})
    return {'schema_version': 1, 'kind': 'metal-packed-f32-operands-reference',
            'sources': sources,
            'activation_lanes': ['Original input rounded to F32', 'F32(original[i] * (0.5 if i % 3 == 0 else -1))'],
            'oracle': 'Independent frozen decoded weights; Python stdlib F32 products, sequential F64 sum, final F32 rounding',
            'precision': {'absolute_tolerance': 2e-5, 'relative_tolerance': 2e-5,
                          'comparison': 'Every finite element: abs(actual-expected) <= absolute + relative*abs(expected)'},
            'scope': 'Packed Q4_0/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K with preserved F32 activations only; no Q8 activation dot, model, throughput or release qualification.',
            'cases': cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='create a new fixture exclusively; never overwrite')
    args = parser.parse_args()
    expected = expected_fixture()
    if args.write:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        with FIXTURE.open('x', encoding='utf-8') as stream:
            json.dump(expected, stream, indent=2, allow_nan=False)
            stream.write('\n')
    elif json.loads(FIXTURE.read_text()) != expected:
        raise SystemExit('Packed GPU references changed; review original block provenance and frozen precision contract')
    print(f'PASS {len(expected["cases"])} independent packed GPU reference cases; no GPU execution claimed')


if __name__ == '__main__':
    main()

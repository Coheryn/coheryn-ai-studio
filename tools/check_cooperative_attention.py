#!/usr/bin/env python3
"""Recompute the authored tile-boundary oracle with independent standard-library math."""
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]

def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]

def reference():
    rows, hidden, heads = 67, 4, 2
    q = [f32(((i*7) % 29-14)/32) for i in range(rows*hidden)]
    k = [f32(((i*11) % 31-15)/64) for i in range(rows*hidden)]
    v = [f32(((i*13) % 37-18)/16) for i in range(rows*hidden)]
    dimension = hidden//heads
    scale = f32(1/f32(math.sqrt(dimension)))
    output = []
    for row in range(rows):
        for head in range(heads):
            scores = [f32(f32(sum(f32(q[row*hidden+head*dimension+j]*k[key*hidden+head*dimension+j])
                      for j in range(dimension)))*scale) for key in range(rows)]
            maximum = max(scores)
            scores = [f32(math.exp(f32(s-maximum))) for s in scores]
            inverse = f32(1/sum(scores))
            scores = [f32(s*inverse) for s in scores]
            output.extend(f32(sum(f32(scores[key]*v[key*hidden+head*dimension+j]) for key in range(rows)))
                          for j in range(dimension))
    return dict(rows=rows, hidden=hidden, heads=heads, q=q, k=k, v=v, expected=output)

def main():
    fixture = json.loads((ROOT/'fixtures/vlm/cooperative-attention-v1.json').read_text())
    assert fixture['schema_version'] == 1 and fixture['absolute_tolerance'] == 2e-6
    for key, value in reference().items():
        assert fixture[key] == value, f'frozen independent attention field differs: {key}'
    print('PASS independent 67-row noncausal attention fixture: two complete tiles and a three-row tail.')

if __name__ == '__main__':
    main()

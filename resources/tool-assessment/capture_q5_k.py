"""Freeze independent Q5_K decoding and dot fixtures; never called by native CI."""
import copy
import json
import struct
from qwen35_gguf_weights import ROOT, np, dequantize, sha
from gguf.constants import GGMLQuantizationType

source = ROOT / 'fixtures/quantized/kernels.json'
original = json.loads(source.read_text())
cases = []
for item in original['cases']:
    if item['dtype'] != 12:
        continue
    case = copy.deepcopy(item)
    data = bytearray()
    for block in range(case['blocks']):
        pattern = case['pattern']
        if pattern == 0:
            data.extend(bytes(176))
            continue
        # Independently exercise every packed coefficient and high-bit lane.
        packed = bytearray((i * 37 + block * 19 + pattern * 23) % 256 for i in range(176))
        packed[:4] = struct.pack('<ee', (block + 1) / 512, pattern / 2048)
        data.extend(packed)
    decoded = dequantize(np.frombuffer(data, dtype=np.uint8), GGMLQuantizationType.Q5_K).reshape(-1)
    activation = np.frombuffer(bytes(case['activation_bytes']), dtype=np.uint8).reshape(-1, 292)
    scales = activation[:, :4].copy().view('<f4')
    values = activation[:, 4:260].copy().view(np.int8).astype(np.float32)
    expanded = (scales * values).reshape(-1)
    case.update(dtype=13, bytes=list(data), decoded=decoded.tolist(),
                dot=float(np.sum(decoded * expanded, dtype=np.float64)))
    cases.append(case)
output = ROOT / 'fixtures/quantized/q5_k-v1.json'
with output.open('x') as stream:
    json.dump({'schema_version': 1, 'purpose': 'Independent assessment data for mixed Qwen3.5 GGUF Q5_K components.',
               'producer': 'resources/tool-assessment/capture_q5_k.py',
               'tool': 'resources/tool-assessment/modern-gguf-tool-v1.json',
               'input_fixture_sha256': sha(source), 'tolerances': original['tolerances'], 'cases': cases},
              stream, separators=(',', ':'))
    stream.write('\n')
print('Captured', len(cases), 'independent Q5_K cases')

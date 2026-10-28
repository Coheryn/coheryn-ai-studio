#!/usr/bin/env python3
"""Compare native CPU vision traces to frozen independent tensors; never run another executor."""
import argparse
import array
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOLERANCES = {'absolute': 0.003, 'relative': 0.001, 'normalized_l2': 0.0005}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def vector(path, count):
    if path.stat().st_size != count * 4:
        raise ValueError(f'wrong tensor byte count: {path}')
    values = array.array('f', path.read_bytes())
    if sys.byteorder != 'little':
        values.byteswap()
    if not all(math.isfinite(x) for x in values):
        raise ValueError(f'nonfinite tensor: {path}')
    return values


def compare(actual, expected):
    outside = 0
    square = norm = maximum = 0.
    for a, b in zip(actual, expected):
        delta = abs(a-b)
        maximum = max(maximum, delta)
        outside += delta > TOLERANCES['absolute'] + TOLERANCES['relative'] * abs(b)
        square += delta * delta
        norm += b * b
    relative = math.sqrt(square / max(norm, 1e-30))
    return {'maximum_absolute_error': maximum, 'outside_tolerance': outside, 'normalized_l2': relative,
            'passed': outside == 0 and relative <= TOLERANCES['normalized_l2']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT/'target/release/examples/vision_probe')
    args = parser.parse_args()
    binary = args.binary.resolve()
    binary_sha256 = sha(binary)
    original_path = ROOT/'fixtures/vlm/image-conditioned-reference-v1.json'
    extra_path = ROOT/'fixtures/vlm/qwen2-vision-extra-v1.json'
    original = json.loads(original_path.read_text())
    extra = json.loads(extra_path.read_text())
    model = next(m for m in json.loads((ROOT/'fixtures/vlm/models.json').read_text())['models'] if m['id'] == 'qwen2-vl-2b')
    component = next(c for c in model['components'] if c['role'] == 'vision-projector')
    weights = ROOT/'.cache/vlm/full/qwen2-vl-2b'/component['file']
    assert weights.stat().st_size == component['bytes'] and sha(weights) == component['sha256']
    assert extra['component_sha256'] == component['sha256']
    cases = []
    for name in ['left-red', 'left-blue']:
        case = next(c for c in original['cases'] if c['model'] == 'qwen2-vl-2b' and c['id'] == name)
        records = [t for t in case['tensors'] if t['name'].startswith('vision_0_layer_out-') or t['name'] == 'image_0_embeddings']
        cases.append((name, case['identity']['images'][0]['canonical'], records,
                      ROOT/'.cache/vlm-qualified/qwen2-vl-2b'/name, 1, 32))
    for case in extra['cases']:
        name = Path(case['canonical']).stem
        cases.append((name, case['canonical'], case['tensors'], ROOT/'.cache/vision-extra'/name, 1, 32))
    # Exercise the owned worker pool and smaller weight reuse tiles with identical weights.
    cases.append(('left-red-workers2', *cases[0][1:4], 2, 7))
    observations = []
    for name, image, records, reference_directory, threads, tile in cases:
        directory = ROOT/'.cache/m2-vision'/name
        directory.mkdir(parents=True, exist_ok=True)
        request = {'image': str(ROOT/image), 'options': {'threads': threads, 'tile_tokens': tile, 'max_patches': 64},
                   'max_memory': 128 << 20, 'trace_directory': str(directory)}
        request_path = directory/'request.json'
        request_path.write_text(json.dumps(request))
        raw = subprocess.check_output([str(binary), str(weights), str(request_path)], cwd=ROOT, text=True, timeout=900)
        result = json.loads(raw)
        (directory/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        assert result['result']['model_sha256'] == component['sha256']
        assert result['result']['image_to_text_supported'] is False
        after = result['memory_after_release']
        assert after['reserved'] == after['session_buffer_capacity_bytes'] == 0
        memory = result['result']['memory_while_output_retained']
        assert memory['peak_session_buffer_capacity_bytes'] <= memory['peak_reserved']
        comparisons = []
        for record in records:
            reference = reference_directory/(record['name']+'.f32')
            assert sha(reference) == record['sha256'], f'frozen raw tensor changed: {reference}'
            native_name = ('projected_embeddings' if record['name'] == 'image_0_embeddings'
                           else record['name'].removeprefix('vision_0_'))
            actual = directory/(native_name+'.f32')
            item = compare(vector(actual, record['count']), vector(reference, record['count']))
            item.update(name=native_name, count=record['count'], native_sha256=sha(actual), reference_sha256=record['sha256'])
            comparisons.append(item)
        assert len(comparisons) == 33
        if name == 'left-red-workers2':
            previous = {c['name']: c['native_sha256'] for c in observations[0]['comparisons']}
            assert all(c['native_sha256'] == previous[c['name']] for c in comparisons), 'thread/tile result changed'
        assert sha(binary) == binary_sha256, 'native executable changed during qualification'
        passed = all(c['passed'] for c in comparisons)
        observations.append({'id': name, 'image': image, 'image_sha256': sha(ROOT/image), 'threads': threads,
                             'tile_tokens': tile, 'native': result, 'comparisons': comparisons, 'passed': passed})
        print(f'{"PASS" if passed else "FAIL"} {name}: {len(comparisons)} full vision/projector tensors', flush=True)
    report = {'schema_version': 1, 'scope': 'Qwen2-VL F16 native CPU vision encoder and merger, v2 F16 lookup-GELU precision profile',
              'host': {'os': platform.platform(), 'architecture': platform.machine()},
              'binary_sha256': binary_sha256, 'model_sha256': component['sha256'], 'source_sha256': sha(Path(__file__)),
              'references': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p)} for p in [original_path, extra_path]],
              'tolerances': TOLERANCES, 'cases': observations, 'passed': all(c['passed'] for c in observations),
              'image_to_text_supported': False,
              'measurement_limits': ['Uncontrolled cache/host conditions; descriptive timing only.',
                                     'Engine buffers/reservations exclude allocator overhead, OS page cache and complete process residency.',
                                     'No language model, GPU, server admission or multi-image session is qualified.']}
    (ROOT/'fixtures/qualification/m2-vision-v2.json').write_text(json.dumps(report, indent=2)+'\n')
    if not report['passed']:
        raise SystemExit('Native numerical qualification failed; frozen tolerances were not changed.')


if __name__ == '__main__':
    main()

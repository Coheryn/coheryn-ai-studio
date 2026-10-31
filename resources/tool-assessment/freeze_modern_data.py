"""Bind existing independent captures and measure quantization differences; no native execution."""
import array
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def vector(path):
    values = array.array('f', path.read_bytes())
    if sys.byteorder != 'little':
        values.byteswap()
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError('invalid independent numeric data')
    return values


def main():
    profiles = [label + '-text' for label in ('f16', 'q8_0', 'q4_0', 'q4_k_m', 'q6_k_m')] + ['f16-image', 'q4_k_m-image']
    inventory = read(ROOT / 'fixtures/cpu-qualification/references-v1.json')
    inventory['purpose'] = 'Immutable combined historical and modern independently captured data; no native expected-value generation.'
    inventory['modern_origin'] = 'resources/tool-assessment/capture_qwen35_gguf.py; exact per-file weights and folded GGUF coefficients'
    for name in ('fixtures/qwen35/gguf/models-v1.json', 'fixtures/qwen35/sharded-layout-v1.json', 'fixtures/quantized/q5_k-v1.json'):
        inventory['manifests'][name] = sha(ROOT / name)
    references = {}
    for profile in profiles:
        path = ROOT / 'fixtures/qwen35/gguf' / (profile + '.reference-v1.json')
        ref = read(path)
        if ref['native_executor_invoked'] or ref['publisher_framework_parity']:
            raise ValueError('unexpected reference provenance')
        references[profile] = ref
        inventory['manifests'][path.relative_to(ROOT).as_posix()] = sha(path)
        entries = list(ref['encoder_checkpoints'])
        for case in ref['cases']:
            entries += case['checkpoints'] + case['teacher_forced']
        seen = set()
        for entry in entries:
            name = entry['name']
            if name in seen:
                raise ValueError('duplicate independent array identity')
            seen.add(name)
            source = ROOT / '.cache/qwen35-gguf-reference' / profile / (name + '.f32')
            if source.stat().st_size != entry['bytes'] or sha(source) != entry['sha256']:
                raise ValueError('independent capture identity mismatch')
            inventory['files'].append({'path': 'references/qwen35-gguf-' + profile + '/' + source.name,
                                       'source': source.relative_to(ROOT).as_posix(), 'bytes': entry['bytes'], 'sha256': entry['sha256']})
    quality = {'schema_version': 1, 'purpose': 'Quantization quality observations on two short text cases; separate from executor correctness.',
               'semantic_quality_claim': False, 'baseline': 'f16-text.reference-v1.json', 'cases': []}
    baseline = references['f16-text']
    for label in ('q8_0', 'q4_0', 'q4_k_m', 'q6_k_m'):
        ref = references[label + '-text']
        for actual, expected in zip(ref['cases'], baseline['cases']):
            if actual['prompt_token_ids'] != expected['prompt_token_ids'] or actual['id'] != expected['id']:
                raise ValueError('incomparable quality inputs')
            name = expected['teacher_forced'][0]['name'] + '.f32'
            a = vector(ROOT / '.cache/qwen35-gguf-reference' / (label + '-text') / name)
            b = vector(ROOT / '.cache/qwen35-gguf-reference/f16-text' / name)
            if len(a) != len(b):
                raise ValueError('quality shape mismatch')
            top_a = sorted(range(len(a)), key=lambda i: (-a[i], i))[:10]
            top_b = sorted(range(len(b)), key=lambda i: (-b[i], i))[:10]
            quality['cases'].append({'label': label, 'case': actual['id'], 'first_logits_values': len(a),
                'first_logits_normalized_l2': math.sqrt(math.fsum((x-y)**2 for x,y in zip(a,b)) / max(math.fsum(x*x for x in b), 1e-30)),
                'top1_same': top_a[0] == top_b[0], 'top10_overlap': len(set(top_a) & set(top_b)),
                'greedy_same': actual['greedy_token_ids'] == expected['greedy_token_ids'],
                'greedy_token_ids': actual['greedy_token_ids'], 'greedy_text': actual['greedy_text'],
                'reference_sha256': inventory['manifests']['fixtures/qwen35/gguf/' + label + '-text.reference-v1.json']})
    quality_path = ROOT / 'fixtures/qwen35/gguf/quantization-quality-v1.json'
    write(quality_path, quality)
    inventory['manifests'][quality_path.relative_to(ROOT).as_posix()] = sha(quality_path)
    write(ROOT / 'fixtures/cpu-qualification/modern-references-v1.json', inventory)
    print('Bound', len(inventory['files']), 'independent arrays,', sum(i['bytes'] for i in inventory['files']), 'data bytes')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Qualify the native E1 encoder against frozen, independently captured E0 fixtures.

Build first: ./tools/cargo build --release -p coheryn-engine --example embedding_probe
Models remain in .cache/models; traces remain in .cache/embedding/native.
No reference inference runtime is executed or linked by this check.
"""
import argparse
import datetime
import json
import math
import platform
import subprocess
from pathlib import Path

from fetch_model import digest

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache/embedding/native'


def compare(actual, expected, absolute, relative, l2_limit=None):
    assert len(actual) == len(expected) and all(math.isfinite(v) for v in actual)
    differences = [abs(a-b) for a, b in zip(actual, expected, strict=True)]
    outside = sum(d > absolute + relative * abs(b)
                  for d, b in zip(differences, expected, strict=True))
    l2 = math.sqrt(sum(d*d for d in differences))
    assert outside == 0, f'{outside} values exceed frozen tolerances; max={max(differences)}'
    normalized_l2 = l2 / max(math.sqrt(sum(v*v for v in expected)), 1e-30)
    if l2_limit is not None:
        assert normalized_l2 <= l2_limit, f'Normalized L2 {normalized_l2} exceeds {l2_limit}'
    return {'max_absolute': max(differences), 'l2_distance': l2,
            'normalized_l2': normalized_l2, 'outside_tolerance': outside}


def run(executable, model, name, inputs, options, trace=False, tokenize=False):
    request = CACHE / f'{name}.request.json'
    output = CACHE / f'{name}.jsonl'
    request.write_text(json.dumps({'inputs': inputs, 'options': options,
                                  'trace': trace, 'tokenize_only': tokenize})+'\n')
    with output.open('w') as stream:
        subprocess.run([str(executable), str(model), str(request)], cwd=ROOT,
                       stdout=stream, check=True, timeout=300)
    return [json.loads(line) for line in output.read_text().splitlines()]


def result(events):
    output = next(e['result'] for e in events if e['event'] == 'result')
    cleanup = next(e['memory'] for e in events if e['event'] == 'cleanup')
    assert cleanup['session_buffer_capacity_bytes'] == 0
    assert cleanup['reserved'] + output['plan']['output_bytes'] == output['memory']['reserved']
    assert cleanup['peak_reserved'] <= cleanup['limit']
    assert cleanup['peak_session_buffer_capacity_bytes'] <= output['plan']['additional_reserved_bytes']
    return output, cleanup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, default=ROOT / 'target/release/examples/embedding_probe')
    parser.add_argument('--output', type=Path, default=ROOT / 'fixtures/qualification/e1-embeddings.json')
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    contract = json.loads((ROOT / 'fixtures/embedding/reference-contract.json').read_text())
    tolerance = contract['tolerances']
    manifest = json.loads((ROOT / 'fixtures/embedding/models.json').read_text())
    report = {'schema_version': 1, 'gate': 'E1-native-CPU',
              'captured_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'hardware': {'machine': platform.machine(), 'platform': platform.platform(),
                           'processor': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
                           if platform.system() == 'Darwin' else platform.processor()},
              'cargo': subprocess.check_output([str(ROOT / 'tools/cargo'), '--version'], text=True).strip(),
              'rust_toolchain_pin': (ROOT / 'rust-toolchain.toml').read_text().strip(),
              'executable_sha256': digest(args.executable),
              'reference_contract': 'fixtures/embedding/reference-contract.json',
              'reference_contract_sha256': digest(ROOT / 'fixtures/embedding/reference-contract.json'),
              'cache_conditions': 'uncontrolled; no cold-cache claim',
              'production_dependencies': 'Coheryn Rust encoder and kernels; no external inference runtime',
              'models': [], 'limitations': ['macOS ARM64 CPU evidence only',
                                          'Embedding CLI/API and GPU qualification remain separate gates',
                                          'RSS, faults and I/O are not captured by this numerical probe',
                                          'Times include checkpoint serialization when tracing is enabled']}
    for model in manifest['models']:
        path = ROOT / '.cache/models' / model['file']
        assert path.stat().st_size == model['bytes'] and digest(path) == model['sha256']
        reference = json.loads((ROOT / model['evidence']).read_text())
        inputs = [case['input'] for case in reference['cases']]
        name = model['id']
        reference_options = {'profile': 'reference', 'threads': 1}
        token_events = run(args.executable, path, name+'-tokens', inputs, reference_options, tokenize=True)
        actual_tokens = token_events[0]['tokens']
        assert actual_tokens == [[case['token_ids'], False] for case in reference['cases']]
        evidence = {'id': name, 'file': model['file'], 'sha256': model['sha256'],
                    'dimension': 768, 'qualified_context': 512, 'tokenizer_cases': len(inputs),
                    'individual': [], 'batches': []}
        singles = []
        for case in reference['cases']:
            events = run(args.executable, path, name+'-'+case['id'], [case['input']], reference_options, trace=True)
            output, cleanup = result(events)
            assert output['dimension'] == 768 and len(output['data']) == 1
            item = output['data'][0]
            assert item['prompt_tokens'] == len(case['token_ids']) and not item['truncated']
            singles.append(item['embedding'])
            comparison = compare(item['embedding'], case['normalized'], tolerance['absolute'],
                                 tolerance['relative'], tolerance['normalized_l2'])
            assert abs(math.sqrt(sum(v*v for v in item['embedding']))-1) <= tolerance['unit_norm_absolute']
            checkpoints = []
            for expected in case['checkpoints']:
                frame = next(e for e in events if e.get('name') == expected['name'])
                assert frame['rows'] == expected['rows'] and frame['width'] == expected['width']
                checkpoints.append({'name': frame['name'],
                                    'first': compare(frame['values'][:768], expected['first_row'],
                                                     tolerance['absolute'], tolerance['relative'], tolerance['normalized_l2']),
                                    'last': compare(frame['values'][-768:], expected['last_row'],
                                                    tolerance['absolute'], tolerance['relative'], tolerance['normalized_l2'])})
            pooled = next(e['values'] for e in events if e.get('name') == 'pooled')
            pooled_comparison = compare(pooled, case['raw_pooled'], tolerance['absolute'], tolerance['relative'], tolerance['normalized_l2'])
            evidence['individual'].append({'case': case['id'], 'embedding': comparison,
                                           'pooled': pooled_comparison, 'checkpoints': checkpoints,
                                           'encoding_ms': output['encoding_ms'], 'cleanup': cleanup})
        batch_inputs = [inputs[i] for i in reference['batch_indices']]
        for profile, threads, size in [('reference', 1, 4), ('balanced', 2, 4),
                                       ('performance', 4, 4), ('low-memory', 1, 4),
                                       ('balanced', 2, 1)]:
            options = {'profile': profile, 'threads': threads, 'batch_size': size}
            events = run(args.executable, path, f'{name}-{profile}-{threads}-{size}', batch_inputs, options)
            output, cleanup = result(events)
            comparisons = []
            for index, (item, expected) in enumerate(zip(output['data'], reference['batch_items'], strict=True)):
                assert item['index'] == index and item['prompt_tokens'] == len(expected['token_ids'])
                comparisons.append({'reference': compare(item['embedding'], expected['normalized'],
                                                        tolerance['absolute'], tolerance['relative'], tolerance['normalized_l2']),
                                    'individual': compare(item['embedding'], singles[reference['batch_indices'][index]],
                                                          tolerance['batch_absolute'], tolerance['batch_relative'])})
            assert output['usage']['prompt_tokens'] == sum(len(i['token_ids']) for i in reference['batch_items'])
            evidence['batches'].append({'profile': profile, 'threads': threads, 'batch_size': size,
                                       'comparisons': comparisons, 'plan': output['plan'],
                                       'encoding_ms': output['encoding_ms'], 'cleanup': cleanup})
        report['models'].append(evidence)
        print(f'PASS {name}: six tokenizer/encoder/pooling/vector cases and five ordered batch configurations', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(f'PASS E1 native evidence: {args.output}')


if __name__ == '__main__':
    main()

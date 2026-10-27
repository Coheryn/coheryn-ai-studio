#!/usr/bin/env python3
"""Report weight-quantization differences separately from native executor acceptance.

Requires tools/check_oracle.py --trace and tools/check_quantized.py captures.
No quality or performance superiority threshold is inferred from this small corpus.
"""
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from fetch_model import digest


def first_logits(path):
    with path.open() as source:
        for line in source:
            frame = json.loads(line)
            if frame['position'] == 0 and frame['name'] == 'logits':
                return frame['values']
    raise ValueError('Missing first-position logits')


def main():
    baseline = json.loads((ROOT / 'fixtures/golden/qwen2.5-0.5b-instruct-fp16.json').read_text())
    source = ROOT / '.cache/oracle/reference-trace.jsonl'
    expected = first_logits(source)
    manifest = json.loads((ROOT / 'fixtures/quantized/models.json').read_text())
    report = {'schema_version': 1, 'baseline_model_sha256': baseline['model_sha256'],
              'reference_revision': baseline['oracle_revision'], 'prompt': 'Hello', 'prompt_token_ids': [9707],
              'baseline_raw_trace_sha256': digest(source), 'models': [],
              'note': 'Independent reference against reference: weight quantization differences, not native executor errors. Short greedy corpus and one logit position do not establish general model quality.'}
    assert next(c for c in baseline['cases'] if c['id'] == 'english')['prompt_token_ids'] == [9707]
    for model in manifest['models']:
        if not model['id'].startswith('qwen2.5-0.5b-'):
            continue
        evidence = json.loads((ROOT / model['evidence']).read_text())
        assert evidence['oracle_revision'] == baseline['oracle_revision']
        assert next(c for c in evidence['cases'] if c['id'] == 'english')['prompt_token_ids'] == [9707]
        trace = ROOT / '.cache/quantized' / model['id'] / 'reference.jsonl'
        actual = first_logits(trace)
        assert len(actual) == len(expected) and all(math.isfinite(v) for v in actual)
        error = [a-b for a, b in zip(actual, expected)]
        cases = []
        for a, b in zip(baseline['cases'], evidence['cases'], strict=True):
            assert a['id'] == b['id'] and a['prompt_token_ids'] == b['prompt_token_ids']
            first, second = a['generated_token_ids'], b['generated_token_ids']
            prefix = 0
            for x, y in zip(first, second):
                if x != y:
                    break
                prefix += 1
            cases.append({'case': a['id'], 'exact_continuation': first == second, 'common_prefix_tokens': prefix,
                          'fp16_generated_tokens': len(first), 'quantized_generated_tokens': len(second)})
        report['models'].append({'id': model['id'], 'sha256': model['sha256'], 'raw_trace_sha256': digest(trace),
                                 'logits': {'max_absolute_difference': max(abs(e) for e in error),
                                            'normalized_l2_difference': math.sqrt(sum(e*e for e in error)/sum(v*v for v in expected)),
                                            'fp16_argmax': max(range(len(expected)), key=expected.__getitem__),
                                            'quantized_argmax': max(range(len(actual)), key=actual.__getitem__)},
                                 'greedy_observations': cases})
    (ROOT / 'fixtures/qualification/g4-quantization-quality.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Recorded separate FP16/quantized reference quality observations for four files.')


if __name__ == '__main__':
    main()

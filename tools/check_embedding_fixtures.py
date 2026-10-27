#!/usr/bin/env python3
"""Check committed E0 evidence offline without enabling native embedding execution."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from fetch_model import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-observations', action='store_true')
    args = parser.parse_args()
    directory = ROOT / 'fixtures/embedding'
    manifest = json.loads((directory / 'models.json').read_text(encoding='utf-8'))
    contract = json.loads((directory / 'reference-contract.json').read_text(encoding='utf-8'))
    assert manifest['native_embedding_execution'] is False
    for asset in manifest['source_assets']:
        path = ROOT / asset['path'] if 'path' in asset else directory / 'source' / asset['file']
        assert path.stat().st_size == asset['bytes'] and digest(path) == asset['sha256']
    fixtures = []
    for model in manifest['models']:
        metadata = json.loads((ROOT / model['metadata']).read_text(encoding='utf-8'))
        assert metadata['model_sha256'] == model['sha256']
        architecture = metadata['architecture_and_tokenizer_metadata']
        assert architecture['general.architecture'] == 'nomic-bert'
        assert architecture['nomic-bert.embedding_length'] == 768
        assert architecture['nomic-bert.attention.causal'] is False
        assert architecture['tokenizer.ggml.model'] == 'bert'
        assert contract['model_policy']['initial_reference_context'] <= architecture['nomic-bert.context_length']
        reference = json.loads((ROOT / model['evidence']).read_text(encoding='utf-8'))
        assert reference['sha256'] == model['sha256']
        assert reference['reference_revision'] == manifest['oracle']['revision']
        assert reference['native_execution_qualified'] is False
        assert reference['policy'] == contract['model_policy']
        assert len(reference['cases']) == 6
        assert len(reference['batch_indices']) == len(reference['batch_items']) == 4
        for case in reference['cases']:
            assert case['token_ids'][0] == architecture['tokenizer.ggml.bos_token_id']
            assert case['token_ids'][-1] == architecture['tokenizer.ggml.eos_token_id']
            values = case['normalized']
            assert len(values) == 768 and all(math.isfinite(v) for v in values)
            assert abs(math.sqrt(sum(v*v for v in values))-1) <= contract['tolerances']['unit_norm_absolute']
            assert case['attention_mask'] == [1] * len(case['token_ids'])
            assert case['position_ids'] == list(range(len(case['token_ids'])))
            assert case['mask_pooling_reconstruction']['outside_tolerance'] == 0
            raw = case['raw_pooled']
            assert len(raw) == 768 and all(math.isfinite(v) for v in raw)
            length = math.sqrt(sum(v*v for v in raw))
            assert length > 0
            assert max(abs(v/length-out) for v, out in zip(raw, values, strict=True)) <= contract['tolerances']['unit_norm_absolute']
        for item, index in zip(reference['batch_items'], reference['batch_indices'], strict=True):
            individual = reference['cases'][index]
            assert item['token_ids'] == individual['token_ids']
            for expected, actual in zip(individual['normalized'], item['normalized'], strict=True):
                assert abs(expected-actual) <= contract['tolerances']['batch_absolute'] + contract['tolerances']['batch_relative']*abs(expected)
        fixtures.append(reference)
    first, second = fixtures
    quality = []
    for a, b in zip(first['cases'], second['cases'], strict=True):
        assert a['id'] == b['id'] and a['token_ids'] == b['token_ids']
        av, bv = a['normalized'], b['normalized']
        cosine = sum(x*y for x, y in zip(av, bv)) / math.sqrt(sum(x*x for x in av)*sum(y*y for y in bv))
        quality.append({'case': a['id'], 'cosine_similarity': cosine,
                        'l2_distance': math.sqrt(sum((x-y)**2 for x, y in zip(av, bv)))})
    report = {'schema_version': 1, 'reference_revision': manifest['oracle']['revision'],
              'model_sha256': [f['sha256'] for f in fixtures], 'cases': quality,
              'note': 'F16 versus Q8_0 independent-reference quality observations only; no native executor or retrieval-quality acceptance claim.'}
    observations = directory / 'quantization-observations.json'
    if args.capture_observations:
        observations.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    else:
        recorded = json.loads(observations.read_text(encoding='utf-8'))
        assert recorded['model_sha256'] == report['model_sha256']
        for actual, expected in zip(quality, recorded['cases'], strict=True):
            assert actual['case'] == expected['case']
            for key in ['cosine_similarity', 'l2_distance']:
                assert abs(actual[key]-expected[key]) < 1e-12
    extensions = json.loads((directory / 'native-reference-assets.json').read_text())
    for asset in extensions['assets']:
        path = ROOT / asset['path']
        assert path.stat().st_size == asset['bytes'] and digest(path) == asset['sha256']
    multilingual = []
    for model in manifest['models']:
        reference = json.loads((directory / f"{model['id']}.multilingual.json").read_text())
        assert reference['model_sha256'] == model['sha256']
        assert reference['reference_revision'] == manifest['oracle']['revision']
        assert reference['inputs_sha256'] == digest(directory / 'multilingual-inputs.json')
        inputs = json.loads((directory / 'multilingual-inputs.json').read_text())['cases']
        assert len(reference['cases']) == len(inputs) == 50
        for case, source in zip(reference['cases'], inputs, strict=True):
            assert case['id'] == source['id'] and case['text'] == source['text']
            assert 2 <= len(case['token_ids']) <= 512
            values = case['normalized']
            assert len(values) == 768 and all(math.isfinite(v) for v in values)
            assert abs(math.sqrt(sum(v*v for v in values))-1) <= contract['tolerances']['unit_norm_absolute']
        multilingual.append(reference)
        boundary = json.loads((directory / f"{model['id']}.boundary.json").read_text())
        assert boundary['sha256'] == model['sha256']
        assert boundary['reference_revision'] == manifest['oracle']['revision']
        assert len(boundary['case']['token_ids']) == 512
        assert len(boundary['case']['normalized']) == 768
    assert [c['token_ids'] for c in multilingual[0]['cases']] == [c['token_ids'] for c in multilingual[1]['cases']]
    print('PASS frozen E0 sources/vectors/batches and independent E1 references: 100 multilingual vectors and two 512-token boundaries. Runtime status is recorded separately in fixtures/compatibility.json.')


if __name__ == '__main__':
    main()

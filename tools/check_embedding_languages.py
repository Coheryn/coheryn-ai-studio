"""Check native multilingual inference against frozen independent vectors only."""
import argparse
import json
import subprocess
from pathlib import Path
from check_native_embeddings import ROOT, compare, result, run
from fetch_model import digest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', action='store_true')
    args = parser.parse_args()
    if args.capture:
        parser.error('reference capture runs only in separately managed assessments; this command checks frozen data')
    directory = ROOT / 'fixtures/embedding'
    source = directory / 'multilingual-inputs.json'
    cases = json.loads(source.read_text())['cases']
    manifest = json.loads((directory / 'models.json').read_text())
    executable = ROOT / 'target/release/examples/embedding_probe'
    cache = ROOT / '.cache/embedding/native'
    cache.mkdir(parents=True, exist_ok=True)
    contract = json.loads((directory / 'reference-contract.json').read_text())
    tolerance = contract['tolerances']
    observations = {'schema_version': 1, 'inputs_sha256': digest(source), 'executable_sha256': digest(executable), 'scope': 'Representative multilingual Unicode and encoder parity; model vocabulary/training still determine semantic coverage.', 'models': []}
    for model in manifest['models']:
        path = ROOT / '.cache/models' / model['file']
        assert digest(path) == model['sha256']
        fixture = directory / f'{model['id']}.multilingual.json'
        reference = json.loads(fixture.read_text())
        assert reference['model_sha256'] == model['sha256'] and reference['inputs_sha256'] == digest(source)
        settings = {'profile': 'balanced', 'threads': 2, 'batch_size': 4}
        inputs = [case['text'] for case in cases]
        tokens = run(executable, path, model['id'] + '-multilingual-tokens', inputs, settings, tokenize=True)[0]['tokens']
        for case, actual, expected in zip(cases, tokens, reference['cases'], strict=True):
            assert case['id'] == expected['id'] and case['text'] == expected['text']
            assert actual == [expected['token_ids'], False], f'{case['id']}: {actual[0]} != {expected['token_ids']}'
        output, cleanup = result(run(executable, path, model['id'] + '-multilingual', inputs, settings))
        comparisons = []
        for case, actual, expected in zip(cases, output['data'], reference['cases'], strict=True):
            comparisons.append({'case': case['id'], **compare(actual['embedding'], expected['normalized'], tolerance['absolute'], tolerance['relative'], tolerance['normalized_l2'])})
        observations['models'].append({'id': model['id'], 'sha256': model['sha256'], 'reference': str(fixture.relative_to(ROOT)), 'cases': comparisons, 'cleanup': cleanup, 'plan': output['plan'], 'encoding_ms': output['encoding_ms']})
        print(f'PASS {model['id']}: {len(cases)} multilingual tokenizer and embedding cases', flush=True)
    (ROOT / 'fixtures/qualification/e1-multilingual.json').write_text(json.dumps(observations, indent=2) + '\n')
if __name__ == '__main__':
    main()

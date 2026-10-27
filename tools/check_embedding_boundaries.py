"""Capture/check independent 512-token encoder boundaries, explicit truncation and memory limits."""
import argparse
import json
import subprocess
from check_native_embeddings import ROOT, CACHE, compare, run, result
from fetch_model import digest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', action='store_true')
    args = parser.parse_args()
    if args.capture:
        parser.error('reference capture runs only in separately managed assessments; this command checks frozen data')
    CACHE.mkdir(parents=True, exist_ok=True)
    directory = ROOT / 'fixtures/embedding'
    manifest = json.loads((directory / 'models.json').read_text())
    tolerances = json.loads((directory / 'reference-contract.json').read_text())['tolerances']
    text = 'hello ' * 510
    input_file = CACHE / 'boundary.txt'
    input_file.write_text(text)
    executable = ROOT / 'target/release/examples/embedding_probe'
    observations = {'schema_version': 1, 'executable_sha256': digest(executable), 'context': 512, 'cache_conditions': 'uncontrolled', 'models': []}
    for model in manifest['models']:
        path = ROOT / '.cache/models' / model['file']
        assert digest(path) == model['sha256']
        fixture = directory / f'{model['id']}.boundary.json'
        expected = json.loads(fixture.read_text())
        assert expected['sha256'] == model['sha256'] and expected['case']['text'] == text
        settings = {'profile': 'balanced', 'threads': 2, 'batch_size': 2}
        output, cleanup = result(run(executable, path, model['id'] + '-boundary', [text, text], settings))
        comparisons = [compare(item['embedding'], expected['case']['normalized'], tolerances['absolute'], tolerances['relative'], tolerances['normalized_l2']) for item in output['data']]
        assert [item['prompt_tokens'] for item in output['data']] == [512, 512]
        assert output['data'][0]['embedding'] == output['data'][1]['embedding']
        truncated, _ = result(run(executable, path, model['id'] + '-truncate', [text + 'hello'], {**settings, 'truncate': True}))
        assert truncated['data'][0]['truncated'] and truncated['data'][0]['prompt_tokens'] == 512
        compare(truncated['data'][0]['embedding'], expected['case']['normalized'], tolerances['absolute'], tolerances['relative'], tolerances['normalized_l2'])
        rejected = []
        for name, inputs, limit in [('tokens', [text + 'hello'], 256 << 20), ('bytes', ['a' * 65537], 256 << 20), ('batch', ['hello'] * 257, 256 << 20), ('memory', [text], 1 << 20)]:
            request = CACHE / f'reject-{name}.json'
            request.write_text(json.dumps({'inputs': inputs, 'max_memory': limit}))
            response = subprocess.run([str(executable), str(path), str(request)], capture_output=True, text=True, timeout=30)
            assert response.returncode == 1 and response.stdout == ''
            failure = json.loads(response.stderr)
            rejected.append({'case': name, 'error': failure})
        observations['models'].append({'id': model['id'], 'sha256': model['sha256'], 'comparisons': comparisons, 'cleanup': cleanup, 'plan': output['plan'], 'encoding_ms': output['encoding_ms'], 'rejected': rejected})
        print(f'PASS {model['id']}: two 512-token sequences, explicit truncation, limits and cleanup', flush=True)
    (ROOT / 'fixtures/qualification/e1-boundaries.json').write_text(json.dumps(observations, indent=2) + '\n')
if __name__ == '__main__':
    main()

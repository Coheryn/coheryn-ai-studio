#!/usr/bin/env python3
"""Qualify E2 standalone/client/API, SDK and external RAG/vector-store interfaces.

Uses cached, hash-pinned Nomic models and independent E0/E1 expected vectors.
No hosted inference, model conversion, or reference runtime is used by this check.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.cache/rag-tools'), str(ROOT / 'examples'), str(ROOT / 'tools')]
from check_native_embeddings import compare
from fetch_model import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, default=ROOT / 'target/release/coheryn')
    args = parser.parse_args()
    import httpx
    from openai import OpenAI, BadRequestError
    from rag_client import demo

    models = json.loads((ROOT / 'fixtures/embedding/models.json').read_text())['models']
    tolerance = json.loads((ROOT / 'fixtures/embedding/reference-contract.json').read_text())['tolerances']
    report = {'schema_version': 1, 'gate': 'E2-CLI-API-RAG',
              'executable_sha256': digest(args.executable),
              'rust_toolchain_pin': (ROOT / 'rust-toolchain.toml').read_text().strip(),
              'cache_conditions': 'uncontrolled; no cold-cache or performance advantage claim',
              'packages': {name: importlib.metadata.version(name) for name in ['openai', 'langchain-openai', 'qdrant-client']},
              'scope': 'Actual macOS ARM64 CPU execution; API/CLI correctness and interoperability, not comparative performance.',
              'models': []}
    log_path = ROOT / '.cache/e2-service.log'
    with tempfile.TemporaryDirectory(prefix='e2-service-', dir=ROOT / '.cache') as temporary, log_path.open('w') as log:
        directory = Path(temporary)
        server = subprocess.Popen([str(args.executable), '--data-dir', str(directory), 'serve', '--port', '0',
                                   '--max-memory', str(512 << 20), '--max-concurrency', '2', '--embedding-concurrency', '2'],
                                  stdout=subprocess.DEVNULL, stderr=log, cwd=ROOT)
        try:
            endpoint = None
            deadline = time.monotonic()+30
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError('daemon exited; inspect .cache/e2-service.log')
                for line in log_path.read_text().splitlines():
                    value = json.loads(line)
                    if value.get('event') == 'listening':
                        endpoint = 'http://'+value['address']
                if endpoint:
                    break
                time.sleep(.05)
            assert endpoint, 'daemon did not become ready'
            credentials = json.loads((directory / 'credentials.json').read_text())
            with httpx.Client(base_url=endpoint, headers={'Authorization': 'Bearer '+credentials['admin_token']}, trust_env=False, timeout=300) as http:
                def api(path, body):
                    response = http.post(path, json=body)
                    assert response.is_success, (path, response.status_code, response.text[:1024])
                    return response

                report['hardware'] = http.get('/api/v1/system/resources').json()['hardware']
                report['operator_limits'] = http.get('/api/v1/server/status').json()['embedding_limits']
                for model in models:
                    path = ROOT / '.cache/models' / model['file']
                    assert path.stat().st_size == model['bytes'] and digest(path) == model['sha256']
                    with path.open('rb') as source:
                        r = http.post('/api/v1/models/import', params={'name': 'encoder', 'sha256': model['sha256']},
                                      content=source, headers={'Content-Type': 'application/octet-stream'})
                        assert r.is_success, r.text[:1024]
                    # Metadata-only planning must work before model residency.
                    plan = api('/api/v1/plans/estimate', {'model': 'encoder', 'task': 'embedding', 'context': 512, 'input_bytes': [64, 128]}).json()
                    assert plan['admissible'] and plan['model_metadata_additional'] > 0
                    api('/api/v1/models/load', {'model': 'encoder', 'context': 512})
                    reference = json.loads((ROOT / model['evidence']).read_text())
                    inputs = [case['input'] for case in reference['cases']]
                    response = api('/v1/embeddings', {'model': 'encoder', 'input': inputs})
                    space = response.headers['x-coheryn-embedding-space']
                    assert space == plan['embedding_space']['id']
                    expected = [case['normalized'] for case in reference['cases']]
                    token_counts = [len(case['token_ids']) for case in reference['cases']]
                    assert response.json()['usage'] == {'prompt_tokens': sum(token_counts), 'total_tokens': sum(token_counts)}
                    assert list(map(int, response.headers['x-coheryn-token-counts'].split(','))) == token_counts
                    comparisons = [compare(item['embedding'], vector, tolerance['absolute'], tolerance['relative'], tolerance['normalized_l2'])
                                   for item, vector in zip(response.json()['data'], expected, strict=True)]
                    multilingual = json.loads((ROOT / f"fixtures/embedding/{model['id']}.multilingual.json").read_text())
                    language_response = api('/v1/embeddings', {'model': 'encoder', 'input': [case['text'] for case in multilingual['cases']]})
                    language_checks = [compare(item['embedding'], case['normalized'], tolerance['absolute'], tolerance['relative'], tolerance['normalized_l2'])
                                       for item, case in zip(language_response.json()['data'], multilingual['cases'], strict=True)]
                    # SDK default Base64 decoding, explicit float and explicit Base64 wire forms.
                    with OpenAI(base_url=endpoint+'/v1', api_key=credentials['inference_token'], max_retries=0,
                                http_client=httpx.Client(trust_env=False, timeout=300)) as sdk:
                        ordinary = sdk.embeddings.create(model='encoder', input=inputs[0])
                        sdk_check = compare(ordinary.data[0].embedding, expected[0], tolerance['absolute'], tolerance['relative'], tolerance['normalized_l2'])
                        floats = sdk.embeddings.create(model='encoder', input=inputs[0], encoding_format='float', dimensions=768)
                        import struct
                        # JSON's shortest decimals and Base64's binary32-to-Python-float
                        # widening differ as binary64 values; compare the float32 contract.
                        assert struct.pack('<768f', *ordinary.data[0].embedding) == struct.pack('<768f', *floats.data[0].embedding)
                        encoded = sdk.embeddings.create(model='encoder', input=inputs[0], encoding_format='base64')
                        import base64, struct
                        decoded = list(struct.unpack('<768f', base64.b64decode(encoded.data[0].embedding)))
                        assert struct.pack('<768f', *decoded) == struct.pack('<768f', *floats.data[0].embedding)
                        try:
                            sdk.embeddings.create(model='encoder', input=[101, 102])
                            raise AssertionError('unsupported pretokenized input was accepted')
                        except BadRequestError:
                            pass
                    # The same corpus crosses several local/client file windows, preserving identity.
                    input_path = directory / 'documents.txt'
                    file_indices = [0, 1, 2, 3, 5, 0]
                    file_inputs = [inputs[i] for i in file_indices]
                    file_expected = [expected[i] for i in file_indices]
                    file_tokens = sum(len(reference['cases'][i]['token_ids']) for i in file_indices)
                    input_path.write_text('\n'.join(file_inputs), encoding='utf-8')
                    output_paths = [directory/f"{model['id']}-standalone.json", directory/f"{model['id']}-client.json"]
                    for mode, output_path in zip(['standalone', 'client'], output_paths):
                        command = [str(args.executable), '--json', '--data-dir', str(directory), '--endpoint', endpoint,
                                   'embed', str(path) if mode == 'standalone' else 'encoder', '--file', str(input_path),
                                   '--request-size', '2', '--output', str(output_path), '--max-memory', str(512 << 20)]
                        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=300)
                        assert completed.returncode == 0, completed.stderr[-2048:]
                        assert completed.stdout == ''
                        output = json.loads(output_path.read_text())
                        assert output['coheryn']['embedding_space'] == space and output['coheryn']['model_sha256'] == model['sha256']
                        assert output['usage'] == {'prompt_tokens': file_tokens, 'total_tokens': file_tokens}
                        for index, (item, vector) in enumerate(zip(output['data'], file_expected, strict=True)):
                            assert item['index'] == index
                            compare(item['embedding'], vector, tolerance['absolute'], tolerance['relative'], tolerance['normalized_l2'])
                    rag = demo(endpoint+'/v1', 'encoder', credentials['inference_token'])
                    assert rag['dimension'] == 768 and rag['documents'] == 2 and len(rag['matches']) == 2
                    api('/api/v1/models/unload', {'model': 'encoder'})
                    status = http.get('/api/v1/server/status').json()
                    assert status['memory']['reserved'] == 0 and status['memory']['session_buffer_capacity_bytes'] == 0
                    assert status['scheduler']['active'] == status['scheduler']['queued'] == 0
                    report['models'].append({'id': model['id'], 'sha256': model['sha256'], 'embedding_space': space,
                                             'usage': response.json()['usage'], 'selected_batch_size': int(response.headers['x-coheryn-batch-size']),
                                             'selected_batch_tokens': int(response.headers['x-coheryn-batch-tokens']), 'reference_comparisons': comparisons, 'multilingual_comparisons': language_checks,
                                             'sdk_default_comparison': sdk_check, 'float_base64_roundtrip': True,
                                             'standalone_client_file_parity': True, 'rag': rag, 'cleanup': status['memory']})
                    assert http.delete('/api/v1/models/encoder').is_success
                    print(f"PASS E2 {model['id']}: CLI/client/API, 50 Unicode cases, SDK float/Base64 and LangChain/Qdrant", flush=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    target = ROOT / 'fixtures/qualification/e2-service.json'
    target.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()

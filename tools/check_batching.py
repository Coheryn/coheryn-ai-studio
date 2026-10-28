#!/usr/bin/env python3
"""Opt-in native dynamic-batch qualification using cached, hash-pinned weights.

Expected sequences come from the original native single-sequence graph in the same
build. Frozen independent numerical/model references remain unchanged. No downloads
or external execution backend is involved. OS/file cache conditions are uncontrolled.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / 'target/release/coheryn'


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT / 'fixtures/qualification/s6-service.json')
    args = parser.parse_args()
    pin = json.loads((ROOT / 'fixtures/tools/manifest.json').read_text())['model']
    model = ROOT / '.cache/models' / pin['file']
    assert model.stat().st_size == pin['bytes'] and digest(model) == pin['sha256']
    environment = {k: v for k, v in os.environ.items() if k not in {'COHERYN_API_TOKEN', 'COHERYN_ADMIN_TOKEN'}}
    report = {'schema_version': 1, 'stage': 'S6 initial exact-file native CPU batching',
              'model_sha256': pin['sha256'], 'model_bytes': pin['bytes'], 'binary_sha256': digest(BINARY),
              'os': platform.platform(), 'architecture': platform.machine(), 'context': 128,
              'backend': 'native-cpu', 'cache_dtype': 'f32', 'threads': 1,
              'batching_profile': 'coheryn-cpu-batch-v1', 'generation_resident_limit': 2,
              'cache_conditions': 'Uncontrolled OS/file cache; request-private KV, grammar and RNG.',
              'scope': 'Native individual-versus-batch correctness, real weight-read reuse and lifecycle. Not independent model-oracle replacement or a throughput/speed advantage claim.',
              'checks': [], 'rss_bytes': None, 'process_private_bytes': None}

    def passed(name, **evidence):
        report['checks'].append({'case': name, **evidence})
        print('PASS', name, flush=True)

    with tempfile.TemporaryDirectory(prefix='coheryn-s6-') as directory:
        temporary = Path(directory)
        data = temporary / 'service'
        log_path = ROOT / '.cache/s6-server.log'
        with log_path.open('w') as log:
            server = subprocess.Popen([str(BINARY), '--data-dir', str(data), 'serve', '--port', '0',
                                       '--max-context', '512', '--max-concurrency', '2',
                                       '--generation-timeout', '300'], cwd=ROOT, env=environment, stdout=log, stderr=log)
            try:
                endpoint = None
                for _ in range(200):
                    assert server.poll() is None, 'native server exited during startup'
                    for line in log_path.read_text().splitlines():
                        try:
                            value = json.loads(line)
                        except ValueError:
                            continue
                        if value.get('event') == 'listening':
                            endpoint = 'http://' + value['address']
                    if endpoint:
                        break
                    time.sleep(.05)
                assert endpoint, 'native startup timed out'
                credentials = json.loads((data / 'credentials.json').read_text())

                def cli(*arguments):
                    result = subprocess.run([str(BINARY), '--data-dir', str(data), '--endpoint', endpoint,
                                             '--json', *arguments], cwd=ROOT, env=environment,
                                            capture_output=True, text=True, timeout=300)
                    assert result.returncode == 0, result.stderr
                    return json.loads(result.stdout)

                def open_request(path, body=None, admin=False):
                    request = urllib.request.Request(endpoint + path,
                        data=None if body is None else json.dumps(body, ensure_ascii=False).encode(),
                        headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + credentials['admin_token' if admin else 'inference_token']})
                    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=300)

                def api(body, expected=200):
                    try:
                        with open_request('/api/v1/generate', body) as response:
                            assert response.status == expected
                            return json.load(response)
                    except urllib.error.HTTPError as error:
                        assert error.code == expected, error.read().decode()
                        return json.load(error)

                def status():
                    with open_request('/api/v1/server/status', admin=True) as response:
                        return json.load(response)

                def wait(predicate):
                    until = time.monotonic() + 60
                    while time.monotonic() < until:
                        value = status()
                        if predicate(value):
                            return value
                        time.sleep(.025)
                    raise AssertionError('native scheduling condition timed out')

                def idle():
                    return wait(lambda s: s['scheduler']['active'] == s['scheduler']['queued'] == s['scheduler']['waiting_generation_units'] == 0)

                cli('models', 'import', str(model), '--name', 'batch', '--sha256', pin['sha256'])
                cli('models', 'load', 'batch', '--context', '512')
                initial = idle()
                assert initial['continuous_batching'] and initial['generation_resident_limit'] == 2
                baseline = initial['memory']['reserved']
                bodies = [
                    {'model': 'batch', 'prompt': 'Say hello.', 'max_tokens': 8, 'context': 128,
                     'profile': 'reference', 'threads': 1, 'temperature': .8, 'top_k': 32, 'top_p': .9,
                     'seed': 42, 'repetition_penalty': 1.1, 'frequency_penalty': .1, 'presence_penalty': .2},
                    {'model': 'batch', 'prompt': 'Write the integers 1 through 100, separated by commas.', 'max_tokens': 32,
                     'context': 128, 'profile': 'reference', 'threads': 1, 'temperature': .8, 'top_k': 32, 'top_p': .9,
                     'seed': 77, 'repetition_penalty': 1.1, 'frequency_penalty': .1, 'presence_penalty': .2},
                ]
                expected = []
                for body in bodies:
                    result = cli('run', str(model), '--standalone', '--prompt', body['prompt'], '--max-tokens', str(body['max_tokens']),
                                 '--context', '128', '--profile', 'reference', '--threads', '1', '--temperature', '.8',
                                 '--top-k', '32', '--top-p', '.9', '--seed', str(body['seed']),
                                 '--repetition-penalty', '1.1', '--frequency-penalty', '.1', '--presence-penalty', '.2')
                    assert result['batching']['strategy'] == 'single-sequence'
                    expected.append(result)
                passed('original-single-sequence-baselines', sequences=[{'prompt_tokens': r['prompt_tokens'], 'token_ids': r['token_ids']} for r in expected])

                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                    both = [pool.submit(api, body) for body in bodies]
                    results = [future.result(timeout=300) for future in both]
                    for actual, reference in zip(results, expected, strict=True):
                        assert actual['token_ids'] == reference['token_ids'] and actual['text'] == reference['text']
                        assert actual['batching']['strategy'] == 'continuous-weight-tiled'
                        assert actual['batching']['batched_forward_units'] > 0
                    passed('simultaneous-seeded-penalty-cli-api-parity', batches=[r['batching'] for r in results])
                    assert idle()['memory']['reserved'] == baseline

                    before = status()['scheduler']['forwarded_tokens']
                    ongoing = pool.submit(api, bodies[1])
                    wait(lambda s: s['scheduler']['forwarded_tokens'] > before)
                    short = api({**bodies[0], 'max_tokens': 2})
                    assert short['token_ids'] == expected[0]['token_ids'][:2]
                    assert not ongoing.done(), 'long sequence must remain live after first retirement'
                    third = api({**bodies[0], 'max_tokens': 3})
                    assert third['token_ids'] == expected[0]['token_ids'][:3]
                    long = ongoing.result(timeout=300)
                    assert long['token_ids'] == expected[1]['token_ids']
                    assert short['batching']['batched_forward_units'] and third['batching']['batched_forward_units']
                    passed('late-join-individual-retirement-and-slot-reuse', short=short['batching'], third=third['batching'], long=long['batching'])
                    assert idle()['memory']['reserved'] == baseline

                    schema = {'type': 'json_schema', 'json_schema': {'name': 'unicode', 'strict': True,
                              'schema': {'type': 'object', 'properties': {'বাংলা': {'type': 'string', 'const': '日本語 🌍'}},
                                         'required': ['বাংলা'], 'additionalProperties': False}}}
                    constrained = {**bodies[0], 'prompt': 'Return this JSON object: {"বাংলা":"日本語 🌍"}',
                                   'max_tokens': 64, 'temperature': 0, 'repetition_penalty': 1,
                                   'frequency_penalty': 0, 'presence_penalty': 0, 'response_format': schema}
                    peer = pool.submit(api, bodies[1])
                    value = api(constrained)
                    assert json.loads(value['text']) == {'বাংলা': '日本語 🌍'} and value['constraint_validation']['valid']
                    assert peer.result(timeout=300)['token_ids'] == expected[1]['token_ids']
                    passed('request-private-unicode-grammar-and-peer-rng', batching=value['batching'])
                    assert idle()['memory']['reserved'] == baseline

                    peer = pool.submit(api, bodies[1])
                    error = api({**constrained, 'max_tokens': 1}, expected=422)
                    assert error['error']['code'] == 'CONSTRAINT_INCOMPLETE'
                    assert peer.result(timeout=300)['token_ids'] == expected[1]['token_ids']
                    passed('incomplete-sequence-failure-isolation')
                    assert idle()['memory']['reserved'] == baseline

                    peer = pool.submit(api, bodies[1])
                    with open_request('/api/v1/generate', {**bodies[1], 'stream': True}) as response:
                        for line in response:
                            if line.startswith(b'data: '):
                                chunk = json.loads(line[6:])
                                if chunk.get('token', {}).get('text'):
                                    break
                        else:
                            raise AssertionError('no actual token before disconnect')
                    assert peer.result(timeout=300)['token_ids'] == expected[1]['token_ids']
                    assert idle()['memory']['reserved'] == baseline
                    passed('stream-disconnect-isolation-and-acknowledged-cleanup')

                    reference = {**bodies[0], 'max_tokens': 2}
                    performance = {**reference, 'profile': 'performance'}
                    mixed = [pool.submit(api, body) for body in (reference, performance)]
                    mixed = [future.result(timeout=300) for future in mixed]
                    assert all(r['batching']['maximum_batch_width'] == 1 for r in mixed)
                    assert all(r['token_ids'] == expected[0]['token_ids'][:2] for r in mixed)
                    passed('incompatible-execution-contracts-are-not-packed')
                final = idle()
                assert final['memory']['reserved'] == baseline and final['memory']['session_buffer_capacity_bytes'] == 0
                assert final['scheduler']['maximum_batch_width'] == 2 and final['scheduler']['matrix_read_bytes'] > 0
                report['final_status'] = final
                report['native_only'] = True
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
            finally:
                if server.poll() is None:
                    server.send_signal(signal.SIGINT)
                    try:
                        server.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait(timeout=5)
    print('PASS pinned native S6 qualification', flush=True)


if __name__ == '__main__':
    main()

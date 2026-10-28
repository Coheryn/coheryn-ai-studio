#!/usr/bin/env python3
"""Opt-in exact-model native S5 checks. No downloads or external model execution."""
import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
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
    parser.add_argument('--report', type=Path, default=ROOT / 'fixtures/qualification/s5-service.json')
    args = parser.parse_args()
    pin = json.loads((ROOT / 'fixtures/tools/manifest.json').read_text())['model']
    model = ROOT / '.cache/models' / pin['file']
    assert model.stat().st_size == pin['bytes'] and digest(model) == pin['sha256']
    environment = {k: v for k, v in os.environ.items() if k not in {'COHERYN_API_TOKEN', 'COHERYN_ADMIN_TOKEN'}}
    report = {'schema_version': 1, 'stage': 'S5 exact-file native CPU preview',
              'model_sha256': pin['sha256'], 'model_bytes': pin['bytes'], 'binary_sha256': digest(BINARY),
              'os': platform.platform(), 'architecture': platform.machine(), 'backend': 'native-cpu',
              'context': 512, 'cache_dtype': 'f32', 'profile': 'reference', 'threads': 1,
              'seed': 0, 'temperature': 0, 'schema_profile': 'coheryn-json-v1',
              'sampling_profile': 'coheryn-sampling-v2', 'checks': [],
              'cache_conditions': 'Uncontrolled OS/file cache; stateless KV and private grammar state rebuilt per request.',
              'rss_bytes': None, 'process_private_bytes': None,
              'scope': 'Correctness, interface parity and lifecycle; no cross-platform/GPU, retrieval-quality or controlled inference throughput claim.'}
    schema = {'type': 'object', 'properties': {'answer': {'type': 'string', 'const': 'ok'}},
              'required': ['answer'], 'additionalProperties': False}

    def formatted(value):
        return {'type': 'json_schema', 'json_schema': {'name': 'result', 'strict': True, 'schema': value}}

    prompt = 'Return exactly the JSON object {"answer":"ok"}.'
    base = {'model': 's5', 'prompt': prompt, 'max_tokens': 64, 'context': 512,
            'profile': 'reference', 'threads': 1, 'temperature': 0, 'seed': 0,
            'response_format': formatted(schema)}

    def passed(name, **evidence):
        report['checks'].append({'case': name, **evidence})
        print('PASS', name, flush=True)

    with tempfile.TemporaryDirectory(prefix='coheryn-s5-') as directory:
        temporary = Path(directory)
        data = temporary / 'service'
        log_path = ROOT / '.cache/s5-server.log'
        log = log_path.open('w')
        server = subprocess.Popen([str(BINARY), '--data-dir', str(data), 'serve', '--port', '0',
                                   '--max-context', '512', '--generation-timeout', '300'],
                                  cwd=ROOT, env=environment, stdout=log, stderr=log)
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

            def cli(*arguments, expected=0):
                result = subprocess.run([str(BINARY), '--data-dir', str(data), '--endpoint', endpoint,
                                         '--json', *arguments], cwd=ROOT, env=environment,
                                        capture_output=True, text=True, timeout=300)
                assert result.returncode == expected, result.stderr
                return json.loads(result.stdout) if expected == 0 else json.loads(result.stderr)

            cli('models', 'import', str(model), '--name', 's5', '--sha256', pin['sha256'])
            cli('models', 'load', 's5', '--context', '512')
            plan = cli('plan', 's5', '--context', '512', '--constraints', '--profile', 'reference', '--threads', '1')
            assert plan['constraint_reserved_bytes'] == 32 << 20
            report['hardware'] = plan['hardware']
            report['constraint_plan'] = plan
            credentials = json.loads((data / 'credentials.json').read_text())
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

            def open_request(path, body):
                return opener.open(urllib.request.Request(endpoint + path,
                    data=json.dumps(body, ensure_ascii=False).encode(),
                    headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + credentials['inference_token']}), timeout=300)

            def api(path, body, expected=200):
                try:
                    with open_request(path, body) as response:
                        assert response.status == expected
                        return json.load(response)
                except urllib.error.HTTPError as error:
                    assert error.code == expected, error.read().decode()
                    return json.load(error)

            def sse(body):
                with open_request('/v1/chat/completions', body) as response:
                    chunks = []
                    done = False
                    for line in response:
                        if not line.startswith(b'data: '):
                            continue
                        payload = line[6:].strip()
                        if payload == b'[DONE]':
                            done = True
                            break
                        chunks.append(json.loads(payload))
                    assert done
                    return chunks

            schema_path = temporary / 'schema.json'
            schema_path.write_text(json.dumps(schema))
            native = cli('run', str(model), '--standalone', '--prompt', prompt, '--max-tokens', '64',
                         '--context', '512', '--profile', 'reference', '--threads', '1',
                         '--response-format', 'json_schema', '--schema', str(schema_path))
            http = api('/api/v1/generate', base)
            assert native['text'] == http['text'] and native['token_ids'] == http['token_ids']
            assert json.loads(http['text']) == {'answer': 'ok'}
            assert http['constraint_validation']['valid'] and http['sampling_metadata']['profile'] == 'coheryn-sampling-v2'
            passed('native-cli-api-exact-schema-parity', token_ids=http['token_ids'], usage=http['completion_tokens'],
                   validation=http['constraint_validation'], memory=http['memory'], sampling=http['sampling_metadata'])
            completion = api('/v1/completions', base)
            assert completion['choices'][0]['text'] == http['text']

            chat = {k: v for k, v in base.items() if k != 'prompt'}
            chat['messages'] = [{'role': 'user', 'content': prompt}]
            chat['stream'] = True
            chat['stream_options'] = {'include_usage': True}
            chunks = sse(chat)
            text = ''.join(choice.get('delta', {}).get('content', '') or '' for chunk in chunks for choice in chunk.get('choices', []))
            assert json.loads(text) == {'answer': 'ok'}
            assert any(chunk.get('coheryn', {}).get('constraint_validation', {}).get('valid') for chunk in chunks)
            assert any(chunk.get('usage') for chunk in chunks)
            passed('openai-completion-chat-sse-and-usage')

            for name, value in [('json-object', {'type': 'json_object'}),
                                ('unicode-schema', formatted({'type': 'object', 'properties': {'বাংলা': {'type': 'string', 'const': '日本語 🌍'}}, 'required': ['বাংলা'], 'additionalProperties': False}))]:
                body = copy.deepcopy(base)
                body['response_format'] = value
                if name == 'unicode-schema':
                    body['prompt'] = 'Return this JSON object: {"বাংলা":"日本語 🌍"}'
                result = api('/api/v1/generate', body)
                parsed = json.loads(result['text'])
                assert isinstance(parsed, dict) and result['constraint_validation']['valid']
                if name == 'unicode-schema':
                    assert parsed == {'বাংলা': '日本語 🌍'}
                passed(name, text=result['text'], validation=result['constraint_validation'])

            strict = {k: v for k, v in chat.items() if k not in {'stream', 'stream_options', 'response_format'}}
            strict.update(max_tokens=128, tools=[{'type': 'function', 'function': {'name': 'answer', 'strict': True,
                'parameters': {'type': 'object', 'properties': {'value': {'type': 'integer', 'const': 4}}, 'required': ['value'], 'additionalProperties': False}}}],
                tool_choice='required', parallel_tool_calls=False)
            strict['messages'] = [{'role': 'user', 'content': 'Call answer with value 4.'}]
            result = api('/v1/chat/completions', strict)
            call = result['choices'][0]['message']['tool_calls'][0]
            assert result['choices'][0]['finish_reason'] == 'tool_calls'
            assert call['function']['name'] == 'answer' and json.loads(call['function']['arguments']) == {'value': 4}
            assert result['coheryn']['constraint_validation']['target'] == 'tool_arguments'
            named = copy.deepcopy(strict)
            named['tool_choice'] = {'type': 'function', 'function': {'name': 'answer'}}
            named_result = api('/v1/chat/completions', named)
            assert json.loads(named_result['choices'][0]['message']['tool_calls'][0]['function']['arguments']) == {'value': 4}
            strict.update(stream=True, stream_options={'include_usage': True})
            chunks = sse(strict)
            arguments = ''.join(delta.get('function', {}).get('arguments', '') for chunk in chunks for choice in chunk.get('choices', []) for delta in choice.get('delta', {}).get('tool_calls', []))
            assert json.loads(arguments) == {'value': 4}
            passed('strict-single-required-named-and-streamed-tool', validation=result['coheryn']['constraint_validation'])

            seeded = {k: v for k, v in base.items() if k != 'response_format'}
            seeded.update(max_tokens=12, temperature=.8, top_k=32, top_p=.9, seed=42,
                          repetition_penalty=1.1, frequency_penalty=.1, presence_penalty=.2)
            result = api('/api/v1/generate', seeded)
            standalone = cli('run', str(model), '--standalone', '--prompt', prompt, '--max-tokens', '12', '--context', '512',
                             '--profile', 'reference', '--threads', '1', '--temperature', '.8', '--top-k', '32', '--top-p', '.9',
                             '--seed', '42', '--repetition-penalty', '1.1', '--frequency-penalty', '.1', '--presence-penalty', '.2')
            assert result['token_ids'] == standalone['token_ids']
            seeded_ids = result['token_ids']
            stopped = api('/api/v1/generate', {**seeded, 'stop_tokens': [result['token_ids'][0]]})
            assert stopped['token_ids'] == result['token_ids'][:1] and stopped['text'] == '' and stopped['finish_reason'] == 'stop'
            passed('seeded-penalty-cli-api-parity-and-token-stop', token_ids=result['token_ids'])

            short = {**base, 'max_tokens': 1}
            error = api('/v1/completions', short, expected=422)
            assert error['error']['code'] == 'CONSTRAINT_INCOMPLETE' and not error['error']['details']['constraint_validation']['valid']
            bad = copy.deepcopy(base)
            bad['response_format']['json_schema']['schema']['pattern'] = '.*'
            assert api('/v1/completions', bad, expected=400)['error']['code'] == 'UNSUPPORTED_FEATURE'
            passed('incomplete-and-unsupported-schema-errors')

            def idle():
                for _ in range(200):
                    status = cli('status')
                    if status['scheduler']['active'] == 0 and status['scheduler']['queued'] == 0:
                        return status
                    time.sleep(.025)
                raise AssertionError('native worker did not acknowledge cleanup')

            baseline = idle()['memory']['reserved']
            long_schema = {'type': 'object', 'properties': {'value': {'type': 'string', 'maxLength': 4096,
                           'const': 'abcdefghijklmnopqrstuvwxyz0123456789' * 80}}, 'required': ['value'], 'additionalProperties': False}
            long = {**base, 'max_tokens': 400, 'response_format': formatted(long_schema), 'timeout_seconds': 1}
            error = api('/v1/completions', long, expected=408)
            assert error['error']['code'] == 'DEADLINE_EXCEEDED'
            assert error['error']['details']['constraint_validation']['status'] == 'deadline'
            assert idle()['memory']['reserved'] == baseline
            disconnect = {k: v for k, v in chat.items() if k != 'stream_options'}
            disconnect.update(max_tokens=400, response_format=formatted(long_schema))
            with open_request('/v1/chat/completions', disconnect) as response:
                for line in response:
                    if line.startswith(b'data: ') and line[6:].strip() != b'[DONE]':
                        chunk = json.loads(line[6:])
                        if any(c.get('delta', {}).get('content') for c in chunk.get('choices', [])):
                            break
            assert idle()['memory']['reserved'] == baseline
            passed('deadline-disconnect-and-reservation-cleanup', baseline_reserved_bytes=baseline)

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                outputs = list(pool.map(lambda _: api('/api/v1/generate', base), range(2)))
            assert all(o['token_ids'] == http['token_ids'] for o in outputs)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                sampled = list(pool.map(lambda _: api('/api/v1/generate', seeded), range(2)))
            assert all(o['token_ids'] == seeded_ids for o in sampled)
            assert idle()['memory']['reserved'] == baseline
            passed('queued-request-grammar-and-rng-isolation')

            session = cli('sessions', 'create', 's5', '--name', 'schema-session', '--context', '512')
            turn_path = temporary / 'turn.json'
            turn_path.write_text(json.dumps({'expected_revision': 0, 'idempotency_key': 'schema-turn', 'content': prompt,
                'settings': {'max_tokens': 64, 'profile': 'reference', 'threads': 1, 'response_format': formatted(schema)}}))
            turn = cli('sessions', 'submit', 'schema-session', '--request', str(turn_path))
            assert json.loads(turn['result']['text']) == {'answer': 'ok'}
            assert turn['result']['constraint_validation']['valid']
            replay = cli('sessions', 'submit', 'schema-session', '--request', str(turn_path))
            assert replay['result']['token_ids'] == turn['result']['token_ids']
            passed('managed-schema-turn-and-idempotent-replay', revision=turn['revision'])

            sys.path.insert(0, str(ROOT / '.cache/rag-tools'))
            try:
                import httpx
                import openai
            except ImportError:
                report['optional_sdk'] = 'unavailable; raw HTTP interoperability checked'
            else:
                with openai.OpenAI(base_url=endpoint + '/v1', api_key=credentials['inference_token'], max_retries=0,
                                   http_client=httpx.Client(trust_env=False, timeout=300)) as client:
                    response = client.chat.completions.create(model='s5', messages=[{'role': 'user', 'content': prompt}],
                        response_format=formatted(schema), max_tokens=64, temperature=0,
                        extra_body={'context': 512, 'profile': 'reference', 'threads': 1})
                    assert json.loads(response.choices[0].message.content) == {'answer': 'ok'}
                report['optional_sdk'] = {'name': 'openai', 'version': openai.__version__}
                passed('standard-sdk-schema-response')
            report['final_status'] = idle()
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
            log.close()
    print('PASS pinned native S5 qualification', flush=True)


if __name__ == '__main__':
    main()

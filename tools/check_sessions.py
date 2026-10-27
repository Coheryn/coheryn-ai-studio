#!/usr/bin/env python3
"""Pinned native CPU session/CLI/API parity; uses weights and inert expected data only."""
import argparse
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
from fetch_model import digest

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / 'target/release/coheryn'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', type=int, default=8, choices=range(1, 9), help='greedy prefix length; eight checks the full frozen continuations')
    args = parser.parse_args()
    tokens = str(args.tokens)
    manifest = json.loads((ROOT / 'fixtures/models.json').read_text())
    pin = manifest['models'][0]
    model = ROOT / '.cache/models' / pin['file']
    assert model.stat().st_size == pin['bytes'] and digest(model) == pin['sha256']
    golden = json.loads((ROOT / 'fixtures/golden/qwen2.5-0.5b-instruct-fp16.json').read_text())
    cases = {case['id']: case for case in golden['cases']}
    environment = {key: value for key, value in os.environ.items() if key not in {'COHERYN_API_TOKEN', 'COHERYN_ADMIN_TOKEN'}}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    report = {'schema_version': 1, 'stage': 'S1-native-CPU-managed-session-preview',
              'model_sha256': pin['sha256'], 'binary_sha256': digest(BINARY),
              'hardware': platform.machine(), 'os': platform.platform(), 'backend': 'native-cpu',
              'context': 512, 'cache_dtype': 'f32', 'profile': 'balanced', 'cache_conditions': 'uncontrolled',
              'max_tokens_per_comparison': args.tokens, 'seed': 0,
              'persistence': False, 'cross_turn_kv_reuse': False, 'checks': []}
    with tempfile.TemporaryDirectory(prefix='coheryn-session-check-') as temporary:
        data = Path(temporary) / 'service'
        log_path = ROOT / '.cache/s1-service.log'
        with log_path.open('w') as log:
            server = subprocess.Popen([str(BINARY), '--data-dir', str(data), 'serve', '--port', '0', '--max-context', '512', '--idle-cache-bytes', '0'],
                                      cwd=ROOT, env=environment, stdout=log, stderr=log)
            try:
                endpoint = None
                for _ in range(200):
                    if server.poll() is not None:
                        raise RuntimeError('native daemon exited before listening')
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
                assert endpoint, 'native daemon startup timed out'
                credentials = json.loads((data / 'credentials.json').read_text())

                def cli(*args, input_text=None, as_json=True):
                    result = subprocess.run([str(BINARY), '--data-dir', str(data), '--endpoint', endpoint, *(['--json'] if as_json else []), *args],
                                            input=input_text, capture_output=True, text=True, env=environment, cwd=ROOT, timeout=300)
                    assert result.returncode == 0, result.stderr
                    return json.loads(result.stdout) if as_json else result.stdout

                def http(method, path, body=None, expected=200):
                    headers = {'Authorization': 'Bearer ' + credentials['inference_token'], 'Content-Type': 'application/json'}
                    request = urllib.request.Request(endpoint + path, method=method, headers=headers,
                                                     data=None if body is None else json.dumps(body).encode())
                    try:
                        response = opener.open(request, timeout=300)
                    except urllib.error.HTTPError as error:
                        response = error
                    with response:
                        value = json.load(response)
                        assert response.status == expected, value
                        return value

                cli('models', 'import', str(model), '--name', 'native-chat', '--sha256', pin['sha256'])
                cli('models', 'load', 'native-chat', '--context', '512')
                plan = cli('plan', 'native-chat', '--context', '512')
                report['threads_per_session'] = plan['threads_per_session']
                report['backend'] = plan['backend']
                native = cli('run', 'native-chat', '--prompt', 'Hello', '--context', '512', '--max-tokens', tokens)
                assert native['token_ids'] == cases['english']['generated_token_ids'][:args.tokens]
                if args.tokens == 8:
                    assert native['text'] == cases['english']['text']
                report['checks'].append({'case': 'retained-independent-english-golden', 'token_ids': native['token_ids']})
                print('PASS unchanged native greedy golden', flush=True)
                created = cli('sessions', 'create', 'native-chat', '--name', 'qualified', '--context', '512', '--system', 'Be brief.')
                first = cli('sessions', 'send', 'qualified', '--input', 'Hi', '--expected-revision', '0', '--idempotency-key', 'first', '--max-tokens', tokens)
                assert first['revision'] == 2 and first['result']['token_ids'] == cases['explicit-system']['generated_token_ids'][:args.tokens]
                if args.tokens == 8:
                    assert first['result']['text'] == cases['explicit-system']['text']
                second_body = {'expected_revision': 2, 'idempotency_key': 'second', 'content': 'What is 2+2?', 'settings': {'max_tokens': args.tokens}}
                second = http('POST', '/coheryn/v1/sessions/qualified/messages', second_body)
                view = cli('sessions', 'show', 'qualified')
                messages = [{'role': message['role'], 'content': message['content']} for message in view['messages'][:-1]]
                stateless = http('POST', '/api/v1/generate', {'model': 'native-chat', 'messages': messages, 'context': 512, 'max_tokens': args.tokens})
                assert second['result']['token_ids'] == stateless['token_ids']
                assert second['context']['prompt_tokens'] == stateless['prompt_tokens']
                assert second['result']['text'] == stateless['text']
                replay = cli('sessions', 'send', 'qualified', '--input', 'Hi', '--expected-revision', '0', '--idempotency-key', 'first', '--max-tokens', tokens)
                assert replay == first
                assert cli('sessions', 'show', 'qualified')['revision'] == 4
                http('POST', '/coheryn/v1/sessions/qualified/messages', {**second_body, 'content': 'changed'}, expected=409)
                report['checks'].append({'case': 'exact-history-native-stateless-token-parity', 'prompt_tokens': stateless['prompt_tokens'],
                                         'token_ids': stateless['token_ids'], 'text': stateless['text'], 'idempotent_replay': True})
                print('PASS managed history, exact token accounting, stateless parity and retry isolation', flush=True)
                output = cli('chat', 'native-chat', '--session', 'qualified', '--resume', '--max-tokens', str(min(4, args.tokens)),
                             input_text='Say goodbye.\n/exit\n', as_json=False)
                assert output.startswith('You: ') and output.endswith('\nYou: ')
                view = cli('sessions', 'show', 'qualified')
                assert view['revision'] == 6 and len(view['messages']) == 7
                report['checks'].append({'case': 'separate-cli-process-resume-and-native-sse', 'revision': 6, 'message_count': 7})
                cli('sessions', 'reset', 'qualified', '--expected-revision', '6')
                cli('sessions', 'delete', 'qualified', '--expected-revision', '7')
                assert cli('sessions', 'list')['sessions'] == []
                cli('models', 'unload', 'native-chat')
                status = cli('status')
                assert status['memory']['reserved'] == 0 and status['scheduler']['active'] == 0 and status['scheduler']['queued'] == 0
                report['checks'].append({'case': 'delete-unload-reservation-cleanup', 'reserved_bytes': 0})
                server.send_signal(signal.SIGINT)
                assert server.wait(timeout=30) == 0
                report['graceful_shutdown'] = True
                print('PASS CLI resume/reset/delete, model unload, zero reservations and graceful shutdown', flush=True)
            finally:
                if server.poll() is None:
                    server.kill()
                    server.wait(timeout=10)
    (ROOT / 'fixtures/qualification/s1-service.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

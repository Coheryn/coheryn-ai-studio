#!/usr/bin/env python3
"""Qualify native prefix reuse and durable restart against pinned weights/data."""
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import tempfile
import time
import urllib.request

from fetch_model import digest

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / 'target/release/coheryn'


def main():
    pin = json.loads((ROOT / 'fixtures/models.json').read_text())['models'][0]
    model = ROOT / '.cache/models' / pin['file']
    assert model.stat().st_size == pin['bytes'] and digest(model) == pin['sha256']
    golden = json.loads((ROOT / 'fixtures/golden/qwen2.5-0.5b-instruct-fp16.json').read_text())
    case = next(c for c in golden['cases'] if c['id'] == 'explicit-system')
    env = {k: v for k, v in os.environ.items() if k not in {'COHERYN_API_TOKEN', 'COHERYN_ADMIN_TOKEN'}}
    compiler = ROOT / '.cache/cargo/bin/rustc'
    compiler_env = env.copy()
    if compiler.is_file():
        compiler_env.update(CARGO_HOME=str(ROOT / '.cache/cargo'),
                            RUSTUP_HOME=str(ROOT / '.cache/rustup'), RUSTUP_AUTO_INSTALL='0')
    else:
        compiler = 'rustc'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    report = {
        'schema_version': 1, 'stage': 'S2-S3-native-CPU-preview',
        'model_sha256': pin['sha256'], 'model_bytes': pin['bytes'],
        'binary_sha256': digest(BINARY), 'os': platform.platform(), 'hardware': platform.machine(),
        'compiler': subprocess.check_output([str(compiler), '--version'], cwd=ROOT, env=compiler_env, text=True).strip(),
        'context': 512, 'backend': 'native-cpu', 'cache_dtype': 'f32', 'profile': 'balanced',
        'seed': 0, 'temperature': 0, 'max_tokens': 1,
        'generation_timeout_seconds': 600,
        'cache_conditions': 'uncontrolled OS/file cache; fresh means no session KV, not a cold OS cache',
        'scope': 'Pinned FP16 CPU smoke, not a performance superiority, platform matrix or full-release claim.',
        'checks': [],
    }
    with tempfile.TemporaryDirectory(prefix='coheryn-durable-check-') as directory:
        data = Path(directory) / 'service'
        server = None
        log = None

        def stop():
            nonlocal server, log
            if server and server.poll() is None:
                server.send_signal(signal.SIGINT)
                assert server.wait(timeout=30) == 0
            if log:
                log.close()
            server, log = None, None

        def start(round_number):
            nonlocal server, log
            path = ROOT / f'.cache/s2-s3-service-{round_number}.log'
            log = path.open('w')
            server = subprocess.Popen([str(BINARY), '--data-dir', str(data), 'serve', '--port', '0',
                                       '--max-context', '512', '--session-persistence',
                                       '--generation-timeout', '600'],
                                      cwd=ROOT, env=env, stdout=log, stderr=log)
            for _ in range(200):
                assert server.poll() is None, path.read_text()
                for line in path.read_text().splitlines():
                    try:
                        value = json.loads(line)
                    except ValueError:
                        continue
                    if value.get('event') == 'listening':
                        return 'http://' + value['address']
                time.sleep(.05)
            raise RuntimeError('native daemon startup timed out')

        try:
            endpoint = start(1)

            def cli(*args):
                run = subprocess.run([str(BINARY), '--data-dir', str(data), '--endpoint', endpoint,
                                      '--json', *args], cwd=ROOT, env=env,
                                     capture_output=True, text=True, timeout=900)
                assert run.returncode == 0, run.stderr
                return json.loads(run.stdout)

            def http(path, body):
                token = json.loads((data / 'credentials.json').read_text())['inference_token']
                req = urllib.request.Request(endpoint + path, data=json.dumps(body).encode(),
                                             headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
                with opener.open(req, timeout=900) as response:
                    return json.load(response)

            baseline = cli('status')['memory']['reserved']
            cli('models', 'import', str(model), '--name', 'native-chat', '--sha256', pin['sha256'])
            cli('models', 'load', 'native-chat', '--context', '512')
            report['threads_per_session'] = cli('plan', 'native-chat', '--context', '512')['threads_per_session']
            cli('sessions', 'create', 'native-chat', '--name', 'durable', '--context', '512',
                '--system', 'Be brief.', '--persist')
            first_args = ('sessions', 'send', 'durable', '--input', 'Hi', '--expected-revision', '0',
                          '--idempotency-key', 'first', '--max-tokens', '1')
            first = cli(*first_args)
            assert first['revision'] == 2 and first['result']['token_ids'] == case['generated_token_ids'][:1]
            assert first['result']['reused_prompt_tokens'] == 0
            print('PASS pinned independent greedy prefix and initial fresh KV', flush=True)
            second = cli('sessions', 'send', 'durable', '--input', 'What is 2+2?', '--expected-revision', '2',
                         '--idempotency-key', 'second', '--max-tokens', '1')
            assert second['result']['reused_prompt_tokens'] > 0
            view = cli('sessions', 'show', 'durable')
            assert view['cached_tokens'] > 0 and 0 < view['cache_reserved_bytes'] <= 64 << 20
            messages = [{'role': m['role'], 'content': m['content']} for m in view['messages'][:-1]]
            fresh = http('/api/v1/generate', {'model': 'native-chat', 'messages': messages, 'context': 512, 'max_tokens': 1})
            assert fresh['token_ids'] == second['result']['token_ids'] and fresh['text'] == second['result']['text']
            assert fresh['prompt_tokens'] == second['result']['prompt_tokens'] and fresh['reused_prompt_tokens'] == 0
            report['checks'].append({
                'case': 'pinned-greedy-and-matched-fresh-reused-continuation',
                'first_token_ids': first['result']['token_ids'], 'token_ids': fresh['token_ids'],
                'prompt_tokens': fresh['prompt_tokens'], 'reused_prompt_tokens': second['result']['reused_prompt_tokens'],
                'cache_reserved_bytes': view['cache_reserved_bytes'],
                'reused_prefill_ms': second['result']['prefill_ms'], 'fresh_prefill_ms': fresh['prefill_ms'],
                'reused_memory': second['result']['memory'], 'fresh_memory': fresh['memory'],
            })
            cli('sessions', 'export', 'durable', '--output', str(Path(directory) / 'history.json'))
            print('PASS native cached/stateless token parity and bounded cache; restarting', flush=True)
            stop()
            endpoint = start(2)
            restarted = cli('sessions', 'show', 'durable')
            assert restarted['messages'] == view['messages'] and restarted['revision'] == 4
            assert restarted['persistent'] and restarted['cached_tokens'] == 0
            assert cli('status')['loaded_models'] == []
            assert cli(*first_args) == first
            report['checks'].append({'case': 'restart-exact-history-replay-without-weights', 'revision': 4,
                                     'message_count': restarted['message_count'], 'cached_tokens': 0})
            cli('models', 'load', 'native-chat', '--context', '512')
            third = cli('sessions', 'send', 'durable', '--input', 'Say goodbye.', '--expected-revision', '4',
                        '--idempotency-key', 'third', '--max-tokens', '1')
            assert third['result']['reused_prompt_tokens'] == 0 and third['revision'] == 6
            cli('sessions', 'reset', 'durable', '--expected-revision', '6')
            cli('sessions', 'delete', 'durable', '--expected-revision', '7')
            cli('models', 'unload', 'native-chat')
            status = cli('status')
            assert status['memory']['reserved'] == baseline
            assert status['scheduler']['active'] == 0 and status['scheduler']['queued'] == 0
            assert cli('sessions', 'list')['sessions'] == []
            report['checks'].append({'case': 'explicit-model-reload-cold-rebuild-reset-delete-unload',
                                     'reused_prompt_tokens': 0, 'remaining_reserved_bytes': baseline,
                                     'remaining_reservation': 'bounded embedded-store baseline, no session/model buffers'})
            stop()
            report['graceful_restarts'] = True
            print('PASS restart replay, explicit cold rebuild and complete session/model cleanup', flush=True)
        finally:
            if server and server.poll() is None:
                server.kill()
                server.wait(timeout=10)
            if log:
                log.close()
    (ROOT / 'fixtures/qualification/s2-s3-service.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

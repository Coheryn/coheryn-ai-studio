#!/usr/bin/env python3
"""Opt-in pinned native tool qualification using inert reference data and local clients."""
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time
from fetch_model import digest

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / 'target/release/coheryn'


def main():
    sys.path.insert(0, str(ROOT / '.cache/rag-tools'))
    try:
        import httpx
        import openai
    except ImportError as error:
        raise SystemExit('Install the optional tools/requirements-rag.txt clients in an isolated environment first.') from error
    reference = json.loads((ROOT / 'fixtures/tools/qwen2.5-reference.json').read_text())
    cases = {c['id']: c for c in reference['cases']}
    pin = reference['model']
    model = ROOT / '.cache/models' / pin['file']
    assert model.stat().st_size == pin['bytes'] and digest(model) == pin['sha256']
    manifest = json.loads((ROOT / 'fixtures/tools/manifest.json').read_text())
    companion_sha256 = manifest['companion_template']['sha256']
    assert digest(ROOT / 'fixtures/tools/qwen2.5-tools-v1.jinja') == companion_sha256
    assert manifest['companion_template']['reference_body_sha256'] == reference['template_sha256']
    env = {k: v for k, v in os.environ.items() if k not in {'COHERYN_API_TOKEN', 'COHERYN_ADMIN_TOKEN'}}
    report = {'schema_version': 1, 'stage': 'S4 native CPU tool-protocol preview',
              'model_sha256': pin['sha256'], 'model_bytes': pin['bytes'],
              'binary_sha256': digest(BINARY), 'reference_sha256': digest(ROOT / 'fixtures/tools/qwen2.5-reference.json'),
              'tool_profile': manifest['profile_identity'],
              'os': platform.platform(), 'architecture': platform.machine(), 'backend': 'native-cpu',
              'context': 512, 'cache_dtype': 'f32', 'profile': 'balanced', 'temperature': 0, 'seed': 0,
              'generation_timeout_seconds': 1200, 'sdk_version': openai.__version__,
              'cache_conditions': 'uncontrolled OS/file cache; session KV reuse reported independently',
              'rss_bytes': None, 'process_private_bytes': None, 'page_faults': None, 'io_bytes': None,
              'unavailable_measurements': 'This correctness/lifecycle smoke does not sample process RSS/private/page-fault/I/O counters.',
              'scope': 'Pinned macOS ARM64 CPU correctness and protocol evidence, not a performance comparison or full-release qualification.',
              'checks': []}
    # Verify tokenizer output against independently captured IDs before executing weights.
    for case in reference['cases']:
        run = subprocess.run([str(BINARY), '--json', 'tokenize', str(model), '--prompt', case['prompt']],
                             cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
        assert run.returncode == 0, run.stderr
        value = json.loads(run.stdout)
        assert value['token_ids'] == case['prompt_token_ids'], case['id']
    report['checks'].append({'case': 'all-independent-tool-prompt-token-ids', 'cases': len(reference['cases'])})
    print('PASS all independent tool prompt token IDs', flush=True)
    with tempfile.TemporaryDirectory(prefix='coheryn-tools-check-') as directory:
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

        def start(number):
            nonlocal server, log
            path = ROOT / f'.cache/s4-service-{number}.log'
            log = path.open('w')
            server = subprocess.Popen([str(BINARY), '--data-dir', str(data), 'serve', '--port', '0',
                                       '--max-context', '512', '--session-persistence', '--generation-timeout', '1200'],
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

            def cli(*args, lines=False):
                run = subprocess.run([str(BINARY), '--data-dir', str(data), '--endpoint', endpoint, '--json', *args],
                                     cwd=ROOT, env=env, capture_output=True, text=True, timeout=1500)
                assert run.returncode == 0, run.stderr
                return [json.loads(s) for s in run.stdout.splitlines()] if lines else json.loads(run.stdout)

            cli('models', 'import', str(model), '--name', 'tools-chat', '--sha256', pin['sha256'])
            cli('models', 'load', 'tools-chat', '--context', '512')
            plan = cli('plan', 'tools-chat', '--context', '512')
            report['hardware'] = plan['hardware']
            report['threads_per_session'] = plan['threads_per_session']
            inspected = cli('models', 'inspect', 'tools-chat')
            assert inspected['tool_model_qualified'] is True
            assert inspected['compatibility']['tool_protocol']['companion_template_sha256'] == companion_sha256
            credentials = json.loads((data / 'credentials.json').read_text())
            with openai.OpenAI(base_url=endpoint + '/v1', api_key=credentials['inference_token'], max_retries=0,
                               http_client=httpx.Client(trust_env=False, timeout=1500), timeout=1500) as client:
                stream = client.chat.completions.create(model='tools-chat',
                    messages=[{'role': 'user', 'content': cases['add']['messages'][0]['content']}],
                    tools=cases['add']['tools'], tool_choice='required', parallel_tool_calls=False,
                    temperature=0, max_tokens=96, stream=True, stream_options={'include_usage': True},
                    extra_body={'context': 512, 'seed': 0, 'top_k': 0, 'threads': plan['threads_per_session']})
                arguments, name, call_id, text, finish, usage = '', '', '', '', None, None
                for chunk in stream:
                    if chunk.usage:
                        usage = chunk.usage.model_dump()
                    for choice in chunk.choices:
                        text += choice.delta.content or ''
                        if choice.finish_reason:
                            finish = choice.finish_reason
                        for delta in choice.delta.tool_calls or []:
                            assert delta.index == 0
                            call_id = delta.id or call_id
                            if delta.function:
                                name = delta.function.name or name
                                arguments += delta.function.arguments or ''
                assert finish == 'tool_calls' and name == 'add' and call_id
                assert json.loads(arguments) == {'a': 2, 'b': 2} and text == ''
                assert usage['prompt_tokens'] == len(cases['add']['prompt_token_ids'])
                assert usage['completion_tokens'] == len(cases['add']['generated_token_ids'])
                report['checks'].append({'case': 'standard-sdk-streaming-required-single-call', 'id': call_id,
                                         'name': name, 'arguments': arguments, 'content': None,
                                         'finish_reason': finish, 'usage': usage})
            print('PASS standard SDK streaming tool call, finish and exact usage', flush=True)
            cli('sessions', 'create', 'tools-chat', '--name', 'proof', '--context', '512', '--persist')
            first = {'expected_revision': 0, 'idempotency_key': 'first',
                     'content': cases['parallel-add']['messages'][0]['content'],
                     'settings': {'max_tokens': 96, 'tools': cases['parallel-add']['tools'],
                                  'tool_choice': 'required', 'parallel_tool_calls': True}}
            first_file = Path(directory) / 'first.json'
            first_file.write_text(json.dumps(first))
            events = cli('sessions', 'submit', 'proof', '--request', str(first_file), '--stream', lines=True)
            result = events[-1]['result']
            assert result['token_ids'] == cases['parallel-add']['generated_token_ids']
            assert result['finish_reason'] == 'tool_calls' and len(result['tool_calls']) == 2
            calls = result['tool_calls']
            assert len({c['id'] for c in calls}) == 2 and call_id not in {c['id'] for c in calls}
            for index, call in enumerate(calls):
                assert call['function']['name'] == 'add'
                fragments = [e['tool_call'] for e in events if e.get('tool_call', {}).get('index') == index]
                assert fragments[0]['id'] == call['id']
                assert ''.join(d['function']['arguments'] for d in fragments) == call['function']['arguments']
                assert json.loads(call['function']['arguments']) == ({'a': 2, 'b': 2} if index == 0 else {'a': 3, 'b': 4})
            view = cli('sessions', 'show', 'proof')
            assert view['revision'] == 2 and view['pending_tool_calls'] == [c['id'] for c in calls]
            assert view['tool_profile'] == report['tool_profile']
            exported = Path(directory) / 'pending.json'
            cli('sessions', 'export', 'proof', '--output', str(exported))
            assert json.loads(exported.read_text())['version'] == 2
            copied = cli('sessions', 'import', str(exported), '--name', 'waiting-copy', '--persist')
            assert copied['pending_tool_calls'] == view['pending_tool_calls']
            report['checks'].append({'case': 'managed-cli-parallel-reference-greedy-and-delta-parity',
                                     'token_ids': result['token_ids'], 'prompt_tokens': result['prompt_tokens'],
                                     'tool_calls': calls, 'memory': result['memory'], 'history_format_version': 2})
            print('PASS native CLI multiple calls, reference greedy IDs and versioned pending export/import', flush=True)
            response = {'expected_revision': 2, 'idempotency_key': 'results',
                        'tool_results': [{'tool_call_id': c['id'], 'content': str(4 if i == 0 else 7)} for i, c in enumerate(calls)],
                        'settings': {'max_tokens': 1}}
            response_file = Path(directory) / 'results.json'
            response_file.write_text(json.dumps(response))
            continuation = cli('sessions', 'submit', 'proof', '--request', str(response_file))
            assert continuation['result']['token_ids'] == cases['parallel-add-results']['generated_token_ids']
            assert continuation['result']['prompt_tokens'] == len(cases['parallel-add-results']['prompt_token_ids'])
            assert continuation['revision'] == 4 and cli('sessions', 'show', 'proof').get('pending_tool_calls', []) == []
            report['checks'].append({'case': 'ordered-tool-results-independent-continuation',
                                     'token_ids': continuation['result']['token_ids'],
                                     'prompt_tokens': continuation['result']['prompt_tokens'],
                                     'reused_prompt_tokens': continuation['result']['reused_prompt_tokens']})
            stop()
            endpoint = start(2)
            assert cli('status')['loaded_models'] == []
            waiting = cli('sessions', 'show', 'waiting-copy')
            assert waiting['pending_tool_calls'] == view['pending_tool_calls'] and waiting['cached_tokens'] == 0
            replay = cli('sessions', 'submit', 'proof', '--request', str(first_file), '--stream', lines=True)
            assert replay[-1]['result'] == result
            assert cli('status')['loaded_models'] == []
            for alias in ['proof', 'waiting-copy']:
                revision = cli('sessions', 'show', alias)['revision']
                cli('sessions', 'delete', alias, '--expected-revision', str(revision))
            status = cli('status')
            assert status['memory']['reserved'] == 16 << 20
            assert status['scheduler']['active'] == 0 and status['scheduler']['queued'] == 0
            report['checks'].append({'case': 'pending-restart-replay-no-auto-execution-and-cleanup',
                                     'pending_call_ids': waiting['pending_tool_calls'], 'loaded_models_after_replay': [],
                                     'remaining_reserved_bytes': 16 << 20})
            stop()
            print('PASS result continuation, cold pending restart, exact replay without weights and cleanup', flush=True)
        finally:
            if server and server.poll() is None:
                server.kill()
                server.wait(timeout=10)
            if log:
                log.close()
    (ROOT / 'fixtures/qualification/s4-service.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

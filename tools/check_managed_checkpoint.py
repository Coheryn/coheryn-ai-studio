#!/usr/bin/env python3
"""Opt-in native managed-checkpoint/API proof using already acquired pinned bytes."""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True, help='New qualification record; existing records are never overwritten.')
    parser.add_argument('--sessions', action='store_true', help='Also qualify Qwen3.5 managed histories, persistence and lifecycle.')
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    manifest_path = ROOT / 'fixtures/qwen35/model.json'
    manifest = json.loads(manifest_path.read_text())
    source = ROOT / manifest['cache_directory']
    for item in manifest['files']:
        path = source / item['file']
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], item['file']
    reference_path = ROOT / 'fixtures/qwen35/reference-v1.json'
    reference = json.loads(reference_path.read_text())
    case = reference['cases'][0]
    binary = ROOT / 'target/release/coheryn'
    output = ROOT / ('.cache/b2-managed' if args.sessions else '.cache/b1-managed')
    output.mkdir(parents=True, exist_ok=True)
    empty = output / 'empty-path'
    empty.mkdir(exist_ok=True)
    env = {**os.environ, 'PATH': str(empty)}
    env.pop('COHERYN_API_TOKEN', None)
    env.pop('COHERYN_ADMIN_TOKEN', None)
    size = sum(item['bytes'] for item in manifest['files'])
    assert shutil.disk_usage(output).free > size + (512 << 20), 'Need space for one private managed copy.'
    results = []
    server = None
    log = None
    endpoint = None
    tokens = None
    with tempfile.TemporaryDirectory(prefix='managed-', dir=output) as temporary:
        vault = Path(temporary) / 'vault'
        messages = Path(temporary) / 'messages.json'
        messages.write_text(json.dumps(case['messages'], ensure_ascii=False))
        def start(label):
            nonlocal server, log, endpoint, tokens
            log = (output / (label + '-daemon.jsonl')).open('w')
            server = subprocess.Popen([str(binary), '--data-dir', str(vault), 'serve', '--port', '0', '--max-memory', str(768 << 20), '--max-context', '512', '--generation-timeout', '300', '--idle-cache-bytes', '0', '--session-cache-bytes', '0'] + (['--session-persistence'] if args.sessions else []), env=env, stdout=log, stderr=log)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                assert server.poll() is None, (output / (label + '-daemon.jsonl')).read_text()
                log.flush()
                for line in (output / (label + '-daemon.jsonl')).read_text().splitlines():
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if value.get('event') == 'listening':
                        endpoint = value['address']
                        tokens = json.loads((vault / 'credentials.json').read_text())
                        return
                time.sleep(.05)
            raise TimeoutError('Daemon did not start.')
        def stop():
            nonlocal server, log
            if server is not None:
                server.send_signal(signal.SIGINT)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
                server = None
            if log is not None:
                log.close()
                log = None
        def cli(label, arguments, expected=0):
            command = [str(binary), '--data-dir', str(vault), '--endpoint', 'http://' + endpoint, *arguments, '--json']
            p = subprocess.run(command, env=env, capture_output=True, text=True, timeout=600)
            (output / (label + '.stdout.json')).write_text(p.stdout)
            (output / (label + '.stderr.txt')).write_text(p.stderr)
            assert p.returncode == expected, (label, p.returncode, p.stderr)
            value = json.loads(p.stdout) if p.stdout.strip() else None
            results.append({'case': label, 'exit': p.returncode, 'stdout_sha256': sha(output / (label + '.stdout.json'))})
            print('PASS', label, flush=True)
            return value
        def request(label, method, path, value=None, expected=200, stream=False):
            connection = http.client.HTTPConnection(endpoint, timeout=360)
            admin = path.startswith('/api/') and path != '/api/v1/generate'
            headers = {'Authorization': 'Bearer ' + tokens['admin_token' if admin else 'inference_token'], 'Content-Type': 'application/json'}
            connection.request(method, path, None if value is None else json.dumps(value).encode(), headers)
            response = connection.getresponse()
            if stream:
                assert response.status == expected, response.read()
                return connection, response
            body = response.read(16 << 20)
            assert len(body) < (16 << 20)
            data = json.loads(body)
            assert response.status == expected, (label, response.status, data)
            connection.close()
            (output / (label + '.json')).write_bytes(body)
            results.append({'case': label, 'status': response.status, 'sha256': hashlib.sha256(body).hexdigest()})
            print('PASS', label, flush=True)
            return data
        def idle(label):
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                connection = http.client.HTTPConnection(endpoint, timeout=10)
                connection.request('GET', '/api/v1/server/status', headers={'Authorization': 'Bearer ' + tokens['admin_token']})
                response = connection.getresponse()
                data = json.loads(response.read())
                connection.close()
                if data['scheduler']['active'] == 0 and data['scheduler']['queued'] == 0 and data['memory']['reserved'] == (256 << 20) + ((16 << 20) if args.sessions else 0):
                    results.append({'case': label, 'memory': data['memory'], 'scheduler': data['scheduler']})
                    return data
                time.sleep(.05)
            raise TimeoutError(('Reservations/workers did not retire', data))
        def check_sessions():
            prefix = '/coheryn/v1/sessions'
            capabilities = request('session-capabilities', 'GET', '/api/v1/models/q35/inspect')
            assert capabilities['compatibility']['sessions'] and capabilities['compatibility']['prefix_kv_reuse']
            request('session-capacity-rejected', 'POST', prefix, {'model': 'q35', 'context': 512}, expected=400)
            request('session-instructions-rejected', 'POST', prefix, {'model': 'q35', 'context': 256, 'system': '<|im_start|>'}, expected=400)
            created = request('session-create', 'POST', prefix, {'model': 'q35', 'alias': 'notes', 'context': 256, 'persist': True})
            route = prefix + '/' + created['id']
            first_body = {'expected_revision': 0, 'idempotency_key': 'first', 'stream': True, 'content': case['messages'][0]['content'], 'settings': {'max_tokens': 4, 'threads': 2, 'profile': 'balanced'}}
            for key, value in [('temperature', .5), ('response_format', {'type': 'json_object'}), ('stop', ['END'])]:
                request('session-unsupported-' + key, 'POST', route + '/messages', {**first_body, 'settings': {**first_body['settings'], key: value}}, expected=400)
            untouched = request('session-unsupported-no-write', 'GET', route)
            assert untouched['revision'] == 0 and untouched['active_turn'] is None and untouched['messages'] == []
            connection, response = request('session-first-sse', 'POST', route + '/messages', {**first_body, 'stream': True}, stream=True)
            raw = response.read(16 << 20).decode(); connection.close()
            frames = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: ') and line[6:] != '[DONE]']
            assert raw.count('data: [DONE]') == 1
            first = next(v for v in frames if v.get('status') == 'completed')
            assert first['result']['token_ids'] == case['greedy_token_ids'] and first['result']['text'] == case['greedy_text']
            assert ''.join(v['token']['text'] for v in frames if 'token' in v) == case['greedy_text']
            assert first['result']['reused_prompt_tokens'] == 0 and first['context']['prompt_tokens'] == len(case['prompt_token_ids'])
            results.append({'case': 'session-first-sse-frozen-reference', 'receipt': first, 'chunks': len(frames)})
            before = request('session-first-history', 'GET', route)
            assert before['revision'] == 2 and before['cached_tokens'] == 0 and before['cache_reserved_bytes'] == 0
            request('session-unload-before-replay', 'POST', '/api/v1/models/unload', {'model': 'q35'})
            replay = request('session-replay-unloaded', 'POST', route + '/messages', first_body)
            assert replay == first
            stop(); start('history-restart')
            restored = request('session-restart-history-before-load', 'GET', route)
            assert restored['messages'] == before['messages'] and restored['revision'] == before['revision'] and restored['cached_tokens'] == 0
            replay = request('session-replay-after-restart-unloaded', 'POST', route + '/messages', first_body)
            assert replay == first
            request('session-reload', 'POST', '/api/v1/models/load', {'model': 'q35', 'context': 256})
            followup = {'expected_revision': 2, 'idempotency_key': 'followup', 'content': '你好. বাংলা again.', 'settings': first_body['settings']}
            file = Path(temporary) / 'followup.json'; file.write_text(json.dumps(followup, ensure_ascii=False))
            second = cli('session-cli-followup', ['sessions', 'submit', 'notes', '--request', str(file)])
            assert second['status'] == 'completed' and second['revision'] == 4 and second['result']['reused_prompt_tokens'] == 0
            history = [{'role': m['role'], 'content': m['content']} for m in before['messages']]
            parity = request('session-followup-stateless-parity', 'POST', '/api/v1/generate', {'model': 'q35', 'messages': history + [{'role': 'user', 'content': followup['content']}], 'context': 256, **first_body['settings']})
            assert parity['token_ids'] == second['result']['token_ids'] and parity['text'] == second['result']['text'] and parity['prompt_tokens'] == second['result']['prompt_tokens']
            after = request('session-followup-history', 'GET', route)
            assert after['message_count'] == 4 and after['cached_tokens'] == 0
            document = request('session-portable-export', 'GET', route + '/export')
            assert document['session']['model_sha256'] == imported['sha256'] and document['session']['cache_reserved_bytes'] == 0
            exported = Path(temporary) / 'history.json'
            cli('session-cli-export', ['sessions', 'export', 'notes', '--output', str(exported)])
            assert json.loads(exported.read_text())['messages'] == document['messages']
            copied = cli('session-cli-import', ['sessions', 'import', str(exported), '--name', 'copy', '--persist'])
            assert copied['id'] != created['id'] and copied['model_sha256'] == imported['sha256']
            changed = json.loads(json.dumps(document)); changed['session']['model_sha256'] = '0' * 64
            mismatch = request('session-import-old-identity', 'POST', prefix + '/import', {'document': changed, 'name': 'mismatch'})
            mismatch_route = prefix + '/' + mismatch['id']
            request('session-identity-rebind-rejected', 'POST', mismatch_route + '/messages', {**followup, 'expected_revision': mismatch['revision'], 'idempotency_key': 'wrong'}, expected=409)
            rejected = request('session-identity-writer-released', 'GET', mismatch_route)
            assert rejected['active_turn'] is None
            request('session-delete-mismatch', 'DELETE', mismatch_route, {'expected_revision': rejected['revision']})
            reset = request('session-copy-reset', 'POST', prefix + '/' + copied['id'] + '/reset', {'expected_revision': copied['revision']})
            request('session-delete-copy', 'DELETE', prefix + '/' + copied['id'], {'expected_revision': reset['revision']})
            connection, response = request('session-cancel-live', 'POST', route + '/messages', {'expected_revision': 4, 'idempotency_key': 'cancel', 'content': 'Continue counting for a long time.', 'stream': True, 'settings': {**first_body['settings'], 'max_tokens': 128}}, stream=True)
            request('session-unload-active-rejected', 'POST', '/api/v1/models/unload', {'model': 'q35'}, expected=409)
            assert response.readline(); response.close(); connection.close()
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                c = http.client.HTTPConnection(endpoint, timeout=10)
                c.request('GET', route, headers={'Authorization': 'Bearer ' + tokens['inference_token']})
                r = c.getresponse(); view = json.loads(r.read()); c.close()
                c = http.client.HTTPConnection(endpoint, timeout=10)
                c.request('GET', '/api/v1/server/status', headers={'Authorization': 'Bearer ' + tokens['admin_token']})
                r = c.getresponse(); status = json.loads(r.read()); c.close()
                if view.get('active_turn') is None and status['scheduler']['active'] == 0 and status['memory']['session_buffer_capacity_bytes'] == 0:
                    break
                time.sleep(.05)
            else:
                raise TimeoutError('Managed worker/writer did not retire after disconnect.')
            cancelled = request('session-cancelled-portable-receipt', 'GET', route + '/export')
            assert any(t['receipt']['status'] == 'cancelled' for t in cancelled['turns'])
            assert view['cached_tokens'] == 0 and view['revision'] == 6
            request('session-delete-original', 'DELETE', route, {'expected_revision': view['revision']})
            remaining = request('session-history-cleanup', 'GET', prefix)
            assert remaining['sessions'] == [] and remaining['persistence_enabled'], remaining
            idle('session-resources-reclaimed')
        try:
            start('initial')
            imported = cli('checkpoint-import', ['models', 'import', str(source), '--name', 'q35'])
            assert imported['schema_version'] == 2 and imported['checkpoint']['adapter'] == 'qwen35-safetensors-cpu-v1'
            assert imported['checkpoint']['entrypoint'] == 'model.safetensors.index.json'
            files = imported['checkpoint']['files']
            pins = {item['file']: item for item in manifest['files']}
            assert all(item['bytes'] == pins[item['name']]['bytes'] and item['sha256'] == pins[item['name']]['sha256'] for item in files)
            assert any(item['name'] == 'LICENSE' for item in files)
            canonical = json.dumps({'schema_version': 1, 'entrypoint': imported['checkpoint']['entrypoint'], 'adapter': imported['checkpoint']['adapter'], 'files': [{'name': f['name'], 'bytes': f['bytes'], 'sha256': f['sha256']} for f in files]}, separators=(',', ':'), ensure_ascii=False).encode()
            assert hashlib.sha256(b'coheryn-checkpoint-v1\0' + canonical).hexdigest() == imported['sha256']
            info = request('managed-inspection', 'GET', '/api/v1/models/q35/inspect')
            assert info['descriptor']['text_adapter']['managed_api'] and info['source_set_sha256'] == imported['sha256']
            planning = {'model': 'q35', 'context': 256, 'concurrency': 1, 'profile': 'balanced', 'threads': 2}
            unloaded = request('unloaded-plan', 'POST', '/api/v1/plans/estimate', planning)
            assert unloaded['admissible'] and unloaded['model_metadata_additional'] == (256 << 20)
            request('load', 'POST', '/api/v1/models/load', {'model': 'q35', 'context': 256})
            loaded = request('loaded-plan', 'POST', '/api/v1/plans/estimate', planning)
            assert loaded['admissible'] and loaded['model_metadata_additional'] == 0
            assert loaded['required_reserved_bytes'] == unloaded['required_reserved_bytes']
            request('remove-loaded-rejected', 'DELETE', '/api/v1/models/q35', expected=409)
            body = {'model': 'q35', 'messages': case['messages'], 'max_tokens': 4, 'context': 256, 'threads': 2, 'profile': 'balanced'}
            complete = request('chat-completion', 'POST', '/v1/chat/completions', body)
            assert complete['choices'][0]['message']['content'] == case['greedy_text']
            assert complete['usage']['prompt_tokens'] == len(case['prompt_token_ids'])
            assert complete['coheryn']['sampling_profile'] is None
            generated = cli('daemon-cli-generation', ['run', 'q35', '--messages', str(messages), '--context', '256', '--max-tokens', '4', '--threads', '2'])
            assert generated['token_ids'] == case['greedy_token_ids'] and generated['text'] == case['greedy_text']
            assert generated['memory']['session_buffer_capacity_bytes'] == 0 and generated['batching']['profile'] == 'coheryn-qwen35-batch-v1'
            assert generated['memory']['peak_reserved'] == loaded['required_reserved_bytes']
            connection, response = request('chat-sse', 'POST', '/v1/chat/completions', {**body, 'stream': True, 'stream_options': {'include_usage': True}}, stream=True)
            raw = response.read(16 << 20).decode()
            connection.close()
            chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: ') and line[6:] != '[DONE]']
            assert raw.count('data: [DONE]') == 1
            assert ''.join(v['choices'][0]['delta'].get('content', '') for v in chunks if v['choices']) == case['greedy_text']
            assert any(v.get('usage') == complete['usage'] for v in chunks)
            results.append({'case': 'chat-sse-parity', 'chunks': len(chunks), 'done': 1})
            unicode_case = reference['cases'][1]
            unicode_result = request('unicode-chat-completion', 'POST', '/v1/chat/completions', {**body, 'messages': unicode_case['messages'], 'max_tokens': len(unicode_case['greedy_token_ids'])})
            assert unicode_result['choices'][0]['message']['content'] == unicode_case['greedy_text']
            assert unicode_result['usage']['prompt_tokens'] == len(unicode_case['prompt_token_ids'])
            request('memory-rejection', 'POST', '/v1/chat/completions', {**body, 'max_memory': 1}, expected=503)
            for key, value in [('temperature', .5), ('stop', ['END']), ('top_k', 2), ('response_format', {'type': 'json_object'})]:
                request('unsupported-' + key, 'POST', '/v1/chat/completions', {**body, key: value}, expected=400)
            # A bounded native streaming request pins its source while active; disconnect
            # cancellation must acknowledge the joined worker before memory is reclaimed.
            connection, response = request('cancel-stream', 'POST', '/api/v1/generate', {**body, 'max_tokens': 128, 'stream': True}, stream=True)
            request('unload-active-rejected', 'POST', '/api/v1/models/unload', {'model': 'q35'}, expected=409)
            first = response.readline()
            assert first
            response.close()
            connection.close()
            after = idle('disconnect-cleanup')
            assert after['scheduler']['cancelled'] >= 1
            request('unload', 'POST', '/api/v1/models/unload', {'model': 'q35'})
            stop()
            start('restart')
            restarted = request('restart-record', 'GET', '/api/v1/models')
            assert restarted['models'][0]['sha256'] == imported['sha256'] and restarted['loaded'] == []
            request('reload', 'POST', '/api/v1/models/load', {'model': 'q35', 'context': 256})
            # Raw completion reaches the same graph as the qualified rendered prompt.
            raw_body = {'model': 'q35', 'prompt': case['prompt'], 'max_tokens': 4, 'threads': 1, 'profile': 'reference'}
            raw_complete = request('raw-completion', 'POST', '/v1/completions', raw_body)
            assert raw_complete['choices'][0]['text'] == case['greedy_text']
            if args.sessions:
                check_sessions()
            request('final-unload', 'POST', '/api/v1/models/unload', {'model': 'q35'})
            # Same-length corruption must fail full verification before becoming resident.
            weight = vault / 'models/model-q35.checkpoint/model.safetensors-00001-of-00001.safetensors'
            with weight.open('r+b') as file:
                file.seek(-1, 2)
                value = file.read(1)
                file.seek(-1, 2)
                file.write(bytes([value[0] ^ 1]))
            corrupt = request('payload-corruption-rejected', 'POST', '/api/v1/models/load', {'model': 'q35', 'context': 256}, expected=409)
            assert corrupt['error']['code'] == 'MODEL_CHANGED'
            request('remove', 'DELETE', '/api/v1/models/q35')
            final = request('final-status', 'GET', '/api/v1/server/status')
            assert final['memory']['reserved'] == ((16 << 20) if args.sessions else 0) and final['loaded_models'] == []
        finally:
            stop()
    report = {'schema_version': 1, 'scope': ('Idle caches explicitly disabled for historical full-prefill regression. Includes managed history/CLI/SSE, durable restart/replay without residency, portable import/export, identity rebinding rejection and cancellation cleanup; no prefix/recurrent cache reuse. ' if args.sessions else '') + 'Opt-in native managed Safetensors source-set import/load/planning, stateless CLI/API/SSE parity, disconnect cleanup, restart and payload integrity on macOS ARM64. One physical source shard; real multi-shard execution, prefix/recurrent state reuse, broad fairness/backpressure campaigns, GPU and other platforms remain unqualified.', 'binary_sha256': sha(binary), 'model_manifest_sha256': sha(manifest_path), 'reference_sha256': sha(reference_path), 'managed_record': imported, 'cases': results, 'cache_condition': 'uncontrolled', 'external_executor_invoked': False, 'managed_sessions': args.sessions}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('PASS', len(results), 'managed checkpoint/API checks; private managed copy removed.', flush=True)

if __name__ == '__main__':
    main()

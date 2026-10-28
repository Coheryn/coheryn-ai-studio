#!/usr/bin/env python3
"""Qualify native image conversation persistence, no-resend parity and shared compute.

Uses only cached pinned GGUF bytes, the release CLI and frozen reference inputs.
Private hardlinks avoid duplicating model weights. Original image assets are copied
by the actual authenticated upload route; no external executor is invoked.
"""
import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    binary = ROOT/'target/release/coheryn'
    binary_sha = digest(binary)
    references = ROOT/'fixtures/vlm/qwen2-image-conditioned-reference-v2.json'
    case = next(c for c in json.loads(references.read_text())['cases'] if c['id'] == 'left-red')
    run = ROOT/'.cache/m4-image-sessions'/uuid.uuid4().hex
    data = run/'data'
    for path in [run, data, data/'models', data/'records']:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    components = [
        ('language', ROOT/'.cache/vlm/full/qwen2-vl-2b/Qwen2-VL-2B-Instruct-f16.gguf', case['identity']['language_sha256']),
        ('vision', ROOT/'.cache/vlm/full/qwen2-vl-2b/mmproj-Qwen2-VL-2B-Instruct-f16.gguf', case['identity']['vision_sha256']),
        ('small-text', ROOT/'.cache/models/qwen2.5-0.5b-instruct-q4_k_m.gguf', '74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db'),
    ]
    for name, source, sha in components:
        assert source.is_file(), 'acquire the pinned component bytes before qualification'
        os.link(source, data/'models'/f'model-{name}.gguf')
        (data/'records'/f'model-{name}.json').write_text(json.dumps(dict(schema_version=1, id=name, sha256=sha, bytes=source.stat().st_size, source=None)))
    env = dict(os.environ, PATH='')
    process = log = None
    address = None
    def start(number):
        nonlocal process, log, address
        log_path = run/f'daemon-{number}.log'
        log = log_path.open('w')
        process = subprocess.Popen([str(binary), '--data-dir', str(data), 'serve', '--port', '0', '--session-persistence',
            '--max-memory', str(1536<<20), '--max-context', '512', '--queue-capacity', '3', '--generation-timeout', '600'],
            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=log)
        address = None
        for _ in range(300):
            if process.poll() is not None:
                raise RuntimeError('native daemon exited; inspect the private run log')
            for line in log_path.read_text().splitlines():
                try:
                    v = json.loads(line)
                except ValueError:
                    continue
                if v.get('event') == 'listening':
                    address = v['address']
            if address:
                return
            time.sleep(.1)
        raise RuntimeError('native daemon startup timeout')
    def stop():
        nonlocal process, log
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=40)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            process = None
        if log is not None:
            log.close(); log = None
    def command(args):
        assert digest(binary) == binary_sha, 'release executable changed during qualification'
        result = subprocess.run([str(binary), '--json', '--endpoint', f'http://{address}', '--data-dir', str(data), *args],
            cwd=ROOT, env=env, capture_output=True, timeout=650)
        if result.returncode:
            raise RuntimeError(result.stderr.decode()[:2048])
        return json.loads(result.stdout)
    def call(method, path, body=None, admin=False, raw=None):
        credentials = json.loads((data/'credentials.json').read_text())
        host, port = address.rsplit(':', 1)
        connection = http.client.HTTPConnection(host, int(port), timeout=650)
        payload = raw if raw is not None else (None if body is None else json.dumps(body, ensure_ascii=False).encode())
        headers = {'Authorization':'Bearer '+credentials['admin_token' if admin else 'inference_token'],
                   'Content-Type':'application/octet-stream' if raw is not None else 'application/json'}
        connection.request(method, path, payload, headers)
        response = connection.getresponse()
        content = response.read(16<<20)
        connection.close()
        value = json.loads(content)
        assert response.status == 200, (response.status, value)
        return value
    def record(name, value):
        path = run/f'{name}.json'
        path.write_text(json.dumps(value, indent=2)+'\n')
        return dict(path=str(path.relative_to(ROOT)), sha256=digest(path))
    try:
        start(1)
        pair = command(['models','register-pair','--name','qwen-vl','--language','language','--vision','vision'])
        command(['models','load','qwen-vl','--context','256'])
        command(['models','load','small-text','--context','128'])
        created = call('POST','/coheryn/v1/sessions',dict(model='qwen-vl', alias='scene', context=256))
        session = created['id']
        image_path = ROOT/'fixtures/vlm/preprocessing/canonical/landscape-png.png'
        image = image_path.read_bytes()
        attached = call('POST', f'/coheryn/v1/sessions/{session}/images?expected_revision=0', raw=image)
        image_id = attached['image']['id']
        question = 'Which color is on the left? Answer with one color word.'
        settings = dict(max_tokens=8,threads=2,profile='balanced')
        request = dict(expected_revision=1,idempotency_key='first',content=question,images=[image_id],settings=settings)
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(call,'POST',f'/coheryn/v1/sessions/{session}/messages',request)
            for _ in range(1200):
                status = call('GET','/api/v1/server/status',admin=True)
                if status['scheduler']['vision_units'] >= 8:
                    break
                if first.done():
                    raise AssertionError('image request ended before the overlap probe')
                time.sleep(.05)
            else:
                raise AssertionError('image numeric work did not start')
            text_start = time.monotonic()
            text = call('POST','/v1/completions',dict(model='small-text',prompt='Hello',max_tokens=1,context=128,profile='reference',threads=1))
            text_elapsed = time.monotonic()-text_start
            assert not first.done(), 'text must progress before image generation completes with max_concurrency=1'
            first = first.result(timeout=650)
        first_elapsed = time.monotonic()-started
        assert first['result']['token_ids'] == [t['token_id'] for t in case['generation']]
        assert first['result']['text'] == case['text']
        assert first['result']['prompt_tokens'] == case['encoded']['physical_prompt_tokens']
        assert first['result']['vision']['cooperative']['yields'] > 0
        assert first['result']['vision']['images'] == 1
        first_receipt = record('first-image', first)
        print('PASS native image reference and concurrent text progress', flush=True)
        before = call('GET',f'/coheryn/v1/sessions/{session}')
        saved = call('POST',f'/coheryn/v1/sessions/{session}/save',dict(expected_revision=first['revision']))
        assert saved['persistent'] and saved['revision'] == 4
        stop()
        start(2)
        # Read authoritative history before loading any weights; KV and pixels are not restored.
        restored = call('GET',f'/coheryn/v1/sessions/{session}')
        status = call('GET','/api/v1/server/status',admin=True)
        assert status['loaded_models'] == []
        assert restored['messages'] == before['messages']
        assert restored['cached_tokens'] == 0 and restored['image_assets'] == before['image_assets']
        command(['models','load','qwen-vl','--context','256'])
        started = time.monotonic()
        followup = command(['sessions','send','scene','--input',question,'--expected-revision','4','--idempotency-key','followup','--max-tokens','8'])
        followup_elapsed = time.monotonic()-started
        assert followup['result']['vision']['images'] == 1
        assert followup['result']['text'] == 'Red'
        followup_receipt = record('cli-resumed-no-image', followup)
        print('PASS CLI saved-session restart and follow-up without resending the image', flush=True)
        # The reconstructed history must match a stateless request with identical original bytes.
        import base64
        url = 'data:image/png;base64,'+base64.b64encode(image).decode()
        messages = [dict(role='user',content=[dict(type='image_url',image_url=dict(url=url)),dict(type='text',text=question)]),
                    dict(role='assistant',content=first['result']['text']),dict(role='user',content=question)]
        parity = call('POST','/api/v1/generate',dict(model='qwen-vl',messages=messages,context=256,max_tokens=8,threads=2,profile='balanced'))
        assert parity['token_ids'] == followup['result']['token_ids']
        assert parity['prompt_tokens'] == followup['result']['prompt_tokens']
        assert parity['vision']['effective_positions'] == followup['result']['vision']['effective_positions']
        parity_receipt = record('stateless-history-parity', parity)
        replay = call('POST',f'/coheryn/v1/sessions/{session}/messages',dict(expected_revision=4,idempotency_key='followup',content=question,settings=dict(max_tokens=8)))
        assert replay['result']['token_ids'] == followup['result']['token_ids'] and replay['revision'] == followup['revision']
        exported = call('GET',f'/coheryn/v1/sessions/{session}/export')
        assert exported['version'] == 3 and len(exported['assets']) == 1
        assert base64.b64decode(exported['assets'][0]['base64']) == image
        exported_receipt = record('portable-image-document', exported)
        reset = call('POST',f'/coheryn/v1/sessions/{session}/reset',dict(expected_revision=followup['revision']))
        assert not reset.get('image_assets') and not any((data/'image-assets').iterdir())
        call('DELETE',f'/coheryn/v1/sessions/{session}',dict(expected_revision=reset['revision']))
        command(['models','unload','qwen-vl'])
        final = call('GET','/api/v1/server/status',admin=True)
        assert final['session_image_assets']['disk_bytes'] == 0
        assert final['memory']['session_buffer_capacity_bytes'] == 0
        assert final['scheduler']['vision_waiting_units'] == 0
        report = dict(schema_version=1,scope='Current-host exact-pair native CPU image histories, opt-in save/restart, CLI no-resend follow-up, stateless parity, original-byte export, cleanup and live text progress. Synthetic tests separately cover embeddings, mutation, ownership, quotas and failed imports. Not a performance or complete release qualification.',
            host=platform.platform(),binary_sha256=binary_sha,checker_sha256=digest(Path(__file__)),reference_sha256=digest(references),
            model_pair=pair,image_sha256=digest(image_path),first_reference_case='left-red',first_text=first['result']['text'],followup_text=followup['result']['text'],
            evidence=dict(first=first_receipt,followup=followup_receipt,stateless_parity=parity_receipt,portable_document=exported_receipt),
            shared_compute=dict(max_concurrency=1,text_completed_before_image=True,text_response=text,text_elapsed_seconds=text_elapsed,image_elapsed_seconds=first_elapsed,policy='FIFO vision weight tiles/32-query attention groups/decoder checkpoints; output delivery releases compute'),
            followup_elapsed_seconds=followup_elapsed,restored_before_loading_weights=True,idempotent_replay=True,cache_condition='uncontrolled',final_status=final,
            run_directory=str(run.relative_to(ROOT)),independence='Native executables run with an empty executable search path; no external inference runtime.')
        assert digest(binary) == binary_sha
        (ROOT/'fixtures/qualification/m4-image-sessions.json').write_text(json.dumps(report,indent=2)+'\n')
        print('PASS native persistent image conversation qualification', flush=True)
    finally:
        stop()

if __name__ == '__main__':
    main()

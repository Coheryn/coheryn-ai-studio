#!/usr/bin/env python3
"""Exercise the native product image CLI/API against frozen exact-pair continuations.

Requires the two pinned cached GGUF files and a release CLI. No downloads or foreign
execution. Qualification uses private hardlinks to immutable cached component bytes
rather than duplicating the 4.4 GB pair; ordinary bounded uploads are covered by tests.
"""
import base64
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
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def main():
    binary = ROOT/'target/release/coheryn'
    reference_path = ROOT/'fixtures/vlm/qwen2-image-conditioned-reference-v2.json'
    reference = json.loads(reference_path.read_text())
    cases = {c['id']: c for c in reference['cases']}
    run = ROOT/'.cache/m3-service'/uuid.uuid4().hex
    data = run/'data'
    for p in [run, data, data/'models', data/'records']:
        p.mkdir(parents=True, exist_ok=True, mode=0o700)
    components = [
        ('language', 'Qwen2-VL-2B-Instruct-f16.gguf', 3093667808, 'ec8b6dc5fd686f12b7d74850561f74b779db67424e477403fc043b98a0f40752'),
        ('vision', 'mmproj-Qwen2-VL-2B-Instruct-f16.gguf', 1331656160, 'ecb20cabcdd8dbc277de06bd6eb980aeb2adfaaba9f199a434e328d205675d03'),
    ]
    for name, filename, size, sha in components:
        source = ROOT/'.cache/vlm/full/qwen2-vl-2b'/filename
        assert source.is_file() and source.stat().st_size == size
        os.link(source, data/'models'/f'model-{name}.gguf')
        (data/'records'/f'model-{name}.json').write_text(json.dumps(dict(schema_version=1, id=name, sha256=sha, bytes=size, source=None)))
    env = dict(os.environ, PATH='')
    log = (run/'daemon.log').open('w')
    process = subprocess.Popen([str(binary), '--data-dir', str(data), 'serve', '--port', '0', '--max-memory', str(768<<20), '--max-context', '256', '--generation-timeout', '600', '--queue-capacity', '1'], env=env, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=log)
    evidence = []
    try:
        address = None
        for _ in range(300):
            if process.poll() is not None:
                raise RuntimeError('native daemon exited; inspect the private run log')
            lines = (run/'daemon.log').read_text().splitlines()
            for line in lines:
                try:
                    v = json.loads(line)
                except ValueError:
                    continue
                if v.get('event') == 'listening':
                    address = v['address']
            if address:
                break
            time.sleep(.1)
        assert address
        host, port = address.rsplit(':', 1)
        credentials = json.loads((data/'credentials.json').read_text())
        def call(path, body=None, admin=False, streaming=False):
            c = http.client.HTTPConnection(host, int(port), timeout=650)
            raw = None if body is None else json.dumps(body, ensure_ascii=False).encode()
            c.request('GET' if body is None else 'POST', path, body=raw, headers={'Authorization':'Bearer '+credentials['admin_token' if admin else 'inference_token'], 'Content-Type':'application/json'})
            r = c.getresponse()
            b = r.read(16<<20)
            assert r.status == 200, (r.status, b[:1024])
            c.close()
            return b.decode() if streaming else json.loads(b)
        def command(args):
            r = subprocess.run([str(binary),'--json','--endpoint',f'http://{address}','--data-dir',str(data), *args], env=env, cwd=ROOT, capture_output=True, timeout=650)
            if r.returncode:
                raise RuntimeError(r.stderr.decode()[:1024])
            return json.loads(r.stdout)
        pair = command(['models','register-pair','--name','qwen-vl','--language','language','--vision','vision'])
        assert pair['bytes'] == sum(c[2] for c in components)
        estimate = command(['plan','qwen-vl','--task','image-to-text','--context','256','--image',str(ROOT/'fixtures/vlm/preprocessing/canonical/landscape-png.png')])
        assert estimate['task'] == 'image-to-text' and call('/api/v1/server/status', admin=True)['loaded_models'] == []
        command(['models','load','qwen-vl','--context','256'])
        baseline = call('/api/v1/server/status', admin=True)['memory']['reserved']
        def messages(case):
            result = []
            for m in case['messages']:
                content = m['content']
                if isinstance(content, list):
                    parts = []
                    for p in content:
                        if p['type'] == 'image':
                            image = ROOT/'fixtures/vlm/preprocessing/canonical'/p['fixture']
                            parts.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(image.read_bytes()).decode()}})
                        else:
                            parts.append(p)
                    content = parts
                result.append({'role':m['role'],'content':content})
            return result
        for name, route, streaming, cli in [
            ('left-red','/api/v1/generate',False,True),
            ('left-blue','/v1/chat/completions',True,False),
            ('ordered-images','/api/v1/generate',False,False),
            ('unicode-multiturn','/v1/chat/completions',False,False),
        ]:
            case = cases[name]
            body = dict(model='qwen-vl', messages=messages(case), context=256, max_tokens=8, threads=2, profile='balanced', stream=streaming)
            started = time.monotonic()
            if cli:
                content = case['messages'][-1]['content']
                prompt = ''.join(p['text'] for p in content if p['type'] == 'text')
                args = ['chat','qwen-vl','--prompt',prompt,'--max-tokens','8','--context','256','--threads','2']
                for p in content:
                    if p['type'] == 'image':
                        args += ['--image', str(ROOT/'fixtures/vlm/preprocessing/canonical'/p['fixture'])]
                result = command(args)
            else:
                if streaming:
                    body['stream_options'] = dict(include_usage=True)
                result = call(route, body, streaming=streaming)
            elapsed = time.monotonic()-started
            path = run/f'{name}.json'
            if streaming:
                assert result.endswith('data: [DONE]\n\n')
                events = [json.loads(l[6:]) for l in result.splitlines() if l.startswith('data: ') and l != 'data: [DONE]']
                assert all('error' not in e for e in events)
                text = ''.join(e.get('choices',[{}])[0].get('delta',{}).get('content','') for e in events if e.get('choices'))
                usage = events[-1]['usage']
                assert usage['prompt_tokens'] == case['encoded']['physical_prompt_tokens']
                assert usage['completion_tokens'] == len(case['generation'])
                path.write_text(json.dumps(events, indent=2)+'\n')
            elif route == '/api/v1/generate':
                text = result['text']
                assert result['token_ids'] == [t['token_id'] for t in case['generation']]
                assert result['prompt_tokens'] == case['encoded']['physical_prompt_tokens']
                assert result['vision']['effective_positions'] == case['encoded']['effective_positions']
                path.write_text(json.dumps(result, indent=2)+'\n')
            else:
                text = result['choices'][0]['message']['content']
                assert result['usage']['prompt_tokens'] == case['encoded']['physical_prompt_tokens']
                assert result['usage']['completion_tokens'] == len(case['generation'])
                path.write_text(json.dumps(result, indent=2)+'\n')
            assert text == case['text'], (name,text,case['text'])
            status = call('/api/v1/server/status', admin=True)
            assert status['memory']['reserved'] == baseline, (name,status['memory'])
            evidence.append(dict(case=name, route='CLI chat -> '+route if cli else route, streaming=streaming, text=text,
                exact_token_ids=not streaming and route == '/api/v1/generate', prompt_tokens=case['encoded']['physical_prompt_tokens'], completion_tokens=len(case['generation']), elapsed_seconds=elapsed,
                response=str(path.relative_to(ROOT)), response_sha256=digest(path), memory_after=status['memory']))
            print('PASS', name, 'via', evidence[-1]['route'], flush=True)
        # Disconnect while native preprocessing/prefill is live; await worker acknowledgement.
        c = http.client.HTTPConnection(host, int(port), timeout=30)
        body = dict(model='qwen-vl',messages=messages(cases['ordered-images']),max_tokens=256,context=256,stream=True,threads=2)
        # Leave sufficient room for the bounded generation allowance.
        body['max_tokens'] = 128
        c.request('POST','/v1/chat/completions',json.dumps(body),{'Authorization':'Bearer '+credentials['inference_token'],'Content-Type':'application/json'})
        response = c.getresponse(); assert response.status == 200
        assert response.read(1)
        c.close()
        for _ in range(600):
            status = call('/api/v1/server/status', admin=True)
            if status['memory']['reserved'] == baseline and status['scheduler']['active'] == 0:
                break
            time.sleep(.1)
        assert status['memory']['reserved'] == baseline and status['scheduler']['cancelled'] >= 1
        command(['models','unload','qwen-vl'])
        report = dict(schema_version=1, scope='Native product image CLI/API on this host; exact pinned pair only, greedy F32 KV. Four frozen cases; structural/security/lifecycle coverage in workspace tests. No foreign execution, performance claim or full M2/release acceptance.',
            host=platform.platform(),binary_sha256=digest(binary), checker_sha256=digest(Path(__file__)),reference_sha256=digest(reference_path),
            model_pair=pair, estimate=estimate, executions=evidence, disconnect_cancelled=True, baseline_reserved=baseline,
            final_status=call('/api/v1/server/status',admin=True),cache_condition='uncontrolled',run_directory=str(run.relative_to(ROOT)))
        (ROOT/'fixtures/qualification/m3-image-service.json').write_text(json.dumps(report,indent=2)+'\n')
        print('PASS product image CLI/API qualification', flush=True)
    finally:
        process.terminate()
        try:
            process.wait(timeout=40)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
        log.close()

if __name__ == '__main__':
    main()

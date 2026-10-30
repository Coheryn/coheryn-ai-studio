"""Real loopback TCP checks using the extracted candidate and private, disposable credentials."""
import base64
import http.client
import json
import os
import shutil
import subprocess
import tempfile
import time

from .common import ROOT, clean_environment, compare, read, require, sha, write
from .suites import generated


def frames(raw):
    lines = [line[6:] for line in raw.decode('utf-8').splitlines() if line.startswith('data: ')]
    require(lines and lines[-1] == '[DONE]' and lines.count('[DONE]') == 1, 'SSE termination mismatch')
    result = [json.loads(line) for line in lines[:-1]]
    require(not any('error' in frame for frame in result), 'SSE error frame')
    return result


def image_content(text, image):
    """Preserve the image-first ordering of the frozen still-image prompt."""
    return [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(image.read_bytes()).decode('ascii')}},
            {'type': 'text', 'text': text}]


def run(context, suite, model):
    records = []
    # Keep credentials and weight copies outside the directory uploaded as evidence,
    # including when a platform refuses deletion of an unexpectedly open file.
    with tempfile.TemporaryDirectory(prefix='cpu-private-vault-', dir=ROOT / '.cache') as temporary:
        vault = ROOT / temporary / 'vault'
        log_path = context.directory / (suite + '-daemon.jsonl')
        require(shutil.disk_usage(context.directory).free >
                (sum(p.stat().st_size for p in model.iterdir() if p.is_file()) if model.is_dir() else model.stat().st_size) + (512 << 20),
                'insufficient space for a private managed model copy')
        with log_path.open('xb') as log:
            process = subprocess.Popen([str(context.binary), '--data-dir', str(vault), 'serve', '--port', '0',
                                        '--max-memory', str(1 << 30), '--max-context', '512',
                                        '--generation-timeout', '1200', '--idle-cache-bytes', '0', '--session-cache-bytes', '0'],
                                       env=clean_environment(), stdout=log, stderr=log)
            try:
                endpoint = None
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    require(process.poll() is None, 'daemon exited before readiness')
                    require(log_path.stat().st_size < 1 << 20, 'daemon readiness log exceeded limit')
                    for line in log_path.read_text(encoding='utf-8', errors='replace').splitlines():
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if item.get('event') == 'listening':
                            endpoint = item['address']
                    if endpoint:
                        break
                    time.sleep(.05)
                require(endpoint and endpoint.startswith('127.0.0.1:'), 'loopback daemon readiness timeout')
                credentials = read(vault / 'credentials.json')

                def request(label, route, body, stream=False):
                    connection = http.client.HTTPConnection(endpoint, timeout=1200)
                    admin = route.startswith('/api/') and route != '/api/v1/generate'
                    try:
                        connection.request('POST', route, json.dumps(body, ensure_ascii=False).encode('utf-8'),
                                           {'Content-Type': 'application/json', 'Authorization': 'Bearer ' +
                                            credentials['admin_token' if admin else 'inference_token']})
                        response = connection.getresponse()
                        raw = response.read((16 << 20) + 1)
                        require(response.status == 200 and len(raw) <= 16 << 20,
                                f'HTTP status/size mismatch: {label}, status={response.status}, body={raw[:512]!r}')
                        if stream:
                            require(response.getheader('Content-Type', '').startswith('text/event-stream'), 'SSE content type mismatch')
                        path = context.directory / (suite + '-' + label + '.response')
                        with path.open('xb') as output:
                            output.write(raw)
                        records.append({'case': label, 'route': route, 'status': response.status, 'response_sha256': sha(path)})
                        return frames(raw) if stream else json.loads(raw)
                    finally:
                        connection.close()

                imported = context.cli(suite + '-import', ['models', 'import', str(model), '--name', 'qualified'], endpoint, vault)
                require(imported, 'missing managed import receipt')
                embedding = suite.startswith('embedding-')
                request('load', '/api/v1/models/load', {'model': 'qualified', 'context': 512 if suite == 'gguf-text' or embedding else 256})
                if embedding:
                    entry = next(e for e in read(ROOT / 'fixtures/embedding/models.json')['models'] if e['file'] == model.name)
                    ref = read(ROOT / entry['evidence'])
                    limit = read(ROOT / 'fixtures/embedding/reference-contract.json')['tolerances']
                    output = request('embedding-batch', '/v1/embeddings', {'model': 'qualified', 'input': [c['input'] for c in ref['cases']],
                                                                         'profile': 'reference', 'threads': 1, 'batch_size': 4})
                    require(output['object'] == 'list' and len(output['data']) == len(ref['cases']) and
                            output['usage']['prompt_tokens'] == sum(len(c['token_ids']) for c in ref['cases']), 'embedding API shape/usage mismatch')
                    for index, (actual, expected) in enumerate(zip(output['data'], ref['cases'])):
                        require(actual['index'] == index and actual['object'] == 'embedding', 'embedding API ordering mismatch')
                        compare(actual['embedding'], expected['normalized'], limit['absolute'], limit['relative'], limit['normalized_l2'])
                else:
                    if suite == 'gguf-text':
                        case = read(ROOT / 'fixtures/golden/qwen2.5-0.5b-instruct-fp16.json')['cases'][0]
                        body = {'prompt': case['prompt']}
                    else:
                        case = read(ROOT / ('fixtures/qwen35/vision/reference-v1.json' if suite == 'qwen35-image' else 'fixtures/qwen35/reference-v1.json'))['cases'][0]
                        body = {'messages': case.get('messages')}
                        if suite == 'qwen35-image':
                            image = ROOT / 'fixtures/qwen35/vision/red.png'
                            body = {'messages': [{'role': 'user', 'content': image_content(case['text'], image)}]}
                    expected_text = case.get('greedy_text', case.get('text'))
                    body.update(model='qualified', max_tokens=len(case.get('greedy_token_ids', case.get('generated_token_ids'))),
                                context=512 if suite == 'gguf-text' else 256, threads=1, profile='reference')
                    route = '/v1/completions' if suite == 'gguf-text' else '/v1/chat/completions'
                    output = request('completion', route, body)
                    text = output['choices'][0]['text'] if suite == 'gguf-text' else output['choices'][0]['message']['content']
                    expected_usage = {'prompt_tokens': len(case['prompt_token_ids']), 'completion_tokens': body['max_tokens'],
                                      'total_tokens': len(case['prompt_token_ids']) + body['max_tokens']}
                    require(text == expected_text and all(output['usage'].get(k) == v for k, v in expected_usage.items()),
                            'API continuation/usage mismatch')
                    chunks = request('sse', route, {**body, 'stream': True, 'stream_options': {'include_usage': True}}, True)
                    streamed = ''.join(choice.get('text', '') if suite == 'gguf-text' else choice['delta'].get('content', '')
                                       for chunk in chunks for choice in chunk['choices'])
                    require(streamed == expected_text and any(c.get('usage') == output['usage'] for c in chunks), 'SSE text/usage mismatch')
                    finishes = [choice['finish_reason'] for chunk in chunks for choice in chunk['choices'] if choice.get('finish_reason')]
                    require(finishes == [output['choices'][0]['finish_reason']], 'SSE finish reason mismatch')
                    if suite != 'qwen35-image':
                        native = request('native', '/api/v1/generate', body)
                        generated(native, case)
                        arguments = ['run', 'qualified', '--prompt', case['prompt'], '--context', str(body['context']),
                                     '--max-tokens', str(body['max_tokens']), '--profile', 'reference', '--threads', '1']
                        generated(context.cli(suite + '-client', arguments, endpoint, vault), case)
                request('unload', '/api/v1/models/unload', {'model': 'qualified'})
            finally:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)
                records.append({'shutdown': 'TerminateProcess' if os.name == 'nt' else 'SIGTERM', 'exit': process.returncode})
        # TemporaryDirectory removes model copies and tokens even after failed assertions.
    return {'transport': 'loopback TCP, authenticated, extracted candidate', 'cases': records,
            'durable_restart_tested': False, 'tls_tested': False}

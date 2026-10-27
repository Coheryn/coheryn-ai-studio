#!/usr/bin/env python3
"""Unix pinned-model smoke: actual daemon, managed model, CLI and OpenAI/SSE parity."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from fetch_model import digest

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT/'target/release/coheryn'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'fixtures/models.json')
    parser.add_argument('--id')
    parser.add_argument('--golden',type=Path,default=ROOT/'fixtures/golden/qwen2.5-0.5b-instruct-fp16.json')
    parser.add_argument('--evidence-out',type=Path,default=ROOT/'fixtures/qualification/g2-service.json')
    args=parser.parse_args()
    models=json.loads(args.manifest.read_text())['models']
    pin=next((m for m in models if m['id']==args.id),None) if args.id else models[0]
    if pin is None:parser.error('unknown pinned model ID')
    model = ROOT/'.cache/models'/pin['file']
    golden = json.loads(args.golden.read_text())
    english = next(c for c in golden['cases'] if c['id'] == 'english')
    explicit = next(c for c in golden['cases'] if c['id'] == 'explicit-system')
    report = {'schema_version':1,'model_sha256':pin['sha256'],'native_executable_sha256':digest(BINARY),'transport':'actual loopback HTTP/TCP','checks':[]}
    log_path = ROOT/'.cache/service-smoke.log'
    with tempfile.TemporaryDirectory(prefix='coheryn-g2-',dir=ROOT/'.cache') as temporary:
        data = Path(temporary)/'data'
        with log_path.open('w') as log:
            server = subprocess.Popen([str(BINARY),'--data-dir',str(data),'serve','--port','0','--max-context','512'],stdout=log,stderr=log)
            try:
                endpoint = None
                for _ in range(200):
                    if server.poll() is not None:
                        raise RuntimeError('Daemon exited before listening; see .cache/service-smoke.log')
                    for line in log_path.read_text().splitlines():
                        try:
                            record = json.loads(line)
                        except ValueError:
                            continue
                        if record.get('event') == 'listening':
                            endpoint = 'http://'+record['address']
                    if endpoint:
                        break
                    time.sleep(0.05)
                if endpoint is None:
                    raise RuntimeError('Daemon startup timed out')
                credentials = json.loads((data/'credentials.json').read_text())
                base = [str(BINARY),'--data-dir',str(data),'--endpoint',endpoint]

                def cli(*args, json_result=True, expected=0):
                    command = base+(['--json'] if json_result else [])+list(args)
                    result = subprocess.run(command,capture_output=True,text=True,timeout=240)
                    if result.returncode != expected:
                        raise AssertionError(f'CLI {args[:2]} exited {result.returncode}: {result.stderr}')
                    if expected:
                        return json.loads(result.stderr)
                    return json.loads(result.stdout) if json_result else result.stdout

                def http(path,body=None,admin=False,expected=200):
                    headers = {'Authorization':'Bearer '+credentials['admin_token' if admin else 'inference_token']}
                    if body is not None:
                        headers['Content-Type'] = 'application/json'
                    request = urllib.request.Request(endpoint+path,data=None if body is None else json.dumps(body).encode(),headers=headers)
                    try:
                        response = urllib.request.urlopen(request,timeout=240)
                    except urllib.error.HTTPError as error:
                        response = error
                    if response.status != expected:
                        raise AssertionError(f'{path}: {response.status} != {expected}: {response.read(8192)}')
                    return response

                imported = cli('models','import',str(model),'--name','qwen-smoke','--sha256',pin['sha256'])
                assert imported['sha256'] == pin['sha256']
                cli('models','load','qwen-smoke','--context','512')
                print('PASS managed import/load and pinned SHA-256',flush=True)
                native = cli('run','qwen-smoke','--prompt','Hello','--context','512','--max-tokens','8')
                assert native['token_ids'] == english['generated_token_ids']
                assert native['text'] == english['text']
                completion = json.load(http('/v1/completions',{'model':'qwen-smoke','prompt':'Hello','max_tokens':8}))
                assert completion['choices'][0]['text'] == native['text']
                stream = cli('run','qwen-smoke','--prompt','Hello','--context','512','--max-tokens','8','--stream',json_result=False)
                assert stream == native['text']+'\n'
                report['checks'].append({'case':'native-cli-openai-completion-stream-parity','token_ids':native['token_ids'],'text':native['text']})
                print('PASS daemon CLI, CLI SSE, OpenAI completion and independent golden parity',flush=True)

                chunks = []
                with http('/v1/chat/completions',{'model':'qwen-smoke','messages':[{'role':'system','content':'Be brief.'},{'role':'user','content':'Hi'}],
                          'max_tokens':8,'stream':True,'stream_options':{'include_usage':True}}) as response:
                    done = False
                    for line in response:
                        line = line.decode().strip()
                        if not line.startswith('data: '):
                            continue
                        if line == 'data: [DONE]':
                            done = True
                            break
                        chunks.append(json.loads(line[6:]))
                assert done
                text = ''.join(choice.get('delta',{}).get('content','') for chunk in chunks for choice in chunk['choices'])
                assert text == explicit['text']
                assert chunks[-1]['choices'] == [] and chunks[-1]['usage']['completion_tokens'] == 8
                assert all(chunk.get('usage') is None for chunk in chunks[:-1])
                report['checks'].append({'case':'chat-sse-golden-parity','text':text,'usage':chunks[-1]['usage']})
                print('PASS OpenAI chat SSE, usage chunk and [DONE]',flush=True)

                response = http('/v1/completions',{'model':'qwen-smoke','prompt':'Hello','max_tokens':128,'stream':True})
                for line in response:
                    if line.startswith(b'data: '):
                        break
                conflict = json.load(http('/api/v1/models/unload',{'model':'qwen-smoke'},admin=True,expected=409))
                assert conflict['error']['code'] == 'CONFLICT'
                response.close()
                for _ in range(200):
                    status = json.load(http('/api/v1/server/status',admin=True))
                    if status['scheduler']['active'] == 0 and status['scheduler']['queued'] == 0:
                        break
                    time.sleep(0.05)
                else:
                    raise AssertionError('Disconnect did not cancel generation')
                assert status['scheduler']['cancelled'] >= 1
                error = cli('run','qwen-smoke','--prompt','Hello','--context','512','--max-tokens','1','--max-memory','1',expected=5)
                assert error['code'] == 'RESOURCE_EXHAUSTED'
                denied = json.load(http('/api/v1/server/status',admin=False,expected=403))
                assert denied['error']['code'] == 'UNAUTHORIZED'
                report['checks'].append({'case':'pinning-disconnect-auth-memory-admission','scheduler':status['scheduler'],'unload_while_active':409,'inference_cannot_admin':403,'client_memory_limit_exit':5})
                cli('models','unload','qwen-smoke')
                cli('models','remove','qwen-smoke')
                assert cli('status')['memory']['reserved'] == 0
                print('PASS active-model pinning, disconnect cancellation, role checks and memory cleanup',flush=True)
            finally:
                if server.poll() is None:
                    server.send_signal(signal.SIGINT)
                    try:
                        server.wait(timeout=45)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait()
                        raise AssertionError('Graceful daemon shutdown timed out')
            assert server.returncode == 0,server.returncode
            result = subprocess.run(base+['--json','run','qwen-smoke','--prompt','Hello','--max-tokens','1','--context','512'],capture_output=True,text=True,timeout=10)
            assert result.returncode == 8,(result.returncode,result.stderr)
            report['checks'].append({'case':'graceful-shutdown-and-no-standalone-fallback','server_exit':server.returncode,'client_exit_without_daemon':result.returncode})
    target = args.evidence_out
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print('PASS graceful shutdown and explicit client failure without daemon',flush=True)

if __name__ == '__main__':
    main()

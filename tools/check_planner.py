#!/usr/bin/env python3
"""Pinned-model G3 smoke: profiles, context/concurrency plans, measured buffers and cleanup.
Uses only local CLI/loopback HTTP and the existing ignored FP16 cache; never downloads weights.
"""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'fixtures/models.json')
    parser.add_argument('--id')
    parser.add_argument('--golden', type=Path, default=ROOT/'fixtures/golden/qwen2.5-0.5b-instruct-fp16.json')
    parser.add_argument('--evidence-out', type=Path, default=ROOT/'fixtures/qualification/g3-planner.json')
    args = parser.parse_args()
    models = json.loads(args.manifest.read_text())['models']
    pin = next((m for m in models if m['id']==args.id), None) if args.id else models[0]
    if pin is None:
        parser.error('unknown pinned model ID')
    golden = next(c for c in json.loads(args.golden.read_text())['cases'] if c['id']=='english')
    report = {'schema_version':1,'model_sha256':pin['sha256'], 'transport':'actual loopback HTTP/TCP',
              'native_executable_sha256':digest(BINARY),
              'cache_condition':'uncontrolled','context_note':'Reservation capacity checks and short-prompt inference; no long-context quality claim',
              'plans':[], 'runs':[], 'checks':[]}
    log_path = ROOT/'.cache/planner-smoke.log'
    with tempfile.TemporaryDirectory(prefix='coheryn-g3-',dir=ROOT/'.cache') as temporary, log_path.open('w') as log:
        data = Path(temporary)/'data'
        server = subprocess.Popen([str(BINARY),'--data-dir',str(data),'serve','--port','0','--max-context','4096','--max-concurrency','2'],stdout=log,stderr=log)
        try:
            endpoint = None
            for _ in range(200):
                if server.poll() is not None:
                    raise RuntimeError('daemon exited; see .cache/planner-smoke.log')
                for line in log_path.read_text().splitlines():
                    try:
                        record=json.loads(line)
                    except ValueError:
                        continue
                    if record.get('event')=='listening':
                        endpoint='http://'+record['address']
                if endpoint:
                    break
                time.sleep(0.05)
            if not endpoint:
                raise RuntimeError('daemon startup timed out')
            credentials=json.loads((data/'credentials.json').read_text())
            base=[str(BINARY),'--data-dir',str(data),'--endpoint',endpoint,'--json']

            def cli(*args):
                result=subprocess.run(base+list(args),capture_output=True,text=True,timeout=240)
                if result.returncode:
                    raise RuntimeError(result.stderr)
                return json.loads(result.stdout)

            def http(path, body=None, admin=True):
                headers={'Authorization':'Bearer '+credentials['admin_token' if admin else 'inference_token']}
                if body is not None:
                    headers['Content-Type']='application/json'
                request=urllib.request.Request(endpoint+path,headers=headers,data=None if body is None else json.dumps(body).encode())
                return urllib.request.urlopen(request,timeout=240)

            cli('models','import',str(ROOT/'.cache/models'/pin['file']),'--name','qwen-g3','--sha256',pin['sha256'])
            for context in [128,512,4096]:
                for profile in ['reference','performance','balanced','low-memory']:
                    for concurrency in [1,2]:
                        plan=cli('plan','qwen-g3','--context',str(context),'--profile',profile,'--concurrency',str(concurrency))
                        assert plan['context']==context and plan['concurrency']==concurrency
                        assert plan['cache_dtype']=='f32' and plan['backend']=='cpu-reference'
                        assert plan['model_metadata_additional']>0
                        if profile=='low-memory' and concurrency==2:
                            assert not plan['admissible']
                        else:
                            assert plan['admissible'], plan['reasons']
                        report['plans'].append(plan)
            assert json.load(http('/api/v1/server/status'))['loaded_models']==[]
            print('PASS 24 metadata-only context/profile/concurrency plans without loading weights',flush=True)
            cli('models','load','qwen-g3','--context','4096')
            for profile in ['reference','performance','balanced','low-memory']:
                before=json.load(http('/api/v1/server/status'))['memory']['reserved']
                result=cli('run','qwen-g3','--prompt','Hello','--context','512','--max-tokens','8','--profile',profile)
                assert result['token_ids']==golden['generated_token_ids'] and result['text']==golden['text']
                assert result['memory']['session_buffer_capacity_bytes']==0
                assert result['memory']['peak_session_buffer_capacity_bytes']>0
                deadline=time.monotonic()+5
                while True:
                    after=json.load(http('/api/v1/server/status'))
                    if after['memory']['reserved']==before and after['scheduler']['active']==0:
                        break
                    if time.monotonic()>deadline:
                        raise RuntimeError('session reservation failed to return to baseline')
                    time.sleep(0.02)
                report['runs'].append({'profile':profile,'result':result,'memory_after':after['memory']})
                print(f'PASS {profile}: exact oracle greedy tokens, observed buffers and reservation cleanup',flush=True)
            streams=[]
            try:
                for _ in range(2):
                    streams.append(http('/v1/chat/completions',{'model':'qwen-g3','messages':[{'role':'user','content':'Hi'}],
                        'max_tokens':128,'context':512,'profile':'performance','stream':True},admin=False))
                deadline=time.monotonic()+10
                while True:
                    status=json.load(http('/api/v1/server/status'))
                    if status['scheduler']['active']==2 and status['memory']['session_buffer_capacity_bytes']>0:
                        break
                    if time.monotonic()>deadline:
                        raise RuntimeError('two requests never became active')
                    time.sleep(0.05)
                report['concurrent_observation']=status
                assert status['memory']['reserved']<=status['memory']['limit']
                assert status['memory']['session_buffer_capacity_bytes']<=status['memory']['reserved']
            finally:
                for response in streams:
                    response.close()
            deadline=time.monotonic()+15
            while True:
                status=json.load(http('/api/v1/server/status'))
                if status['scheduler']['active']==0 and status['scheduler']['queued']==0:
                    break
                if time.monotonic()>deadline:
                    raise RuntimeError('concurrent disconnect cleanup timed out')
                time.sleep(0.05)
            assert status['memory']['session_buffer_capacity_bytes']==0
            assert status['scheduler']['cancelled']>=2
            report['checks'].append('two-live-pinned-model-sessions-disconnect-within-budget')
            report['hardware']=json.load(http('/api/v1/system/resources'))['hardware']
            cli('models','unload','qwen-g3')
            assert json.load(http('/api/v1/server/status'))['memory']['reserved']==0
            cli('models','remove','qwen-g3')
            report['checks'].append('unload-remove-returns-reservations-to-zero')
            print('PASS concurrent pinned-model admission, disconnect cleanup and zero-reservation unload',flush=True)
            server.send_signal(signal.SIGINT)
            assert server.wait(timeout=40)==0
            args.evidence_out.write_text(json.dumps(report,indent=2)+'\n')
        finally:
            if server.poll() is None:
                server.send_signal(signal.SIGINT)
                try:
                    server.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    server.kill();server.wait()


if __name__ == '__main__':
    main()

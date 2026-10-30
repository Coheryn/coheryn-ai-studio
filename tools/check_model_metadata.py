#!/usr/bin/env python3
"""Offline native descriptor/plan validation using pinned inputs; no external executor."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=ROOT/'fixtures/qualification/a1-metadata-runtime.json')
    parser.add_argument('--admission',action='store_true',help='Also verify profiled runtime headroom and rejection parity.')
    args=parser.parse_args()
    model=json.loads((ROOT/'fixtures/qwen35/model.json').read_text())
    checkpoint=ROOT/model['cache_directory']
    for item in model['files']:
        path=checkpoint/item['file']
        assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'], item['file']
    binary=ROOT/'target/release/coheryn'
    directory=ROOT/'.cache/a1-metadata';directory.mkdir(parents=True,exist_ok=True)
    empty=directory/'empty-path';empty.mkdir(exist_ok=True)
    environment={**os.environ,'PATH':str(empty)}
    environment.pop('COHERYN_API_TOKEN',None);environment.pop('COHERYN_ADMIN_TOKEN',None)
    results=[]
    def run(label,args,expected=0,cwd=ROOT):
        command=[str(binary),'--data-dir',str(directory/'unused-vault'),'--endpoint','http://127.0.0.1:1',*args,'--json']
        process=subprocess.run(command,cwd=cwd,env=environment,capture_output=True,text=True,timeout=180)
        assert process.returncode==expected,(label,process.returncode,process.stderr)
        (directory/(label+'.stdout.json')).write_text(process.stdout)
        (directory/(label+'.stderr.txt')).write_text(process.stderr)
        value=json.loads(process.stdout) if process.stdout.strip() else None
        results.append({'case':label,'exit':process.returncode,'stdout_sha256':sha(directory/(label+'.stdout.json'))})
        return value
    descriptor=run('inspect',['models','inspect',str(checkpoint)])
    assert descriptor['schema_version']==1 and descriptor['format']=='safetensors' and descriptor['layout']=='indexed'
    assert descriptor['tensor_count']==488 and descriptor['architecture']=='qwen3_5'
    assert descriptor['declared_context']==262144 and descriptor['runtime_context_limit']==8192
    assert descriptor['weight_payloads']=='not-scanned' and descriptor['qualification_status']=='unqualified-payload-identities-not-verified'
    assert descriptor['executable'] and descriptor['text_adapter']['managed_api'] and not descriptor['text_adapter']['images']
    pinned={f['file']:f['sha256'] for f in model['files']}
    for item in descriptor['source_files']:
        if item['hash_scope']=='complete-file':assert item['metadata_sha256']==pinned[item['name']]
    indexed=run('explicit-index',['models','inspect',str(checkpoint/'model.safetensors.index.json')])
    assert indexed['metadata_identity']==descriptor['metadata_identity']
    single=run('relative-single',['models','inspect','model.safetensors-00001-of-00001.safetensors'],cwd=checkpoint)
    assert single['layout']=='single-file' and single['metadata_identity']!=descriptor['metadata_identity']
    basic=run('validate',['models','validate',str(checkpoint)])
    assert basic['executable'] and not basic['payload_scan_completed']
    deep=run('deep',['models','validate',str(checkpoint),'--deep'])
    assert deep['executable'] and deep['payload_scan_completed']
    plans=[]
    for context in (128,256,512,4096):
        estimate=run(f'plan-{context}',['plan',str(checkpoint),'--standalone','--context',str(context),'--profile','reference','--threads','1','--max-memory',str(768<<20)])
        assert estimate['model_descriptor']['metadata_identity']==descriptor['metadata_identity']
        plan=estimate['plan'];assert plan['admissible'] and plan['context']==context
        assert plan['session']['kv_capacity_bytes']==6*context*2*256*2*4
        assert plan['session']['workspace_bytes']>18*16*128*128*4
        plans.append(plan)
    assert len({p['session']['workspace_bytes'] for p in plans})==1
    rejected=run('rejected-memory',['plan',str(checkpoint),'--standalone','--context','256','--profile','reference','--max-memory','1'],5)
    assert not rejected['plan']['admissible']
    run('rejected-context',['plan',str(checkpoint),'--standalone','--context','8193'],2)
    run('unsupported-constraints',['plan',str(checkpoint),'--standalone','--constraints'],4)
    reference=json.loads((ROOT/'fixtures/qwen35/reference-v1.json').read_text())
    generated=run('qwen35-generation',['run',str(checkpoint),'--standalone','--messages',str(ROOT/'fixtures/qwen35/smoke-messages.json'),'--context','256','--max-tokens','4','--profile','reference','--threads','1','--max-memory',str(768<<20)])
    assert generated['token_ids']==reference['cases'][0]['greedy_token_ids']
    assert generated['text']==reference['cases'][0]['greedy_text']
    assert generated['memory']['peak_reserved']==plans[1]['required_reserved_bytes']
    assert generated['memory']['reserved']==plans[1]['model_metadata_additional']
    legacy=ROOT/'.cache/models/qwen2.5-0.5b-instruct-fp16.gguf'
    golden=json.loads((ROOT/'fixtures/golden/qwen2.5-0.5b-instruct-fp16.json').read_text())
    assert sha(legacy)==golden['model_sha256']
    old=run('gguf-inspect',['models','inspect',str(legacy)])
    assert old['version']==3 and old['model_descriptor']['format']=='gguf'
    run('gguf-plan',['plan',str(legacy),'--standalone','--context','128','--profile','reference'])
    old_result=run('gguf-generation',['run',str(legacy),'--standalone','--prompt','Hello','--context','128','--max-tokens','1','--profile','reference'])
    assert old_result['token_ids']==golden['cases'][0]['generated_token_ids'][:1]
    admission=None
    if args.admission:
        balanced=run('balanced-plan',['plan',str(checkpoint),'--standalone','--context','256','--profile','balanced','--threads','2','--max-memory',str(768<<20)])['plan']
        assert balanced['admissible'] and balanced['safety_headroom_bytes']>0
        executed=run('balanced-generation',['run',str(checkpoint),'--standalone','--messages',str(ROOT/'fixtures/qwen35/smoke-messages.json'),'--context','256','--max-tokens','4','--profile','balanced','--threads','2','--max-memory',str(balanced['required_reserved_bytes'])])
        assert executed['token_ids']==reference['cases'][0]['greedy_token_ids']
        assert executed['memory']['peak_reserved']==balanced['required_reserved_bytes']
        assert executed['memory']['reserved']==balanced['model_metadata_additional']
        without_headroom=balanced['required_reserved_bytes']-balanced['safety_headroom_bytes']
        run('balanced-reject-without-headroom',['run',str(checkpoint),'--standalone','--messages',str(ROOT/'fixtures/qwen35/smoke-messages.json'),'--context','256','--max-tokens','4','--profile','balanced','--threads','2','--max-memory',str(without_headroom)],5)
        admission={'balanced_plan':balanced,'balanced_generation':executed,'without_headroom_rejected':True,'failure_cleanup':'Separately acquired-model native Rust opt-in covers consumer failure, in-compute cancellation, budget/buffer reclamation and model drop.'}
    report={'schema_version':1,'scope':'Local macOS ARM64 metadata/CLI planning and native smoke; not managed/API, broader format/backend/platform or publisher-framework qualification. No admission reservations held by inspection/planning.','binary_sha256':sha(binary),'model_manifest_sha256':sha(ROOT/'fixtures/qwen35/model.json'),'descriptor':descriptor,'plans':plans,'qwen35_generation':generated,'legacy_generation':old_result,'cases':results,'environment':'Empty executable search path and no credentials, unreachable endpoint; all commands explicitly local','runtime_admission':admission,'cache_condition':'uncontrolled','os_rss':None,'observed_session_buffers':'reported by generation separately from conservative reservations'}
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('PASS local shared metadata, explicit deep scan, context/resource limits, exact runtime reservation prediction and native regression.')
if __name__=='__main__':main()

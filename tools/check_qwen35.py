#!/usr/bin/env python3
"""Run native Qwen3.5 against frozen independent data, without an external executor."""
import argparse
import array
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def vector(path):
    result=array.array('f');result.frombytes(path.read_bytes())
    if sys.byteorder!='little':result.byteswap()
    return result

def compare(path, expected, tolerance):
    import math
    actual=vector(path)
    assert len(actual)==expected['length'] and all(math.isfinite(x) for x in actual)
    maximum=0.
    for i,value in zip(expected['indices'],expected['values']):
        difference=abs(actual[i]-value);maximum=max(maximum,difference)
        assert difference<=tolerance['logits_absolute']+abs(value)*tolerance['logits_relative'], (path,i,actual[i],value)
    reference=ROOT/'.cache/q35-reference'/(expected['name']+'.f32')
    full=None
    if not reference.exists():
        raise FileNotFoundError('Prepare the independently captured reference arrays before full numerical qualification: '+str(reference))
    if reference.exists():
        assert sha(reference)==expected['sha256']
        values=vector(reference)
        norm=math.sqrt(sum(float(v)*v for v in values))
        full=math.sqrt(sum((float(a)-b)**2 for a,b in zip(actual,values)))/max(norm,1e-30)
        assert full<=tolerance['normalized_l2'], (path,full)
        assert all(abs(a-b)<=tolerance['logits_absolute']+abs(b)*tolerance['logits_relative'] for a,b in zip(actual,values)), path
    return {'sample_max_absolute':maximum,'full_array_normalized_l2':full,'full_reference_available':full is not None,'actual_sha256':sha(path)}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    if args.report.exists():raise FileExistsError(args.report)
    model=json.loads((ROOT/'fixtures/qwen35/model.json').read_text())
    ref=json.loads((ROOT/'fixtures/qwen35/reference-v1.json').read_text())
    assert ref['model_revision']==model['revision'] and ref['tolerances']==model['oracle']['tolerances']
    weights=ROOT/model['cache_directory']
    for entry in model['files']:
        assert (weights/entry['file']).stat().st_size==entry['bytes'] and sha(weights/entry['file'])==entry['sha256']
    binary=ROOT/'target/release/examples/qwen35_probe'
    empty=ROOT/'.cache/q35-native/empty-path';empty.mkdir(parents=True,exist_ok=True)
    runs=[]
    for case in ref['cases']:
        for threads in (1,2):
            directory=ROOT/'.cache/q35-native'/f"{case['id']}-workers{threads}";directory.mkdir(parents=True,exist_ok=True)
            request={'messages':case['messages'],'teacher_tokens':case['greedy_token_ids'][:-1],'threads':threads,'max_tokens':len(case['greedy_token_ids']),'trace_directory':str(directory),'lifecycle':threads==1}
            request_path=directory/'request.json';request_path.write_text(json.dumps(request,ensure_ascii=False))
            print('START',case['id'],threads,flush=True)
            selected=weights if threads==1 else weights/'model.safetensors-00001-of-00001.safetensors'
            process=subprocess.run([str(binary),str(selected),str(request_path)],capture_output=True,text=True,env={**os.environ,'PATH':str(empty)},timeout=900)
            (directory/'stderr.txt').write_text(process.stderr)
            assert process.returncode==0, process.stderr
            actual=json.loads(process.stdout);(directory/'result.json').write_text(process.stdout)
            assert actual['prompt']==case['prompt'] and actual['prompt_token_ids']==case['prompt_token_ids']
            assert actual['generation']['token_ids']==case['greedy_token_ids'], (actual['generation']['token_ids'],case['greedy_token_ids'])
            assert actual['generation']['text']==case['greedy_text']==''.join(e['text'] for e in actual['events'])
            assert actual['after']['reserved']==actual['resident']['reserved']
            arrays=[]
            last=len(case['prompt_token_ids'])-1
            for expected in case['checkpoints']:
                name=expected['name'].split('-prompt-')[1]
                arrays.append(compare(directory/f'{last}-{name}.f32',expected,ref['tolerances']))
            for i,expected in enumerate(case['teacher_forced']):
                arrays.append(compare(directory/f'{last+i}-logits.f32',expected,ref['tolerances']))
            runs.append({'case':case['id'],'threads':threads,'tokens':actual['generation']['token_ids'],'text':actual['generation']['text'],'memory':actual['generation']['memory'],'arrays':arrays,'result_sha256':sha(directory/'result.json'),'layout':'indexed' if threads==1 else 'single-file','lifecycle':actual['lifecycle']})
            print('PASS',case['id'],threads,repr(actual['generation']['text']),flush=True)
    cli=ROOT/'target/release/coheryn'
    first=ref['cases'][0]
    messages=ROOT/'fixtures/qwen35/smoke-messages.json'
    options=['--standalone','--context','256','--max-tokens','4','--threads','1','--profile','reference']
    environment={**os.environ,'PATH':str(empty)}
    process=subprocess.run([str(cli),'run','model.safetensors-00001-of-00001.safetensors','--messages',str(messages),*options,'--json'],cwd=weights,capture_output=True,text=True,env=environment,timeout=900)
    assert process.returncode==0,process.stderr
    cli_result=json.loads(process.stdout)
    assert cli_result['token_ids']==first['greedy_token_ids'] and cli_result['text']==first['greedy_text']
    process=subprocess.run([str(cli),'chat',str(weights),*options],input='Say hello.\n/exit\n',capture_output=True,text=True,env=environment,timeout=900)
    assert process.returncode==0 and first['greedy_text'] in process.stdout,process.stderr
    interactive=process.stdout
    process=subprocess.run([str(cli),'run',str(weights),'--prompt','Hello',*options,'--temperature','0.5','--json'],capture_output=True,text=True,env=environment,timeout=120)
    assert process.returncode==4 and 'UNSUPPORTED_FEATURE' in process.stderr,process.stderr
    unsupported=process.returncode
    process=subprocess.run([str(cli),'run',str(weights),'--prompt','Hello',*options,'--max-memory',str(64<<20),'--json'],capture_output=True,text=True,env=environment,timeout=120)
    assert process.returncode==5 and 'RESOURCE_EXHAUSTED' in process.stderr,process.stderr
    resource_exit=process.returncode
    legacy=ROOT/'.cache/models/qwen2.5-0.5b-instruct-fp16.gguf'
    golden=json.loads((ROOT/'fixtures/golden/qwen2.5-0.5b-instruct-fp16.json').read_text())
    assert sha(legacy)==golden['model_sha256']
    process=subprocess.run([str(cli),'run',str(legacy),'--prompt','Hello','--standalone','--context','128','--max-tokens','1','--threads','1','--profile','reference','--json'],capture_output=True,text=True,env=environment,timeout=120)
    assert process.returncode==0,process.stderr
    legacy_result=json.loads(process.stdout)
    assert legacy_result['token_ids']==golden['cases'][0]['generated_token_ids'][:1]
    cli_evidence={'binary_sha256':sha(cli),'qwen35_run':cli_result,'qwen35_interactive_stdout':interactive,'unsupported_sampling_exit':unsupported,'resource_exhaustion_exit':resource_exit,'legacy_gguf_result':legacy_result,'native_child_executable_path':'absolute binary with empty executable search path','os_rss':'unavailable in sandbox; reservation and buffer capacities reported separately','cache_condition':'uncontrolled'}
    report={'schema_version':1,'scope':'Development CPU text profile; independent array equations, not full publisher framework/vision/API/Metal/platform qualification. Existing GGUF acceptance unaffected.','model_manifest_sha256':sha(ROOT/'fixtures/qwen35/model.json'),'reference_sha256':sha(ROOT/'fixtures/qwen35/reference-v1.json'),'native_binary_sha256':sha(binary),'host':{'system':platform.system(),'machine':platform.machine()},'runs':runs,'cli':cli_evidence}
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('PASS native Qwen3.5 development text',flush=True)
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Qualify native still-image arrays against separately frozen independent fixtures."""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def vector(path):
    v=array.array('f');v.frombytes(path.read_bytes())
    if sys.byteorder!='little':v.byteswap()
    return v
def compare(actual,expected,limits,kind):
    reference=ROOT/'.cache/q35-vision-reference'/(expected['name']+'.f32')
    assert sha(reference)==expected['sha256'],reference
    a,b=vector(actual),vector(reference)
    assert len(a)==len(b)==math.prod(expected['shape']),actual
    delta=sum((float(x)-y)**2 for x,y in zip(a,b));norm=sum(float(y)*y for y in b)
    l2=math.sqrt(delta/max(norm,1e-30));max_abs=max(abs(x-y) for x,y in zip(a,b))
    assert all(math.isfinite(x) and abs(x-y)<=limits[kind+'_absolute']+abs(y)*limits[kind+'_relative'] for x,y in zip(a,b)),(actual,max_abs,l2)
    assert l2<=limits[kind+'_normalized_l2'],(actual,max_abs,l2)
    return {'name':expected['name'],'actual_sha256':sha(actual),'max_absolute':max_abs,'normalized_l2':l2,'values':len(a)}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',required=True,type=Path);a=p.parse_args()
    if a.report.exists():raise FileExistsError(a.report)
    contract_path=ROOT/'fixtures/qwen35/vision/contract-v1.json';reference_path=ROOT/'fixtures/qwen35/vision/reference-v1.json'
    contract=json.loads(contract_path.read_text());ref=json.loads(reference_path.read_text());assert sha(contract_path)==ref['contract_sha256']
    manifest_path=ROOT/'fixtures/qwen35/model.json';manifest=json.loads(manifest_path.read_text());assert sha(manifest_path)==ref['model_manifest_sha256']
    source=ROOT/manifest['cache_directory']
    for item in manifest['files']:
        path=source/item['file'];assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],path
    images={i['id']:i for i in contract['images']}
    for i in images.values():assert sha(ROOT/i['file'])==i['sha256']
    binary=ROOT/'target/release/examples/qwen35_image_probe';out=ROOT/'.cache/b3-vision-native';out.mkdir(exist_ok=True);empty=out/'empty-path';empty.mkdir(exist_ok=True);runs=[]
    for case in ref['cases']:
        directory=out/case['id'];directory.mkdir(exist_ok=True)
        request={'images':[str(ROOT/images[k]['file']) for k in case['images']],'text':case['text'],'teacher':case['greedy_token_ids'][:-1]}
        request_path=directory/'request.json';request_path.write_text(json.dumps(request));print('START',case['id'],flush=True)
        process=subprocess.run([str(binary),str(source),str(request_path),str(directory)],capture_output=True,text=True,env={**os.environ,'PATH':str(empty)},timeout=1200)
        (directory/'stderr.txt').write_text(process.stderr);assert process.returncode==0,process.stderr
        actual=json.loads(process.stdout);(directory/'result.json').write_text(process.stdout)
        assert actual['prompt']==case['prompt'] and actual['prompt_token_ids']==case['prompt_token_ids'],'prompt mismatch'
        assert actual['generation']['token_ids']==case['greedy_token_ids'],(actual['generation']['token_ids'],case['greedy_token_ids'])
        assert actual['generation']['text']==case['greedy_text']==''.join(e['text'] for e in actual['events'])
        assert actual['after']['reserved']==256<<20 and actual['after']['session_buffer_capacity_bytes']==0,actual['after']
        arrays=[]
        for index,key in enumerate(case['images']):
            for expected in ref['encoder_checkpoints']:
                prefix=key+'-'
                if expected['name'].startswith(prefix) and not expected['name'].endswith('-patches'):
                    arrays.append(compare(directory/(str(index)+'-'+expected['name'][len(prefix):]+'.f32'),expected,contract['tolerances'],'vision'))
        last=len(case['prompt_token_ids'])-1
        for i,expected in enumerate(case['checkpoints']):arrays.append(compare(directory/f'{last}-layer.{i}.f32',expected,contract['tolerances'],'language'))
        for i,expected in enumerate(case['teacher_forced']):arrays.append(compare(directory/f'{last+i}-logits.f32',expected,contract['tolerances'],'language'))
        runs.append({'case':case['id'],'tokens':actual['generation']['token_ids'],'text':actual['generation']['text'],'arrays':arrays,'generation':actual['generation'],'after':actual['after'],'result_sha256':sha(directory/'result.json')})
        print('PASS',case['id'],len(arrays),'full arrays',flush=True)
    report={'schema_version':1,'scope':'Experimental CPU native still-image equations for authored constant-color PNGs; square, portrait and ordered two-image input. Independent F32/wide-reduction oracle; no publisher framework, broad image semantics, GPU or other-platform claim.','host':{'system':platform.system(),'machine':platform.machine()},'contract_sha256':sha(contract_path),'reference_sha256':sha(reference_path),'model_manifest_sha256':sha(manifest_path),'binary_sha256':sha(binary),'cache_condition':'uncontrolled','external_executor_invoked':False,'runs':runs}
    a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print('PASS native Qwen3.5 image fixtures',flush=True)
if __name__=='__main__':main()

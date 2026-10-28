#!/usr/bin/env python3
"""Qualify the offline native image-to-text runner against frozen independent arrays."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from check_native_vision import compare, vector, TOLERANCES
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def request_for(case,directory,threads=1,context=256):
    images=[]
    messages=[]
    for message in case['messages']:
        content=message['content']
        if isinstance(content,list):
            parts=[]
            for part in content:
                if part['type']=='image':
                    parts.append({'type':'image','index':len(images)})
                    images.append(str(ROOT/'fixtures/vlm/preprocessing/canonical'/part['fixture']))
                else:
                    parts.append(part)
            content=parts
        messages.append({'role':message['role'],'content':content})
    return {'messages':messages,'images':images,'context':context,'max_memory':768<<20,
            'options':{'max_tokens':8,'threads':threads,'vision':{'threads':threads,'tile_tokens':32,'max_patches':64}},
            'trace_directory':str(directory)}

def check_plan(plan,case):
    assert plan['rendered_prompt']==case['rendered_prompt']
    assert plan['physical_prompt_tokens']==case['encoded']['physical_prompt_tokens']
    assert plan['effective_positions']==case['encoded']['effective_positions']
    chunks=[c for c in case['chunks'] if 'chunk' in c]
    assert len(chunks)==len(plan['chunks'])
    for expected,actual in zip(chunks,plan['chunks']):
        if expected['type']==0:
            assert actual['kind']=='text'
            assert actual['token_ids']==expected['token_ids']
            assert actual['start_position']==expected['start_position']
        else:
            assert actual['kind']=='image' and actual['index']==expected['image_index']
            span=actual['positions']
            assert span['start']==expected['start_position']
            assert span['physical_tokens']==expected['tokens']
            assert span['next']-span['start']==expected['position_count']
            coordinates=next(c['positions'] for c in case['chunks'] if c.get('image_positions')==actual['index'])
            assert coordinates==[[span['start'],span['start']+i%span['columns'],span['start']+i//span['columns'],0] for i in range(span['physical_tokens'])]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-only', action='store_true', help='Run native cases and save verified receipts; no numerical acceptance claim.')
    parser.add_argument('--verify-captured', action='store_true', help='Compare receipt-verified native outputs without executing models again.')
    parser.add_argument('--case',action='append',help='Select reference case IDs; default: all six plus an alternate worker/context case.')
    args=parser.parse_args()
    binary=ROOT/'target/release/examples/vlm_chat'
    digest=sha(binary)
    assert not (args.capture_only and args.verify_captured)
    original_path=ROOT/('fixtures/vlm/image-conditioned-reference-v1.json' if args.capture_only else 'fixtures/vlm/qwen2-image-conditioned-reference-v2.json')
    layer_path=ROOT/'fixtures/vlm/qwen2-language-reference-v2.json'
    original=json.loads(original_path.read_text())
    layers=None if args.capture_only else json.loads(layer_path.read_text())
    if layers is not None:
        assert layers['source_record_sha256']==original['source_record_sha256']
        assert len(original['cases'])==len(layers['cases'])==6
    cases=[c for c in original['cases'] if c['model']=='qwen2-vl-2b' and (not args.case or c['id'] in args.case)]
    assert cases
    weights=ROOT/'.cache/vlm/full/qwen2-vl-2b'
    language=weights/'Qwen2-VL-2B-Instruct-f16.gguf'
    vision=weights/'mmproj-Qwen2-VL-2B-Instruct-f16.gguf'
    assert sha(language)==cases[0]['identity']['language_sha256']
    assert sha(vision)==cases[0]['identity']['vision_sha256']
    cases.sort(key=lambda c:(c['id']!='left-red',c['id']))
    runs=[(c,c['id'],2,256) for c in cases]
    if not args.case:
        red=next(c for c in cases if c['id']=='left-red')
        runs.append((red,'left-red-workers1-context128',1,128))
    observations=[]
    empty=ROOT/'.cache/m2-offline/empty-path'
    empty.mkdir(parents=True,exist_ok=True)
    for case,name,threads,context in runs:
        print('START',name,flush=True)
        directory=ROOT/'.cache/m2-offline-qualified'/name
        directory.mkdir(parents=True,exist_ok=True)
        request=request_for(case,directory,threads,context)
        request_path=directory/'request.json'
        encoded_request=json.dumps(request)
        receipt_path=directory/'receipt.json'
        if args.verify_captured:
            receipt=json.loads(receipt_path.read_text())
            assert receipt['binary_sha256']==digest, 'captured executable does not match current build'
            assert receipt['request_sha256']==hashlib.sha256(encoded_request.encode()).hexdigest()
            assert receipt['result_sha256']==sha(directory/'result.json')
            for filename,expected in receipt['traces'].items():
                assert sha(directory/filename)==expected, 'captured native tensor changed'
            result=json.loads((directory/'result.json').read_text())
        else:
            request_path.write_text(encoded_request)
            env=dict(os.environ,PATH=str(empty))
            result=json.loads(subprocess.check_output([str(binary),str(language),str(vision),str(request_path)],env=env,cwd=ROOT,text=True,timeout=1800))
            (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            receipt={'binary_sha256':digest,'request_sha256':sha(request_path),'result_sha256':sha(directory/'result.json'),'traces':{p.name:sha(p) for p in directory.glob('*.f32')}}
            receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
        if args.capture_only:
            check_plan(result['result']['response']['plan'],case)
            assert result['memory_after_release']['reserved']==result['memory_after_release']['session_buffer_capacity_bytes']==0
            print('CAPTURED',name,'native output saved; numerical acceptance pending',flush=True)
            continue
        assert sha(binary)==digest,'native executable changed during qualification'
        response=result['result']['response']
        assert response['language_sha256']==case['identity']['language_sha256']
        assert response['vision_sha256']==case['identity']['vision_sha256']
        check_plan(response['plan'],case)
        assert response['token_ids']==case['decoded']['generated_token_ids'],f'greedy token mismatch: {name}'
        assert response['text']==case['text'],f'decoded text mismatch: {name}'
        assert response['finish_reason']==('stop' if case['generation'][-1]['eos'] else 'length')
        after=result['memory_after_release']
        assert after['reserved']==after['session_buffer_capacity_bytes']==0
        assert response['memory']['peak_session_buffer_capacity_bytes']<=response['memory']['peak_reserved']
        comparisons=[]
        stage=ROOT/'.cache/qwen-language-reference-v2'/case['id']
        checkpoints=[(r,stage,r['name']) for r in case['tensors'] if r['name'].startswith(('logits_','image_'))]
        layer=next(c for c in layers['cases'] if c['id']==case['id'])
        checkpoints.extend((r,ROOT/'.cache/qwen-language-reference-v2'/case['id'],r['native_checkpoint']) for r in layer['tensors'])
        for record,source,native_name in checkpoints:
            expected=source/(record['name']+'.f32')
            assert sha(expected)==record['sha256'],'frozen reference raw bytes changed'
            actual=directory/(native_name+'.f32')
            item=compare(vector(actual,record['count']),vector(expected,record['count']))
            item.update(name=native_name,count=record['count'],native_sha256=sha(actual),reference_sha256=record['sha256'])
            comparisons.append(item)
        passed=all(c['passed'] for c in comparisons)
        if name=='left-red-workers1-context128':
            previous=next(c for c in observations if c['id']=='left-red')
            expected={c['name']:c['native_sha256'] for c in previous['comparisons']}
            assert all(c['native_sha256']==expected[c['name']] for c in comparisons),'worker/context result changed'
        observations.append({'id':name,'reference_case':case['id'],'threads':threads,'context':context,'native':result,'comparisons':comparisons,'passed':passed})
        print(('PASS' if passed else 'FAIL'),name,repr(response['text']),len(comparisons),'full arrays',flush=True)
    if args.capture_only:
        print('Native capture complete; verify against frozen corrected references before acceptance.')
        return
    report={'schema_version':1,'scope':'Exact selected Qwen2-VL F16 pair, offline native CPU image-to-text; not product CLI/API or release qualification.',
            'host':{'os':platform.platform(),'architecture':platform.machine()},'binary_sha256':digest,'source_sha256':sha(Path(__file__)),
            'reference_manifests':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in [original_path,layer_path]],
            'tolerances':TOLERANCES,'cases':observations,'passed':all(c['passed'] for c in observations),'empty_executable_search_path':True,
            'measurement_limits':['Uncontrolled host/cache conditions; descriptive observations only.','Reservations and vector capacities exclude complete allocator/process/OS cache/driver accounting.','No GPU, secure network ingress, managed sessions, sustained security or other-platform execution is qualified.']}
    name='m2-offline-selected-cases.json' if args.case else 'm2-offline-vlm.json'
    (ROOT/'fixtures/qualification'/name).write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:
        raise SystemExit('Frozen numerical criteria failed; acceptance remains open.')

if __name__=='__main__':
    main()

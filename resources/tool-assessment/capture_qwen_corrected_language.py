#!/usr/bin/env python3
"""Capture the isolated metadata-correct Qwen language profile; preserve all v1 records."""
import copy,hashlib,json,math,os,re,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    original_path=ROOT/'fixtures/vlm/image-conditioned-reference-v1.json'
    original=json.loads(original_path.read_text());source_record=ROOT/'resources/tool-assessment/qwen-language-source-record.json'
    record=json.loads(source_record.read_text());binary=ROOT/'.cache/qwen-language-qualified/bin/vlm-language'
    weights=ROOT/'.cache/vlm/full/qwen2-vl-2b/Qwen2-VL-2B-Instruct-f16.gguf'
    cases=[c for c in original['cases'] if c['model']=='qwen2-vl-2b']
    cases.sort(key=lambda c:(c['id']!='left-red',c['id']))
    assert sha(weights)==cases[0]['identity']['language_sha256']
    lib=binary.parent/'libllama.0.7.0.dylib';assert sha(lib)==record['library_sha256']
    records=[];layers=[]
    for previous in cases:
        directory=ROOT/'.cache/qwen-language-reference-v2'/previous['id'];directory.mkdir(parents=True,exist_ok=True)
        old=ROOT/'.cache/vlm-qualified/qwen2-vl-2b'/previous['id']
        shutil.copyfile(old/'chunks.dat',directory/'chunks.dat');assert sha(directory/'chunks.dat')==previous['staging_sha256']
        env=dict(os.environ,DYLD_LIBRARY_PATH=str(binary.parent))
        with (directory/'capture.jsonl').open('w') as out,(directory/'capture.log').open('w') as err:
            subprocess.run([str(binary),'decode',str(weights),str(directory),'256','8'],env=env,stdout=out,stderr=err,check=True,timeout=900)
        logs=(directory/'capture.log').read_text();assert re.search(r'f_norm_rms_eps\s*=\s*1\.0e-06',logs),'corrected library was not selected'
        lines=[json.loads(line) for line in (directory/'capture.jsonl').read_text().splitlines()]
        case=copy.deepcopy(previous);case['tensors']=[]
        for t in previous['tensors']:
            if t['name'].startswith('image_'):
                src=old/(t['name']+'.f32');assert sha(src)==t['sha256'];shutil.copyfile(src,directory/src.name);case['tensors'].append(t)
        generation=[line for line in lines if 'greedy_step' in line]
        for step in generation:
            name=f"logits_{step['greedy_step']}";t=next(line for line in lines if line.get('name')==name);t['sha256']=sha(directory/(name+'.f32'));case['tensors'].append(t)
        case['generation']=generation;case['decoded']=next(line for line in lines if line.get('stage')=='decoded');case['output_bytes_hex']=''.join(step['piece_hex'] for step in generation);case['text']=bytes.fromhex(case['output_bytes_hex']).decode('utf8',errors='replace')
        case['language_reference']={'binary_sha256':sha(binary),'library_sha256':sha(lib),'rms_norm_epsilon':1e-6,'original_diagnostic_token_ids':previous['decoded']['generated_token_ids'],'source_record_sha256':sha(source_record)}
        case['decoder_output_sha256']=sha(directory/'capture.jsonl');case.pop('memory',None)
        records.append(case)
        batch=sum(math.ceil(c['tokens']/32) for c in case['chunks'] if 'chunk' in c)
        tensors=[]
        for i in range(28):
            name=f'batch_{batch}_l_out-{i}';t=next(line for line in lines if line.get('name')==name);assert t['count']==1536;t['sha256']=sha(directory/(name+'.f32'));t['samples']=t['samples'][:4];t['native_checkpoint']=f'prompt_blk.{i}.output';tensors.append(t)
        layers.append({'id':case['id'],'last_prompt_batch':batch,'tensors':tensors})
        print('Captured metadata-correct Qwen language:',case['id'],repr(case['text']),flush=True)
    common={'schema_version':2,'scope':'Independent metadata-correct Qwen2-VL language reference; original epsilon-zero records retained as diagnostics. No native qualification.','reference_revision':original['reference_revision'],'source_record':'resources/tool-assessment/qwen-language-source-record.json','source_record_sha256':sha(source_record),'capture_source_sha256':sha(Path(__file__)),'wrapper_source_sha256':sha(ROOT/'resources/tool-assessment/vlm_language_oracle.cpp'),'base_wrapper_sha256':sha(ROOT/'resources/tool-assessment/vlm_oracle.cpp'),'binary_sha256':sha(binary),'original_reference_sha256':sha(original_path),'settings':original['settings'],'native_image_understanding_qualified':False}
    report=dict(common,cases=records)
    (ROOT/'fixtures/vlm/qwen2-image-conditioned-reference-v2.json').write_text(json.dumps(report,indent=2)+'\n')
    layer_report=dict(common,cases=layers)
    (ROOT/'fixtures/vlm/qwen2-language-reference-v2.json').write_text(json.dumps(layer_report,indent=2)+'\n')
if __name__=='__main__':main()

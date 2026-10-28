#!/usr/bin/env python3
"""Freeze independent decoder layers from immutable staged image references."""
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    source_path=ROOT/'fixtures/vlm/image-conditioned-reference-v1.json'
    source=json.loads(source_path.read_text())
    binary=ROOT/'.cache/image-oracle/vlm-language'
    weights=ROOT/'.cache/vlm/full/qwen2-vl-2b/Qwen2-VL-2B-Instruct-f16.gguf'
    cases=[c for c in source['cases'] if c['model']=='qwen2-vl-2b']
    assert sha(weights)==cases[0]['identity']['language_sha256']
    records=[]
    for case in cases:
        original=ROOT/'.cache/vlm-qualified/qwen2-vl-2b'/case['id']
        directory=ROOT/'.cache/vlm-language-reference'/case['id']
        directory.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(original/'chunks.dat',directory/'chunks.dat')
        assert sha(directory/'chunks.dat')==case['staging_sha256']
        with (directory/'capture.jsonl').open('w') as out,(directory/'capture.log').open('w') as err:
            subprocess.run([str(binary),'decode',str(weights),str(directory),'256','8'],stdout=out,stderr=err,check=True,timeout=900)
        lines=[json.loads(line) for line in (directory/'capture.jsonl').read_text().splitlines()]
        tensors=[]
        batch=sum(math.ceil(chunk['tokens']/32) for chunk in case['chunks'] if 'chunk' in chunk)
        for layer in range(28):
            name=f'batch_{batch}_l_out-{layer}'
            item=next(line for line in lines if line.get('name')==name)
            assert item['count']==1536
            item['samples']=item['samples'][:4]
            item['sha256']=sha(directory/(name+'.f32'))
            item['native_checkpoint']=f'prompt_blk.{layer}.output'
            tensors.append(item)
        logits=[t for t in case['tensors'] if t['name'].startswith('logits_')]
        for item in logits:
            assert sha(directory/(item['name']+'.f32'))==item['sha256'],'layer instrumentation changed frozen logits'
        decoded=next(line for line in lines if line.get('stage')=='decoded')
        assert decoded['generated_token_ids']==case['decoded']['generated_token_ids']
        records.append({'id':case['id'],'staging_sha256':case['staging_sha256'],'last_prompt_batch':batch,'tensors':tensors,'unchanged_logit_arrays':len(logits),'greedy_token_ids':decoded['generated_token_ids']})
        print('Captured independent decoder layers:',case['id'],flush=True)
    report={'schema_version':1,'scope':'Independent Qwen2-VL decoder checkpoints; does not qualify native execution.','reference_revision':source['reference_revision'],'component_sha256':cases[0]['identity']['language_sha256'],'staged_reference_sha256':sha(source_path),'wrapper_source_sha256':sha(ROOT/'resources/tool-assessment/vlm_language_oracle.cpp'),'base_wrapper_sha256':sha(ROOT/'resources/tool-assessment/vlm_oracle.cpp'),'capture_source_sha256':sha(Path(__file__)),'binary_sha256':sha(binary),'cases':records}
    (ROOT/'fixtures/vlm/qwen2-language-reference-v1.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    main()

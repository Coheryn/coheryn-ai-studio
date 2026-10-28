#!/usr/bin/env python3
"""Compare cooperative native image inference with the frozen full-array reference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import uuid
from check_native_vision import compare, vector, TOLERANCES
from check_native_vlm import check_plan, request_for

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    binary = ROOT/'target/release/examples/vlm_chat'
    binary_sha = digest(binary)
    reference = ROOT/'fixtures/vlm/qwen2-image-conditioned-reference-v2.json'
    layers = ROOT/'fixtures/vlm/qwen2-language-reference-v2.json'
    case = next(c for c in json.loads(reference.read_text())['cases'] if c['id'] == 'left-red')
    layer = next(c for c in json.loads(layers.read_text())['cases'] if c['id'] == case['id'])
    run = ROOT/'.cache/m4-native-numerics'/uuid.uuid4().hex
    run.mkdir(parents=True)
    request = request_for(case, run, threads=2, context=256)
    request['cooperative'] = True
    request_path = run/'request.json'
    request_path.write_text(json.dumps(request, ensure_ascii=False)+'\n')
    language = ROOT/'.cache/vlm/full/qwen2-vl-2b/Qwen2-VL-2B-Instruct-f16.gguf'
    vision = ROOT/'.cache/vlm/full/qwen2-vl-2b/mmproj-Qwen2-VL-2B-Instruct-f16.gguf'
    output = subprocess.check_output([str(binary),str(language),str(vision),str(request_path)],env=dict(os.environ,PATH=''),cwd=ROOT,text=True,timeout=1800)
    result = json.loads(output)
    result_path = run/'result.json'
    result_path.write_text(json.dumps(result,indent=2)+'\n')
    response = result['result']['response']
    check_plan(response['plan'],case)
    assert response['token_ids'] == case['decoded']['generated_token_ids'] and response['text'] == case['text']
    assert response['language_sha256'] == case['identity']['language_sha256'] and response['vision_sha256'] == case['identity']['vision_sha256']
    assert result['result']['cooperative_yields'] > 100
    assert result['memory_after_release']['reserved'] == result['memory_after_release']['session_buffer_capacity_bytes'] == 0
    source = ROOT/'.cache/qwen-language-reference-v2'/case['id']
    tensors = [(r,r['name']) for r in case['tensors'] if r['name'].startswith(('logits_','image_'))]
    tensors.extend((r,r['native_checkpoint']) for r in layer['tensors'])
    observations = []
    for record,name in tensors:
        expected = source/(record['name']+'.f32')
        actual = run/(name+'.f32')
        assert digest(expected) == record['sha256']
        comparison = compare(vector(actual,record['count']),vector(expected,record['count']))
        comparison.update(name=name,count=record['count'],native_sha256=digest(actual),reference_sha256=record['sha256'])
        assert comparison['passed'], comparison
        observations.append(comparison)
    assert digest(binary) == binary_sha
    report = dict(schema_version=1,scope='Current-host exact-pair cooperative CPU numerical checkpoint. All projected embeddings, final-prompt decoder layers and teacher-forced decode logits for frozen left-red; unchanged precision thresholds. No external execution or performance claim.',
        host=platform.platform(),binary_sha256=binary_sha,checker_sha256=digest(Path(__file__)),reference_sha256=digest(reference),layer_reference_sha256=digest(layers),
        tolerances=TOLERANCES,case=case['id'],threads=2,context=256,cooperative_yields=result['result']['cooperative_yields'],comparisons=observations,
        all_arrays_bit_exact=all(r['native_sha256']==r['reference_sha256'] for r in observations),passed=True,
        text=response['text'],token_ids=response['token_ids'],memory_after_release=result['memory_after_release'],run_directory=str(run.relative_to(ROOT)),
        request_sha256=digest(request_path),result_sha256=digest(result_path),execution_search_path='empty')
    (ROOT/'fixtures/qualification/m4-cooperative-numerics.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS cooperative native full-array reference:',len(observations),'arrays;',report['cooperative_yields'],'yield checkpoints; bit exact:',report['all_arrays_bit_exact'],flush=True)

if __name__ == '__main__':
    main()

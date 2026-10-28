#!/usr/bin/env python3
"""Sequential isolated VLM reference capture with pinned inputs and resume receipts."""
import argparse
from array import array
import math
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.cache/devtools'))
sys.path.insert(0,str(ROOT/'.cache/image-reference'))
import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment
from PIL import Image
CACHE=ROOT/'.cache/vlm-qualified'
BINARY=ROOT/'.cache/image-oracle/vlm-corrected'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def records(path):return [json.loads(line) for line in path.read_text().splitlines()]


def receipt(stage,directory,identity,command,timeout):
    record_path=directory/(stage+'.receipt.json')
    if record_path.exists():
        record=json.loads(record_path.read_text())
        assert record['identity']==identity,'capture identity changed; preserve old candidate directory'
        assert all(sha(directory/path)==value for path,value in record['files'].items()),'cached capture changed'
        print('Verified resume',directory.name,stage,flush=True)
        return record
    start=time.monotonic()
    with (directory/(stage+'.jsonl')).open('w') as output,(directory/(stage+'.log')).open('w') as errors:
        subprocess.run(command,stdout=output,stderr=errors,check=True,timeout=timeout)
    lines=records(directory/(stage+'.jsonl'))
    files={stage+'.jsonl':sha(directory/(stage+'.jsonl'))}
    if stage=='encode':files['chunks.dat']=sha(directory/'chunks.dat')
    for item in lines:
        if 'name' in item:
            path=directory/(item['name']+'.f32');assert path.stat().st_size==item['count']*4
            files[path.name]=sha(path)
    record={'identity':identity,'files':files,'elapsed_seconds':time.monotonic()-start,
            'process_peak_rss_bytes':lines[-1]['process_peak_rss_bytes']}
    record_path.write_text(json.dumps(record,indent=2)+'\n')
    print('Captured',directory.name,stage,flush=True)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['llava-1.5-7b','qwen2-vl-2b'],action='append')
    parser.add_argument('--case',action='append')
    parser.add_argument('--stage',choices=['encode','decode','all','collect'],default='all')
    parser.add_argument('--hardware-record',type=Path)
    args=parser.parse_args()
    manifest_path=ROOT/'fixtures/vlm/models.json'
    request_path=ROOT/'fixtures/vlm/requests-v1.json'
    manifest=json.loads(manifest_path.read_text());requests=json.loads(request_path.read_text())
    environment=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
    # The source template uses generation tags; reference rendering has no token mask.
    selected_records=[]
    for group in requests['models']:
        mid=group['model']
        if args.model and mid not in args.model:continue
        model=next(m for m in manifest['models'] if m['id']==mid)
        components={c['role']:c for c in model['components']}
        paths={role:ROOT/'.cache/vlm/full'/mid/c['file'] for role,c in components.items()}
        for role,path in paths.items():assert path.stat().st_size==components[role]['bytes'] and sha(path)==components[role]['sha256']
        template_path=ROOT/'fixtures/vlm'/mid/'chat_template.json'
        template=json.loads(template_path.read_text())['chat_template']
        # Removing these inert annotation tags does not change rendered prompt bytes.
        rendered_template=template.replace('{% generation %}','').replace('{% endgeneration %}','')
        for request in group['cases']:
            if args.case and request['id'] not in args.case:continue
            directory=CACHE/mid/request['id'];directory.mkdir(parents=True,exist_ok=True)
            images=[];messages=[]
            for source in request['messages']:
                message=dict(source)
                if isinstance(message['content'],list):
                    parts=[]
                    for item in message['content']:
                        if item['type']=='image':
                            image_path=ROOT/'fixtures/vlm/preprocessing/canonical'/item['fixture']
                            image=Image.open(image_path).convert('RGB')
                            raw=directory/f'image-{len(images)}.rgb';raw.write_bytes(image.tobytes())
                            images.append({'canonical':str(image_path.relative_to(ROOT)),'sha256':sha(image_path),'rgb_sha256':sha(raw),'width':image.width,'height':image.height})
                            parts.append({'type':'image'})
                        else:parts.append(item)
                    message['content']=parts
                messages.append(message)
            prompt=environment.from_string(rendered_template).render(messages=messages,add_generation_prompt=True,add_vision_id=False)
            marker_prompt=prompt.replace('<image>','<__media__>') if mid=='llava-1.5-7b' else prompt.replace('<|vision_start|><|image_pad|><|vision_end|>','<__media__>')
            assert marker_prompt.count('<__media__>')==len(images)
            (directory/'prompt.txt').write_text(marker_prompt)
            (directory/'images.txt').write_text(''.join(f'{image["width"]} {image["height"]} "{directory/f"image-{i}.rgb"}"\n' for i,image in enumerate(images)))
            identity={'model':mid,'case':request['id'],'language_sha256':components['language']['sha256'],'vision_sha256':components['vision-projector']['sha256'],
                'binary_sha256':sha(BINARY),'wrapper_source_sha256':sha(ROOT/'resources/tool-assessment/vlm_oracle.cpp'),
                'corrected_source_record_sha256':sha(ROOT/'.cache/vision-corrected-source-record.json'),
                'marker_prompt_sha256':sha(directory/'prompt.txt'),'images':images,'settings':requests['settings']}
            if args.stage in ('encode','all'):
                receipt('encode',directory,identity,[str(BINARY),'encode',str(paths['language']),str(paths['vision-projector']),str(directory/'prompt.txt'),str(directory/'images.txt'),str(directory)],1200)
            if args.stage in ('decode','all'):
                assert (directory/'encode.receipt.json').is_file(),'encode and verify staged features first'
                receipt('encode',directory,identity,[],0)
                if mid=='llava-1.5-7b':
                    comparison=directory/'numpy-comparison.json'
                    if not comparison.exists():
                        png=ROOT/'fixtures/vlm/preprocessing/resized'/('llava15-pad336-v1-'+images[0]['canonical'].split('/')[-1])
                        with (directory/'numpy.log').open('w') as output:
                            subprocess.run([sys.executable,str(ROOT/'resources/tool-assessment/llava_numpy.py'),str(directory),str(png)],stdout=output,stderr=subprocess.STDOUT,check=True,timeout=1200)
                    compared=json.loads(comparison.read_text())
                    assert compared['source_sha256']==sha(ROOT/'resources/tool-assessment/llava_numpy.py')
                    assert all(item['outside_tolerance']==0 and item['normalized_l2']<=0.0005 for item in compared['comparisons'])
                receipt('decode',directory,identity,[str(BINARY),'decode',str(paths['language']),str(directory),str(requests['settings']['context'][mid]),str(requests['settings']['max_tokens'])],7200)
            if args.stage=='collect':
                assert (directory/'decode.receipt.json').exists(),'finish every numerical capture before collection'
                receipt('encode',directory,identity,[],0)
                receipt('decode',directory,identity,[],0)
            if not (directory/'decode.receipt.json').exists():continue
            encoded=records(directory/'encode.jsonl');decoded=records(directory/'decode.jsonl')
            tensors=[]
            for item in encoded+decoded:
                if 'name' in item:
                    item=dict(item);item['sha256']=sha(directory/(item['name']+'.f32'));tensors.append(item)
            generation=[item for item in decoded if 'greedy_step' in item]
            output_bytes=b''.join(bytes.fromhex(item['piece_hex']) for item in generation)
            row={'model':mid,'id':request['id'],'identity':identity,'messages':request['messages'],
                'rendered_prompt':prompt,'marker_prompt':marker_prompt,'template_sha256':sha(template_path),
                'chunks':[item for item in encoded if 'chunk' in item or 'image_positions' in item],
                'encoded':next(item for item in encoded if item.get('stage')=='encoded'),
                'decoded':next(item for item in decoded if item.get('stage')=='decoded'),
                'generation':generation,'text':output_bytes.decode('utf8',errors='replace'),
                'output_bytes_hex':output_bytes.hex(),'tensors':tensors,
                'staging_sha256':sha(directory/'chunks.dat'),'encoder_output_sha256':sha(directory/'encode.jsonl'),'decoder_output_sha256':sha(directory/'decode.jsonl'),
                'memory':{stage:json.loads((directory/(stage+'.receipt.json')).read_text())['process_peak_rss_bytes'] for stage in ['encode','decode']}}
            if mid=='llava-1.5-7b':row['independent_vision_comparison']=json.loads((directory/'numpy-comparison.json').read_text())
            selected_records.append(row)
    if selected_records:
        path=ROOT/'fixtures/vlm/image-conditioned-reference-v1.json'
        previous=json.loads(path.read_text()) if path.exists() else {'cases':[]}
        updated={(item['model'],item['id']):item for item in previous['cases']}
        updated.update({(item['model'],item['id']):item for item in selected_records})
        result={'schema_version':1,'scope':'independent corrected reference evidence; not native image-to-text execution',
            'host':{'os':platform.platform(),'architecture':platform.machine(),'python':platform.python_version()},'compiler':subprocess.check_output(['clang++','--version'],text=True).splitlines()[0],'reference_source_record':'resources/tool-assessment/vision-corrected-source-record.json','reference_source_sha256':sha(ROOT/'.cache/vision-corrected-source-record.json'),
            'reference_revision':manifest['oracle']['revision'],'capture_source_sha256':sha(Path(__file__)),
            'requests_sha256':sha(request_path),'template_renderer':{'implementation':'Jinja2','version':jinja2.__version__,'generation_tags':'removed as inert rendering annotations'},
            'native_image_understanding_qualified':False,'settings':requests['settings'],'cache_condition':'uncontrolled',
            'cases':[updated[key] for key in sorted(updated)],'reference_libraries':{p.name:sha(p) for p in sorted((ROOT/'.cache/vision-corrected-build/bin').glob('*.dylib')) if not p.is_symlink()}}
        result['image_sensitivity']=[]
        case_map={(c['model'],c['id']):c for c in result['cases']}
        for mid,left,right in [('llava-1.5-7b','left-red','left-blue'),('qwen2-vl-2b','left-red','left-blue'),('qwen2-vl-2b','ordered-images','reversed-images')]:
            if (mid,left) not in case_map or (mid,right) not in case_map:continue
            a=array('f');b=array('f')
            a.frombytes((CACHE/mid/left/'logits_0.f32').read_bytes());b.frombytes((CACHE/mid/right/'logits_0.f32').read_bytes())
            if sys.byteorder!='little':a.byteswap();b.byteswap()
            assert len(a)==len(b)>0
            difference=[float(x)-float(y) for x,y in zip(a,b)]
            first,second=case_map[(mid,left)],case_map[(mid,right)]
            assert first['marker_prompt']==second['marker_prompt']
            result['image_sensitivity'].append({'model':mid,'cases':[left,right],'prompt_bytes_equal':True,
                'maximum_absolute_logit_change':max(map(abs,difference)),
                'relative_l2_change':math.sqrt(sum(d*d for d in difference))/max(math.sqrt(sum(float(v)*v for v in a)),1e-30),
                'bounded_greedy_prefix_equal':first['decoded']['generated_token_ids']==second['decoded']['generated_token_ids'],
                'note':'Conditioning sensitivity, not numerical error or a complete semantic-quality benchmark.'})
        result['host'].update({'cpu_model':None,'ram_total_bytes':None})
        if args.hardware_record:
            hardware=json.loads(args.hardware_record.read_text())
            assert isinstance(hardware['cpu_model'],str) and isinstance(hardware['ram_total_bytes'],int) and hardware['ram_total_bytes']>0
            result['host'].update(hardware)
        result['unavailable_measurements']=['private memory','OS file-cache residency','per-request mapped residency','page-fault and I/O attribution','NumPy process peak memory']
        path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
        print('Published candidate reference cases:',len(result['cases']),flush=True)


if __name__=='__main__':main()

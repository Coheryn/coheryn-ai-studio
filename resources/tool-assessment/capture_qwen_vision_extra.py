#!/usr/bin/env python3
"""Capture additional independent Qwen vision geometries; outside native CI/runtime."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.cache/image-reference'))
from PIL import Image

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    model=next(m for m in json.loads((ROOT/'fixtures/vlm/models.json').read_text())['models'] if m['id']=='qwen2-vl-2b')
    component=next(c for c in model['components'] if c['role']=='vision-projector')
    path=ROOT/'.cache/vlm/full'/model['id']/component['file'];assert path.stat().st_size==component['bytes'] and sha(path)==component['sha256']
    binary=ROOT/'.cache/image-oracle/vision';records=[]
    for name in ['portrait-png.png','minimum-png.png','odd-rgb-png.png']:
        source=ROOT/'fixtures/vlm/preprocessing/canonical'/name;image=Image.open(source).convert('RGB')
        directory=ROOT/'.cache/vision-extra'/source.stem;directory.mkdir(parents=True,exist_ok=True)
        (directory/'input.rgb').write_bytes(image.tobytes())
        with (directory/'capture.jsonl').open('w') as out,(directory/'capture.log').open('w') as err:
            subprocess.run([str(binary),str(path),'qwen2vl-still-v1',*map(str,image.size),str(directory/'input.rgb'),str(directory)],stdout=out,stderr=err,check=True,timeout=900)
        lines=[json.loads(s) for s in (directory/'capture.jsonl').read_text().splitlines()]
        for t in lines[1:]:
            raw=directory/(t['name']+'.f32');assert raw.stat().st_size==t['count']*4;t['sha256']=sha(raw)
        records.append({'canonical':str(source.relative_to(ROOT)),'canonical_sha256':sha(source),'settings':lines[0],'tensors':lines[1:]})
        print('Captured independent geometry:',name,flush=True)
    report={'schema_version':1,'reference_revision':'2280f0fd163b03d4331d49e08460f66502e35854','component_sha256':component['sha256'],'binary_sha256':sha(binary),'wrapper_source_sha256':sha(ROOT/'resources/tool-assessment/vision_oracle.cpp'),'capture_source_sha256':sha(Path(__file__)),'native_execution_qualified':False,'cases':records}
    (ROOT/'fixtures/vlm/qwen2-vision-extra-v1.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

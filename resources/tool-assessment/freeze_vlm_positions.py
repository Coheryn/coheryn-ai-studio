#!/usr/bin/env python3
"""Freeze existing independent encoder positions; never generate expected native values."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    requests=json.loads((ROOT/'fixtures/vlm/requests-v1.json').read_text())
    spans=[];sources=[]
    for group in requests['models']:
        for case in group['cases']:
            directory=ROOT/'.cache/vlm-qualified'/group['model']/case['id']
            receipt=json.loads((directory/'encode.receipt.json').read_text())
            assert receipt['identity']['wrapper_source_sha256']==sha(ROOT/'resources/tool-assessment/vlm_oracle.cpp')
            assert all(sha(directory/name)==digest for name,digest in receipt['files'].items())
            lines=[json.loads(line) for line in (directory/'encode.jsonl').read_text().splitlines()]
            for chunk in [line for line in lines if line.get('type')==1 and 'chunk' in line]:
                index=chunk['image_index']
                coordinates=next(line['positions'] for line in lines if line.get('image_positions')==index)
                axes=[[p[0],p[2],p[1],p[3]] for p in coordinates]
                spans.append({'model':group['model'],'case':case['id'],'image_index':index,
                    'processor':'qwen2vl-still-v1' if group['model']=='qwen2-vl-2b' else 'llava15-pad336-v1',
                    'start':chunk['start_position'],'physical_tokens':chunk['tokens'],'position_count':chunk['position_count'],
                    'next':chunk['start_position']+chunk['position_count'],'decoder_axes':axes})
            sources.append({'model':group['model'],'case':case['id'],'encoder_output_sha256':sha(directory/'encode.jsonl')})
    assert len(spans)==10
    output=ROOT/'fixtures/vlm/positions-v1.json'
    output.write_text(json.dumps({'schema_version':1,'scope':'Independent image positions, not inference or complete memory qualification',
        'axes':['temporal','vertical','horizontal','reserved'],'wrapper_source_sha256':sha(ROOT/'resources/tool-assessment/vlm_oracle.cpp'),
        'freeze_source_sha256':sha(Path(__file__)),'encoder_sources':sources,'spans':spans},indent=2)+'\n')
    print('Frozen ten independent positional spans')


if __name__=='__main__':main()

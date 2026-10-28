#!/usr/bin/env python3
"""Independent LLaVA vision math from identical GGUF weights, outside native execution."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
sys.path.insert(0, str(ROOT / '.cache/vision-numpy'))
sys.path.insert(0, str(ROOT / '.cache/image-reference'))
import numpy as np
from PIL import Image


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case_directory',type=Path)
    parser.add_argument('resized_png',type=Path)
    args=parser.parse_args()
    directory=args.case_directory.resolve()
    manifest=json.loads((ROOT/'fixtures/vlm/models.json').read_text())
    model=next(m for m in manifest['models'] if m['id']=='llava-1.5-7b')
    component=next(c for c in model['components'] if c['role']=='vision-projector')
    weights=ROOT/'.cache/vlm/full'/model['id']/component['file']
    assert weights.stat().st_size==component['bytes'] and sha(weights)==component['sha256']
    inventory=json.loads((ROOT/component['inventory']).read_text())['descriptor']
    descriptors={t['name']:t for t in inventory['tensors']}
    raw=np.memmap(weights,mode='r',dtype=np.uint8)
    def tensor(name):
        t=descriptors[name]
        assert t['dtype'] in (0,1)
        result=np.ndarray(tuple(reversed(t['dimensions'])),dtype='<f4' if t['dtype']==0 else '<f2',buffer=raw,offset=inventory['data_offset']+t['offset'])
        return result
    def matrix(x,name):
        w=tensor(name)
        # The assessment scalar kernel converts F16 matrix operands before its dot.
        activation=x.astype(np.float16).astype(np.float64) if w.dtype==np.dtype('<f2') else x.astype(np.float64)
        return (activation@w.astype(np.float64).T).astype(np.float32)
    def affine(x,prefix):return matrix(x,prefix+'.weight')+tensor(prefix+'.bias')
    def norm(x,prefix):
        mean=x.sum(axis=-1,keepdims=True,dtype=np.float64).astype(np.float32)/np.float32(x.shape[-1])
        centered=x-mean
        variance=(centered*centered).sum(axis=-1,keepdims=True,dtype=np.float64)/x.shape[-1]
        variance=variance.astype(np.float32)
        y=centered*(np.float32(1)/np.sqrt(variance+np.float32(1e-5)))
        return y*tensor(prefix+'.weight')+tensor(prefix+'.bias')
    lib=ctypes.CDLL(str(ROOT/'.cache/image-oracle/scalar-math.dylib'))
    pointer=ctypes.POINTER(ctypes.c_float)
    for operation in [lib.coheryn_assessment_exp,lib.coheryn_assessment_tanh]:
        operation.argtypes=[ctypes.c_size_t,pointer,pointer]
        operation.restype=None
    def pointwise(operation,x):
        x=np.ascontiguousarray(x,dtype=np.float32);out=np.empty_like(x)
        operation(x.size,x.ctypes.data_as(pointer),out.ctypes.data_as(pointer))
        return out
    def dot32(a,b):
        out=np.empty((a.shape[0],b.shape[1]),dtype=np.float32)
        for offset in range(0,a.shape[0],16):
            out[offset:offset+16]=np.sum(a[offset:offset+16,:,None]*b[None,:,:],axis=1,dtype=np.float64)
        return out
    reports=[]
    def compare(name,actual):
        reference=np.fromfile(directory/(name+'.f32'),dtype='<f4').reshape(actual.shape)
        assert actual.dtype==np.float32 and np.isfinite(actual).all()
        diff=np.abs(actual.astype(np.float64)-reference)
        threshold=0.003+0.001*np.abs(reference.astype(np.float64))
        l2=float(np.linalg.norm(diff.ravel())/max(float(np.linalg.norm(reference.astype(np.float64).ravel())),1e-30))
        row={'name':name,'shape':list(actual.shape),'maximum_absolute':float(diff.max()),'outside_tolerance':int((diff>threshold).sum()),'normalized_l2':l2}
        reports.append(row)
        print(json.dumps(row),flush=True)
        np.asarray(actual,dtype='<f4').tofile(directory/(name+'.numpy.f32'))
        row['independent_sha256']=sha(directory/(name+'.numpy.f32'))
        row['reference_sha256']=sha(directory/(name+'.f32'))
        assert row['outside_tolerance']==0 and l2<=0.0005, f'independent numeric mismatch: {name}'
    start=time.monotonic()
    rgb=np.asarray(Image.open(args.resized_png).convert('RGB'))
    assert rgb.shape==(336,336,3)
    normalized=(rgb.astype(np.float32)/np.float32(255)-np.array([0.48145466,0.4578275,0.40821073],dtype=np.float32))/np.array([0.26862954,0.26130258,0.27577711],dtype=np.float32)
    patches=normalized.reshape(24,14,24,14,3).transpose(0,2,4,1,3).reshape(576,588)
    convolution=tensor('v.patch_embd.weight').reshape(1024,588)
    projected=(patches.astype(np.float16).astype(np.float64)@convolution.astype(np.float64).T).astype(np.float32)
    x=np.concatenate([tensor('v.class_embd')[None,:],projected],axis=0)+tensor('v.position_embd.weight').astype(np.float32)
    x=norm(x,'v.pre_ln')
    compare('vision_0_pre_ln',x)
    for layer in range(23):
        prefix=f'v.blk.{layer}.'
        h=norm(x,prefix+'ln1')
        q=affine(h,prefix+'attn_q').reshape(577,16,64).transpose(1,0,2)
        k=affine(h,prefix+'attn_k').reshape(577,16,64).transpose(1,0,2)
        v=affine(h,prefix+'attn_v').reshape(577,16,64).transpose(1,0,2)
        scores=np.stack([dot32(q[h],k[h].T) for h in range(16)])*np.float32(0.125)
        exp=pointwise(lib.coheryn_assessment_exp,scores-scores.max(axis=-1,keepdims=True))
        probabilities=exp*((1.0/exp.sum(axis=-1,keepdims=True,dtype=np.float64)).astype(np.float32))
        attended=np.stack([dot32(probabilities[h],v[h]) for h in range(16)]).transpose(1,0,2).reshape(577,1024)
        x=x+affine(attended,prefix+'attn_out')
        h=norm(x,prefix+'ln2')
        # Legacy names are inverted relative to the actual FFN tensor dimensions.
        h=affine(h,prefix+'ffn_down')
        h=h*(np.float32(1)/(np.float32(1)+pointwise(lib.coheryn_assessment_exp,np.float32(-1.702)*h)))
        x=x+affine(h,prefix+'ffn_up')
        compare(f'vision_0_layer_out-{layer}',x)
    x=affine(x[1:],'mm.0')
    x=np.float32(0.5)*x*(np.float32(1)+pointwise(lib.coheryn_assessment_tanh,np.float32(0.7978845608028654)*x*(np.float32(1)+np.float32(0.044715)*x*x)))
    x=affine(x,'mm.2')
    compare('image_0_embeddings',x)
    report={'schema_version':1,'implementation':'independent NumPy operations, no external inference runtime and no native Coheryn numeric code',
            'numpy_version':np.__version__,'source_sha256':sha(Path(__file__)),'standard_math_source_sha256':sha(ROOT/'resources/tool-assessment/scalar_math.c'),'standard_math_binary_sha256':sha(ROOT/'.cache/image-oracle/scalar-math.dylib'),'component_sha256':component['sha256'],
            'resized_input_sha256':sha(args.resized_png),'elapsed_seconds':time.monotonic()-start,
            'policy':'23 stored blocks, CLS first, omit CLS before projector, exact pinned GGUF pad336 pixels; half operands for F16 matrices, F32 activations',
            'tolerances':{'absolute':0.003,'relative':0.001,'normalized_l2':0.0005},'comparisons':reports}
    (directory/'numpy-comparison.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

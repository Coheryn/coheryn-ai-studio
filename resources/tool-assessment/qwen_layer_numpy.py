#!/usr/bin/env python3
"""Independent first Qwen decoder layer, checking the epsilon correction both ways."""
import ctypes,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
os.environ['VECLIB_MAXIMUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
sys.path.insert(0,str(ROOT/'.cache/vision-numpy'))
import numpy as np

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    original=json.loads((ROOT/'fixtures/vlm/image-conditioned-reference-v1.json').read_text())
    case=next(c for c in original['cases'] if c['model']=='qwen2-vl-2b' and c['id']=='left-red')
    weights=ROOT/'.cache/vlm/full/qwen2-vl-2b/Qwen2-VL-2B-Instruct-f16.gguf';assert sha(weights)==case['identity']['language_sha256']
    inventory=json.loads((ROOT/'fixtures/vlm/qwen2-vl-2b/language.inventory.json').read_text())
    eps=inventory['architecture_and_processor_metadata']['qwen2vl.attention.layer_norm_rms_epsilon'];assert abs(eps-1e-6)<1e-12
    d=inventory['descriptor'];descriptors={t['name']:t for t in d['tensors']};raw=np.memmap(weights,mode='r',dtype=np.uint8)
    def tensor(name):
        t=descriptors[name];return np.ndarray(tuple(reversed(t['dimensions'])),dtype='<f4' if t['dtype']==0 else '<f2',buffer=raw,offset=d['data_offset']+t['offset'])
    def matrix(x,name):return (x.astype(np.float16).astype(np.float64)@tensor(name).astype(np.float64).T).astype(np.float32)
    def affine(x,name):return matrix(x,name+'.weight')+tensor(name+'.bias')
    def norm(x,name,epsilon):
        variance=((x*x).sum(axis=-1,keepdims=True,dtype=np.float64)/x.shape[-1]).astype(np.float32)
        return (x*(np.float32(1)/np.sqrt(variance+np.float32(epsilon))))*tensor(name)
    lib=ctypes.CDLL(str(ROOT/'.cache/image-oracle/qwen-scalar-math.dylib'));pointer=ctypes.POINTER(ctypes.c_float)
    lib.coheryn_assessment_exp.argtypes=[ctypes.c_size_t,pointer,pointer];lib.coheryn_assessment_exp.restype=None
    lib.coheryn_assessment_sincos.argtypes=[ctypes.c_size_t,pointer,pointer,pointer];lib.coheryn_assessment_sincos.restype=None
    lib.coheryn_assessment_pow.argtypes=[ctypes.c_float,ctypes.c_float];lib.coheryn_assessment_pow.restype=ctypes.c_float
    def exp(x):
        x=np.ascontiguousarray(x);out=np.empty_like(x);lib.coheryn_assessment_exp(x.size,x.ctypes.data_as(pointer),out.ctypes.data_as(pointer));return out
    inputs=[];positions=[]
    for chunk in case['chunks']:
        if 'chunk' not in chunk:continue
        if chunk['type']==0:
            inputs.append(tensor('token_embd.weight')[chunk['token_ids']].astype(np.float32));positions.extend([[chunk['start_position']+i]*3+[0] for i in range(chunk['tokens'])])
        else:
            name=f"image_{chunk['image_index']}_embeddings";path=ROOT/'.cache/vlm-qualified/qwen2-vl-2b/left-red'/(name+'.f32');record=next(t for t in case['tensors'] if t['name']==name);assert sha(path)==record['sha256'];inputs.append(np.fromfile(path,dtype='<f4').reshape(-1,1536));coordinates=next(c['positions'] for c in case['chunks'] if c.get('image_positions')==chunk['image_index']);positions.extend([[p[0],p[2],p[1],p[3]] for p in coordinates])
    x=np.concatenate(inputs,axis=0);n=len(x);assert n==42
    angles=np.empty((n,64),dtype=np.float32);step=np.float32(lib.coheryn_assessment_pow(1e6,np.float32(-2/128)))
    for row,p in enumerate(positions):
        theta=np.array(p,dtype=np.float32)
        for i in range(64):angles[row,i]=theta[0 if i<16 else 1 if i<40 else 2];theta*=step
    cosine=np.empty_like(angles);sine=np.empty_like(angles);lib.coheryn_assessment_sincos(angles.size,angles.ctypes.data_as(pointer),cosine.ctypes.data_as(pointer),sine.ctypes.data_as(pointer))
    def rotate(x):
        a=x[:,:,:64].copy();b=x[:,:,64:].copy();x[:,:,:64]=a*cosine[:,None,:]-b*sine[:,None,:];x[:,:,64:]=a*sine[:,None,:]+b*cosine[:,None,:];return x
    reports=[]
    for epsilon,kind in [(0.,'original-diagnostic'),(eps,'metadata-correct')]:
        start=time.monotonic();normalized=norm(x,'blk.0.attn_norm.weight',epsilon)
        q=rotate(affine(normalized,'blk.0.attn_q').reshape(n,12,128));k=rotate(affine(normalized,'blk.0.attn_k').reshape(n,2,128));v=affine(normalized,'blk.0.attn_v').reshape(n,2,128)
        attended=np.empty((n,12,128),dtype=np.float32);scale=np.float32(1)/np.sqrt(np.float32(128))
        for row in range(n):
            for head in range(12):
                group=head//6;scores=(q[row,head][None,:]*k[:row+1,group]).sum(axis=-1,dtype=np.float64).astype(np.float32)*scale
                probabilities=exp(scores-scores.max());probabilities*=np.float32(1./probabilities.sum(dtype=np.float64))
                attended[row,head]=(probabilities[:,None]*v[:row+1,group]).sum(axis=0,dtype=np.float64).astype(np.float32)
        h=x+matrix(attended.reshape(n,1536),'blk.0.attn_output.weight');normalized=norm(h,'blk.0.ffn_norm.weight',epsilon)
        gate=matrix(normalized,'blk.0.ffn_gate.weight');up=matrix(normalized,'blk.0.ffn_up.weight');hidden=(gate/(np.float32(1)+exp(-gate)))*up
        output=h+matrix(hidden,'blk.0.ffn_down.weight');actual=output[-1]
        parent='.cache/vlm-language-reference' if epsilon==0 else '.cache/qwen-language-reference-v2'
        path=ROOT/parent/'left-red/batch_3_l_out-0.f32';expected=np.fromfile(path,dtype='<f4');diff=np.abs(actual.astype(np.float64)-expected)
        outside=int((diff>0.003+0.001*np.abs(expected.astype(np.float64))).sum());l2=float(np.linalg.norm(diff)/np.linalg.norm(expected.astype(np.float64)))
        destination=ROOT/'.cache/m2-offline'/('independent-'+kind+'.f32');actual.astype('<f4').tofile(destination)
        item={'profile':kind,'epsilon':epsilon,'count':len(actual),'independent_sha256':sha(destination),'reference_sha256':sha(path),'maximum_absolute':float(diff.max()),'outside_tolerance':outside,'normalized_l2':l2,'elapsed_seconds':time.monotonic()-start}
        reports.append(item);print(json.dumps(item),flush=True);assert outside==0 and l2<=0.0005,'independent layer math mismatch'
    report={'schema_version':1,'scope':'Independent first decoder layer under both original epsilon-zero and metadata-correct profiles; identical weights, token inputs, frozen image embeddings and positions. This does not replace native end-to-end PNG execution.','numpy_version':np.__version__,'source_sha256':sha(Path(__file__)),'math_source_sha256':sha(ROOT/'resources/tool-assessment/qwen_scalar_math.c'),'base_math_source_sha256':sha(ROOT/'resources/tool-assessment/scalar_math.c'),'math_binary_sha256':sha(ROOT/'.cache/image-oracle/qwen-scalar-math.dylib'),'language_sha256':case['identity']['language_sha256'],'tolerances':{'absolute':0.003,'relative':0.001,'normalized_l2':0.0005},'comparisons':reports}
    (ROOT/'fixtures/vlm/qwen2-language-independent-v2.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()

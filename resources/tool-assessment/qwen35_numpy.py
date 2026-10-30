"""Independent array-math assessment; never imported by native builds or tests.

Implements the published recurrent equations, independently of Coheryn's Rust code.
Uses identical BF16 source weights with a declared F32 activation / wide-sum profile.
This is not a capture from the publisher's full framework execution.
"""
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for relative in ('.cache/vision-numpy', '.cache/devtools', '.cache/q35-assessment-pip'):
    sys.path.insert(0, str(ROOT / relative))
import numpy as np
from jinja2.sandbox import SandboxedEnvironment
from tokenizers import Tokenizer

SOURCE = ROOT / '.cache/qwen3.5-0.8b-source'
OUT = ROOT / '.cache/q35-reference'
OUT.mkdir(exist_ok=True)
manifest = json.loads((ROOT / 'fixtures/qwen35/model.json').read_text())
for entry in manifest['files']:
    data = SOURCE / entry['file']
    digest = hashlib.file_digest(data.open('rb'), 'sha256').hexdigest()
    if data.stat().st_size != entry['bytes'] or digest != entry['sha256']:
        raise ValueError('source identity mismatch: ' + entry['file'])
weights = SOURCE / 'model.safetensors-00001-of-00001.safetensors'
with weights.open('rb') as f:
    header_size = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(header_size))
raw = np.memmap(weights, mode='r', dtype=np.uint8)
c = json.loads((SOURCE / 'config.json').read_text())['text_config']
F = np.float32

def weight(name, start=None, end=None):
    h = header[name]
    shape = tuple(h['shape'])
    dtype = {'BF16':'<u2', 'F32':'<f4'}[h['dtype']]
    view = raw[8+header_size+h['data_offsets'][0]:8+header_size+h['data_offsets'][1]].view(dtype).reshape(shape)
    if start is not None:
        view = view[start:end]
    if h['dtype'] == 'BF16':
        return (view.astype(np.uint32) << 16).view(np.float32)
    return np.array(view, dtype=np.float32)

def project(name, x):
    rows = header[name]['shape'][0]
    result = np.empty(rows, dtype=np.float32)
    for start in range(0, rows, 64):
        w = weight(name, start, min(rows, start+64))
        result[start:start+len(w)] = np.sum(w*x, axis=-1, dtype=np.float64).astype(np.float32)
    return result

def sigmoid(x):
    # Stable scalar/vector expression, including both tails.
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, F(1)/(F(1)+e), e/(F(1)+e)).astype(np.float32)

def silu(x):
    return x*sigmoid(x)

def rms(x, w, centered=True):
    mean = np.mean(x*x, axis=-1, keepdims=True, dtype=np.float64).astype(np.float32)
    return (x*(F(1)/np.sqrt(mean+F(c['rms_norm_eps']))))*(w+F(1) if centered else w)

def rope(x, position):
    d = int(c['head_dim']*c['rope_parameters']['partial_rotary_factor'])
    half = d//2
    inv = np.power(F(c['rope_parameters']['rope_theta']), -np.arange(0,d,2,dtype=np.float32)/F(d))
    angle = F(position)*inv
    a, b = x[:, :half].copy(), x[:, half:d].copy()
    x[:, :half] = a*np.cos(angle)-b*np.sin(angle)
    x[:, half:d] = a*np.sin(angle)+b*np.cos(angle)
    return x

class Reference:
    def __init__(self):
        self.position = 0
        self.state = {}
    def forward(self, token, logits=False):
        hidden = weight('model.language_model.embed_tokens.weight', token, token+1)[0]
        layers = []
        for i, kind in enumerate(c['layer_types']):
            p = f'model.language_model.layers.{i}'
            x = rms(hidden, weight(p+'.input_layernorm.weight'))
            if kind == 'linear_attention':
                p += '.linear_attn'
                nk, nv = c['linear_num_key_heads'], c['linear_num_value_heads']
                dk, dv = c['linear_key_head_dim'], c['linear_value_head_dim']
                channels = 2*nk*dk+nv*dv
                if i not in self.state:
                    self.state[i] = (np.zeros((channels,c['linear_conv_kernel_dim']),dtype=np.float32), np.zeros((nv,dk,dv),dtype=np.float32))
                conv, state = self.state[i]
                conv[:, :-1] = conv[:, 1:]
                conv[:, -1] = project(p+'.in_proj_qkv.weight', x)
                mixed = silu(np.sum(conv*weight(p+'.conv1d.weight')[:,0,:], axis=-1,dtype=np.float64).astype(np.float32))
                q = mixed[:nk*dk].reshape(nk,dk)
                k = mixed[nk*dk:2*nk*dk].reshape(nk,dk)
                value = mixed[2*nk*dk:].reshape(nv,dv)
                q = q/(np.sqrt(np.sum(q*q,axis=-1,keepdims=True,dtype=np.float64).astype(np.float32)+F(1e-6))*F(np.sqrt(F(dk))))
                k = k/np.sqrt(np.sum(k*k,axis=-1,keepdims=True,dtype=np.float64).astype(np.float32)+F(1e-6))
                q, k = np.repeat(q,nv//nk,axis=0), np.repeat(k,nv//nk,axis=0)
                a = project(p+'.in_proj_a.weight', x)+weight(p+'.dt_bias')
                decay = np.exp(-np.exp(weight(p+'.A_log'))*(np.maximum(a,F(0))+np.log1p(np.exp(-np.abs(a)))))
                beta = sigmoid(project(p+'.in_proj_b.weight', x))
                state *= decay[:,None,None]
                prediction = np.sum(state*k[:,:,None],axis=1,dtype=np.float64).astype(np.float32)
                correction = (value-prediction)*beta[:,None]
                state += k[:,:,None]*correction[:,None,:]
                y = np.sum(state*q[:,:,None],axis=1,dtype=np.float64).astype(np.float32)
                y = rms(y,weight(p+'.norm.weight'),False)*silu(project(p+'.in_proj_z.weight',x).reshape(nv,dv))
                y = project(p+'.out_proj.weight', y.ravel())
            else:
                p += '.self_attn'
                nh, nk, d = c['num_attention_heads'],c['num_key_value_heads'],c['head_dim']
                qg = project(p+'.q_proj.weight', x).reshape(nh,2*d)
                q = rope(rms(qg[:,:d],weight(p+'.q_norm.weight')), self.position)
                k = rope(rms(project(p+'.k_proj.weight',x).reshape(nk,d),weight(p+'.k_norm.weight')), self.position)
                v = project(p+'.v_proj.weight',x).reshape(nk,d)
                if i not in self.state:
                    self.state[i] = ([],[])
                self.state[i][0].append(k)
                self.state[i][1].append(v)
                keys, values = np.array(self.state[i][0]), np.array(self.state[i][1])
                y = np.empty((nh,d),dtype=np.float32)
                for h in range(nh):
                    g=h//(nh//nk)
                    scores = np.sum(keys[:,g,:]*q[h],axis=-1,dtype=np.float64).astype(np.float32)/F(np.sqrt(F(d)))
                    probabilities = np.exp(scores-np.max(scores))
                    probabilities /= np.sum(probabilities,dtype=np.float32)
                    y[h] = np.sum(probabilities[:,None]*values[:,g,:],axis=0,dtype=np.float64).astype(np.float32)
                y *= sigmoid(qg[:,d:])
                y = project(p+'.o_proj.weight', y.ravel())
            hidden = hidden+y
            p = f'model.language_model.layers.{i}'
            x = rms(hidden,weight(p+'.post_attention_layernorm.weight'))
            hidden += project(p+'.mlp.down_proj.weight',silu(project(p+'.mlp.gate_proj.weight',x))*project(p+'.mlp.up_proj.weight',x))
            layers.append(hidden.copy())
        self.position += 1
        result = project('model.language_model.embed_tokens.weight',rms(hidden,weight('model.language_model.norm.weight'))) if logits else None
        return layers,result

def summarize(name, array):
    data = array.astype('<f4').tobytes()
    (OUT / (name+'.f32')).write_bytes(data)
    indices = sorted(set([0,len(array)-1,*range(0,len(array),max(1,len(array)//64))]))
    return {'name':name,'length':len(array),'sha256':hashlib.sha256(data).hexdigest(),'indices':indices,'values':[float(array[i]) for i in indices], 'norm_l2':float(np.linalg.norm(array.astype(np.float64)))}

tokenizer = Tokenizer.from_file(str(SOURCE/'tokenizer.json'))
env = SandboxedEnvironment()
template = env.from_string((SOURCE/'chat_template.jinja').read_text())
cases = [
    {'id':'english','messages':[{'role':'user','content':'Say hello.'}]},
    {'id':'unicode','messages':[{'role':'system','content':'Reply briefly.'},{'role':'user','content':'你好! বাংলা 😊'}]},
]
begin = time.monotonic()
for case in cases:
    prompt = template.render(messages=case['messages'],tools=[],add_generation_prompt=True,enable_thinking=False,add_vision_id=False)
    ids = tokenizer.encode(prompt,add_special_tokens=False).ids
    case.update(prompt=prompt,prompt_token_ids=ids)
    reference = Reference()
    for position,token in enumerate(ids):
        layers, logits = reference.forward(token, position==len(ids)-1)
        print(case['id'], 'prefill',position+1,'/',len(ids),flush=True)
    case['checkpoints'] = [summarize(f"{case['id']}-prompt-layer.{i}",v) for i,v in enumerate(layers)]
    generated=[]
    case['teacher_forced'] = []
    for position in range(4):
        case['teacher_forced'].append(summarize(f"{case['id']}-logits-{position}",logits))
        token=int(np.argmax(logits));generated.append(token)
        if token in (248044,248046):break
        if position+1<4:
            _,logits=reference.forward(token,True)
        print(case['id'],'decode',position+1, token,flush=True)
    case['greedy_token_ids']=generated
    case['greedy_text']=tokenizer.decode(generated,skip_special_tokens=True)
    print(case['id'],repr(case['greedy_text']),flush=True)
report={'schema_version':1,'profile':'independent NumPy equations, source BF16 / F32 activations / wide products; not publisher framework parity','model_revision':manifest['revision'],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'architecture_source':json.loads((SOURCE/'architecture-source.json').read_text()),'numpy':np.__version__,'cases':cases,'wall_seconds':time.monotonic()-begin,'tolerances':manifest['oracle']['tolerances']}
(ROOT/'fixtures/qwen35/reference-v1.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('Reference captured',flush=True)

"""Independent Qwen3.5 still-image equations. Inert assessment outside native tools.

Original array arithmetic; F32 activations/products, wide reductions, BF16 source.
Does not execute the publisher framework or import any native Coheryn execution code.
"""
import ast,hashlib,json,math,os,struct,sys,zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for relative in ('.cache/vision-numpy','.cache/devtools','.cache/q35-assessment-pip'):
    sys.path.insert(0,str(ROOT/relative))
import numpy as np
from tokenizers import Tokenizer
from jinja2.sandbox import SandboxedEnvironment
F=np.float32
SOURCE=ROOT/'.cache/qwen3.5-0.8b-source'
OUT=ROOT/'.cache/q35-vision-reference';OUT.mkdir(exist_ok=True)
CONTRACT=ROOT/'fixtures/qwen35/vision/contract-v1.json'
manifest=json.loads((ROOT/'fixtures/qwen35/model.json').read_text())
for item in manifest['files']:
    path=SOURCE/item['file']
    assert path.stat().st_size==item['bytes'] and hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==item['sha256']
weights=SOURCE/'model.safetensors-00001-of-00001.safetensors'
with weights.open('rb') as f:
    header_size=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(header_size))
raw=np.memmap(weights,mode='r',dtype=np.uint8)
config=json.loads((SOURCE/'config.json').read_text());c=config['text_config'];vc=config['vision_config']
# Reuse the authored, independent text equations without their capture entrypoint.
source=(ROOT/'resources/tool-assessment/qwen35_numpy.py').read_text()
source=source.replace('def forward(self, token, logits=False):','def forward(self, token, logits=False, embedding=None, coordinate=None):')
source=source.replace("hidden = weight('model.language_model.embed_tokens.weight', token, token+1)[0]","hidden = weight('model.language_model.embed_tokens.weight', token, token+1)[0] if embedding is None else np.array(embedding,dtype=np.float32)")
source=source.replace('self.position)','self.position if coordinate is None else coordinate)')
module=ast.parse(source)
selected=[node for node in module.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in ('weight','project','sigmoid','silu','rms','Reference')]
exec(compile(ast.Module(body=selected,type_ignores=[]),'independent-text-equations','exec'),globals())
def rope(x,position):
    dim=int(c['head_dim']*c['rope_parameters']['partial_rotary_factor']);half=dim//2
    axes=position if isinstance(position,tuple) else (position,)*3
    positions=np.full(half,axes[0],dtype=np.float32)
    for axis,offset in ((1,1),(2,2)):
        positions[offset:c['rope_parameters']['mrope_section'][axis]*3:3]=axes[axis]
    angle=positions*np.power(F(c['rope_parameters']['rope_theta']),-np.arange(0,dim,2,dtype=np.float32)/F(dim))
    a,b=x[:,:half].copy(),x[:,half:dim].copy()
    x[:,:half]=a*np.cos(angle)-b*np.sin(angle);x[:,half:dim]=a*np.sin(angle)+b*np.cos(angle)
    return x

def project_many(name,x):
    shape=header[name]['shape'];rows=shape[0];width=math.prod(shape[1:]);result=np.empty((len(x),rows),dtype=np.float32)
    for out in range(0,rows,16):
        w=weight(name,out,min(rows,out+16)).reshape(-1,width)
        for start in range(0,len(x),8):
            products=x[start:start+8,None,:]*w[None,:,:]
            result[start:start+8,out:out+len(w)]=np.sum(products,axis=-1,dtype=np.float64).astype(np.float32)
    bias=name.removesuffix('.weight')+'.bias'
    if bias in header:result+=weight(bias)
    return result

def norm(x,prefix):
    mean=np.sum(x,axis=-1,keepdims=True,dtype=np.float64).astype(np.float32)/F(x.shape[-1])
    centered=x-mean
    variance=np.mean(centered*centered,axis=-1,keepdims=True,dtype=np.float64).astype(np.float32)
    return (centered*(F(1)/np.sqrt(variance+F(1e-6))))*weight(prefix+'.weight')+weight(prefix+'.bias')
def gelu(x,approximate=True):
    if approximate:return F(.5)*x*(F(1)+np.tanh(F(.7978846)*x*(F(1)+F(.044715)*x*x)))
    erf=np.fromiter((math.erf(float(v)/math.sqrt(2)) for v in x.flat),dtype=np.float32,count=x.size).reshape(x.shape)
    return F(.5)*x*(F(1)+erf)
def save(name,value):
    value=np.asarray(value,dtype='<f4');data=value.tobytes();path=OUT/(name+'.f32');path.write_bytes(data)
    return {'name':name,'shape':list(value.shape),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
def encode(name,rgb,grid):
    gh,gw=grid;h=vc['hidden_size'];heads=vc['num_heads'];dim=h//heads;patch=vc['patch_size'];rows=gh*gw
    coordinates=[(y*2+dy,x*2+dx) for y in range(gh//2) for x in range(gw//2) for dy in range(2) for dx in range(2)]
    pixel=(np.asarray(rgb,dtype=np.float32)/F(255)-F(.5))/F(.5)
    patches=np.tile(np.repeat(pixel,2*patch*patch),(rows,1))
    checkpoints=[save(name+'-patches',patches)]
    x=project_many('model.visual.patch_embed.proj.weight',patches)
    table=weight('model.visual.pos_embed.weight');side=math.isqrt(len(table))
    for i,(y,col) in enumerate(coordinates):
        fy=F(y*(side-1)/(gh-1));fx=F(col*(side-1)/(gw-1));y0=int(fy);x0=int(fx);y1=min(y0+1,side-1);x1=min(x0+1,side-1)
        dy=fy-F(y0);dx=fx-F(x0)
        terms=[table[y0*side+x0]*((F(1)-dy)*(F(1)-dx)),table[y0*side+x1]*((F(1)-dy)*dx),table[y1*side+x0]*(dy*(F(1)-dx)),table[y1*side+x1]*(dy*dx)]
        x[i]+=np.sum(terms,axis=0,dtype=np.float64).astype(np.float32)
    checkpoints.append(save(name+'-embedding',x))
    inv=np.power(F(10000),-np.arange(0,dim//2,2,dtype=np.float32)/F(dim//2))
    angles=np.stack([np.concatenate((F(y)*inv,F(col)*inv)) for y,col in coordinates]);cs=np.cos(angles);sn=np.sin(angles)
    for i in range(vc['depth']):
        p=f'model.visual.blocks.{i}';qkv=project_many(p+'.attn.qkv.weight',norm(x,p+'.norm1')).reshape(rows,3,heads,dim)
        q,k,v=[qkv[:,j].copy() for j in range(3)]
        for value in (q,k):
            a,b=value[:,:,:dim//2].copy(),value[:,:,dim//2:].copy();value[:,:,:dim//2]=a*cs[:,None,:]-b*sn[:,None,:];value[:,:,dim//2:]=a*sn[:,None,:]+b*cs[:,None,:]
        attended=np.empty_like(q)
        for row in range(rows):
            for head in range(heads):
                scores=np.sum(k[:,head]*q[row,head],axis=-1,dtype=np.float64).astype(np.float32)/F(math.sqrt(dim))
                probabilities=np.exp(scores-scores.max());probabilities/=F(np.sum(probabilities,dtype=np.float64))
                attended[row,head]=np.sum(v[:,head]*probabilities[:,None],axis=0,dtype=np.float64).astype(np.float32)
        x+=project_many(p+'.attn.proj.weight',attended.reshape(rows,h))
        x+=project_many(p+'.mlp.linear_fc2.weight',gelu(project_many(p+'.mlp.linear_fc1.weight',norm(x,p+'.norm2'))))
        checkpoints.append(save(name+f'-layer.{i}',x));print(name,'encoder layer',i,flush=True)
    merged=norm(x,'model.visual.merger.norm').reshape(-1,4*h)
    merged=project_many('model.visual.merger.linear_fc2.weight',gelu(project_many('model.visual.merger.linear_fc1.weight',merged),False))
    checkpoints.append(save(name+'-merged',merged));return merged,checkpoints

def main():
    contract=json.loads(CONTRACT.read_text());target=ROOT/'fixtures/qwen35/vision/reference-v1.json'
    if target.exists():raise FileExistsError(target)
    images={};arrays=[]
    for image in contract['images']:
        merged,capture=encode(image['id'],image['rgb'],image['grid']);images[image['id']]=merged;arrays+=capture
    tokenizer=Tokenizer.from_file(str(SOURCE/'tokenizer.json'));template=SandboxedEnvironment().from_string((SOURCE/'chat_template.jinja').read_text());cases=[]
    for case in contract['cases']:
        content=[{'type':'image'} for _ in case['images']]+[{'type':'text','text':case['text']}];messages=[{'role':'user','content':content}]
        prompt=template.render(messages=messages,tools=[],add_generation_prompt=True,enable_thinking=False,add_vision_id=False)
        original=tokenizer.encode(prompt,add_special_tokens=False).ids;ids=[];entries=[];position=0;image_cursor=0
        for token in original:
            if token==config['image_token_id']:
                key=case['images'][image_cursor];image=next(i for i in contract['images'] if i['id']==key);gh,gw=image['grid'];features=images[key]
                for j,embedding in enumerate(features):
                    ids.append(token);entries.append((embedding,(position,position+j//(gw//2),position+j%(gw//2))))
                position+=max(gh//2,gw//2);image_cursor+=1
            else:ids.append(token);entries.append((None,(position,)*3));position+=1
        reference=Reference()
        for index,(token,(embedding,coordinate)) in enumerate(zip(ids,entries)):
            layers,logits=reference.forward(token,index+1==len(ids),embedding,coordinate)
            if index%16==0:print(case['id'],'language',index+1,'/',len(ids),flush=True)
        checkpoints=[save(case['id']+f'-language.{i}',v) for i,v in enumerate(layers)];generated=[];teacher=[]
        for step in range(contract['max_tokens']):
            teacher.append(save(case['id']+f'-logits.{step}',logits));token=int(np.argmax(logits));generated.append(token)
            if token in (248044,248046):break
            if step+1<contract['max_tokens']:
                _,logits=reference.forward(token,True,None,(position,)*3);position+=1
        cases.append({**case,'prompt':prompt,'prompt_token_ids':ids,'checkpoints':checkpoints,'teacher_forced':teacher,'greedy_token_ids':generated,'greedy_text':tokenizer.decode(generated,skip_special_tokens=True)})
        print('PASS capture',case['id'],repr(cases[-1]['greedy_text']),flush=True)
    target.write_text(json.dumps({'schema_version':1,'contract_sha256':hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'model_manifest_sha256':hashlib.sha256((ROOT/'fixtures/qwen35/model.json').read_bytes()).hexdigest(),'profile':'Independent array equations, F32 activations/products, wide reductions, BF16 weights; not publisher framework parity','numpy':np.__version__,'encoder_checkpoints':arrays,'cases':cases},ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()

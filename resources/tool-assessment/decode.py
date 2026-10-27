"""Separate research decoder; never invoked by native tests/build/runtime."""
import struct,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
import hashlib
model=ROOT/'.cache/models/qwen2.5-0.5b-instruct-q4_k_m.gguf'
with model.open('rb') as source:assert hashlib.file_digest(source,'sha256').hexdigest()=='74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db'
f=model.open('rb');f.read(16);n=struct.unpack('<Q',f.read(8))[0]
def val(t):
 if t==8:return f.read(struct.unpack('<Q',f.read(8))[0]).decode()
 if t==9:
  typ,n=struct.unpack('<IQ',f.read(12));return [val(typ) for _ in range(n)]
 s={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}[t];return struct.unpack('<'+s,f.read(struct.calcsize(s)))[0]
meta={}
for _ in range(n):
 key=val(8);meta[key]=val(struct.unpack('<I',f.read(4))[0])
a=list(range(33,127))+list(range(161,173))+list(range(174,256)); chars=a[:];b=a[:];extra=0
for i in range(256):
 if i not in a:chars.append(256+extra);b.append(i);extra+=1
dec=dict(zip(map(chr,chars),b));v=json.loads((ROOT/'fixtures/tools/qwen2.5-reference.json').read_text())
for c in v['cases']:
 if 'generated_token_ids' in c:
  parts=[]
  for i in c['generated_token_ids']:
   if i in [meta['tokenizer.ggml.eos_token_id'],151643,151645]:continue
   token=meta['tokenizer.ggml.tokens'][i]
   parts.append(bytes(dec[ch] for ch in token))
  c['generated_text']=b''.join(parts).decode();print(c['generated_text'])
(ROOT/'fixtures/tools/qwen2.5-reference.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

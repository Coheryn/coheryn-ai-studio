"""Independent scalar candidate capture; never part of the native execution chain."""
import json,math,struct
from pathlib import Path
f=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
def dot(a,b):return f(sum(f(x*y) for x,y in zip(a,b)))
heads,key,value=2,3,2
state=[0.]*(heads*key*value);steps=[]
for t in range(3):
 q=[f(((i+2*t)%7-3)/4) for i in range(heads*key)]
 k=[f(((i*2+t)%9-4)/8) for i in range(heads*key)]
 v=[f(((i+t)%5-2)/2) for i in range(heads*value)]
 g=[f(-math.log(2)),f(-math.log(4))];beta=[.25,.75];out=[]
 for h in range(heads):
  qh=q[h*key:(h+1)*key];kh=k[h*key:(h+1)*key]
  qs=f(f(1/math.sqrt(f(dot(qh,qh)+f(1e-6))))/f(math.sqrt(key)))
  ks=f(1/math.sqrt(f(dot(kh,kh)+f(1e-6))))
  dec=f(math.exp(g[h]));base=h*key*value
  for j in range(value):
   for i in range(key):state[base+i*value+j]=f(state[base+i*value+j]*dec)
   old=f(sum(f(state[base+i*value+j]*f(kh[i]*ks)) for i in range(key)))
   delta=f(f(v[h*value+j]-old)*beta[h])
   for i in range(key):state[base+i*value+j]=f(state[base+i*value+j]+f(f(kh[i]*ks)*delta))
   out.append(f(sum(f(state[base+i*value+j]*f(qh[i]*qs)) for i in range(key))))
 steps.append(dict(q=q,k=k,v=v,log_decay=g,beta=beta,output=out,state=list(state)))
report=dict(schema_version=1,scope='Authored rational scalar fixtures; independently evaluated binary32 gated delta equations, not Coheryn-generated expected output.',precision='binary32 state/products, wide reference reductions',absolute_tolerance=0.000002,heads=heads,key_dimension=key,value_dimension=value,steps=steps)
(Path(__file__).resolve().parents[2]/'fixtures/qwen35/operations-v1.json').write_text(json.dumps(report,indent=2)+'\n')

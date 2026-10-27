"""Separate research capture. Never invoked by native build/test/runtime tools."""
import hashlib,json,os,struct,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.cache/devtools'))
import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment
pin=next(x for x in json.loads((ROOT/'fixtures/quantized/models.json').read_text())['models'] if x['id']=='qwen2.5-0.5b-instruct-q4_k_m')
model=ROOT/'.cache/models'/pin['file']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(model)==pin['sha256']
template=(ROOT/'fixtures/tools/qwen2.5-tools-v1.jinja').read_text()
env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
env.filters['tojson']=lambda obj:json.dumps(obj,ensure_ascii=False,sort_keys=True)
tool={'type':'function','function':{'name':'add','description':'Add two integers.','parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},'required':['a','b'],'additionalProperties':False}}}
def msg(role,content,**kw):return {'role':role,'content':content,'tool_calls':[],**kw}
cases=[{'id':'add','messages':[msg('user','Use add to calculate 2 + 2.')],'tools':[tool]},
 {'id':'unicode-system','messages':[msg('system','বাংলা 日本語 🌍'),msg('user','Use add for 3 + 4.')],'tools':[tool]},
 {'id':'parallel-results','messages':[msg('user','Add twice.'),msg('assistant','',tool_calls=[{'id':'call-a','type':'function','function':{'name':'add','arguments':{'a':2,'b':2}}},{'id':'call-b','type':'function','function':{'name':'add','arguments':{'a':3,'b':4}}}]),msg('tool','4',tool_call_id='call-a'),msg('tool','7',tool_call_id='call-b')],'tools':[tool]}]
for case in cases:
 case['messages']=json.loads(json.dumps(case['messages'],sort_keys=True));case['tools']=json.loads(json.dumps(case['tools'],sort_keys=True))
 for m in case['messages']:
  for c in m.get('tool_calls',[]):
   if isinstance(c['function']['arguments'],dict):c['function']['arguments']=json.dumps(c['function']['arguments'],ensure_ascii=False,sort_keys=True)
 case['prompt']=env.from_string(template).render(messages=case['messages'],tools=case['tools'],add_generation_prompt=True)
 path=ROOT/'.cache/s4-prompt.txt';path.write_text(case['prompt'])
 with (ROOT/'.cache/s4-reference.log').open('a') as log:
  raw=subprocess.check_output([str(ROOT/'.cache/reference-oracle'),str(model),'tokenize',str(path)],stderr=log)
 case['prompt_token_ids']=json.loads(raw)['token_ids']
 if case['id']=='add':
  with (ROOT/'.cache/s4-reference.log').open('a') as log:
   raw=subprocess.check_output([str(ROOT/'.cache/reference-oracle'),str(model),'generate',str(path),'96'],stderr=log)
  case['generated_token_ids']=json.loads(raw)['token_ids']
report={'schema_version':1,'model':pin,'template_sha256':sha(ROOT/'fixtures/tools/qwen2.5-tools-v1.jinja'),'template_fingerprint':hashlib.sha256(json.dumps(template,ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),'oracle_revision':(ROOT/'.cache/reference/source-revision').read_text().strip(),'oracle_binary_sha256':sha(ROOT/'.cache/reference-oracle'),'template_reference':{'implementation':'Jinja2','version':jinja2.__version__,'json_profile':'sorted keys, UTF-8, default Python separators; recursively canonical input maps'},'settings':{'context':512,'threads':1,'cache_dtype':'f32','temperature':0,'seed':0,'max_tokens':96},'cases':cases,'acceptance':{'rendered_bytes':'exact','prompt_token_ids':'exact','greedy_token_ids':'exact','call_ids':'native unique IDs checked separately; not part of model text'},'production_dependency':False,'tool_template_profile':'qwen-tools-v1; explicit inert companion corrects the double-braced schematic example; original GGUF text template unchanged'}
(ROOT/'fixtures/tools/qwen2.5-reference.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('Captured independent tool prompt/token/greedy fixtures.',flush=True)

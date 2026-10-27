"""Capture a tool-result continuation independently, preserving all prior cases."""
import json,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'.cache/devtools'))
from jinja2.sandbox import ImmutableSandboxedEnvironment
path=ROOT/'fixtures/tools/qwen2.5-reference.json';v=json.loads(path.read_text());source=next(c for c in v['cases'] if c['id']=='parallel-add');raw=source['generated_text']
objects=[json.loads(s) for s in re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>',raw,re.S)]
assert len(objects)==2
calls=[{'id':f'fixture-{i}','type':'function','function':obj}for i,obj in enumerate(objects)]
messages=source['messages']+[{'role':'assistant','content':re.sub(r'<tool_call>.*?</tool_call>','',raw,flags=re.S),'tool_calls':calls}]+[{'role':'tool','content':str(obj['arguments']['a']+obj['arguments']['b']),'tool_call_id':f'fixture-{i}','tool_calls':[]}for i,obj in enumerate(objects)]
for m in messages:
 for c in m.get('tool_calls',[]):
  if isinstance(c['function']['arguments'],dict):c['function']['arguments']=json.dumps(c['function']['arguments'],ensure_ascii=False,sort_keys=True)
case={'id':'parallel-add-results','messages':messages,'tools':source['tools'],'max_tokens':1}
env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True);env.filters['tojson']=lambda obj:json.dumps(obj,ensure_ascii=False,sort_keys=True)
case['prompt']=env.from_string((ROOT/'fixtures/tools/qwen2.5-tools-v1.jinja').read_text()).render(messages=messages,tools=case['tools'],add_generation_prompt=True)
prompt=ROOT/'.cache/s4-results-prompt.txt';prompt.write_text(case['prompt'])
with (ROOT/'.cache/s4-results-reference.log').open('w')as log:
 for mode in ['tokenize','generate']:
  raw=subprocess.check_output([str(ROOT/'.cache/reference-oracle'),str(ROOT/'.cache/models'/v['model']['file']),mode,str(prompt),'1'],stderr=log)
  case['prompt_token_ids' if mode=='tokenize' else 'generated_token_ids']=json.loads(raw)['token_ids']
v['cases'].append(case);path.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');print('Captured independent tool-result continuation.')

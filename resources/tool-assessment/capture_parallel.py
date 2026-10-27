"""Append an independently generated multiple-call case; preserve existing expected cases."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'.cache/devtools'))
from jinja2.sandbox import ImmutableSandboxedEnvironment
path=ROOT/'fixtures/tools/qwen2.5-reference.json';v=json.loads(path.read_text());before=v['cases'][:]
case={'id':'parallel-add','messages':[{'role':'user','content':'Use add twice: calculate 2 + 2, then 3 + 4.','tool_calls':[]}],'tools':v['cases'][0]['tools']}
env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True);env.filters['tojson']=lambda obj:json.dumps(obj,ensure_ascii=False,sort_keys=True)
case['prompt']=env.from_string((ROOT/'fixtures/tools/qwen2.5-tools-v1.jinja').read_text()).render(messages=case['messages'],tools=case['tools'],add_generation_prompt=True)
prompt=ROOT/'.cache/s4-parallel-prompt.txt';prompt.write_text(case['prompt'])
with (ROOT/'.cache/s4-parallel-reference.log').open('w')as log:
 for mode in ['tokenize','generate']:
  raw=subprocess.check_output([str(ROOT/'.cache/reference-oracle'),str(ROOT/'.cache/models'/v['model']['file']),mode,str(prompt),'96'],stderr=log)
  case['prompt_token_ids' if mode=='tokenize' else 'generated_token_ids']=json.loads(raw)['token_ids']
v['cases'].append(case);assert v['cases'][:-1]==before
path.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');print('Appended independent parallel-call case.')

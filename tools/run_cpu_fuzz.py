#!/usr/bin/env python3
"""Run bounded, instrumented native parser campaigns and retain exact evidence."""
import argparse,hashlib,json,os,platform,re,subprocess,struct,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TARGETS=('gguf','safetensors','images','tools','constraints')
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def corpus(target):
    path=ROOT/'fuzz/corpus'/target;path.mkdir(parents=True,exist_ok=True);seeds=[b'',b'{}',b'[]',b'\xff'*16]
    if target=='gguf':
        def string(x):return struct.pack('<Q',len(x))+x
        meta=string(b'general.architecture')+struct.pack('<I',8)+string(b'qwen2')
        seeds += [b'GGUF'+struct.pack('<IQQ',3,0,1)+meta]
    elif target=='safetensors':
        for shape,payload in [([],b'\0'*4),([2],b'\0'*8),([0],b'')]:
            header=json.dumps({'x':{'dtype':'F32','shape':shape,'data_offsets':[0,len(payload)]}},separators=(',',':')).encode();seeds.append(struct.pack('<Q',len(header))+header+payload)
        seeds += [b'{"x":1,"x":2}',b'{"a":[1,2,3]}']
    elif target=='images':seeds += [p.read_bytes() for p in sorted((ROOT/'fixtures/vlm/images').glob('*')) if p.suffix in ('.png','.jpg')]+[(ROOT/'fixtures/qwen35/vision/red.png').read_bytes()]
    elif target=='tools':seeds += [b'{"tools":[{"type":"function","function":{"name":"add","parameters":{"type":"object"}}}]}',b'<tool_call>{"name":"add","arguments":{}}</tool_call>']
    else:seeds += [b'{"schema":{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]},"output":"{\\"x\\":1}"}']
    for seed in seeds:
        if seed:(path/hashlib.sha256(seed).hexdigest()).write_bytes(seed)
    return path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--toolchain',required=True);p.add_argument('--runs',type=int,default=100000);p.add_argument('--target',action='append',choices=TARGETS);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    if a.report.exists():raise FileExistsError(a.report)
    if a.runs<1 or not re.fullmatch(r'nightly-\d{4}-\d{2}-\d{2}',a.toolchain):p.error('use a pinned nightly and positive iteration count')
    out=ROOT/'.cache/cpu-fuzz'/a.report.stem;out.mkdir(parents=True,exist_ok=True);cargo=str(ROOT/'tools/cargo');env={**os.environ,'CARGO_NET_OFFLINE':'true','CARGO_BUILD_JOBS':'2','CARGO_INCREMENTAL':'0'}
    local_rust=ROOT/'.cache/cargo/bin/rustc'
    rust_command=str(local_rust) if local_rust.exists() else shutil.which('rustc')
    rust_env={**env,'RUSTUP_HOME':str(ROOT/'.cache/rustup')} if local_rust.exists() else env
    lock=sha(ROOT/'fuzz/Cargo.lock');rust=subprocess.check_output([rust_command,'+'+a.toolchain,'-vV'],cwd=ROOT,env=rust_env,text=True)
    sources=list((ROOT/'crates/coheryn-engine/src').rglob('*.rs'))+list((ROOT/'fuzz/fuzz_targets').glob('*.rs'))
    snapshot={str(s.relative_to(ROOT)):sha(s) for s in sorted(sources)}
    runs=[]
    for target in a.target or TARGETS:
        seed_dir=corpus(target);seeds={x.name:sha(x) for x in seed_dir.iterdir() if x.is_file()}
        log=out/(target+'.log');command=[cargo,'+'+a.toolchain,'fuzz','run','--sanitizer','address',target,str(seed_dir),'--',f'-runs={a.runs}','-seed=1337','-max_len=65536','-timeout=10','-rss_limit_mb=1024','-malloc_limit_mb=128']
        with log.open('xb') as output:result=subprocess.run(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
        content=log.read_text(errors='replace');done=re.findall(r'#(\d+)\s+DONE\s+cov:\s*(\d+)\s+ft:\s*(\d+)',content)
        valid=result.returncode==0 and bool(done) and int(done[-1][0])>=a.runs and int(done[-1][1])>0 and bool(re.search(r'(inline 8-bit counters|PC tables|PCs)',content))
        runs.append({'target':target,'exit':result.returncode,'outcome':'pass' if valid else 'fail','requested_iterations':a.runs,'executed_iterations':int(done[-1][0]) if done else None,'edges':int(done[-1][1]) if done else None,'features':int(done[-1][2]) if done else None,'log_sha256':sha(log),'seeds':seeds,'command':command})
        print(('PASS' if valid else 'FAIL'),target,flush=True)
        if not valid:break
    if sha(ROOT/'fuzz/Cargo.lock')!=lock:raise ValueError('fuzz lock changed during campaign')
    unchanged=snapshot=={str(s.relative_to(ROOT)):sha(s) for s in sorted(sources)}
    report={'schema_version':1,'kind':'bounded-native-fuzz','scope':'Current-host AddressSanitizer and edge-feedback campaigns; iteration counts and coverage are observations, not proof of absence of bugs or a sustained release campaign.','outcome':'pass' if unchanged and len(runs)==len(a.target or TARGETS) and all(r['outcome']=='pass' for r in runs) else 'fail','source_unchanged':unchanged,'host':{'system':platform.system(),'machine':platform.machine()},'rust':rust,'toolchain':a.toolchain,'cargo_fuzz':subprocess.check_output([cargo,'+'+a.toolchain,'fuzz','--version'],env=env,cwd=ROOT,text=True).strip(),'fuzz_lock_sha256':lock,'source_snapshot':snapshot,'runs':runs}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+'\n')
    if report['outcome']!='pass':raise SystemExit(1)
if __name__=='__main__':main()

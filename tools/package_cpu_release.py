#!/usr/bin/env python3
"""Build a deterministic, offline CPU candidate archive with complete legal payload."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tarfile
import tempfile
import tomllib
from check_repository_policy import ROOT, load_policy, scan
import license_notices as legal
from check_cpu_release import COMMANDS, bound_file, source_identity

TARGETS=('aarch64-apple-darwin','x86_64-unknown-linux-gnu','x86_64-pc-windows-msvc')
DOCS=('README.md',)
def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def regular(path):
    path=Path(path)
    if not path.is_file() or any(p.is_symlink() for p in (path,*path.parents)):raise ValueError('missing or linked package input: '+str(path))
    return path

def executable_target(path,target):
    with regular(path).open('rb') as f:
        header=f.read(4096)
    if target=='aarch64-apple-darwin':valid=header[:4]==b'\xcf\xfa\xed\xfe' and len(header)>=8 and struct.unpack_from('<I',header,4)[0]==0x0100000c
    elif target=='x86_64-unknown-linux-gnu':valid=header[:6]==b'\x7fELF\x02\x01' and len(header)>=20 and struct.unpack_from('<H',header,18)[0]==62
    else:
        offset=struct.unpack_from('<I',header,60)[0] if len(header)>=64 else 0
        valid=header[:2]==b'MZ' and offset>=64 and offset+6<=len(header) and header[offset:offset+4]==b'PE\0\0' and struct.unpack_from('<H',header,offset+4)[0]==0x8664
    if not valid:raise ValueError('executable does not match declared CPU target: '+str(path))

def archive_bytes(root,prefix):
    raw=io.BytesIO()
    with tarfile.open(fileobj=raw,mode='w',format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(root.rglob('*')):
            if path.is_dir():continue
            regular(path);relative=path.relative_to(root).as_posix();data=path.read_bytes();info=tarfile.TarInfo(prefix+'/'+relative)
            info.size=len(data);info.mode=0o755 if relative.startswith('bin/') else 0o644;info.mtime=0;info.uid=info.gid=0;info.uname=info.gname='';tar.addfile(info,io.BytesIO(data))
    out=io.BytesIO()
    with gzip.GzipFile(fileobj=out,mode='wb',filename='',mtime=0,compresslevel=9) as stream:stream.write(raw.getvalue())
    return out.getvalue()

def build(binary_dir,output,target,build_record):
    if target not in TARGETS:raise ValueError('unsupported release target')
    record=json.loads(regular(build_record).read_text())
    snapshot=source_identity()
    if (record.get('kind')!='cpu-native-checks' or record.get('outcome')!='pass'
            or record.get('source_unchanged') is not True or record.get('target')!=target
            or record.get('source_snapshot')!=snapshot
            or [tuple(c['command']) for c in record['commands']]!=list(COMMANDS)
            or any(c['exit']!=0 or c.get('validated') is not True for c in record['commands'])):
        raise ValueError('requires passing native checks from the current source and target')
    for command in record['commands']:bound_file(command['log'],command['log_sha256'])
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('linked package output')
    output=output.resolve();boundary=(ROOT/'target').resolve()
    if not output.is_relative_to(boundary):raise ValueError('package output must be an unlinked target/ directory')
    version=tomllib.loads((ROOT/'Cargo.toml').read_text())['workspace']['package']['version']
    if not re.fullmatch(r'[0-9A-Za-z.+-]+',version):raise ValueError('invalid package version')
    prefix=f'coheryn-ai-studio-{version}-{target}-cpu-candidate';destination=output/(prefix+'.tar.gz')
    sidecar=destination.with_suffix(destination.suffix+'.sha256')
    if destination.exists() or sidecar.exists():raise FileExistsError(destination)
    legal.check();output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='cpu-package-',dir=boundary) as temporary:
        stage=Path(temporary);suffix='.exe' if target.endswith('msvc') else ''
        for name in ['coheryn','coheryn-bench']:
            source=regular(binary_dir/(name+suffix));executable_target(source,target)
            if record['binaries'].get(source.name)!={'sha256':digest(source),'bytes':source.stat().st_size}:raise ValueError('binary changed after native checks')
            path=stage/'bin'/source.name;path.parent.mkdir(exist_ok=True);shutil.copyfile(source,path)
        documents=list(DOCS)+[p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'docs').rglob('*.md'))]+['fixtures/compatibility.json','fixtures/cpu-release/scope.json']
        for relative in documents:
            source=regular(ROOT/relative);path=stage/relative;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,path)
        legal.bundle(stage)
        files=[{'path':p.relative_to(stage).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(stage.rglob('*')) if p.is_file()]
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        dirty=bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=normal'],cwd=ROOT,text=True))
        manifest={'schema_version':1,'classification':'cpu-candidate','production_ready':False,'version':version,'target':target,'source_commit':source_commit,'source_dirty':dirty,'cargo_lock_sha256':digest(ROOT/'Cargo.lock'),'rust_toolchain_sha256':digest(ROOT/'rust-toolchain.toml'),'models_bundled':False,'files':files,'notes':['Deterministic archive for identical inputs; this does not establish reproducible compiler output.','Checksums verify integrity; signing and platform/model qualification remain separate release gates.']}
        manifest['native_check_record_sha256']=digest(build_record)
        manifest['source_snapshot_sha256']=hashlib.sha256(json.dumps(snapshot,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        manifest['compiler']=record['compiler']
        (stage/'PACKAGE.json').write_text(json.dumps(manifest,indent=2)+'\n')
        if source_identity()!=snapshot:raise ValueError('source changed while staging candidate')
        payload=scan(stage,load_policy(),artifact=True)
        if payload['issues']:raise ValueError('controlled package policy failed; preserve original legal notices and inspect source policy')
        archive=archive_bytes(stage,prefix)
        if archive!=archive_bytes(stage,prefix):raise ValueError('package serialization is not reproducible')
        # Exclusive publication. Only remove files created by this invocation.
        checksum=hashlib.sha256(archive).hexdigest()
        created=[]
        try:
            with destination.open('xb') as f:
                created.append(destination);f.write(archive)
            with sidecar.open('x') as f:
                created.append(sidecar);f.write(checksum+'  '+destination.name+'\n')
        except BaseException:
            for path in created:path.unlink(missing_ok=True)
            raise
        return {'schema_version':1,'archive':destination.name,'sha256':checksum,'bytes':len(archive),'manifest':manifest,'policy_findings':0,'archive_reproducible':True,'platform_qualification':'pending; executable format only is checked'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--target',choices=TARGETS,required=True);p.add_argument('--binary-dir',type=Path,default=ROOT/'target/release');p.add_argument('--output',type=Path,default=ROOT/'target/cpu-release');p.add_argument('--build-record',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    if a.report.exists():raise FileExistsError(a.report)
    result=build(a.binary_dir,a.output,a.target,a.build_record);a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2)+'\n');print('PASS CPU candidate package:',result['archive'])
if __name__=='__main__':main()

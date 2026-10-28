#!/usr/bin/env python3
"""Create an isolated reference library with the declared Qwen RMSNorm epsilon.

Reuse hashed unchanged objects, replacing only the unity object containing the
one-line metadata-loader correction. Never mutate the original sources/builds.
"""
import hashlib,json,shlex,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    source=ROOT/'.cache/vision-corrected-source'
    build=ROOT/'.cache/vision-corrected-build'
    destination=ROOT/'.cache/qwen-language-qualified'
    binary=destination/'bin';binary.mkdir(parents=True,exist_ok=True)
    source_path=source/'src/models/qwen2vl.cpp';original=source_path.read_text()
    assert sha(source_path)=='2d00d03c371b87e158f95d69a89bf91cfbcbfa066c8d07f3738e264b3fe919c0', 'unexpected pinned Qwen source'
    expected='    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, true);'
    assert original.count(expected)==1
    corrected=original.replace(expected,'    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);\n'+expected)
    patched=destination/'qwen2vl.cpp';patched.write_text(corrected)
    unit=next(p for p in (build/'src/CMakeFiles/llama.dir/Unity').glob('*.cxx') if str(source_path) in p.read_text())
    unity=destination/unit.name;unity.write_text(unit.read_text().replace(str(source_path),str(patched)))
    commands=json.loads((build/'compile_commands.json').read_text())
    command=next(c for c in commands if Path(c['file'])==unit)
    args=shlex.split(command['command']);object_path=destination/'qwen-unity.o'
    args[args.index('-o')+1]=str(object_path);args[args.index('-c')+1]=str(unity)
    args.append('-I'+str(source/'src/models'))
    subprocess.run(args,cwd=command['directory'],check=True)
    link=shlex.split((build/'src/CMakeFiles/llama.dir/link.txt').read_text())
    old_object=str(unit.relative_to(build/'src'))+'.o'
    assert old_object in link
    objects={arg:sha(build/'src'/arg) for arg in link if arg.endswith('.o')}
    link[link.index(old_object)]=str(object_path)
    link[link.index('-o')+1]=str(binary/'libllama.0.7.0.dylib')
    for path in (build/'bin').iterdir():
        if path.is_file() and not path.is_symlink():shutil.copyfile(path,binary/path.name)
    subprocess.run(link,cwd=build/'src',check=True)
    for path in (build/'bin').iterdir():
        if path.is_symlink():
            target=binary/path.name
            if target.is_symlink():target.unlink()
            target.symlink_to(path.readlink())
    record={'schema_version':1,'scope':'Isolated assessment correction; no production dependency or change to original reference files.','reason':'The selected Qwen2-VL loader reads rotary sections but omits the GGUF RMSNorm epsilon. Original execution reports zero; the GGUF and source configuration declare approximately 1e-6.','base_source_record':'resources/tool-assessment/vision-corrected-source-record.json','base_source_record_sha256':sha(ROOT/'resources/tool-assessment/vision-corrected-source-record.json'),'changed_source':'src/models/qwen2vl.cpp','source_before_sha256':sha(source_path),'source_after_sha256':sha(patched),'correction':'Read LLM_KV_ATTENTION_LAYERNORM_RMS_EPS into hparams.f_norm_rms_eps before rotary sections. No graph or weight changes.','unity_before_sha256':sha(unit),'unity_after_sha256':sha(unity),'reused_object_sha256':objects,'replacement_object_sha256':sha(object_path),'library_sha256':sha(binary/'libllama.0.7.0.dylib'),'compile_argv':args,'link_argv':link,'preparation_source_sha256':sha(Path(__file__))}
    (ROOT/'resources/tool-assessment/qwen-language-source-record.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Prepared isolated Qwen language library; original source/build files unchanged',flush=True)
if __name__=='__main__':main()

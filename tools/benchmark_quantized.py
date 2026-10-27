#!/usr/bin/env python3
"""Measure pinned packed CPU profiles serially; never infer a cold cache or speed advantage.

Run after qualification/build jobs finish. Each child is a fresh process so observed
session peaks are not inherited from a different execution profile. Unix only.
"""
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from fetch_model import digest


def main():
    binary = ROOT / 'target/release/coheryn'
    manifest = json.loads((ROOT / 'fixtures/quantized/models.json').read_text())
    model = next(m for m in manifest['models'] if m['id'] == 'qwen2.5-0.5b-instruct-q4_k_m')
    path = ROOT / '.cache/models' / model['file']
    if path.stat().st_size != model['bytes'] or digest(path) != model['sha256']:
        raise SystemExit('Model byte verification failed')
    reference = json.loads((ROOT / model['evidence']).read_text())
    expected = next(c for c in reference['cases'] if c['id'] == 'english')['generated_token_ids']
    compiler = subprocess.check_output([ROOT / 'tools/cargo', '--version'], text=True).strip()
    rust_environment = os.environ.copy()
    rustc = shutil.which('rustc')
    if (ROOT / '.cache/cargo/bin/rustc').exists():
        rustc = str(ROOT / '.cache/cargo/bin/rustc')
        rust_environment['RUSTUP_HOME'] = str(ROOT / '.cache/rustup')
        rust_environment['CARGO_HOME'] = str(ROOT / '.cache/cargo')
    rust_version = subprocess.check_output([rustc, '--version'], env=rust_environment, text=True).strip() if rustc else None
    report = {'schema_version': 1, 'model_sha256': model['sha256'],
              'native_executable_sha256': digest(binary), 'cargo': compiler, 'compiler': rust_version,
              'platform': platform.platform(), 'hardware': {}, 'cache_condition': 'uncontrolled',
              'backend': 'cpu-reference', 'kv_dtype': 'F32', 'concurrency': 1, 'runs': [],
              'measurement_limits': ['Engine buffer capacities exclude worker stacks, libraries and allocator overhead',
                                     'Output delivery measures a no-op engine sink; final JSON transport is excluded',
                                     'CPU affinity, OS cache residency and background workload are uncontrolled'],
              'note': 'Short prompt; context is reservation capacity. Ordered local samples are not controlled performance superiority evidence.'}
    for key in ['hw.model', 'hw.memsize', 'hw.ncpu', 'machdep.cpu.brand_string']:
        value = subprocess.run(['sysctl', '-n', key], capture_output=True, text=True)
        report['hardware'][key] = value.stdout.strip() if value.returncode == 0 else None
    for context in [128, 512, 4096]:
        for profile, threads in [('reference', 1), ('balanced', 2), ('performance', 4), ('low-memory', 1)]:
            command = [str(binary), '--json', 'bench', '--model', str(path), '--standalone',
                       '--prompt', 'Hello', '--max-tokens', '8', '--iterations', '2',
                       '--context', str(context), '--profile', profile, '--threads', str(threads)]
            started = time.monotonic()
            with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
                child = subprocess.Popen(command, stdout=output, stderr=errors)
                _, status, usage = os.wait4(child.pid, 0)
                child.returncode = os.waitstatus_to_exitcode(status)
                if child.returncode:
                    errors.seek(0)
                    raise RuntimeError(errors.read(8192).decode())
                output.seek(0)
                result = json.load(output)
            for run in result['runs']:
                assert run['token_ids'] == expected, 'Profile changed qualified output'
                assert run['memory']['session_buffer_capacity_bytes'] == 0, 'Leaked session buffers'
                assert run['memory']['peak_reserved'] <= run['memory']['limit']
            entry = {'context': context, 'profile': profile, 'threads': threads, 'result': result,
                     'wall_ms': (time.monotonic() - started) * 1000,
                     'peak_rss_bytes': usage.ru_maxrss if sys.platform == 'darwin' else usage.ru_maxrss * 1024,
                     'minor_page_faults': usage.ru_minflt, 'major_page_faults': usage.ru_majflt,
                     'input_blocks': usage.ru_inblock, 'output_blocks': usage.ru_oublock,
                     'private_memory_bytes': None, 'gpu_memory_bytes': None,
                     'mapped_weight_bytes': 0,
                     'measurement': 'wait4 of this benchmark child; block counters are not byte counts'}
            report['runs'].append(entry)
            print(f'PASS benchmark context={context} {profile} threads={threads}', flush=True)
    (ROOT / 'fixtures/qualification/g4-benchmarks.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

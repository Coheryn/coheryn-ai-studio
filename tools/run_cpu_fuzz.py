#!/usr/bin/env python3
"""Run host-guarded, bounded sanitizer campaigns and retain partial evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import struct
import time
import tomllib

from check_cpu_release import ROOT, sha, source_identity
from cpu_qualification.host import Guard
from run_cpu_qualification import checkpoint
from run_cpu_soak import run_child

TARGETS = ('gguf', 'safetensors', 'images', 'tools', 'constraints')
CARGO_FUZZ = '0.13.2'  # Latest stable, non-yanked release verified in the upstream registry.
MAX_CORPUS_BYTES = 64 << 20
MAX_CORPUS_FILES = 10000


def corpus(target, guard):
    """Add immutable seeds without following links or overwriting existing bytes."""
    path = ROOT / 'fuzz/corpus' / target
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('linked fuzz corpus directory')
    path.mkdir(parents=True, exist_ok=True)
    seeds = [b'{}', b'[]', b'\xff' * 16]
    if target == 'gguf':
        def string(value): return struct.pack('<Q', len(value)) + value
        metadata = string(b'general.architecture') + struct.pack('<I', 8) + string(b'qwen2')
        seeds.append(b'GGUF' + struct.pack('<IQQ', 3, 0, 1) + metadata)
    elif target == 'safetensors':
        for shape, payload in [([], b'\0' * 4), ([2], b'\0' * 8), ([0], b'')]:
            header = json.dumps({'x': {'dtype': 'F32', 'shape': shape, 'data_offsets': [0, len(payload)]}},
                                separators=(',', ':')).encode()
            seeds.append(struct.pack('<Q', len(header)) + header + payload)
        seeds += [b'{"x":1,"x":2}', b'{"a":[1,2,3]}']
    elif target == 'images':
        images = [p for p in sorted((ROOT / 'fixtures/vlm/images').glob('*')) if p.suffix in ('.png', '.jpg')]
        images.append(ROOT / 'fixtures/qwen35/vision/red.png')
        for image in images:
            guard.check()
            if not image.is_file() or image.is_symlink() or image.stat().st_size > 1 << 20:
                raise ValueError('missing, linked or oversized image fuzz seed')
            seeds.append(image.read_bytes())
    elif target == 'tools':
        seeds += [b'{"tools":[{"type":"function","function":{"name":"add","parameters":{"type":"object"}}}]}',
                  b'<tool_call>{"name":"add","arguments":{}}</tool_call>']
    else:
        seeds.append(b'{"schema":{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]},"output":"{\\"x\\":1}"}')
    # Reject an existing hostile corpus before adding anything or launching a fuzzer.
    inventory(path, guard)
    for seed in seeds:
        guard.check()
        file = path / hashlib.sha256(seed).hexdigest()
        if file.exists():
            if file.is_symlink() or not file.is_file() or file.read_bytes() != seed:
                raise ValueError('changed or linked immutable fuzz seed')
        else:
            with file.open('xb') as stream: stream.write(seed)
    return path, inventory(path, guard)


def inventory(path, guard):
    result, total = {}, 0
    for file in path.iterdir():
        guard.check()
        if file.is_symlink() or not file.is_file():
            raise ValueError('fuzz corpus must contain regular, unlinked files')
        total += file.stat().st_size
        if total > MAX_CORPUS_BYTES or len(result) >= MAX_CORPUS_FILES:
            raise ValueError('fuzz corpus exceeds input inventory limits')
        result[file.name] = sha(file)
    return dict(sorted(result.items()))


def coverage(log, requested):
    if log.stat().st_size > 64 << 20:
        raise ValueError('fuzz diagnostic log exceeds limit')
    content = log.read_text(encoding='utf-8', errors='replace')
    done = re.findall(r'^#(\d+)\s+DONE\s+cov:\s*(\d+)\s+ft:\s*(\d+)', content, re.M)
    if (len(done) != 1 or int(done[0][0]) < requested or int(done[0][1]) <= 0 or int(done[0][2]) <= 0
            or not re.search(r'(inline 8-bit counters|PC tables|PCs)', content)):
        raise ValueError('missing or incomplete instrumented fuzz completion')
    return dict(zip(('executed_iterations', 'edges', 'features'), map(int, done[0])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--toolchain', required=True)
    parser.add_argument('--runs', type=int, default=100000)
    parser.add_argument('--target', action='append', choices=TARGETS)
    parser.add_argument('--prepare', action='store_true',
                        help='explicitly allow guarded tool installation and locked dependency acquisition')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    targets = args.target or list(TARGETS)
    if args.report.exists(): parser.error('use a fresh fuzz report path')
    if (not 1 <= args.runs <= 100000000 or not re.fullmatch(r'nightly-\d{4}-\d{2}-\d{2}', args.toolchain)
            or len(targets) != len(set(targets))):
        parser.error('use a pinned nightly, bounded positive iteration count and unique targets')
    if platform.system() not in ('Darwin', 'Linux'):
        parser.error('instrumented CPU fuzzing requires native macOS or Linux')
    def interrupted(_signal, _frame): raise KeyboardInterrupt('fuzz campaign interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    directory = ROOT / '.cache/cpu-fuzz' / args.report.stem
    directory.mkdir(parents=True, exist_ok=False)
    source = source_identity()
    environment = {**os.environ, 'CARGO_NET_OFFLINE': 'true', 'CARGO_BUILD_JOBS': '1', 'CARGO_INCREMENTAL': '0'}
    local = ROOT / '.cache/cargo/bin'
    if (local / 'cargo').exists():
        environment.update(CARGO_HOME=str(ROOT / '.cache/cargo'), RUSTUP_HOME=str(ROOT / '.cache/rustup'),
                           PATH=str(local) + os.pathsep + environment.get('PATH', ''))
    cargo = str(ROOT / 'tools/cargo')
    rustc = str(local / 'rustc') if (local / 'rustc').exists() else shutil.which('rustc')
    rustup = str(local / 'rustup') if (local / 'rustup').exists() else shutil.which('rustup')
    result = {'schema_version': 1, 'kind': 'bounded-native-fuzz', 'outcome': 'running', 'production_ready': False,
              'scope': 'Native AddressSanitizer/edge-feedback campaigns under the host guard; no sustained-security or production qualification.',
              'host': {'system': platform.system(), 'machine': platform.machine()},
              'toolchain': args.toolchain, 'cargo_fuzz': None, 'rust': None,
              'fuzz_lock_sha256': sha(ROOT / 'fuzz/Cargo.lock'), 'source_snapshot': source,
              'source_unchanged': False, 'requested_targets': targets, 'commands': [], 'runs': [],
              'execution_settings': {'cargo_build_jobs': 1, 'cargo_incremental': 0, 'prepare': args.prepare},
              'ci': {key: os.environ[key] for key in ('GITHUB_SHA', 'GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT') if key in os.environ}}
    checkpoint(args.report, result)
    guard, children = None, []
    started = time.monotonic()

    def execute(command, name, *, online=False, timeout=3600):
        log = directory / (name + '.log')
        record = {'command': command, 'exit': None, 'validated': False,
                  'log': log.relative_to(ROOT).as_posix(), 'timeout_seconds': timeout}
        result['commands'].append(record)
        checkpoint(args.report, result)
        try:
            record['exit'] = run_child(command, ROOT,
                {**environment, 'CARGO_NET_OFFLINE': 'false'} if online else environment,
                log, timeout, guard, children.append)
            record['validated'] = record['exit'] == 0
        finally:
            if log.exists(): record['log_sha256'] = sha(log)
            checkpoint(args.report, result)
        if not record['validated']:
            raise ValueError('fuzz command failed: ' + name)
        return log

    try:
        guard = Guard(directory)
        guard.start()
        if environment.get('CARGO_BUILD_TARGET'):
            raise ValueError('unset CARGO_BUILD_TARGET; sanitizer campaigns must execute natively')
        if not rustc or (args.prepare and not rustup):
            raise ValueError('native Rust toolchain manager/compiler not found')
        if args.prepare:
            stable = tomllib.loads((ROOT / 'rust-toolchain.toml').read_text())['toolchain']['channel']
            execute([rustup, 'toolchain', 'install', stable, '--profile', 'minimal'], 'prepare-stable', online=True)
            execute([rustup, 'toolchain', 'install', args.toolchain, '--profile', 'minimal'], 'prepare-nightly', online=True)
            execute([cargo, 'install', 'cargo-fuzz', '--version', CARGO_FUZZ, '--locked'], 'prepare-fuzzer', online=True)
            execute([cargo, 'fetch', '--manifest-path', 'fuzz/Cargo.toml', '--locked'], 'prepare-dependencies', online=True)
        compiler_log = execute([rustc, '+' + args.toolchain, '-vV'], 'rustc', timeout=30)
        fuzz_log = execute([cargo, '+' + args.toolchain, 'fuzz', '--version'], 'cargo-fuzz', timeout=30)
        if compiler_log.stat().st_size > 64 << 10 or fuzz_log.stat().st_size > 64 << 10:
            raise ValueError('fuzz tool identity exceeds limit')
        result['rust'] = compiler_log.read_text(encoding='utf-8')
        result['cargo_fuzz'] = fuzz_log.read_text(encoding='utf-8').strip()
        if '-nightly' not in result['rust'] or result['cargo_fuzz'] != 'cargo-fuzz ' + CARGO_FUZZ:
            raise ValueError('fuzz toolchain identity differs from the required tools')
        for target in targets:
            seed_dir, seeds = corpus(target, guard)
            command = [cargo, '+' + args.toolchain, 'fuzz', 'run', '--sanitizer', 'address', target, str(seed_dir), '--',
                       f'-runs={args.runs}', '-seed=1337', '-max_len=65536', '-timeout=10', '-rss_limit_mb=1024', '-malloc_limit_mb=128']
            row = {'target': target, 'outcome': 'running', 'requested_iterations': args.runs,
                   'executed_iterations': None, 'edges': None, 'features': None, 'seeds': seeds, 'command': command}
            result['runs'].append(row)
            checkpoint(args.report, result)
            try:
                log = execute(command, target)
                row.update(coverage(log, args.runs))
                row['final_corpus'] = inventory(seed_dir, guard)
                row['outcome'] = 'pass'
            finally:
                record = result['commands'][-1]
                row.update({key: record[key] for key in ('exit', 'log', 'log_sha256') if key in record})
                if row['outcome'] != 'pass': row['outcome'] = 'fail'
                checkpoint(args.report, result)
            print('PASS', target, flush=True)
        result['outcome'] = 'pass'
    except (Exception, KeyboardInterrupt) as error:
        result['outcome'], result['error'] = 'fail', str(error)
    finally:
        if guard:
            try: guard.close()
            except Exception as error: result['outcome'], result['observer_close_error'] = 'fail', str(error)
            result['host_guard'] = guard.report()
            if guard.stream.closed: result['host_guard']['log_sha256'] = sha(guard.log)
            if guard.reason or guard.stop_errors: result['outcome'] = 'fail'
        try:
            result['source_unchanged'] = source_identity() == source
        except Exception as error:
            result['source_unchanged'], result['source_observation_error'] = False, str(error)
        result['owned_cleanup_confirmed'] = all(p.poll() is not None for p in children) and not (guard and guard.stop_errors)
        if not result['source_unchanged'] or not result['owned_cleanup_confirmed']: result['outcome'] = 'fail'
        result['elapsed_seconds'] = time.monotonic() - started
        checkpoint(args.report, result)
    print(result['outcome'].upper(), 'bounded native fuzz:', args.report, flush=True)
    raise SystemExit(result['outcome'] != 'pass')


if __name__ == '__main__':
    main()

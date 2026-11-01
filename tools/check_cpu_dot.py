#!/usr/bin/env python3
"""Guarded, source-bound synthetic CPU dot measurements; no model qualification."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

from check_cpu_release import ROOT, sha, source_identity
from cpu_qualification.host import Guard
from run_cpu_qualification import checkpoint
from run_cpu_soak import run_child
from verify_cpu_package import executable_target, native_target

BUILD = ('build', '--release', '--workspace', '--example', 'dot_bench', '--locked')
CASES = tuple((width, False) for width in (256, 384, 512, 768, 896, 1024, 1536, 3072,
                                        3584, 4864, 8960, 16384, 32768, 65536)) + ((1024, True),)


def observations(path):
    with path.open('rb') as stream:
        data = stream.read((1 << 20) + 1)
    if len(data) > 1 << 20:
        raise ValueError('kernel observations exceed limit')
    result = json.loads(data)
    if (not isinstance(result, dict) or result.get('schema_version') != 1
            or result.get('kind') != 'cpu-dot-observations' or result.get('outcome') != 'pass'
            or type(result.get('threads')) is not int or result['threads'] != 1
            or result.get('seed') != '57024f001e5b90e3'
            or not isinstance(result.get('cases'), list) or len(result['cases']) != len(CASES)):
        raise ValueError('incomplete kernel observations')
    for row, (width, cancellation) in zip(result['cases'], CASES):
        if (not isinstance(row, dict) or type(row.get('width')) is not int or row['width'] != width
                or type(row.get('cancellation')) is not bool or row['cancellation'] != cancellation
                or row.get('rows') != 64 or row.get('repeats') != 128 or row.get('samples') != 7
                or row.get('bitwise_parity') is not True):
            raise ValueError('kernel case identity or parity changed')
        for field in ('reference_ms', 'candidate_ms'):
            samples = row.get(field)
            if (not isinstance(samples, list) or len(samples) != 7
                    or any(type(x) not in (float, int) or not math.isfinite(x) or x <= 0 for x in samples)):
                raise ValueError('invalid or incomplete kernel timing samples')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error('use a fresh kernel report path')
    if os.name != 'nt':
        def interrupted(_signal, _frame): raise KeyboardInterrupt('kernel observation interrupted')
        signal.signal(signal.SIGTERM, interrupted)
    target = native_target()
    if target is None:
        parser.error('unsupported native kernel target')
    directory = ROOT / '.cache/cpu-dot-bench' / args.report.stem
    directory.mkdir(parents=True, exist_ok=False)
    source = source_identity()
    result = {'schema_version': 1, 'kind': 'cpu-kernel-observations', 'outcome': 'running',
              'production_ready': False, 'target': target, 'source_snapshot': source,
              'commands': [], 'scope': 'Synthetic, same-process alternating kernel timings with bitwise reference parity. '
                  'No model-throughput, performance-SLA, complete numerical or production qualification.'}
    result['ci'] = {key: os.environ[key] for key in ('GITHUB_REPOSITORY', 'GITHUB_SHA', 'GITHUB_RUN_ID',
                                                   'GITHUB_RUN_ATTEMPT', 'GITHUB_REF') if key in os.environ}
    guard, children = None, []
    started = time.monotonic()
    checkpoint(args.report, result)
    try:
        guard = Guard(directory)
        guard.start()
        if os.environ.get('CARGO_BUILD_TARGET'):
            raise ValueError('unset CARGO_BUILD_TARGET; kernel checks must execute natively')
        environment = {**os.environ, 'CARGO_BUILD_JOBS': '1', 'CARGO_INCREMENTAL': '0',
                       'CARGO_TARGET_DIR': str(ROOT / 'target'), 'CARGO_NET_OFFLINE': 'true'}
        cargo = str(ROOT / 'tools/cargo') if os.name != 'nt' else 'cargo'
        local_rust = ROOT / '.cache/cargo/bin/rustc'
        compiler = str(local_rust) if os.name != 'nt' and local_rust.exists() else 'rustc'
        compiler_env = {**environment, 'RUSTUP_HOME': str(ROOT / '.cache/rustup')} if compiler == str(local_rust) else environment
        result['toolchain'] = {}
        for name, command, env in [('cargo', [cargo, '--version'], environment), ('rustc', [compiler, '-vV'], compiler_env)]:
            log = directory / (name + '.log')
            code = run_child(command, ROOT, env, log, 30, guard, children.append)
            if code or log.stat().st_size > 64 << 10:
                raise ValueError('kernel compiler identity probe failed')
            result['toolchain'][name] = {'output': log.read_text(encoding='utf-8'),
                                         'log': log.relative_to(ROOT).as_posix(), 'log_sha256': sha(log)}
        binary = ROOT / 'target/release/examples' / ('dot_bench.exe' if target.endswith('msvc') else 'dot_bench')
        for command, log_name, timeout in [([cargo, *BUILD], 'build.log', 3600), ([str(binary)], 'native.json', 300)]:
            log = directory / log_name
            record = {'command': command, 'exit': None, 'log': log.relative_to(ROOT).as_posix()}
            result['commands'].append(record)
            checkpoint(args.report, result)
            try:
                record['exit'] = run_child(command, ROOT, environment, log, timeout, guard, children.append)
            finally:
                if log.exists(): record['log_sha256'] = sha(log)
                checkpoint(args.report, result)
            if record['exit'] != 0:
                raise ValueError('native kernel command failed: ' + log_name)
            if log_name == 'build.log':
                if (not binary.is_file() or any(p.is_symlink() for p in (binary, *binary.parents))
                        or not 0 < binary.stat().st_size <= 64 << 20):
                    raise ValueError('missing, linked or oversized kernel executable')
                executable_target(binary, target)
                result['binary_sha256'] = sha(binary)
        result['observations'] = observations(directory / 'native.json')
        if sha(binary) != result['binary_sha256']:
            raise ValueError('kernel executable changed during measurement')
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
        result['owned_cleanup_confirmed'] = all(p.poll() is not None for p in children) and not (guard and guard.stop_errors)
        result['source_unchanged'] = source_identity() == source
        if not result['source_unchanged'] or not result['owned_cleanup_confirmed']: result['outcome'] = 'fail'
        result['elapsed_seconds'] = time.monotonic() - started
        checkpoint(args.report, result)
    print(result['outcome'].upper(), 'CPU kernel observations:', args.report)
    raise SystemExit(result['outcome'] != 'pass')


if __name__ == '__main__':
    main()

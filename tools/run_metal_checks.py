#!/usr/bin/env python3
"""Guarded, source-bound optional Metal checks; never promotes a model release."""
import argparse
import os
from pathlib import Path
import platform
import signal
import sys
import time

from check_cpu_release import ROOT, sha, source_identity
from cpu_qualification.driver import test_counts
from cpu_qualification.host import Guard
from run_cpu_qualification import checkpoint
from run_cpu_soak import run_child


def commands(gpu):
    result = [
        ('cargo', 'fmt', '--all', '--', '--check'),
        ('cargo', 'clippy', '--release', '-p', 'coheryn-metal', '-p', 'coheryn-engine',
         '--features', 'coheryn-engine/metal,coheryn-metal/metal', '--all-targets', '--locked', '--', '-D', 'warnings'),
        ('cargo', 'test', '--profile', 'qualification', '-p', 'coheryn-metal', '--features', 'metal', '--lib', '--locked'),
        ('cargo', 'test', '--profile', 'qualification', '-p', 'coheryn-engine', '--features', 'metal', '--lib', '--locked'),
    ]
    if gpu:
        result.append(('cargo', 'test', '--profile', 'qualification', '-p', 'coheryn-metal',
                       '--features', 'metal', '--test', 'dense', '--locked', '--', '--ignored', '--nocapture'))
    return tuple(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--gpu', action='store_true', help='require real GPU numerical execution; absent devices fail')
    args = parser.parse_args()
    if args.report.exists():
        parser.error('use a fresh Metal check report path')
    if platform.system() != 'Darwin' or platform.machine().lower() not in ('arm64', 'aarch64'):
        parser.error('Metal qualification requires a native Apple Silicon Mac')
    if os.environ.get('CARGO_BUILD_TARGET'):
        parser.error('unset CARGO_BUILD_TARGET; qualification must execute natively')
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt('Metal checks interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    directory = ROOT / '.cache/metal-checks' / args.report.stem
    directory.mkdir(parents=True, exist_ok=False)
    source = source_identity()
    report = {'schema_version': 1, 'kind': 'metal-primitive-checks', 'outcome': 'running',
              'production_ready': False, 'gpu_execution_required': args.gpu,
              'source_snapshot': source, 'source_unchanged': False, 'commands': [],
              'fixture_sha256': sha(ROOT / 'fixtures/gpu/metal-dense-v1.json'),
              'scope': 'Optional backend compile/lint and independent dense primitives only; no model, packed-kernel, performance or release qualification.',
              'host': {'system': platform.system(), 'machine': platform.machine(), 'release': platform.release()}}
    environment = {**os.environ, 'CARGO_BUILD_JOBS': '1', 'CARGO_INCREMENTAL': '0',
                   'RUST_TEST_THREADS': '1', 'CARGO_NET_OFFLINE': 'true',
                   'CARGO_TARGET_DIR': str(ROOT / 'target'),
                   'MTL_DEBUG_LAYER': '1', 'MTL_SHADER_VALIDATION': '1'}
    cargo = str(ROOT / 'tools/cargo') if (ROOT / '.cache/cargo/bin/cargo').exists() else 'cargo'
    guard, children = None, []
    started = time.monotonic()
    checkpoint(args.report, report)
    try:
        guard = Guard(directory)
        guard.start()
        for number, command in enumerate(commands(args.gpu)):
            log = directory / f'{number}.log'
            row = {'command': list(command), 'exit': None, 'validated': False, 'log': log.relative_to(ROOT).as_posix()}
            report['commands'].append(row)
            checkpoint(args.report, report)
            try:
                row['exit'] = run_child([cargo, *command[1:]], ROOT, environment, log, 3600, guard, children.append)
            finally:
                if log.exists(): row['log_sha256'] = sha(log)
                checkpoint(args.report, report)
            valid = row['exit'] == 0
            if command[1] == 'test':
                row['tests'] = counts = test_counts(log)
                valid = valid and counts['passed'] > 0 and counts['failed'] == counts['filtered'] == 0
                if '--test' in command:
                    valid = valid and counts['passed'] == counts['suites'] == 1 and counts['ignored'] == 0
            row['validated'] = valid
            checkpoint(args.report, report)
            print(('PASS' if valid else 'FAIL') + ' ' + ' '.join(command), flush=True)
            if not valid:
                raise ValueError('Metal check failed: ' + ' '.join(command))
        report['outcome'] = 'pass'
    except (Exception, KeyboardInterrupt) as error:
        report['outcome'], report['error'] = 'fail', str(error)
    finally:
        if guard:
            try: guard.close()
            except Exception as error: report['outcome'], report['observer_close_error'] = 'fail', str(error)
            report['host_guard'] = guard.report()
            if guard.stream.closed: report['host_guard']['log_sha256'] = sha(guard.log)
            if guard.reason or guard.stop_errors: report['outcome'] = 'fail'
        report['source_unchanged'] = source_identity() == source
        report['owned_cleanup_confirmed'] = all(p.poll() is not None for p in children) and not (guard and guard.stop_errors)
        if not report['source_unchanged'] or not report['owned_cleanup_confirmed']: report['outcome'] = 'fail'
        report['elapsed_seconds'] = time.monotonic() - started
        checkpoint(args.report, report)
    print(report['outcome'].upper(), 'Metal primitive checks:', args.report, flush=True)
    raise SystemExit(report['outcome'] != 'pass')


if __name__ == '__main__':
    main()

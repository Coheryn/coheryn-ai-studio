#!/usr/bin/env python3
"""Guarded, source-bound Rust checks for the mixed CPU soak driver."""
import argparse
import os
from pathlib import Path
import platform
import re
import signal
import sys
import time

from check_cpu_release import ROOT, sha, source_identity
from cpu_qualification.host import Guard
from run_cpu_qualification import checkpoint
from run_cpu_soak import run_child
from verify_cpu_package import native_target

COMMANDS = (
    ('cargo', 'test', '--release', '--workspace', '--example', 'cpu_soak', '--locked'),
    ('cargo', 'clippy', '--release', '--workspace', '--example', 'cpu_soak', '--tests', '--locked', '--', '-D', 'warnings'),
)
FETCH_COMMANDS = (
    ('cargo', 'fetch', '--locked'),
    ('cargo', 'fetch', '--locked', '--manifest-path', 'fuzz/Cargo.toml'),
)


def test_counts(path):
    totals = dict(suites=0, passed=0, failed=0, ignored=0, filtered=0)
    with path.open(encoding='utf-8', errors='replace') as stream:
        for line in stream:
            match = re.search(r'test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored; \d+ measured; (\d+) filtered out', line)
            if match:
                totals['suites'] += 1
                for key, value in zip(('passed', 'failed', 'ignored', 'filtered'), match.groups()):
                    totals[key] += int(value)
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--fetch-only', action='store_true',
                        help='prepare both locked Cargo graphs for offline legal checks; do not compile')
    args = parser.parse_args()
    if args.report.exists(): raise FileExistsError(args.report)
    if os.name != 'nt':
        def interrupted(_signal, _frame): raise KeyboardInterrupt('soak checks interrupted')
        signal.signal(signal.SIGTERM, interrupted)
    target = native_target()
    if target is None: raise ValueError('unsupported native platform')
    directory = ROOT / '.cache/cpu-soak-checks' / args.report.stem
    directory.mkdir(parents=True, exist_ok=False)
    source = source_identity()
    commands = FETCH_COMMANDS if args.fetch_only else COMMANDS
    result = {'schema_version': 1, 'kind': 'cpu-soak-dependencies' if args.fetch_only else 'cpu-soak-checks', 'outcome': 'running',
              'production_ready': False, 'target': target, 'source_snapshot': source,
              'compatibility_sha256': sha(ROOT / 'fixtures/compatibility.json'), 'commands': [],
              'host': {'system': platform.system(), 'release': platform.release(), 'machine': platform.machine()},
              'scope': 'Soak driver compilation, example tests and strict Clippy only; no model, sustained-soak or release qualification.'}
    if args.fetch_only:
        result['scope'] = 'Locked workspace/fuzz dependency acquisition only; no compilation, model execution or release qualification.'
    result['ci'] = {key: os.environ[key] for key in ('GITHUB_REPOSITORY', 'GITHUB_SHA', 'GITHUB_RUN_ID',
                                                  'GITHUB_RUN_ATTEMPT', 'GITHUB_REF') if key in os.environ}
    guard, children = None, []
    started = time.monotonic()
    checkpoint(args.report, result)
    try:
        guard = Guard(directory)
        guard.start()
        if os.environ.get('CARGO_BUILD_TARGET'):
            raise ValueError('unset CARGO_BUILD_TARGET; checks must execute on the native target')
        cargo = str(ROOT / 'tools/cargo') if os.name != 'nt' else 'cargo'
        environment = {**os.environ, 'CARGO_BUILD_JOBS': '1', 'CARGO_INCREMENTAL': '0',
                       'CARGO_TARGET_DIR': str(ROOT / 'target')}
        result['build_environment'] = {key: environment[key] for key in ('CARGO_BUILD_JOBS', 'CARGO_INCREMENTAL')}
        local_rust = ROOT / '.cache/cargo/bin/rustc'
        compiler = str(local_rust) if os.name != 'nt' and local_rust.exists() else 'rustc'
        compiler_env = {**environment, 'RUSTUP_HOME': str(ROOT / '.cache/rustup')} if compiler == str(local_rust) else environment
        result['toolchain'] = {}
        for name, arguments, env in [('cargo', [cargo, '--version'], environment),
                                     ('rustc', [compiler, '-vV'], compiler_env)]:
            log = directory / f'{name}.log'
            code = run_child(arguments, ROOT, env, log, 30, guard, children.append)
            if code or log.stat().st_size > 64 << 10: raise ValueError('toolchain identity probe failed: ' + name)
            result['toolchain'][name] = {'output': log.read_text(encoding='utf-8'),
                                         'log': log.relative_to(ROOT).as_posix(), 'log_sha256': sha(log)}
            checkpoint(args.report, result)
        for number, command in enumerate(commands):
            log = directory / f'{number}.log'
            record = {'command': list(command), 'exit': None, 'validated': False,
                      'log': log.relative_to(ROOT).as_posix()}
            result['commands'].append(record)
            checkpoint(args.report, result)
            try:
                record['exit'] = run_child([cargo, *command[1:]], ROOT, environment, log, 1800, guard, children.append)
                valid = record['exit'] == 0
                if command[1] == 'test':
                    record['tests'] = counts = test_counts(log)
                    valid = valid and counts['passed'] > 0 and counts['failed'] == counts['filtered'] == 0
                record['validated'] = valid
                if not valid: raise ValueError('native soak check failed: ' + ' '.join(command))
            finally:
                if log.exists(): record['log_sha256'] = sha(log)
                checkpoint(args.report, result)
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
    print(result['outcome'].upper(), 'native soak checks:', args.report)
    raise SystemExit(result['outcome'] != 'pass')


if __name__ == '__main__':
    main()

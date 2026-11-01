#!/usr/bin/env python3
"""Run guarded native release checks and retain complete source-bound evidence."""
import argparse
import os
from pathlib import Path
import platform
import shutil
import signal
import sys
import time
import tomllib

from check_cpu_release import ROOT, COMMANDS, source_identity, sha
from cpu_qualification.driver import test_counts
from cpu_qualification.host import Guard
from run_cpu_qualification import checkpoint
from run_cpu_soak import run_child
from verify_cpu_package import native_target

ENGINE_COMMANDS = (
    COMMANDS[0],
    ('cargo', 'clippy', '--release', '-p', 'coheryn-engine', '--all-targets', '--locked', '--', '-D', 'warnings'),
    ('cargo', 'test', '--release', '-p', 'coheryn-engine', '--lib', '--locked'),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--engine-only', action='store_true',
                        help='guarded formatting/lint/library feedback; cannot qualify a native release')
    args = parser.parse_args()
    if args.report.exists():
        parser.error('use a fresh native report path')
    target = native_target()
    if target is None:
        parser.error('unsupported native platform')
    if os.name != 'nt':
        def interrupted(_signal, _frame): raise KeyboardInterrupt('native checks interrupted')
        signal.signal(signal.SIGTERM, interrupted)
    source = source_identity()
    directory = ROOT / '.cache/cpu-checks' / args.report.stem
    directory.mkdir(parents=True, exist_ok=False)
    cargo = str(ROOT / 'tools/cargo') if os.name != 'nt' and (ROOT / '.cache/cargo/bin/cargo').exists() else 'cargo'
    environment = {**os.environ, 'CARGO_BUILD_JOBS': '1', 'CARGO_INCREMENTAL': '0',
                   'RUST_TEST_THREADS': '1', 'CARGO_NET_OFFLINE': 'true',
                   'CARGO_TARGET_DIR': str(ROOT / 'target'),
                   'CARGO_PROFILE_DEV_DEBUG': '0', 'CARGO_PROFILE_TEST_DEBUG': '0'}
    commands = ENGINE_COMMANDS if args.engine_only else COMMANDS
    result = {'schema_version': 1, 'kind': 'cpu-engine-checks' if args.engine_only else 'cpu-native-checks', 'target': target,
              'outcome': 'running', 'production_ready': False,
              'host': {'system': platform.system(), 'release': platform.release(), 'machine': platform.machine()},
              'rust': None, 'compiler': None, 'commands': [], 'binaries': {},
              'source_snapshot': source, 'source_unchanged': False,
              'compatibility_sha256': sha(ROOT / 'fixtures/compatibility.json'),
              'execution_settings': {'cargo_build_jobs': 1, 'rust_test_threads': 1,
                                     'test_profile': 'release', 'cargo_incremental': 0},
              'scope': 'Native automated checks only; no downloaded-model, security-review or production qualification claim.'}
    if args.engine_only:
        result['scope'] = 'Formatting, strict engine lint and ordinary engine library tests only; no complete workspace, executable, model or release qualification.'
    guard, children = None, []
    started = time.monotonic()
    checkpoint(args.report, result)
    try:
        guard = Guard(directory)
        guard.start()
        if environment.get('CARGO_BUILD_TARGET'):
            raise ValueError('unset CARGO_BUILD_TARGET; checks must execute natively')
        local_rust = ROOT / '.cache/cargo/bin/rustc'
        compiler = str(local_rust) if local_rust.exists() else shutil.which('rustc')
        if not compiler:
            raise ValueError('native compiler not found')
        compiler_env = {**environment, 'RUSTUP_HOME': str(ROOT / '.cache/rustup')} if local_rust.exists() else environment
        version = tomllib.loads((ROOT / 'rust-toolchain.toml').read_text())['toolchain']['channel']
        result['toolchain'] = {}
        for name, command, env in [('cargo', [cargo, '--version'], environment), ('rustc', [compiler, '-vV'], compiler_env)]:
            log = directory / (name + '.log')
            code = run_child(command, ROOT, env, log, 30, guard, children.append)
            if code or log.stat().st_size > 64 << 10:
                raise ValueError('native compiler identity probe failed')
            output = log.read_text(encoding='utf-8')
            if not output.startswith(f'{name} {version} '):
                raise ValueError('native compiler differs from the pinned toolchain')
            result['toolchain'][name] = {'output': output, 'log': log.relative_to(ROOT).as_posix(), 'log_sha256': sha(log)}
            result['rust' if name == 'cargo' else 'compiler'] = output.strip() if name == 'cargo' else output
        for number, command in enumerate(commands):
            executable = cargo if command[0] == 'cargo' else sys.executable
            log = directory / f'{number}.log'
            record = {'command': list(command), 'exit': None, 'log': log.relative_to(ROOT).as_posix(), 'validated': False}
            result['commands'].append(record)
            checkpoint(args.report, result)
            try:
                record['exit'] = run_child([executable, *command[1:]], ROOT, environment, log, 3600, guard, children.append)
            finally:
                if log.exists(): record['log_sha256'] = sha(log)
                checkpoint(args.report, result)
            valid = record['exit'] == 0
            if command[:2] == ('cargo', 'test'):
                record['tests'] = counts = test_counts(log)
                valid = valid and counts['suites'] > 0 and counts['passed'] > 0 and counts['failed'] == counts['filtered'] == 0
            record['validated'] = valid
            checkpoint(args.report, result)
            print(('PASS' if valid else 'FAIL') + ' ' + ' '.join(command), flush=True)
            if not valid:
                raise ValueError('native check failed: ' + ' '.join(command))
        if not args.engine_only:
            suffix = '.exe' if os.name == 'nt' else ''
            result['binaries'] = {name + suffix: {'sha256': sha(ROOT / 'target/release' / (name + suffix)),
                                                  'bytes': (ROOT / 'target/release' / (name + suffix)).stat().st_size}
                                  for name in ('coheryn', 'coheryn-bench')}
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
        result['source_unchanged'] = source_identity() == source
        result['owned_cleanup_confirmed'] = all(p.poll() is not None for p in children) and not (guard and guard.stop_errors)
        if not result['source_unchanged'] or not result['owned_cleanup_confirmed']: result['outcome'] = 'fail'
        result['elapsed_seconds'] = time.monotonic() - started
        checkpoint(args.report, result)
    label = 'engine checks:' if args.engine_only else 'native CPU checks:'
    print(result['outcome'].upper(), label, args.report, flush=True)
    raise SystemExit(result['outcome'] != 'pass')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Guarded pinned-model benchmark routing and frozen greedy-output checks."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

from check_cpu_release import ROOT, sha, source_identity
from cpu_qualification.assets import models
from cpu_qualification.driver import test_counts
from cpu_qualification.host import Guard
from run_cpu_qualification import checkpoint
from run_cpu_soak import run_child
from verify_cpu_package import executable_target, native_target

SUITES = ('gguf-text', 'qwen35-gguf-q4_k_m-text', 'qwen35-text')
COMMAND = ('test', '--release', '--workspace', '--test', 'native_text_bench', '--locked', '--', '--ignored')


def validate_native(path, suite):
    with path.open('rb') as stream:
        data = stream.read((2 << 20) + 1)
    if len(data) > 2 << 20:
        raise ValueError('native benchmark output exceeds limit')
    value = json.loads(data)
    entries = 3 if suite == 'qwen35-text' else 1
    if not isinstance(value, dict):
        raise ValueError('native benchmark output must be an object')
    threads = value.get('threads')
    if (value.get('schema_version') != 1 or value.get('outcome') != 'pass' or value.get('suite') != suite
            or value.get('source_entrypoints') != entries or value.get('cases') != 2
            or threads not in ([1, 2], [1, 2, 4])
            or any(type(t) is not int for t in threads)
            or not isinstance(value.get('runs'), list) or len(value['runs']) != entries * 2 * len(threads)):
        raise ValueError('incomplete native benchmark output')
    expected_cases = {'english', 'bengali' if suite == 'gguf-text' else 'unicode'}
    for row in value['runs']:
        if (not isinstance(row, dict) or type(row.get('entrypoint')) is not int or type(row.get('workers')) is not int
                or not isinstance(row.get('case'), str)):
            raise ValueError('invalid benchmark coverage identity')
        benchmark = row.get('benchmark')
        if (not isinstance(benchmark, dict) or benchmark.get('mode') != 'standalone'
                or benchmark.get('cache_condition') != 'uncontrolled'
                or not isinstance(benchmark.get('runs'), list) or len(benchmark['runs']) != 2):
            raise ValueError('incomplete standalone benchmark iterations')
        for run in benchmark['runs']:
            if not isinstance(run, dict):
                raise ValueError('invalid benchmark iteration')
            for field in ('prefill_ms', 'decode_ms', 'tokenization_ms', 'output_delivery_ms', 'first_token_ms'):
                timing = run.get(field)
                if type(timing) not in (float, int) or not math.isfinite(timing) or timing < 0:
                    raise ValueError('invalid benchmark timing observation')
    actual = {(row['entrypoint'], row['case'], row['workers']) for row in value['runs']}
    expected = {(entry, case, worker) for entry in range(entries) for case in expected_cases for worker in threads}
    if actual != expected:
        raise ValueError('benchmark entrypoint/case/worker coverage changed')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=SUITES, required=True)
    parser.add_argument('--fetch', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error('use a fresh benchmark report path')
    if os.name != 'nt':
        def interrupted(_signal, _frame): raise KeyboardInterrupt('benchmark checks interrupted')
        signal.signal(signal.SIGTERM, interrupted)
    target = native_target()
    if target is None:
        parser.error('unsupported native benchmark target')
    directory = ROOT / '.cache/text-benchmark' / args.report.stem
    directory.mkdir(parents=True, exist_ok=False)
    native_output = directory / 'native.json'
    source = source_identity()
    result = {'schema_version': 1, 'kind': 'native-text-benchmark-checks', 'outcome': 'running',
              'production_ready': False, 'suite': args.suite, 'target': target,
              'source_snapshot': source, 'compatibility_sha256': sha(ROOT / 'fixtures/compatibility.json'),
              'commands': [], 'scope': 'Pinned greedy/tokenizer and CLI benchmark routing with fresh sessions. '
                  'Timing observations only: no full-array, controlled-cache, performance-SLA or release qualification.'}
    result['ci'] = {key: os.environ[key] for key in ('GITHUB_REPOSITORY', 'GITHUB_SHA', 'GITHUB_RUN_ID',
                                                   'GITHUB_RUN_ATTEMPT', 'GITHUB_REF') if key in os.environ}
    guard, children = None, []
    started = time.monotonic()
    checkpoint(args.report, result)
    try:
        guard = Guard(directory)
        guard.start()
        if os.environ.get('CARGO_BUILD_TARGET'):
            raise ValueError('unset CARGO_BUILD_TARGET; benchmark checks must execute natively')
        _, result['model_inputs'] = models([args.suite], args.fetch, guard)
        checkpoint(args.report, result)
        cargo = str(ROOT / 'tools/cargo') if os.name != 'nt' else 'cargo'
        environment = {**os.environ, 'CARGO_BUILD_JOBS': '1', 'CARGO_INCREMENTAL': '0',
                       'CARGO_TARGET_DIR': str(ROOT / 'target'), 'COHERYN_TEXT_BENCH_SUITE': args.suite,
                       'COHERYN_TEXT_BENCH_REPORT': str(native_output)}
        result['toolchain'] = {}
        local_rust = ROOT / '.cache/cargo/bin/rustc'
        compiler = str(local_rust) if os.name != 'nt' and local_rust.exists() else 'rustc'
        compiler_env = {**environment, 'RUSTUP_HOME': str(ROOT / '.cache/rustup')} if compiler == str(local_rust) else environment
        for name, arguments, env in [('cargo', [cargo, '--version'], environment),
                                     ('rustc', [compiler, '-vV'], compiler_env)]:
            log = directory / (name + '.log')
            code = run_child(arguments, ROOT, env, log, 30, guard, children.append)
            if code or log.stat().st_size > 64 << 10:
                raise ValueError('benchmark toolchain identity probe failed')
            result['toolchain'][name] = {'output': log.read_text(encoding='utf-8'),
                                         'log': log.relative_to(ROOT).as_posix(), 'log_sha256': sha(log)}
        command = [cargo, *COMMAND]
        log = directory / 'checks.log'
        record = {'command': command, 'exit': None, 'validated': False, 'log': log.relative_to(ROOT).as_posix()}
        result['commands'].append(record)
        checkpoint(args.report, result)
        try:
            record['exit'] = run_child(command, ROOT, environment, log, 7200, guard, children.append)
        finally:
            if log.exists(): record['log_sha256'] = sha(log)
            checkpoint(args.report, result)
        record['tests'] = counts = test_counts(log)
        if record['exit'] != 0 or counts != dict(suites=1, passed=1, failed=0, ignored=0, filtered=0):
            raise ValueError('native text benchmark test did not execute completely')
        record['validated'] = True
        result['benchmark'] = validate_native(native_output, args.suite)
        result['native_output_sha256'] = sha(native_output)
        binary = ROOT / 'target/release' / ('coheryn.exe' if target.endswith('msvc') else 'coheryn')
        executable_target(binary, target)
        result['binary_sha256'] = sha(binary)
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
    print(result['outcome'].upper(), 'native text benchmark checks:', args.report)
    raise SystemExit(result['outcome'] != 'pass')


if __name__ == '__main__':
    main()

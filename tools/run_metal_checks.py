#!/usr/bin/env python3
"""Guarded, source-bound optional Metal checks; never promotes a model release."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import sys
import time
import tomllib

from check_cpu_release import ROOT, sha, source_identity
from cpu_qualification.driver import test_counts
from cpu_qualification.assets import models
from cpu_qualification.profiles import PROFILES
from cpu_qualification.host import Guard
from run_cpu_qualification import checkpoint
from run_cpu_soak import run_child


def commands(gpu, cross_check=False):
    result = [
        ('cargo', 'fmt', '--all', '--', '--check'),
        ('cargo', 'clippy', '--release', '-p', 'coheryn-metal',
         '--features', 'metal', '--all-targets', '--locked', '--', '-D', 'warnings'),
        ('cargo', 'clippy', '--release', '-p', 'coheryn-engine',
         '--features', 'metal', '--all-targets', '--locked', '--', '-D', 'warnings'),
        ('cargo', 'clippy', '--release', '-p', 'coheryn-cli',
         '--features', 'metal', '--all-targets', '--locked', '--', '-D', 'warnings'),
        ('cargo', 'clippy', '--release', '-p', 'coheryn-api',
         '--features', 'metal', '--all-targets', '--locked', '--', '-D', 'warnings'),
        ('cargo', 'test', '--profile', 'qualification', '-p', 'coheryn-metal', '--features', 'metal', '--lib', '--locked'),
        ('cargo', 'test', '--profile', 'qualification', '-p', 'coheryn-engine', '--features', 'metal', '--lib', '--locked'),
        ('cargo', 'test', '--profile', 'qualification', '-p', 'coheryn-cli', '--features', 'metal', '--bin', 'coheryn', '--locked'),
        ('cargo', 'test', '--profile', 'qualification', '-p', 'coheryn-api', '--features', 'metal', '--test', 'service', '--locked'),
        ROUTING_COMMAND,
    ]
    if cross_check:
        # Compile the macOS FFI and engine tests, never execute a foreign binary.
        # The HTTP CLI includes C TLS dependencies requiring an Apple SDK; check
        # its platform-neutral routing natively here and its Metal build on Mac.
        for index in (1, 2):
            result[index] = (*result[index][:-3], '--target', 'aarch64-apple-darwin', *result[index][-3:])
        return (*result[:5], ROUTING_COMMAND)
    if gpu:
        result.append(('cargo', 'test', '--profile', 'qualification', '-p', 'coheryn-metal',
                       '--features', 'metal', '--test', 'dense', '--locked', '--', '--ignored', '--nocapture'))
        result.append(('cargo', 'test', '--profile', 'qualification', '-p', 'coheryn-metal',
                       '--features', 'metal', '--test', 'packed', '--locked', '--', '--ignored', '--nocapture'))
        result.append(('cargo', 'test', '--profile', 'qualification', '-p', 'coheryn-metal',
                       '--features', 'metal', '--test', 'simd', '--locked', '--', '--ignored', '--nocapture'))
        result.append(('cargo', 'test', '--profile', 'qualification', '-p', 'coheryn-engine',
                       '--features', 'metal', '--test', 'metal', '--locked', '--', '--ignored', '--nocapture'))
    return tuple(result)


MODEL_SUITES = ('qwen35-gguf-q4_k_m-text', 'qwen35-text')
ROUTING_COMMAND = ('cargo', 'test', '--profile', 'qualification', '-p', 'coheryn-engine',
                   '--features', 'metal', '--test', 'gguf', '--test', 'gguf_probe', '--test', 'adapters', '--locked')
MODEL_COMMAND = ('cargo', 'test', '--profile', 'qualification', '-p', 'coheryn-engine',
                 '--features', 'metal', '--test', 'metal_model', '--locked', '--', '--ignored', '--nocapture')
SIMD_MODEL_COMMAND = ('cargo', 'test', '--profile', 'qualification', '-p', 'coheryn-engine',
                      '--features', 'metal', '--test', 'metal_simd_model', '--locked', '--', '--ignored', '--nocapture')
CLI_MODEL_COMMAND = ('cargo', 'test', '--profile', 'qualification', '-p', 'coheryn-cli',
                     '--features', 'metal', '--test', 'metal_model', '--locked', '--', '--ignored', '--nocapture')
API_MODEL_COMMAND = ('cargo', 'test', '--profile', 'qualification', '-p', 'coheryn-api',
                     '--features', 'metal', '--test', 'metal_model', '--locked', '--', '--ignored', '--nocapture')
SERVICE_NETWORK_TEST = 'pinned_public_https_download_follows_verified_redirects'
SERVICE_BACKEND_TEST = 'explicit_backend_load_keeps_authentication_cpu_defaults_and_atomic_state'
CLI_REJECTED_ARGUMENTS = [
    ['run', 'missing.gguf', '--backend', 'metal', '--prompt', 'hello'],
    ['chat', 'missing.gguf', '--standalone', '--backend', 'metal', '--image', 'missing.png'],
    ['chat', 'missing.gguf', '--standalone', '--backend', 'metal', '--request', 'missing.json'],
    ['chat', 'missing.gguf', '--standalone', '--backend', 'metal', '--session', 'test'],
]


def validate_cli(path, suite):
    if path.stat().st_size > 2 << 20:
        raise ValueError('GPU CLI output exceeds bound')
    value = json.loads(path.read_text())
    reference = json.loads((ROOT / PROFILES[suite]['reference']).read_text())
    if (not isinstance(value, dict) or value.get('schema_version') != 1
            or value.get('outcome') != 'pass' or value.get('suite') != suite
            or not isinstance(value.get('cases'), list) or len(value['cases']) != len(reference['cases'])):
        raise ValueError('incomplete GPU CLI output coverage')
    for result, expected in zip(value['cases'], reference['cases'], strict=True):
        if (not isinstance(result, dict) or result.get('token_ids') != expected['greedy_token_ids']
                or result.get('text') != expected['greedy_text']
                or result.get('prompt_tokens') != len(expected['prompt_token_ids'])
                or result.get('completion_tokens') != len(expected['greedy_token_ids'])
                or result.get('sampling_metadata', {}).get('backend') != 'native-metal-matmul-f32-reference'):
            raise ValueError('GPU CLI frozen output/backend mismatch')
    if value.get('streams') != [case['greedy_text'] + '\n' for case in reference['cases']]:
        raise ValueError('GPU CLI streaming reconstruction mismatch')
    rejected = value.get('rejected')
    expected_codes = ['INVALID_REQUEST', *(['UNSUPPORTED_FEATURE'] * 3)]
    if (not isinstance(rejected, list) or len(rejected) != 4
            or any(not isinstance(row, dict) or type(row.get('exit')) is not int or row['exit'] != status
                   or row.get('arguments') != arguments
                   or row.get('error', {}).get('code') != code
                   for row, code, status, arguments in zip(rejected, expected_codes, [2, 4, 4, 4], CLI_REJECTED_ARGUMENTS, strict=True))):
        raise ValueError('GPU CLI rejection coverage missing')
    return value


def validate_api(path, suite):
    if path.stat().st_size > 2 << 20:
        raise ValueError('GPU API output exceeds bound')
    value = json.loads(path.read_text())
    reference = json.loads((ROOT / PROFILES[suite]['reference']).read_text())
    if (not isinstance(value, dict) or value.get('schema_version') != 1
            or value.get('outcome') != 'pass' or value.get('suite') != suite):
        raise ValueError('invalid GPU API report identity')
    for name in ('cases', 'streams'):
        if not isinstance(value.get(name), list) or len(value[name]) != len(reference['cases']):
            raise ValueError('incomplete GPU API reference coverage')
    def backend(result):
        return isinstance(result, dict) and isinstance(result.get('sampling_metadata'), dict) and result['sampling_metadata'].get('backend') == 'native-metal-matmul-f32-reference'
    def output(result, expected):
        return (backend(result) and result.get('token_ids') == expected['greedy_token_ids']
                and result.get('text') == expected['greedy_text']
                and result.get('completion_tokens') == len(expected['greedy_token_ids']))
    for result, stream, expected in zip(value['cases'], value['streams'], reference['cases'], strict=True):
        if not output(result, expected) or not isinstance(stream, dict):
            raise ValueError('GPU API frozen output/backend mismatch')
        finish = stream.get('finish', {})
        if (stream.get('text') != expected['greedy_text'] or type(stream.get('done')) is not int or stream['done'] != 1
                or not isinstance(finish, dict) or finish.get('coheryn', {}).get('sampling', {}).get('backend') != 'native-metal-matmul-f32-reference'
                or stream.get('usage', {}).get('completion_tokens') != len(expected['greedy_token_ids'])
                or stream.get('usage', {}).get('prompt_tokens') != len(expected['prompt_token_ids'])):
            raise ValueError('GPU API SSE termination/output/backend mismatch')
    if not output(value.get('healthy_after_disconnect'), reference['cases'][0]):
        raise ValueError('GPU API did not survive disconnect')
    cached, fresh = value.get('cached_result'), value.get('fresh_result')
    if (not backend(cached) or not backend(fresh) or not cached.get('token_ids')
            or cached.get('token_ids') != fresh.get('token_ids') or cached.get('text') != fresh.get('text')
            or type(cached.get('reused_prompt_tokens')) is not int or cached['reused_prompt_tokens'] <= 0):
        raise ValueError('GPU API cached/fresh history parity missing')
    plan = value.get('plan', {})
    if (not isinstance(plan, dict) or plan.get('backend') != 'native-metal-matmul-f32-reference'
            or plan.get('concurrency') != 2 or type(plan.get('gpu_allocation_bytes')) is not int or plan['gpu_allocation_bytes'] <= 0):
        raise ValueError('GPU API backend-aware plan missing')
    info = value.get('gpu', {})
    if (not isinstance(info, dict) or not isinstance(info.get('device'), str) or not 1 <= len(info['device']) <= 256
            or info.get('kernel_profile') != 'reference-f32-compensated-v1'
            or any(type(info.get(name)) is not int or info[name] <= 0
                   for name in ('matrices', 'encoded_weight_bytes', 'driver_weight_allocation_bytes', 'request_workspace_bytes', 'completed_model_dispatches'))
            or info['driver_weight_allocation_bytes'] < info['encoded_weight_bytes']):
        raise ValueError('GPU API native dispatch/residency evidence missing')
    if plan['gpu_allocation_bytes'] != info['encoded_weight_bytes'] + 2 * info['request_workspace_bytes']:
        raise ValueError('GPU API plan did not account for shared weights and private concurrency buffers')
    memory = value.get('final_memory', {})
    if (not isinstance(memory, dict) or type(memory.get('reserved')) is not int or memory['reserved'] != 0
            or type(memory.get('session_buffer_capacity_bytes')) is not int or memory['session_buffer_capacity_bytes'] != 0
            or type(memory.get('peak_reserved')) is not int or type(memory.get('limit')) is not int
            or not 0 < memory['peak_reserved'] <= memory['limit']):
        raise ValueError('GPU API allocations were not reclaimed')
    if any(value.get(name) is not True for name in ('backend_conflict_rejected', 'pinned_unload_rejected',
            'disconnect_cleanup', 'image_profile_rejected', 'plan_without_reservation', 'unload_cache_release', 'history_deleted')):
        raise ValueError('GPU API lifecycle/planner evidence missing')
    if type(value.get('post_unload_history_reserved')) is not int or value['post_unload_history_reserved'] <= 0:
        raise ValueError('GPU API history ownership was not distinguished from resident allocations')
    return value


def validate_model(path, suite, kernel='reference-f32-compensated-v1'):
    if path.stat().st_size > 2 << 20:
        raise ValueError('GPU model output exceeds bound')
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('GPU model output must be an object')
    reference = json.loads((ROOT / PROFILES[suite]['reference']).read_text())
    expected_samples = sum(len(checkpoint['indices']) for case in reference['cases']
                           for checkpoint in [*case['checkpoints'], *case['teacher_forced']])
    if (value.get('schema_version') != 2 or value.get('outcome') != 'pass' or value.get('suite') != suite
            or value.get('full_cpu_gpu_arrays') != 50 or value.get('full_cpu_gpu_elements') != 545792
            or value.get('independent_sampled_values') != expected_samples
            or not isinstance(value.get('cases'), list) or len(value['cases']) != len(reference['cases'])):
        raise ValueError('incomplete GPU model numerical coverage')
    for result, expected in zip(value['cases'], reference['cases'], strict=True):
        if (not isinstance(result, dict) or not isinstance(result.get('token_ids'), list)
                or any(type(token) is not int for token in result['token_ids'])
                or result.get('token_ids') != expected['greedy_token_ids'] or result.get('text') != expected['greedy_text']
                or result.get('completion_tokens') != len(expected['greedy_token_ids'])
                or result.get('sampling_metadata', {}).get('backend') != 'native-metal-matmul-f32-reference'):
            raise ValueError('GPU frozen output/backend mismatch')
    info = value.get('gpu', {})
    if kernel not in ('reference-f32-compensated-v1', 'simd-f32-compensated-v1') or info.get('kernel_profile') != kernel:
        raise ValueError('GPU model kernel profile mismatch')
    for name in ('matrices', 'encoded_weight_bytes', 'request_workspace_bytes', 'completed_model_dispatches'):
        if type(info.get(name)) is not int or info[name] <= 0:
            raise ValueError('missing GPU allocation/dispatch evidence')
    if not isinstance(info.get('device'), str) or not 1 <= len(info['device']) <= 256:
        raise ValueError('missing Metal device identity')
    memory = value.get('memory_after_unload', {})
    if (type(memory.get('reserved')) is not int or memory['reserved'] != 0
            or type(memory.get('session_buffer_capacity_bytes')) is not int or memory['session_buffer_capacity_bytes'] != 0
            or type(memory.get('limit')) is not int
            or type(memory.get('peak_reserved')) is not int or not 0 < memory['peak_reserved'] <= memory.get('limit', 0)):
        raise ValueError('GPU model admission was not reclaimed')
    if any(value.get(name) is not True for name in ('trace_failure_cleanup', 'in_graph_cancellation_cleanup', 'unqualified_image_profile_rejected', 'gpu_plan_prepaid')):
        raise ValueError('GPU model lifecycle/profile checks missing')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--gpu', action='store_true', help='require real GPU numerical execution; absent devices fail')
    modes.add_argument('--cross-check', action='store_true', help='Linux-hosted macOS compilation only; cannot qualify GPU execution')
    parser.add_argument('--model-suite', choices=MODEL_SUITES, help='also require pinned pretrained native GPU graph checks')
    parser.add_argument('--fetch', action='store_true', help='explicitly acquire pinned model inputs under the host guard')
    args = parser.parse_args()
    if args.model_suite and not args.gpu:
        parser.error('pretrained model checks require explicit --gpu')
    if args.fetch and not args.model_suite:
        parser.error('--fetch requires a model suite')
    if args.report.exists():
        parser.error('use a fresh Metal check report path')
    native_host = ('aarch64-apple-darwin' if not args.cross_check else 'x86_64-unknown-linux-gnu')
    if args.cross_check:
        if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
            parser.error('the explicit cross-check requires a Linux x86-64 host')
    elif platform.system() != 'Darwin' or platform.machine().lower() not in ('arm64', 'aarch64'):
        parser.error('Metal qualification requires a native Apple Silicon Mac')
    if os.environ.get('CARGO_BUILD_TARGET'):
        parser.error('unset CARGO_BUILD_TARGET; qualification must execute natively')
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt('Metal checks interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    directory = ROOT / '.cache/metal-checks' / args.report.stem
    directory.mkdir(parents=True, exist_ok=False)
    source = source_identity()
    report = {'schema_version': 1, 'kind': 'metal-cross-check' if args.cross_check else 'metal-primitive-checks', 'outcome': 'running',
              'production_ready': False, 'gpu_execution_required': args.gpu,
              'compile_target': 'aarch64-apple-darwin', 'cross_check': args.cross_check,
              'cli_compile_target': native_host, 'api_compile_target': native_host, 'routing_execution_target': native_host,
              'source_snapshot': source, 'source_unchanged': False, 'commands': [],
              'toolchain': {},
              'fixture_sha256': sha(ROOT / 'fixtures/gpu/metal-dense-v1.json'),
              'packed_fixture_sha256': sha(ROOT / 'fixtures/gpu/metal-packed-v1.json'),
              'simd_scope': 'Explicit component and library profiles are checked separately; CLI/API retain the reference kernel pending promotion.',
              'scope': 'Optional backend compile/lint and independent dense/packed primitives with F32 operands only; no Q8 activation dot, model, performance or release qualification.',
              'host': {'system': platform.system(), 'machine': platform.machine(), 'release': platform.release()}}
    if args.model_suite:
        report['kind'] = 'metal-model-checks'
        report['model_suite'] = args.model_suite
        report['scope'] = 'Selected pretrained native library, standalone CLI and authenticated in-process API text checks; no TCP, performance or release qualification.'
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
        local_rust = ROOT / '.cache/cargo/bin/rustc'
        compiler = str(local_rust) if local_rust.exists() else shutil.which('rustc')
        if not compiler:
            raise ValueError('native compiler not found')
        compiler_env = {**environment, 'RUSTUP_HOME': str(ROOT / '.cache/rustup')} if local_rust.exists() else environment
        version = tomllib.loads((ROOT / 'rust-toolchain.toml').read_text())['toolchain']['channel']
        for name, command, env in [('cargo', [cargo, '--version'], environment), ('rustc', [compiler, '-vV'], compiler_env)]:
            log = directory / (name + '.log')
            code = run_child(command, ROOT, env, log, 30, guard, children.append)
            if code or log.stat().st_size > 64 << 10:
                raise ValueError('native compiler identity probe failed')
            output = log.read_text(encoding='utf-8')
            if not output.startswith(f'{name} {version} '):
                raise ValueError('native compiler differs from the pinned toolchain')
            if name == 'rustc' and f'\nhost: {native_host}\n' not in output:
                raise ValueError('Metal compiler host differs from the declared check mode')
            report['toolchain'][name] = {'output': output, 'log': log.relative_to(ROOT).as_posix(), 'log_sha256': sha(log)}
        inventory = (*commands(args.gpu, args.cross_check), *([MODEL_COMMAND, SIMD_MODEL_COMMAND, CLI_MODEL_COMMAND, API_MODEL_COMMAND] if args.model_suite else []))
        for number, command in enumerate(inventory):
            if command == MODEL_COMMAND:
                _, report['model_inputs'] = models([args.model_suite], args.fetch, guard)
                environment['COHERYN_METAL_MODEL_SUITE'] = args.model_suite
                environment['COHERYN_METAL_MODEL_REPORT'] = str(directory / 'model.json')
                environment['COHERYN_METAL_CLI_REPORT'] = str(directory / 'cli-model.json')
                environment['COHERYN_METAL_API_REPORT'] = str(directory / 'api-model.json')
                checkpoint(args.report, report)
            if command == SIMD_MODEL_COMMAND:
                environment['COHERYN_METAL_MODEL_REPORT'] = str(directory / 'simd-model.json')
            if command == API_MODEL_COMMAND:
                # The authenticated import creates a private managed model copy.
                # Check its disk headroom before the child starts copying bytes.
                copy_bytes = sum(item['bytes'] for item in report['model_inputs']) + (128 << 20)
                guard.sample(disk_reservation=copy_bytes)
                report['api_managed_copy_disk_reservation'] = copy_bytes
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
                    target = command[command.index('--test') + 1]
                    if command == ROUTING_COMMAND:
                        valid = valid and counts['suites'] == 3 and counts['passed'] >= 21 and counts['ignored'] == 0
                    elif target == 'service':
                        text = log.read_text(encoding='utf-8')
                        excluded = re.findall(r'^test (\S+) \.\.\. ignored\b.*$', text, re.MULTILINE)
                        row['excluded_tests'] = excluded
                        valid = (valid and counts['suites'] == 1 and counts['ignored'] == 1
                                 and excluded == [SERVICE_NETWORK_TEST]
                                 and f'test {SERVICE_BACKEND_TEST} ... ok' in text.splitlines())
                    else:
                        valid = valid and counts['passed'] == counts['suites'] == 1 and counts['ignored'] == 0
            row['validated'] = valid
            checkpoint(args.report, report)
            print(('PASS' if valid else 'FAIL') + ' ' + ' '.join(command), flush=True)
            # Preserve compiler/GPU feedback on a formatting-only failure. The
            # overall report still fails until every required command passes.
            if not valid and command[1] != 'fmt':
                raise ValueError('Metal check failed: ' + ' '.join(command))
            if command == MODEL_COMMAND:
                report['model'] = validate_model(directory / 'model.json', args.model_suite)
                report['model_output_sha256'] = sha(directory / 'model.json')
            if command == SIMD_MODEL_COMMAND:
                report['simd_model'] = validate_model(directory / 'simd-model.json', args.model_suite, 'simd-f32-compensated-v1')
                report['simd_model_output_sha256'] = sha(directory / 'simd-model.json')
            if command == CLI_MODEL_COMMAND:
                report['cli_model'] = validate_cli(directory / 'cli-model.json', args.model_suite)
                report['cli_model_output_sha256'] = sha(directory / 'cli-model.json')
            if command == API_MODEL_COMMAND:
                report['api_model'] = validate_api(directory / 'api-model.json', args.model_suite)
                report['api_model_output_sha256'] = sha(directory / 'api-model.json')
        if any(not row['validated'] for row in report['commands']):
            raise ValueError('Metal formatting failed; other guarded check results retained')
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
        try:
            report['source_unchanged'] = source_identity() == source
        except Exception as error:
            report['source_identity_error'] = str(error)
        report['owned_cleanup_confirmed'] = all(p.poll() is not None for p in children) and not (guard and guard.stop_errors)
        if not report['source_unchanged'] or not report['owned_cleanup_confirmed']: report['outcome'] = 'fail'
        report['elapsed_seconds'] = time.monotonic() - started
        checkpoint(args.report, report)
    print(report['outcome'].upper(), 'Metal primitive checks:', args.report, flush=True)
    raise SystemExit(report['outcome'] != 'pass')


if __name__ == '__main__':
    main()

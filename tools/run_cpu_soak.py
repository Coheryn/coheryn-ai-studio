#!/usr/bin/env python3
"""Run a source-bound native CPU workload soak with frozen acceptance limits."""
import argparse
import json
import math
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time

from check_cpu_release import ROOT, sha, source_identity

SCENARIOS = {'queue_saturation_and_disconnect', 'slow_consumer_timeout',
             'image_backpressure_peer_progress_and_drop', 'resource_rejection_and_model_reload'}
PROFILE_KEYS = {'schema_version', 'name', 'cycles', 'warmup_cycles', 'restart_every', 'fault_every',
                'concurrency', 'execution_profiles', 'max_memory_bytes', 'max_rss_bytes',
                'max_drained_rss_growth_bytes', 'require_rss', 'sample_ms',
                'request_timeout_seconds', 'cleanup_timeout_seconds', 'consumer_timeout_seconds'}


def load(path):
    with path.open('rb') as stream:
        data = stream.read((64 << 10) + 1)
    if len(data) > 64 << 10:
        raise ValueError('configuration exceeds 64 KiB')
    return json.loads(data)


def validate_profile(profile):
    if set(profile) != PROFILE_KEYS or profile['schema_version'] != 1:
        raise ValueError('unsupported or incomplete soak contract')
    integer_ranges = {'cycles': (1, 10000), 'warmup_cycles': (1, 10),
                      'restart_every': (1, profile['cycles']), 'fault_every': (1, profile['cycles']),
                      'max_memory_bytes': (128 << 20, 8 << 30), 'max_rss_bytes': (1, 1 << 50),
                      'max_drained_rss_growth_bytes': (0, profile['max_rss_bytes']),
                      'sample_ms': (10, 1000), 'request_timeout_seconds': (1, 1800),
                      'cleanup_timeout_seconds': (1, 60), 'consumer_timeout_seconds': (1, 60)}
    for key, (low, high) in integer_ranges.items():
        if type(profile[key]) is not int or not low <= profile[key] <= high:
            raise ValueError('invalid frozen threshold: ' + key)
    for key, allowed in [('concurrency', {1, 2}), ('execution_profiles', {'balanced', 'low-memory'})]:
        values = profile[key]
        if not isinstance(values, list) or not values or len(values) > 2 or len(set(values)) != len(values) or not set(values) <= allowed:
            raise ValueError('unsupported or duplicate configuration: ' + key)
    if any(type(n) is not int for n in profile['concurrency']):
        raise ValueError('concurrency must use integer counts')
    if type(profile['require_rss']) is not bool or not isinstance(profile['name'], str) or not 1 <= len(profile['name'].encode('utf-8')) <= 128:
        raise ValueError('invalid profile identity/RSS policy')


def summarize(path, profile):
    """Stream observations: memory use is independent of cycles/log length."""
    validate_profile(profile)
    keys = {(c, p) for c in profile['concurrency'] for p in profile['execution_profiles']}
    progress, baselines, residents = {key: 0 for key in keys}, {}, {}
    faults = dict.fromkeys(SCENARIOS, 0)
    fault_progress = {(key, scenario, phase): 0 for key in keys for scenario in SCENARIOS for phase in ('warmup', 'measured')}
    operations = {}
    peak_rss, peak_reserved, peak_buffers, growth = None, 0, 0, None
    samples, epochs, contracts, closed = 0, 0, 0, 0
    with path.open('rb') as stream:
        while line := stream.readline((2 << 20) + 1):
            if len(line) > 2 << 20 or not line.endswith(b'\n'):
                raise ValueError('oversized or truncated observer record')
            row = json.loads(line)
            kind = row['kind']
            if kind in ('cycle_complete', 'sample'):
                rss = row['rss_bytes']
                if rss is not None and (type(rss) is not int or rss <= 0):
                    raise ValueError('invalid physical memory observation')
                for field in ('reserved', 'peak_reserved', 'session_buffer_capacity_bytes', 'peak_session_buffer_capacity_bytes'):
                    if type(row['memory'][field]) is not int or row['memory'][field] < 0:
                        raise ValueError('invalid tracked memory observation')
            if kind == 'contract':
                contracts += 1
                if row['profile'] != profile:
                    raise ValueError('runtime changed the frozen contract')
            elif kind == 'observer_failure':
                raise ValueError(row['reason'])
            elif kind == 'warmup_complete':
                key = (row['concurrency'], row['profile'])
                if key not in keys or key in baselines:
                    raise ValueError('duplicate or unknown warmup')
                baselines[key] = row['rss_baseline_bytes']
                if profile['require_rss'] and baselines[key] is None:
                    raise ValueError('required warmup RSS missing')
            elif kind == 'history_baseline':
                if row['reserved_bytes'] < 16 << 20:
                    raise ValueError('missing persistent-store reservation')
            elif kind == 'resident_baseline':
                key = (row['concurrency'], row['execution_profile'])
                if key not in keys or key in residents:
                    raise ValueError('duplicate or unknown resident baseline')
                residents[key] = row['reserved_bytes']
            elif kind == 'bootstrap_complete':
                if row['reserved_after_close'] != 0:
                    raise ValueError('bootstrap retained reservations')
            elif kind == 'cycle_complete':
                key = (row['concurrency'], row['profile'])
                if key not in keys or row['cycle'] != progress[key] or key not in baselines:
                    raise ValueError('missing, duplicate or reordered measured cycle')
                progress[key] += 1
                memory = row['memory']
                if key not in residents or memory['reserved'] != residents[key]:
                    raise ValueError('resident reservations leaked after drain')
                if memory['session_buffer_capacity_bytes'] != 0:
                    raise ValueError('drained engine buffers leaked')
                rss = row['rss_bytes']
                if profile['require_rss'] and rss is None:
                    raise ValueError('required drained RSS missing')
                if rss is not None:
                    peak_rss = max(peak_rss or 0, rss)
                base = baselines[key]
                if base is not None and rss is not None:
                    delta = max(0, rss - base)
                    growth = max(growth or 0, delta)
                    if delta > profile['max_drained_rss_growth_bytes']:
                        raise ValueError('drained RSS growth exceeds frozen threshold')
            elif kind == 'sample':
                samples += 1
                rss = row['rss_bytes']
                if profile['require_rss'] and rss is None:
                    raise ValueError('required sampled RSS missing')
                if rss is not None:
                    peak_rss = max(peak_rss or 0, rss)
                memory = row['memory']
                peak_reserved = max(peak_reserved, memory['peak_reserved'])
                peak_buffers = max(peak_buffers, memory['peak_session_buffer_capacity_bytes'])
            elif kind == 'operation':
                latency = row['elapsed_ms']
                if row['outcome'] != 'pass' or not math.isfinite(latency) or latency < 0 or latency > profile['request_timeout_seconds'] * 1000:
                    raise ValueError('operation failed or exceeded frozen latency threshold')
                key = (row['concurrency'], row['execution_profile'])
                if key not in keys or row['task'] not in ('text', 'embedding', 'image'):
                    raise ValueError('unknown operation configuration/task')
                name = f'{key[0]}/{key[1]}/{row["task"]}'
                item = operations.setdefault(name, {'count': 0, 'total_ms': 0., 'max_ms': 0., 'path': row['path']})
                if item['path'] != row['path']:
                    raise ValueError('operation route changed during run')
                item['count'] += 1
                item['total_ms'] += latency
                item['max_ms'] = max(item['max_ms'], latency)
            elif kind == 'fault':
                scenario = row['scenario']
                if scenario not in faults:
                    raise ValueError('unknown fault evidence')
                key = ((row['concurrency'], row['execution_profile']), scenario, row['phase'])
                if key not in fault_progress or row['cycle'] != fault_progress[key]:
                    raise ValueError('missing, duplicated or misplaced fault-cycle evidence')
                fault_progress[key] += 1 if row['phase'] == 'warmup' else profile['fault_every']
                faults[scenario] += 1
                if scenario == 'slow_consumer_timeout' and (row['buffered_token_events'] != 16 or row['terminal_error_observed'] is not True):
                    raise ValueError('bounded slow-consumer termination not demonstrated')
                if scenario == 'image_backpressure_peer_progress_and_drop' and (row['vision_backpressure_observed'] is not True or row['peer_completion_ms'] > profile['consumer_timeout_seconds'] * 1000):
                    raise ValueError('peer progress before consumer timeout not demonstrated')
            elif kind == 'epoch_complete':
                epochs += 1
                if row['scheduler']['active'] or row['scheduler']['queued']:
                    raise ValueError('epoch ended with active/queued work')
            elif kind == 'service_closed':
                closed += 1
                if row['memory']['reserved'] or row['memory']['session_buffer_capacity_bytes']:
                    raise ValueError('service teardown retained tracked memory')
            else:
                raise ValueError('unknown observer record')
    expected_faults = len(keys) * (profile['warmup_cycles'] + (profile['cycles'] - 1) // profile['fault_every'] + 1)
    expected_epochs = len(keys) * math.ceil(profile['cycles'] / profile['restart_every'])
    if contracts != 1 or samples < expected_epochs or epochs != expected_epochs or closed != expected_epochs or set(baselines) != keys:
        raise ValueError('incomplete contract/sampling/restart evidence')
    if any(value != profile['cycles'] for value in progress.values()) or any(n != expected_faults for n in faults.values()):
        raise ValueError('incomplete measured cycle or fault coverage')
    for (_, _, phase), number in fault_progress.items():
        expected = profile['warmup_cycles'] if phase == 'warmup' else math.ceil(profile['cycles'] / profile['fault_every']) * profile['fault_every']
        if number != expected:
            raise ValueError('incomplete per-configuration fault coverage')
    for concurrency, execution in keys:
        normal = profile['cycles'] + profile['warmup_cycles'] + 1
        peers = profile['warmup_cycles'] + math.ceil(profile['cycles'] / profile['fault_every'])
        for task in ('text', 'image', 'embedding'):
            name = f'{concurrency}/{execution}/{task}'
            route = '/v1/embeddings' if task == 'embedding' else '/api/v1/generate'
            if name not in operations or operations[name]['count'] != normal + (peers if task != 'image' else 0) or operations[name]['path'] != route:
                raise ValueError('missing mixed task latency/coverage measurements')
    if peak_rss is not None and peak_rss > profile['max_rss_bytes'] or peak_reserved > profile['max_memory_bytes']:
        raise ValueError('sampled memory exceeds frozen threshold')
    for item in operations.values():
        item['mean_ms'] = item.pop('total_ms') / item['count']
    return {'measured_cycles': sum(progress.values()), 'service_epochs': epochs, 'samples': samples,
            'faults': faults, 'operations': operations, 'sampled_peak_rss_bytes': peak_rss,
            'max_drained_rss_growth_bytes': growth, 'peak_reserved_bytes': peak_reserved,
            'peak_engine_buffer_bytes': peak_buffers,
            'rss_acceptance': 'evaluated' if profile['require_rss'] else 'not required; missing observations do not satisfy RSS qualification'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, default=ROOT / 'fixtures/soak/cpu-synthetic-v1.json')
    parser.add_argument('--models', type=Path, default=ROOT / 'fixtures/soak/synthetic-models-v1.json')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=int, default=3600,
                        help='hard process deadline; timeout records failure and kills the owned driver')
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    profile, sources = load(args.profile), load(args.models)
    validate_profile(profile)
    if not 1 <= args.timeout_seconds <= 86400:
        parser.error('timeout must be between 1 and 86400 seconds')
    args.report = args.report.absolute()
    run = ROOT / '.cache/cpu-soak' / args.report.stem
    run.mkdir(parents=True, exist_ok=False)
    source = source_identity()
    environment = {**os.environ, 'CARGO_BUILD_JOBS': '2', 'CARGO_INCREMENTAL': '0', 'CARGO_TARGET_DIR': str(ROOT / 'target')}
    if environment.get('CARGO_BUILD_TARGET'):
        raise ValueError('unset CARGO_BUILD_TARGET; this is a native host qualification')
    cargo = str(ROOT / 'tools/cargo') if os.name != 'nt' else 'cargo'
    command = [cargo, 'build', '--release', '-p', 'coheryn-api', '--example', 'cpu_soak', '--locked']
    result = {'schema_version': 1, 'kind': 'cpu-mixed-soak', 'outcome': 'fail', 'production_ready': False,
              'profile': profile, 'profile_sha256': sha(args.profile), 'models': sources,
              'models_sha256': sha(args.models), 'source_snapshot': source,
              'hard_process_timeout_seconds': args.timeout_seconds,
              'host': {'system': platform.system(), 'release': platform.release(), 'machine': platform.machine()},
              'compatibility_sha256': sha(ROOT / 'fixtures/compatibility.json'),
              'scope': 'Development-only application/service soak. Synthetic mode exercises real native inference with deterministic small weights, not pretrained-model numerical or semantic qualification. Service restarts occur within one process; TCP transport and process-crash qualification are separate.'}
    started = time.monotonic()
    try:
        with (run / 'build.log').open('xb') as output:
            build = subprocess.run(command, cwd=ROOT, env=environment, stdout=output, stderr=subprocess.STDOUT, timeout=1800)
        if build.returncode:
            raise ValueError('optimized driver build failed; inspect build.log')
        binary = ROOT / 'target/release/examples' / ('cpu_soak.exe' if os.name == 'nt' else 'cpu_soak')
        result['binary_sha256'] = sha(binary)
        # Freeze exact inputs under this exclusive run directory before execution.
        for name, value in [('profile.json', profile), ('models.json', sources)]:
            (run / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
        native_env = {k: v for k, v in os.environ.items() if k in ('SystemRoot', 'WINDIR', 'TMP', 'TEMP', 'TMPDIR')}
        native_env['PATH'] = ''
        command = [str(binary), str(run / 'profile.json'), str(run / 'models.json'), str(run / 'runtime'), str(run / 'runtime.json')]
        with (run / 'driver.log').open('xb') as output:
            process = subprocess.run(command, cwd=ROOT, env=native_env, stdout=output, stderr=subprocess.STDOUT, timeout=args.timeout_seconds)
        result['runtime_exit'] = process.returncode
        runtime = json.loads((run / 'runtime.json').read_text()) if (run / 'runtime.json').exists() else None
        result['runtime'] = runtime
        if process.returncode or runtime is None or runtime['outcome'] != 'pass':
            raise ValueError('native soak failed: ' + str(runtime.get('error') if runtime else 'see driver.log'))
        if runtime['profile'] != profile or runtime['sources'] != sources:
            raise ValueError('runtime did not retain the frozen inputs')
        result['measurements'] = summarize(run / 'runtime/observations.jsonl', profile)
        if source_identity() != source or sha(binary) != result['binary_sha256']:
            raise ValueError('source or executable changed during soak')
        result['outcome'] = 'pass'
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        result['error'] = str(error)
    finally:
        # The child has exited: remove only disposable stores under this run, including
        # failed-run model copies. Keep contract, logs and all evidence for diagnosis.
        for store in (run / 'runtime').glob('store-*'):
            if store.is_symlink():
                result['outcome'] = 'fail'
                result['error'] = 'unexpected linked disposable store'
            else:
                try:
                    shutil.rmtree(store)
                except OSError as error:
                    result['outcome'] = 'fail'
                    result['error'] = 'disposable-store cleanup failed: ' + str(error)
        result['elapsed_seconds'] = time.monotonic() - started
        result['source_unchanged'] = source_identity() == source
        result['artifacts'] = [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size}
                               for p in sorted(run.rglob('*')) if p.is_file()]
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open('x', encoding='utf-8') as output:
            json.dump(result, output, indent=2)
            output.write('\n')
    print(result['outcome'].upper(), 'native CPU mixed-workload soak:', args.report)
    if result['outcome'] != 'pass':
        print(result.get('error', 'qualification failed'), file=sys.stderr)
    raise SystemExit(result['outcome'] != 'pass')


if __name__ == '__main__':
    main()

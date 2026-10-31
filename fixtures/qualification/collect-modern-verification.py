#!/usr/bin/env python3
"""Collect the pre-guard v2 campaign; reject changed source and incomplete evidence."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from check_cpu_release import source_identity
from cpu_qualification.common import read, require, sha, write
from cpu_qualification.profiles import PROFILES


def bound(relative):
    path = ROOT / relative
    return {'path': relative, 'bytes': path.stat().st_size, 'sha256': sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT / 'fixtures/qualification/cpu-modern-verification.json')
    args = parser.parse_args()
    native_path = 'fixtures/qualification/cpu-modern-native-v2.json'
    native = read(ROOT / native_path)
    source = source_identity()
    require(native['outcome'] == 'pass' and native['source_unchanged']
            and native['source_snapshot'] == source, 'native record is stale')
    package_path = 'fixtures/qualification/cpu-modern-package-v2.json'
    package = read(ROOT / package_path)
    require(sha(ROOT / 'target/cpu-modern-release-v2' / package['archive']) == package['sha256'],
            'package archive changed')
    require(package['manifest']['native_check_record_sha256'] == sha(ROOT / native_path),
            'package/native binding differs')
    recovery_path = 'fixtures/qualification/cpu-modern-models-v3-recovered.json'
    recovery = read(ROOT / recovery_path)
    records = [(recovery_path, recovery)]
    for suite in PROFILES:
        if suite not in recovery['selected_suites']:
            relative = 'fixtures/qualification/cpu-modern-' + suite + '-v3.json'
            records.append((relative, read(ROOT / relative)))
    require(set().union(*(set(r['selected_suites']) for _, r in records)) == set(PROFILES),
            'incomplete profile coverage')
    suites = {}
    array_count = value_count = vector_count = http_count = 0
    for relative, record in records:
        require(record['outcome'] == 'pass' and record['source_unchanged']
                and record['source_snapshot'] == source and record['target'] == native['target']
                and record['native_checks_sha256'] == sha(ROOT / native_path)
                and record['archive_sha256'] == package['sha256']
                and record['binary_sha256'] == native['binaries']['coheryn']['sha256']
                and record['production_ready'] is False and record['external_executor_invoked'] is False,
                'failed, stale or mismatched model record: ' + relative)
        require(set(record['results']) == set(record['selected_suites']), 'incomplete model result')
        for name, result in record['results'].items():
            require(name not in suites, 'duplicate profile evidence')
            numeric = result['numerical_and_packaged_cli']
            arrays = numeric.get('full_arrays', []) + [a for run in numeric.get('runs', []) for a in run['arrays']]
            vectors = numeric.get('vectors', [])
            calls = [c for c in result['http']['cases'] if 'status' in c]
            require(calls and all(c['status'] == 200 for c in calls), 'incomplete HTTP evidence')
            suites[name] = {'record': bound(relative),
                            'provenance': record.get('recovery_scope', 'fresh native execution'),
                            'group': PROFILES[name]['group'], 'model_inputs': record['models'],
                            'full_array_comparisons': len(arrays),
                            'full_array_values': sum(a['values'] for a in arrays),
                            'individual_embedding_vectors': len(vectors),
                            'batch_configurations': numeric.get('batch_configurations', []),
                            'http_requests': len(calls), 'http_routes': sorted({c['route'] for c in calls}),
                            'numerical_runs': len(numeric.get('runs', [])),
                            'publisher_framework_parity': numeric.get('publisher_framework_parity', 'not claimed')}
            array_count += len(arrays)
            value_count += sum(a['values'] for a in arrays)
            vector_count += len(vectors)
            http_count += len(calls)
    rust_log = (ROOT / native['commands'][2]['log']).read_text()
    counts = [tuple(map(int, match)) for match in
              re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;', rust_log)]
    rust = dict(zip(('passed', 'failed', 'ignored'), map(sum, zip(*counts))))
    python_tests = {}
    for command in native['commands'][4:7]:
        matches = re.findall(r'Ran (\d+) tests?', (ROOT / command['log']).read_text())
        require(len(matches) == 1, 'missing Python test summary')
        python_tests[command['command'][1]] = int(matches[0])
    readiness_path = 'fixtures/qualification/cpu-modern-readiness-v1.json'
    readiness = read(ROOT / readiness_path)
    require(readiness['source_snapshot'] == source and readiness['production_ready'] is False,
            'release preflight is stale')
    equivalence_path = 'fixtures/qualification/cpu-modern-trace-equivalence-v1.json'
    equivalence = read(ROOT / equivalence_path)
    require(equivalence['outcome'] == 'pass' and equivalence['production_binaries_identical'],
            'trace observer equivalence missing')
    report = {'schema_version': 1, 'kind': 'modern-cpu-local-verification',
              'outcome': 'pass', 'production_ready': False, 'target': native['target'],
              'compiler': native['compiler'], 'source_unchanged': True,
              'source_snapshot_sha256': package['manifest']['source_snapshot_sha256'],
              'binaries': native['binaries'], 'native_checks': bound(native_path),
              'package': bound(package_path), 'archive_sha256': package['sha256'],
              'compatibility': bound('fixtures/compatibility.json'),
              'reference_bundle': bound('fixtures/cpu-qualification/modern-reference-bundle-v1.json'),
              'quantization_quality': bound('fixtures/qwen35/gguf/quantization-quality-v1.json'),
              'trace_observer_equivalence': bound(equivalence_path),
              'recovery_procedure': bound('fixtures/qualification/revalidate-modern-v2.py'),
              'collection_procedure': bound('fixtures/qualification/collect-modern-verification.py'),
              'rust_tests': rust, 'python_tests': python_tests,
              'summary': {'suites': len(suites), 'full_array_comparisons': array_count,
                          'full_array_values': value_count, 'individual_embedding_vectors': vector_count,
                          'http_requests': http_count}, 'suites': suites,
              'release_readiness': {'record': bound(readiness_path),
                                    'accepted': sum(c['status'] == 'pass' for c in readiness['checks']),
                                    'blocked': sum(c['status'] == 'blocked' for c in readiness['checks'])},
              'interruptions': [
                  {'record': bound('fixtures/qualification/cpu-modern-models-v1.json'),
                   'reason': 'Deliberately stopped to buffer diagnostic trace writes; historical results retained.'},
                  {'artifacts': '.cache/cpu-qualification/cpu-modern-models-v2',
                   'reason': 'Parent process ended during interruption without a final report. Five preserved numerical/CLI suites and four completed image HTTP records were revalidated. Incomplete text HTTP checks reran. Original exit codes remain unavailable.'},
                  {'record': bound('fixtures/qualification/cpu-modern-models-v2-recovered.json'),
                   'reason': 'Redundant image API replay deliberately stopped after finding complete original status logs and responses; its partial fresh response is retained separately.'}],
              'limitations': [
                  'Local macOS ARM64 CPU evidence only; configured Linux/Windows jobs have not been executed.',
                  'Selected Qwen3.5-0.8B files and encodings only; the three physical Safetensors shards are a lossless fixture derivation, not a publisher multi-shard release.',
                  'Short prompts/continuations at admitted contexts 256/512; broad language, semantic image quality, larger configurations and all context sizes are not qualified.',
                  'Independent frozen array-equation references do not establish publisher-framework parity. Quantization quality observations remain separate from executor correctness.',
                  'Cache conditions uncontrolled; diagnostic trace I/O is not an inference performance benchmark.',
                  'Internal allocation cleanup checked; portable peak RSS, private memory, faults and I/O are unavailable in this runner.',
                  'GPU/Metal, sustained resource/security campaigns, complete release matrix and signing/review remain open.']}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write(args.report, report)
    print(json.dumps(report['summary']))


if __name__ == '__main__':
    main()

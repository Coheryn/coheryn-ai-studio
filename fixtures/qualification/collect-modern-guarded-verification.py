#!/usr/bin/env python3
"""Index exact local evidence across tool revisions; never promote release gates.

Historical records keep their original native/source bindings. Reuse requires
identical production bytes, frozen inputs and numerical comparison procedures.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from check_cpu_release import source_identity
from cpu_qualification.common import read, require, sha
from cpu_qualification.profiles import PROFILES
from run_cpu_qualification import checkpoint


def bound(relative):
    path = ROOT / relative
    return {'path': relative, 'sha256': sha(path), 'bytes': path.stat().st_size}


def runtime_input(name):
    return (name.startswith('fixtures/')
            or name.startswith('crates/') and '/examples/' not in name
            or name in ('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml',
                        'tools/cpu_qualification/common.py', 'tools/cpu_qualification/suites.py'))


def host_observations(record):
    observer = record['host_guard']
    require(observer['samples'] > 0 and observer['stop_reason'] is None and not observer['stop_errors'],
            'guarded campaign stopped or observer cleanup failed')
    path = ROOT / record['artifacts'] / observer['log']
    require(sha(path) == observer['log_sha256'], 'host log differs')
    rows = []
    with path.open('rb') as stream:
        while data := stream.readline((64 << 10) + 1):
            require(len(data) <= 64 << 10 and len(rows) <= observer['policy']['max_samples'], 'oversized host evidence')
            row = json.loads(data)
            require(row['sample'] == len(rows) + 1 and row['rejection'] is None, 'invalid host sample or cutoff')
            rows.append(row['observation'])
    require(len(rows) == observer['samples'] and rows[-1] == observer['last'], 'host sample count/tail differs')
    return {'samples': len(rows), 'minimum_available_memory_bytes': min(r['available_memory_bytes'] for r in rows),
            'minimum_disk_free_bytes': min(r['disk_free_bytes'] for r in rows),
            'maximum_compressed_memory_bytes': max(r['compressed_memory_bytes'] or 0 for r in rows),
            'thermal_states': sorted({r['thermal_state'] for r in rows}),
            'maximum_load_1m': max(r['load_1m'] or 0 for r in rows), 'log_sha256': observer['log_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT / 'fixtures/qualification/cpu-modern-verification.json')
    args = parser.parse_args()
    native_path = 'fixtures/qualification/cpu-modern-native-v4.json'
    native = read(ROOT / native_path)
    current = source_identity()
    require(native['outcome'] == 'pass' and native['source_unchanged'] and native['source_snapshot'] == current,
            'current complete native evidence required')
    driver = read(ROOT / 'fixtures/qualification/cpu-modern-guarded-native-driver-v1.json')
    require(driver['outcome'] == 'pass' and driver['source_unchanged'] and driver['source_snapshot'] == current
            and driver['host_guard']['stop_reason'] is None, 'guarded native driver is incomplete or stale')
    for name, item in native['binaries'].items():
        require(sha(ROOT / 'target/release' / name) == item['sha256'], 'current production binary differs')
    natives = {}
    for version in (1, 2, 4):
        relative = f'fixtures/qualification/cpu-modern-native-v{version}.json'
        n = read(ROOT / relative)
        require(n['outcome'] == 'pass' and n['source_unchanged'] and n['binaries'] == native['binaries']
                and n['compiler'] == native['compiler'], 'native executable/compiler identity differs')
        natives[sha(ROOT / relative)] = n
    records = [('fixtures/qualification/cpu-modern-models-v3-recovered.json', 'historical revalidation'),
               ('fixtures/qualification/cpu-modern-qwen35-gguf-q8_0-text-v3.json', 'historical execution'),
               ('fixtures/qualification/cpu-modern-qwen35-gguf-q4_0-text-v3.json', 'historical execution'),
               ('fixtures/qualification/cpu-modern-retained-components-v1.json', 'retained completed components')]
    for suite in ('qwen35-gguf-q6_k_m-text', 'qwen35-text', 'gguf-text', 'embedding-f16', 'embedding-q8'):
        version = 5 if suite == 'qwen35-text' else 4
        records.append((f'fixtures/qualification/cpu-modern-{suite}-v{version}.json', 'current guarded execution'))
    suites, summaries = {}, []
    array_count = values = vectors = http = 0
    for relative, scope in records:
        record = read(ROOT / relative)
        require(record['outcome'] == 'pass' and record['target'] == native['target']
                and record['binary_sha256'] == native['binaries']['coheryn']['sha256']
                and record['production_ready'] is False, 'incomplete or incompatible component evidence')
        basis = record
        if scope == 'retained completed components':
            origin = record['source_campaign']
            require(sha(ROOT / origin['path']) == origin['sha256'], 'source campaign changed')
            basis = read(ROOT / origin['path'])
            require(basis['outcome'] == 'fail' and basis['error'] == 'qualification interrupted'
                    and record['runtime_frozen_inputs_and_numeric_validators_unchanged'], 'invalid component retention')
            require(sha(ROOT / 'target/release/examples/qwen35_probe') == record['probe_sha256'], 'retained text probe differs')
            for suite in record['selected_suites']:
                require(record['results'][suite] == basis['results'][suite], 'retained component was rewritten')
        require(basis['source_unchanged'], 'source changed during original execution/revalidation')
        require(basis['native_checks_sha256'] in natives, 'unbound native build record')
        original_native = natives[basis['native_checks_sha256']]
        require(basis['source_snapshot'] == original_native['source_snapshot'], 'original source/native binding differs')
        changed = [p for p in current.keys() | basis['source_snapshot'].keys()
                   if runtime_input(p) and current.get(p) != basis['source_snapshot'].get(p)]
        require(not changed, 'runtime/frozen inputs/numerical validators changed: ' + str(changed))
        require(set(record['results']) == set(record['selected_suites']), 'incomplete selected component set')
        host = None
        if scope == 'current guarded execution':
            require(record['source_snapshot'] == current and record['native_checks_sha256'] == sha(ROOT / native_path),
                    'guarded source evidence is stale')
            host = host_observations(record)
            summaries.append(host)
            if 'resume' in record:
                resumed = record['resume']
                require(sha(ROOT / resumed['source_record']) == resumed['source_record_sha256'], 'resume origin differs')
                origin = read(ROOT / resumed['source_record'])
                require(origin['outcome'] == 'fail' and origin['source_unchanged'] and origin['source_snapshot'] == current,
                        'invalid resumed source')
                retained = [c for c in record['commands'] if c.get('execution_scope') == 'retained successful child; not reexecuted']
                require({c['case'] for c in retained} == set(resumed['retained_children'])
                        and all(c['exit'] == 0 for c in retained), 'invalid retained child set')
        for name, result in record['results'].items():
            require(name in PROFILES and name not in suites, 'unknown or duplicate profile')
            numeric = result['numerical_and_packaged_cli']
            arrays = numeric.get('full_arrays', []) + [a for r in numeric.get('runs', []) for a in r['arrays']]
            calls = [c for c in result['http']['cases'] if 'status' in c]
            require(calls and all(c['status'] == 200 for c in calls), 'failed/incomplete HTTP component')
            suites[name] = {'record': bound(relative), 'evidence_scope': scope,
                            'original_complete_source_matches_current': basis['source_snapshot'] == current,
                            'runtime_binary_and_frozen_inputs_match_current': True,
                            'full_array_comparisons': len(arrays), 'full_array_values': sum(a['values'] for a in arrays),
                            'individual_embedding_vectors': len(numeric.get('vectors', [])),
                            'batch_configurations': numeric.get('batch_configurations', []),
                            'http_requests': len(calls), 'http_routes': sorted({c['route'] for c in calls}),
                            'host_observations': host, 'publisher_framework_parity': False}
            if 'resume' in record:
                suites[name]['resume'] = record['resume']
            array_count += len(arrays)
            values += sum(a['values'] for a in arrays)
            vectors += len(numeric.get('vectors', []))
            http += len(calls)
    require(set(suites) == set(PROFILES), 'profile matrix is incomplete')
    readiness_path = 'fixtures/qualification/cpu-modern-readiness-v4.json'
    readiness = read(ROOT / readiness_path)
    require(readiness['source_snapshot'] == current and readiness['production_ready'] is False, 'release preflight stale')
    package_path = 'fixtures/qualification/cpu-modern-package-v4.json'
    package = read(ROOT / package_path)
    require(sha(ROOT / 'target/cpu-modern-release-v4' / package['archive']) == package['sha256']
            and package['manifest']['native_check_record_sha256'] == sha(ROOT / native_path), 'package/native binding differs')
    result = {'schema_version': 1, 'kind': 'modern-cpu-local-verification', 'outcome': 'pass',
              'classification': 'experimental selected-profile evidence', 'production_ready': False,
              'target': native['target'], 'compiler': native['compiler'],
              'current_native_checks': bound(native_path), 'current_package': bound(package_path),
              'current_guarded_native_driver': bound('fixtures/qualification/cpu-modern-guarded-native-driver-v1.json'),
              'reference_bundle': bound('fixtures/cpu-qualification/modern-reference-bundle-v1.json'),
              'compatibility': bound('fixtures/compatibility.json'),
              'quantization_quality': bound('fixtures/qwen35/gguf/quantization-quality-v1.json'),
              'observer_byte_equivalence': bound('fixtures/qualification/cpu-modern-trace-equivalence-v1.json'),
              'host_interruption_assessment': bound('fixtures/qualification/cpu-modern-host-interruption-v1.json'),
              'collection_procedure': bound('fixtures/qualification/collect-modern-guarded-verification.py'),
              'summary': {'suites': len(suites), 'current_guarded_suites': 5, 'historical_completed_components': 9,
                          'full_array_comparisons': array_count, 'full_array_values': values,
                          'individual_embedding_vectors': vectors, 'http_requests': http}, 'suites': suites,
              'guarded_host_observations': summaries,
              'release_readiness': {'record': bound(readiness_path),
                                    'accepted': sum(c['status'] == 'pass' for c in readiness['checks']),
                                    'blocked': sum(c['status'] == 'blocked' for c in readiness['checks'])},
              'limitations': ['Historical completed components retain their original complete source snapshots; identical runtime/probe bytes, frozen inputs and numeric validators justify retaining those bounded results. Interrupted parent campaigns remain interrupted.',
                              'Only five unfinished profiles were executed with the revised host guard; historical image evidence does not retroactively qualify host monitoring. Polling does not guarantee prevention of OS/hardware faults.',
                              'Local macOS ARM64 CPU only; no actual Linux/Windows, GPU/Metal or publisher-framework qualification.',
                              'Short text/image fixtures at contexts 256/512. The three physical Safetensors shards are a lossless derivation, not a publisher multi-file release. Broad image/language semantics, larger configurations and all context sizes remain open.',
                              'Quantization quality is separate from executor correctness. Cache conditions uncontrolled; trace/observer overhead is not a performance benchmark. Peak RSS/private memory/faults/I/O are not qualified here.',
                              'Complete model/history, sustained resource, fuzz/security, dependency/signing and release review gates remain open.']}
    if args.report.exists():
        prior = read(args.report)
        require(prior.get('kind') == result['kind'] and prior.get('outcome') in ('running', 'incomplete')
                and prior.get('production_ready') is False, 'refuse to replace completed/unrelated evidence')
        archive = args.report.with_name('cpu-modern-verification-running-v2.json')
        require(not archive.exists(), 'prior progress archive exists')
        archive.write_bytes(args.report.read_bytes())
    checkpoint(args.report, result)
    print(json.dumps(result['summary']))


if __name__ == '__main__':
    main()

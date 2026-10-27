#!/usr/bin/env python3
"""Audit the complete Cargo graph and immutable evidence without external execution."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from check_repository_policy import ROOT, load_policy, scan


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT / 'fixtures/qualification/s0-independence.json')
    args = parser.parse_args()
    policy = load_policy()
    result = scan(ROOT, policy)
    if result['issues']:
        raise SystemExit('source/path policy failed; use the detailed policy report for triage')
    cargo = ['cmd', '/c', 'cargo'] if sys.platform == 'win32' else [str(ROOT / 'tools/cargo')]
    graphs = []
    inventory = {}
    for manifest in ['Cargo.toml', 'fuzz/Cargo.toml']:
        metadata = json.loads(subprocess.check_output([*cargo, 'metadata', '--manifest-path', manifest, '--format-version', '1', '--locked', '--offline', '--all-features'], cwd=ROOT))
        labels = {p['id']: p['name'] + '@' + p['version'] for p in metadata['packages']}
        graphs.append({'manifest': manifest, 'nodes': [{'package': labels[n['id']], 'features': n['features'], 'dependencies': sorted(labels[d] for d in n['dependencies'])} for n in metadata['resolve']['nodes']]})
        for package in metadata['packages']:
            inventory[(package['name'], package['version'], package.get('source'))] = package
    patterns = [re.compile(p) for p in policy['prohibited_package_patterns']]
    packages = []
    violations = []
    for package in inventory.values():
        if any(p.search(package['name']) for p in patterns):
            violations.append(hashlib.sha256(package['name'].encode()).hexdigest())
        packages.append({key: package.get(key) for key in ['name', 'version', 'license', 'source', 'checksum']})
    if violations:
        raise SystemExit('prohibited dependency detected; review the resource policy catalog')
    archive = json.loads((ROOT / 'resources/reference-archive/relocations.json').read_text(encoding='utf-8'))
    for item in archive['records']:
        path = (ROOT / item['archive_path']).resolve()
        if not path.is_relative_to(ROOT / 'resources/reference-archive') or path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            raise SystemExit('historical evidence bytes or relocation boundary changed')
    # No source/build integration remains for archived external builders. Native-only
    # helpers may read expected data; they cannot execute the archived source records.
    source_files = list((ROOT / 'crates').rglob('*.rs')) + list((ROOT / 'tools').glob('*.py')) + list((ROOT / '.github').rglob('*'))
    for path in source_files:
        if not path.is_file():
            continue
        text = path.read_text(encoding='utf-8')
        if re.search(r'(?:\.cache/(?:reference-oracle|embedding-oracle|quant-oracle|rope-oracle)|setup_reference\.py|setup_quant_reference\.py)', text):
            raise SystemExit('legacy external execution integration remains in controlled source')
    report = {'schema_version': 1, 'scope': 'All-feature workspace dependency graph, controlled source/path policy, archived evidence integrity. Synthetic native CLI tests separately exercise an empty executable search path. Not a signed release, legal audit, or proof about arbitrary remote deployments.',
              'lockfile_sha256': digest(ROOT / 'Cargo.lock'), 'policy_sha256': digest(ROOT / 'resources/policy/repository.json'),
              'fuzz_lockfile_sha256': digest(ROOT / 'fuzz/Cargo.lock'), 'graphs': graphs,
              'source_files_checked': result['files_checked'], 'forbidden_dependency_count': 0,
              'archive_records_verified': len(archive['records']), 'packages': sorted(packages, key=lambda p: (p['name'], p['version'])),
              'release_artifact_qualification': 'pending', 'license_review': 'declared SPDX metadata inventoried; full review remains a release gate'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'PASS native all-feature graph: {len(packages)} packages; {len(archive["records"])} original records byte-verified; controlled source policy clean.')


if __name__ == '__main__':
    main()

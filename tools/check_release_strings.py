#!/usr/bin/env python3
"""Inspect the controlled native executables, separately from build staging/caches."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from check_repository_policy import ROOT, load_policy, scan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT / 'fixtures/qualification/s0-artifacts.json')
    args = parser.parse_args()
    suffix = '.exe' if sys.platform == 'win32' else ''
    policy = load_policy()
    records = []
    with tempfile.TemporaryDirectory(prefix='coheryn-release-check-') as directory:
        staging = Path(directory)
        for name in ['coheryn', 'coheryn-bench']:
            path = ROOT / 'target/release' / (name + suffix)
            if not path.is_file():
                raise SystemExit('build the release workspace before inspecting executables')
            shutil.copyfile(path, staging / path.name)
            records.append({'file': path.name, 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        report = scan(staging, policy, artifact=True)
    if report['issues']:
        target = ROOT / 'resources/policy/artifact-findings.json'
        target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        raise SystemExit('controlled executable string policy failed; detailed findings are in the resource policy directory')
    evidence = {'schema_version': 1, 'scope': 'Current-host release executables: UTF-8 and UTF-16 in data and metadata, excluding declared instruction sections. Unknown formats receive full scanning; malformed recognized section tables fail. Not an installer/signature/notice or other-platform qualification.',
                'encoding_scopes': report['encoding_scopes'],
                'policy_sha256': hashlib.sha256((ROOT / 'resources/policy/repository.json').read_bytes()).hexdigest(),
                'files': records, 'finding_count': 0}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    print(f'PASS controlled release strings: {len(records)} native executables.')


if __name__ == '__main__':
    main()

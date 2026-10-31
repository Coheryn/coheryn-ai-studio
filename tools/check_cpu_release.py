#!/usr/bin/env python3
"""Fail closed on missing CPU qualification; never promote a candidate by label alone."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ('aarch64-apple-darwin', 'x86_64-unknown-linux-gnu', 'x86_64-pc-windows-msvc')
PROFILES = ('qwen2-gguf-text', 'nomic-bert-gguf-embedding', 'qwen2vl-gguf-paired-image', 'qwen35-safetensors-text-image', 'qwen35-gguf-text-image', 'qwen35-real-shards')
REQUIREMENTS = {'native-checks': True, 'model-matrix': True, 'resource-soak': True,
                'offline-package': True, 'fuzz-security': False,
                'dependency-review': False, 'release-review': False}
COMMANDS = (
    ('cargo', 'fmt', '--all', '--', '--check'),
    ('cargo', 'clippy', '--workspace', '--all-targets', '--locked', '--', '-D', 'warnings'),
    ('cargo', 'test', '--workspace', '--locked'),
    ('cargo', 'build', '--release', '--workspace', '--locked'),
    ('python', 'tools/test_cpu_release.py'),
    ('python', 'tools/test_cpu_qualification.py'),
    ('python', 'tools/test_cpu_shards.py'),
    ('python', 'tools/license_notices.py', '--tracked'),
    ('python', 'tools/check_repository_policy.py'),
    ('python', 'tools/check_independence.py', '--report', 'target/cpu-independence.json'),
)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_identity():
    files = [ROOT / name for name in ('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'README.md',
                                      'LICENSE', 'THIRD_PARTY_NOTICES.md', 'AGENT.md', '.gitignore')]
    for directory, suffixes in [('crates', ('.rs', '.toml')), ('tools', ('.py',)),
                                ('docs', ('.md',)), ('fuzz', ('.rs', '.toml', '.lock')),
                                ('.github/workflows', ('.yml', '.yaml'))]:
        files += [p for p in (ROOT / directory).rglob('*')
                  if p.is_file() and p.suffix in suffixes and 'target' not in p.relative_to(ROOT / directory).parts]
    # Bind frozen expected data as well as implementation. Qualification outputs and
    # evidence registration are excluded to avoid making a record hash itself.
    files += [p for p in (ROOT / 'fixtures').rglob('*') if p.is_file()
              and not p.relative_to(ROOT / 'fixtures').parts[0] in ('qualification', 'cpu-release')]
    files += [p for p in (ROOT / 'resources/policy').rglob('*') if p.is_file()
              and 'findings' not in p.name]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}


def bound_file(relative, checksum):
    path = ROOT / relative
    if (Path(relative).is_absolute() or '..' in Path(relative).parts
            or any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file()
            or path.stat().st_size > 64 << 20 or sha(path) != checksum):
        raise ValueError('missing, unsafe or changed evidence artifact: ' + str(relative))
    return path


def evaluate(scope):
    if (scope.get('schema_version') != 1 or scope.get('classification') != 'cpu-candidate'
            or scope.get('production_ready') is not False or tuple(scope.get('targets', [])) != TARGETS
            or tuple(scope.get('profiles', [])) != PROFILES
            or scope.get('compatibility_matrix') != 'fixtures/compatibility.json'):
        raise ValueError('unsupported CPU scope')
    requirements = scope['requirements']
    declared = {item['id']: item['per_target'] for item in requirements}
    if len(declared) != len(requirements) or declared != REQUIREMENTS:
        raise ValueError('required qualification gates cannot be removed or duplicated')
    source = source_identity()
    evidence = {}
    for item in scope['evidence']:
        key = (item['id'], item.get('target'))
        if item['id'] not in REQUIREMENTS or key in evidence:
            raise ValueError('unknown or duplicate evidence key')
        evidence[key] = item
    checks = []
    for requirement, per_target in REQUIREMENTS.items():
        for target in TARGETS if per_target else (None,):
            item = evidence.get((requirement, target))
            reason = 'No accepted execution/review evidence registered.'
            passed = False
            if item:
                try:
                    record = json.loads(bound_file(item['path'], item['sha256']).read_text())
                    if record.get('source_snapshot') != source:
                        raise ValueError('evidence does not bind the current complete source snapshot')
                    if record.get('compatibility_sha256') != sha(ROOT / 'fixtures/compatibility.json'):
                        raise ValueError('evidence does not bind the current compatibility matrix')
                    if requirement == 'native-checks':
                        if (record.get('kind') != 'cpu-native-checks' or record.get('target') != target
                                or record.get('outcome') != 'pass' or record.get('source_unchanged') is not True
                                or [tuple(c['command']) for c in record['commands']] != list(COMMANDS)
                                or any(c['exit'] != 0 or c.get('validated') is not True for c in record['commands'])):
                            raise ValueError('incomplete native execution record')
                        for command in record['commands']:
                            bound_file(command['log'], command['log_sha256'])
                        passed = True
                        reason = 'Exact native check commands and log/source identities verified; model qualification is separate.'
                    else:
                        reason = ('Independent model, sustained-resource, security or release review is required. '
                                  'This preflight has no validator that can accept this gate automatically.')
                except (OSError, ValueError, KeyError, TypeError) as error:
                    reason = str(error)
            checks.append({'id': requirement, 'target': target,
                           'status': 'pass' if passed else 'blocked', 'reason': reason})
    version = tomllib.loads((ROOT / 'Cargo.toml').read_text())['workspace']['package']['version']
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=ROOT))
    checks.append({'id': 'stable-source', 'target': None,
                   'status': 'blocked' if dirty or '-' in version else 'pass',
                   'reason': 'A clean stable release version is required; current version=' + version})
    return {'schema_version': 1, 'kind': 'cpu-release-readiness',
            'status': 'blocked' if any(c['status'] != 'pass' for c in checks) else 'ready-for-review',
            'production_ready': False, 'checks': checks,
            'source_snapshot': source, 'compatibility_sha256': sha(ROOT / 'fixtures/compatibility.json'),
            'scope': 'Read-only qualification preflight. This tool does not authorize a release or authenticate evidence authors.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(json.loads((ROOT / 'fixtures/cpu-release/scope.json').read_text()))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    blocked = [c for c in result['checks'] if c['status'] != 'pass']
    print(f'CPU qualification: {len(result["checks"]) - len(blocked)} checks accepted, {len(blocked)} blocked; production label retained as experimental.')
    raise SystemExit(bool(blocked))


if __name__ == '__main__':
    main()

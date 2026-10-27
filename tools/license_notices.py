#!/usr/bin/env python3
"""Collect/check pinned license records, or stage their complete distribution payload.

Collection reads checksum-verified registry archives already acquired by Cargo; it
never downloads or executes upstream code. Supplemental upstream notices are pinned
separately when a crate archive omits its license. Default checking works offline.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'resources/licensing'
MANIFESTS = ('Cargo.toml', 'fuzz/Cargo.toml')
LOCKS = ('Cargo.lock', 'fuzz/Cargo.lock')
LEGAL_NAME = re.compile(r'^(licen[cs]e|copying|notice|copyright|authors|third.party)([._-].*|$)', re.I)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def checked_file(root, relative, expected):
    path = root / relative
    if Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('license path escaped its distribution boundary')
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f'missing or linked legal record: {relative}')
    if sha(path.read_bytes()) != expected:
        raise ValueError(f'legal record checksum mismatch: {relative}')
    return path


def packages():
    cargo = ['cmd', '/c', 'cargo'] if sys.platform == 'win32' else [str(ROOT / 'tools/cargo')]
    result = {}
    for manifest in MANIFESTS:
        graph = json.loads(subprocess.check_output([*cargo, 'metadata', '--manifest-path', manifest,
                          '--format-version', '1', '--all-features', '--locked', '--offline'], cwd=ROOT))
        for package in graph['packages']:
            if package['source']:
                result[(package['name'], package['version'], package['source'])] = package
            elif not package['license']:
                raise ValueError(f'local package lacks license metadata: {package["name"]}')
    return result


def lock_checksums():
    result = {}
    for relative in LOCKS:
        for package in tomllib.loads((ROOT / relative).read_text())['package']:
            if 'source' in package:
                key = (package['name'], package['version'], package['source'])
                previous = result.setdefault(key, package['checksum'])
                if previous != package['checksum']:
                    raise ValueError('conflicting locked package checksums')
    return result


def collect():
    overrides = read_json(DIRECTORY / 'upstream/sources.json')['dependency_supplements']
    checksums = lock_checksums()
    records = []
    for key, package in sorted(packages().items()):
        name, version, source = key
        base = Path(package['manifest_path']).parent
        cache = base.parents[2] / 'cache' / base.parent.name / f'{name}-{version}.crate'
        if sha(cache.read_bytes()) != checksums[key]:
            raise ValueError(f'published crate archive checksum mismatch: {name}@{version}')
        documents = []
        with tarfile.open(cache, 'r:gz') as archive:
            for member in sorted(archive.getmembers(), key=lambda m: m.name):
                relative = Path(*Path(member.name).parts[1:])
                declared = package.get('license_file')
                if not member.isfile() or not (LEGAL_NAME.match(relative.name)
                       or (declared and relative.as_posix() == declared)):
                    continue
                if member.size > 2 << 20:
                    raise ValueError('oversized legal record')
                data = archive.extractfile(member).read()
                data.decode('utf-8')
                digest = sha(data)
                target = DIRECTORY / 'texts' / (digest + '.txt')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                documents.append({'original_path': relative.as_posix(),
                                  'path': str(target.relative_to(ROOT)), 'sha256': digest})
        for supplement in overrides.get(name + '@' + version, []):
            checked_file(ROOT, supplement['path'], supplement['sha256'])
            documents.append(supplement)
        if not documents:
            raise ValueError(f'no license/notice text for {name}@{version}')
        if not package['license']:
            raise ValueError(f'undeclared dependency license: {name}@{version}')
        records.append({'name': name, 'version': version, 'source': source,
                        'package_checksum': checksums[key], 'license': package['license'],
                        'authors': package['authors'],
                        'repository': package['repository'], 'documents': documents})
    write_json(DIRECTORY / 'dependencies.json', {
        'schema_version': 1,
        'scope': 'Conservative all-feature/all-target workspace and fuzz graph; includes build/test dependencies. SPDX expressions are retained verbatim, not rewritten as blanket permission.',
        'lockfiles': {p: sha((ROOT / p).read_bytes()) for p in LOCKS}, 'packages': records})
    retained = {record['path'] for package in records for record in package['documents']}
    for path in (DIRECTORY / 'texts').glob('*.txt'):
        if path.relative_to(ROOT).as_posix() not in retained:
            path.unlink()
    collect_toolchain()


def collect_toolchain():
    local = ROOT / '.cache/cargo/bin/rustc'
    environment = os.environ.copy()
    if local.exists():
        environment['CARGO_HOME'] = str(ROOT / '.cache/cargo')
        environment['RUSTUP_HOME'] = str(ROOT / '.cache/rustup')
    rustc = str(local) if local.exists() else 'rustc'
    sysroot = Path(subprocess.check_output([rustc, '--print', 'sysroot'], env=environment, cwd=ROOT, text=True).strip())
    version = subprocess.check_output([rustc, '-vV'], env=environment, cwd=ROOT, text=True).strip()
    docs = sysroot / 'share/doc/rust'
    sources = [docs / 'COPYRIGHT-library.html'] + sorted((docs / 'licenses').glob('*'))
    if not sources[0].is_file() or len(sources) < 2:
        raise ValueError('installed Rust distribution lacks its standard-library legal records')
    records = []
    for source in sources:
        if not source.is_file() or source.is_symlink():
            raise ValueError('invalid standard-library notice source')
        target = DIRECTORY / 'rust-standard-library' / source.relative_to(docs)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        records.append({'path': str(target.relative_to(ROOT)), 'sha256': sha(target.read_bytes()),
                        'original_path': 'share/doc/rust/' + source.relative_to(docs).as_posix()})
    write_json(DIRECTORY / 'toolchain.json', {
        'schema_version': 1, 'rustc': version,
        'toolchain_pin_sha256': sha((ROOT / 'rust-toolchain.toml').read_bytes()),
        'scope': 'Original standard-library copyright inventory and referenced license texts from the pinned installed Rust distribution. Broad source inventory does not imply all listed components are linked; the compiler executable is not redistributed. Target-specific system/SDK runtime packaging remains a release qualification.',
        'records': records})


def mirrored_paths():
    paths = set()
    for directory in ('fixtures/embedding/source', 'resources/reference-archive/fixtures/embedding/source',
                      'fixtures/vlm/llava-1.5-7b', 'fixtures/vlm/qwen2-vl-2b'):
        paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT / directory).rglob('*') if p.is_file())
    paths.update(('fixtures/tools/qwen2.5.jinja', 'fixtures/tools/qwen2.5-tools-v1.jinja',
                  'fixtures/text/qwen-chat-template.json'))
    return paths


def check():
    for relative in ('LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md', 'resources/licensing/Notice'):
        path = ROOT / relative
        if not path.is_file() or path.is_symlink() or not path.read_text(encoding='utf-8').strip():
            raise ValueError(f'required distribution legal file missing: {relative}')
    index = read_json(DIRECTORY / 'dependencies.json')
    if index['schema_version'] != 1:
        raise ValueError('unsupported dependency license inventory')
    expected = lock_checksums()
    graph = packages()
    recorded = {}
    for package in index['packages']:
        key = (package['name'], package['version'], package['source'])
        if key in recorded or key not in graph or package['package_checksum'] != expected[key]:
            raise ValueError('dependency notice inventory differs from locked graph')
        if package['license'] != graph[key]['license'] or not package['documents']:
            raise ValueError('dependency license declaration or text missing')
        recorded[key] = package
        for doc in package['documents']:
            checked_file(ROOT, doc['path'], doc['sha256'])
    if recorded.keys() != graph.keys():
        raise ValueError('dependency notice coverage is incomplete')
    if index['lockfiles'] != {p: sha((ROOT / p).read_bytes()) for p in LOCKS}:
        raise ValueError('license inventory requires regeneration after lockfile changes')
    sources = read_json(DIRECTORY / 'upstream/sources.json')
    if sources['schema_version'] != 1:
        raise ValueError('unsupported upstream legal inventory')
    for record in sources['records']:
        checked_file(ROOT, record['path'], record['sha256'])
    toolchain = read_json(DIRECTORY / 'toolchain.json')
    if toolchain['schema_version'] != 1 or toolchain['toolchain_pin_sha256'] != sha((ROOT / 'rust-toolchain.toml').read_bytes()):
        raise ValueError('standard-library notices require regeneration after toolchain changes')
    for record in toolchain['records']:
        checked_file(ROOT, record['path'], record['sha256'])
    assets = read_json(DIRECTORY / 'assets.json')
    if assets['schema_version'] != 1:
        raise ValueError('unsupported asset license inventory')
    covered = set()
    for group in assets['groups']:
        if not group['license'] or not group['source'] or not group['license_files']:
            raise ValueError('asset license/provenance mapping incomplete')
        for path in group['license_files']:
            if not (ROOT / path).is_file():
                raise ValueError('asset license text missing')
        for asset in group['files']:
            if asset['path'] in covered:
                raise ValueError('duplicate asset attribution')
            covered.add(asset['path'])
            checked_file(ROOT, asset['path'], asset['sha256'])
    if not mirrored_paths() <= covered:
        raise ValueError('new mirrored assets require attribution: ' + ', '.join(sorted(mirrored_paths() - covered)))
    archive = read_json(ROOT / 'resources/reference-archive/relocations.json')
    for record in archive['records']:
        checked_file(ROOT, record['archive_path'], record['sha256'])
    approved = set(distribution_files())
    present = {p.relative_to(ROOT).as_posix() for p in DIRECTORY.rglob('*') if p.is_file() or p.is_symlink()}
    if present - approved:
        raise ValueError('unindexed files in legal payload: ' + ', '.join(sorted(present - approved)))
    return len(recorded), len(covered)


def distribution_files():
    paths = {'LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md', 'resources/licensing/README.md',
             'resources/licensing/Notice', 'resources/licensing/assets.json',
             'resources/licensing/dependencies.json', 'resources/licensing/toolchain.json',
             'resources/licensing/upstream/sources.json'}
    for package in read_json(DIRECTORY / 'dependencies.json')['packages']:
        paths.update(record['path'] for record in package['documents'])
    for inventory in ('toolchain.json', 'upstream/sources.json'):
        paths.update(record['path'] for record in read_json(DIRECTORY / inventory)['records'])
    result = {}
    for relative in sorted(paths):
        if (Path(relative).is_absolute() or '..' in Path(relative).parts
                or not (relative in {'LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md'}
                        or relative.startswith('resources/licensing/'))):
            raise ValueError('legal payload record escaped its allowed boundary')
        path = ROOT / relative
        if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('missing or linked legal payload file')
        result[relative] = sha(path.read_bytes())
    return result


def bundle(destination):
    destination = destination.resolve()
    if not destination.is_relative_to(ROOT / 'target'):
        raise ValueError('license staging destination must be under target/')
    for relative, digest in distribution_files().items():
        target = destination / relative
        if any(p.is_symlink() for p in (target, *target.parents)):
            raise ValueError('linked staging destination')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checked_file(ROOT, relative, digest), target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect', action='store_true', help='regenerate notices from acquired locked archives')
    parser.add_argument('--bundle', type=Path, help='copy complete legal payload into a target/ staging directory')
    args = parser.parse_args()
    try:
        if args.collect:
            collect()
        packages_count, assets_count = check()
        if args.bundle:
            bundle(args.bundle)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        raise SystemExit(f'License inventory failed: {error}') from error
    print(f'PASS legal records: {packages_count} locked dependencies, {assets_count} attributed assets; archive bytes preserved.')


if __name__ == '__main__':
    main()

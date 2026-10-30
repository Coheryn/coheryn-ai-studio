#!/usr/bin/env python3
"""Verify a bounded candidate archive before extraction and smoke its native binaries."""
import argparse
import hashlib
import gzip
import json
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import tarfile
import tempfile

from package_cpu_release import ROOT, TARGETS, digest, executable_target, regular
import license_notices as legal

MAX_FILES = 4096
MAX_BYTES = 256 << 20


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate manifest key')
        result[key] = value
    return result


def verify(archive, expected_sha256, destination):
    """Extract only verified regular files into a new private empty directory.

    expected_sha256 must come from a separately trusted channel for authenticity;
    a sidecar downloaded with an archive supplies corruption detection only.
    """
    regular(archive)
    if archive.stat().st_size > MAX_BYTES or digest(archive) != expected_sha256:
        raise ValueError('archive checksum or size mismatch')
    if destination.is_symlink() or not destination.is_dir() or any(destination.iterdir()):
        raise ValueError('verification requires an empty private directory')
    # Bound decompressed tar bytes before its parser can allocate extended headers.
    with tempfile.SpooledTemporaryFile(max_size=8 << 20) as raw:
        expanded = 0
        with gzip.open(archive, 'rb') as compressed:
            while chunk := compressed.read(1 << 20):
                expanded += len(chunk)
                if expanded > MAX_BYTES:
                    raise ValueError('expanded archive size limit')
                raw.write(chunk)
        raw.seek(0)
        return unpack(raw, destination)


def unpack(raw, destination):
    with tarfile.open(fileobj=raw, mode='r:') as tar:
        members = {}
        portable_names = set()
        total = 0
        prefix = None
        # Iterate rather than getmembers(): hostile member counts stay bounded.
        for entry in tar:
            parts = PurePosixPath(entry.name).parts
            if (len(members) >= MAX_FILES or not entry.isfile() or len(parts) < 2
                    or any(x in ('', '.', '..') or any(c in x for c in '<>:"|?*\\')
                           or x.endswith((' ', '.')) or x.split('.')[0].upper() in
                           {'CON', 'PRN', 'AUX', 'NUL', *('COM' + str(n) for n in range(1, 10)),
                            *('LPT' + str(n) for n in range(1, 10))} for x in parts)
                    or entry.name.startswith('/') or entry.name != '/'.join(parts)
                    or entry.name.casefold() in portable_names or entry.size < 0):
                raise ValueError('unsafe or duplicate archive member')
            prefix = prefix or parts[0]
            total += entry.size
            if parts[0] != prefix or total > MAX_BYTES:
                raise ValueError('archive prefix or expanded size limit')
            members[entry.name] = entry
            portable_names.add(entry.name.casefold())
        manifest_path = str(prefix) + '/PACKAGE.json'
        if manifest_path not in members or members[manifest_path].size > 2 << 20:
            raise ValueError('missing or oversized package manifest')
        manifest = json.loads(tar.extractfile(members[manifest_path]).read(), object_pairs_hook=unique_object)
        if (manifest.get('schema_version') != 1 or manifest.get('classification') != 'cpu-candidate'
                or manifest.get('production_ready') is not False or manifest.get('models_bundled') is not False
                or manifest.get('target') not in TARGETS):
            raise ValueError('unsupported package manifest')
        expected = {}
        for item in manifest['files']:
            relative = item['path']
            if relative in expected or relative == 'PACKAGE.json':
                raise ValueError('duplicate manifest path')
            expected[relative] = item
        actual = {name[len(prefix) + 1:] for name in members if name != manifest_path}
        if actual != set(expected):
            raise ValueError('package file set does not match manifest')
        # Require the complete locally reviewed legal payload, including original bytes.
        for relative, checksum in legal.distribution_files().items():
            if relative not in expected or expected[relative]['sha256'] != checksum:
                raise ValueError('legal payload missing or changed: ' + relative)
        suffix = '.exe' if manifest['target'].endswith('msvc') else ''
        required = {'bin/coheryn' + suffix, 'bin/coheryn-bench' + suffix,
                    'docs/CPU_RELEASE.md', 'docs/OPERATIONS.md',
                    'fixtures/compatibility.json', 'fixtures/cpu-release/scope.json'}
        if not required <= actual:
            raise ValueError('missing required binary, contract or operator guide')
        for relative, item in expected.items():
            member = members[prefix + '/' + relative]
            if member.size != item['bytes']:
                raise ValueError('member length mismatch')
            data = tar.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != item['sha256']:
                raise ValueError('member checksum mismatch: ' + relative)
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(data)
            path.chmod(0o755 if relative.startswith('bin/') else 0o644)
        for name in ('coheryn', 'coheryn-bench'):
            executable_target(destination / 'bin' / (name + suffix), manifest['target'])
    return manifest


def native_target():
    key = (platform.system(), platform.machine().lower())
    return {('Darwin', 'arm64'): TARGETS[0], ('Linux', 'x86_64'): TARGETS[1],
            ('Windows', 'amd64'): TARGETS[2], ('Windows', 'x86_64'): TARGETS[2]}.get(key)


def smoke(directory, manifest):
    if manifest['target'] != native_target():
        raise ValueError('package must execute on its declared native platform')
    suffix = '.exe' if platform.system() == 'Windows' else ''
    environment = {key: value for key, value in os.environ.items()
                   if key in ('SystemRoot', 'WINDIR', 'TMP', 'TEMP', 'TMPDIR')}
    environment.update(PATH='', COHERYN_ENDPOINT='http://127.0.0.1:1')
    results = []
    for name, arguments in [('coheryn', ['--version']), ('coheryn', ['doctor']),
                            ('coheryn-bench', ['--help'])]:
        command = [str(directory / 'bin' / (name + suffix)), *arguments]
        process = subprocess.run(command, cwd=directory, env=environment,
                                 capture_output=True, timeout=30, check=True)
        if name == 'coheryn' and arguments == ['doctor']:
            doctor = json.loads(process.stdout)
            if doctor.get('telemetry') is not False or doctor.get('backend') != 'cpu-reference':
                raise ValueError('unexpected package capabilities')
        results.append({'command': [name, *arguments], 'exit': process.returncode,
                        'stdout_sha256': hashlib.sha256(process.stdout).hexdigest()})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    with tempfile.TemporaryDirectory(prefix='cpu-offline-') as temporary:
        directory = Path(temporary).resolve()
        manifest = verify(args.archive, args.sha256, directory)
        commands = smoke(directory, manifest)
    result = {'schema_version': 1, 'kind': 'cpu-package-smoke', 'outcome': 'pass',
              'target': manifest['target'], 'archive_sha256': digest(args.archive),
              'host': {'system': platform.system(), 'machine': platform.machine()},
              'commands': commands, 'manifest': manifest,
              'scope': 'Native version/doctor/help with empty PATH and no credentials; model inference and network isolation require separate evidence.'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('PASS verified native candidate package; full model qualification remains separate')


if __name__ == '__main__':
    main()

"""Security regression checks for candidate archives and fail-closed release claims."""
import copy
import io
import json
from pathlib import Path
import struct
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import check_cpu_release as gate
from package_cpu_release import archive_bytes, executable_target, build
import verify_cpu_package as verifier


class PackageTests(unittest.TestCase):
    def fixture(self, root):
        files = {'bin/coheryn', 'bin/coheryn-bench', 'docs/CPU_RELEASE.md',
                 'docs/OPERATIONS.md', 'fixtures/compatibility.json', 'fixtures/cpu-release/scope.json'}
        for name in files:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'\xcf\xfa\xed\xfe' + struct.pack('<I', 0x0100000c) if name.startswith('bin/') else b'fixture')
        manifest = {'schema_version': 1, 'classification': 'cpu-candidate', 'production_ready': False,
                    'models_bundled': False, 'target': gate.TARGETS[0],
                    'files': [{'path': name, 'bytes': (root / name).stat().st_size,
                               'sha256': gate.sha(root / name)} for name in sorted(files)]}
        (root / 'PACKAGE.json').write_text(json.dumps(manifest))

    def verify(self, archive, directory):
        with patch.object(verifier.legal, 'distribution_files', return_value={}):
            return verifier.verify(archive, gate.sha(archive), directory)

    def test_deterministic_archive_and_complete_verified_file_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / 'source'
            source.mkdir()
            self.fixture(source)
            first = archive_bytes(source, 'candidate')
            (source / 'bin/coheryn').chmod(0o600)
            self.assertEqual(first, archive_bytes(source, 'candidate'))
            archive = root / 'package.tgz'
            archive.write_bytes(first)
            destination = root / 'out'
            destination.mkdir()
            self.assertEqual(self.verify(archive, destination)['target'], gate.TARGETS[0])
            self.assertTrue((destination / 'bin/coheryn').is_file())

    def test_tampered_missing_extra_and_original_legal_bytes_are_rejected(self):
        for mutation in ('tamper', 'missing', 'extra', 'legal'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                source = root / 'source'
                source.mkdir()
                self.fixture(source)
                if mutation == 'tamper':
                    (source / 'bin/coheryn').write_bytes(b'changed')
                elif mutation == 'missing':
                    (source / 'docs/OPERATIONS.md').unlink()
                elif mutation == 'extra':
                    (source / 'unlisted.txt').write_text('undeclared')
                archive = root / 'bad.tgz'
                archive.write_bytes(archive_bytes(source, 'candidate'))
                destination = root / 'out'
                destination.mkdir()
                with patch.object(verifier.legal, 'distribution_files', return_value={'LICENSE': '0' * 64} if mutation == 'legal' else {}):
                    with self.assertRaises(ValueError):
                        verifier.verify(archive, gate.sha(archive), destination)

    def test_hostile_archive_members_fail_before_extraction(self):
        for name, kind in [('candidate/../escape', tarfile.REGTYPE),
                           ('/absolute/file', tarfile.REGTYPE),
                           ('candidate/link', tarfile.SYMTYPE),
                           ('candidate/CON.txt', tarfile.REGTYPE),
                           ('candidate/alternate:stream', tarfile.REGTYPE),
                           ('candidate/back\\slash', tarfile.REGTYPE)]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                archive = root / 'bad.tgz'
                with tarfile.open(archive, 'w:gz') as tar:
                    member = tarfile.TarInfo(name)
                    member.type = kind
                    member.linkname = '../../escape' if kind == tarfile.SYMTYPE else ''
                    tar.addfile(member, io.BytesIO())
                destination = root / 'out'
                destination.mkdir()
                with self.assertRaises(ValueError):
                    self.verify(archive, destination)
                self.assertFalse(list(destination.iterdir()))

    def test_archive_checksum_and_expanded_size_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            archive = root / 'bad.tgz'
            archive.write_bytes(b'not a package')
            with self.assertRaises(ValueError):
                verifier.verify(archive, '0' * 64, root)
            with tarfile.open(archive, 'w:gz') as tar:
                member = tarfile.TarInfo('candidate/large')
                member.size = 8192
                tar.addfile(member, io.BytesIO(b'\0' * 8192))
            destination = root / 'out'
            destination.mkdir()
            with patch.object(verifier, 'MAX_BYTES', 4096), self.assertRaises(ValueError):
                self.verify(archive, destination)

    def test_declared_architecture_is_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / 'executable'
            path.write_bytes(b'\xcf\xfa\xed\xfe' + struct.pack('<I', 0x0100000c))
            executable_target(path, gate.TARGETS[0])
            for target in gate.TARGETS[1:]:
                with self.assertRaises(ValueError):
                    executable_target(path, target)

    def test_package_refuses_unverified_or_stale_native_build_record(self):
        with tempfile.TemporaryDirectory(dir=gate.ROOT / 'target') as temporary:
            root = Path(temporary).resolve()
            record = root / 'record.json'
            record.write_text(json.dumps({'kind': 'cpu-native-checks', 'outcome': 'pass',
                                          'source_unchanged': True, 'target': gate.TARGETS[0],
                                          'source_snapshot': {}}))
            with self.assertRaisesRegex(ValueError, 'passing native checks'):
                build(root, root / 'out', gate.TARGETS[0], record)
            self.assertFalse((root / 'out').exists())


class ReadinessTests(unittest.TestCase):
    def scope(self):
        return json.loads((gate.ROOT / 'fixtures/cpu-release/scope.json').read_text())

    def test_missing_evidence_cannot_claim_production(self):
        result = gate.evaluate(self.scope())
        self.assertEqual(result['status'], 'blocked')
        self.assertFalse(result['production_ready'])
        self.assertTrue(any(c['id'] == 'model-matrix' and c['status'] == 'blocked' for c in result['checks']))

    def test_removing_or_duplicating_required_gate_cannot_shrink_release(self):
        for change in ('remove', 'duplicate'):
            scope = self.scope()
            if change == 'remove':
                scope['requirements'].pop()
            else:
                scope['requirements'].append(copy.deepcopy(scope['requirements'][0]))
            with self.assertRaises(ValueError):
                gate.evaluate(scope)

    def test_arbitrary_pass_or_changed_artifact_cannot_qualify(self):
        with tempfile.TemporaryDirectory(dir=gate.ROOT / 'target') as temporary:
            record = Path(temporary) / 'fake.json'
            record.write_text('{"outcome":"pass","production_ready":true}')
            scope = self.scope()
            item = {'id': 'native-checks', 'target': gate.TARGETS[0],
                    'path': record.relative_to(gate.ROOT).as_posix(), 'sha256': gate.sha(record)}
            scope['evidence'] = [item]
            for mutation in ('forged', 'tampered'):
                if mutation == 'tampered':
                    record.write_text('{}')
                result = gate.evaluate(scope)
                check = next(c for c in result['checks'] if c['id'] == 'native-checks' and c['target'] == gate.TARGETS[0])
                self.assertEqual(check['status'], 'blocked')
                self.assertFalse(result['production_ready'])


if __name__ == '__main__':
    unittest.main()

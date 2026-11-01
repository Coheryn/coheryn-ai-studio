"""Security regression checks for candidate archives and fail-closed release claims."""
import copy
import io
import json
from pathlib import Path
import struct
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

import check_cpu_release as gate
import run_cpu_checks as native_checks
import test_cpu_qualification as qualification_tests
from cpu_qualification.host import Guard
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


class NativeChecksTests(unittest.TestCase):
    def execute(self, root, *, unhealthy=False, failed=False, empty=False, filtered=False, changed=False):
        source = {'owned.rs': 'frozen'}
        report = root / 'checks.json'
        (root / 'fixtures').mkdir()
        (root / 'fixtures/compatibility.json').write_text('{}')
        (root / 'rust-toolchain.toml').write_text('[toolchain]\nchannel="1.99.1"\n')
        binary_dir = root / 'target/release'; binary_dir.mkdir(parents=True)
        suffix = '.exe' if native_checks.os.name == 'nt' else ''
        for name in ('coheryn', 'coheryn-bench'): (binary_dir / (name + suffix)).write_bytes(b'owned fixture')
        def child(command, directory, environment, log, timeout, guard, on_started):
            self.assertEqual(environment['CARGO_BUILD_JOBS'], '1')
            self.assertEqual(environment['RUST_TEST_THREADS'], '1')
            self.assertEqual(environment['CARGO_NET_OFFLINE'], 'true')
            on_started(Mock(poll=Mock(return_value=0)))
            if command[1:] == ['--version']: log.write_text('cargo 1.99.1 (fixture)\n')
            elif command[1:] == ['-vV']: log.write_text('rustc 1.99.1 (fixture)\n')
            elif command[1] == 'test':
                self.assertIn('--release', command)
                log.write_text(f'test result: ok. {0 if empty else 7} passed; 0 failed; 2 ignored; 0 measured; {int(filtered)} filtered out\n')
                return 1 if failed else 0
            else: log.write_text('native command complete\n')
            return 0
        make_guard = lambda directory: Guard(directory, sampler=lambda: {
            **qualification_tests.QualificationTests.healthy_host(), **({'thermal_state': 'serious'} if unhealthy else {})})
        with patch.object(native_checks, 'ROOT', root), \
             patch.object(native_checks, 'source_identity', side_effect=[source, {}] if changed else [source, source]), \
             patch.object(native_checks, 'native_target', return_value='aarch64-apple-darwin'), \
             patch.object(native_checks, 'Guard', side_effect=make_guard), \
             patch.object(native_checks, 'run_child', side_effect=child) as run, \
             patch.object(native_checks.shutil, 'which', return_value='virtual-rustc'), \
             patch.object(native_checks.signal, 'signal'), patch('builtins.print'), \
             patch.dict(native_checks.os.environ, {'CARGO_BUILD_TARGET': ''}), \
             patch.object(native_checks.sys, 'argv', ['checks', '--report', str(report)]):
            with self.assertRaises(SystemExit) as outcome: native_checks.main()
        return json.loads(report.read_text()), outcome.exception.code, run.call_count

    def test_complete_inventory_preserves_release_profile_logs_and_owned_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, code, calls = self.execute(Path(temporary).resolve())
            self.assertEqual((code, calls), (0, len(gate.COMMANDS) + 2))
            self.assertEqual([tuple(c['command']) for c in result['commands']], list(gate.COMMANDS))
            self.assertTrue(all(c['validated'] and c['exit'] == 0 and c['log_sha256'] for c in result['commands']))
            self.assertEqual(result['commands'][2]['tests']['passed'], 7)
            self.assertEqual(result['commands'][2]['tests']['ignored'], 2)
            self.assertTrue(result['source_unchanged'] and result['owned_cleanup_confirmed'])
            self.assertEqual(result['execution_settings']['rust_test_threads'], 1)
            self.assertFalse(result['production_ready'])

    def test_guard_rejection_starts_no_native_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, code, calls = self.execute(Path(temporary).resolve(), unhealthy=True)
            self.assertEqual((code, calls, result['outcome']), (1, 0, 'fail'))
            self.assertIn('thermal', result['host_guard']['stop_reason'])

    def test_empty_failed_filtered_or_source_changed_results_never_pass(self):
        for setting in ('empty', 'failed', 'filtered', 'changed'):
            with self.subTest(setting=setting), tempfile.TemporaryDirectory() as temporary:
                result, code, _ = self.execute(Path(temporary).resolve(), **{setting: True})
                self.assertEqual((code, result['outcome']), (1, 'fail'))
                self.assertTrue(all(c['log_sha256'] for c in result['commands']))


class SourceIdentityTests(unittest.TestCase):
    def test_desktop_metadata_is_excluded_but_hidden_and_unknown_fixture_bytes_remain_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for name in ('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'README.md',
                         'LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md', 'AGENT.md', '.gitignore', '.gitattributes'):
                (root / name).write_text('authored fixture')
            fixtures = root / 'fixtures/nested'; fixtures.mkdir(parents=True)
            policy = root / 'resources/policy'; policy.mkdir(parents=True)
            inputs = [fixtures / '.hidden-reference.bin', fixtures / '._reference.bin',
                      fixtures / '.DS_Store.reference', policy / '.hidden-policy.json']
            for path in inputs: path.write_bytes(b'bound input')
            with patch.object(gate, 'ROOT', root):
                baseline = gate.source_identity()
                for name in ('.DS_Store', 'Thumbs.db', 'desktop.ini'):
                    for directory in (fixtures, policy):
                        path = directory / name; path.write_bytes(b'desktop metadata')
                        self.assertEqual(gate.source_identity(), baseline)
                        path.write_bytes(b'changed desktop metadata')
                        self.assertEqual(gate.source_identity(), baseline)
                for path in inputs:
                    with self.subTest(path=path.name):
                        self.assertIn(path.relative_to(root).as_posix(), baseline)
                        path.write_bytes(b'changed bound input')
                        self.assertNotEqual(gate.source_identity(), baseline)
                        path.write_bytes(b'bound input')

    def test_notice_build_configuration_and_complete_legal_payload_changes_invalidate_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for name in ('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'README.md',
                         'LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md', 'AGENT.md', '.gitignore', '.gitattributes'):
                (root / name).write_bytes(b'authored fixture')
            legal = root / 'resources/licensing'
            legal.mkdir(parents=True)
            for name in ('dependencies.json', 'assets.json', 'Notice', '.hidden-notice', 'licenses/upstream.txt'):
                path = legal / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'original legal bytes')
            with patch.object(gate, 'ROOT', root):
                baseline = gate.source_identity()
                for path in (root / 'NOTICE', root / '.gitattributes', *(p for p in legal.rglob('*') if p.is_file())):
                    with self.subTest(path=path.relative_to(root).as_posix()):
                        original = path.read_bytes()
                        self.assertIn(path.relative_to(root).as_posix(), baseline)
                        path.write_bytes(b'changed source/legal bytes')
                        self.assertNotEqual(gate.source_identity(), baseline)
                        path.write_bytes(original)
                (root / '.cargo').mkdir()
                for name in ('config', 'config.toml'):
                    path = root / '.cargo' / name
                    path.write_bytes(b'build settings')
                    self.assertNotEqual(gate.source_identity(), baseline)
                    path.unlink()
                    self.assertEqual(gate.source_identity(), baseline)
                added = legal / 'additional-notice'
                added.write_bytes(b'new legal bytes')
                self.assertNotEqual(gate.source_identity(), baseline)
                added.unlink()
                self.assertEqual(gate.source_identity(), baseline)
                (legal / 'Notice').unlink()
                self.assertNotEqual(gate.source_identity(), baseline)


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

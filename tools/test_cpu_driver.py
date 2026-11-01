"""Prepared-driver provenance and no-rebuild execution regression tests."""
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import check_cpu_release
from cpu_qualification import driver
from cpu_qualification.host import Guard, Policy
import run_cpu_soak as runner
import test_cpu_qualification as qualification_tests
import test_cpu_soak as soak_tests


class PreparedDriverTests(unittest.TestCase):
    target = 'aarch64-apple-darwin'
    source = {'source.rs': 'frozen'}

    def fixture(self, root):
        (root / 'rust-toolchain.toml').write_text('[toolchain]\nchannel = "1.99.1"\n')
        path = root / 'target/release/examples/cpu_soak'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'\xcf\xfa\xed\xfe\x0c\x00\x00\x01' + b'original test executable bytes')
        healthy = qualification_tests.QualificationTests.healthy_host()
        host_log = root / '.cache/cpu-soak-checks/prepared/host-health.jsonl'
        host_log.parent.mkdir(parents=True)
        host_log.write_text(json.dumps({'sample': 1, 'observation': healthy, 'rejection': None}) + '\n')
        record = {'schema_version': 1, 'kind': 'cpu-soak-checks', 'outcome': 'pass',
                  'target': self.target, 'source_snapshot': self.source,
                  'source_unchanged': True, 'owned_cleanup_confirmed': True,
                  'toolchain': {},
                  'host_guard': {'stop_reason': None, 'stop_errors': [], 'policy': asdict(Policy()),
                                 'samples': 1, 'last': healthy, 'log': 'host-health.jsonl',
                                 'log_sha256': hashlib.sha256(host_log.read_bytes()).hexdigest()}, 'commands': [],
                  'binary': {'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size,
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}
        for name in ('cargo', 'rustc'):
            log = host_log.parent / f'{name}.log'
            text = f'{name} 1.99.1 (frozen identity)\n'
            log.write_text(text)
            record['toolchain'][name] = {'output': text, 'log': log.relative_to(root).as_posix(),
                                         'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest()}
        for index, command in enumerate(driver.CHECK_COMMANDS):
            log = root / f'logs/{index}.log'
            log.parent.mkdir(exist_ok=True)
            log.write_bytes(b'test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n')
            item = {'command': list(command), 'exit': 0, 'validated': True,
                    'log': log.relative_to(root).as_posix(),
                    'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest()}
            if command[1] == 'test':
                item['tests'] = dict(suites=1, passed=4, failed=0, ignored=0, filtered=0)
            record['commands'].append(item)
        return path, record

    def validate(self, root, record, *, source=None, target=None):
        path = root / 'prepared.json'
        path.write_text(json.dumps(record))
        with patch.object(driver, 'ROOT', root), patch.object(check_cpu_release, 'ROOT', root):
            return driver.prepared_driver(path, self.source if source is None else source,
                                          self.target if target is None else target)

    def test_complete_record_binds_source_logs_and_exact_binary_without_a_compiler(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            binary, record = self.fixture(root)
            with patch('subprocess.Popen') as spawn, patch('subprocess.run') as run:
                result = self.validate(root, record)
            self.assertEqual(result['mode'], 'verified-record')
            self.assertEqual(result['binary']['sha256'], hashlib.sha256(binary.read_bytes()).hexdigest())
            spawn.assert_not_called()
            run.assert_not_called()

    def test_changed_source_target_status_commands_or_cleanup_never_allow_reuse(self):
        for mutation in ('source', 'target', 'fetch-only', 'failed', 'unchecked', 'missing-command',
                         'changed-command', 'empty-tests', 'ignored-tests', 'filtered-tests',
                         'failed-tests', 'cleanup', 'guard', 'stop-error', 'wrong-binary-path'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                _, record = self.fixture(root)
                if mutation == 'source': record['source_snapshot'] = {}
                elif mutation == 'target': record['target'] = 'x86_64-unknown-linux-gnu'
                elif mutation == 'fetch-only': record['kind'] = 'cpu-soak-dependencies'
                elif mutation == 'failed': record['outcome'] = 'fail'
                elif mutation == 'unchecked': record['commands'][0]['validated'] = False
                elif mutation == 'missing-command': record['commands'].pop()
                elif mutation == 'changed-command': record['commands'][0]['command'].append('--ignore-rust-version')
                elif mutation == 'empty-tests': record['commands'][0]['tests']['passed'] = 0
                elif mutation.endswith('-tests'):
                    record['commands'][0]['tests'][mutation.split('-')[0]] = 1
                elif mutation == 'cleanup': record['owned_cleanup_confirmed'] = False
                elif mutation == 'guard': record['host_guard']['stop_reason'] = 'host load exceeds cutoff'
                elif mutation == 'stop-error': record['host_guard']['stop_errors'] = ['termination unconfirmed']
                else: record['binary']['path'] = 'different-binary'
                with self.assertRaises((ValueError, KeyError)):
                    self.validate(root, record)

    def test_changed_missing_or_linked_binary_and_logs_fail(self):
        for mutation in ('binary', 'log', 'missing-binary', 'missing-log', 'linked-binary', 'linked-log'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                binary, record = self.fixture(root)
                path = binary if mutation.endswith('binary') or mutation == 'binary' else root / record['commands'][0]['log']
                if mutation.startswith('missing'): path.unlink()
                elif mutation.startswith('linked'):
                    original = path.with_suffix('.original')
                    path.rename(original)
                    try: path.symlink_to(original)
                    except OSError: continue
                else: path.write_bytes(b'changed')
                with self.assertRaises((ValueError, OSError)):
                    self.validate(root, record)

    def test_oversized_malformed_and_linked_records_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            _, record = self.fixture(root)
            path = root / 'prepared.json'
            with patch.object(driver, 'ROOT', root):
                for content in (b'{' , b'[]', b' ' * ((2 << 20) + 1)):
                    path.write_bytes(content)
                    with self.assertRaises(ValueError): driver.prepared_driver(path, self.source, self.target)
                path.write_text(json.dumps(record))
                link = root / 'linked.json'
                try: link.symlink_to(path)
                except OSError: return
                with self.assertRaises(ValueError): driver.prepared_driver(link, self.source, self.target)

    def test_forged_test_totals_and_host_summaries_do_not_count_as_completed_checks(self):
        for mutation in ('test-count', 'filtered-log', 'empty-log', 'missing-host-log', 'rejected-host',
                         'truncated-host', 'host-count', 'host-policy', 'host-last', 'binary-header',
                         'missing-compiler', 'compiler-log', 'compiler-version'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                binary, record = self.fixture(root)
                log = root / record['commands'][0]['log']
                host = root / '.cache/cpu-soak-checks/prepared/host-health.jsonl'
                if mutation == 'test-count': record['commands'][0]['tests']['passed'] += 1
                elif mutation == 'filtered-log':
                    log.write_text('test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out\n')
                    record['commands'][0]['log_sha256'] = hashlib.sha256(log.read_bytes()).hexdigest()
                elif mutation == 'empty-log':
                    log.write_text('compilation only\n')
                    record['commands'][0]['log_sha256'] = hashlib.sha256(log.read_bytes()).hexdigest()
                elif mutation == 'missing-host-log': host.unlink()
                elif mutation in ('rejected-host', 'truncated-host'):
                    row = json.loads(host.read_text())
                    if mutation == 'rejected-host': row['observation']['thermal_state'] = 'serious'
                    host.write_text(json.dumps(row) + ('\n' if mutation == 'rejected-host' else ''))
                    record['host_guard']['log_sha256'] = hashlib.sha256(host.read_bytes()).hexdigest()
                elif mutation == 'host-count': record['host_guard']['samples'] += 1
                elif mutation == 'host-policy': record['host_guard']['policy']['max_load_per_cpu'] = 4
                elif mutation == 'host-last': record['host_guard']['last'] = {}
                elif mutation == 'missing-compiler': record['toolchain'].pop('rustc')
                elif mutation == 'compiler-log':
                    (root / record['toolchain']['rustc']['log']).write_text('changed identity\n')
                elif mutation == 'compiler-version':
                    log = root / record['toolchain']['rustc']['log']
                    log.write_text('rustc 1.98.0 (wrong version)\n')
                    record['toolchain']['rustc'].update(output=log.read_text(),
                                                      log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
                elif mutation == 'binary-header':
                    binary.write_bytes(b'wrong platform signature')
                    record['binary'].update(bytes=binary.stat().st_size,
                                            sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
                with self.assertRaises((ValueError, OSError)):
                    self.validate(root, record)

    def test_reuse_runs_only_the_driver_and_invalid_evidence_has_no_fallback_build(self):
        for scenario in ('valid', 'invalid-record', 'changed-before-execution'):
            valid = scenario == 'valid'
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                binary, record = self.fixture(root)
                original = soak_tests.SoakEvidenceTests()
                original.setUp()
                (root / 'fixtures').mkdir()
                (root / 'fixtures/compatibility.json').write_text('{}')
                profile = copy.deepcopy(original.profile)
                sources = {'schema_version': 1, 'kind': 'synthetic', 'text': None, 'embedding': None, 'vision': None}
                profile_path, models_path = root / 'profile.json', root / 'models.json'
                profile_path.write_text(json.dumps(profile)); models_path.write_text(json.dumps(sources))
                report = root / 'result.json'
                def child(command, directory, environment, log, timeout, guard, on_started):
                    self.assertEqual(command[0], str(binary))
                    runtime = Path(command[3]); runtime.mkdir()
                    (runtime / 'observations.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in original.rows()))
                    Path(command[4]).write_text(json.dumps({'outcome': 'pass', 'profile': profile, 'sources': sources}))
                    log.write_text('native execution\n')
                    on_started(Mock(poll=Mock(return_value=0)))
                    return 0
                preparation = {'mode': 'verified-record', 'record': 'prepared.json', 'record_sha256': '0' * 64,
                               'binary': record['binary']}
                if scenario == 'changed-before-execution': binary.write_bytes(b'changed executable')
                make_guard = lambda directory: Guard(directory, sampler=qualification_tests.QualificationTests.healthy_host)
                with patch.object(runner, 'ROOT', root), patch.object(driver, 'ROOT', root), \
                     patch.object(runner, 'source_identity', return_value=self.source), \
                     patch.object(runner, 'native_target', return_value=self.target), \
                     patch.object(runner, 'prepared_driver', return_value=preparation,
                                  side_effect=ValueError('changed prepared input') if scenario == 'invalid-record' else None), \
                     patch.object(runner, 'Guard', side_effect=make_guard), \
                     patch.object(runner, 'run_child', side_effect=child) as execute, \
                     patch.object(runner.signal, 'signal'), patch('builtins.print'), \
                     patch.object(sys, 'argv', ['soak', '--profile', str(profile_path), '--models', str(models_path),
                                              '--report', str(report), '--build-record', str(root / 'prepared.json')]):
                    with self.assertRaises(SystemExit) as result: runner.main()
                evidence = json.loads(report.read_text())
                self.assertEqual(result.exception.code, 0 if valid else 1)
                self.assertEqual(execute.call_count, 1 if valid else 0)
                self.assertEqual(evidence['outcome'], 'pass' if valid else 'fail')
                self.assertNotIn('build_exit', evidence)
                self.assertTrue(evidence['owned_cleanup_confirmed'])


if __name__ == '__main__':
    unittest.main()

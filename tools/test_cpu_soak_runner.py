"""Failure-path tests for remote soak preparation/checks; no compiler or model execution."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import check_cpu_soak as runner
import prepare_cpu_soak as preparation
from cpu_qualification.host import Guard
import test_cpu_qualification as qualification_tests


class SoakRunnerTests(unittest.TestCase):
    def run_checks(self, root, *, output=None, exit_code=0, unhealthy=False, changed=False, fetch_only=False):
        report = root / 'result.json'
        source = {'example.rs': 'frozen'}
        make_guard = lambda directory: Guard(directory, sampler=lambda: {
            **qualification_tests.QualificationTests.healthy_host(), **({'thermal_state': 'serious'} if unhealthy else {})})
        def child(command, directory, environment, log, timeout, guard, on_started):
            self.assertEqual(environment['CARGO_BUILD_JOBS'], '1')
            if command[1] in ('--version', '-vV'):
                log.write_text('toolchain identity\n')
                on_started(Mock(poll=Mock(return_value=0)))
                return 0
            log.write_text(output if output is not None else
                           'test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n')
            on_started(Mock(poll=Mock(return_value=exit_code)))
            return exit_code
        with patch.object(runner, 'ROOT', root), \
             patch.object(runner, 'source_identity', side_effect=[source, {'example.rs': 'changed'}] if changed else [source, source]), \
             patch.object(runner, 'sha', return_value='0' * 64), \
             patch.object(runner, 'native_target', return_value='aarch64-apple-darwin'), \
             patch.object(runner, 'Guard', side_effect=make_guard), \
             patch.object(runner, 'run_child', side_effect=child) as execute, \
             patch.object(runner.signal, 'signal'), \
             patch('builtins.print'), \
             patch.dict(runner.os.environ, {'CARGO_BUILD_TARGET': ''}), \
             patch.object(sys, 'argv', ['checks', '--report', str(report)] + (['--fetch-only'] if fetch_only else [])):
            with self.assertRaises(SystemExit) as done: runner.main()
        return json.loads(report.read_text()), done.exception.code, execute.call_count

    def test_complete_checks_record_executed_tests_and_do_not_claim_native_release_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, code, calls = self.run_checks(Path(temporary).resolve())
            self.assertEqual(code, 0)
            self.assertEqual(calls, len(runner.COMMANDS) + 2)
            self.assertEqual(result['kind'], 'cpu-soak-checks')
            self.assertFalse(result['production_ready'])
            self.assertEqual(result['commands'][0]['tests']['passed'], 4)
            self.assertTrue(all(c['validated'] for c in result['commands']))
            self.assertTrue(result['source_unchanged'] and result['owned_cleanup_confirmed'])
            self.assertEqual(set(result['toolchain']), {'cargo', 'rustc'})

    def test_compile_failure_preserves_log_and_prevents_later_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            result, code, calls = self.run_checks(root, output='compiler diagnostic\n', exit_code=1)
            self.assertEqual((code, calls, result['outcome']), (1, 3, 'fail'))
            self.assertFalse(result['commands'][0]['validated'])
            self.assertIn('log_sha256', result['commands'][0])
            self.assertEqual((root / result['commands'][0]['log']).read_text(), 'compiler diagnostic\n')

    def test_empty_filtered_or_failed_tests_cannot_be_accepted(self):
        for output in ('compile only\n',
                       'test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n',
                       'test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out\n',
                       'test result: FAILED. 3 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out\n'):
            with self.subTest(output=output), tempfile.TemporaryDirectory() as temporary:
                result, code, calls = self.run_checks(Path(temporary).resolve(), output=output)
                self.assertEqual((code, calls, result['outcome']), (1, 3, 'fail'))

    def test_preflight_blocks_compilation_and_keeps_failed_host_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, code, calls = self.run_checks(Path(temporary).resolve(), unhealthy=True)
            self.assertEqual((code, calls), (1, 0))
            self.assertEqual(result['commands'], [])
            self.assertIn('thermal', result['host_guard']['stop_reason'])

    def test_source_changes_invalidate_successful_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, code, calls = self.run_checks(Path(temporary).resolve(), changed=True)
            self.assertEqual(code, 1)
            self.assertEqual(calls, len(runner.COMMANDS) + 2)
            self.assertFalse(result['source_unchanged'])

    def test_fresh_runner_bootstrap_fetches_both_locked_graphs_without_compiling(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, code, calls = self.run_checks(Path(temporary).resolve(), fetch_only=True)
            self.assertEqual(code, 0)
            self.assertEqual(calls, len(runner.FETCH_COMMANDS) + 2)
            self.assertEqual(result['kind'], 'cpu-soak-dependencies')
            self.assertEqual([c['command'] for c in result['commands']], [
                ['cargo', 'fetch', '--locked'],
                ['cargo', 'fetch', '--locked', '--manifest-path', 'fuzz/Cargo.toml']])
            self.assertTrue(all(c['validated'] and 'tests' not in c for c in result['commands']))
            self.assertFalse(result['production_ready'])

    @staticmethod
    def input_fixture(root):
        pins = {}
        for name, data in [('language.gguf', b'language'), ('vision.gguf', b'vision'),
                           ('embed.gguf', b'embedding'), ('image.png', b'image')]:
            (root / name).write_bytes(data)
            pins[name] = hashlib.sha256(data).hexdigest()
        return {'schema_version': 2, 'kind': 'pinned', 'text': None,
                'text_checkpoint': {'path': str(root / 'language.gguf'),
                                    'pinned_files': {n: pins[n] for n in ('language.gguf', 'vision.gguf')}},
                'embedding': {'path': str(root / 'embed.gguf'), 'sha256': pins['embed.gguf']},
                'vision': {'kind': 'shared_text', 'image': {'path': str(root / 'image.png'), 'sha256': pins['image.png']}}}

    def test_preparation_verifies_shared_inputs_once_and_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            sources = self.input_fixture(root)
            with patch.object(preparation, 'ROOT', root):
                records = preparation.verify_sources(sources, Mock())
                self.assertEqual(len(records), 4)
                self.assertEqual(sum(r['bytes'] for r in records), 28)
                (root / 'vision.gguf').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'pinned soak input changed'):
                    preparation.verify_sources(sources, Mock())

    def test_preparation_rejects_missing_sources_and_cooperates_with_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            sources = self.input_fixture(root)
            guard = Mock()
            guard.check.side_effect = [None, RuntimeError('host stop')]
            with patch.object(preparation, 'ROOT', root):
                with self.assertRaisesRegex(RuntimeError, 'host stop'):
                    preparation.verify_sources(sources, guard)
                (root / 'language.gguf').unlink()
                with self.assertRaisesRegex(ValueError, 'missing or linked'):
                    preparation.verify_sources(sources, Mock())

    def test_preparation_host_failure_prevents_downloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            report = root / 'result.json'
            make_guard = lambda directory: Guard(directory, sampler=lambda: {
                **qualification_tests.QualificationTests.healthy_host(), 'thermal_state': 'serious'})
            with patch.object(preparation, 'ROOT', root), \
                 patch.object(preparation, 'source_identity', return_value={'model-set.json': 'frozen'}), \
                 patch.object(preparation, 'sha', return_value='0' * 64), \
                 patch.object(preparation, 'Guard', side_effect=make_guard), \
                 patch.object(preparation, 'models') as acquire, \
                 patch.object(preparation.signal, 'signal'), \
                 patch('builtins.print'), \
                 patch.object(sys, 'argv', ['prepare', '--fetch', '--report', str(report)]):
                with self.assertRaises(SystemExit) as done: preparation.main()
            self.assertEqual(done.exception.code, 1)
            acquire.assert_not_called()
            result = json.loads(report.read_text())
            self.assertEqual(result['outcome'], 'fail')
            self.assertFalse(result['execution_qualification'])

    def test_preparation_requests_only_the_frozen_modern_subset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            model_set = root / 'models.json'
            model_set.write_text(json.dumps(self.input_fixture(root)))
            report = root / 'result.json'
            make_guard = lambda directory: Guard(directory, sampler=qualification_tests.QualificationTests.healthy_host)
            with patch.object(preparation, 'ROOT', root), \
                 patch.object(preparation, 'MODEL_SET', model_set), \
                 patch.object(preparation, 'source_identity', return_value={'model-set.json': 'frozen'}), \
                 patch.object(preparation, 'Guard', side_effect=make_guard), \
                 patch.object(preparation, 'models', return_value=({}, [])) as acquire, \
                 patch.object(preparation.signal, 'signal'), patch('builtins.print'), \
                 patch.object(sys, 'argv', ['prepare', '--fetch', '--report', str(report)]):
                with self.assertRaises(SystemExit) as done: preparation.main()
            self.assertEqual(done.exception.code, 0)
            self.assertEqual(acquire.call_args.args[:2], (('embedding-q8', 'qwen35-gguf-q4_k_m-image'), True))
            result = json.loads(report.read_text())
            self.assertEqual(result['outcome'], 'pass')
            self.assertEqual(len(result['verified_inputs']), 4)
            self.assertEqual(result['managed_copy_reservation_bytes'], (128 << 20) + 23)
            self.assertFalse(result['execution_qualification'])


if __name__ == '__main__':
    unittest.main()

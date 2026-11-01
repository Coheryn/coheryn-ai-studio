"""Offline failure-injection tests; no compiler or GPU is executed."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import run_metal_checks as runner
from cpu_qualification.host import Guard
import test_cpu_qualification as qualification_tests


class MetalChecksTests(unittest.TestCase):
    def execute(self, root, mutation=None, gpu=True, cross_check=False):
        report = root / 'report.json'
        fixture = root / 'fixtures/gpu/metal-dense-v1.json'
        fixture.parent.mkdir(parents=True)
        fixture.write_text('{}')
        (root / 'rust-toolchain.toml').write_text('[toolchain]\nchannel="1.99.1"\n')
        source = {'dense.metal': 'frozen'}
        health = qualification_tests.QualificationTests.healthy_host()
        if mutation == 'unhealthy': health['memory_pressure_level'] = 2

        def child(command, directory, environment, log, timeout, guard, on_started):
            self.assertEqual(environment['CARGO_BUILD_JOBS'], '1')
            self.assertEqual(environment['CARGO_NET_OFFLINE'], 'true')
            self.assertEqual(environment['MTL_SHADER_VALIDATION'], '1')
            on_started(Mock(poll=Mock(return_value=None if mutation == 'cleanup' else 0)))
            if command[1:] == ['--version']:
                log.write_text('cargo 1.99.1 (fixture)\n')
            elif command[1:] == ['-vV']:
                host = 'x86_64-apple-darwin' if mutation == 'compiler' else 'aarch64-apple-darwin'
                if cross_check: host = 'x86_64-unknown-linux-gnu'
                log.write_text(f'rustc 1.99.1 (fixture)\nhost: {host}\n')
            elif command[1] == 'test':
                live = '--test' in command
                passed = 0 if mutation == 'empty' else 1
                ignored = int(live and mutation == 'skip')
                filtered = int(mutation == 'filtered')
                log.write_text(f'test result: ok. {passed} passed; 0 failed; {ignored} ignored; 0 measured; {filtered} filtered out\n')
                if mutation == 'failed': return 1
            else:
                log.write_text('controlled native preparation\n')
            return 0

        end_source = OSError('source read failed') if mutation == 'source-error' else ({} if mutation == 'source' else source)
        with patch.object(runner, 'ROOT', root), \
             patch.object(runner, 'source_identity', side_effect=[source, end_source]), \
             patch.object(runner.platform, 'system', return_value='Linux' if cross_check else 'Darwin'), \
             patch.object(runner.platform, 'machine', return_value='x86_64' if cross_check else 'arm64'), \
             patch.object(runner.shutil, 'which', return_value='controlled-rustc'), \
             patch.object(runner, 'Guard', side_effect=lambda directory: Guard(directory, sampler=lambda: dict(health))), \
             patch.object(runner, 'run_child', side_effect=child) as run, \
             patch.object(runner.signal, 'signal'), patch('builtins.print'), \
             patch.dict(runner.os.environ, {'CARGO_BUILD_TARGET': ''}), \
             patch.object(runner.sys, 'argv', ['metal', '--report', str(report), *(['--gpu'] if gpu else []), *(['--cross-check'] if cross_check else [])]):
            with self.assertRaises(SystemExit) as outcome: runner.main()
        return json.loads(report.read_text()), outcome.exception.code, run.call_count

    def test_live_mode_requires_exact_test_and_preserves_logs_scope_and_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            report, code, calls = self.execute(Path(temporary).resolve())
            self.assertEqual((code, calls), (0, 8))
            self.assertTrue(report['gpu_execution_required'])
            self.assertFalse(report['production_ready'])
            self.assertTrue(report['source_unchanged'] and report['owned_cleanup_confirmed'])
            for row in report['commands']:
                self.assertTrue(row['validated'])
                self.assertEqual(row['log_sha256'], runner.sha(Path(temporary) / row['log']))

    def test_compile_mode_cannot_claim_gpu_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            report, code, calls = self.execute(Path(temporary).resolve(), gpu=False)
            self.assertEqual((code, calls), (0, 7))
            self.assertFalse(report['gpu_execution_required'])
            self.assertTrue(all('--test' not in row['command'] for row in report['commands']))

    def test_unhealthy_host_starts_no_native_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            report, code, calls = self.execute(Path(temporary).resolve(), 'unhealthy')
            self.assertEqual((code, calls, report['outcome']), (1, 0, 'fail'))

    def test_explicit_cross_check_compiles_foreign_target_without_executing_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            report, code, calls = self.execute(Path(temporary).resolve(), gpu=False, cross_check=True)
            self.assertEqual((code, calls), (0, 5))
            self.assertEqual(report['kind'], 'metal-cross-check')
            self.assertFalse(report['gpu_execution_required'])
            self.assertTrue(report['cross_check'])
            self.assertEqual(report['commands'][1]['command'][-5:], ['--target', 'aarch64-apple-darwin', '--', '-D', 'warnings'])
            self.assertTrue(all(row['command'][1] != 'test' for row in report['commands']))

    def test_skipped_filtered_failed_changed_or_unretired_work_cannot_pass(self):
        for mutation in ('skip', 'empty', 'filtered', 'failed', 'source', 'source-error', 'cleanup', 'compiler'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                report, code, _ = self.execute(Path(temporary).resolve(), mutation)
                self.assertEqual((code, report['outcome']), (1, 'fail'))


if __name__ == '__main__':
    unittest.main()

"""Fuzz lifecycle and evidence regressions without a compiler or sanitizer."""
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import run_cpu_fuzz as runner
from cpu_qualification.host import Guard
import test_cpu_qualification as qualification_tests


class FuzzChecks(unittest.TestCase):
    def execute(self, root, *, mutation=None, prepare=False):
        report = root / 'report.json'
        (root / 'fuzz').mkdir()
        (root / 'fuzz/Cargo.lock').write_text('controlled lock fixture')
        (root / 'rust-toolchain.toml').write_text('[toolchain]\nchannel = "1.99.1"\n')
        source = {'fuzz/Cargo.lock': 'frozen'}
        health = qualification_tests.QualificationTests.healthy_host()
        if mutation == 'unhealthy': health['memory_pressure_level'] = 2
        observed = []

        def child(command, directory, environment, log, timeout, guard, on_started):
            guard.sample()
            self.assertEqual(environment['CARGO_BUILD_JOBS'], '1')
            online = command[1:2] in (['install'], ['fetch'], ['toolchain'])
            self.assertEqual(environment['CARGO_NET_OFFLINE'], 'false' if online else 'true')
            self.assertLessEqual(timeout, 3600)
            on_started(Mock(poll=Mock(return_value=None if mutation == 'cleanup' else 0)))
            observed.append(command)
            if command[-1] == '-vV':
                log.write_text('rustc 1.101.0-nightly (fixture)\n' if mutation != 'stable' else 'rustc 1.99.1 (fixture)\n')
            elif command[-1] == '--version':
                log.write_text('cargo-fuzz ' + ('0.0.0' if mutation == 'tool' else runner.CARGO_FUZZ) + '\n')
            elif 'run' not in command:
                log.write_text('controlled preparation\n')
                if mutation == 'prepare-failed': return 1
            else:
                text = 'INFO: Loaded 1 modules (123 inline 8-bit counters)\n#100 DONE cov: 42 ft: 65\n'
                if mutation == 'short': text = text.replace('#100', '#99')
                if mutation == 'coverage': text = text.replace('cov: 42', 'cov: 0')
                if mutation == 'features': text = text.replace('ft: 65', 'ft: 0')
                if mutation == 'instrumentation': text = text.split('\n', 1)[1]
                if mutation == 'incomplete': text = 'INFO: Loaded 1 modules (123 inline 8-bit counters)\n'
                if mutation == 'duplicate': text += '#100 DONE cov: 42 ft: 65\n'
                log.write_text(text)
                if mutation == 'interrupted': raise KeyboardInterrupt('controlled interruption')
                if mutation == 'timeout': raise subprocess.TimeoutExpired(command, timeout)
                if mutation == 'pressure':
                    health['memory_pressure_level'] = 2
                    guard.sample()
                if mutation == 'failed': return 1
            return 0

        def seeds(target, guard):
            guard.check()
            path = root / 'fuzz/corpus' / target
            path.mkdir(parents=True)
            file = path / 'seed'
            file.write_bytes(b'controlled immutable seed')
            return path, {'seed': runner.sha(file)}

        end_source = OSError('controlled source read failure') if mutation == 'source-error' else (
            {} if mutation == 'source' else source)
        arguments = ['fuzz', '--toolchain', 'nightly-2026-10-30', '--runs', '100', '--report', str(report)]
        if prepare: arguments.append('--prepare')
        with patch.object(runner, 'ROOT', root), \
             patch.object(runner, 'source_identity', side_effect=[source, end_source]), \
             patch.object(runner, 'Guard', side_effect=lambda directory: Guard(directory, sampler=lambda: dict(health))), \
             patch.object(runner, 'run_child', side_effect=child) as run, \
             patch.object(runner, 'corpus', side_effect=seeds) as seed, \
             patch.object(runner.shutil, 'which', side_effect=lambda name: name), \
             patch.object(runner.platform, 'system', return_value='Darwin'), \
             patch.object(runner.signal, 'signal'), patch('builtins.print'), \
             patch.dict(runner.os.environ, {'CARGO_BUILD_TARGET': ''}), \
             patch.object(sys, 'argv', arguments):
            with self.assertRaises(SystemExit) as outcome: runner.main()
        return json.loads(report.read_text()), outcome.exception.code, run.call_count, seed.call_count, observed

    def test_all_targets_require_complete_instrumented_results_and_clean_retirement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            result, code, calls, seeds, _ = self.execute(root)
            self.assertEqual((code, calls, seeds), (0, 7, 5))
            self.assertEqual(result['outcome'], 'pass')
            self.assertFalse(result['production_ready'])
            self.assertTrue(result['source_unchanged'] and result['owned_cleanup_confirmed'])
            self.assertEqual([row['target'] for row in result['runs']], list(runner.TARGETS))
            self.assertTrue(all(row['executed_iterations'] == 100 and row['edges'] == 42 for row in result['runs']))
            for command in result['commands']:
                self.assertTrue(command['validated'])
                self.assertEqual(runner.sha(root / command['log']), command['log_sha256'])

    def test_preparation_is_explicit_guarded_and_version_pinned(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, code, calls, _, commands = self.execute(Path(temporary).resolve(), prepare=True)
            self.assertEqual((code, calls), (0, 11))
            self.assertTrue(result['execution_settings']['prepare'])
            self.assertEqual(commands[0][1:], ['toolchain', 'install', '1.99.1', '--profile', 'minimal'])
            self.assertEqual(commands[1][1:], ['toolchain', 'install', 'nightly-2026-10-30', '--profile', 'minimal'])
            self.assertEqual(commands[2][1:], ['install', 'cargo-fuzz', '--version', runner.CARGO_FUZZ, '--locked'])
            self.assertEqual(commands[3][1:], ['fetch', '--manifest-path', 'fuzz/Cargo.toml', '--locked'])

    def test_failed_preparation_preserves_diagnostics_and_starts_no_campaign(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, code, calls, seeds, _ = self.execute(Path(temporary).resolve(), mutation='prepare-failed', prepare=True)
            self.assertEqual((code, calls, seeds), (1, 1, 0))
            self.assertEqual(result['runs'], [])
            self.assertFalse(result['commands'][0]['validated'])
            self.assertIn('log_sha256', result['commands'][0])

    def test_initial_host_rejection_starts_no_install_compiler_seed_or_campaign(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, code, calls, seeds, _ = self.execute(Path(temporary).resolve(), mutation='unhealthy', prepare=True)
            self.assertEqual((code, calls, seeds), (1, 0, 0))
            self.assertEqual(result['outcome'], 'fail')
            self.assertIn('memory pressure', result['host_guard']['stop_reason'])

    def test_failures_and_partial_or_untrusted_evidence_never_pass(self):
        for mutation in ('short', 'coverage', 'features', 'instrumentation', 'incomplete', 'duplicate',
                         'interrupted', 'timeout', 'pressure', 'failed', 'stable', 'tool', 'cleanup', 'source', 'source-error'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                result, code, _, _, _ = self.execute(root, mutation=mutation)
                self.assertEqual((code, result['outcome']), (1, 'fail'))
                self.assertTrue(all('log_sha256' in command for command in result['commands']))
                if mutation in ('short', 'interrupted', 'timeout', 'pressure', 'failed'):
                    self.assertEqual(len(result['runs']), 1)
                    self.assertEqual(result['runs'][0]['outcome'], 'fail')
                self.assertTrue((root / '.cache/cpu-fuzz/report/host-health.jsonl').exists())

    def test_corpus_preserves_seed_bytes_and_rejects_links_and_inventory_exhaustion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            guard = Mock()
            with patch.object(runner, 'ROOT', root):
                path, original = runner.corpus('gguf', guard)
                self.assertTrue(original)
                self.assertEqual(runner.corpus('gguf', guard), (path, original))
                seed = b'{}'
                (path / hashlib.sha256(seed).hexdigest()).write_bytes(b'changed bytes')
                with self.assertRaisesRegex(ValueError, 'immutable'): runner.corpus('gguf', guard)
                with patch.object(runner, 'MAX_CORPUS_FILES', 1):
                    with self.assertRaisesRegex(ValueError, 'limits'): runner.inventory(path, guard)
                with patch.object(runner, 'MAX_CORPUS_BYTES', 1):
                    with self.assertRaisesRegex(ValueError, 'limits'): runner.inventory(path, guard)
                if sys.platform != 'win32':
                    (path / 'linked-seed').symlink_to(root / 'outside-seed')
                    with self.assertRaisesRegex(ValueError, 'unlinked'): runner.inventory(path, guard)
                    (path / 'linked-seed').unlink()
                    linked = root / 'fuzz/corpus/tools'
                    linked.symlink_to(path, target_is_directory=True)
                    with self.assertRaisesRegex(ValueError, 'linked'): runner.corpus('tools', guard)

    def test_invalid_requests_fail_before_work_begins(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for extra in (['--runs', '0'], ['--runs', '100000001'], ['--toolchain', 'nightly'],
                          ['--target', 'gguf', '--target', 'gguf']):
                with self.subTest(extra=extra), patch.object(runner, 'Guard') as guard, \
                     patch.object(sys, 'argv', ['fuzz', '--toolchain', 'nightly-2026-10-30', '--report', str(root / 'r.json'), *extra]), \
                     patch.object(sys, 'stderr'):
                    with self.assertRaises(SystemExit) as result: runner.main()
                    self.assertEqual(result.exception.code, 2)
                    guard.assert_not_called()


if __name__ == '__main__':
    unittest.main()

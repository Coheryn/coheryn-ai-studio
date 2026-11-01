"""Offline failure-injection tests; no compiler or GPU is executed."""
import json
import copy
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
        (fixture.parent / 'metal-packed-v1.json').write_text('{}')
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
                if mutation == 'format' and command[1] == 'fmt': return 1
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
            self.assertEqual((code, calls), (0, 10))
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

    def test_format_failure_retains_numeric_feedback_but_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            report, code, calls = self.execute(Path(temporary).resolve(), 'format')
            self.assertEqual((code, calls, report['outcome']), (1, 10, 'fail'))
            self.assertFalse(report['commands'][0]['validated'])
            self.assertTrue(all(row['validated'] for row in report['commands'][1:]))

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


class ModelOutputTests(unittest.TestCase):
    def fixture(self, suite):
        reference = json.loads((runner.ROOT / runner.PROFILES[suite]['reference']).read_text())
        return {'schema_version': 1, 'outcome': 'pass', 'suite': suite,
                'full_cpu_gpu_arrays': 50, 'full_cpu_gpu_elements': 545792,
                'independent_sampled_values': sum(len(checkpoint['indices']) for case in reference['cases']
                    for checkpoint in [*case['checkpoints'], *case['teacher_forced']]),
                'cases': [{'token_ids': case['greedy_token_ids'], 'text': case['greedy_text'],
                           'completion_tokens': len(case['greedy_token_ids']),
                           'sampling_metadata': {'backend': 'native-metal-matmul-f32-reference'}} for case in reference['cases']],
                'gpu': {'device': 'controlled test fixture', 'matrices': 1, 'encoded_weight_bytes': 1024,
                        'request_workspace_bytes': 512, 'completed_model_dispatches': 1},
                'memory_after_unload': {'reserved': 0, 'session_buffer_capacity_bytes': 0, 'peak_reserved': 2048, 'limit': 4096},
                'trace_failure_cleanup': True, 'in_graph_cancellation_cleanup': True,
                'unqualified_image_profile_rejected': True, 'gpu_plan_prepaid': True}

    def test_complete_model_contract_keeps_source_and_runtime_scope_separate(self):
        for suite in runner.MODEL_SUITES:
            with self.subTest(suite=suite), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'model.json'; value = self.fixture(suite)
                path.write_text(json.dumps(value))
                self.assertEqual(runner.validate_model(path, suite), value)

    def test_partial_wrong_backend_nonnumeric_or_unreclaimed_model_cannot_pass(self):
        base = self.fixture(runner.MODEL_SUITES[0])
        for mutation in ('arrays', 'elements', 'samples', 'greedy', 'backend', 'dispatches', 'leak', 'boolean', 'cancel', 'plan', 'missing_plan', 'empty'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                value = copy.deepcopy(base)
                if mutation == 'arrays': value['full_cpu_gpu_arrays'] -= 1
                elif mutation == 'elements': value['full_cpu_gpu_elements'] = 0
                elif mutation == 'samples': value['independent_sampled_values'] -= 1
                elif mutation == 'greedy': value['cases'][0]['token_ids'][0] += 1
                elif mutation == 'backend': value['cases'][0]['sampling_metadata']['backend'] = 'native-cpu'
                elif mutation == 'dispatches': value['gpu']['completed_model_dispatches'] = 0
                elif mutation == 'leak': value['memory_after_unload']['reserved'] = 1
                elif mutation == 'boolean': value['gpu']['encoded_weight_bytes'] = True
                elif mutation == 'cancel': value['in_graph_cancellation_cleanup'] = False
                elif mutation == 'plan': value['gpu_plan_prepaid'] = False
                elif mutation == 'missing_plan': del value['gpu_plan_prepaid']
                elif mutation == 'empty': value = []
                path = Path(temporary) / 'model.json'; path.write_text(json.dumps(value))
                with self.assertRaises(ValueError): runner.validate_model(path, runner.MODEL_SUITES[0])


if __name__ == '__main__':
    unittest.main()

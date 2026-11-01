"""Benchmark coverage and host-failure regressions without compilers or models."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import check_text_bench as runner
from cpu_qualification.host import Guard
import test_cpu_qualification as qualification_tests


def sampling_result():
    return {'token_ids': [1, 2], 'text': 'sample', 'completion_tokens': 2,
            'sampling_metadata': {'profile': 'coheryn-sampling-v2', 'threads': 1, 'backend': 'native-cpu', 'cache_dtype': 'f32',
                'settings': {'temperature': 0.8, 'top_k': 8, 'top_p': 0.9, 'seed': 73},
                'penalties': {'repetition_penalty': 1.1, 'frequency_penalty': 0.1, 'presence_penalty': 0.2}}}


def api_result():
    return {'schema_version': 1, 'outcome': 'pass',
            'sampling': {'concurrent_results': [sampling_result(), sampling_result()],
                         'image_result': sampling_result(), 'image_stream_parity': True,
                         'cached_result': {**sampling_result(), 'reused_prompt_tokens': 4}, 'fresh_result': sampling_result()},
            'model_manifest_sha256': '0' * 64, 'image_reference_sha256': '0' * 64,
            'test_executable_sha256': '1' * 64, 'external_executor_invoked': False,
            'final_memory': {'reserved': 0, 'session_buffer_capacity_bytes': 0}}


def native_result(suite):
    rows = []
    for entry in range(3 if suite == 'qwen35-text' else 1):
        for case in ('english', 'bengali' if suite == 'gguf-text' else 'unicode'):
            for workers in (1, 2):
                run = dict.fromkeys(('prefill_ms', 'decode_ms', 'tokenization_ms', 'output_delivery_ms', 'first_token_ms'), 1.)
                rows.append({'entrypoint': entry, 'case': case, 'workers': workers,
                             'benchmark': {'mode': 'standalone', 'cache_condition': 'uncontrolled',
                                           'runs': [copy.deepcopy(run), copy.deepcopy(run)]}})
    return {'schema_version': 1, 'outcome': 'pass', 'suite': suite,
            'source_entrypoints': 3 if suite == 'qwen35-text' else 1, 'cases': 2,
            'threads': [1, 2], 'runs': rows,
            'sampling_runs': [{'entrypoint': i, 'results': [sampling_result(), sampling_result()]}
                              for i in range(3 if suite == 'qwen35-text' else 1)] if suite != 'gguf-text' else []}


class BenchmarkChecks(unittest.TestCase):
    def test_seeded_sampling_requires_every_entrypoint_exact_controls_and_repeatability(self):
        for mutation in ('missing', 'repeated', 'seed', 'nonfinite', 'tokens', 'negative', 'boolean', 'profile', 'settings', 'entry'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                value = native_result('qwen35-text')
                result = value['sampling_runs'][0]['results'][0]
                if mutation == 'missing': value['sampling_runs'].pop()
                elif mutation == 'repeated': result['text'] = 'different'
                elif mutation == 'seed': result['sampling_metadata']['settings']['seed'] = 74
                elif mutation == 'nonfinite': result['sampling_metadata']['settings']['top_p'] = float('nan')
                elif mutation == 'tokens': result['completion_tokens'] = 3
                elif mutation == 'negative': result['token_ids'][0] = -1
                elif mutation == 'boolean': result['token_ids'][0] = True
                elif mutation == 'profile': result['sampling_metadata']['profile'] = 'different'
                elif mutation == 'settings': result['sampling_metadata'].pop('penalties')
                else: value['sampling_runs'][0]['entrypoint'] = True
                path = Path(temporary) / 'native.json'; path.write_text(json.dumps(value))
                with self.assertRaises(ValueError): runner.validate_native(path, 'qwen35-text')

    def test_sampling_api_requires_text_image_stream_identity_and_complete_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(runner, 'sha', return_value='0' * 64):
            path = Path(temporary) / 'api.json'
            value = api_result(); path.write_text(json.dumps(value))
            self.assertEqual(runner.validate_api(path), value)
            for mutation in ('status', 'missing', 'stream', 'identity', 'executable', 'cleanup', 'output', 'external'):
                value = api_result()
                if mutation == 'status': value['outcome'] = 'running'
                elif mutation == 'missing': value['sampling'].pop('image_result')
                elif mutation == 'stream': value['sampling']['image_stream_parity'] = False
                elif mutation == 'identity': value['model_manifest_sha256'] = '1' * 64
                elif mutation == 'executable': value['test_executable_sha256'] = ''
                elif mutation == 'cleanup': value['final_memory']['session_buffer_capacity_bytes'] = 1
                elif mutation == 'output': value['sampling']['concurrent_results'][0]['text'] = 'different'
                else: value['external_executor_invoked'] = True
                path.write_text(json.dumps(value))
                with self.subTest(mutation=mutation), self.assertRaises(ValueError): runner.validate_api(path)

    def test_each_format_requires_complete_entrypoint_case_worker_and_iteration_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'native.json'
            for suite in runner.SUITES:
                value = native_result(suite)
                path.write_text(json.dumps(value))
                self.assertEqual(runner.validate_native(path, suite), value)

    def test_missing_duplicate_misidentified_nonfinite_and_oversized_results_fail(self):
        for mutation in ('missing', 'duplicate', 'suite', 'mode', 'iterations', 'timing', 'boolean', 'case', 'entry'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'native.json'
                value = native_result('qwen35-text')
                if mutation == 'missing': value['runs'].pop()
                elif mutation == 'duplicate': value['runs'][-1] = copy.deepcopy(value['runs'][0])
                elif mutation == 'suite': value['suite'] = 'different'
                elif mutation == 'mode': value['runs'][0]['benchmark']['mode'] = 'daemon-client'
                elif mutation == 'iterations': value['runs'][0]['benchmark']['runs'].pop()
                elif mutation == 'timing': value['runs'][0]['benchmark']['runs'][0]['decode_ms'] = float('nan')
                elif mutation == 'boolean': value['threads'][0] = True
                elif mutation == 'case': value['runs'][0]['case'] = 'unqualified'
                else: value['runs'][0]['entrypoint'] = -1
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError): runner.validate_native(path, 'qwen35-text')
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'large.json'; path.write_bytes(b' ' * ((2 << 20) + 1))
                with self.assertRaises(ValueError): runner.validate_native(path, 'qwen35-text')

    def execute(self, root, *, unhealthy=False, filtered=False, failed=False, changed=False, missing=False,
                api_failed=False, api_filtered=False, api_missing=False):
        source = {'benchmark.rs': 'frozen'}
        report = root / 'report.json'
        (root / 'fixtures').mkdir()
        (root / 'fixtures/compatibility.json').write_text('{}')
        def child(command, directory, environment, log, timeout, guard, on_started):
            self.assertEqual(environment['CARGO_BUILD_JOBS'], '1')
            on_started(Mock(poll=Mock(return_value=0)))
            if command[1] in ('--version', '-vV'):
                log.write_text('compiler identity\n'); return 0
            if tuple(command[1:]) == runner.API_COMMAND:
                log.write_text(f'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; {int(api_filtered)} filtered out\n')
                if not api_missing:
                    Path(environment['COHERYN_Q35_REPORT']).write_text(json.dumps(api_result()))
                return 1 if api_failed else 0
            self.assertEqual(tuple(command[1:]), runner.COMMAND)
            log.write_text(f'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; {int(filtered)} filtered out\n')
            if not missing:
                Path(environment['COHERYN_TEXT_BENCH_REPORT']).write_text(json.dumps(native_result('qwen35-text')))
            return 1 if failed else 0
        make_guard = lambda directory: Guard(directory, sampler=lambda: {
            **qualification_tests.QualificationTests.healthy_host(), **({'thermal_state': 'serious'} if unhealthy else {})})
        with patch.object(runner, 'ROOT', root), \
             patch.object(runner, 'source_identity', side_effect=[source, {}] if changed else [source, source]), \
             patch.object(runner, 'native_target', return_value='aarch64-apple-darwin'), \
             patch.object(runner, 'Guard', side_effect=make_guard), \
             patch.object(runner, 'models', return_value=({}, [])) as acquire, \
             patch.object(runner, 'run_child', side_effect=child) as run, \
             patch.object(runner, 'executable_target'), patch.object(runner, 'sha', return_value='0' * 64), \
             patch.object(runner.signal, 'signal'), patch('builtins.print'), \
             patch.dict(runner.os.environ, {'CARGO_BUILD_TARGET': ''}), \
             patch.object(sys, 'argv', ['checks', '--suite', 'qwen35-text', '--report', str(report)]):
            with self.assertRaises(SystemExit) as outcome: runner.main()
        return json.loads(report.read_text()), outcome.exception.code, run.call_count, acquire.call_count

    def test_complete_native_benchmark_evidence_preserves_exact_source_and_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, code, calls, acquisitions = self.execute(Path(temporary).resolve())
            self.assertEqual((code, calls, acquisitions), (0, 4, 1))
            self.assertEqual(result['outcome'], 'pass')
            self.assertFalse(result['production_ready'])
            self.assertTrue(result['owned_cleanup_confirmed'] and result['source_unchanged'])
            self.assertEqual(result['commands'][0]['tests']['passed'], 1)
            self.assertEqual(set(result['toolchain']), {'cargo', 'rustc'})

    def test_host_rejection_starts_neither_acquisition_nor_a_compiler(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, code, calls, acquisitions = self.execute(Path(temporary).resolve(), unhealthy=True)
            self.assertEqual((code, calls, acquisitions), (1, 0, 0))
            self.assertEqual(result['outcome'], 'fail')
            self.assertIn('thermal', result['host_guard']['stop_reason'])

    def test_failed_filtered_missing_and_changed_results_never_pass(self):
        for setting in ('filtered', 'failed', 'changed', 'missing'):
            with self.subTest(setting=setting), tempfile.TemporaryDirectory() as temporary:
                result, code, _, _ = self.execute(Path(temporary).resolve(), **{setting: True})
                self.assertEqual((code, result['outcome']), (1, 'fail'))
                self.assertIn('log_sha256', result['commands'][0])

    def test_partial_api_execution_cannot_promote_passing_cli_results(self):
        for setting in ('api_failed', 'api_filtered', 'api_missing'):
            with self.subTest(setting=setting), tempfile.TemporaryDirectory() as temporary:
                result, code, _, _ = self.execute(Path(temporary).resolve(), **{setting: True})
                self.assertEqual((code, result['outcome']), (1, 'fail'))
                self.assertEqual(len(result['commands']), 2)
                self.assertFalse(result['commands'][-1]['validated'])
                self.assertTrue(result['owned_cleanup_confirmed'])


if __name__ == '__main__':
    unittest.main()

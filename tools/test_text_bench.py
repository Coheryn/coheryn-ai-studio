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
            'threads': [1, 2], 'runs': rows}


class BenchmarkChecks(unittest.TestCase):
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

    def execute(self, root, *, unhealthy=False, filtered=False, failed=False, changed=False, missing=False):
        source = {'benchmark.rs': 'frozen'}
        report = root / 'report.json'
        (root / 'fixtures').mkdir()
        (root / 'fixtures/compatibility.json').write_text('{}')
        def child(command, directory, environment, log, timeout, guard, on_started):
            self.assertEqual(environment['CARGO_BUILD_JOBS'], '1')
            on_started(Mock(poll=Mock(return_value=0)))
            if command[1] in ('--version', '-vV'):
                log.write_text('compiler identity\n'); return 0
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
            self.assertEqual((code, calls, acquisitions), (0, 3, 1))
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


if __name__ == '__main__':
    unittest.main()

"""Kernel evidence and admission regressions without native compilation or inference."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import check_cpu_dot as runner
from cpu_qualification.host import Guard
import test_cpu_qualification as qualification_tests


def native():
    return {'schema_version': 1, 'kind': 'cpu-dot-observations', 'outcome': 'pass', 'threads': 1,
            'seed': '57024f001e5b90e3', 'cases': [
                {'width': width, 'cancellation': cancellation, 'rows': 64, 'repeats': 128, 'samples': 7,
                 'bitwise_parity': True, 'reference_ms': [2.] * 7, 'candidate_ms': [1.] * 7}
                for width, cancellation in runner.CASES]}


class DotChecks(unittest.TestCase):
    def test_complete_observations_require_all_cases_and_finite_positive_samples(self):
        for mutation in ('valid', 'missing', 'duplicate', 'parity', 'nan', 'zero', 'samples', 'bool', 'oversized'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'native.json'; value = native()
                if mutation == 'missing': value['cases'].pop()
                elif mutation == 'duplicate': value['cases'][-1] = copy.deepcopy(value['cases'][0])
                elif mutation == 'parity': value['cases'][0]['bitwise_parity'] = False
                elif mutation == 'nan': value['cases'][0]['candidate_ms'][0] = float('nan')
                elif mutation == 'zero': value['cases'][0]['reference_ms'][0] = 0
                elif mutation == 'samples': value['cases'][0]['candidate_ms'].pop()
                elif mutation == 'bool': value['threads'] = True
                path.write_text(json.dumps(value))
                if mutation == 'oversized': path.write_bytes(b' ' * ((1 << 20) + 1))
                if mutation == 'valid': self.assertEqual(runner.observations(path), value)
                else:
                    with self.assertRaises(ValueError): runner.observations(path)

    def execute(self, root, *, unhealthy=False, failed=False, changed=False, incomplete=False):
        source = {'cpu.rs': 'frozen'}
        report = root / 'report.json'
        def child(command, directory, environment, log, timeout, guard, on_started):
            self.assertEqual(environment['CARGO_BUILD_JOBS'], '1')
            self.assertEqual(environment['CARGO_NET_OFFLINE'], 'true')
            on_started(Mock(poll=Mock(return_value=0)))
            if command[1:] in (['--version'], ['-vV']):
                log.write_text('compiler identity\n')
            elif tuple(command[1:]) == runner.BUILD:
                log.write_text('build complete\n')
                if failed: return 1
                binary = root / 'target/release/examples/dot_bench'
                binary.parent.mkdir(parents=True)
                binary.write_bytes(b'controlled native executable fixture')
            else:
                value = native()
                if incomplete: value['cases'].pop()
                log.write_text(json.dumps(value))
            return 0
        make_guard = lambda directory: Guard(directory, sampler=lambda: {
            **qualification_tests.QualificationTests.healthy_host(), **({'memory_pressure_level': 2} if unhealthy else {})})
        with patch.object(runner, 'ROOT', root), \
             patch.object(runner, 'source_identity', side_effect=[source, {}] if changed else [source, source]), \
             patch.object(runner, 'native_target', return_value='aarch64-apple-darwin'), \
             patch.object(runner, 'Guard', side_effect=make_guard), \
             patch.object(runner, 'run_child', side_effect=child) as run, \
             patch.object(runner, 'executable_target'), patch.object(runner, 'sha', return_value='0' * 64), \
             patch.object(runner.signal, 'signal'), patch('builtins.print'), \
             patch.dict(runner.os.environ, {'CARGO_BUILD_TARGET': ''}), \
             patch.object(sys, 'argv', ['checks', '--report', str(report)]):
            with self.assertRaises(SystemExit) as outcome: runner.main()
        return json.loads(report.read_text()), outcome.exception.code, run.call_count

    def test_healthy_complete_run_preserves_observations_source_and_owned_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, code, calls = self.execute(Path(temporary).resolve())
            self.assertEqual((code, calls, result['outcome']), (0, 4, 'pass'))
            self.assertFalse(result['production_ready'])
            self.assertTrue(result['source_unchanged'] and result['owned_cleanup_confirmed'])
            self.assertEqual(len(result['observations']['cases']), 8)
            self.assertTrue(all('log_sha256' in c for c in result['commands']))

    def test_host_rejection_starts_no_compiler_or_native_measurement(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, code, calls = self.execute(Path(temporary).resolve(), unhealthy=True)
            self.assertEqual((code, calls, result['outcome']), (1, 0, 'fail'))
            self.assertIn('memory pressure', result['host_guard']['stop_reason'])

    def test_failed_commands_incomplete_samples_and_changed_source_never_pass(self):
        for setting in ('failed', 'incomplete', 'changed'):
            with self.subTest(setting=setting), tempfile.TemporaryDirectory() as temporary:
                result, code, _ = self.execute(Path(temporary).resolve(), **{setting: True})
                self.assertEqual((code, result['outcome']), (1, 'fail'))
                self.assertTrue(all('log_sha256' in c for c in result['commands']))


if __name__ == '__main__':
    unittest.main()

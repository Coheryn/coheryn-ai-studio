#!/usr/bin/env python3
"""Regression checks for qualification trust boundaries; no models or network required."""
import gzip
from concurrent.futures import ThreadPoolExecutor
import errno
import hashlib
import io
import json
from pathlib import Path
import subprocess
import signal
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

from cpu_qualification import references
from cpu_qualification import host
from cpu_qualification.assets import acquire, HTTPSRedirect
from cpu_qualification.common import ROOT, compare, pinned, read, sha
from cpu_qualification.transport import frames, image_content
from run_cpu_qualification import Context, checkpoint, main
from cpu_qualification.host import Guard, HostLimitError, Policy, bounded_text, mac_memory, rejection
from cpu_qualification.profiles import PROFILES, selected
from check_cpu_host import readiness


class QualificationTests(unittest.TestCase):
    @staticmethod
    def healthy_host():
        return {'system': 'Darwin', 'disk_free_bytes': 32 << 30, 'physical_memory_bytes': 8 << 30,
                'available_memory_bytes': 2 << 30, 'compressed_memory_bytes': 1 << 30,
                'memory_pressure_level': 1, 'thermal_state': 'nominal', 'load_1m': 1., 'cpu_count': 8}

    def test_host_limits_reject_pressure_thermal_compression_load_and_missing_measurements(self):
        good = self.healthy_host()
        self.assertIsNone(rejection(good, Policy()))
        for changes in ({'disk_free_bytes': 7 << 30}, {'available_memory_bytes': (1 << 30) - 1},
                        {'memory_pressure_level': 2}, {'memory_pressure_level': None},
                        {'compressed_memory_bytes': 3 << 30}, {'thermal_state': 'serious'},
                        {'thermal_state': 'critical'}, {'thermal_state': 'unavailable'},
                        {'load_1m': 17.}, {'load_1m': float('nan')}, {'load_1m': 'unknown'},
                        {'load_1m': None}, {'system': 'unknown'}, {'memory_pressure_level': True},
                        {'cpu_count': 0}, {'available_memory_bytes': 9 << 30}):
            with self.subTest(changes=changes):
                self.assertIsNotNone(rejection({**good, **changes}, Policy()))
        self.assertIsNone(rejection({**good, 'thermal_state': 'fair'}, Policy()))
        self.assertIsNotNone(rejection({**good, 'disk_free_bytes': 9 << 30}, Policy(), 2 << 30))
        with self.assertRaises(ValueError): rejection(good, Policy(), -1)

    def test_startup_load_wait_retains_rejections_without_weakening_admission(self):
        tick = [0.]
        def sleep(seconds): tick[0] += seconds
        good = self.healthy_host()
        sample = Mock(side_effect=[{**good, 'load_1m': 30.}, good])
        result = readiness(Policy(), sample, 10, now=lambda: tick[0], sleep=sleep)
        self.assertEqual(result['outcome'], 'pass')
        self.assertEqual(len(result['samples']), 2)
        self.assertEqual(result['samples'][0]['reason'], 'host load exceeds cutoff')
        self.assertEqual(tick[0], 5.)

    def test_startup_wait_is_bounded_and_never_starts_work_on_an_unhealthy_host(self):
        tick = [0.]
        def sleep(seconds): tick[0] += seconds
        sample = lambda: {**self.healthy_host(), 'load_1m': 30.}
        result = readiness(Policy(), sample, 10, now=lambda: tick[0], sleep=sleep)
        self.assertEqual(result['outcome'], 'fail')
        self.assertEqual(len(result['samples']), 3)
        self.assertEqual(tick[0], 10.)
        for invalid in (-1, 301, True):
            with self.assertRaises(ValueError): readiness(Policy(), sample, invalid)

    def test_startup_wait_does_not_retry_other_violations_or_invalid_load_values(self):
        for changes in ({'load_1m': 30., 'compressed_memory_bytes': 3 << 30},
                        {'load_1m': 30., 'available_memory_bytes': 0},
                        {'load_1m': 30., 'thermal_state': 'serious'},
                        {'load_1m': float('nan')}, {'load_1m': -1.}):
            sleep = Mock()
            result = readiness(Policy(), lambda: {**self.healthy_host(), **changes}, 30, sleep=sleep)
            self.assertEqual(result['outcome'], 'fail')
            json.dumps(result, allow_nan=False)
            sleep.assert_not_called()
            self.assertEqual(len(result['samples']), 1)

    def test_non_mac_sensors_are_explicitly_unavailable_without_weakening_memory_disk_limits(self):
        row = {**self.healthy_host(), 'system': 'Linux', 'thermal_state': 'unavailable',
               'memory_pressure_level': None, 'compressed_memory_bytes': None}
        self.assertIsNone(rejection(row, Policy()))
        self.assertIsNotNone(rejection({**row, 'available_memory_bytes': 0}, Policy()))

    def test_vm_memory_estimate_uses_observed_page_size_and_requires_complete_counters(self):
        text = ('Mach Virtual Memory Statistics: (page size of 16384 bytes)\n'
                'Pages free: 1000.\nPages inactive: 2000.\nPages speculative: 3000.\n'
                'Pages occupied by compressor: 4000.\n')
        actual = mac_memory(text, 8 << 30, 1)
        self.assertEqual(actual['available_memory_bytes'], 6000 * 16384)
        self.assertEqual(actual['compressed_memory_bytes'], 4000 * 16384)
        for invalid in ('', text.replace('16384', '0'), text.replace('4000.', '999999999999.')):
            with self.assertRaises((ValueError, KeyError)): mac_memory(invalid, 8 << 30, 1)

    def test_host_observer_input_limit_is_enforced_while_reading(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'counters'
            path.write_bytes(b'x' * ((64 << 10) + 1))
            with self.assertRaises(ValueError): bounded_text(path)
            path.write_bytes(b'counters')
            self.assertEqual(bounded_text(path), 'counters')

    def test_mac_sampler_routes_live_measurements_into_thermal_rejection_without_real_sensors(self):
        text = ('Mach Virtual Memory Statistics: (page size of 16384 bytes)\n'
                'Pages free: 65536.\nPages inactive: 65536.\nPages speculative: 0.\n'
                'Pages occupied by compressor: 65536.\n')
        def counters(command):
            return text if command[0].endswith('vm_stat') else '8589934592' if command[-1] == 'hw.memsize' else '1'
        with patch.object(host.platform, 'system', return_value='Darwin'), \
             patch.object(host, 'output', side_effect=counters), \
             patch.object(host, 'MacThermal') as thermal, \
             patch.object(host.shutil, 'disk_usage') as disk, \
             patch.object(host.os, 'getloadavg', return_value=(1., 1., 1.), create=True):
            disk.return_value.free = 32 << 30
            thermal.return_value.return_value = 'serious'
            row = host.Sampler(Path('.'))()
            self.assertEqual(row['available_memory_bytes'], 2 << 30)
            self.assertEqual(row['compressed_memory_bytes'], 1 << 30)
            self.assertIn('thermal', rejection(row, Policy()))
            thermal.return_value.assert_called_once_with()

    def test_host_policy_cannot_disable_cutoffs(self):
        for changes in ({'min_free_disk_bytes': 0}, {'min_available_memory_bytes': 0},
                        {'max_compressed_fraction': 1}, {'max_load_per_cpu': 99},
                        {'sample_seconds': 60}, {'max_samples': 999999}):
            with self.assertRaises(ValueError): Policy(**changes)

    def test_host_preflight_rejects_before_spawning_any_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            guard = Guard(directory, sampler=lambda: {**self.healthy_host(), 'thermal_state': 'serious'})
            try:
                context = Context(directory, guard)
                with patch('run_cpu_qualification.subprocess.Popen') as spawn:
                    with self.assertRaises(HostLimitError): context.command('blocked', ['unused'])
                    spawn.assert_not_called()
                self.assertEqual(context.commands, [])
            finally:
                guard.close()
            self.assertIn('thermal', guard.report()['stop_reason'])

    def test_host_observer_failure_and_sample_count_are_fail_closed_and_bounded(self):
        for sampler in (lambda: {'bad': float('nan')}, lambda: {**self.healthy_host(), 'load_1m': float('nan')}):
            with tempfile.TemporaryDirectory() as temporary:
                guard = Guard(Path(temporary), sampler=sampler)
                try:
                    with self.assertRaises(HostLimitError): guard.sample()
                finally:
                    guard.close()
                self.assertIsNotNone(guard.report()['stop_reason'])
                json.loads(guard.log.read_text())
        with tempfile.TemporaryDirectory() as temporary:
            guard = Guard(Path(temporary), sampler=self.healthy_host, policy=Policy(max_samples=1))
            try:
                guard.sample()
                for _ in range(3):
                    with self.assertRaises(HostLimitError): guard.sample()
            finally:
                guard.close()
            self.assertEqual(len(guard.log.read_text().splitlines()), 2)

    def test_host_abort_terminates_owned_child_and_preserves_failure_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            ready = directory / 'work-started'
            def sample():
                return {**self.healthy_host(), **({'thermal_state': 'serious'} if ready.exists() else {})}
            guard = Guard(directory, sampler=sample, policy=Policy(sample_seconds=.05))
            context = Context(directory, guard)
            try:
                guard.start()
                with self.assertRaises(HostLimitError):
                    context.command('owned', [sys.executable, '-c',
                                    'import pathlib,time;pathlib.Path("work-started").touch();time.sleep(30)'])
                self.assertNotEqual(context.commands[0]['exit'], 0)
                self.assertEqual(len(guard.children), 0)
            finally:
                guard.close()
            self.assertIsNotNone(guard.report()['stop_reason'])
            self.assertEqual(guard.report()['stop_errors'], [])

    @unittest.skipIf(sys.platform == 'win32', 'POSIX process-group signal contract')
    def test_concurrent_shutdown_signals_a_retired_owned_group_only_once(self):
        process = Mock(pid=12345)
        process.wait.return_value = -signal.SIGTERM
        with patch.object(host.os, 'killpg') as send, ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: host.stop_owned(process), range(2)))
        self.assertEqual(send.call_args_list, [unittest.mock.call(12345, signal.SIGTERM),
                                               unittest.mock.call(12345, signal.SIGKILL)])
        self.assertEqual(results, [['SIGTERM', 'SIGKILL']] * 2)
        process.wait.assert_any_call(timeout=5)

    @unittest.skipIf(sys.platform == 'win32', 'POSIX process-group signal contract')
    def test_denied_descendant_shutdown_remains_an_explicit_cleanup_failure(self):
        process = Mock(pid=12346)
        process.poll.return_value = -signal.SIGTERM
        with tempfile.TemporaryDirectory() as temporary:
            guard = Guard(Path(temporary), sampler=self.healthy_host)
            try:
                with patch.object(host.os, 'killpg', side_effect=[None, PermissionError(errno.EPERM, 'denied')]):
                    with self.assertRaisesRegex(PermissionError, 'SIGKILL.*descendant termination is unconfirmed'):
                        guard.stop(process)
                self.assertEqual(len(guard.report()['stop_errors']), 1)
                self.assertIn('root exit=-15', guard.report()['stop_errors'][0])
                self.assertNotIn(process, host._stopped)
            finally:
                guard.close()

    def test_windows_tree_termination_failure_is_not_hidden_by_root_termination(self):
        process = Mock(pid=12347)
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as temporary:
            system_path = str(Path(temporary).resolve())
            executable = str(Path(system_path) / 'System32/taskkill.exe')
            with patch.object(host.os, 'name', 'nt'), \
                 patch.dict(host.os.environ, {'SystemRoot': system_path}), \
                 patch.object(host, 'Path') as path, \
                 patch.object(host.subprocess, 'run', return_value=Mock(returncode=1)):
                path.return_value.__truediv__.return_value = executable
                with self.assertRaises(subprocess.CalledProcessError): host.stop_owned(process)
            process.kill.assert_called_once_with()
            process.wait.assert_called_once_with(timeout=5)
            self.assertNotIn(process, host._stopped)

    def test_atomic_progress_keeps_previous_complete_record_on_serialization_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'record.json'
            checkpoint(path, {'outcome': 'running', 'completed': 2})
            with self.assertRaises(ValueError): checkpoint(path, {'bad': float('nan')})
            self.assertEqual(read(path), {'outcome': 'running', 'completed': 2})
            self.assertEqual(list(path.parent.iterdir()), [path])
            checkpoint(path, {'outcome': 'pass', 'completed': 3})
            self.assertEqual(read(path)['outcome'], 'pass')

    def test_preflight_failure_is_finalized_as_failed_evidence_without_starting_model_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            report = root / 'result.json'
            arguments = ['qualification', '--suite', 'gguf-text', '--archive', 'unused',
                         '--archive-sha256', 'unused', '--native-checks', 'unused', '--report', str(report)]
            make_guard = lambda directory: Guard(directory, sampler=lambda: {
                **self.healthy_host(), 'thermal_state': 'serious'})
            with patch('run_cpu_qualification.ROOT', root), \
                 patch('run_cpu_qualification.source_identity', return_value={'unit.rs': 'frozen'}), \
                 patch('run_cpu_qualification.hardware', return_value={}), \
                 patch('run_cpu_qualification.native_target', return_value='unit-target'), \
                 patch('run_cpu_qualification.host.Guard', side_effect=make_guard), \
                 patch('run_cpu_qualification.subprocess.Popen') as spawn, \
                 patch('run_cpu_qualification.traceback.print_exc'), patch.object(sys, 'argv', arguments):
                with self.assertRaises(SystemExit) as error: main()
                self.assertEqual(error.exception.code, 1)
                spawn.assert_not_called()
            result = read(report)
            self.assertEqual(result['outcome'], 'fail')
            self.assertEqual(result['results'], {})
            self.assertEqual(result['commands'], [])
            self.assertTrue(result['source_unchanged'])
            self.assertIn('thermal', result['host_guard']['stop_reason'])
            self.assertEqual(result['host_guard']['log_sha256'], sha(root / result['artifacts'] / 'host-health.jsonl'))

    def test_modern_groups_cover_all_mandatory_encodings_pairs_and_real_shards(self):
        text = [PROFILES[name] for name in selected('gguf-quantized')]
        self.assertEqual({p['label'] for p in text}, {'f16', 'q8_0', 'q4_0', 'q4_k_m', 'q6_k_m'})
        self.assertEqual({PROFILES[name]['label'] for name in selected('gguf-vision')}, {'f16', 'q4_k_m'})
        self.assertEqual(sum(PROFILES[name]['layout'] == 'sharded' for name in selected('safetensors')), 2)
        self.assertTrue(all(PROFILES[name]['family'] != 'qwen2' for group in ('safetensors', 'gguf-quantized', 'gguf-vision') for name in selected(group)))
    def test_numeric_tail_and_shapes_are_checked(self):
        expected = [0.0] * 1024
        actual = expected.copy()
        actual[-1] = .1  # Unsampled tail must fail a full-array comparison.
        with self.assertRaises(ValueError):
            compare(actual, expected, .003, .001, .0005)
        with self.assertRaises(ValueError):
            compare([1.0], [1.0, 2.0], .003, .001)

    def test_finiteness_and_relative_l2(self):
        for value in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                compare([value], [0.], 1., 0.)
        with self.assertRaises(ValueError):
            compare([1.01], [1.], .1, 0., .001)
        self.assertEqual(compare([1.], [1.], 0., 0., 0.)['normalized_l2'], 0.)

    def test_optimized_python_cannot_disable_checks(self):
        process = subprocess.run([sys.executable, '-O', '-c',
                                  'from cpu_qualification.common import compare; compare([9.],[0.],.001,0.)'],
                                 cwd=Path(__file__).parent, capture_output=True)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn(b'numeric mismatch', process.stderr)

    def test_existing_corrupt_cache_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / 'weight'
            path.write_bytes(b'bad')
            with self.assertRaises(ValueError):
                acquire(path, {'bytes': 3, 'sha256': hashlib.sha256(b'yes').hexdigest()}, 'https://example.test/model', True)
            self.assertEqual(path.read_bytes(), b'bad')

    def test_download_reserves_host_headroom_before_opening_a_connection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            guard = Mock()
            guard.sample.side_effect = HostLimitError('disk reserve')
            with patch('cpu_qualification.assets.urllib.request.build_opener') as opener:
                with self.assertRaises(HostLimitError):
                    acquire(root / 'weight', {'bytes': 3, 'sha256': hashlib.sha256(b'yes').hexdigest()},
                            'https://example.test/weight', True, guard)
                opener.assert_not_called()
            guard.sample.assert_called_once_with(disk_reservation=3)
            self.assertEqual(list(root.iterdir()), [])

    def test_host_abort_during_download_removes_partial_file_without_publishing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            guard = Mock()
            guard.check.side_effect = [None, HostLimitError('memory pressure')]
            response = io.BytesIO(b'yes')
            response.url = 'https://example.test/weight'
            with patch('cpu_qualification.assets.urllib.request.build_opener') as opener:
                opener.return_value.open.return_value = response
                with self.assertRaises(HostLimitError):
                    acquire(root / 'weight', {'bytes': 3, 'sha256': hashlib.sha256(b'yes').hexdigest()},
                            response.url, True, guard)
            self.assertFalse((root / 'weight').exists())
            self.assertEqual(list(root.glob('.download-*')), [])

    def test_linked_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / 'data').write_bytes(b'x')
            try:
                (root / 'link').symlink_to(root / 'data')
            except OSError:
                self.skipTest('symlink creation not permitted on this host')
            with self.assertRaises(ValueError):
                pinned(root / 'link', {'bytes': 1, 'sha256': sha(root / 'data')})

    def test_insecure_redirect_is_rejected(self):
        with self.assertRaises(ValueError):
            HTTPSRedirect().redirect_request(None, None, 302, '', {}, 'http://example.test/file')

    def bundle(self, entries, expected=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            archive = root / 'bundle.tgz'
            with tarfile.open(archive, 'w:gz') as stream:
                for name, data, kind in entries:
                    item = tarfile.TarInfo(name)
                    item.size, item.type = len(data), kind
                    stream.addfile(item, io.BytesIO(data))
            (root / 'out').mkdir()
            expected = expected or {'references/a.f32': {'bytes': 1, 'sha256': hashlib.sha256(b'a').hexdigest()}}
            with patch.object(references, 'files', return_value=expected), patch.object(references, 'INVENTORY', archive):
                references.unpack(archive, sha(archive), root / 'out')
            return sorted(p.relative_to(root / 'out').as_posix() for p in (root / 'out').rglob('*') if p.is_file())

    def test_exact_reference_archive(self):
        self.assertEqual(self.bundle([('references/a.f32', b'a', tarfile.REGTYPE)]), ['references/a.f32'])

    def test_archive_rejects_unexpected_missing_duplicate_and_links(self):
        valid = ('references/a.f32', b'a', tarfile.REGTYPE)
        for entries in ([('../escape', b'a', tarfile.REGTYPE)], [valid, valid], [],
                        [('references/a.f32', b'a', tarfile.SYMTYPE)],
                        [('references/a.f32', b'changed', tarfile.REGTYPE)],
                        [valid, ('extra.py', b'a', tarfile.REGTYPE)]):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                self.bundle(entries)

    def test_gzip_expansion_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            archive = root / 'bomb.gz'
            archive.write_bytes(gzip.compress(b'\0' * 8192))
            (root / 'out').mkdir()
            with patch.object(references, 'files', return_value={}), patch.object(references, 'MAX_EXPANDED', 1024):
                with self.assertRaises(ValueError):
                    references.unpack(archive, sha(archive), root / 'out')

    def test_portable_archive_names(self):
        for name in ('/root', '../x', 'x/../y', 'x\\y', 'C:/x', 'x//y', 'x/CON.txt', 'x/..', 'x/a.'):
            self.assertFalse(references.safe_name(name), name)
        self.assertTrue(references.safe_name('references/english-layer.0.f32'))

    def test_sse_terminal_and_error_frames(self):
        valid = b'data: {"choices":[]}\n\ndata: [DONE]\n\n'
        self.assertEqual(frames(valid), [{'choices': []}])
        for raw in (valid + b'data: {}\n', valid + b'data: [DONE]\n', b'data: {}\n', b'data: {"error":{}}\ndata: [DONE]\n'):
            with self.assertRaises(ValueError):
                frames(raw)

    def test_failed_child_preserves_exit_and_log_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            context = Context(Path(directory).resolve())
            with self.assertRaises(ValueError):
                context.command('failure', [sys.executable, '-c', 'import sys; print("failure evidence"); sys.exit(7)'])
            record = context.commands[0]
            self.assertEqual(record['exit'], 7)
            self.assertEqual(record['stdout_sha256'], sha(context.directory / record['stdout']))
            self.assertIn('failure evidence', (context.directory / record['stdout']).read_text())

    def test_api_image_order_matches_frozen_prompt(self):
        case = read(ROOT / 'fixtures/qwen35/vision/reference-v1.json')['cases'][0]
        self.assertLess(case['prompt'].index('<|image_pad|>'), case['prompt'].index(case['text']))
        content = image_content(case['text'], ROOT / 'fixtures/qwen35/vision/red.png')
        self.assertEqual([part['type'] for part in content], ['image_url', 'text'])
        self.assertEqual(content[1]['text'], case['text'])


if __name__ == '__main__':
    unittest.main()

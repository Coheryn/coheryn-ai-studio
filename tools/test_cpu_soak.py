"""Fail-closed measurement/coverage regressions for the CPU soak controller."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from run_cpu_soak import ROOT, SCENARIOS, summarize, validate_profile


class SoakEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads((ROOT / 'fixtures/soak/cpu-smoke-v1.json').read_text())

    def rows(self):
        p = self.profile
        memory = {'reserved': 17 << 20, 'peak_reserved': 64 << 20,
                  'session_buffer_capacity_bytes': 0, 'peak_session_buffer_capacity_bytes': 1 << 20}
        context = {'concurrency': 1, 'execution_profile': 'balanced'}
        rows = [{'kind': 'contract', 'profile': p},
                {'kind': 'bootstrap_complete', 'reserved_after_close': 0},
                {'kind': 'history_baseline', 'reserved_bytes': 16 << 20},
                {'kind': 'resident_baseline', **context, 'reserved_bytes': 17 << 20},
                {'kind': 'warmup_complete', 'concurrency': 1, 'profile': 'balanced', 'rss_baseline_bytes': 32 << 20}]
        for task, count in [('text', 7), ('embedding', 7), ('image', 4)]:
            rows.extend({'kind': 'operation', **context, 'task': task, 'path': '/v1/embeddings' if task == 'embedding' else '/api/v1/generate', 'elapsed_ms': 1., 'outcome': 'pass'} for _ in range(count))
        for phase, cycles in [('warmup', [0]), ('measured', [0, 1])]:
            for cycle in cycles:
                for scenario in sorted(SCENARIOS):
                    row = {'kind': 'fault', **context, 'phase': phase, 'cycle': cycle, 'scenario': scenario}
                    if scenario == 'slow_consumer_timeout':
                        row.update(buffered_token_events=16, terminal_error_observed=True)
                    if scenario == 'image_backpressure_peer_progress_and_drop':
                        row.update(vision_backpressure_observed=True, peer_completion_ms=1.)
                    rows.append(row)
        for cycle in (0, 1):
            rows.extend([{'kind': 'sample', 'rss_bytes': 33 << 20, 'memory': copy.deepcopy(memory)},
                         {'kind': 'cycle_complete', 'concurrency': 1, 'profile': 'balanced',
                          'cycle': cycle, 'rss_bytes': 33 << 20, 'memory': copy.deepcopy(memory)},
                         {'kind': 'epoch_complete', 'scheduler': {'active': 0, 'queued': 0}},
                         {'kind': 'service_closed', 'memory': {'reserved': 0, 'session_buffer_capacity_bytes': 0}}])
        return rows

    def evaluate(self, rows, suffix=b''):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'observer.jsonl'
            path.write_bytes(b''.join(json.dumps(row).encode() + b'\n' for row in rows) + suffix)
            return summarize(path, self.profile)

    def test_complete_stream_has_bounded_summary_and_distinct_memory_metrics(self):
        result = self.evaluate(self.rows())
        self.assertEqual(result['measured_cycles'], 2)
        self.assertEqual(result['service_epochs'], 2)
        self.assertEqual(result['peak_reserved_bytes'], 64 << 20)
        self.assertEqual(result['sampled_peak_rss_bytes'], 33 << 20)
        self.assertEqual(result['max_drained_rss_growth_bytes'], 1 << 20)
        self.assertTrue(all(n == 3 for n in result['faults'].values()))

    def test_missing_or_duplicated_faults_cannot_count_as_coverage(self):
        for mutation in ('missing', 'duplicate', 'wrong-phase'):
            with self.subTest(mutation=mutation):
                rows = self.rows()
                index = next(i for i, row in enumerate(rows) if row['kind'] == 'fault')
                if mutation == 'missing':
                    rows.pop(index)
                elif mutation == 'duplicate':
                    rows.insert(index, copy.deepcopy(rows[index]))
                else:
                    rows[index]['phase'] = 'measured'
                with self.assertRaises(ValueError):
                    self.evaluate(rows)

    def test_dropped_or_duplicated_cycles_and_missing_teardown_fail(self):
        for kind in ('cycle_complete', 'service_closed'):
            for duplicate in (False, True):
                with self.subTest(kind=kind, duplicate=duplicate):
                    rows = self.rows()
                    index = next(i for i, row in enumerate(rows) if row['kind'] == kind)
                    if duplicate:
                        rows.insert(index, copy.deepcopy(rows[index]))
                    else:
                        rows.pop(index)
                    with self.assertRaises(ValueError):
                        self.evaluate(rows)

    def test_reservation_leaks_rss_growth_and_missing_metrics_fail(self):
        for change in ('reservation', 'buffer', 'growth', 'peak', 'missing', 'nan', 'teardown'):
            with self.subTest(change=change):
                rows = self.rows()
                row = next(r for r in rows if r['kind'] == 'cycle_complete')
                if change == 'reservation':
                    row['memory']['reserved'] += 1
                elif change == 'buffer':
                    row['memory']['session_buffer_capacity_bytes'] = 1
                elif change == 'growth':
                    row['rss_bytes'] = (32 << 20) + self.profile['max_drained_rss_growth_bytes'] + 1
                elif change == 'peak':
                    next(r for r in rows if r['kind'] == 'sample')['memory']['peak_reserved'] = self.profile['max_memory_bytes'] + 1
                elif change == 'missing':
                    row['rss_bytes'] = None
                elif change == 'nan':
                    row['rss_bytes'] = float('nan')
                else:
                    next(r for r in rows if r['kind'] == 'service_closed')['memory']['reserved'] = 1
                with self.assertRaises(ValueError):
                    self.evaluate(rows)

    def test_late_peer_progress_unbounded_events_and_bad_latency_fail(self):
        for change in ('peer', 'events', 'latency', 'notfinite'):
            with self.subTest(change=change):
                rows = self.rows()
                if change == 'peer':
                    next(r for r in rows if r.get('scenario') == 'image_backpressure_peer_progress_and_drop')['peer_completion_ms'] = self.profile['consumer_timeout_seconds'] * 1000 + 1
                elif change == 'events':
                    next(r for r in rows if r.get('scenario') == 'slow_consumer_timeout')['buffered_token_events'] = 17
                else:
                    next(r for r in rows if r['kind'] == 'operation')['elapsed_ms'] = float('nan') if change == 'notfinite' else self.profile['request_timeout_seconds'] * 1000 + 1
                with self.assertRaises(ValueError):
                    self.evaluate(rows)

    def test_truncation_changed_contract_and_explicit_observer_failure_fail(self):
        with self.assertRaises(ValueError):
            self.evaluate(self.rows(), b'{"kind":')
        rows = self.rows()
        rows[0]['profile'] = {**self.profile, 'cycles': 1}
        with self.assertRaises(ValueError):
            self.evaluate(rows)
        with self.assertRaises(ValueError):
            self.evaluate(self.rows() + [{'kind': 'observer_failure', 'reason': 'injected'}])

    def test_invalid_or_ambiguous_contracts_fail_before_execution(self):
        for change in ({'cycles': 0}, {'concurrency': [1, 1]}, {'concurrency': [True]},
                       {'require_rss': 'yes'}, {'max_rss_bytes': -1}, {'unknown': 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_profile({**self.profile, **change})


if __name__ == '__main__':
    unittest.main()

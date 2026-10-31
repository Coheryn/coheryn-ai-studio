#!/usr/bin/env python3
"""Check host headroom without loading a model, compiling or starting a server."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

from cpu_qualification.common import ROOT, write
from cpu_qualification.host import Policy, Sampler, rejection


def readiness(policy, sampler, wait_seconds=0, *, now=time.monotonic, sleep=time.sleep):
    """Wait only for startup load; admit no work and retain every observation."""
    if type(wait_seconds) is not int or not 0 <= wait_seconds <= 300:
        raise ValueError('startup load wait must be between 0 and 300 seconds')
    started, samples = now(), []
    while True:
        try:
            row = sampler()
            json.dumps(row, allow_nan=False)
            reason = rejection(row, policy)
        except Exception as error:
            row = {'observation_error': str(error)[:512]}
            reason = 'required host observer failed: ' + str(error)[:512]
        elapsed = now() - started
        samples.append({'elapsed_seconds': elapsed, 'observation': row, 'reason': reason})
        load = row.get('load_1m')
        waitable = (reason == 'host load exceeds cutoff' and type(load) in (int, float)
                    and math.isfinite(load) and load > row['cpu_count'] * policy.max_load_per_cpu
                    and rejection({**row, 'load_1m': 0.}, policy) is None)
        if not waitable or elapsed >= wait_seconds:
            return {'observation': row, 'reason': reason, 'samples': samples,
                    'outcome': 'pass' if reason is None else 'fail'}
        sleep(min(5, wait_seconds - elapsed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--wait-for-load-seconds', type=int, default=0,
                        help='bounded idle startup wait for load only; all cutoffs remain enforced')
    args = parser.parse_args()
    policy = Policy()
    report = {'schema_version': 1, 'kind': 'cpu-host-preflight', 'policy': asdict(policy),
              'production_ready': False, 'outcome': 'fail',
              'startup_load_wait_seconds': args.wait_for_load_seconds}
    try:
        report.update(readiness(policy, Sampler(ROOT), args.wait_for_load_seconds))
    except Exception as error:
        report['reason'] = str(error)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write(args.report, report)
    print(json.dumps(report, indent=2))
    raise SystemExit(report['outcome'] != 'pass')


if __name__ == '__main__':
    main()

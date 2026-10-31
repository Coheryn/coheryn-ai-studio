#!/usr/bin/env python3
"""Check host headroom without loading a model, compiling or starting a server."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from cpu_qualification.common import ROOT, write
from cpu_qualification.host import Policy, Sampler, rejection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    policy = Policy()
    report = {'schema_version': 1, 'kind': 'cpu-host-preflight', 'policy': asdict(policy),
              'production_ready': False, 'outcome': 'fail'}
    try:
        report['observation'] = Sampler(ROOT)()
        report['reason'] = rejection(report['observation'], policy)
        report['outcome'] = 'pass' if report['reason'] is None else 'fail'
    except Exception as error:
        report['reason'] = str(error)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write(args.report, report)
    print(json.dumps(report, indent=2))
    raise SystemExit(report['outcome'] != 'pass')


if __name__ == '__main__':
    main()

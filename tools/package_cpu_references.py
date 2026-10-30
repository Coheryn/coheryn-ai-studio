#!/usr/bin/env python3
"""Package existing independently captured data and its complete legal payload; no inference."""
import argparse
from pathlib import Path
from cpu_qualification.common import write
from cpu_qualification.references import pack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    write(args.report, pack(args.output))
    print('PASS frozen data-only reference bundle')


if __name__ == '__main__':
    main()

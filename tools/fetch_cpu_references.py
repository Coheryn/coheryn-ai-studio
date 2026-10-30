#!/usr/bin/env python3
"""Acquire an explicitly hosted copy of the exact frozen reference-data bundle."""
import argparse
from pathlib import Path
from cpu_qualification.assets import acquire
from cpu_qualification.common import ROOT, read, require, sha
from cpu_qualification.references import INVENTORY, MAX_ARCHIVE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    item = read(ROOT / 'fixtures/cpu-qualification/reference-bundle-v1.json')
    require(item['inventory_sha256'] == sha(INVENTORY) and item['bytes'] <= MAX_ARCHIVE, 'reference descriptor changed')
    acquire(args.output.resolve(), item, args.url, True)
    print('PASS pinned reference-data acquisition')


if __name__ == '__main__':
    main()

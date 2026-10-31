#!/usr/bin/env python3
"""Create a lossless three-file real-model Safetensors fixture from the pinned publisher source."""
import argparse
from pathlib import Path
from cpu_qualification.common import ROOT, read, require, write
from cpu_qualification.shards import derive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / '.cache/qwen35-three-shard')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    require(not args.report.exists(), 'report already exists')
    source = read(ROOT / 'fixtures/qwen35/model.json')
    weight = next(item for item in source['files'] if item['file'].endswith('.safetensors'))
    assets = [item for item in source['files'] if item['file'] not in (weight['file'], 'model.safetensors.index.json')]
    result = derive(ROOT / source['cache_directory'] / weight['file'], args.output.resolve(), weight, assets)
    result.update(source_repository=source['repository'], source_revision=source['revision'])
    write(args.report, result)
    print('PASS lossless physical shards; named tensor bytes unchanged; native inference qualification remains separate')


if __name__ == '__main__':
    main()

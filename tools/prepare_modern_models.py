#!/usr/bin/env python3
"""Acquire/verify explicitly pinned modern model sets and losslessly derived physical shards."""
import argparse
from pathlib import Path
from cpu_qualification import assets
from cpu_qualification.common import read, require, write, ROOT
from cpu_qualification.profiles import selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=('safetensors', 'gguf-quantized', 'gguf-vision'), required=True)
    parser.add_argument('--fetch', action='store_true')
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    require(not args.report.exists(), 'preparation report already exists')
    models, identities = assets.models(selected(args.group), args.fetch)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write(args.report, {'schema_version': 1, 'kind': 'pinned-modern-input-preparation', 'group': args.group,
                        'execution_qualification': False, 'models': {k: str(v) for k, v in models.items()},
                        'inputs': identities, 'gguf_supplier': read(ROOT / 'fixtures/qwen35/gguf/models-v1.json')['supplier']})
    print('PASS pinned input preparation; execution qualification remains separate')


if __name__ == '__main__':
    main()

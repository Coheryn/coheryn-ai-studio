#!/usr/bin/env python3
"""Acquire complete pinned VLM components sequentially; never execute model weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import ssl
import tempfile
import time
import urllib.request
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache/vlm/full'


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='append', choices=['qwen2-vl-2b', 'llava-1.5-7b'])
    parser.add_argument('--minimum-free-bytes', type=int, default=4 << 30)
    parser.add_argument('--timeout-seconds', type=int, default=3600)
    parser.add_argument('--report', type=Path, default=ROOT / 'fixtures/qualification/m0-components.json')
    args = parser.parse_args()
    assert args.minimum_free_bytes >= 1 << 30 and 1 <= args.timeout_seconds <= 14400
    source = ROOT / 'fixtures/vlm/models.json'
    manifest = json.loads(source.read_text())
    selected = args.model or ['qwen2-vl-2b', 'llava-1.5-7b']
    context = ssl.create_default_context(cafile='/etc/ssl/cert.pem') if Path('/etc/ssl/cert.pem').exists() else ssl.create_default_context()
    records = []
    CACHE.mkdir(parents=True, exist_ok=True)
    for identity in selected:
        model = next(m for m in manifest['models'] if m['id'] == identity)
        assert re.fullmatch(r'[0-9a-f]{40}', model['revision'])
        directory = CACHE / identity
        directory.mkdir(exist_ok=True)
        assert directory.resolve().is_relative_to(CACHE.resolve()) and not directory.is_symlink()
        for component in model['components']:
            filename, size, checksum = component['file'], component['bytes'], component['sha256']
            assert re.fullmatch(r'[A-Za-z0-9._-]{1,256}', filename) and re.fullmatch(r'[0-9a-f]{64}', checksum)
            target = directory / filename
            assert not target.is_symlink()
            if target.exists():
                assert target.stat().st_size == size and digest(target) == checksum, 'existing component identity mismatch'
            else:
                assert shutil.disk_usage(directory).free >= size + args.minimum_free_bytes, 'insufficient disk headroom for this component'
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(dir=directory, suffix='.part', delete=False) as output:
                        temporary = Path(output.name)
                        url = f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/{filename}"
                        deadline = time.monotonic() + args.timeout_seconds
                        h, received = hashlib.sha256(), 0
                        print(f'Acquire {identity}/{filename}: exact ceiling {size} bytes', flush=True)
                        with urllib.request.urlopen(url, timeout=60, context=context) as response:
                            declared = response.headers.get('Content-Length')
                            assert declared is None or int(declared) == size, 'response byte count differs from pin'
                            while chunk := response.read1(1 << 20):
                                assert time.monotonic() < deadline, 'acquisition deadline exceeded'
                                received += len(chunk)
                                assert received <= size, 'component exceeds pin'
                                output.write(chunk)
                                h.update(chunk)
                        assert received == size and h.hexdigest() == checksum, 'component size/checksum mismatch'
                        output.flush()
                        os.fsync(output.fileno())
                    temporary.replace(target)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
            records.append({'model': identity, 'role': component['role'], 'file': filename, 'bytes': size,
                            'sha256': checksum, 'repository': model['repository'], 'revision': model['revision'],
                            'status': 'full-file-byte-verified', 'model_execution_qualified': False})
            print(f'PASS complete bytes {identity}/{filename}', flush=True)
    report = {'schema_version': 1, 'stage': 'M0 complete component identity, not inference qualification',
              'selection_manifest_sha256': digest(source), 'components': records,
              'minimum_free_bytes': args.minimum_free_bytes, 'acquisition': 'Sequential atomic writes, exact byte/checksum ceilings and bounded stream buffers; no model execution.',
              'remaining': ['component/processor semantic alignment', 'independent vision/projector/logit/greedy captures', 'native vision-language execution and release license qualification']}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

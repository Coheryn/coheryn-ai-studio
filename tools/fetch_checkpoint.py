#!/usr/bin/env python3
"""Acquire the pinned Qwen3.5 development checkpoint as one verified asset set."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import ssl
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'fixtures/qwen35/model.json'

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def verify(directory, files):
    for item in files:
        path = directory / item['file']
        if path.is_symlink() or not path.is_file() or path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('checkpoint identity mismatch: ' + item['file'])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    model = json.loads(MANIFEST.read_text())
    files = model['files']
    assert re.fullmatch('[0-9a-f]{40}', model['revision'])
    assert model['repository'] == 'Qwen/Qwen3.5-0.8B'
    assert 1 <= len(files) <= 32
    assert len({f['file'].lower() for f in files}) == len(files)
    for item in files:
        assert re.fullmatch('[A-Za-z0-9_.-]{1,200}', item['file']) and item['file'] not in ('.','..')
        assert 0 < item['bytes'] <= 4 << 30 and re.fullmatch('[0-9a-f]{64}', item['sha256'])
    target = ROOT / '.cache/qwen3.5-0.8b-source'
    if target.exists():
        verify(target, files)
        print('Verified', target)
        return
    if args.verify_only:
        raise SystemExit('checkpoint is not cached')
    target.parent.mkdir(exist_ok=True)
    total = sum(f['bytes'] for f in files)
    if shutil.disk_usage(target.parent).free < total + (512 << 20):
        raise SystemExit('insufficient space for checkpoint staging')
    staging = Path(tempfile.mkdtemp(prefix='.qwen35-stage-', dir=target.parent))
    try:
        context = ssl.create_default_context(cafile='/etc/ssl/cert.pem' if Path('/etc/ssl/cert.pem').exists() else None)
        for item in files:
            url = f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/{item['file']}"
            request = urllib.request.Request(url, headers={'User-Agent':'Coheryn-development-checkpoint'})
            size = 0
            digest = hashlib.sha256()
            with urllib.request.urlopen(request,context=context,timeout=60) as response, (staging/item['file']).open('xb') as output:
                if not response.url.startswith('https://'):
                    raise ValueError('insecure checkpoint redirect')
                while chunk := response.read(1 << 20):
                    size += len(chunk)
                    if size > item['bytes']:
                        raise ValueError('download exceeds pinned byte length')
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if size != item['bytes'] or digest.hexdigest() != item['sha256']:
                raise ValueError('download identity mismatch: ' + item['file'])
        verify(staging, files)
        if target.exists():
            raise ValueError('checkpoint destination appeared during download')
        os.rename(staging, target)
        print('Verified', target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

if __name__ == '__main__':
    main()

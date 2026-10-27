#!/usr/bin/env python3
"""Pinned, bounded, atomic development fixture acquisition. Never executes a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import ssl
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id', default='qwen2.5-0.5b-instruct-fp16')
    parser.add_argument('--manifest', type=Path, default=ROOT/'fixtures/models.json')
    args = parser.parse_args()
    models = json.loads(args.manifest.read_text())['models']
    model = next(m for m in models if m['id'] == args.id)
    directory = ROOT/'.cache/models'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory/model['file']
    if target.exists():
        if digest(target) != model['sha256']:
            raise SystemExit('Existing model has the wrong checksum; remove it before retrying.')
        print(target)
        return
    if shutil.disk_usage(directory).free < model['maximum_download_bytes'] + (512 << 20):
        raise SystemExit('Insufficient free space for bounded model download.')
    url = f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/{model['file']}"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, suffix='.part', delete=False) as output:
            temporary = Path(output.name)
            h, size = hashlib.sha256(), 0
            # Python.org macOS builds may not have populated their own CA bundle.
            context = ssl.create_default_context(cafile='/etc/ssl/cert.pem') if Path('/etc/ssl/cert.pem').exists() else ssl.create_default_context()
            with urllib.request.urlopen(url, timeout=60, context=context) as response:
                while chunk := response.read(1 << 20):
                    size += len(chunk)
                    if size > model['maximum_download_bytes']:
                        raise ValueError('Model exceeds manifest download ceiling')
                    output.write(chunk)
                    h.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        if size != model['bytes']:
            raise ValueError('Model byte count mismatch')
        if h.hexdigest() != model['sha256']:
            raise ValueError('Model SHA-256 mismatch')
        temporary.replace(target)
        print(json.dumps({'path': str(target), 'bytes': size, 'sha256': h.hexdigest()}))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

if __name__ == '__main__':
    main()

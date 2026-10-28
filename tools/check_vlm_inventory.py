#!/usr/bin/env python3
"""Check cached complete VLM bytes and native GGUF inventories; no inference."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    binary = ROOT / 'target/release' / ('coheryn.exe' if os.name == 'nt' else 'coheryn')
    manifest_path = ROOT / 'fixtures/vlm/models.json'
    models = json.loads(manifest_path.read_text())
    rows = []
    for model in models['models']:
        for component in model['components']:
            path = ROOT / '.cache/vlm/full' / model['id'] / component['file']
            assert path.stat().st_size == component['bytes'] and sha(path) == component['sha256']
            descriptor = json.loads(subprocess.check_output([str(binary), 'models', 'inspect', str(path), '--json'], text=True, timeout=60))
            prefix = json.loads((ROOT / component['inventory']).read_text())['descriptor']
            for field in prefix:
                assert descriptor[field] == prefix[field], (model['id'], component['role'], field)
            rows.append({'model':model['id'],'role':component['role'],'sha256':component['sha256'],
                'file_bytes':descriptor['file_bytes'],'architecture':descriptor['architecture'],
                'tensor_count':len(descriptor['tensors']),'encodings':descriptor['encodings'],
                'header_sha256':descriptor['header_sha256'],'matches_frozen_prefix_inventory':True,
                'native_inference_qualified':False})
            print(f"PASS native full-file container {model['id']}/{component['role']}",flush=True)
    output = ROOT / 'fixtures/qualification/m0-container-inventories.json'
    output.write_text(json.dumps({'schema_version':1,'scope':'complete byte verification and native GGUF descriptor/range validation, not tensor numerical execution',
        'binary_sha256':sha(binary),'selection_manifest_sha256':sha(manifest_path),'components':rows},indent=2)+'\n')


if __name__ == '__main__':
    main()

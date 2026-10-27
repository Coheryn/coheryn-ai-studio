#!/usr/bin/env python3
"""Pin/reproduce M0 VLM provenance and bounded header inventories; never execute weights.

--pin explicitly selects published revisions and writes pins. Normal runs use those exact
pins, verify small source assets, and inspect GGUF prefixes without downloading full weights.
Published whole-file LFS hashes are recorded, not falsely described as byte-verified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import ssl
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'fixtures/vlm'
CACHE = ROOT / '.cache/vlm'
MANIFEST = DIRECTORY / 'models.json'
TLS = ssl.create_default_context(cafile='/etc/ssl/cert.pem') if Path('/etc/ssl/cert.pem').exists() else ssl.create_default_context()
TARGETS = [
    ('llava-1.5-7b', 'second-state/Llava-v1.5-7B-GGUF',
     ['llava-v1.5-7b-Q4_K_M.gguf', 'llava-v1.5-7b-mmproj-model-f16.gguf'], 'llava-hf/llava-1.5-7b-hf'),
    ('qwen2-vl-2b', 'ggml-org/Qwen2-VL-2B-Instruct-GGUF',
     ['Qwen2-VL-2B-Instruct-f16.gguf', 'mmproj-Qwen2-VL-2B-Instruct-f16.gguf'], 'Qwen/Qwen2-VL-2B-Instruct'),
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_url(url, limit, headers=None):
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=45, context=TLS) as response:
        data = response.read(limit + 1)
        if len(data) > limit:
            raise ValueError('response exceeded bounded acquisition ceiling')
        return data, response.status, dict(response.headers)


def info(repo):
    data, _, _ = read_url(f'https://huggingface.co/api/models/{repo}?blobs=true', 4 << 20)
    result = json.loads(data)
    if not re.fullmatch('[0-9a-f]{40}', result['sha']):
        raise ValueError('unqualified repository revision')
    return result


def pin():
    entries = []
    for identity, repository, filenames, source_repo in TARGETS:
        model, source = info(repository), info(source_repo)
        files = {f['rfilename']: f for f in model['siblings']}
        missing = set(filenames) - files.keys()
        if missing:
            raise ValueError(f'{repository}: missing exact required files {missing}; inspect publisher inventory before changing selection')
        components = []
        for filename, role in zip(filenames, ['language', 'vision-projector']):
            f = files[filename]
            components.append({'role': role, 'file': filename, 'bytes': f['size'],
                               'sha256': f['lfs']['sha256'], 'hash_status': 'publisher-declared-not-full-file-verified'})
        available = {f['rfilename'] for f in source['siblings']}
        assets = [f for f in ['config.json', 'preprocessor_config.json', 'processor_config.json',
                  'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.json',
                  'tokenizer.model', 'tokenizer.json', 'merges.txt', 'vocab.json', 'LICENSE', 'LICENSE.txt'] if f in available]
        if not {'config.json', 'preprocessor_config.json', 'tokenizer_config.json'} <= set(assets):
            raise ValueError('source processor/config/tokenizer identity incomplete')
        entries.append({'id': identity, 'repository': repository, 'revision': model['sha'],
                        'declared_license': model.get('cardData', {}).get('license'),
                        'components': components,
                        'source_repository': source_repo, 'source_revision': source['sha'],
                        'source_declared_license': source.get('cardData', {}).get('license'),
                        'source_asset_names': assets, 'qualification': 'not-executable-not-qualified'})
    document = {'schema_version': 1, 'stage': 'M0-provenance', 'models': entries,
                'runtime_dependency': False, 'full_weight_downloaded': False,
                'oracle': {'source_record': 'resources/reference-archive/fixtures/vlm/models.json',
                           'source_record_sha256': hashlib.sha256((ROOT/'resources/reference-archive/fixtures/vlm/models.json').read_bytes()).hexdigest(),
                           'revision': json.loads((ROOT/'fixtures/models.json').read_text())['oracle']['revision'],
                           'production_dependency': False, 'vision_reference_outputs_captured': False},
                'remaining_acceptance': ['review component/processor alignment and licenses',
                    'byte-verify complete GGUF assets before executing any reference',
                    'capture independent preprocessing/feature/embedding/logit/greedy fixtures',
                    'qualify multi-image behavior and freeze exact per-variant policies']}
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(document, indent=2) + '\n')


def reproduce():
    document = json.loads(MANIFEST.read_text())
    subprocess.run([str(ROOT/'tools/cargo'), 'build', '--release', '--locked', '-p', 'coheryn-engine', '--example', 'inspect_header'], cwd=ROOT, check=True)
    for model in document['models']:
        output_dir = DIRECTORY/model['id']
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = CACHE/model['id']
        cache_dir.mkdir(parents=True, exist_ok=True)
        assets = []
        for filename in model['source_asset_names']:
            url = f"https://huggingface.co/{model['source_repository']}/resolve/{model['source_revision']}/{filename}"
            data, _, _ = read_url(url, 20 << 20)
            target = cache_dir/filename
            expected = next((a for a in model.get('source_assets', []) if a['file'] == filename), None)
            if expected and digest(data) != expected['sha256']:
                raise ValueError('pinned source asset checksum mismatch')
            target.write_bytes(data)
            assets.append({'file': filename, 'bytes': len(data), 'sha256': digest(data)})
            if filename.endswith('.json') and filename not in ['tokenizer.json', 'vocab.json']:
                (output_dir/filename).write_bytes(data)
            if filename in ['LICENSE', 'LICENSE.txt']:
                (output_dir/filename).write_bytes(data)
        model['source_assets'] = assets
        for component in model['components']:
            prefix_bytes = min(component['bytes'], 8 << 20)
            url = f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/{component['file']}?download=true&coheryn_prefix={prefix_bytes}"
            data, status, headers = read_url(url, prefix_bytes, {'Range': f'bytes=0-{prefix_bytes-1}'})
            normalized = {k.lower(): v for k, v in headers.items()}
            expected_range = f"bytes 0-{prefix_bytes-1}/{component['bytes']}"
            if status != 206 or normalized.get('content-range') != expected_range or len(data) != prefix_bytes or data[:4] != b'GGUF':
                raise ValueError('server did not return the exact bounded GGUF prefix')
            prefix = cache_dir/(component['file']+'.prefix')
            previous = component.get('prefix_sha256')
            if previous and digest(data) != previous:
                raise ValueError('pinned header-prefix checksum mismatch')
            prefix.write_bytes(data)
            inventory = subprocess.check_output([str(ROOT/'target/release/examples/inspect_header'), str(prefix), str(component['bytes'])], cwd=ROOT)
            inventory_path = output_dir/(component['role']+'.inventory.json')
            inventory_path.write_bytes(inventory + b'\n')
            component.update({'prefix_bytes': len(data), 'prefix_sha256': digest(data),
                              'inventory': str(inventory_path.relative_to(ROOT)),
                              'header_status': 'bounded-prefix-parsed-payload-not-verified'})
            print(f"Inventoried {model['id']} {component['role']} header; payload remains unverified", flush=True)
    MANIFEST.write_text(json.dumps(document, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pin', action='store_true')
    args = parser.parse_args()
    if args.pin:
        pin()
    reproduce()


if __name__ == '__main__':
    main()

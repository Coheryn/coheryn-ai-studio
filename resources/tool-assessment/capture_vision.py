#!/usr/bin/env python3
"""Capture isolated CPU vision/projector references; never used by native CI/runtime."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.cache/image-reference'))
from PIL import Image


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    manifest = json.loads((ROOT / 'fixtures/vlm/models.json').read_text())
    binary = ROOT / '.cache/image-oracle/vision'
    revision = (ROOT / '.cache/reference/source-revision').read_text().strip()
    assert revision == '2280f0fd163b03d4331d49e08460f66502e35854'
    records = []
    for model in manifest['models']:
        component = next(c for c in model['components'] if c['role'] == 'vision-projector')
        weights = ROOT / '.cache/vlm/full' / model['id'] / component['file']
        assert weights.stat().st_size == component['bytes'] and sha(weights) == component['sha256']
        profile = 'llava15-pad336-v1' if model['id'] == 'llava-1.5-7b' else 'qwen2vl-still-v1'
        for filename in ['landscape-png.png', 'swapped-png.png']:
            image_path = ROOT / 'fixtures/vlm/preprocessing/canonical' / filename
            image = Image.open(image_path).convert('RGB')
            directory = ROOT / '.cache/vision-oracle' / model['id'] / image_path.stem
            directory.mkdir(parents=True, exist_ok=True)
            (directory / 'input.rgb').write_bytes(image.tobytes())
            start = time.monotonic()
            with (directory / 'capture.jsonl').open('w') as output, (directory / 'capture.log').open('w') as errors:
                subprocess.run([str(binary), str(weights), profile, *map(str, image.size), str(directory / 'input.rgb'), str(directory)], stdout=output, stderr=errors, check=True, timeout=900)
            lines = [json.loads(line) for line in (directory / 'capture.jsonl').read_text().splitlines()]
            assert 'hparams' in lines[0] and lines[-1]['name'] == 'projected_embeddings'
            for tensor in lines[1:]:
                raw = directory / (tensor['name'] + '.f32')
                assert raw.stat().st_size == tensor['count'] * 4
                tensor['sha256'] = sha(raw)
            records.append({'model': model['id'], 'component_sha256': component['sha256'], 'processor': profile,
                            'canonical': str(image_path.relative_to(ROOT)), 'canonical_sha256': sha(image_path),
                            'elapsed_seconds': time.monotonic() - start, 'settings': lines[0], 'tensors': lines[1:]})
            print(f"PASS independent vision/projector {model['id']} {filename}: {len(lines)-1} tensors", flush=True)
    target = ROOT / 'fixtures/vlm/vision-reference.json'
    target.write_text(json.dumps({'schema_version': 1, 'reference_revision': revision, 'binary_sha256': sha(binary),
        'wrapper_source_sha256': sha(ROOT / 'resources/tool-assessment/vision_oracle.cpp'),
        'capture_source_sha256': sha(Path(__file__)), 'backend': 'scalar CPU, one thread, F32 outputs, no GPU/BLAS/half intrinsics',
        'native_vision_execution_qualified': False, 'language_logits_captured': False,
        'limitations': ['Legacy selected LLaVA component/source layer and preprocessing alignment needs review before native graph acceptance.',
                        'No image-conditioned language logits, greedy continuation or multi-image qualification.'], 'cases': records}, indent=2) + '\n')


if __name__ == '__main__':
    main()

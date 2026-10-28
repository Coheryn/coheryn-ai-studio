#!/usr/bin/env python3
"""Validate committed image evidence offline; never execute reference tools."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    directory = ROOT / 'fixtures/vlm'
    reference = json.loads((directory / 'preprocessing/reference.json').read_text())
    assert reference['profile'] == 'coheryn-image-preprocess-v1'
    assert reference['reference_revision'] == json.loads((ROOT / 'fixtures/models.json').read_text())['oracle']['revision']
    for field, path in [('capture_source_sha256', 'resources/tool-assessment/capture_images.py'),
                        ('wrapper_source_sha256', 'resources/tool-assessment/image_oracle.cpp')]:
        assert sha(ROOT / path) == reference[field], path
    cases = reference['cases']
    assert len(cases) == len({case['id'] for case in cases}) == 20
    profile_manifest = json.loads((directory / 'preprocessing/profiles.json').read_text())
    assert profile_manifest['profile'] == reference['profile']
    assert profile_manifest['native_image_understanding_supported'] is False
    profiles = {'llava15-pad336-v1', 'qwen2vl-still-v1'}
    assert {item['id'] for item in profile_manifest['profiles']} == profiles
    assert {case['processor'] for case in cases} == profiles
    for case in cases:
        for key in ['source', 'canonical', 'resized']:
            path = (ROOT / case[key]).resolve()
            assert path.is_relative_to(directory.resolve())
            assert sha(path) == case[key + '_sha256'], str(path)
        width, height = case['processed_size']
        assert width % 14 == height % 14 == 0
        assert case['patch_rows'] == width // 14 * (height // 14)
        qwen = case['processor'] == 'qwen2vl-still-v1'
        assert case['patch_features'] == (1176 if qwen else 588)
        assert case['image_feature_tokens'] == case['patch_rows'] // (4 if qwen else 1)
        assert all(math.isfinite(sample['value']) for sample in case['normalized_samples'])
        for patch in case['patch_samples']:
            assert 0 <= patch['row'] < case['patch_rows']
            assert all(0 <= value['index'] < case['patch_features'] and math.isfinite(value['value']) for value in patch['values'])
    model_manifest = directory / 'models.json'
    components = json.loads((ROOT / 'fixtures/qualification/m0-components.json').read_text())
    assert components['selection_manifest_sha256'] == sha(model_manifest)
    selected = json.loads(model_manifest.read_text())
    expected = {(model['id'], component['role']): component for model in selected['models'] for component in model['components']}
    assert len(components['components']) == len(expected) == 4
    for profile in profile_manifest['profiles']:
        assert profile['component_sha256'] == expected[(profile['selected_model'], 'vision-projector')]['sha256']
    for item in components['components']:
        source = expected[(item['model'], item['role'])]
        assert all(item[key] == source[key] for key in ['file', 'bytes', 'sha256'])
        assert item['status'] == 'full-file-byte-verified' and not item['model_execution_qualified']
    alignment = json.loads((directory / 'alignment.json').read_text())
    assert alignment['status'] == 'review-required-before-native-vision-model-acceptance'
    assert alignment['vision_reference_sha256'] == sha(directory / 'vision-reference.json')
    for item in alignment['issues']:
        base = directory / item['model']
        assert sha(base / 'config.json') == item['source_config_sha256']
        assert sha(base / 'vision-projector.inventory.json') == item['inventory_sha256']
        assert item['image_sensitive_projector_outputs'] is True
    vision_path = directory / 'vision-reference.json'
    if vision_path.exists():
        vision = json.loads(vision_path.read_text())
        assert vision['reference_revision'] == reference['reference_revision']
        assert sha(ROOT / 'resources/tool-assessment/vision_oracle.cpp') == vision['wrapper_source_sha256']
        assert sha(ROOT / 'resources/tool-assessment/capture_vision.py') == vision['capture_source_sha256']
        assert len(vision['cases']) == 4
        assert not vision['native_vision_execution_qualified'] and not vision['language_logits_captured']
        for case in vision['cases']:
            assert case['component_sha256'] == expected[(case['model'], 'vision-projector')]['sha256']
            assert sha(ROOT / case['canonical']) == case['canonical_sha256']
            assert case['tensors'][-1]['name'] == 'projected_embeddings'
            for tensor in case['tensors']:
                assert math.prod(tensor['shape']) == tensor['count']
                assert all(math.isfinite(tensor[key]) for key in ['sum', 'l2', 'minimum', 'maximum'])
                assert all(0 <= sample['index'] < tensor['count'] and math.isfinite(sample['value']) for sample in tensor['samples'])
    print('PASS 20 image processor cases, four component identities, and available independent vision records (offline; no inference claim)')


if __name__ == '__main__':
    main()

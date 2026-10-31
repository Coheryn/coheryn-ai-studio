"""Explicit qualification coverage; architecture support and exact evidence remain separate."""
PROFILES = {
    'gguf-text': {'group': 'legacy-regression', 'family': 'qwen2', 'image': False},
    'embedding-f16': {'group': 'legacy-regression', 'family': 'embedding', 'image': False},
    'embedding-q8': {'group': 'legacy-regression', 'family': 'embedding', 'image': False},
}
for layout, group in [('publisher', 'safetensors'), ('sharded', 'safetensors')]:
    for image in (False, True):
        name = 'qwen35-' + ('' if layout == 'publisher' else 'sharded-') + ('image' if image else 'text')
        PROFILES[name] = {'group': group, 'family': 'qwen35', 'image': image, 'layout': layout,
                          'reference': 'fixtures/qwen35/vision/reference-v1.json' if image else 'fixtures/qwen35/reference-v1.json',
                          'arrays': 'qwen35-image' if image else 'qwen35-text'}
for label in ('f16', 'q8_0', 'q4_0', 'q4_k_m', 'q6_k_m'):
    for image in ((False, True) if label in ('f16', 'q4_k_m') else (False,)):
        profile = label + ('-image' if image else '-text')
        PROFILES['qwen35-gguf-' + profile] = {
            'group': 'gguf-vision' if image else 'gguf-quantized', 'family': 'qwen35', 'layout': 'gguf',
            'label': label, 'image': image, 'reference': 'fixtures/qwen35/gguf/' + profile + '.reference-v1.json',
            'arrays': 'qwen35-gguf-' + profile,
        }
GROUPS = tuple(dict.fromkeys(p['group'] for p in PROFILES.values()))


def selected(group):
    return [name for name, profile in PROFILES.items() if group == 'all' or profile['group'] == group]

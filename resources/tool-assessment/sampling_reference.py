"""Independent scalar arithmetic fixtures; never reads native sampler output."""
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[2]


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def reference(case):
    settings, penalties = case['settings'], case['penalties']
    counts = [0] * len(case['logits'])
    for token in case['history']:
        counts[token] += 1
    state = settings['seed']
    result = []
    for _ in range(24):
        scores = []
        for token, raw in enumerate(case['logits']):
            if not case['allowed'][token]:
                continue
            score = f32(raw)
            if counts[token]:
                repeat = f32(penalties['repetition_penalty'])
                score = score * repeat if score < 0 else score / repeat
                score -= f32(penalties['presence_penalty']) + counts[token] * f32(penalties['frequency_penalty'])
            scores.append((token, score))
        scores.sort(key=lambda pair: (-pair[1], pair[0]))
        if settings['temperature'] == 0:
            selected = scores[0][0]
        else:
            if settings['top_k']:
                scores = scores[:settings['top_k']]
            probabilities = [math.exp((s - scores[0][1]) / f32(settings['temperature'])) for _, s in scores]
            total = sum(probabilities)
            cumulative = 0
            keep = 0
            for probability in probabilities:
                cumulative += probability / total
                keep += 1
                if cumulative >= f32(settings['top_p']):
                    break
            probabilities = probabilities[:keep]
            state = (state + 0x9e3779b97f4a7c15) & ((1 << 64) - 1)
            z = state
            z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & ((1 << 64) - 1)
            z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & ((1 << 64) - 1)
            random = ((z ^ (z >> 31)) >> 11) / (1 << 53)
            threshold = random * sum(probabilities)
            cumulative = 0
            selected = scores[keep - 1][0]
            for (token, _), probability in zip(scores, probabilities):
                cumulative += probability
                if threshold < cumulative:
                    selected = token
                    break
        counts[selected] += 1
        result.append(selected)
    return result


cases = [
    {'id': 'greedy-tie', 'logits': [0, 0, -1], 'allowed': [True, True, True], 'history': [], 'settings': {'temperature': 0, 'top_k': 0, 'top_p': 1, 'seed': 0}, 'penalties': {'repetition_penalty': 1, 'frequency_penalty': 0, 'presence_penalty': 0}},
    {'id': 'seeded-masked-top-k-p', 'logits': [1, 200, 3, 0, 100], 'allowed': [True, False, True, True, False], 'history': [], 'settings': {'temperature': 0.8, 'top_k': 2, 'top_p': 0.99, 'seed': 42}, 'penalties': {'repetition_penalty': 1, 'frequency_penalty': 0, 'presence_penalty': 0}},
    {'id': 'penalties-and-history', 'logits': [2, 2, 1, 0], 'allowed': [True, True, True, True], 'history': [0, 1, 0], 'settings': {'temperature': 0.7, 'top_k': 0, 'top_p': 0.95, 'seed': 7}, 'penalties': {'repetition_penalty': 1.1, 'frequency_penalty': 0.5, 'presence_penalty': 0.1}},
    {'id': 'negative-frequency-and-max-seed', 'logits': [-1, 0.5, 1, -2], 'allowed': [True, True, True, True], 'history': [0, 0, 3], 'settings': {'temperature': 1.2, 'top_k': 3, 'top_p': 0.8, 'seed': (1 << 64) - 1}, 'penalties': {'repetition_penalty': 0.75, 'frequency_penalty': -0.25, 'presence_penalty': -0.5}},
]
for case in cases:
    case['expected_ids'] = reference(case)
output = ROOT / 'fixtures/constraints/sampling-v2.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({'schema_version': 1, 'profile': 'coheryn-sampling-v2', 'source': 'resources/tool-assessment/sampling_reference.py', 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'scope': 'Scalar arithmetic/RNG reference with f32 inputs and f64 probabilities; not model inference or cross-device bitwise qualification.', 'cases': cases}, indent=2) + '\n')

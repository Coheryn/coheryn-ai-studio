"""Freeze independent Qwen3.5 GGUF arrays before native executor acceptance.

Own model equations plus a separately pinned GGUF reader/dequantizer. This is an
isolated assessment command; production/native CI only consumes frozen data.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from qwen35_gguf_weights import ROOT, Weights, np, sha
for relative in ('.cache/devtools', '.cache/q35-assessment-pip'):
    sys.path.insert(0, str(ROOT / relative))
from jinja2.sandbox import SandboxedEnvironment
from tokenizers import Tokenizer
F = np.float32


def equations(store, image):
    namespace = {'np': np, 'F': F, 'math': math, 'c': config['text_config'], 'vc': config['vision_config'],
                 'header': store.header, 'weight': store.weight, 'project': store.project}
    source = (ROOT / 'resources/tool-assessment/qwen35_numpy.py').read_text()
    # GGUF stores these coefficients already folded. The equations consume the
    # actual folded values, avoiding inverse transforms of quantized parameters.
    source = source.replace('(w+F(1) if centered else w)', 'w')
    source = source.replace("-np.exp(weight(p+'.A_log'))", "weight(p+'.A_log')")
    if image:
        source = source.replace('def forward(self, token, logits=False):', 'def forward(self, token, logits=False, embedding=None, coordinate=None):')
        source = source.replace("hidden = weight('model.language_model.embed_tokens.weight', token, token+1)[0]", "hidden = weight('model.language_model.embed_tokens.weight', token, token+1)[0] if embedding is None else np.array(embedding,dtype=np.float32)")
        source = source.replace('self.position)', 'self.position if coordinate is None else coordinate)')
    names = {'sigmoid', 'silu', 'rms', 'Reference'} | (set() if image else {'rope'})
    nodes = [n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'independent-GGUF-equations', 'exec'), namespace)
    if image:
        vision_source = (ROOT / 'resources/tool-assessment/qwen35_vision_array.py').read_text()
        nodes = [n for n in ast.parse(vision_source).body if isinstance(n, ast.FunctionDef) and n.name in
                 {'rope', 'project_many', 'norm', 'gelu', 'encode'}]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'independent-GGUF-image-equations', 'exec'), namespace)
    return namespace


def save(name, values):
    value = np.asarray(values, dtype='<f4')
    data = value.tobytes()
    path = output / (name + '.f32')
    with path.open('xb') as stream: stream.write(data)
    flat = value.ravel()
    indices = sorted({0, len(flat) - 1, *range(0, len(flat), max(1, len(flat) // 64))})
    return {'name': name, 'shape': list(value.shape), 'bytes': len(data), 'length': len(flat),
            'sha256': hashlib.sha256(data).hexdigest(), 'indices': indices, 'values': [float(flat[i]) for i in indices]}


def main():
    global config, output
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True, choices=['f16', 'q4_0', 'q8_0', 'q4_k_m', 'q6_k_m'])
    parser.add_argument('--image', action='store_true')
    args = parser.parse_args()
    profile = args.label + ('-image' if args.image else '-text')
    target = ROOT / 'fixtures/qwen35/gguf' / (profile + '.reference-v1.json')
    if target.exists(): raise FileExistsError(target)
    output = ROOT / '.cache/qwen35-gguf-reference' / profile
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / '.cache/qwen3.5-0.8b-source/config.json').read_text())
    store = Weights(args.label, vision=args.image)
    functions = equations(store, args.image)
    functions['save'] = save
    tokenizer = Tokenizer.from_file(str(ROOT / '.cache/qwen3.5-0.8b-source/tokenizer.json'))
    template = SandboxedEnvironment().from_string((ROOT / '.cache/qwen3.5-0.8b-source/chat_template.jinja').read_text())
    base = json.loads((ROOT / ('fixtures/qwen35/vision/reference-v1.json' if args.image else 'fixtures/qwen35/reference-v1.json')).read_text())
    contract = json.loads((ROOT / 'fixtures/qwen35/vision/contract-v1.json').read_text())
    features, encoder = {}, []
    if args.image:
        for item in contract['images']:
            features[item['id']], arrays = functions['encode'](item['id'], item['rgb'], item['grid'])
            encoder += arrays
    cases = []
    begin = time.monotonic()
    for original in base['cases']:
        messages = original.get('messages')
        if args.image:
            messages = [{'role': 'user', 'content': [{'type': 'image'} for _ in original['images']] + [{'type': 'text', 'text': original['text']}]}]
        prompt = template.render(messages=messages, tools=[], add_generation_prompt=True, enable_thinking=False, add_vision_id=False)
        ids, entries, position, cursor = [], [], 0, 0
        for token in tokenizer.encode(prompt, add_special_tokens=False).ids:
            if args.image and token == config['image_token_id']:
                key = original['images'][cursor]
                item = next(i for i in contract['images'] if i['id'] == key)
                gh, gw = item['grid']
                for j, embedding in enumerate(features[key]):
                    ids.append(token); entries.append((embedding, (position, position + j // (gw // 2), position + j % (gw // 2))))
                position += max(gh // 2, gw // 2); cursor += 1
            else:
                ids.append(token); entries.append((None, (position,) * 3)); position += 1
        reference = functions['Reference']()
        for index, (token, (embedding, coordinate)) in enumerate(zip(ids, entries)):
            layers, logits = reference.forward(token, index + 1 == len(ids), embedding, coordinate) if args.image else reference.forward(token, index + 1 == len(ids))
            if index % 16 == 0: print(profile, original['id'], 'prefill', index + 1, '/', len(ids), flush=True)
        checkpoints = [save(original['id'] + f'-layer.{i}', value) for i, value in enumerate(layers)]
        tokens, teacher = [], []
        for step in range(4):
            teacher.append(save(original['id'] + f'-logits.{step}', logits))
            token = int(np.argmax(logits)); tokens.append(token)
            if token in (248044, 248046): break
            if step < 3:
                _, logits = reference.forward(token, True, None, (position,) * 3) if args.image else reference.forward(token, True)
                position += 1
        cases.append({**{k: v for k, v in original.items() if k in ('id', 'text', 'messages', 'images')},
                      'prompt': prompt, 'prompt_token_ids': ids, 'checkpoints': checkpoints, 'teacher_forced': teacher,
                      'greedy_token_ids': tokens, 'greedy_text': tokenizer.decode(tokens, skip_special_tokens=True)})
        print('CAPTURED', profile, original['id'], repr(cases[-1]['greedy_text']), flush=True)
    report = {'schema_version': 1, 'label': args.label, 'profile': 'Independent GGUF weights / F32 activations-products / wide reductions; folded norms and SSM coefficients.',
              'provenance': store.provenance(), 'script_sha256': sha(Path(__file__)),
              'equation_sources': {name: sha(ROOT / 'resources/tool-assessment' / name) for name in
                                   ('qwen35_numpy.py', 'qwen35_vision_array.py', 'qwen35_gguf_weights.py')},
              'cases': cases, 'encoder_checkpoints': encoder, 'wall_seconds': time.monotonic() - begin,
              'tolerances': contract['tolerances'] if args.image else json.loads((ROOT / 'fixtures/qwen35/reference-v1.json').read_text())['tolerances'],
              'publisher_framework_parity': False, 'native_executor_invoked': False}
    with target.open('x') as stream: json.dump(report, stream, ensure_ascii=False, indent=2); stream.write('\n')


if __name__ == '__main__':
    main()

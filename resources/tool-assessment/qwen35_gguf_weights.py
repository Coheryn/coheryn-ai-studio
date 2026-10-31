"""Independent, ignored-wheel GGUF reading/dequantization for numerical assessment.

Only the reader, constants, lazy arrays and dequantizer modules are loaded. Writer,
network, vocabulary and GUI utilities are not executed. No native Coheryn module
is imported, and this module is outside production and native qualification jobs.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
for directory in ('.cache/vision-numpy', '.cache/devtools', '.cache/q35-assessment-pip'):
    sys.path.insert(0, str(ROOT / directory))
import numpy as np

ASSESSMENT = ROOT / '.cache/qwen35-gguf-assessment'
namespace = types.ModuleType('gguf')
namespace.__path__ = [str(ASSESSMENT / 'gguf')]
sys.modules['gguf'] = namespace
from gguf.gguf_reader import GGUFReader
from gguf.quants import dequantize


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Weights:
    def __init__(self, label, vision=True):
        manifest = json.loads((ROOT / 'fixtures/qwen35/gguf/models-v1.json').read_text())
        self.manifest = manifest
        self.readers, self.tensors, self.files = [], {}, []
        for role in ('language', 'vision') if vision else ('language',):
            item = next(i for i in manifest['files'] if i['role'] == role and (role == 'vision' or i['label'] == label))
            path = ROOT / manifest['cache_directory'] / item.get('cache_file', item['file'])
            if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
                raise ValueError('independent GGUF input identity mismatch: ' + item['file'])
            reader = GGUFReader(str(path))
            self.readers.append(reader)
            for tensor in reader.tensors:
                if tensor.name in self.tensors:
                    raise ValueError('duplicate physical GGUF tensor')
                self.tensors[tensor.name] = tensor
            self.files.append(item)
        source = json.loads((ROOT / 'fixtures/qwen35/sharded-layout-v1.json').read_text())
        self.header = {t['name']: {'shape': t['shape']} for t in source['tensors']
                       if t['name'].startswith('model.language_model.') or (vision and t['name'].startswith('model.visual.'))}
        used = set()
        for name in self.header:
            physical = self.resolve(name)
            used.update(physical)
            for key in physical:
                if key not in self.tensors:
                    raise ValueError('missing independent GGUF tensor: ' + key)
        if used != set(self.tensors):
            raise ValueError('unmapped GGUF tensors: ' + repr(set(self.tensors) - used))

    @staticmethod
    def resolve(name):
        if name == 'model.language_model.embed_tokens.weight': return ['token_embd.weight']
        if name == 'model.language_model.norm.weight': return ['output_norm.weight']
        if name.startswith('model.language_model.layers.'):
            index, stem = name.removeprefix('model.language_model.layers.').split('.', 1)
            suffix = {
                'input_layernorm.weight': 'attn_norm.weight', 'post_attention_layernorm.weight': 'post_attention_norm.weight',
                'mlp.gate_proj.weight': 'ffn_gate.weight', 'mlp.up_proj.weight': 'ffn_up.weight', 'mlp.down_proj.weight': 'ffn_down.weight',
                'self_attn.q_proj.weight': 'attn_q.weight', 'self_attn.k_proj.weight': 'attn_k.weight', 'self_attn.v_proj.weight': 'attn_v.weight',
                'self_attn.o_proj.weight': 'attn_output.weight', 'self_attn.q_norm.weight': 'attn_q_norm.weight', 'self_attn.k_norm.weight': 'attn_k_norm.weight',
                'linear_attn.in_proj_qkv.weight': 'attn_qkv.weight', 'linear_attn.in_proj_z.weight': 'attn_gate.weight',
                'linear_attn.in_proj_a.weight': 'ssm_alpha.weight', 'linear_attn.in_proj_b.weight': 'ssm_beta.weight',
                'linear_attn.conv1d.weight': 'ssm_conv1d.weight', 'linear_attn.A_log': 'ssm_a', 'linear_attn.dt_bias': 'ssm_dt.bias',
                'linear_attn.norm.weight': 'ssm_norm.weight', 'linear_attn.out_proj.weight': 'ssm_out.weight',
            }[stem]
            return [f'blk.{index}.{suffix}']
        name = name.removeprefix('model.visual.')
        if name == 'patch_embed.proj.weight': return ['v.patch_embd.weight', 'v.patch_embd.weight.1']
        if name.startswith('blocks.'):
            index, stem = name.removeprefix('blocks.').split('.', 1)
            stem, suffix = stem.rsplit('.', 1)
            key = {'norm1': 'ln1', 'norm2': 'ln2', 'attn.qkv': 'attn_qkv', 'attn.proj': 'attn_out',
                   'mlp.linear_fc1': 'ffn_up', 'mlp.linear_fc2': 'ffn_down'}[stem]
            return [f'v.blk.{index}.{key}.{suffix}']
        return [{'pos_embed.weight': 'v.position_embd.weight', 'patch_embed.proj.bias': 'v.patch_embd.bias',
                 'merger.norm.weight': 'v.post_ln.weight', 'merger.norm.bias': 'v.post_ln.bias',
                 'merger.linear_fc1.weight': 'mm.0.weight', 'merger.linear_fc1.bias': 'mm.0.bias',
                 'merger.linear_fc2.weight': 'mm.2.weight', 'merger.linear_fc2.bias': 'mm.2.bias'}[name]]

    def weight(self, name, start=None, end=None):
        shape = self.header[name]['shape']
        keys = self.resolve(name)
        if len(keys) == 2:
            values = []
            for key in keys:
                t = self.tensors[key]
                data = t.data if start is None else t.data[start:end]
                values.append(dequantize(data, t.tensor_type).reshape(-1, 3, 16, 16))
            return np.stack(values, axis=2).astype(np.float32)
        t = self.tensors[keys[0]]
        data = t.data if start is None else t.data[start:end]
        result = dequantize(data, t.tensor_type).astype(np.float32)
        return result.reshape(shape if start is None else (len(result), *shape[1:]))

    def project(self, name, x):
        rows = self.header[name]['shape'][0]
        result = np.empty(rows, dtype=np.float32)
        for start in range(0, rows, 64):
            w = self.weight(name, start, min(rows, start + 64))
            result[start:start + len(w)] = np.sum(w * x, axis=-1, dtype=np.float64).astype(np.float32)
        return result

    def provenance(self):
        tool = json.loads((ROOT / 'resources/tool-assessment/modern-gguf-tool-v1.json').read_text())
        version = importlib.metadata.Distribution.at(ASSESSMENT / f"gguf-{tool['version']}.dist-info").version
        if version != tool['version']: raise ValueError('independent reader version changed')
        return {'tool': tool, 'numpy': np.__version__, 'files': self.files,
                'reader_modules': {name: sha(ASSESSMENT / 'gguf' / name) for name in ('constants.py', 'gguf_reader.py', 'quants.py', 'lazy.py')}}

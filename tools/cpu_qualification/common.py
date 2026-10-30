"""Fail-closed identities and numerical comparisons, including under Python -O."""
import array
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def pinned(path, item):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'linked input: ' + str(path))
    require(path.is_file() and path.stat().st_size == item['bytes'] and sha(path) == item['sha256'],
            'missing or changed pinned input: ' + str(path))
    return path


def vector(path):
    values = array.array('f')
    values.frombytes(Path(path).read_bytes())
    if sys.byteorder != 'little':
        values.byteswap()
    return values


def compare(actual, expected, absolute, relative, l2_limit=None):
    require(len(actual) == len(expected) and len(actual) > 0, 'numeric shape mismatch')
    require(all(math.isfinite(x) for x in actual) and all(math.isfinite(x) for x in expected),
            'non-finite numeric value')
    differences = [abs(a - b) for a, b in zip(actual, expected)]
    outside = sum(d > absolute + relative * abs(b) for d, b in zip(differences, expected))
    l2 = math.sqrt(math.fsum(d * d for d in differences)) / max(
        math.sqrt(math.fsum(b * b for b in expected)), 1e-30)
    require(outside == 0 and (l2_limit is None or l2 <= l2_limit),
            f'numeric mismatch: outside={outside}, max={max(differences)}, normalized_l2={l2}')
    return {'values': len(actual), 'max_absolute': max(differences), 'normalized_l2': l2}


def clean_environment():
    env = {k: v for k, v in os.environ.items() if k in ('SystemRoot', 'WINDIR', 'TMP', 'TEMP', 'TMPDIR')}
    return {**env, 'PATH': '', 'COHERYN_ENDPOINT': 'http://127.0.0.1:1'}

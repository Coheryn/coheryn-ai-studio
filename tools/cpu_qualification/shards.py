"""Deterministic whole-tensor resharding of pinned source bytes, without numeric conversion."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile

from .common import pinned, require, sha, write

VERSION = 'whole-tensor-balanced-v1'
DTYPES = {'F32': 4, 'F16': 2, 'BF16': 2}


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate Safetensors JSON key')
        result[key] = value
    return result


def catalog(stream, size):
    prefix = stream.read(8)
    require(len(prefix) == 8, 'truncated Safetensors prefix')
    count = struct.unpack('<Q', prefix)[0]
    require(2 <= count <= 8 << 20 and count <= size - 8, 'Safetensors header outside limits')
    header = json.loads(stream.read(count), object_pairs_hook=unique)
    require(isinstance(header, dict) and 1 <= len(header) <= 100001, 'invalid tensor catalog')
    metadata = header.pop('__metadata__', {})
    require(isinstance(metadata, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()), 'invalid source metadata')
    tensors = []
    for name, tensor in header.items():
        require(isinstance(name, str) and 0 < len(name) <= 1024 and isinstance(tensor, dict)
                and set(tensor) == {'dtype', 'shape', 'data_offsets'}, 'invalid tensor descriptor')
        dtype, shape, offsets = tensor['dtype'], tensor['shape'], tensor['data_offsets']
        require(isinstance(dtype, str) and dtype in DTYPES and isinstance(shape, list) and len(shape) <= 8
                and all(type(n) is int and 0 < n <= 1 << 32 for n in shape), 'unsupported tensor shape/dtype')
        elements = 1
        for dimension in shape:
            elements *= dimension
        require(isinstance(offsets, list) and len(offsets) == 2 and all(type(n) is int for n in offsets), 'invalid tensor offsets')
        start, end = offsets
        require(0 <= start < end <= size - 8 - count and end - start == elements * DTYPES[dtype], 'tensor byte range mismatch')
        tensors.append({'name': name, 'dtype': dtype, 'shape': shape, 'offset': start, 'bytes': end - start})
    cursor = 0
    for tensor in sorted(tensors, key=lambda item: item['offset']):
        require(tensor['offset'] == cursor, 'tensor overlap or uncovered source bytes')
        cursor += tensor['bytes']
    require(tensors and cursor == size - 8 - count, 'uncovered source payload')
    return 8 + count, metadata, sorted(tensors, key=lambda item: item['name'])


def derive(source, destination, identity, assets, shards=3, check=None):
    """Return a byte-verifiable real-model layout; never create expected inference values."""
    source, destination = Path(source), Path(destination)
    pinned(source, identity)
    require(2 <= shards <= 16, 'derived layout requires two to sixteen physical shards')
    require(not destination.exists(), 'derived layout already exists; verify its frozen manifest instead')
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(not any(p.is_symlink() for p in (destination.parent, *destination.parent.parents)), 'linked shard destination')
    require(shutil.disk_usage(destination.parent).free >= identity['bytes'] + (128 << 20), 'insufficient space for lossless shards')
    staging = Path(tempfile.mkdtemp(prefix='.reshard-', dir=destination.parent))
    try:
        with source.open('rb') as input_stream:
            initial = os.fstat(input_stream.fileno())
            offset, metadata, tensors = catalog(input_stream, initial.st_size)
            require(shards <= len(tensors), 'every shard must contain real tensors')
            buckets, sizes = [[] for _ in range(shards)], [0] * shards
            for tensor in tensors:
                selected = min(range(shards), key=lambda i: (sizes[i], i))
                buckets[selected].append(tensor)
                sizes[selected] += tensor['bytes']
            index, records, payloads = {}, [], []
            for number, bucket in enumerate(buckets, 1):
                name = f'model-{number:05}-of-{shards:05}.safetensors'
                header, cursor = {'__metadata__': metadata}, 0
                for tensor in bucket:
                    header[tensor['name']] = {'dtype': tensor['dtype'], 'shape': tensor['shape'], 'data_offsets': [cursor, cursor + tensor['bytes']]}
                    index[tensor['name']] = name
                    cursor += tensor['bytes']
                encoded = json.dumps(header, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
                encoded += b' ' * (-len(encoded) % 8)
                require(len(encoded) <= 8 << 20, 'derived shard header exceeds native bound')
                with (staging / name).open('xb') as output:
                    output.write(struct.pack('<Q', len(encoded)))
                    output.write(encoded)
                    for tensor in bucket:
                        input_stream.seek(offset + tensor['offset'])
                        remaining, digest = tensor['bytes'], hashlib.sha256()
                        while remaining:
                            if check:
                                check()
                            chunk = input_stream.read(min(1 << 20, remaining))
                            require(chunk, 'source truncated during resharding')
                            output.write(chunk)
                            digest.update(chunk)
                            remaining -= len(chunk)
                        records.append({**tensor, 'shard': name, 'sha256': digest.hexdigest()})
                    output.flush()
                    os.fsync(output.fileno())
                payloads.append({'file': name, 'bytes': (staging / name).stat().st_size, 'sha256': sha(staging / name)})
            final = os.fstat(input_stream.fileno())
            require((initial.st_size, initial.st_mtime_ns, initial.st_ino) == (final.st_size, final.st_mtime_ns, final.st_ino), 'source changed while resharding')
        pinned(source, identity)
        write(staging / 'model.safetensors.index.json', {'metadata': {'total_size': sum(sizes)}, 'weight_map': dict(sorted(index.items()))})
        for asset in assets:
            if check:
                check()
            name = asset['file']
            require(Path(name).name == name and name not in index.values() and name != 'model.safetensors.index.json', 'unsafe or colliding sidecar')
            original = pinned(source.parent / name, asset)
            shutil.copyfile(original, staging / name)
            pinned(staging / name, asset)
        for path in sorted(staging.iterdir()):
            if not any(item['file'] == path.name for item in payloads):
                payloads.append({'file': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path)})
        record = {'schema_version': 1, 'recipe': VERSION, 'source': identity, 'physical_shards': shards,
                  'tensor_count': len(tensors), 'payload_bytes': sum(sizes), 'numeric_conversion': False,
                  'publisher_sharded_artifact': False, 'files': sorted(payloads, key=lambda item: item['file']),
                  'tensors': sorted(records, key=lambda item: item['name'])}
        write(staging / 'DERIVATION.json', record)
        if check:
            check()
        require(not destination.exists(), 'derived destination appeared during staging')
        os.rename(staging, destination)
        return record
    finally:
        if staging.exists():
            shutil.rmtree(staging)

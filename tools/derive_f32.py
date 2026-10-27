#!/usr/bin/env python3
"""Reproducibly promote the pinned GGUF's F16 tensors to F32, development only."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile

from fetch_model import digest

ROOT = Path(__file__).resolve().parents[1]

def read_number(file, fmt):
    size = struct.calcsize(fmt)
    data = file.read(size)
    if len(data) != size:
        raise ValueError('Truncated pinned GGUF')
    return struct.unpack(fmt, data)[0]

def read_string(file):
    size = read_number(file, '<Q')
    if size > 8 << 20:
        raise ValueError('Oversized string')
    return file.read(size).decode('utf-8')

def skip_value(file, kind, depth=0):
    if depth > 4:
        raise ValueError('Nested metadata exceeds fixture bounds')
    sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    if kind in sizes:
        file.seek(sizes[kind], 1)
    elif kind == 8:
        read_string(file)
    elif kind == 9:
        item, count = read_number(file, '<I'), read_number(file, '<Q')
        if count > 1_000_000:
            raise ValueError('Oversized metadata array')
        for _ in range(count):
            skip_value(file, item, depth+1)
    else:
        raise ValueError('Unknown metadata type')

def main():
    pin = json.loads((ROOT/'fixtures/models.json').read_text())['models'][0]
    source = ROOT/'.cache/models'/pin['file']
    if digest(source) != pin['sha256']:
        raise SystemExit('Derivation requires the exact pinned source artifact')
    target = source.with_name('qwen2.5-0.5b-instruct-derived-f32.gguf')
    if target.exists():
        raise SystemExit('Derived artifact already exists; remove it explicitly to regenerate')
    if shutil.disk_usage(source.parent).free < 3_000_000_000:
        raise SystemExit('Insufficient disk space for F32 derivation')
    with source.open('rb') as file:
        if file.read(8) != b'GGUF\x03\0\0\0':
            raise ValueError('Expected little-endian GGUF v3')
        count, metadata = read_number(file, '<Q'), read_number(file, '<Q')
        alignment, recipe_position = 32, None
        for _ in range(metadata):
            key, kind = read_string(file), read_number(file, '<I')
            if key == 'general.file_type':
                if kind != 4:
                    raise ValueError('Expected U32 file type')
                recipe_position = file.tell()
            if key == 'general.alignment':
                if kind != 4:
                    raise ValueError('Expected U32 alignment')
                alignment = read_number(file, '<I')
            else:
                skip_value(file, kind)
        tensors, next_offset = [], 0
        for _ in range(count):
            name, rank = read_string(file), read_number(file, '<I')
            dimensions = [read_number(file, '<Q') for _ in range(rank)]
            dtype_position = file.tell()
            dtype, offset = read_number(file, '<I'), read_number(file, '<Q')
            if dtype not in (0,1):
                raise ValueError('Only F32/F16 source tensors are accepted')
            elements = 1
            for dim in dimensions:
                elements *= dim
            next_offset = (next_offset+alignment-1)//alignment*alignment
            tensors.append((name,dtype_position,dtype,offset,elements,next_offset))
            next_offset += elements*4
        data_offset = (file.tell()+alignment-1)//alignment*alignment
        file.seek(0)
        header = bytearray(file.read(data_offset))
        if recipe_position is not None:
            struct.pack_into('<I',header,recipe_position,0)
        for _,position,_,_,_,offset in tensors:
            struct.pack_into('<IQ',header,position,0,offset)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=source.parent,suffix='.part',delete=False) as output:
                temporary = Path(output.name)
                output.write(header)
                for _,_,dtype,offset,elements,destination in tensors:
                    output.write(b'\0'*(data_offset+destination-output.tell()))
                    file.seek(data_offset+offset)
                    size = 4 if dtype == 0 else 2
                    while elements:
                        n = min(elements,32768)
                        chunk = file.read(n*size)
                        if len(chunk) != n*size:
                            raise ValueError('Truncated weight tensor')
                        if dtype == 1:
                            values = struct.unpack(f'<{n}e',chunk)
                            chunk = struct.pack(f'<{n}f',*values)
                        output.write(chunk)
                        elements -= n
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    provenance = {'source_sha256':pin['sha256'],'conversion':'F16-to-F32 exact promotion; existing F32 preserved; general.file_type=0',
                  'file':target.name,'bytes':target.stat().st_size,'sha256':digest(target)}
    target.with_suffix('.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance))

if __name__ == '__main__':
    main()

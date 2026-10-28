#!/usr/bin/env python3
"""Independent, isolated image-processor capture; never imports Coheryn numeric code.

Uses the pinned assessment checkout and Pillow for canonical decoding/resizing.
This is not part of native CI/build/runtime and is never an inference fallback.
"""
import array
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.cache/image-reference'))
from PIL import Image, ImageOps
import PIL
OUT = ROOT / 'fixtures/vlm/preprocessing'
REFERENCE = ROOT / '.cache/reference'
REVISION = '2280f0fd163b03d4331d49e08460f66502e35854'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def geometry(profile, width, height):
    if profile == 'llava15-pad336-v1':
        scale = min(f32(336 / width), f32(336 / height))
        return (336, 336), (min(336, math.ceil(f32(width * scale))), min(336, math.ceil(f32(height * scale))))
    w = max(28, round(f32(width / 28)) * 28)
    h = max(28, round(f32(height / 28)) * 28)
    if w * h > 12845056:
        factor = f32(math.sqrt(f32(f32(width * height) / 12845056)))
        w = max(28, math.floor(f32(f32(width / factor) / 28)) * 28)
        h = max(28, math.floor(f32(f32(height / factor) / 28)) * 28)
    elif w * h < 3136:
        factor = f32(math.sqrt(f32(3136 / f32(width * height))))
        w = math.ceil(f32(f32(width * factor) / 28)) * 28
        h = math.ceil(f32(f32(height * factor) / 28)) * 28
    return (w, h), (w, h)


def patches(values, width, height, temporal):
    grid_w, grid_h = width // 14, height // 14
    digest_value = hashlib.sha256()
    selected = []
    for row in range(grid_w * grid_h):
        if temporal == 2:
            group = row // 4
            py = group // (grid_w // 2) * 2 + row % 4 // 2
            px = group % (grid_w // 2) * 2 + row % 2
        else:
            py, px = divmod(row, grid_w)
        patch = [values[((py * 14 + y) * width + px * 14 + x) * 3 + channel]
                 for channel in range(3) for _ in range(temporal) for y in range(14) for x in range(14)]
        digest_value.update(struct.pack('<' + 'f' * len(patch), *patch))
        if row in {0, 1, 2, grid_w * grid_h // 2, grid_w * grid_h - 1}:
            selected.append({'row': row, 'values': [{'index': index, 'value': patch[index]} for index in [0, 13, 195, 196, len(patch) - 1]]})
    return digest_value.hexdigest(), selected


def main():
    assert PIL.__version__ == '13.0.0'
    assert (REFERENCE / 'source-revision').read_text().strip() == REVISION
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'inputs').mkdir(exist_ok=True)
    (OUT / 'canonical').mkdir(exist_ok=True)
    (OUT / 'resized').mkdir(exist_ok=True)
    for name, width, height in [('odd-rgb.png', 37, 19), ('downscale.png', 1025, 769)]:
        image = Image.new('RGB', (width, height))
        image.putdata([((x * 13 + y * 7) % 256, (x * 3 + y * 29) % 256, (x * 19 + y * 11) % 256)
                       for y in range(height) for x in range(width)])
        image.save(OUT / 'inputs' / name)
    gray = Image.new('L', (33, 17))
    gray.putdata([(x * 17 + y * 5) % 256 for y in range(17) for x in range(33)])
    gray.save(OUT / 'inputs/gray.png')
    original = (ROOT / 'fixtures/vlm/images/landscape.jpg').read_bytes()
    # Retain the immutable source JPEG, but replace its original EXIF in the new
    # orientation fixture so the image has one unambiguous primary directory.
    retained = bytearray(original[:2])
    cursor = 2
    while original[cursor:cursor + 2] != b'\xff\xda':
        marker = original[cursor + 1]
        length = int.from_bytes(original[cursor + 2:cursor + 4], 'big')
        segment = original[cursor:cursor + 2 + length]
        if not (marker == 0xe1 and segment[4:].startswith(b'Exif\0\0')):
            retained.extend(segment)
        cursor += 2 + length
    retained.extend(original[cursor:])
    original = bytes(retained)
    exif = b'Exif\0\0II' + struct.pack('<HIH', 42, 8, 1) + struct.pack('<HHIHHI', 0x112, 3, 1, 6, 0, 0)
    (OUT / 'inputs/exif6.jpg').write_bytes(original[:2] + b'\xff\xe1' + struct.pack('>H', len(exif) + 2) + exif + original[2:])
    inputs = [ROOT / 'fixtures/vlm/images' / name for name in
              ['landscape.png', 'swapped.png', 'portrait.png', 'minimum.png', 'alpha.png', 'landscape.jpg']]
    inputs += [OUT / 'inputs' / name for name in ['odd-rgb.png', 'downscale.png', 'gray.png', 'exif6.jpg']]
    source_files = ['tools/mtmd/mtmd-image.cpp', 'tools/mtmd/mtmd-image.h', 'tools/mtmd/clip-impl.h', 'tools/mtmd/clip-model.h', 'tools/mtmd/clip.cpp']
    report = {'schema_version': 1, 'profile': 'coheryn-image-preprocess-v1', 'reference_revision': REVISION,
              'reference_sources': {name: digest(REFERENCE / name) for name in source_files},
              'capture_source_sha256': digest(Path(__file__)), 'wrapper_source_sha256': digest(ROOT / 'resources/tool-assessment/image_oracle.cpp'),
              'reference_binary_sha256': digest(ROOT / '.cache/image-oracle/processor'),
              'decoder_reference': {'package': 'Pillow', 'version': PIL.__version__},
              'settings': {'compiler': 'clang++ -std=c++17 -O2 -ffp-contract=off', 'alpha': 'discard alpha; retain RGB samples',
                           'orientation': 'apply primary EXIF orientation before resizing', 'normalization': 'HWC; each division/subtraction rounds to f32',
                           'resize_pass_order': 'horizontal then vertical, u8 rounding after each; explicitly separate Pillow calls to match the pinned assessment processor',
                           'qwen_limits': 'explicit source config min_pixels=3136, max_pixels=12845056; not assessment defaults'},
              'tolerances': {'png_pixels_absolute': 0, 'jpeg_pixels_absolute': 2, 'canonical_resized_pixels_absolute': 0,
                             'normalized_absolute': 1e-5, 'normalized_relative': 1e-5},
              'vision_encoder_outputs_captured': False, 'cases': []}
    for path in inputs:
        with Image.open(path) as original:
            oriented = ImageOps.exif_transpose(original).convert('RGB')
            rgb = oriented.tobytes()
            canonical = OUT / 'canonical' / (path.name.replace('.', '-') + '.png')
            oriented.save(canonical)
        for profile in ['llava15-pad336-v1', 'qwen2vl-still-v1']:
            target, resized = geometry(profile, *oriented.size)
            rendered = oriented.resize((resized[0], oriented.height), Image.Resampling.BICUBIC).resize(resized, Image.Resampling.BICUBIC)
            if resized != target:
                padded = Image.new('RGB', target, (122, 116, 104))
                padded.paste(rendered, ((target[0] - resized[0]) // 2, (target[1] - resized[1]) // 2))
                rendered = padded
            with tempfile.TemporaryDirectory(dir=ROOT / '.cache/image-oracle') as temporary:
                temporary = Path(temporary)
                (temporary / 'input.rgb').write_bytes(rgb)
                subprocess.run([str(ROOT / '.cache/image-oracle/processor'), profile, str(oriented.width), str(oriented.height),
                                str(temporary / 'input.rgb'), str(temporary / 'output.f32'), str(temporary / 'output.json')], check=True)
                metadata = json.loads((temporary / 'output.json').read_text())
                assert (metadata['width'], metadata['height']) == target
                encoded_values = (temporary / 'output.f32').read_bytes()
                values = array.array('f')
                values.frombytes(encoded_values)
                if sys.byteorder != 'little':
                    values.byteswap()
                mean, std = [f32(v) for v in [.48145466, .4578275, .40821073]], [f32(v) for v in [.26862954, .26130258, .27577711]]
                restored = bytes(min(255, max(0, round((v * std[i % 3] + mean[i % 3]) * 255))) for i, v in enumerate(values))
                assert restored == rendered.tobytes(), (profile, path.name, 'independent processors disagree')
                resized_file = OUT / 'resized' / (profile + '-' + path.name.replace('.', '-') + '.png')
                rendered.save(resized_file)
                temporal = 2 if profile == 'qwen2vl-still-v1' else 1
                patch_digest, selected = patches(values, *target, temporal)
                indices = sorted(set([0, 1, 2, 3, len(values) // 2, len(values) - 1] + list(range(0, len(values), max(1, len(values) // 64)))))
                report['cases'].append({'id': profile + ':' + path.name, 'processor': profile,
                    'source': str(path.relative_to(ROOT)), 'source_sha256': digest(path),
                    'canonical': str(canonical.relative_to(ROOT)), 'canonical_sha256': digest(canonical),
                    'decoded_rgb_sha256': hashlib.sha256(rgb).hexdigest(), 'source_oriented_size': list(oriented.size),
                    'resized': str(resized_file.relative_to(ROOT)), 'resized_sha256': digest(resized_file),
                    'processed_size': list(target), 'resized_inner_size': list(resized),
                    'processed_rgb_sha256': hashlib.sha256(restored).hexdigest(),
                    'normalized_sha256': hashlib.sha256(encoded_values).hexdigest(),
                    'normalized_samples': [{'index': index, 'value': values[index]} for index in indices],
                    'patch_rows': target[0] // 14 * (target[1] // 14), 'patch_features': 3 * temporal * 196,
                    'image_feature_tokens': target[0] // 14 * (target[1] // 14) // (4 if temporal == 2 else 1),
                    'patches_sha256': patch_digest, 'patch_samples': selected})
            print('CAPTURE', profile, path.name, flush=True)
    (OUT / 'reference.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()

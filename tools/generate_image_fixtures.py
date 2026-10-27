#!/usr/bin/env python3
"""Create small procedural PNG inputs with no external assets or runtime codec dependency.

JPEG is a committed, byte-pinned development conversion of landscape.png; regenerate
only deliberately, recording the encoder. These are inputs, not vision oracle outputs.
"""
import hashlib
import json
from pathlib import Path
import struct
import zlib

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT/'fixtures/vlm/images'


def chunk(kind, data):
    return struct.pack('>I', len(data))+kind+data+struct.pack('>I', zlib.crc32(kind+data))


def png(width, height, channels, pixel):
    rows = b''.join(b'\0'+bytes(v for x in range(width) for v in pixel(x, y)) for y in range(height))
    return (b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2 if channels == 3 else 6, 0, 0, 0))
            +chunk(b'IDAT', zlib.compress(rows, 9))+chunk(b'IEND', b''))


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    fixtures = [
        ('landscape.png', 112, 56, 3, lambda x, y: (255,0,0) if x < 56 else (0,0,255)),
        ('swapped.png', 112, 56, 3, lambda x, y: (0,0,255) if x < 56 else (255,0,0)),
        ('portrait.png', 56, 112, 3, lambda x, y: (0,255,0) if y < 56 else (255,255,255)),
        ('minimum.png', 1, 1, 3, lambda x, y: (255,0,0)),
        ('alpha.png', 28, 28, 4, lambda x, y: (255,0,0,0 if x < 14 else 255)),
    ]
    entries = []
    for name, width, height, channels, pixel in fixtures:
        data = png(width, height, channels, pixel)
        (DIRECTORY/name).write_bytes(data)
        entries.append({'file':name, 'bytes':len(data), 'sha256':hashlib.sha256(data).hexdigest(),
                        'width':width, 'height':height, 'channels':channels, 'format':'png'})
    jpeg = DIRECTORY/'landscape.jpg'
    if jpeg.exists():
        data = jpeg.read_bytes()
        entries.append({'file':jpeg.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                        'width':112,'height':56,'channels':3,'format':'jpeg',
                        'provenance':'Committed sips JPEG conversion of landscape.png on macOS; exact encoded bytes are the fixture input'})
    document = {'schema_version':1,'source':'Original procedural color blocks; no external artwork',
                'fixtures':entries,'oracle_outputs_captured':False,
                'cases':[
                    {'id':'image-sensitive-pair','images':['landscape.png'],'paired_images':['swapped.png'],
                     'text':'Which color is on the left?','expected_semantic_answers':['red','blue']},
                    {'id':'ordered-images','images':['landscape.png','swapped.png'],
                     'text':'Compare the color on the left in the first and second images.'},
                    {'id':'portrait','images':['portrait.png'],'text':'Which color is above the white area?'},
                    {'id':'jpeg','images':['landscape.jpg'],'text':'Which color is on the left?'},
                    {'id':'minimum-resolution','images':['minimum.png'],'expectation':'qualified processor resize or explicit bounded rejection'},
                    {'id':'alpha-policy','images':['alpha.png'],'expectation':'qualified alpha conversion or explicit rejection'}],
                'pending_cases':['EXIF orientation','malformed/truncated bytes','oversized headers/decompression bombs',
                                 'per-variant maximum resolutions','interleaved multi-turn content and image markers']}
    (DIRECTORY/'manifest.json').write_text(json.dumps(document,indent=2)+'\n')


if __name__ == '__main__':
    main()

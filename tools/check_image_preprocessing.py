#!/usr/bin/env python3
"""Qualify the native image example against frozen independent pixels, without weights."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    suffix = '.exe' if os.name == 'nt' else ''
    binary = ROOT / 'target/debug/examples' / ('prepare_image' + suffix)
    if not binary.is_file():
        raise SystemExit('Build the native coheryn-engine prepare_image example first.')
    reference_path = ROOT / 'fixtures/vlm/preprocessing/reference.json'
    reference = json.loads(reference_path.read_text())
    cases = []
    for expected in reference['cases']:
        result = json.loads(subprocess.check_output([str(binary), expected['processor'], str(ROOT / expected['canonical'])], text=True, timeout=60))
        item = result['result']
        assert item['image_understanding_supported'] is False
        assert item['processed_rgb_sha256'] == expected['processed_rgb_sha256']
        plan = item['plan']
        assert [plan['width'], plan['height']] == expected['processed_size']
        assert plan['patch_rows'] == expected['patch_rows']
        stats = item['memory_while_retained']
        assert stats['reserved'] == plan['output_reserved_bytes']
        assert stats['peak_session_buffer_capacity_bytes'] <= stats['peak_reserved']
        after = result['memory_after_release']
        assert after['reserved'] == after['session_buffer_capacity_bytes'] == 0
        cases.append({'id':expected['id'],'elapsed_seconds':item['elapsed_seconds'],
                      'peak_reserved_bytes':stats['peak_reserved'],
                      'peak_engine_buffer_capacity_bytes':stats['peak_session_buffer_capacity_bytes'],
                      'retained_output_reserved_bytes':stats['reserved'],
                      'processed_rgb_sha256':item['processed_rgb_sha256'],
                      'cleanup_reserved_bytes':after['reserved']})
    report = {'schema_version':1,'stage':'M1 native image preprocessing foundation, not image understanding',
              'host':{'os':platform.platform(),'architecture':platform.machine()},
              'binary_sha256':sha(binary),'reference_sha256':sha(reference_path),
              'cases':cases,'image_to_text_supported':False,
              'measurement_limits':['Uncontrolled host/cache conditions; descriptive observations, no comparative speed claim.',
                                    'Engine vector capacities exclude codec allocations, allocator overhead and OS residency.',
                                    'No vision encoder, projector, language-model or GPU allocation is included.']}
    path = ROOT / 'fixtures/qualification/m1-preprocessing.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(f'PASS {len(cases)} native processor outputs and reservation cleanup observations')


if __name__ == '__main__':
    main()

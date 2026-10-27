#!/usr/bin/env python3
"""Check frozen tool fixture identities without weights, rendering or model execution."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((ROOT / 'fixtures/tools/manifest.json').read_text())
    reference = json.loads((ROOT / manifest['reference']).read_text())
    original = ROOT / manifest['original_template']['path']
    companion = ROOT / manifest['companion_template']['path']
    assert digest(original) == manifest['original_template']['sha256']
    assert digest(companion) == manifest['companion_template']['sha256']
    notice = manifest['companion_template']['license_notice']
    assert notice.startswith('{#-\n') and notice.endswith('-#}\n')
    assert '#}' not in notice[:-4] and '{#' not in notice[3:]
    assert companion.read_text().startswith(notice)
    body = companion.read_text()[len(notice):]
    assert hashlib.sha256(body.encode()).hexdigest() == manifest['companion_template']['reference_body_sha256'] == reference['template_sha256']
    assert digest(ROOT / manifest['reference']) == manifest['reference_sha256']
    assert digest(ROOT / manifest['negative_original']) == manifest['negative_sha256']
    fingerprint = hashlib.sha256(json.dumps(original.read_text(), ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    assert fingerprint == manifest['original_template']['metadata_fingerprint']
    expected = original.read_text().replace('{{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}}', '{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}')
    expected = expected.replace('{{- tool_call.arguments | tojson }}', '{{- tool_call.arguments }}')
    assert body == expected, 'unreviewed companion transformation'
    assert manifest['profile_identity'] == 'qwen-tools-v1:' + digest(companion)
    pin = next(m for m in json.loads((ROOT / 'fixtures/quantized/models.json').read_text())['models'] if m['sha256'] == manifest['model']['sha256'])
    assert pin == manifest['model'] == reference['model']
    assert {c['id'] for c in reference['cases']} == {'add', 'unicode-system', 'parallel-results', 'parallel-add', 'parallel-add-results'}
    for case in reference['cases']:
        assert 0 < len(case['prompt'].encode()) <= 1 << 20
        assert 0 < len(case['prompt_token_ids']) < 512
        for field in ['prompt_token_ids', 'generated_token_ids']:
            assert all(isinstance(t, int) and 0 <= t < 151936 for t in case.get(field, []))
        if case['id'] in {'add', 'parallel-add'}:
            assert case['generated_token_ids'][-1] == 151645
    assert not manifest['strict_schema'] and not manifest['automatic_execution']
    print('PASS frozen tool model/template/negative/reference identities and five bounded prompt cases.')


if __name__ == '__main__':
    main()

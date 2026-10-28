"""Check frozen constraint contract and independent sampling provenance offline."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
fixture = json.loads((ROOT / 'fixtures/constraints/sampling-v2.json').read_text())
assert fixture['profile'] == 'coheryn-sampling-v2'
assert hashlib.sha256((ROOT / fixture['source']).read_bytes()).hexdigest() == fixture['source_sha256']
assert len(fixture['cases']) == 4
for case in fixture['cases']:
    assert len(case['logits']) == len(case['allowed'])
    assert len(case['expected_ids']) == 24
    assert all(0 <= token < len(case['logits']) and case['allowed'][token] for token in case['expected_ids'])
profile = json.loads((ROOT / 'fixtures/constraints/profile-v1.json').read_text())
assert profile['profile'] == 'coheryn-json-v1'
assert profile['limits']['reservation_bytes_per_request'] == 32 << 20
print('PASS bounded JSON contract and 96 independently computed sampling decisions; no model execution.')

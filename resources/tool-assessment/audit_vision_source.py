#!/usr/bin/env python3
"""Verify that the isolated assessment copy differs only at the declared graph file."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def files(base):
    result={}
    for path in base.rglob('*'):
        relative=path.relative_to(base)
        if any(part in {'build','.git','__pycache__'} for part in relative.parts) or path.name=='.DS_Store' or not path.is_file():continue
        with path.open('rb') as f:result[str(relative)]=hashlib.file_digest(f,'sha256').hexdigest()
    return result


def fingerprint(mapping):return hashlib.sha256(json.dumps(mapping,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    original=files(ROOT/'.cache/reference');corrected=files(ROOT/'.cache/vision-corrected-source')
    assert original.keys()==corrected.keys()
    changes=[path for path in original if original[path]!=corrected[path]]
    assert changes==['tools/mtmd/models/llava.cpp'],changes
    patch=json.loads((ROOT/'resources/tool-assessment/vision-corrected-source-record.json').read_text())
    assert original[changes[0]]==patch['original_source_sha256'] and corrected[changes[0]]==patch['corrected_source_sha256']
    report={'schema_version':1,'base_revision':patch['base_revision'],'file_count':len(original),
        'original_tree_sha256':fingerprint(original),'corrected_tree_sha256':fingerprint(corrected),
        'hash_algorithm':'SHA-256 of compact JSON sorted relative-path to file-SHA256 mapping; build/git/bytecode/OS metadata excluded',
        'changed_files':changes,'changes_record_sha256':hashlib.sha256((ROOT/'resources/tool-assessment/vision-corrected-source-record.json').read_bytes()).hexdigest(),
        'source_archive_sha256':json.loads((ROOT/'fixtures/models.json').read_text())['oracle']['source_tar_sha256'],
        'archive_reverified_in_this_run':False,'archive_note':'Original extracted cache and prior archive pin retained; source tree identities are independently reproducible from that pin.'}
    (ROOT/'resources/tool-assessment/vision-source-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS complete assessment source comparison:',len(original),'files, only declared graph file changed')


if __name__=='__main__':main()

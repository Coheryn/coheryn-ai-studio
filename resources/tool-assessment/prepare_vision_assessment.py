#!/usr/bin/env python3
"""Prepare an explicitly corrected, isolated reference copy; never a native helper."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / '.cache/reference'
TARGET = ROOT / '.cache/vision-corrected-source'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert (SOURCE / 'source-revision').read_text().strip() == '2280f0fd163b03d4331d49e08460f66502e35854'
    original = SOURCE / 'tools/mtmd/models/llava.cpp'
    code = original.read_text()
    # Deliberately narrow to the selected legacy LLaVA graph, not all projector families.
    before = 'int il_last = hparams.n_layer - 1;'
    after = 'int il_last = hparams.n_layer - (proj_type == PROJECTOR_TYPE_MLP ? 0 : 1);'
    assert code.count(before) == 1
    corrected = code.replace(before, after)
    before_cls = 'inp = ggml_concat(ctx0, inp, model.class_embedding, 1);'
    after_cls = 'inp = ggml_concat(ctx0, model.class_embedding, inp, 1);'
    assert corrected.count(before_cls) == 1
    corrected = corrected.replace(before_cls, after_cls)
    if TARGET.exists():
        assert (TARGET / 'tools/mtmd/models/llava.cpp').read_text() == corrected, 'preserve unknown assessment directory'
    else:
        shutil.copytree(SOURCE, TARGET, ignore=shutil.ignore_patterns('build', '.git', '__pycache__'))
        (TARGET / 'tools/mtmd/models/llava.cpp').write_text(corrected)
    report = {'schema_version':1,'base_revision':(SOURCE/'source-revision').read_text().strip(),
        'original_source_sha256':sha(original),'corrected_source_sha256':sha(TARGET/'tools/mtmd/models/llava.cpp'),
        'changes':[{'before':before,'after':after},{'before':before_cls,'after':after_cls}],
        'scope':'research-only corrected legacy MLP reference; original files/weights/captures remain unchanged',
        'upstream_reports':['https://github.com/ggml-org/llama.cpp/issues/25817','https://github.com/ggml-org/llama.cpp/issues/25814'],
        'qualification':'Requires independent numerical cross-check; upstream reports alone are not acceptance.'}
    (ROOT/'.cache/vision-corrected-source-record.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Prepared separate corrected assessment source')


if __name__ == '__main__':
    main()

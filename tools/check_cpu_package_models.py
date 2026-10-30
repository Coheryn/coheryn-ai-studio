#!/usr/bin/env python3
"""Check extracted CPU executables against pinned local text/image/vector references."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

from verify_cpu_package import ROOT, verify, native_target
from check_cpu_release import sha, source_identity


def pinned(path, checksum, size=None):
    if not path.is_file() or (size is not None and path.stat().st_size != size) or sha(path) != checksum:
        raise ValueError('missing or changed pinned model input: ' + str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-report', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    package = json.loads(args.package_report.read_text())
    if Path(package['archive']).name != package['archive']:
        raise ValueError('package report must name a direct archive basename')
    archive = ROOT / 'target/cpu-release' / package['archive']
    manifest = json.loads((ROOT / 'fixtures/qwen35/model.json').read_text())
    checkpoint = ROOT / manifest['cache_directory']
    for item in manifest['files']:
        pinned(checkpoint / item['file'], item['sha256'], item['bytes'])
    gguf = ROOT / '.cache/models/qwen2.5-0.5b-instruct-fp16.gguf'
    gguf_reference_path = ROOT / 'fixtures/golden/qwen2.5-0.5b-instruct-fp16.json'
    gguf_reference = json.loads(gguf_reference_path.read_text())
    pinned(gguf, gguf_reference['model_sha256'])
    embedding = ROOT / '.cache/models/nomic-embed-text-v1.5.Q8_0.gguf'
    embedding_reference_path = ROOT / 'fixtures/embedding/nomic-embed-text-v1.5.Q8_0.reference.json'
    embedding_reference = json.loads(embedding_reference_path.read_text())
    pinned(embedding, embedding_reference['sha256'])
    text_reference_path = ROOT / 'fixtures/qwen35/reference-v1.json'
    image_reference_path = ROOT / 'fixtures/qwen35/vision/reference-v1.json'
    text_case = json.loads(text_reference_path.read_text())['cases'][0]
    image_case = json.loads(image_reference_path.read_text())['cases'][0]
    embedding_case = embedding_reference['cases'][0]
    image = ROOT / 'fixtures/qwen35/vision/red.png'
    records = []
    source = source_identity()
    with tempfile.TemporaryDirectory(prefix='coheryn-model-smoke-') as temporary:
        directory = Path(temporary).resolve()
        extracted = verify(archive, package['sha256'], directory)
        if extracted['target'] != native_target():
            raise ValueError('requires the declared native CPU platform')
        suffix = '.exe' if os.name == 'nt' else ''
        binary = directory / 'bin' / ('coheryn' + suffix)
        env = {k: v for k, v in os.environ.items() if k in ('SystemRoot', 'WINDIR', 'TMP', 'TEMP', 'TMPDIR')}
        env.update(PATH='', COHERYN_ENDPOINT='http://127.0.0.1:1')

        def run(name, arguments):
            process = subprocess.run([str(binary), '--json', *arguments], cwd=directory, env=env,
                                     capture_output=True, timeout=600, check=True)
            result = json.loads(process.stdout)
            records.append({'case': name, 'arguments': arguments, 'exit': process.returncode,
                            'stdout_sha256': hashlib.sha256(process.stdout).hexdigest()})
            return result

        common = ['--standalone', '--context', '256', '--max-tokens', '4', '--profile', 'reference', '--threads', '1']
        generated = run('qwen35-text', ['run', str(checkpoint), '--messages', str(ROOT / 'fixtures/qwen35/smoke-messages.json'), *common])
        if generated['token_ids'] != text_case['greedy_token_ids'] or generated['text'] != text_case['greedy_text']:
            raise ValueError('Qwen3.5 packaged text continuation mismatch')
        pictured = run('qwen35-image', ['chat', str(checkpoint), '--image', str(image), '--prompt', image_case['text'], *common])
        if pictured['token_ids'] != image_case['greedy_token_ids'] or pictured['text'] != image_case['greedy_text']:
            raise ValueError('Qwen3.5 packaged image continuation mismatch')
        old = run('qwen2-gguf', ['run', str(gguf), '--prompt', 'Hello', '--standalone', '--context', '128', '--max-tokens', '8', '--profile', 'reference'])
        if old['token_ids'] != gguf_reference['cases'][0]['generated_token_ids']:
            raise ValueError('GGUF packaged continuation mismatch')
        embedded = run('nomic-q8', ['embed', str(embedding), '--input', embedding_case['input'], '--standalone'])
        values = embedded['data'][0]['embedding']
        expected = embedding_case['normalized']
        tolerance = json.loads((ROOT / 'fixtures/embedding/reference-contract.json').read_text())['tolerances']
        if len(values) != 768 or not all(math.isfinite(v) for v in values):
            raise ValueError('packaged embedding dimension/finiteness mismatch')
        error = math.sqrt(sum((a - b) ** 2 for a, b in zip(values, expected)))
        if (error > tolerance['normalized_l2']
                or any(abs(a - b) > tolerance['absolute'] + tolerance['relative'] * abs(b) for a, b in zip(values, expected))
                or abs(math.sqrt(sum(a * a for a in values)) - 1) > tolerance['unit_norm_absolute']):
            raise ValueError('packaged embedding reference mismatch')
    if source_identity() != source:
        raise ValueError('source changed during model smoke')
    result = {'schema_version': 1, 'kind': 'cpu-package-model-smoke', 'outcome': 'pass',
              'target': native_target(), 'archive_sha256': package['sha256'],
              'source_snapshot': source, 'compatibility_sha256': sha(ROOT / 'fixtures/compatibility.json'),
              'reference_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in
                                   [gguf_reference_path, text_reference_path, image_reference_path, embedding_reference_path]},
              'cases': records, 'embedding_normalized_l2': error,
              'model_sha256': {'qwen2': gguf_reference['model_sha256'], 'nomic': embedding_reference['sha256']},
              'qwen35_manifest_sha256': sha(ROOT / 'fixtures/qwen35/model.json'),
              'image_sha256': sha(image), 'network': 'No credentials, empty PATH, unreachable configured daemon; explicit local inference only. This is not OS network isolation.',
              'cache_condition': 'uncontrolled',
              'scope': 'One current-host packaged smoke per implemented task; not the complete encoding/model/vision-pair/platform or numerical qualification matrix.'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('PASS extracted candidate: frozen GGUF/Qwen3.5 text, Qwen3.5 image and Nomic vector smoke')


if __name__ == '__main__':
    main()

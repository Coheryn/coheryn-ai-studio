#!/usr/bin/env python3
"""Check controlled source, authored ignored documents, paths and supplied artifacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import unicodedata
from binary_sections import BinaryLayoutError, data_ranges, instruction_ranges

ROOT = Path(__file__).resolve().parents[1]


def normalized(text):
    return unicodedata.normalize('NFKC', text).casefold()


def load_policy(root=ROOT):
    policy = json.loads((root / 'resources/policy/repository.json').read_text(encoding='utf-8'))
    if policy.get('schema_version') != 1 or policy.get('exception_root') != 'resources':
        raise ValueError('unsupported repository policy or exception boundary')
    return policy


def scan(root, policy, artifact=False, legal_payload=None):
    root = Path(root).resolve()
    patterns = [re.compile(p) for p in policy['patterns']]
    issues, excluded, binary_assets, encoding_scopes = [], [], [], []
    seen = set()
    files = 0
    if artifact and legal_payload is None:
        from license_notices import distribution_files
        legal_payload = distribution_files()

    def issue(path, kind, pattern=None):
        identity = (path.as_posix(), kind, pattern)
        if identity in seen:
            return
        seen.add(identity)
        issues.append({'path': path.as_posix(), 'kind': kind, 'pattern_index': pattern})

    def match(path, text, kind):
        text = normalized(text)
        for i, pattern in enumerate(patterns):
            if pattern.search(text):
                issue(path, kind, i)

    for directory, dirs, names in os.walk(root, followlinks=False):
        parent = Path(directory).relative_to(root)
        for name in list(dirs):
            path = parent / name
            full = root / path
            if full.is_symlink():
                issue(path, 'symlink-directory'); dirs.remove(name); continue
            if not artifact and path.parts[0] == 'resources':
                dirs.remove(name); continue
            match(path, str(path), 'path')
            reason = None
            if not artifact:
                reason = policy['excluded_relative_directories'].get(path.as_posix())
                reason = reason or policy['excluded_nested_directories'].get(name)
                if parent == Path('.'):
                    reason = reason or policy['excluded_root_directories'].get(name)
            if reason:
                excluded.append({'path': path.as_posix(), 'reason': reason}); dirs.remove(name)
        for name in names:
            path = parent / name
            full = root / path
            if full.is_symlink():
                issue(path, 'symlink-file'); continue
            if not artifact and path.parts[0] == 'resources':
                continue
            if artifact and path.as_posix() in legal_payload:
                try:
                    if (full.stat().st_size <= policy['maximum_text_bytes']
                            and hashlib.sha256(full.read_bytes()).hexdigest() == legal_payload[path.as_posix()]):
                        excluded.append({'path': path.as_posix(), 'reason': 'Byte-identical source-controlled legal payload; original notices preserved'})
                        files += 1
                        continue
                except OSError as error:
                    issue(path, 'unreadable-or-unclassified-' + type(error).__name__)
                    continue
            match(path, str(path), 'path')
            if not artifact and name in policy['excluded_files']:
                excluded.append({'path': path.as_posix(), 'reason': policy['excluded_files'][name]}); continue
            try:
                size = full.stat().st_size
                if full.suffix.lower() in policy['binary_asset_extensions']:
                    with full.open('rb') as f:
                        signature = f.read(12)
                    valid = ((full.suffix.lower() == '.gguf' and signature[:4] in (b'GGUF', b'FUGG'))
                             or (full.suffix.lower() == '.png' and signature.startswith(b'\x89PNG\r\n\x1a\n'))
                             or (full.suffix.lower() in ('.jpg', '.jpeg') and signature.startswith(b'\xff\xd8')))
                    if valid:
                        binary_assets.append({'path': path.as_posix(), 'bytes': size}); continue
                if size > policy['maximum_text_bytes'] and not artifact:
                    issue(path, 'oversized-unclassified-file'); continue
                if artifact:
                    # UTF-8 is checked throughout the file. Instructions are not
                    # UTF-16 text: interpreting relative operands as Unicode can
                    # normalize arithmetic bytes into unrelated short labels.
                    try:
                        instructions = instruction_ranges(full)
                    except BinaryLayoutError:
                        issue(path, 'invalid-executable-layout')
                        instructions = []
                    if instructions:
                        encoding_scopes.append({'path': path.as_posix(),
                                                'utf16_instruction_ranges': instructions,
                                                'utf8': 'complete file'})
                    with full.open('rb') as f:
                        tail = b''
                        while chunk := f.read(1 << 20):
                            data = tail + chunk
                            match(path, data.decode('utf-8', errors='replace'), 'artifact-string')
                            tail = data[-4096:]
                        for start, end in data_ranges(size, instructions):
                            f.seek(start)
                            remaining, tail = end - start, b''
                            while remaining:
                                chunk = f.read(min(1 << 20, remaining))
                                if not chunk:
                                    raise OSError('artifact changed during text inspection')
                                remaining -= len(chunk)
                                data = tail + chunk
                                for encoding in ('utf-16-le', 'utf-16-be'):
                                    for offset in (0, 1):
                                        match(path, data[offset:].decode(encoding, errors='replace'), 'artifact-string')
                                tail = data[-4096:]
                else:
                    match(path, full.read_text(encoding='utf-8'), 'text')
                files += 1
            except (OSError, UnicodeError) as error:
                issue(path, 'unreadable-or-unclassified-' + type(error).__name__)
    return {'files_checked': files, 'issues': issues, 'excluded': excluded, 'binary_assets': binary_assets,
            'encoding_scopes': encoding_scopes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, action='append', default=[])
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    policy = load_policy()
    report = {'schema_version': 1, 'policy_sha256': hashlib.sha256((ROOT / 'resources/policy/repository.json').read_bytes()).hexdigest(),
              'source': scan(ROOT, policy), 'artifacts': []}
    for path in args.artifact:
        if not path.is_dir():
            parser.error('artifact must be a directory containing the controlled release payload')
        report['artifacts'].append({'root': path.as_posix(), 'scan': scan(path, policy, artifact=True)})
    if args.report:
        target = args.report.resolve()
        if not target.is_relative_to(ROOT / 'resources'):
            parser.error('detailed findings must be written under the resource exception boundary')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    count = len(report['source']['issues']) + sum(len(r['scan']['issues']) for r in report['artifacts'])
    print(f'Repository policy: {report["source"]["files_checked"]} source files; {count} findings; {len(args.artifact)} artifact directories.')
    if count:
        print('Detailed findings require --report under the resource exception boundary.', file=sys.stderr)
    return 1 if count else 0


if __name__ == '__main__':
    raise SystemExit(main())

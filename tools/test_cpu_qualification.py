#!/usr/bin/env python3
"""Regression checks for qualification trust boundaries; no models or network required."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from cpu_qualification import references
from cpu_qualification.assets import acquire, HTTPSRedirect
from cpu_qualification.common import ROOT, compare, pinned, read, sha
from cpu_qualification.transport import frames, image_content
from run_cpu_qualification import Context


class QualificationTests(unittest.TestCase):
    def test_numeric_tail_and_shapes_are_checked(self):
        expected = [0.0] * 1024
        actual = expected.copy()
        actual[-1] = .1  # Unsampled tail must fail a full-array comparison.
        with self.assertRaises(ValueError):
            compare(actual, expected, .003, .001, .0005)
        with self.assertRaises(ValueError):
            compare([1.0], [1.0, 2.0], .003, .001)

    def test_finiteness_and_relative_l2(self):
        for value in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                compare([value], [0.], 1., 0.)
        with self.assertRaises(ValueError):
            compare([1.01], [1.], .1, 0., .001)
        self.assertEqual(compare([1.], [1.], 0., 0., 0.)['normalized_l2'], 0.)

    def test_optimized_python_cannot_disable_checks(self):
        process = subprocess.run([sys.executable, '-O', '-c',
                                  'from cpu_qualification.common import compare; compare([9.],[0.],.001,0.)'],
                                 cwd=Path(__file__).parent, capture_output=True)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn(b'numeric mismatch', process.stderr)

    def test_existing_corrupt_cache_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / 'weight'
            path.write_bytes(b'bad')
            with self.assertRaises(ValueError):
                acquire(path, {'bytes': 3, 'sha256': hashlib.sha256(b'yes').hexdigest()}, 'https://example.test/model', True)
            self.assertEqual(path.read_bytes(), b'bad')

    def test_linked_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / 'data').write_bytes(b'x')
            try:
                (root / 'link').symlink_to(root / 'data')
            except OSError:
                self.skipTest('symlink creation not permitted on this host')
            with self.assertRaises(ValueError):
                pinned(root / 'link', {'bytes': 1, 'sha256': sha(root / 'data')})

    def test_insecure_redirect_is_rejected(self):
        with self.assertRaises(ValueError):
            HTTPSRedirect().redirect_request(None, None, 302, '', {}, 'http://example.test/file')

    def bundle(self, entries, expected=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            archive = root / 'bundle.tgz'
            with tarfile.open(archive, 'w:gz') as stream:
                for name, data, kind in entries:
                    item = tarfile.TarInfo(name)
                    item.size, item.type = len(data), kind
                    stream.addfile(item, io.BytesIO(data))
            (root / 'out').mkdir()
            expected = expected or {'references/a.f32': {'bytes': 1, 'sha256': hashlib.sha256(b'a').hexdigest()}}
            with patch.object(references, 'files', return_value=expected):
                references.unpack(archive, sha(archive), root / 'out')
            return sorted(p.relative_to(root / 'out').as_posix() for p in (root / 'out').rglob('*') if p.is_file())

    def test_exact_reference_archive(self):
        self.assertEqual(self.bundle([('references/a.f32', b'a', tarfile.REGTYPE)]), ['references/a.f32'])

    def test_archive_rejects_unexpected_missing_duplicate_and_links(self):
        valid = ('references/a.f32', b'a', tarfile.REGTYPE)
        for entries in ([('../escape', b'a', tarfile.REGTYPE)], [valid, valid], [],
                        [('references/a.f32', b'a', tarfile.SYMTYPE)],
                        [('references/a.f32', b'changed', tarfile.REGTYPE)],
                        [valid, ('extra.py', b'a', tarfile.REGTYPE)]):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                self.bundle(entries)

    def test_gzip_expansion_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            archive = root / 'bomb.gz'
            archive.write_bytes(gzip.compress(b'\0' * 8192))
            (root / 'out').mkdir()
            with patch.object(references, 'files', return_value={}), patch.object(references, 'MAX_EXPANDED', 1024):
                with self.assertRaises(ValueError):
                    references.unpack(archive, sha(archive), root / 'out')

    def test_portable_archive_names(self):
        for name in ('/root', '../x', 'x/../y', 'x\\y', 'C:/x', 'x//y', 'x/CON.txt', 'x/..', 'x/a.'):
            self.assertFalse(references.safe_name(name), name)
        self.assertTrue(references.safe_name('references/english-layer.0.f32'))

    def test_sse_terminal_and_error_frames(self):
        valid = b'data: {"choices":[]}\n\ndata: [DONE]\n\n'
        self.assertEqual(frames(valid), [{'choices': []}])
        for raw in (valid + b'data: {}\n', valid + b'data: [DONE]\n', b'data: {}\n', b'data: {"error":{}}\ndata: [DONE]\n'):
            with self.assertRaises(ValueError):
                frames(raw)

    def test_failed_child_preserves_exit_and_log_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            context = Context(Path(directory).resolve())
            with self.assertRaises(ValueError):
                context.command('failure', [sys.executable, '-c', 'import sys; print("failure evidence"); sys.exit(7)'])
            record = context.commands[0]
            self.assertEqual(record['exit'], 7)
            self.assertEqual(record['stdout_sha256'], sha(context.directory / record['stdout']))
            self.assertIn('failure evidence', (context.directory / record['stdout']).read_text())

    def test_api_image_order_matches_frozen_prompt(self):
        case = read(ROOT / 'fixtures/qwen35/vision/reference-v1.json')['cases'][0]
        self.assertLess(case['prompt'].index('<|image_pad|>'), case['prompt'].index(case['text']))
        content = image_content(case['text'], ROOT / 'fixtures/qwen35/vision/red.png')
        self.assertEqual([part['type'] for part in content], ['image_url', 'text'])
        self.assertEqual(content[1]['text'], case['text'])


if __name__ == '__main__':
    unittest.main()

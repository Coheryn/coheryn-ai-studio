"""Regression checks for exception boundaries, normalization and fail-closed scanning."""
from pathlib import Path
import hashlib
import tempfile
import unittest
from check_repository_policy import load_policy, scan


class PolicyTests(unittest.TestCase):
    def test_artifact_legal_payload_requires_exact_path_bytes_and_regular_file(self):
        policy = load_policy()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'resources/licensing/notice.txt'
            path.parent.mkdir(parents=True)
            data = policy['test_terms'][0].encode()
            path.write_bytes(data)
            payload = {'resources/licensing/notice.txt': hashlib.sha256(data).hexdigest()}
            self.assertFalse(scan(root, policy, artifact=True, legal_payload=payload)['issues'])
            path.write_bytes(data + b' changed')
            self.assertTrue(scan(root, policy, artifact=True, legal_payload=payload)['issues'])
            path.unlink()
            (path.parent / 'unreviewed.txt').write_bytes(data)
            self.assertTrue(scan(root, policy, artifact=True, legal_payload=payload)['issues'])
            try:
                path.symlink_to(path.parent / 'unreviewed.txt')
            except OSError:
                return
            issues = scan(root, policy, artifact=True, legal_payload=payload)['issues']
            self.assertIn('symlink-file', {i['kind'] for i in issues})

    def test_case_paths_ignored_docs_and_true_exception(self):
        policy = load_policy()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'resources').mkdir()
            (root / 'resources/research.md').write_text(policy['test_terms'][0])
            (root / 'docs').mkdir()
            (root / '.gitignore').write_text('/docs/\n')
            (root / 'docs/notes.md').write_text(policy['test_terms'][0].swapcase())
            (root / 'resource-lookalike').mkdir()
            (root / 'resource-lookalike' / (policy['test_terms'][1] + '.md')).write_text('neutral')
            report = scan(root, policy)
            self.assertEqual(len(report['issues']), 2)
            self.assertEqual({i['kind'] for i in report['issues']}, {'path', 'text'})

    def test_symlinks_invalid_text_and_fake_binary_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'invalid.txt').write_bytes(b'\xff\x00')
            (root / 'fake.png').write_bytes(b'\xff\x00')
            try:
                (root / 'alias').symlink_to(root / 'invalid.txt')
                expected = 3
            except OSError:
                expected = 2
            self.assertEqual(len(scan(root, load_policy())['issues']), expected)

    def test_catalog_terms_unicode_and_artifact_overlap(self):
        policy = load_policy()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, term in enumerate(policy['test_terms']):
                (root / f'case-{index}.txt').write_text(term.upper())
            self.assertEqual(len(scan(root, policy)['issues']), len(policy['test_terms']))
            for path in root.iterdir():
                path.unlink()
            term = policy['test_terms'][0]
            (root / 'binary').write_bytes(b'\x00' * ((1 << 20) - 2) + term.encode() + b'\x00')
            self.assertTrue(scan(root, policy, artifact=True)['issues'])
            (root / 'binary').write_bytes(b'\x00' + term.encode('utf-16-le') + b'\x00')
            self.assertTrue(scan(root, policy, artifact=True)['issues'])
            (root / 'binary').unlink()
            (root / 'normalized.txt').write_text(''.join(chr(ord(c) + 0xfee0) if c.isascii() and c.isalnum() else c for c in term))
            self.assertTrue(scan(root, policy)['issues'])


if __name__ == '__main__':
    unittest.main()

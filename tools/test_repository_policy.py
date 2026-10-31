"""Regression checks for exception boundaries, normalization and fail-closed scanning."""
from pathlib import Path
import hashlib
import struct
import tempfile
import unittest
from check_repository_policy import load_policy, scan


class PolicyTests(unittest.TestCase):
    @staticmethod
    def executable(kind, code, data=b''):
        if kind == 'elf':
            offset = 64 + 3 * 64
            header = struct.pack('<16sHHIQQQIHHHHHH', b'\x7fELF\x02\x01\x01' + b'\0' * 9,
                                 1, 62, 1, 0, 0, 64, 0, 64, 0, 0, 64, 3, 0)
            section = lambda flags, start, length: struct.pack('<IIQQQQIIQQ', 0, 1, flags, 0, start, length, 0, 0, 1, 0)
            return header + b'\0' * 64 + section(6, offset, len(code)) + section(2, offset + len(code), len(data)) + code + data
        if kind == 'pe':
            header = bytearray(64)
            header[:2] = b'MZ'
            struct.pack_into('<I', header, 60, 64)
            offset = 64 + 24 + 240 + 80
            coff = struct.pack('<HHIIIHH', 0x8664, 2, 0, 0, 0, 240, 0)
            optional = struct.pack('<H', 0x20b) + b'\0' * 238
            section = lambda name, flags, start, length: struct.pack('<8sIIIIIIHHI', name, length, 0, length, start, 0, 0, 0, 0, flags)
            return (bytes(header) + b'PE\0\0' + coff + optional
                    + section(b'.text', 0x60000020, offset, len(code))
                    + section(b'.rdata', 0x40000040, offset + len(code), len(data)) + code + data)
        if kind == 'macho':
            command_size = 72 + 160
            offset = 32 + command_size
            header = struct.pack('<8I', 0xfeedfacf, 0x100000c, 0, 2, 1, command_size, 0, 0)
            segment = struct.pack('<II16sQQQQIIII', 0x19, command_size, b'__TEXT', 0,
                                  offset + len(code) + len(data), 0, offset + len(code) + len(data), 7, 5, 2, 0)
            section = lambda name, flags, start, length: struct.pack('<16s16sQQIIIIIIII', name, b'__TEXT', start, length, start, 0, 0, 0, flags, 0, 0, 0)
            return (header + segment + section(b'__text', 0x80000400, offset, len(code))
                    + section(b'__const', 0, offset + len(code), len(data)) + code + data)
        raise ValueError(kind)

    def test_instruction_operands_are_not_utf16_text_but_real_data_labels_still_fail(self):
        policy = load_policy()
        # Actual x86 call operand observed in a native CI executable. Decoding
        # its displacement as UTF-16 invents a normalized textual label.
        operand = bytes.fromhex('ff15963358004c8b')
        for kind in ('elf', 'pe', 'macho'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path = root / 'program'
                path.write_bytes(self.executable(kind, operand))
                report = scan(root, policy, artifact=True, legal_payload={})
                self.assertEqual(report['issues'], [])
                self.assertEqual(len(report['encoding_scopes']), 1)
                for encoding in ('utf-8', 'utf-16-le', 'utf-16-be'):
                    path.write_bytes(self.executable(kind, operand, b'\0' + policy['test_terms'][5].encode(encoding) + b'\0'))
                    self.assertTrue(scan(root, policy, artifact=True, legal_payload={})['issues'])
                path.write_bytes(self.executable(kind, b'\0' + policy['test_terms'][5].encode() + b'\0'))
                self.assertTrue(scan(root, policy, artifact=True, legal_payload={})['issues'])

    def test_unknown_or_malformed_binary_layout_never_waives_full_text_inspection(self):
        policy = load_policy()
        operand = bytes.fromhex('ff15963358004c8b')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'program'
            path.write_bytes(operand)
            self.assertTrue(scan(root, policy, artifact=True, legal_payload={})['issues'])
            for kind in ('elf', 'pe', 'macho'):
                image = self.executable(kind, operand)
                # Header truncation must not become a code-region exemption.
                path.write_bytes(image[:24])
                issues = scan(root, policy, artifact=True, legal_payload={})['issues']
                self.assertIn('invalid-executable-layout', {i['kind'] for i in issues})

    def test_section_bounds_and_hostile_counts_fail_closed(self):
        policy = load_policy()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'program'
            # Each layout has a section file offset and a bounded table count.
            for kind, offset_field, count_field in (
                    ('elf', 64 + 64 + 24, 60),
                    ('pe', 64 + 24 + 240 + 20, 64 + 6),
                    ('macho', 32 + 72 + 48, 32 + 64)):
                image = bytearray(self.executable(kind, b'abc'))
                struct.pack_into('<Q' if kind == 'elf' else '<I', image, offset_field, len(image) + 1)
                path.write_bytes(image)
                self.assertIn('invalid-executable-layout',
                              {i['kind'] for i in scan(root, policy, artifact=True, legal_payload={})['issues']})
                image = bytearray(self.executable(kind, b'abc'))
                struct.pack_into('<I' if kind == 'macho' else '<H', image, count_field, 65535)
                path.write_bytes(image)
                self.assertIn('invalid-executable-layout',
                              {i['kind'] for i in scan(root, policy, artifact=True, legal_payload={})['issues']})

    def test_normalized_utf16_data_labels_are_detected_in_every_layout(self):
        policy = load_policy()
        term = policy['test_terms'][5]
        fullwidth = ''.join(chr(ord(c) + 0xfee0) if c.isascii() and c.isalnum() else c for c in term)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for kind in ('elf', 'pe', 'macho'):
                for encoding in ('utf-16-le', 'utf-16-be'):
                    for alignment in (0, 1):
                        (root / 'program').write_bytes(self.executable(
                            kind, b'abc', b'\0' * alignment + fullwidth.encode(encoding)))
                        self.assertTrue(scan(root, policy, artifact=True, legal_payload={})['issues'])

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

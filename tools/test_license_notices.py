"""Exercise missing, corrupt and newly introduced legal evidence in isolated copies."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import license_notices as legal


class LicenseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = legal.ROOT
        cls.graph = legal.packages()
        cls.payload = legal.distribution_files()
        cls.assets = legal.read_json(legal.DIRECTORY / 'assets.json')
        cls.archive = legal.read_json(cls.source / 'resources/reference-archive/relocations.json')

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        files = set(self.payload)
        files.update(('Cargo.lock', 'fuzz/Cargo.lock', 'rust-toolchain.toml',
                      'resources/reference-archive/relocations.json'))
        files.update(a['path'] for g in self.assets['groups'] for a in g['files'])
        files.update(a['archive_path'] for a in self.archive['records'])
        for relative in files:
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.source / relative, destination)
        for name, value in [('ROOT', self.root), ('DIRECTORY', self.root / 'resources/licensing'),
                            ('packages', lambda: self.graph)]:
            mock = patch.object(legal, name, value)
            mock.start()
            self.addCleanup(mock.stop)
        self.assertEqual(legal.check(), (len(self.graph), sum(len(g['files']) for g in self.assets['groups'])))

    def test_corrupt_license_and_linked_legal_record_are_rejected(self):
        path = self.root / next(p for p in self.payload if '/texts/' in p)
        path.write_bytes(path.read_bytes() + b' changed')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            legal.check()
        path.unlink()
        try:
            path.symlink_to(self.root / 'LICENSE')
        except OSError:
            return
        with self.assertRaisesRegex(ValueError, 'linked legal record'):
            legal.check()

    def test_removed_dependency_notice_fails_coverage(self):
        path = legal.DIRECTORY / 'dependencies.json'
        data = json.loads(path.read_text())
        data['packages'].pop()
        legal.write_json(path, data)
        with self.assertRaisesRegex(ValueError, 'coverage is incomplete'):
            legal.check()

    def test_new_mirrored_asset_and_unindexed_payload_are_rejected(self):
        path = self.root / 'fixtures/embedding/source/new-asset.json'
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'require attribution'):
            legal.check()
        path.unlink()
        (legal.DIRECTORY / 'unexpected.txt').write_text('unreviewed payload')
        with self.assertRaisesRegex(ValueError, 'unindexed files'):
            legal.check()

    def test_changed_toolchain_pin_and_missing_required_notice_are_rejected(self):
        path = self.root / 'rust-toolchain.toml'
        original = path.read_bytes()
        path.write_bytes(original + b'\n')
        with self.assertRaisesRegex(ValueError, 'toolchain changes'):
            legal.check()
        path.write_bytes(original)
        (self.root / 'NOTICE').unlink()
        with self.assertRaisesRegex(ValueError, 'required distribution legal file missing'):
            legal.check()


if __name__ == '__main__':
    unittest.main()

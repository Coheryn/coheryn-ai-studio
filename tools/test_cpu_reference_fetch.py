"""Reference-asset authentication/redirect/cache checks without network or models."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.request

from cpu_qualification import assets
import fetch_cpu_references as fetcher


class ReferenceFetchTests(unittest.TestCase):
    def test_asset_ids_are_canonical_and_fixed_to_the_project_repository(self):
        self.assertEqual(fetcher.asset_url('123'), fetcher.API_BASE + '123')
        for value in ('0', '-1', '01', '1/other', '1?token=x', '1\n', '١', str(1 << 63), '1' * 100, True, None):
            with self.subTest(value=value), self.assertRaises(ValueError): fetcher.asset_url(value)

    def test_every_redirect_drops_credentials_including_same_origin(self):
        source = urllib.request.Request(fetcher.asset_url('123'), headers={
            'Authorization': 'Bearer fixture-secret', 'Cookie': 'fixture-cookie', 'Proxy-Authorization': 'fixture-proxy'})
        for host in fetcher.ASSET_HOSTS:
            redirected = fetcher.AssetRedirect().redirect_request(source, None, 302, '', {}, f'https://{host}/asset?sig=fixture')
            self.assertEqual(redirected.get_method(), 'GET')
            for header in ('Authorization', 'Cookie', 'Proxy-Authorization'):
                self.assertIsNone(redirected.get_header(header))
            self.assertEqual(redirected.get_header('Accept'), 'application/octet-stream')
        self.assertEqual(source.get_header('Authorization'), 'Bearer fixture-secret')

    def test_redirects_reject_unapproved_origins_credentials_fragments_and_methods(self):
        source = urllib.request.Request(fetcher.asset_url('123'))
        for url in ('http://api.github.com/asset', 'https://example.test/asset', 'https://api.github.com.evil.test/x',
                    'https://127.0.0.1/x', 'https://api.github.com:444/x', 'https://user@api.github.com/x',
                    'https://@api.github.com/x', 'https://api.github.com/x#fragment'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                fetcher.AssetRedirect().redirect_request(source, None, 302, '', {}, url)
        with self.assertRaises(ValueError):
            fetcher.AssetRedirect().redirect_request(urllib.request.Request(source.full_url, method='POST'), None, 302, '', {}, source.full_url)

    def test_authentication_is_only_attached_to_the_fixed_initial_api_request(self):
        client = fetcher.AuthenticatedAsset('123', 'fixture-secret')
        self.assertNotIn('fixture-secret', repr(client))
        with patch.object(fetcher.urllib.request, 'build_opener') as build:
            client.open(client.url, 20)
            request = build.return_value.open.call_args.args[0]
            self.assertEqual(request.full_url, fetcher.asset_url('123'))
            self.assertEqual(request.get_header('Authorization'), 'Bearer fixture-secret')
            self.assertEqual(request.get_header('Accept'), 'application/octet-stream')
            self.assertEqual(request.get_header('X-github-api-version'), '2026-03-10')
            build.reset_mock()
            with self.assertRaises(ValueError): client.open('https://example.test/x', 20)
            build.assert_not_called()
        for token in (None, '', 'with space', 'with\nnewline', 'é', 'x' * 4097):
            with self.subTest(token_type=type(token).__name__), patch.object(fetcher.urllib.request, 'build_opener') as build:
                with self.assertRaisesRegex(ValueError, 'authentication required'):
                    fetcher.AuthenticatedAsset('123', token).open(fetcher.asset_url('123'), 20)
                build.assert_not_called()

    def test_direct_and_redirected_streams_keep_integrity_atomicity_and_cached_offline_use(self):
        item = {'bytes': 3, 'sha256': hashlib.sha256(b'yes').hexdigest()}
        for redirected in (False, True):
            with self.subTest(redirected=redirected), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary).resolve() / 'data.tar.gz'
                response = io.BytesIO(b'yes')
                response.url = 'https://release-assets.githubusercontent.com/asset' if redirected else fetcher.asset_url('123')
                with patch.object(fetcher.urllib.request, 'build_opener') as build:
                    build.return_value.open.return_value = response
                    assets.acquire(path, item, fetcher.asset_url('123'), True,
                                   opener=fetcher.AuthenticatedAsset('123', 'fixture-secret'))
                self.assertEqual(path.read_bytes(), b'yes')
                with patch.object(fetcher.urllib.request, 'build_opener') as build:
                    assets.acquire(path, item, fetcher.asset_url('123'), True,
                                   opener=fetcher.AuthenticatedAsset('123', None))
                    build.assert_not_called()
                self.assertEqual(list(path.parent.glob('.download-*')), [])
        for body in (b'no', b'bad', b'too much'):
            with self.subTest(body=body), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary).resolve() / 'data.tar.gz'
                response = io.BytesIO(body); response.url = fetcher.asset_url('123')
                with patch.object(fetcher.urllib.request, 'build_opener') as build:
                    build.return_value.open.return_value = response
                    with self.assertRaises(ValueError):
                        assets.acquire(path, item, response.url, True, opener=fetcher.AuthenticatedAsset('123', 'fixture-secret'))
                self.assertFalse(path.exists())
                self.assertEqual(list(path.parent.glob('.download-*')), [])

    def fixture(self, root):
        inventory = root / 'inventory.json'; inventory.write_text('{}')
        descriptor = root / 'fixtures/cpu-qualification/modern-reference-bundle-v1.json'
        descriptor.parent.mkdir(parents=True)
        descriptor.write_text(json.dumps({'bytes': 3, 'sha256': hashlib.sha256(b'yes').hexdigest(),
                                          'inventory_sha256': fetcher.sha(inventory)}))
        return inventory

    def test_cli_public_mode_ignores_token_and_private_mode_keeps_it_out_of_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); inventory = self.fixture(root)
            for source in (['--url', 'https://example.test/data'], ['--github-asset-id', '123']):
                output = io.StringIO()
                with patch.object(fetcher, 'ROOT', root), patch.object(fetcher, 'INVENTORY', inventory), \
                     patch.object(fetcher, 'acquire') as acquire, \
                     patch.dict(os.environ, {'COHERYN_REFERENCE_TOKEN': 'fixture-secret'}), \
                     patch.object(sys, 'argv', ['fetch', *source, '--output', str(root / 'data')]), \
                     contextlib.redirect_stdout(output):
                    fetcher.main()
                self.assertNotIn('fixture-secret', output.getvalue())
                if source[0] == '--url': self.assertNotIn('opener', acquire.call_args.kwargs)
                else: self.assertIsInstance(acquire.call_args.kwargs['opener'], fetcher.AuthenticatedAsset)

    def test_cli_requires_exactly_one_source(self):
        for source in ([], ['--url', 'https://example.test/data', '--github-asset-id', '123']):
            with self.subTest(source=source), patch.object(fetcher, 'acquire') as acquire, \
                 patch.object(sys, 'argv', ['fetch', *source, '--output', 'unused']), patch.object(sys, 'stderr'):
                with self.assertRaises(SystemExit) as result: fetcher.main()
                self.assertEqual(result.exception.code, 2)
                acquire.assert_not_called()

    def test_cli_preserves_symlink_rejection_instead_of_resolving_it_away(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); inventory = self.fixture(root)
            target = root / 'data'; target.write_bytes(b'yes')
            link = root / 'link'
            try: link.symlink_to(target)
            except OSError: self.skipTest('symlink creation unavailable')
            with patch.object(fetcher, 'ROOT', root), patch.object(fetcher, 'INVENTORY', inventory), \
                 patch.object(sys, 'argv', ['fetch', '--url', 'https://example.test/data', '--output', str(link)]):
                with self.assertRaises(ValueError): fetcher.main()
            self.assertEqual(target.read_bytes(), b'yes')

    def test_linked_cache_parent_is_rejected_before_authentication_or_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            directory = root / 'real'; directory.mkdir()
            alias = root / 'alias'
            try: alias.symlink_to(directory, target_is_directory=True)
            except OSError: self.skipTest('symlink creation unavailable')
            item = {'bytes': 3, 'sha256': hashlib.sha256(b'yes').hexdigest()}
            with patch.object(fetcher.urllib.request, 'build_opener') as build:
                with self.assertRaisesRegex(ValueError, 'linked cache directory'):
                    assets.acquire(alias / 'data', item, fetcher.asset_url('123'), True,
                                   opener=fetcher.AuthenticatedAsset('123', 'fixture-secret'))
                build.assert_not_called()
            self.assertEqual(list(directory.iterdir()), [])


if __name__ == '__main__':
    unittest.main()

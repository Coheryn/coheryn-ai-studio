#!/usr/bin/env python3
"""Acquire an explicitly hosted copy of the exact frozen reference-data bundle."""
import argparse
import os
from pathlib import Path
import re
import ssl
import urllib.parse
import urllib.request
from cpu_qualification.assets import acquire
from cpu_qualification.common import ROOT, read, require, sha
from cpu_qualification.references import INVENTORY, MAX_ARCHIVE

API_BASE = 'https://api.github.com/repos/Coheryn/coheryn-ai-studio/releases/assets/'
ASSET_HOSTS = frozenset(('api.github.com', 'github.com', 'release-assets.githubusercontent.com',
                         'objects.githubusercontent.com', 'github-releases.githubusercontent.com'))


def asset_url(asset_id):
    require(isinstance(asset_id, str) and re.fullmatch(r'[1-9][0-9]{0,19}', asset_id) is not None
            and int(asset_id) < 1 << 63, 'invalid GitHub reference asset ID')
    return API_BASE + asset_id


class AssetRedirect(urllib.request.HTTPRedirectHandler):
    """An asset CDN never receives the API token, cookies or proxy credentials."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        require(parsed.scheme == 'https' and parsed.hostname in ASSET_HOSTS
                and parsed.port in (None, 443) and parsed.username is None and parsed.password is None
                and not parsed.fragment, 'unexpected reference asset redirect')
        require(req.get_method() == 'GET' and code in (301, 302, 303, 307, 308), 'unexpected reference asset method')
        return urllib.request.Request(newurl, headers={'User-Agent': 'Coheryn-reference-qualification',
                                                      'Accept': 'application/octet-stream'}, method='GET')


class AuthenticatedAsset:
    def __init__(self, asset_id, token):
        self.url, self._token = asset_url(asset_id), token

    def open(self, url, timeout):
        require(url == self.url, 'reference API origin mismatch')
        require(isinstance(self._token, str) and 0 < len(self._token) <= 4096
                and self._token.isascii() and all(33 <= ord(char) <= 126 for char in self._token),
                'GitHub reference asset authentication required')
        request = urllib.request.Request(url, headers={
            'Authorization': 'Bearer ' + self._token, 'Accept': 'application/octet-stream',
            'User-Agent': 'Coheryn-reference-qualification', 'X-GitHub-Api-Version': '2026-03-10'})
        context = ssl.create_default_context(cafile='/etc/ssl/cert.pem' if Path('/etc/ssl/cert.pem').exists() else None)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), AssetRedirect(),
                                            urllib.request.HTTPSHandler(context=context))
        return opener.open(request, timeout=timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--url')
    source.add_argument('--github-asset-id', help='same-repository asset, including an authorized draft; token comes from COHERYN_REFERENCE_TOKEN')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    item = read(ROOT / 'fixtures/cpu-qualification/modern-reference-bundle-v1.json')
    require(item['inventory_sha256'] == sha(INVENTORY) and item['bytes'] <= MAX_ARCHIVE, 'reference descriptor changed')
    if args.github_asset_id:
        opener = AuthenticatedAsset(args.github_asset_id, os.environ.get('COHERYN_REFERENCE_TOKEN'))
        acquire(args.output.absolute(), item, opener.url, True, opener=opener)
    else:
        acquire(args.output.absolute(), item, args.url, True)
    print('PASS pinned reference-data acquisition')


if __name__ == '__main__':
    main()

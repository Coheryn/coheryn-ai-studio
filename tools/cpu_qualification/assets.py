"""Bounded, revision-pinned acquisition; existing corrupt inputs are never repaired silently."""
import hashlib
import os
from pathlib import Path
import re
import shutil
import ssl
import tempfile
import time
import urllib.parse
import urllib.request

from .common import ROOT, pinned, read, require


class HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        require(urllib.parse.urlsplit(newurl).scheme == 'https', 'insecure download redirect')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def acquire(path, item, url, fetch):
    path = Path(path)
    require(re.fullmatch('[0-9a-f]{64}', item['sha256']) is not None
            and 0 < item['bytes'] <= 4 << 30, 'invalid acquisition identity')
    if path.exists() or path.is_symlink():
        return pinned(path, item)
    require(fetch, 'input not cached; explicitly enable --fetch: ' + str(path))
    parsed = urllib.parse.urlsplit(url)
    require(parsed.scheme == 'https' and parsed.hostname and not parsed.username
            and not parsed.password and not parsed.fragment, 'HTTPS download URL required')
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not any(p.is_symlink() for p in (path.parent, *path.parent.parents)), 'linked cache directory')
    require(shutil.disk_usage(path.parent).free >= item['bytes'] + (256 << 20), 'insufficient download space')
    context = ssl.create_default_context(cafile='/etc/ssl/cert.pem' if Path('/etc/ssl/cert.pem').exists() else None)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), HTTPSRedirect(),
                                        urllib.request.HTTPSHandler(context=context))
    descriptor, temporary = tempfile.mkstemp(prefix='.download-', dir=path.parent)
    try:
        size, digest, deadline = 0, hashlib.sha256(), time.monotonic() + 1800
        with os.fdopen(descriptor, 'wb') as output, opener.open(url, timeout=60) as response:
            require(urllib.parse.urlsplit(response.url).scheme == 'https', 'insecure final download URL')
            while chunk := response.read(1 << 20):
                size += len(chunk)
                require(size <= item['bytes'] and time.monotonic() < deadline, 'download size/deadline exceeded')
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        require(size == item['bytes'] and digest.hexdigest() == item['sha256'], 'download identity mismatch')
        # Atomic, exclusive publication on the supported native filesystems; no clobber race.
        os.link(temporary, path)
        return pinned(path, item)
    finally:
        Path(temporary).unlink(missing_ok=True)


def models(suites, fetch):
    results, identities = {}, []
    entries = []
    if 'gguf-text' in suites:
        entries.append(('gguf-text', read(ROOT / 'fixtures/models.json')['models'][0]))
    for suite, entry in zip(('embedding-f16', 'embedding-q8'), read(ROOT / 'fixtures/embedding/models.json')['models']):
        if suite in suites:
            entries.append((suite, entry))
    for suite, entry in entries:
        require(re.fullmatch('[0-9a-f]{40}', entry['revision']) is not None, 'unpinned repository revision')
        require(re.fullmatch('[A-Za-z0-9_.-]+', entry['file']) is not None, 'unsafe asset name')
        url = f"https://huggingface.co/{entry['repository']}/resolve/{entry['revision']}/{entry['file']}"
        results[suite] = acquire(ROOT / '.cache/models' / entry['file'], entry, url, fetch)
        identities.append({k: entry[k] for k in ('repository', 'revision', 'file', 'bytes', 'sha256')})
    if any(s.startswith('qwen35-') for s in suites):
        manifest = read(ROOT / 'fixtures/qwen35/model.json')
        directory = ROOT / manifest['cache_directory']
        for item in manifest['files']:
            require(re.fullmatch('[A-Za-z0-9_.-]+', item['file']) is not None, 'unsafe checkpoint name')
            url = f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}/{item['file']}"
            acquire(directory / item['file'], item, url, fetch)
            identities.append({**item, 'repository': manifest['repository'], 'revision': manifest['revision']})
        results['qwen35-text'] = results['qwen35-image'] = directory
    return results, identities

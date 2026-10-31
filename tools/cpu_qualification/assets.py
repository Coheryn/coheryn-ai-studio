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
from .profiles import PROFILES


class HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        require(urllib.parse.urlsplit(newurl).scheme == 'https', 'insecure download redirect')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def acquire(path, item, url, fetch, guard=None):
    path = Path(path)
    if guard:
        guard.check()
    require(re.fullmatch('[0-9a-f]{64}', item['sha256']) is not None
            and 0 < item['bytes'] <= 4 << 30, 'invalid acquisition identity')
    if path.exists() or path.is_symlink():
        return pinned(path, item)
    require(fetch, 'input not cached; explicitly enable --fetch: ' + str(path))
    if guard:
        guard.sample(disk_reservation=item['bytes'])
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
                if guard:
                    guard.check()
                size += len(chunk)
                require(size <= item['bytes'] and time.monotonic() < deadline, 'download size/deadline exceeded')
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        require(size == item['bytes'] and digest.hexdigest() == item['sha256'], 'download identity mismatch')
        if guard:
            guard.check()
        # Atomic, exclusive publication on the supported native filesystems; no clobber race.
        os.link(temporary, path)
        return pinned(path, item)
    finally:
        Path(temporary).unlink(missing_ok=True)


def models(suites, fetch, guard=None):
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
        results[suite] = acquire(ROOT / '.cache/models' / entry['file'], entry, url, fetch, guard)
        identities.append({k: entry[k] for k in ('repository', 'revision', 'file', 'bytes', 'sha256')})
    qwen = [s for s in suites if PROFILES[s]['family'] == 'qwen35']
    if qwen:
        manifest = read(ROOT / 'fixtures/qwen35/model.json')
        directory = ROOT / manifest['cache_directory']
        source_weights = any(PROFILES[s]['layout'] != 'gguf' for s in qwen)
        for item in manifest['files']:
            if not source_weights and item['file'] not in ('config.json', 'tokenizer.json', 'chat_template.jinja', 'preprocessor_config.json', 'tokenizer_config.json', 'LICENSE'):
                continue
            require(re.fullmatch('[A-Za-z0-9_.-]+', item['file']) is not None, 'unsafe checkpoint name')
            url = f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}/{item['file']}"
            acquire(directory / item['file'], item, url, fetch, guard)
            identities.append({**item, 'repository': manifest['repository'], 'revision': manifest['revision']})
        for name in qwen:
            if PROFILES[name]['layout'] == 'publisher': results[name] = directory
        if any(PROFILES[s]['layout'] == 'sharded' for s in qwen):
            from .shards import derive
            fixture = read(ROOT / 'fixtures/qwen35/sharded-layout-v1.json')
            target = ROOT / '.cache/qwen35-three-shard'
            if not target.exists():
                original = next(i for i in manifest['files'] if i['file'].endswith('.safetensors'))
                sidecars = [i for i in manifest['files'] if i['file'] not in (original['file'], 'model.safetensors.index.json')]
                if guard:
                    guard.sample(disk_reservation=original['bytes'] + sum(i['bytes'] for i in sidecars) + (64 << 20))
                derive(directory / original['file'], target, original, sidecars, check=guard.check if guard else None)
            for item in fixture['files']:
                pinned(target / item['file'], item)
                identities.append({**item, 'derivation': fixture['recipe'], 'source_sha256': fixture['source']['sha256']})
            for name in qwen:
                if PROFILES[name]['layout'] == 'sharded': results[name] = target
        if any(PROFILES[s]['layout'] == 'gguf' for s in qwen):
            import shutil
            gguf = read(ROOT / 'fixtures/qwen35/gguf/models-v1.json')
            require(re.fullmatch('[0-9a-f]{40}', gguf['revision']) is not None, 'unpinned GGUF revision')
            target = ROOT / gguf['cache_directory']
            needed = {PROFILES[s]['label'] for s in qwen if PROFILES[s]['layout'] == 'gguf'}
            for item in gguf['files']:
                if item['role'] != 'vision' and item['label'] not in needed: continue
                for name in (item['file'], item.get('cache_file', item['file'])):
                    require(re.fullmatch('[A-Za-z0-9_.-]+', name) is not None and name not in ('.', '..'), 'unsafe GGUF asset name')
                url = f"https://huggingface.co/{gguf['repository']}/resolve/{gguf['revision']}/{item['file']}"
                acquire(target / item.get('cache_file', item['file']), item, url, fetch, guard)
                identities.append({**item, 'repository': gguf['repository'], 'revision': gguf['revision']})
            for item in manifest['files']:
                if item['file'] not in ('config.json', 'tokenizer.json', 'chat_template.jinja', 'preprocessor_config.json', 'tokenizer_config.json', 'LICENSE'): continue
                destination = target / item['file']
                if not destination.exists():
                    if guard:
                        guard.sample(disk_reservation=item['bytes'])
                    with (directory / item['file']).open('rb') as original, destination.open('xb') as output:
                        shutil.copyfileobj(original, output, 1 << 20)
                pinned(destination, item)
            for name in qwen:
                if PROFILES[name]['layout'] == 'gguf':
                    entry = next(i for i in gguf['files'] if i['role'] == 'language' and i['label'] == PROFILES[name]['label'])
                    results[name] = target / entry['file']
    return results, identities

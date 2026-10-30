"""A bounded data-only transport for immutable independent numerical references."""
import gzip
import io
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

import license_notices
from .common import ROOT, pinned, read, require, sha

INVENTORY = ROOT / 'fixtures/cpu-qualification/references-v1.json'
MAX_ARCHIVE = 128 << 20
MAX_EXPANDED = 192 << 20


def files():
    inventory = read(INVENTORY)
    for name, digest in inventory['manifests'].items():
        require(sha(ROOT / name) == digest, 'reference manifest changed: ' + name)
    result = {item['path']: item for item in inventory['files']}
    require(len(result) == len(inventory['files']), 'duplicate reference inventory')
    for name, digest in license_notices.distribution_files().items():
        require(name not in result, 'legal/reference name collision')
        result[name] = {'path': name, 'source': name, 'bytes': (ROOT / name).stat().st_size, 'sha256': digest}
    require(len(result) <= 4096 and sum(i['bytes'] for i in result.values()) < MAX_EXPANDED, 'reference bundle exceeds limits')
    for name in result:
        require(safe_name(name), 'unsafe reference inventory name')
    return result


def safe_name(name):
    return (isinstance(name, str) and 0 < len(name) <= 240 and '\\' not in name
            and not PurePosixPath(name).is_absolute() and ':' not in name
            and all(part not in ('', '.', '..') and part == part.rstrip(' .')
                    and part.split('.')[0].upper() not in
                    {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}
                    for part in name.split('/')))


def pack(destination):
    """Copy verified data bytes, never generate expected results from the native engine."""
    expected = files()
    for item in expected.values():
        pinned(ROOT / item['source'], item)
    with Path(destination).open('xb') as output:
        with gzip.GzipFile(fileobj=output, mode='wb', filename='', mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.USTAR_FORMAT) as archive:
                for name, item in sorted(expected.items()):
                    data = (ROOT / item['source']).read_bytes()
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = len(data), 0o644, 0
                    archive.addfile(info, io.BytesIO(data))
        output.flush()
        os.fsync(output.fileno())
    require(Path(destination).stat().st_size <= MAX_ARCHIVE, 'compressed reference bundle exceeds limit')
    return {'bytes': Path(destination).stat().st_size, 'sha256': sha(destination),
            'inventory_sha256': sha(INVENTORY), 'files': len(expected), 'contains_executable': False}


def unpack(archive, checksum, destination):
    expected = files()
    require(Path(archive).stat().st_size <= MAX_ARCHIVE and sha(archive) == checksum, 'reference archive identity mismatch')
    destination = Path(destination)
    require(destination.is_dir() and not any(destination.iterdir())
            and not any(p.is_symlink() for p in (destination, *destination.parents)), 'reference extraction requires an empty unlinked directory')
    # Limit gzip expansion before the tar reader can allocate extended headers.
    with tempfile.TemporaryFile() as raw:
        total = 0
        with gzip.open(archive, 'rb') as compressed:
            while chunk := compressed.read(1 << 20):
                total += len(chunk)
                require(total <= MAX_EXPANDED, 'reference archive expansion limit exceeded')
                raw.write(chunk)
        raw.seek(0)
        with tarfile.open(fileobj=raw, mode='r:') as bundle:
            seen = set()
            for entry in bundle:
                name = entry.name
                require(safe_name(name) and name in expected and name.casefold() not in seen
                        and entry.isreg() and not entry.pax_headers and entry.size == expected[name]['bytes'],
                        'unexpected, linked, duplicate or oversized reference member: ' + name)
                seen.add(name.casefold())
                path = destination / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(entry) as source, path.open('xb') as output:
                    while chunk := source.read(1 << 20):
                        output.write(chunk)
                pinned(path, expected[name])
            require(seen == {name.casefold() for name in expected}, 'incomplete reference bundle')
    return {'archive_sha256': checksum, 'inventory_sha256': sha(INVENTORY), 'files': len(expected)}

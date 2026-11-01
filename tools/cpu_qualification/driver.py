"""Bind reusable native soak drivers to completed checks and immutable source inputs."""
import json
from dataclasses import asdict
from pathlib import Path
import re
import tomllib

from check_cpu_release import bound_file
from verify_cpu_package import executable_target
from .common import ROOT, require, sha
from .host import Policy, rejection


CHECK_COMMANDS = (
    ('cargo', 'test', '--release', '--workspace', '--example', 'cpu_soak', '--locked'),
    ('cargo', 'clippy', '--release', '--workspace', '--example', 'cpu_soak', '--tests', '--locked', '--', '-D', 'warnings'),
    ('cargo', 'build', '--release', '--workspace', '--example', 'cpu_soak', '--locked'),
)


def test_counts(path):
    totals = dict(suites=0, passed=0, failed=0, ignored=0, filtered=0)
    with path.open(encoding='utf-8', errors='replace') as stream:
        for line in stream:
            match = re.search(r'test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored; \d+ measured; (\d+) filtered out', line)
            if match:
                totals['suites'] += 1
                for key, value in zip(('passed', 'failed', 'ignored', 'filtered'), match.groups()):
                    totals[key] += int(value)
    return totals


def driver_path(target):
    require(target in ('aarch64-apple-darwin', 'x86_64-unknown-linux-gnu', 'x86_64-pc-windows-msvc'),
            'unsupported native soak target')
    suffix = '.exe' if target.endswith('msvc') else ''
    return ROOT / 'target/release/examples' / ('cpu_soak' + suffix)


def driver_identity(target):
    path = driver_path(target)
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            'missing or linked native soak driver')
    require(0 < path.stat().st_size <= 64 << 20, 'native soak driver exceeds size limit')
    executable_target(path, target)
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def prepared_driver(path, source, target):
    """Require exact current-source checks, complete logs and the unchanged executable.

    This checks local evidence consistency, not authorship or release approval.
    No compiler is invoked and no replacement or fallback build is permitted.
    """
    path = Path(path).absolute()
    require(path.is_relative_to(ROOT) and '..' not in path.parts and path.is_file()
            and not any(p.is_symlink() for p in (path, *path.parents)),
            'prepared driver record must be an unlinked workspace file')
    with path.open('rb') as stream:
        data = stream.read((2 << 20) + 1)
    require(len(data) <= 2 << 20, 'prepared driver record exceeds size limit')
    record = json.loads(data)
    require(isinstance(record, dict) and record.get('schema_version') == 1
            and record.get('kind') == 'cpu-soak-checks' and record.get('outcome') == 'pass'
            and record.get('target') == target and record.get('source_snapshot') == source
            and record.get('source_unchanged') is True and record.get('owned_cleanup_confirmed') is True,
            'prepared driver requires passing checks for the current source and native target')
    guard = record.get('host_guard')
    policy = Policy()
    require(isinstance(guard, dict) and 'stop_reason' in guard and guard['stop_reason'] is None
            and guard.get('stop_errors') == [] and guard.get('policy') == asdict(policy)
            and type(guard.get('samples')) is int and 0 < guard['samples'] <= policy.max_samples
            and guard.get('log') == 'host-health.jsonl',
            'prepared driver checks did not retire a healthy guard')
    host_log = bound_file(f'.cache/cpu-soak-checks/{path.stem}/host-health.jsonl', guard['log_sha256'])
    count, last = 0, None
    with host_log.open('rb') as stream:
        while line := stream.readline((64 << 10) + 1):
            require(len(line) <= 64 << 10 and line.endswith(b'\n'), 'invalid prepared host observation')
            row = json.loads(line)
            count += 1
            require(count <= policy.max_samples and row.get('sample') == count
                    and 'rejection' in row and row['rejection'] is None
                    and rejection(row['observation'], policy) is None,
                    'prepared host observations rejected or incomplete')
            last = row['observation']
    require(count == guard['samples'] and last == guard.get('last'),
            'prepared host observation summary changed')
    toolchain = record.get('toolchain')
    require(isinstance(toolchain, dict) and set(toolchain) == {'cargo', 'rustc'},
            'prepared driver compiler identity is incomplete')
    version = tomllib.loads((ROOT / 'rust-toolchain.toml').read_text())['toolchain']['channel']
    for name in ('cargo', 'rustc'):
        item = toolchain[name]
        log = bound_file(item['log'], item['log_sha256'])
        require(log.stat().st_size <= 64 << 10 and isinstance(item.get('output'), str)
                and log.read_text(encoding='utf-8') == item['output']
                and item['output'].startswith(f'{name} {version} '),
                'prepared driver compiler identity changed or differs from the pinned toolchain')
    commands = record.get('commands')
    require(isinstance(commands, list) and len(commands) == len(CHECK_COMMANDS),
            'prepared driver check command inventory is incomplete')
    for item, expected in zip(commands, CHECK_COMMANDS):
        require(isinstance(item, dict) and item.get('command') == list(expected)
                and item.get('exit') == 0 and item.get('validated') is True,
                'prepared driver check command failed or changed')
        log = bound_file(item['log'], item['log_sha256'])
        if expected[1] == 'test':
            counts = item.get('tests')
            require(isinstance(counts, dict) and counts == test_counts(log) and type(counts.get('passed')) is int
                    and counts['passed'] > 0 and type(counts.get('suites')) is int and counts['suites'] > 0
                    and all(type(counts.get(k)) is int and counts[k] == 0
                            for k in ('failed', 'filtered', 'ignored')),
                    'prepared driver tests are empty, filtered, ignored or failed')
    identity = driver_identity(target)
    require(record.get('binary') == identity, 'native soak driver changed after its checks')
    return {'record': path.relative_to(ROOT).as_posix(), 'record_sha256': sha(path),
            'binary': identity, 'mode': 'verified-record'}

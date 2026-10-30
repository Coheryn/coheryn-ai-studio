#!/usr/bin/env python3
"""Execute native CPU build checks and retain source-bound commands and complete logs."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from check_cpu_release import ROOT, COMMANDS, source_identity, sha
from verify_cpu_package import native_target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    target = native_target()
    if target is None:
        raise ValueError('unsupported native platform')
    source = source_identity()
    directory = ROOT / '.cache/cpu-checks' / args.report.stem
    directory.mkdir(parents=True, exist_ok=True)
    cargo = str(ROOT / 'tools/cargo') if os.name != 'nt' and (ROOT / '.cache/cargo/bin/cargo').exists() else 'cargo'
    environment = {**os.environ, 'CARGO_BUILD_JOBS': '2', 'CARGO_INCREMENTAL': '0',
                   'CARGO_PROFILE_DEV_DEBUG': '0', 'CARGO_PROFILE_TEST_DEBUG': '0'}
    records = []
    for number, command in enumerate(COMMANDS):
        executable = cargo if command[0] == 'cargo' else sys.executable
        log = directory / f'{number}.log'
        with log.open('xb') as output:
            process = subprocess.run([executable, *command[1:]], cwd=ROOT, env=environment,
                                     stdout=output, stderr=subprocess.STDOUT)
        valid = process.returncode == 0
        if command[:2] == ('cargo', 'test'):
            summaries = re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored; (\d+) measured; (\d+) filtered out', log.read_text(errors='replace'))
            valid = valid and bool(summaries) and sum(int(s[0]) for s in summaries) > 0 and all(int(s[1]) == 0 and int(s[4]) == 0 for s in summaries)
        records.append({'command': list(command), 'exit': process.returncode,
                        'log': log.relative_to(ROOT).as_posix(), 'log_sha256': sha(log), 'validated': valid})
        print(('PASS' if valid else 'FAIL') + ' ' + ' '.join(command), flush=True)
        if not valid:
            break
    unchanged = source_identity() == source
    local_rust = ROOT / '.cache/cargo/bin/rustc'
    compiler = str(local_rust) if local_rust.exists() else shutil.which('rustc')
    compiler_env = {**environment, 'RUSTUP_HOME': str(ROOT / '.cache/rustup')} if local_rust.exists() else environment
    result = {'schema_version': 1, 'kind': 'cpu-native-checks', 'target': target,
              'host': {'system': platform.system(), 'release': platform.release(), 'machine': platform.machine()},
              'rust': subprocess.check_output([cargo, '--version'], cwd=ROOT, text=True).strip(),
              'compiler': subprocess.check_output([compiler, '-vV'], cwd=ROOT, env=compiler_env, text=True),
              'source_snapshot': source, 'compatibility_sha256': sha(ROOT / 'fixtures/compatibility.json'),
              'commands': records, 'source_unchanged': unchanged,
              'outcome': 'pass' if unchanged and len(records) == len(COMMANDS) and all(r['validated'] for r in records) else 'fail',
              'scope': 'Native automated checks only; no downloaded-model, security-review or production qualification claim.'}
    suffix = '.exe' if os.name == 'nt' else ''
    result['binaries'] = {name + suffix: {'sha256': sha(ROOT / 'target/release' / (name + suffix)),
                                          'bytes': (ROOT / 'target/release' / (name + suffix)).stat().st_size}
                          for name in ('coheryn', 'coheryn-bench')} if result['outcome'] == 'pass' else {}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    raise SystemExit(result['outcome'] != 'pass')


if __name__ == '__main__':
    main()

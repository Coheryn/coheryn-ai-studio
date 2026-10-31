#!/usr/bin/env python3
"""Qualify selected pinned real models on this native host; never promote production status."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import time
import traceback

from check_cpu_release import source_identity
from verify_cpu_package import native_target, verify
from cpu_qualification import assets, host, references, suites, transport
from cpu_qualification.common import ROOT, clean_environment, read, require, sha, write
from cpu_qualification.profiles import PROFILES, GROUPS, selected as group_suites

SUITES = tuple(PROFILES)


def hardware():
    result = {'cpu': platform.processor() or 'unavailable', 'cpu_count': os.cpu_count(),
              'physical_memory_bytes': 'unavailable'}
    if platform.system() == 'Darwin':
        for key, name in [('machdep.cpu.brand_string', 'cpu'), ('hw.model', 'hardware_model'), ('hw.memsize', 'physical_memory_bytes')]:
            try:
                value = subprocess.check_output(['/usr/sbin/sysctl', '-n', key], text=True, timeout=10).strip()
                result[name] = int(value) if name == 'physical_memory_bytes' else value
            except (OSError, ValueError, subprocess.SubprocessError):
                result[name] = 'unavailable'
    elif platform.system() == 'Linux':
        try:
            result['cpu'] = next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                                 if line.startswith('model name'))
            result['physical_memory_bytes'] = os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
        except (OSError, ValueError, StopIteration):
            pass
    elif os.name == 'nt':
        result['cpu'] = os.environ.get('PROCESSOR_IDENTIFIER', result['cpu'])
    return result


class Context:
    def __init__(self, directory, guard=None):
        self.directory = directory
        self.commands = []
        self.suffix = '.exe' if os.name == 'nt' else ''
        self.binary = directory / 'package/bin' / ('coheryn' + self.suffix)
        self.references = directory / 'reference-data'
        self.guard = guard
        self.on_command = None

    @staticmethod
    def json_lines(path):
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

    def command(self, name, arguments, *, json_lines=True, build=False):
        stdout, stderr = self.directory / (name + '.stdout'), self.directory / (name + '.stderr')
        started = time.monotonic()
        with stdout.open('xb') as output, stderr.open('xb') as errors:
            if self.guard:
                self.guard.sample()
            environment = {**os.environ, 'CARGO_BUILD_JOBS': '2', 'CARGO_INCREMENTAL': '0'} if build else clean_environment()
            process = subprocess.Popen(arguments, cwd=ROOT if build else self.directory, env=environment,
                                       stdin=subprocess.DEVNULL, stdout=output, stderr=errors, **host.child_options())
            try:
                with self.guard.owned(process) if self.guard else nullcontext():
                    while process.poll() is None:
                        if self.guard:
                            self.guard.check()
                        require(time.monotonic() - started < (1800 if build else 1200), 'subprocess deadline exceeded: ' + name)
                        require(stdout.stat().st_size + stderr.stat().st_size <= 64 << 20, 'subprocess output limit exceeded: ' + name)
                        try:
                            process.wait(timeout=.1)
                        except subprocess.TimeoutExpired:
                            pass
            finally:
                if process.poll() is None:
                    self.guard.stop(process) if self.guard else host.stop_owned(process)
                self.commands.append({'case': name, 'arguments': arguments, 'exit': process.returncode,
                                      'elapsed_seconds': time.monotonic() - started,
                                      'stdout': stdout.name, 'stdout_sha256': sha(stdout), 'stderr': stderr.name, 'stderr_sha256': sha(stderr)})
                if self.on_command:
                    self.on_command()
        require(process.returncode == 0, 'native command failed: ' + name + '; see ' + str(stderr))
        require(stdout.stat().st_size + stderr.stat().st_size <= 64 << 20, 'completed subprocess output limit exceeded: ' + name)
        print('PASS command ' + name, flush=True)
        if build:
            return None
        return self.json_lines(stdout) if json_lines else read(stdout)

    def cli(self, name, arguments, endpoint=None, vault=None):
        common = ['--json']
        if endpoint:
            common += ['--endpoint', 'http://' + endpoint, '--data-dir', str(vault)]
        return self.command(name, [str(self.binary), *common, *arguments], json_lines=False)

    def probe(self, binary, name, model, request, extra=None):
        path = self.directory / (name + '.request.json')
        write(path, request)
        result = self.command(name, [str(ROOT / 'target/release/examples' / (binary + self.suffix)), str(model), str(path), *(extra or [])],
                              json_lines=binary == 'embedding_probe')
        return result if binary == 'embedding_probe' else [result]


def checkpoint(path, report):
    """Atomically preserve progress; an abrupt restart cannot publish partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.' + path.name, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', action='append', choices=SUITES)
    parser.add_argument('--group', choices=('all', *GROUPS), default='all')
    parser.add_argument('--fetch', action='store_true', help='allow bounded pinned model downloads')
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--archive-sha256', required=True)
    parser.add_argument('--native-checks', required=True, type=Path)
    parser.add_argument('--reference-bundle', type=Path)
    parser.add_argument('--reference-sha256')
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    if os.name != 'nt':
        def interrupted(_signal, _frame):
            raise KeyboardInterrupt('qualification interrupted')
        signal.signal(signal.SIGTERM, interrupted)
    require(not args.report.exists(), 'report already exists')
    selected = args.suite or group_suites(args.group)
    require(len(set(selected)) == len(selected), 'duplicate qualification suite')
    directory = ROOT / '.cache/cpu-qualification' / args.report.stem
    directory.mkdir(parents=True, exist_ok=False)
    source = source_identity()
    context = Context(directory)
    guard = None
    report = {'schema_version': 1, 'kind': 'cpu-real-model-qualification', 'outcome': 'running', 'target': native_target(),
              'host': {'system': platform.system(), 'release': platform.release(), 'machine': platform.machine(), **hardware()},
              'source_snapshot': source, 'source_unchanged': False, 'selected_suites': selected, 'production_ready': False,
              'cache_condition': 'uncontrolled', 'external_executor_invoked': False, 'results': {},
              'limitations': ['Selected fixtures only; other sizes/encodings, publisher-sharded checkpoints, legacy vision paths and broad image semantics remain separate qualification.',
                              'No GPU, cross-platform extrapolation, publisher-framework Qwen3.5 parity, TLS, soak, fuzz or durable-session qualification.',
                              'Internal reservation cleanup checked; RSS, private memory, faults and I/O unavailable in this runner.']}
    report['commands'] = context.commands
    report['artifacts'] = directory.relative_to(ROOT).as_posix()
    def save():
        if guard:
            report['host_guard'] = guard.report()
            if guard.stream.closed:
                report['host_guard']['log_sha256'] = sha(guard.log)
        checkpoint(args.report, report)
    context.on_command = save
    save()
    try:
        guard = host.Guard(directory)
        context.guard = guard
        guard.start()
        native = read(args.native_checks)
        require(report['target'] and native['target'] == report['target'] and native['outcome'] == 'pass'
                and native['source_unchanged'] and native['source_snapshot'] == source, 'current native build evidence required')
        report.update(compiler=native['compiler'], native_checks_sha256=sha(args.native_checks), cargo_lock_sha256=sha(ROOT / 'Cargo.lock'),
                      fuzz_lock_sha256=sha(ROOT / 'fuzz/Cargo.lock'), compatibility_sha256=sha(ROOT / 'fixtures/compatibility.json'))
        package = directory / 'package'
        package.mkdir()
        manifest = verify(args.archive.resolve(), args.archive_sha256, package)
        require(manifest['target'] == report['target'] and manifest['native_check_record_sha256'] == sha(args.native_checks)
                and manifest['source_snapshot_sha256'] == hashlib.sha256(json.dumps(source, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                'package source/native evidence mismatch')
        report.update(archive_sha256=args.archive_sha256, binary_sha256=sha(context.binary))
        if any(PROFILES[s]['family'] != 'embedding' for s in selected):
            require(args.reference_bundle and args.reference_sha256, 'numerical suites require the frozen data bundle and its SHA-256')
            context.references.mkdir()
            report['references'] = references.unpack(args.reference_bundle.resolve(), args.reference_sha256, context.references)
        models, identities = assets.models(selected, args.fetch, guard)
        report['models'] = identities
        guard.sample()
        probes = sorted(set('embedding_probe' if s.startswith('embedding-') else 'qwen35_image_probe' if PROFILES[s]['image'] else 'qwen35_probe'
                            for s in selected if s != 'gguf-text'))
        if probes:
            cargo = str(ROOT / 'tools/cargo') if os.name != 'nt' else 'cargo'
            command = [cargo, 'build', '--release', '--locked', '-p', 'coheryn-engine']
            for probe in probes:
                command += ['--example', probe]
            context.command('build-probes', command, build=True)
        report['probes'] = {p: sha(ROOT / 'target/release/examples' / (p + context.suffix)) for p in probes}
        for suite in selected:
            guard.sample()
            print('START ' + suite, flush=True)
            numerical = suites.gguf(context, models[suite]) if suite == 'gguf-text' else suites.embedding(context, suite, models[suite]) if suite.startswith('embedding-') else suites.qwen35(context, suite, models[suite])
            # Preserve completed numerical evidence even if a later transport check fails.
            report['results'][suite] = {'numerical_and_packaged_cli': numerical}
            save()
            report['results'][suite]['http'] = transport.run(context, suite, models[suite])
            save()
            print('PASS ' + suite, flush=True)
        require(source_identity() == source, 'source changed during qualification')
        report['outcome'] = 'pass'
    except (Exception, KeyboardInterrupt) as error:
        report['outcome'] = 'fail'
        report['error'] = str(error)
        traceback.print_exc()
    finally:
        if guard:
            try:
                guard.close()
            except Exception as error:
                report['outcome'] = 'fail'
                report['observer_close_error'] = str(error)
            report['host_guard'] = guard.report()
        report['source_unchanged'] = source_identity() == source
        report['commands'] = context.commands
        # The private vault is destroyed by transport before report/artifact publication.
        report['artifacts'] = directory.relative_to(ROOT).as_posix()
        save()
    raise SystemExit(report['outcome'] != 'pass')


if __name__ == '__main__':
    main()

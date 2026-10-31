#!/usr/bin/env python3
"""Resume current-source children after a host cutoff; retain no interrupted child."""
from collections import Counter
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from check_cpu_release import source_identity
from cpu_qualification import assets, references, suites, transport
from cpu_qualification.common import pinned, read, require, sha
from cpu_qualification.host import Guard
from run_cpu_qualification import Context, checkpoint, hardware
from verify_cpu_package import verify

ORIGINAL = ROOT / 'fixtures/qualification/cpu-modern-qwen35-text-v4.json'
REPORT = ROOT / 'fixtures/qualification/cpu-modern-qwen35-text-v5.json'
DIRECTORY = ROOT / '.cache/cpu-qualification/cpu-modern-qwen35-text-v5'
RETAINED = {'qwen35-text-english-1', 'qwen35-text-english-2', 'qwen35-text-cli-english'}


class CaptureDirectory(type(Path())):
    def mkdir(self, mode=0o777, parents=False, exist_ok=False):
        if self.parent == DIRECTORY and self.name in RETAINED and self.is_dir():
            return
        return super().mkdir(mode, parents, exist_ok)


class ResumeContext(Context):
    def command(self, name, arguments, *, json_lines=True, build=False):
        if name not in RETAINED:
            return super().command(name, arguments, json_lines=json_lines, build=build)
        self.guard.check()
        original = self.original_commands[name]
        expected = [a.replace(str(self.original_directory), str(DIRECTORY)) for a in original['arguments']]
        require(arguments == expected, 'retained command arguments differ')
        request = self.original_directory / (name + '.request.json')
        if request.exists():
            normalized = json.loads(json.dumps(read(request)).replace(str(self.original_directory), str(DIRECTORY)))
            require(read(DIRECTORY / request.name) == normalized, 'retained request settings differ')
        for field in ('stdout', 'stderr'):
            require(sha(self.directory / original[field]) == original[field + '_sha256'], 'retained command output differs')
        record = {**original, 'execution_scope': 'retained successful child; not reexecuted',
                  'source_record_sha256': sha(ORIGINAL)}
        self.commands.append(record)
        if self.on_command:
            self.on_command()
        output = self.directory / original['stdout']
        return self.json_lines(output) if json_lines else read(output)


def main():
    require(not REPORT.exists(), 'resume report exists')
    source = source_identity()
    native_path = ROOT / 'fixtures/qualification/cpu-modern-native-v4.json'
    native = read(native_path)
    old = read(ORIGINAL)
    require(native['source_snapshot'] == source and native['outcome'] == 'pass' and native['source_unchanged'], 'native evidence is stale')
    require(old['source_snapshot'] == source and old['source_unchanged'] and old['outcome'] == 'fail'
            and old['host_guard']['stop_reason'] == 'macOS compressed memory exceeds cutoff', 'unexpected interrupted source')
    require(old['binary_sha256'] == native['binaries']['coheryn']['sha256']
            and sha(ROOT / 'target/release/examples/qwen35_probe') == old['probes']['qwen35_probe'], 'executable/probe differs')
    DIRECTORY.mkdir(exist_ok=False)
    guard = Guard(DIRECTORY)
    context = ResumeContext(CaptureDirectory(DIRECTORY), guard)
    context.original_directory = ROOT / old['artifacts']
    context.original_commands = {c['case']: c for c in old['commands'] if c['case'] in RETAINED and c['exit'] == 0}
    require(set(context.original_commands) == RETAINED, 'missing completed child evidence')
    report = {'schema_version': 1, 'kind': 'cpu-real-model-qualification', 'outcome': 'running',
              'target': native['target'], 'host': old['host'], 'source_snapshot': source, 'source_unchanged': False,
              'selected_suites': ['qwen35-text'], 'production_ready': False, 'cache_condition': 'uncontrolled',
              'external_executor_invoked': False, 'results': {}, 'commands': context.commands,
              'artifacts': DIRECTORY.relative_to(ROOT).as_posix(), 'compiler': native['compiler'],
              'native_checks_sha256': sha(native_path), 'binary_sha256': old['binary_sha256'],
              'probes': old['probes'], 'cargo_lock_sha256': sha(ROOT / 'Cargo.lock'),
              'fuzz_lock_sha256': sha(ROOT / 'fuzz/Cargo.lock'), 'compatibility_sha256': sha(ROOT / 'fixtures/compatibility.json'),
              'resume': {'source_record': ORIGINAL.relative_to(ROOT).as_posix(), 'source_record_sha256': sha(ORIGINAL),
                         'source_outcome': 'fail', 'retained_children': sorted(RETAINED),
                         'scope': 'Revalidate completed English outputs; rerun interrupted Unicode children and all HTTP checks. Original host-cutoff campaign remains failed.'},
              'limitations': old['limitations']}
    def save():
        report['host_guard'] = guard.report()
        if guard.stream.closed:
            report['host_guard']['log_sha256'] = sha(guard.log)
        checkpoint(REPORT, report)
    context.on_command = save
    save()
    try:
        guard.start()
        package = read(ROOT / 'fixtures/qualification/cpu-modern-package-v4.json')
        (DIRECTORY / 'package').mkdir()
        manifest = verify(ROOT / 'target/cpu-modern-release-v4' / package['archive'], package['sha256'], DIRECTORY / 'package')
        require(manifest['native_check_record_sha256'] == sha(native_path), 'package/native binding differs')
        report['archive_sha256'] = package['sha256']
        context.references = context.original_directory / 'reference-data'
        for item in references.files().values():
            pinned(context.references / item['path'], item)
        report['references'] = old['references']
        models, identities = assets.models(['qwen35-text'], False, guard)
        report['models'] = identities
        for name, command in context.original_commands.items():
            for field in ('stdout', 'stderr'):
                original = context.original_directory / command[field]
                require(sha(original) == command[field + '_sha256'], 'original output changed')
                os.link(original, DIRECTORY / original.name)
            original_trace = context.original_directory / name
            if original_trace.is_dir():
                (DIRECTORY / name).mkdir()
                for path in original_trace.iterdir():
                    require(path.is_file() and not path.is_symlink(), 'unexpected original capture member')
                    os.link(path, DIRECTORY / name / path.name)
        print('REVALIDATE English; RESUME Unicode and HTTP', flush=True)
        numeric = suites.qwen35(context, 'qwen35-text', models['qwen35-text'])
        report['results']['qwen35-text'] = {'numerical_and_packaged_cli': numeric}
        save()
        report['results']['qwen35-text']['http'] = transport.run(context, 'qwen35-text', models['qwen35-text'])
        require(source_identity() == source, 'source changed during resume')
        report['outcome'] = 'pass'
    except BaseException as error:
        report['outcome'], report['error'] = 'fail', str(error)
        traceback.print_exc()
    finally:
        guard.close()
        report['source_unchanged'] = source_identity() == source
        save()
    print(report['outcome'], flush=True)
    raise SystemExit(report['outcome'] != 'pass')


if __name__ == '__main__':
    main()

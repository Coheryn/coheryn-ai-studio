#!/usr/bin/env python3
"""Revalidate interrupted local captures; rerun transport, never invent process exits.

This is an evidence-recovery procedure for the preserved v2 artifacts, not a
production runner or a claim that the interrupted parent process succeeded.
The original numerical validators and frozen tolerances are used unchanged.
"""
import json
from pathlib import Path
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import run_cpu_qualification as runner
from check_cpu_release import source_identity
from cpu_qualification import assets, references, suites, transport
from cpu_qualification.common import pinned, read, require, sha, write
from verify_cpu_package import verify

ARTIFACTS = ROOT / '.cache/cpu-qualification/cpu-modern-models-v2'
REPORT = ROOT / 'fixtures/qualification/cpu-modern-models-v2-recovered.json'
SELECTED = ['qwen35-gguf-f16-image', 'qwen35-gguf-q4_k_m-image',
            'qwen35-sharded-image', 'qwen35-image', 'qwen35-gguf-f16-text']


class ExistingDirectory(type(Path())):
    def mkdir(self, mode=0o777, parents=False, exist_ok=False):
        require(self.is_dir() and not self.is_symlink(), 'missing capture directory: ' + str(self))


def existing_json(path, value):
    require(read(path) == value, 'preserved request differs: ' + str(path))


class CaptureContext(runner.Context):
    def command(self, name, arguments, *, json_lines=True, build=False):
        require(not build, 'recovery must not build or execute cached commands')
        output = self.directory / (name + '.stdout')
        errors = self.directory / (name + '.stderr')
        require(output.is_file() and errors.is_file() and errors.stat().st_size == 0,
                'missing or diagnostic-bearing capture: ' + name)
        result = self.json_lines(output) if json_lines else read(output)
        self.commands.append({'case': name, 'exit': None, 'original_exit_unavailable': True,
                              'revalidated_output': True, 'stdout_sha256': sha(output),
                              'stderr_sha256': sha(errors)})
        return result


def checkpoint(report):
    temporary = REPORT.with_suffix('.pending')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(REPORT)


def main():
    require(not REPORT.exists(), 'recovery record already exists')
    native = read(ROOT / 'fixtures/qualification/cpu-modern-native-v2.json')
    source = source_identity()
    require(native['source_snapshot'] == source and native['outcome'] == 'pass'
            and native['source_unchanged'], 'current native source evidence required')
    package = read(ROOT / 'fixtures/qualification/cpu-modern-package-v2.json')
    with tempfile.TemporaryDirectory(prefix='cpu-package-recheck-', dir=ROOT / '.cache') as temporary:
        manifest = verify(ROOT / 'target/cpu-modern-release-v2' / package['archive'], package['sha256'], Path(temporary))
    require(manifest['native_check_record_sha256'] == sha(ROOT / 'fixtures/qualification/cpu-modern-native-v2.json'),
            'package/native binding differs')
    context = CaptureContext(ExistingDirectory(ARTIFACTS))
    require(sha(context.binary) == native['binaries']['coheryn']['sha256'], 'extracted executable differs')
    for item in references.files().values():
        pinned(context.references / item['path'], item)
    models, identities = assets.models(SELECTED, False)
    # Require every frozen request to remain byte-identical; do not regenerate it.
    runner.write = existing_json
    suites.write = existing_json
    report = {'schema_version': 1, 'kind': 'cpu-real-model-recovery', 'outcome': 'running',
              'target': native['target'], 'source_snapshot': source, 'source_unchanged': False,
              'native_checks_sha256': sha(ROOT / 'fixtures/qualification/cpu-modern-native-v2.json'),
              'archive_sha256': package['sha256'], 'binary_sha256': sha(context.binary),
              'procedure_sha256': sha(Path(__file__)), 'selected_suites': SELECTED,
              'models': identities, 'production_ready': False, 'cache_condition': 'uncontrolled',
              'external_executor_invoked': False, 'results': {},
              'limitations': ['Original parent process was interrupted without a final report; its command exit codes are unavailable.',
                              'Preserved numerical/CLI outputs are revalidated against frozen references; all HTTP checks run again.',
                              'Local ARM64 host only; publisher-framework parity, broad image semantics, peak RSS and production qualification remain open.']}
    checkpoint(report)
    try:
        for suite in SELECTED:
            print('REVALIDATE ' + suite, flush=True)
            numerical = suites.qwen35(context, suite, models[suite])
            report['results'][suite] = {'numerical_and_packaged_cli': numerical}
            checkpoint(report)
            directory = ROOT / '.cache/modern-validation-resume' / suite
            directory.mkdir(exist_ok=False)
            live = runner.Context(directory)
            live.binary = context.binary
            print('RERUN transport ' + suite, flush=True)
            report['results'][suite]['http'] = transport.run(live, suite, models[suite])
            report['results'][suite]['live_commands'] = live.commands
            require(source_identity() == source, 'source changed during recovery')
            checkpoint(report)
            print('PASS ' + suite, flush=True)
        report['outcome'] = 'pass'
    except (Exception, KeyboardInterrupt) as error:
        report['outcome'] = 'fail'
        report['error'] = str(error)
        traceback.print_exc()
    finally:
        report['source_unchanged'] = source_identity() == source
        report['revalidated_commands'] = context.commands
        checkpoint(report)
    raise SystemExit(report['outcome'] != 'pass')


if __name__ == '__main__':
    main()

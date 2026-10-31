#!/usr/bin/env python3
"""Revalidate interrupted local captures and complete saved HTTP evidence.

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
REPORT = ROOT / 'fixtures/qualification/cpu-modern-models-v3-recovered.json'
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


def saved_image_http(context, suite, model):
    """Check actual daemon status logs, immutable import bytes and saved replies.

    The unchanged original writer checked status/content type before preserving
    responses. Retain the distinction between response validation and a new call.
    """
    from cpu_qualification.profiles import PROFILES
    log = context.directory / (suite + '-daemon.jsonl')
    events = context.json_lines(log)
    calls = [e for e in events if e.get('event') == 'http_request']
    require([(e['method'], e['path'], e['status']) for e in calls] == [
        ('POST', '/api/v1/models/import-checkpoint', 200),
        ('POST', '/api/v1/models/load', 200),
        ('POST', '/v1/chat/completions', 200),
        ('POST', '/v1/chat/completions', 200),
        ('POST', '/api/v1/models/unload', 200)], 'incomplete saved HTTP status log')
    require(events[0].get('event') == 'listening'
            and events[0]['address'].startswith('127.0.0.1:'), 'missing saved loopback readiness')
    receipt_path = context.directory / (suite + '-import.stdout')
    receipt = read(receipt_path)
    require(receipt['id'] == 'qualified' and receipt['schema_version'] == 2, 'invalid import receipt')
    root = model if model.is_dir() else model.parent
    for item in receipt['checkpoint']['files']:
        require(Path(item['name']).name == item['name'], 'unsafe receipt name')
        pinned(root / item['name'], item)
    profile = PROFILES[suite]
    case = read(ROOT / profile['reference'])['cases'][0]
    expected_usage = {'prompt_tokens': len(case['prompt_token_ids']),
                      'completion_tokens': len(case['greedy_token_ids']),
                      'total_tokens': len(case['prompt_token_ids']) + len(case['greedy_token_ids'])}
    records = []
    replies = {}
    for label, route in [('load', '/api/v1/models/load'), ('completion', '/v1/chat/completions'),
                         ('sse', '/v1/chat/completions'), ('unload', '/api/v1/models/unload')]:
        path = context.directory / (suite + '-' + label + '.response')
        require(path.is_file() and path.stat().st_size <= 16 << 20, 'missing or oversized saved reply')
        replies[label] = transport.frames(path.read_bytes()) if label == 'sse' else read(path)
        records.append({'case': label, 'route': route, 'status': 200,
                        'response_sha256': sha(path), 'revalidated_saved_response': True})
    require(replies['load']['loaded'] is True and replies['load']['model'] == 'qualified'
            and replies['load']['context'] == 256
            and replies['load']['adapter'] == receipt['checkpoint']['adapter'], 'saved load differs')
    output = replies['completion']
    require(output['choices'][0]['message']['content'] == case['greedy_text']
            and output['usage'] == expected_usage, 'saved completion text/usage differs')
    chunks = replies['sse']
    text = ''.join(c['delta'].get('content', '') for chunk in chunks for c in chunk['choices'])
    finishes = [c['finish_reason'] for chunk in chunks for c in chunk['choices'] if c.get('finish_reason')]
    require(text == case['greedy_text'] and any(c.get('usage') == expected_usage for c in chunks)
            and finishes == [output['choices'][0]['finish_reason']], 'saved SSE text/usage/finish differs')
    require(replies['unload'] == {'loaded': False, 'model': 'qualified'}, 'saved unload differs')
    return {'transport': 'preserved authenticated loopback TCP evidence; no new image request',
            'cases': records, 'daemon_log_sha256': sha(log), 'import_receipt_sha256': sha(receipt_path),
            'original_exit_unavailable': True, 'raw_response_headers_retained': False,
            'durable_restart_tested': False, 'tls_tested': False}


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
              'recovery_scope': 'Preserved numerical/CLI captures and completed image HTTP records revalidated; incomplete text HTTP checks run fresh.',
              'limitations': ['Original parent process was interrupted without a final report; its command exit codes are unavailable.',
                              'Image HTTP status logs/import bytes/replies are revalidated, not represented as fresh execution. Raw response headers and original daemon exits were not retained.',
                              'Local ARM64 host only; publisher-framework parity, broad image semantics, peak RSS and production qualification remain open.']}
    checkpoint(report)
    try:
        for suite in SELECTED:
            print('REVALIDATE ' + suite, flush=True)
            numerical = suites.qwen35(context, suite, models[suite])
            report['results'][suite] = {'numerical_and_packaged_cli': numerical}
            checkpoint(report)
            if suite.endswith('-image'):
                report['results'][suite]['http'] = saved_image_http(context, suite, models[suite])
            else:
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

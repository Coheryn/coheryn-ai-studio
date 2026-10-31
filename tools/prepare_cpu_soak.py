#!/usr/bin/env python3
"""Prepare only the frozen modern mixed-soak inputs, without executing models."""
import argparse
import hashlib
import os
from pathlib import Path
import signal
import time
import urllib.error

from check_cpu_release import ROOT, sha, source_identity
from cpu_qualification.assets import models
from cpu_qualification.host import Guard
from run_cpu_qualification import checkpoint
from run_cpu_soak import load, model_copy_bytes, validate_sources

MODEL_SET = ROOT / 'fixtures/soak/qwen35-shared-nomic-models-v2.json'
SUITES = ('embedding-q8', 'qwen35-gguf-q4_k_m-image')


def transient_download(error):
    if isinstance(error, urllib.error.HTTPError):
        return error.code in (408, 429, 500, 502, 503, 504)
    if isinstance(error, urllib.error.URLError):
        return isinstance(error.reason, (TimeoutError, ConnectionError))
    return isinstance(error, (TimeoutError, ConnectionError))


def download_retry_delay(error):
    if not transient_download(error):
        return None
    if isinstance(error, urllib.error.HTTPError):
        value = (error.headers or {}).get('Retry-After')
        if value is not None:
            try: seconds = int(value)
            except (ValueError, TypeError): return None
            return max(1, seconds) if 0 <= seconds <= 5 else None
    return 1


def verify_sources(sources, guard):
    """Stream exact declared bytes; do not infer compatibility from the model set."""
    validate_sources(sources)
    checkpoint_source = sources.get('text_checkpoint')
    if sources['schema_version'] != 2 or sources['vision']['kind'] != 'shared_text' or checkpoint_source is None:
        raise ValueError('preparation requires the frozen shared modern checkpoint')
    selected = ROOT / checkpoint_source['path']
    base = selected if selected.is_dir() else selected.parent
    files = [(base / name, digest) for name, digest in checkpoint_source['pinned_files'].items()]
    files += [(ROOT / entry['path'], entry['sha256']) for entry in (sources['embedding'], sources['vision']['image'])]
    records = []
    for path, expected in files:
        guard.check()
        if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('missing or linked soak input: ' + str(path))
        digest, count = hashlib.sha256(), 0
        with path.open('rb') as stream:
            while chunk := stream.read(1 << 20):
                guard.check()
                count += len(chunk)
                if count > 4 << 30: raise ValueError('soak input exceeds file bound')
                digest.update(chunk)
        if digest.hexdigest() != expected: raise ValueError('pinned soak input changed: ' + str(path))
        records.append({'path': path.relative_to(ROOT).as_posix(), 'bytes': count, 'sha256': expected})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true', help='explicitly allow revision-pinned HTTPS acquisition')
    parser.add_argument('--download-attempts', type=int, choices=(1, 2), default=1,
                        help='explicit preparation attempt limit for transient network failures only')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists(): raise FileExistsError(args.report)
    if os.name != 'nt':
        def interrupted(_signal, _frame): raise KeyboardInterrupt('soak preparation interrupted')
        signal.signal(signal.SIGTERM, interrupted)
    directory = ROOT / '.cache/cpu-soak-inputs' / args.report.stem
    directory.mkdir(parents=True, exist_ok=False)
    source = source_identity()
    result = {'schema_version': 1, 'kind': 'cpu-soak-input-preparation', 'outcome': 'running',
              'production_ready': False, 'execution_qualification': False, 'source_snapshot': source,
              'models_sha256': sha(MODEL_SET), 'selected_suites': list(SUITES),
              'download_attempt_limit': args.download_attempts, 'download_retry_max_delay_seconds': 5,
              'download_attempts': []}
    guard = None
    checkpoint(args.report, result)
    try:
        guard = Guard(directory)
        guard.start()
        sources = load(MODEL_SET)
        validate_sources(sources)
        for number in range(1, args.download_attempts + 1):
            attempt = {'attempt': number, 'outcome': 'running'}
            result['download_attempts'].append(attempt)
            def current_input(path, item):
                attempt['current_input'] = {'path': path.relative_to(ROOT).as_posix(),
                                             'bytes': item['bytes'], 'sha256': item['sha256']}
                checkpoint(args.report, result)
            try:
                _, result['acquired_inputs'] = models(SUITES, args.fetch, guard, on_input=current_input)
                attempt['outcome'] = 'pass'
                break
            except Exception as error:
                attempt.update(outcome='fail', error=str(error), error_type=type(error).__name__)
                checkpoint(args.report, result)
                delay = download_retry_delay(error)
                if not args.fetch or number == args.download_attempts or delay is None:
                    raise
                attempt['retry_delay_seconds'] = delay
                checkpoint(args.report, result)
                guard.check()
                time.sleep(delay)
                guard.check()
        result['verified_inputs'] = verify_sources(sources, guard)
        result['managed_copy_reservation_bytes'] = model_copy_bytes(sources)
        guard.sample(disk_reservation=result['managed_copy_reservation_bytes'])
        result['outcome'] = 'pass'
    except (Exception, KeyboardInterrupt) as error:
        result['outcome'], result['error'] = 'fail', str(error)
    finally:
        if guard:
            try: guard.close()
            except Exception as error: result['outcome'], result['observer_close_error'] = 'fail', str(error)
            result['host_guard'] = guard.report()
            if guard.stream.closed: result['host_guard']['log_sha256'] = sha(guard.log)
            if guard.reason: result['outcome'] = 'fail'
        result['source_unchanged'] = source_identity() == source
        if not result['source_unchanged']: result['outcome'] = 'fail'
        checkpoint(args.report, result)
    print(result['outcome'].upper(), 'pinned modern soak inputs:', args.report)
    raise SystemExit(result['outcome'] != 'pass')


if __name__ == '__main__':
    main()

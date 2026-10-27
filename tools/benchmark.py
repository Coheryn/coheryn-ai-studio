#!/usr/bin/env python3
"""Wrap one benchmark child with OS peak-RSS and resource observations (Unix)."""
import argparse
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('args', nargs=argparse.REMAINDER, help='Arguments passed to coheryn-bench')
    options = parser.parse_args()
    arguments = options.args[1:] if options.args[:1] == ['--'] else options.args
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    child = subprocess.run([str(ROOT/'target/release/coheryn-bench'), *arguments], check=True, capture_output=True, text=True)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = json.loads(child.stdout)
    result['host'] = {'platform': platform.platform(), 'cpu': platform.processor(), 'machine': platform.machine()}
    if sys.platform == 'darwin':
        for key in ['hw.memsize','hw.ncpu','hw.model','machdep.cpu.brand_string']:
            value = subprocess.run(['sysctl','-n',key],capture_output=True,text=True)
            result['host'][key] = value.stdout.strip() if value.returncode == 0 else None
    result['wall_ms'] = (time.monotonic()-started)*1000
    result['measurements'].update({
        'peak_rss_bytes': usage.ru_maxrss if sys.platform == 'darwin' else usage.ru_maxrss*1024,
        'minor_page_faults': usage.ru_minflt-before.ru_minflt,
        'major_page_faults': usage.ru_majflt-before.ru_majflt,
        'input_blocks': usage.ru_inblock-before.ru_inblock,
        'output_blocks': usage.ru_oublock-before.ru_oublock,
        'user_cpu_seconds': usage.ru_utime-before.ru_utime,
        'system_cpu_seconds': usage.ru_stime-before.ru_stime,
    })
    result['measurement_method'] = 'getrusage of a single benchmark child; block counters are not byte counts'
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()

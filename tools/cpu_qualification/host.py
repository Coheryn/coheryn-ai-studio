"""Bounded host observations and owned-process cutoffs for extended local tests.

These are observation limits, not engine reservations or a guarantee against an
OS/hardware fault. No third-party package or workload is used to sample the host.
"""
from contextlib import contextmanager
import ctypes
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import threading
import time
import weakref


class HostLimitError(RuntimeError):
    pass


@dataclass(frozen=True)
class Policy:
    min_free_disk_bytes: int = 8 << 30
    min_available_memory_bytes: int = 1 << 30
    max_compressed_fraction: float = .35
    max_load_per_cpu: float = 2.
    sample_seconds: float = 2.
    max_samples: int = 16384

    def __post_init__(self):
        if (self.min_free_disk_bytes < 8 << 30 or self.min_available_memory_bytes < 1 << 30
                or not 0 < self.max_compressed_fraction <= .35
                or not 0 < self.max_load_per_cpu <= 2
                or not .05 <= self.sample_seconds <= 5 or not 1 <= self.max_samples <= 16384):
            raise ValueError('invalid or weakened host observer policy')


def output(command):
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, timeout=2, check=True)
    if len(result.stdout) > 64 << 10 or len(result.stderr) > 64 << 10:
        raise ValueError('host probe output exceeds bound')
    return result.stdout.decode('utf-8', errors='strict')


def bounded_text(path):
    with Path(path).open('rb') as stream:
        data = stream.read((64 << 10) + 1)
    if len(data) > 64 << 10:
        raise ValueError('host observer input exceeds bound')
    return data.decode('utf-8', errors='strict')


def mac_memory(text, physical, pressure):
    page = re.search(r'page size of (\d+) bytes', text)
    if not page:
        raise ValueError('missing VM page size')
    size = int(page[1])
    pages = {name: int(value) for name, value in re.findall(r'^([^:\n]+):\s+(\d+)\.', text, re.M)}
    available = sum(pages[name] for name in ('Pages free', 'Pages inactive', 'Pages speculative')) * size
    compressed = pages['Pages occupied by compressor'] * size
    if not 0 < size <= 1 << 20 or not 0 < physical or not 0 <= available <= physical or not 0 <= compressed <= physical:
        raise ValueError('invalid VM memory counters')
    return {'physical_memory_bytes': physical, 'available_memory_bytes': available,
            'available_memory_kind': 'free + inactive + speculative estimate; not a reservation',
            'compressed_memory_bytes': compressed, 'memory_pressure_level': pressure}


class MacThermal:
    """Use Foundation's public NSInteger thermalState ABI on 64-bit macOS.

    The process singleton is retained by Foundation; no borrowed pointer escapes.
    Pointer-returning and NSInteger-returning messages use distinct prototypes.
    """
    def __init__(self):
        self.foundation = ctypes.CDLL('/System/Library/Frameworks/Foundation.framework/Foundation')
        self.objc = ctypes.CDLL('/usr/lib/libobjc.A.dylib')
        self.objc.objc_getClass.argtypes = [ctypes.c_char_p]
        self.objc.objc_getClass.restype = ctypes.c_void_p
        self.objc.sel_registerName.argtypes = [ctypes.c_char_p]
        self.objc.sel_registerName.restype = ctypes.c_void_p
        pointer = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(('objc_msgSend', self.objc))
        self.integer = ctypes.CFUNCTYPE(ctypes.c_ssize_t, ctypes.c_void_p, ctypes.c_void_p)(('objc_msgSend', self.objc))
        cls = self.objc.objc_getClass(b'NSProcessInfo')
        if not cls:
            raise ValueError('NSProcessInfo unavailable')
        self.process = pointer(cls, self.objc.sel_registerName(b'processInfo'))
        self.selector = self.objc.sel_registerName(b'thermalState')
        if not self.process or not self.selector:
            raise ValueError('thermal observation unavailable')

    def __call__(self):
        value = self.integer(self.process, self.selector)
        if value not in (0, 1, 2, 3):
            raise ValueError('unknown macOS thermal state')
        return ('nominal', 'fair', 'serious', 'critical')[value]


def windows_memory():
    from ctypes import wintypes
    class Status(ctypes.Structure):
        _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD)] + [
            (name, ctypes.c_ulonglong) for name in ('total_physical', 'available_physical', 'total_page',
                                                  'available_page', 'total_virtual', 'available_virtual', 'extended')]
    status = Status()
    status.length = ctypes.sizeof(status)
    call = ctypes.WinDLL('kernel32', use_last_error=True).GlobalMemoryStatusEx
    call.argtypes, call.restype = [ctypes.POINTER(Status)], wintypes.BOOL
    if not call(ctypes.byref(status)):
        raise OSError(ctypes.get_last_error(), 'GlobalMemoryStatusEx failed')
    return {'physical_memory_bytes': status.total_physical,
            'available_memory_bytes': status.available_physical,
            'available_memory_kind': 'GlobalMemoryStatusEx available physical memory'}


class Sampler:
    def __init__(self, directory):
        self.directory = directory
        self.system = platform.system()
        self.thermal = MacThermal() if self.system == 'Darwin' else None

    def __call__(self):
        row = {'system': self.system, 'disk_free_bytes': shutil.disk_usage(self.directory).free,
               'cpu_count': os.cpu_count() or 1, 'thermal_state': 'unavailable',
               'compressed_memory_bytes': None, 'memory_pressure_level': None, 'load_1m': None}
        if self.system == 'Darwin':
            row.update(mac_memory(output(['/usr/bin/vm_stat']),
                                  int(output(['/usr/sbin/sysctl', '-n', 'hw.memsize'])),
                                  int(output(['/usr/sbin/sysctl', '-n', 'kern.memorystatus_vm_pressure_level']))))
            row['thermal_state'] = self.thermal()
        elif self.system == 'Linux':
            values = {key: int(value) * 1024 for key, value in
                      re.findall(r'^(MemTotal|MemAvailable):\s+(\d+) kB$', bounded_text('/proc/meminfo'), re.M)}
            row.update(physical_memory_bytes=values['MemTotal'], available_memory_bytes=values['MemAvailable'],
                       available_memory_kind='Linux MemAvailable estimate')
        elif self.system == 'Windows':
            row.update(windows_memory())
        else:
            raise ValueError('unsupported host observer platform')
        if hasattr(os, 'getloadavg'):
            row['load_1m'] = os.getloadavg()[0]
        return row


def rejection(row, policy, disk_reservation=0):
    if type(disk_reservation) is not int or disk_reservation < 0:
        raise ValueError('invalid pending disk reservation')
    if row.get('system') not in ('Darwin', 'Linux', 'Windows'):
        return 'missing or unsupported host platform'
    for key in ('physical_memory_bytes', 'available_memory_bytes', 'disk_free_bytes', 'cpu_count'):
        if type(row.get(key)) is not int or row[key] < 0:
            return 'missing or invalid required host measurement: ' + key
    if row['physical_memory_bytes'] == 0 or row['cpu_count'] == 0:
        return 'invalid physical memory/CPU count'
    if row['available_memory_bytes'] > row['physical_memory_bytes']:
        return 'invalid available memory observation'
    if row['disk_free_bytes'] < policy.min_free_disk_bytes + disk_reservation:
        return 'insufficient host disk reserve'
    if row['available_memory_bytes'] < policy.min_available_memory_bytes:
        return 'insufficient estimated available host memory'
    load = row.get('load_1m')
    if row['system'] != 'Windows' and load is None:
        return 'required host load observation is unavailable'
    if load is not None and (type(load) not in (int, float) or not math.isfinite(load)
                             or load < 0 or load > row['cpu_count'] * policy.max_load_per_cpu):
        return 'host load exceeds cutoff'
    if row['system'] == 'Darwin':
        compressed = row.get('compressed_memory_bytes')
        if (type(compressed) is not int or not 0 <= compressed <= row['physical_memory_bytes']
                or type(row.get('memory_pressure_level')) is not int or row['memory_pressure_level'] != 1):
            return 'macOS memory pressure is not normal or is unavailable'
        if compressed > row['physical_memory_bytes'] * policy.max_compressed_fraction:
            return 'macOS compressed memory exceeds cutoff'
        if row.get('thermal_state') not in ('nominal', 'fair'):
            return 'macOS thermal state is serious/critical or unavailable'
    return None


def child_options():
    return {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}


_stop_lock = threading.RLock()
_stopped = weakref.WeakKeyDictionary()


def stop_owned(process):
    """Only called for children started in a dedicated group by this observer."""
    # The observer and command owner can notice a cutoff simultaneously. Complete
    # one group shutdown before another caller acts on the same PID, and never
    # signal a successfully retired group again. Weak keys retain no processes.
    with _stop_lock:
        if process not in _stopped:
            _stopped[process] = tuple(_stop_group(process))
        return list(_stopped[process])


def _stop_group(process):
    actions = []
    failure = None
    if os.name == 'nt':
        if process.poll() is None:
            command = [str(Path(os.environ['SystemRoot']) / 'System32/taskkill.exe'),
                       '/PID', str(process.pid), '/T', '/F']
            completed = subprocess.run(command, capture_output=True, timeout=5)
            if completed.returncode:
                failure = subprocess.CalledProcessError(completed.returncode, command)
            actions.append('taskkill /T /F')
            if process.poll() is None:
                process.kill()
                actions.append('TerminateProcess')
    else:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, sig)
                actions.append(sig.name)
            except ProcessLookupError:
                break
            except PermissionError as error:
                raise PermissionError(error.errno, f'{sig.name} denied for owned process group '
                                      f'{process.pid}; root exit={process.poll()}; '
                                      'descendant termination is unconfirmed') from error
            if sig == signal.SIGTERM:
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
    process.wait(timeout=5)
    if failure:
        raise failure
    return actions


class Guard:
    def __init__(self, directory, *, policy=None, sampler=None):
        self.policy = policy or Policy()
        self.sampler = sampler or Sampler(directory)
        self.lock = threading.RLock()
        self.ended = threading.Event()
        self.children = set()
        self.reason = None
        self.last = None
        self.samples = 0
        self.log = Path(directory) / 'host-health.jsonl'
        self.stream = self.log.open('x', encoding='utf-8')
        self.thread = None
        self.stop_errors = []

    def stop(self, process):
        try:
            return stop_owned(process)
        except (OSError, subprocess.SubprocessError) as error:
            with self.lock:
                if len(self.stop_errors) < 16:
                    self.stop_errors.append(str(error)[:512])
            raise

    def check(self):
        with self.lock:
            if self.reason:
                raise HostLimitError(self.reason)

    def sample(self, disk_reservation=0):
        with self.lock:
            self.check()
            try:
                row = self.sampler()
                reason = rejection(row, self.policy, disk_reservation)
                json.dumps(row, allow_nan=False)
            except Exception as error:
                row, reason = {'observation_error': str(error)}, 'required host observer failed: ' + str(error)
            self.samples += 1
            if self.samples > self.policy.max_samples:
                reason = 'host observation count exceeds bound'
            self.last = row
            if reason:
                self.reason = reason
            self.stream.write(json.dumps({'sample': self.samples, 'monotonic_seconds': time.monotonic(),
                                          'observation': row, 'rejection': reason}, allow_nan=False) + '\n')
            self.stream.flush()
            if reason:
                raise HostLimitError(reason)
            return row

    def start(self):
        self.sample()
        def observe():
            while not self.ended.wait(self.policy.sample_seconds):
                try:
                    self.sample()
                except Exception as error:
                    with self.lock:
                        self.reason = self.reason or 'host observer failed: ' + str(error)
                        children = list(self.children)
                    for child in children:
                        try:
                            self.stop(child)
                        except (OSError, subprocess.SubprocessError):
                            pass  # stop() preserves the failure before rethrowing.
                    break
        self.thread = threading.Thread(target=observe, name='qualification-host-observer', daemon=True)
        self.thread.start()

    @contextmanager
    def owned(self, process):
        with self.lock:
            if self.reason:
                self.stop(process)
                self.check()
            if len(self.children) >= 2:
                self.reason = 'qualification owned-process limit exceeded'
                self.stop(process)
                self.check()
            self.children.add(process)
        try:
            yield process
            self.check()
        finally:
            with self.lock:
                self.children.discard(process)

    def close(self):
        self.ended.set()
        if self.thread:
            self.thread.join(timeout=15)
            if self.thread.is_alive():
                self.reason = self.reason or 'host observer did not stop'
                raise HostLimitError('host observer did not stop')
        self.stream.close()

    def report(self):
        with self.lock:
            return {'policy': asdict(self.policy), 'samples': self.samples, 'last': self.last,
                    'stop_reason': self.reason, 'log': self.log.name, 'stop_errors': self.stop_errors.copy(),
                    'scope': 'Host observations only; unavailable sensors are explicit. Polling cannot prevent every OS/hardware fault.'}

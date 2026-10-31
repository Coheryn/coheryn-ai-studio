#!/usr/bin/env python3
"""Lossless fixture derivation: physical files, exact bytes, hostile headers and atomic cleanup."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from cpu_qualification.shards import catalog, derive
from cpu_qualification.common import sha
from cpu_qualification.host import HostLimitError


class ShardTests(unittest.TestCase):
    def source(self, root):
        header = {'__metadata__': {'format': 'pt'}}
        payload = b''
        for index, count in enumerate((3, 7, 2, 5, 1)):
            data = bytes(range(index * 16, index * 16 + count * 2))
            header[f'tensor.{index}'] = {'dtype': 'BF16', 'shape': [count], 'data_offsets': [len(payload), len(payload) + len(data)]}
            payload += data
        encoded = json.dumps(header, separators=(',', ':')).encode()
        encoded += b' ' * (-len(encoded) % 8)
        path = root / 'source.safetensors'
        path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + payload)
        return path, {'file': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path)}

    def test_physical_shards_preserve_all_named_bytes_and_reproduce(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source, identity = self.source(root)
            with patch('cpu_qualification.shards.shutil.disk_usage') as space:
                space.return_value.free = 1 << 40
                first = derive(source, root / 'one', identity, [], 3)
                second = derive(source, root / 'two', identity, [], 3)
            self.assertEqual(first, second)
            self.assertEqual(first['physical_shards'], 3)
            self.assertFalse(first['numeric_conversion'])
            self.assertEqual(len(list((root / 'one').glob('*.safetensors'))), 3)
            with source.open('rb') as stream:
                start, _, tensors = catalog(stream, source.stat().st_size)
                expected = {}
                for t in tensors:
                    stream.seek(start + t['offset']); expected[t['name']] = stream.read(t['bytes'])
            actual = {}
            for path in (root / 'one').glob('*.safetensors'):
                with path.open('rb') as stream:
                    start, _, tensors = catalog(stream, path.stat().st_size)
                    for t in tensors:
                        stream.seek(start + t['offset']); actual[t['name']] = stream.read(t['bytes'])
            self.assertEqual(actual, expected)

    def test_corrupt_inputs_and_existing_destination_are_not_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); source, identity = self.source(root)
            (root / 'existing').mkdir(); (root / 'existing/keep').write_bytes(b'keep')
            with self.assertRaises(ValueError): derive(source, root / 'existing', identity, [], 3)
            self.assertEqual((root / 'existing/keep').read_bytes(), b'keep')
            source.write_bytes(b'corrupt')
            with self.assertRaises(ValueError): derive(source, root / 'new', identity, [], 3)
            self.assertFalse((root / 'new').exists())

    def test_hostile_header_counts_duplicates_and_ranges_fail_without_publication(self):
        examples = [struct.pack('<Q', 1 << 63),
                    struct.pack('<Q', 15) + b'{"a":1,"a":2}  ',
                    b'{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,8]}}']
        for data in examples:
            with self.subTest(data=data[:16]), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve(); source = root / 'bad.safetensors'; source.write_bytes(data)
                identity = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                with patch('cpu_qualification.shards.shutil.disk_usage') as space:
                    space.return_value.free = 1 << 40
                    with self.assertRaises((ValueError, UnicodeDecodeError)): derive(source, root / 'out', identity, [], 2)
                self.assertFalse((root / 'out').exists())
                self.assertEqual(list(root.glob('.reshard-*')), [])

    def test_failed_sidecar_copy_cleans_staging(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); source, identity = self.source(root)
            sidecar = root / 'config.json'; sidecar.write_bytes(b'{}')
            asset = {'file': sidecar.name, 'bytes': 2, 'sha256': sha(sidecar)}
            with patch('cpu_qualification.shards.shutil.disk_usage') as space, patch('cpu_qualification.shards.shutil.copyfile', side_effect=OSError('injected write failure')):
                space.return_value.free = 1 << 40
                with self.assertRaises(OSError): derive(source, root / 'out', identity, [asset], 3)
            self.assertFalse((root / 'out').exists()); self.assertEqual(list(root.glob('.reshard-*')), [])

    def test_host_abort_during_copy_cleans_staging_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); source, identity = self.source(root)
            def stop(): raise HostLimitError('host pressure')
            with patch('cpu_qualification.shards.shutil.disk_usage') as space:
                space.return_value.free = 1 << 40
                with self.assertRaises(HostLimitError): derive(source, root / 'out', identity, [], 3, check=stop)
            self.assertFalse((root / 'out').exists())
            self.assertEqual(list(root.glob('.reshard-*')), [])
            self.assertEqual(sha(source), identity['sha256'])

    def test_host_abort_before_publication_cleans_completed_staging(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); source, identity = self.source(root)
            def stop_before_publication():
                if list(root.glob('.reshard-*/DERIVATION.json')): raise HostLimitError('host pressure')
            with patch('cpu_qualification.shards.shutil.disk_usage') as space:
                space.return_value.free = 1 << 40
                with self.assertRaises(HostLimitError):
                    derive(source, root / 'out', identity, [], 3, check=stop_before_publication)
            self.assertFalse((root / 'out').exists())
            self.assertEqual(list(root.glob('.reshard-*')), [])


if __name__ == '__main__': unittest.main()

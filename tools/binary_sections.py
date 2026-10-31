"""Bounded instruction-section discovery for native artifact text inspection.

This is not an executable loader or architecture compatibility validator. Unknown
payloads retain full scanning. Recognized malformed section tables fail closed.
"""
import struct


class BinaryLayoutError(ValueError):
    pass


def instruction_ranges(path):
    size = path.stat().st_size
    regions = []
    with path.open('rb') as stream:
        def read(offset, count):
            if offset < 0 or count < 0 or count > 1 << 20 or offset > size or count > size - offset:
                raise BinaryLayoutError('executable section metadata exceeds bounds')
            stream.seek(offset)
            data = stream.read(count)
            if len(data) != count:
                raise BinaryLayoutError('truncated executable section metadata')
            return data

        def add(offset, count):
            if offset > size or count > size - offset:
                raise BinaryLayoutError('instruction section exceeds file bounds')
            if count:
                regions.append((offset, offset + count))

        magic = read(0, min(4, size))
        if magic == b'\x7fELF':
            header = read(0, 64)
            if header[4] != 2 or header[5] not in (1, 2):
                raise BinaryLayoutError('unsupported ELF section metadata encoding')
            endian = '<' if header[5] == 1 else '>'
            fields = struct.unpack(endian + '16sHHIQQQIHHHHHH', header)
            table, entry_size, count = fields[6], fields[11], fields[12]
            if count:
                if fields[8] != 64 or table < 64 or entry_size != 64 or count > 4096:
                    raise BinaryLayoutError('invalid ELF section table')
                entries = read(table, count * entry_size)
                for index in range(count):
                    section = struct.unpack_from(endian + 'IIQQQQIIQQ', entries, index * 64)
                    if section[1] != 8 and (section[4] > size or section[5] > size - section[4]):
                        raise BinaryLayoutError('ELF section exceeds file bounds')
                    if section[1] == 1 and section[2] & 4:
                        add(section[4], section[5])
        elif magic in (b'\xcf\xfa\xed\xfe', b'\xfe\xed\xfa\xcf'):
            endian = '<' if magic == b'\xcf\xfa\xed\xfe' else '>'
            fields = struct.unpack(endian + '8I', read(0, 32))
            count, commands_size = fields[4], fields[5]
            if count > 4096 or commands_size > 1 << 20:
                raise BinaryLayoutError('Mach-O load commands exceed bounds')
            commands = read(32, commands_size)
            offset = 0
            for _ in range(count):
                if offset + 8 > len(commands):
                    raise BinaryLayoutError('truncated Mach-O load command')
                kind, length = struct.unpack_from(endian + 'II', commands, offset)
                if length < 8 or length > len(commands) - offset:
                    raise BinaryLayoutError('invalid Mach-O load command size')
                if kind == 0x19:
                    if length < 72:
                        raise BinaryLayoutError('truncated Mach-O segment')
                    segment = struct.unpack_from(endian + 'II16sQQQQIIII', commands, offset)
                    sections = segment[9]
                    if sections > 4096 or 72 + sections * 80 > length:
                        raise BinaryLayoutError('invalid Mach-O section count')
                    for index in range(sections):
                        section = struct.unpack_from(endian + '16s16sQQIIIIIIII', commands, offset + 72 + index * 80)
                        # Zerofill sections have virtual sizes rather than file data.
                        if section[8] & 0xff not in (1, 0xc, 0x12):
                            if section[4] > size or section[3] > size - section[4]:
                                raise BinaryLayoutError('Mach-O section exceeds file bounds')
                            if section[8] & (0x80000000 | 0x400):
                                add(section[4], section[3])
                offset += length
            if offset != len(commands):
                raise BinaryLayoutError('Mach-O load command coverage mismatch')
        elif magic[:2] == b'MZ':
            pe_offset = struct.unpack_from('<I', read(0, 64), 60)[0]
            if pe_offset > 1 << 20 or read(pe_offset, 4) != b'PE\0\0':
                raise BinaryLayoutError('invalid PE header')
            fields = struct.unpack('<HHIIIHH', read(pe_offset + 4, 20))
            count, optional_size = fields[1], fields[5]
            if count > 4096 or optional_size > 4096:
                raise BinaryLayoutError('PE section metadata exceeds bounds')
            entries = read(pe_offset + 24 + optional_size, count * 40)
            for index in range(count):
                section = struct.unpack_from('<8sIIIIIIHHI', entries, index * 40)
                if section[4] > size or section[3] > size - section[4]:
                    raise BinaryLayoutError('PE section exceeds file bounds')
                if section[9] & (0x20 | 0x20000000):
                    add(section[4], section[3])
    merged = []
    for start, end in sorted(regions):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def data_ranges(size, instructions):
    position = 0
    for start, end in instructions:
        if start > position:
            yield position, start
        position = end
    if position < size:
        yield position, size

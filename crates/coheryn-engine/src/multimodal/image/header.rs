//! Checked structural image preflight. Compressed-pixel integrity is checked by decoding.
use super::*;
fn invalid(message: &str) -> Error {
    Error::new(ErrorCode::InvalidImage, message)
}
fn unsupported(message: &str) -> Error {
    Error::new(ErrorCode::UnsupportedImageFormat, message)
}
fn range(bytes: &[u8], start: usize, length: usize) -> Result<&[u8]> {
    bytes
        .get(
            start
                ..start
                    .checked_add(length)
                    .ok_or_else(|| invalid("image range overflow"))?,
        )
        .ok_or_else(|| invalid("truncated image structure"))
}
fn be32(b: &[u8]) -> u32 {
    u32::from_be_bytes(b.try_into().expect("checked four-byte field"))
}
fn orientation(bytes: &[u8]) -> Result<u8> {
    let le = match range(bytes, 0, 2)? {
        b"II" => true,
        b"MM" => false,
        _ => return Err(invalid("invalid EXIF byte order")),
    };
    let u16_at = |p| -> Result<u16> {
        let b: [u8; 2] = range(bytes, p, 2)?.try_into().expect("checked");
        Ok(if le {
            u16::from_le_bytes(b)
        } else {
            u16::from_be_bytes(b)
        })
    };
    let u32_at = |p| -> Result<u32> {
        let b: [u8; 4] = range(bytes, p, 4)?.try_into().expect("checked");
        Ok(if le {
            u32::from_le_bytes(b)
        } else {
            u32::from_be_bytes(b)
        })
    };
    if u16_at(2)? != 42 {
        return Err(invalid("unsupported EXIF TIFF header"));
    }
    let offset = u32_at(4)? as usize;
    if offset < 8 {
        return Err(invalid("EXIF directory overlaps TIFF header"));
    }
    let count = u16_at(offset)? as usize;
    if count > 512 {
        return Err(limit("EXIF directory exceeds 512 entries"));
    }
    range(bytes, offset + 2, count * 12 + 4)?;
    let mut value = None;
    for index in 0..count {
        let entry = offset + 2 + index * 12;
        if u16_at(entry)? == 0x0112 {
            if value.is_some() || u16_at(entry + 2)? != 3 || u32_at(entry + 4)? != 1 {
                return Err(invalid("invalid or duplicate EXIF orientation"));
            }
            let tag = u16_at(entry + 8)?;
            if !(1..=8).contains(&tag) {
                return Err(invalid("EXIF orientation must be in 1..=8"));
            }
            value = Some(tag as u8);
        }
    }
    Ok(value.unwrap_or(1))
}
const fn crc_table() -> [u32; 256] {
    let mut table = [0; 256];
    let mut i = 0;
    while i < 256 {
        let mut value = i as u32;
        let mut bit = 0;
        while bit < 8 {
            value = if value & 1 == 0 {
                value >> 1
            } else {
                (value >> 1) ^ 0xedb88320
            };
            bit += 1;
        }
        table[i] = value;
        i += 1;
    }
    table
}
const CRC: [u32; 256] = crc_table();
fn crc_valid(kind: &[u8], data: &[u8], expected: u32, cancel: &Cancellation) -> Result<bool> {
    let mut crc = u32::MAX;
    for bytes in [kind, data] {
        for chunk in bytes.chunks(4096) {
            cancel.check()?;
            for byte in chunk {
                crc = CRC[((crc ^ u32::from(*byte)) & 255) as usize] ^ (crc >> 8);
            }
        }
    }
    Ok(crc ^ u32::MAX == expected)
}
fn png(bytes: &[u8], cancel: &Cancellation) -> Result<ImageInfo> {
    let mut cursor = 8;
    let mut info = None;
    let mut ended = false;
    let mut exif = None;
    let mut idat = false;
    for chunk_index in 0..4096 {
        cancel.check()?;
        let length = be32(range(bytes, cursor, 4)?) as usize;
        if length > bytes.len() {
            return Err(invalid("PNG chunk exceeds encoded input"));
        }
        let kind = range(bytes, cursor + 4, 4)?;
        if !kind.iter().all(u8::is_ascii_alphabetic) || !kind[2].is_ascii_uppercase() {
            return Err(invalid("invalid PNG chunk type"));
        }
        let data = range(bytes, cursor + 8, length)?;
        let checksum = be32(range(bytes, cursor + 8 + length, 4)?);
        if !crc_valid(kind, data, checksum, cancel)? {
            return Err(invalid("PNG chunk checksum mismatch"));
        }
        if chunk_index == 0 && kind != b"IHDR" {
            return Err(invalid("PNG must start with IHDR"));
        }
        match kind {
            b"IHDR" => {
                if info.is_some() || length != 13 {
                    return Err(invalid("invalid duplicate PNG header"));
                }
                let channels = match data[9] {
                    0 => 1,
                    2 => 3,
                    4 => 2,
                    6 => 4,
                    _ => return Err(unsupported("PNG palette images are not qualified")),
                };
                if data[8] != 8 {
                    return Err(unsupported("only 8-bit PNG is qualified"));
                }
                if data[10] != 0 || data[11] != 0 || data[12] > 1 {
                    return Err(invalid("invalid PNG method fields"));
                }
                info = Some(ImageInfo {
                    format: ImageFormat::Png,
                    width: be32(&data[..4]),
                    height: be32(&data[4..8]),
                    channels,
                    orientation: 1,
                    encoded_bytes: bytes.len() as u64,
                });
            }
            b"acTL" | b"fcTL" | b"fdAT" => {
                return Err(unsupported("animated PNG is not qualified"));
            }
            b"iCCP" | b"cICP" => {
                return Err(unsupported("PNG color-profile conversion is not qualified"));
            }
            b"PLTE" => return Err(unsupported("PNG palettes are not qualified")),
            b"tRNS" => return Err(unsupported("PNG color-key transparency is not qualified")),
            b"eXIf" => {
                if exif.is_some() {
                    return Err(invalid("duplicate PNG EXIF"));
                }
                exif = Some(orientation(data)?);
            }
            b"IDAT" => {
                idat = true;
            }
            b"IEND" => {
                if length != 0 || !idat {
                    return Err(invalid("invalid PNG end/image data"));
                }
                ended = true;
            }
            _ => {
                if kind[0].is_ascii_uppercase() {
                    return Err(unsupported("unknown critical PNG chunk"));
                }
            }
        }
        cursor = cursor
            .checked_add(length + 12)
            .ok_or_else(|| invalid("PNG chunk range overflow"))?;
        if ended {
            break;
        }
    }
    if !ended || cursor != bytes.len() {
        return Err(invalid(
            "PNG must have a bounded complete container without trailing bytes",
        ));
    }
    let mut info = info.ok_or_else(|| invalid("missing PNG header"))?;
    info.orientation = exif.unwrap_or(1);
    Ok(info)
}
fn jpeg(bytes: &[u8], cancel: &Cancellation) -> Result<ImageInfo> {
    let mut cursor = 2;
    let mut info = None;
    let mut exif = None;
    let mut scans = 0;
    let mut entropy = false;
    let mut ended = false;
    for _ in 0..4096 {
        cancel.check()?;
        if entropy {
            loop {
                cancel.check()?;
                let tail = bytes
                    .get(cursor..)
                    .ok_or_else(|| invalid("truncated JPEG entropy"))?;
                let window = &tail[..tail.len().min(4096)];
                let Some(relative) = window.iter().position(|b| *b == 0xff) else {
                    if window.is_empty() {
                        return Err(invalid("JPEG missing end marker"));
                    }
                    cursor += window.len();
                    continue;
                };
                cursor += relative;
                let mut next = cursor + 1;
                while *range(bytes, next, 1)?.first().expect("checked") == 0xff {
                    next += 1;
                    if next % 4096 == 0 {
                        cancel.check()?;
                    }
                }
                let marker = range(bytes, next, 1)?[0];
                if marker == 0 || (0xd0..=0xd7).contains(&marker) {
                    cursor = next + 1;
                    continue;
                }
                entropy = false;
                break;
            }
        }
        if range(bytes, cursor, 1)?[0] != 0xff {
            return Err(invalid("JPEG marker prefix missing"));
        }
        while range(bytes, cursor, 1)?[0] == 0xff {
            cursor += 1;
            if cursor % 4096 == 0 {
                cancel.check()?;
            }
        }
        let marker = range(bytes, cursor, 1)?[0];
        cursor += 1;
        if marker == 0xd9 {
            ended = true;
            break;
        }
        if marker == 0xd8 || marker == 0 || (0xd0..=0xd7).contains(&marker) {
            return Err(invalid("unexpected JPEG marker"));
        }
        let count =
            u16::from_be_bytes(range(bytes, cursor, 2)?.try_into().expect("checked")) as usize;
        if count < 2 {
            return Err(invalid("invalid JPEG segment length"));
        }
        let data = range(bytes, cursor + 2, count - 2)?;
        match marker {
            0xc0 => {
                if info.is_some() || data.len() < 6 {
                    return Err(invalid("invalid JPEG frame"));
                }
                let channels = data[5];
                if !matches!(channels, 1 | 3) {
                    return Err(unsupported("CMYK/YCCK JPEG is not qualified"));
                }
                if data[0] != 8 {
                    return Err(unsupported("only baseline 8-bit JPEG is qualified"));
                }
                if data.len() != 6 + channels as usize * 3 {
                    return Err(invalid("JPEG frame component length mismatch"));
                }
                info = Some(ImageInfo {
                    format: ImageFormat::Jpeg,
                    width: u16::from_be_bytes([data[3], data[4]]) as u32,
                    height: u16::from_be_bytes([data[1], data[2]]) as u32,
                    channels,
                    orientation: 1,
                    encoded_bytes: bytes.len() as u64,
                });
            }
            0xc1..=0xcf if !matches!(marker, 0xc4 | 0xc8 | 0xcc) => {
                return Err(unsupported("only baseline sequential JPEG is qualified"));
            }
            0xe1 if data.starts_with(b"Exif\0\0") => {
                if exif.is_some() {
                    return Err(invalid("duplicate JPEG EXIF"));
                }
                exif = Some(orientation(&data[6..])?);
            }
            0xe2 if data.starts_with(b"ICC_PROFILE") || data.starts_with(b"MPF\0") => {
                return Err(unsupported(
                    "JPEG profiles and multi-picture containers are not qualified",
                ));
            }
            0xda => {
                scans += 1;
                let channels = info
                    .as_ref()
                    .ok_or_else(|| invalid("JPEG scan precedes frame"))?
                    .channels;
                if scans != 1 || data.first().copied() != Some(channels) {
                    return Err(unsupported("only one interleaved JPEG scan is qualified"));
                }
                if data.len() != 1 + channels as usize * 2 + 3
                    || data[data.len() - 3..] != [0, 63, 0]
                {
                    return Err(unsupported("unsupported baseline JPEG scan parameters"));
                }
                entropy = true;
            }
            _ => {}
        }
        cursor += count;
    }
    if !ended || scans != 1 || cursor != bytes.len() {
        return Err(invalid(
            "JPEG must end with one complete image without trailing bytes",
        ));
    }
    let mut info = info.ok_or_else(|| invalid("JPEG frame missing"))?;
    info.orientation = exif.unwrap_or(1);
    Ok(info)
}
pub(super) fn inspect(
    bytes: &[u8],
    limits: &ImageLimits,
    cancel: &Cancellation,
) -> Result<ImageInfo> {
    limits.validate()?;
    cancel.check()?;
    if bytes.is_empty() || bytes.len() as u64 > limits.max_encoded_bytes {
        return Err(limit("encoded image byte limit exceeded"));
    }
    let info = if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        png(bytes, cancel)?
    } else if bytes.starts_with(b"\xff\xd8") {
        jpeg(bytes, cancel)?
    } else {
        return Err(unsupported(
            "only PNG and baseline JPEG signatures are supported",
        ));
    };
    if info.width == 0
        || info.height == 0
        || info.width > limits.max_edge
        || info.height > limits.max_edge
        || u64::from(info.width) * u64::from(info.height) > limits.max_source_pixels
    {
        return Err(limit("image dimensions/pixels exceed decoded-input limits"));
    }
    Ok(info)
}

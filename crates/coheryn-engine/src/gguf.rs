//! Bounded GGUF v3 metadata reader. Tensor payloads are never copied by inspection.
use crate::{Error, ErrorCode, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufReader, Read, Seek},
    path::Path,
};

struct DigestWriter(Sha256);
impl std::io::Write for DigestWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ParseLimits {
    pub max_metadata_entries: u64,
    pub max_tensors: u64,
    pub max_array_items: u64,
    pub max_string_bytes: u64,
    pub max_header_bytes: u64,
    pub max_metadata_allocation: u64,
    pub max_depth: u8,
}
impl Default for ParseLimits {
    fn default() -> Self {
        Self {
            max_metadata_entries: 65_536,
            max_tensors: 100_000,
            max_array_items: 1_000_000,
            max_string_bytes: 8 << 20,
            max_header_bytes: 128 << 20,
            max_metadata_allocation: 256 << 20,
            max_depth: 4,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum Value {
    Unsigned(u64),
    Signed(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}
impl Value {
    pub fn uint(&self) -> Option<u64> {
        if let Self::Unsigned(v) = self {
            Some(*v)
        } else {
            None
        }
    }
    pub fn float(&self) -> Option<f64> {
        if let Self::Float(v) = self {
            Some(*v)
        } else {
            None
        }
    }
    pub fn string(&self) -> Option<&str> {
        if let Self::String(v) = self {
            Some(v)
        } else {
            None
        }
    }
    pub fn array(&self) -> Option<&[Value]> {
        if let Self::Array(v) = self {
            Some(v)
        } else {
            None
        }
    }
    pub fn integer(&self) -> Option<u64> {
        match self {
            Self::Unsigned(n) => Some(*n),
            Self::Signed(n) if *n >= 0 => Some(*n as u64),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Tensor {
    pub name: String,
    /// GGML order: contiguous input dimension first.
    pub dimensions: Vec<u64>,
    pub dtype: u32,
    pub offset: u64,
    pub bytes: u64,
}
pub fn encoding(dtype: u32) -> Option<(&'static str, u64, u64)> {
    Some(match dtype {
        0 => ("F32", 1, 4),
        1 => ("F16", 1, 2),
        2 => ("Q4_0", 32, 18),
        3 => ("Q4_1", 32, 20),
        6 => ("Q5_0", 32, 22),
        7 => ("Q5_1", 32, 24),
        8 => ("Q8_0", 32, 34),
        9 => ("Q8_1", 32, 40),
        10 => ("Q2_K", 256, 84),
        11 => ("Q3_K", 256, 110),
        12 => ("Q4_K", 256, 144),
        13 => ("Q5_K", 256, 176),
        14 => ("Q6_K", 256, 210),
        15 => ("Q8_K", 256, 292),
        24 => ("I8", 1, 1),
        25 => ("I16", 1, 2),
        26 => ("I32", 1, 4),
        27 => ("I64", 1, 8),
        28 => ("F64", 1, 8),
        30 => ("BF16", 1, 2),
        _ => return None,
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct Descriptor {
    pub version: u32,
    pub byte_order: String,
    pub file_bytes: u64,
    pub data_offset: u64,
    pub architecture: Option<String>,
    pub tensor_bytes: u64,
    pub encodings: BTreeMap<String, usize>,
    pub metadata_fingerprints: BTreeMap<String, String>,
    pub header_sha256: String,
    pub tensors: Vec<Tensor>,
    #[serde(skip_serializing)]
    pub metadata: BTreeMap<String, Value>,
}
impl Descriptor {
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }
    pub fn text(&self, key: &str) -> Option<&str> {
        self.get(key)?.string()
    }
    pub fn uint(&self, key: &str) -> Option<u64> {
        self.get(key)?.uint()
    }
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

struct Reader<R> {
    inner: R,
    length: u64,
    pos: u64,
    allocation: u64,
    big: bool,
    limits: ParseLimits,
    hasher: Sha256,
}
impl<R: Read + Seek> Reader<R> {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        self.guard(N as u64)?;
        let mut b = [0; N];
        self.inner
            .read_exact(&mut b)
            .map_err(|_| Error::invalid("truncated GGUF header"))?;
        self.hasher.update(b);
        self.pos += N as u64;
        Ok(b)
    }
    fn guard(&self, n: u64) -> Result<()> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| Error::invalid("header offset overflow"))?;
        if end > self.length || end > self.limits.max_header_bytes {
            return Err(Error::invalid("header exceeds file or parser byte limit"));
        }
        Ok(())
    }
    fn charge(&mut self, n: u64) -> Result<()> {
        self.allocation = self
            .allocation
            .checked_add(n)
            .ok_or_else(|| Error::resource("metadata allocation overflow"))?;
        if self.allocation > self.limits.max_metadata_allocation {
            return Err(Error::resource("metadata allocation limit exceeded"));
        }
        Ok(())
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.bytes::<1>()?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        let b = self.bytes()?;
        Ok(if self.big {
            u16::from_be_bytes(b)
        } else {
            u16::from_le_bytes(b)
        })
    }
    fn u32(&mut self) -> Result<u32> {
        let b = self.bytes()?;
        Ok(if self.big {
            u32::from_be_bytes(b)
        } else {
            u32::from_le_bytes(b)
        })
    }
    fn u64(&mut self) -> Result<u64> {
        let b = self.bytes()?;
        Ok(if self.big {
            u64::from_be_bytes(b)
        } else {
            u64::from_le_bytes(b)
        })
    }
    fn string(&mut self, max: u64) -> Result<String> {
        let n = self.u64()?;
        if n > max.min(self.limits.max_string_bytes) {
            return Err(Error::invalid("string length exceeds parser limit"));
        }
        self.guard(n)?;
        self.charge(n + 24)?;
        let mut b = Vec::new();
        b.try_reserve_exact(n as usize)
            .map_err(|_| Error::resource("string allocation failed"))?;
        b.resize(n as usize, 0);
        self.inner
            .read_exact(&mut b)
            .map_err(|_| Error::invalid("truncated string"))?;
        self.hasher.update(&b);
        self.pos += n;
        String::from_utf8(b).map_err(|_| Error::invalid("invalid UTF-8 string"))
    }
    fn value(&mut self, ty: u32, depth: u8) -> Result<Value> {
        Ok(match ty {
            0 => Value::Unsigned(self.u8()? as u64),
            1 => Value::Signed(self.u8()? as i8 as i64),
            2 => Value::Unsigned(self.u16()? as u64),
            3 => Value::Signed(self.u16()? as i16 as i64),
            4 => Value::Unsigned(self.u32()? as u64),
            5 => Value::Signed(self.u32()? as i32 as i64),
            6 => Value::Float(f32::from_bits(self.u32()?) as f64),
            7 => match self.u8()? {
                0 => Value::Bool(false),
                1 => Value::Bool(true),
                _ => return Err(Error::invalid("invalid boolean")),
            },
            8 => Value::String(self.string(self.limits.max_string_bytes)?),
            9 => {
                if depth >= self.limits.max_depth {
                    return Err(Error::invalid("metadata nesting limit exceeded"));
                }
                let item_ty = self.u32()?;
                if item_ty > 12 {
                    return Err(Error::invalid("unknown array element type"));
                }
                let n = self.u64()?;
                if n > self.limits.max_array_items {
                    return Err(Error::invalid("array count exceeds parser limit"));
                }
                let minimum = match item_ty {
                    0 | 1 | 7 => 1,
                    2 | 3 => 2,
                    4..=6 => 4,
                    8 | 10 | 11 | 12 => 8,
                    9 => 12,
                    _ => unreachable!(),
                };
                self.guard(
                    n.checked_mul(minimum)
                        .ok_or_else(|| Error::invalid("array byte extent overflow"))?,
                )?;
                self.charge(
                    n.checked_mul(std::mem::size_of::<Value>() as u64)
                        .ok_or_else(|| Error::resource("array overflow"))?,
                )?;
                let mut values = Vec::new();
                values
                    .try_reserve_exact(n as usize)
                    .map_err(|_| Error::resource("array allocation failed"))?;
                for _ in 0..n {
                    values.push(self.value(item_ty, depth + 1)?);
                }
                Value::Array(values)
            }
            10 => Value::Unsigned(self.u64()?),
            11 => Value::Signed(self.u64()? as i64),
            12 => Value::Float(f64::from_bits(self.u64()?)),
            _ => return Err(Error::invalid("unknown metadata value type")),
        })
    }
}

pub fn inspect(path: impl AsRef<Path>, limits: ParseLimits) -> Result<Descriptor> {
    let file = File::open(path)?;
    let length = file.metadata()?.len();
    parse(BufReader::new(file), length, limits)
}

pub fn parse<R: Read + Seek>(inner: R, length: u64, limits: ParseLimits) -> Result<Descriptor> {
    let mut r = Reader {
        inner,
        length,
        pos: 0,
        allocation: 0,
        big: false,
        limits,
        hasher: Sha256::new(),
    };
    if r.bytes::<4>()? != *b"GGUF" {
        return Err(Error::invalid("GGUF magic is missing"));
    }
    let version_bytes = r.bytes::<4>()?;
    r.big = u32::from_be_bytes(version_bytes) == 3;
    let version = if r.big {
        u32::from_be_bytes(version_bytes)
    } else {
        u32::from_le_bytes(version_bytes)
    };
    if version != 3 {
        return Err(Error::new(
            ErrorCode::UnsupportedGgufVersion,
            format!("GGUF version {version}; expected v3"),
        ));
    }
    let tensor_count = r.u64()?;
    let metadata_count = r.u64()?;
    if tensor_count > limits.max_tensors || metadata_count > limits.max_metadata_entries {
        return Err(Error::invalid("descriptor count exceeds parser limit"));
    }
    r.guard(
        metadata_count
            .checked_mul(13)
            .ok_or_else(|| Error::invalid("metadata minimum size overflow"))?,
    )?;
    r.charge(
        metadata_count
            .checked_mul(128)
            .ok_or_else(|| Error::resource("metadata count overflow"))?,
    )?;
    let mut metadata = BTreeMap::new();
    for _ in 0..metadata_count {
        let key = r.string(65535)?;
        if key.is_empty() || !key.is_ascii() {
            return Err(Error::invalid("metadata key must be nonempty ASCII"));
        }
        let ty = r.u32()?;
        let value = r.value(ty, 0)?;
        if metadata.insert(key, value).is_some() {
            return Err(Error::invalid("duplicate metadata key"));
        }
    }
    let alignment = match metadata.get("general.alignment") {
        Some(Value::Unsigned(n)) if *n > 0 && n.is_power_of_two() && *n <= 65536 => *n,
        None => 32,
        _ => {
            return Err(Error::invalid(
                "alignment must be a positive power of two <= 65536",
            ));
        }
    };
    r.guard(
        tensor_count
            .checked_mul(32)
            .ok_or_else(|| Error::invalid("tensor descriptor minimum size overflow"))?,
    )?;
    r.charge(
        tensor_count
            .checked_mul(256)
            .ok_or_else(|| Error::resource("tensor count overflow"))?,
    )?;
    let mut tensors = Vec::new();
    tensors
        .try_reserve_exact(tensor_count as usize)
        .map_err(|_| Error::resource("tensor descriptor allocation failed"))?;
    let mut names = std::collections::BTreeSet::new();
    let mut encodings = BTreeMap::new();
    let mut tensor_bytes = 0u64;
    for _ in 0..tensor_count {
        let name = r.string(64)?;
        if name.is_empty() || !names.insert(name.clone()) {
            return Err(Error::invalid("empty or duplicate tensor name"));
        }
        let dims = r.u32()?;
        if !(1..=4).contains(&dims) {
            return Err(Error::invalid("tensor rank must be 1..=4"));
        }
        let mut dimensions = Vec::with_capacity(dims as usize);
        let mut elements = 1u64;
        for _ in 0..dims {
            let dim = r.u64()?;
            if dim == 0 {
                return Err(Error::invalid("zero tensor dimension"));
            }
            elements = elements
                .checked_mul(dim)
                .ok_or_else(|| Error::invalid("tensor shape overflow"))?;
            dimensions.push(dim);
        }
        let dtype = r.u32()?;
        let (label, block, size) = encoding(dtype).ok_or_else(|| {
            Error::new(
                ErrorCode::UnsupportedGgufTensorType,
                format!("cannot validate byte extent of tensor {name}: dtype {dtype}"),
            )
        })?;
        if dimensions[0] % block != 0 {
            return Err(Error::invalid(
                "tensor row is not divisible by encoding block size",
            ));
        }
        let bytes = (elements / block)
            .checked_mul(size)
            .ok_or_else(|| Error::invalid("tensor byte size overflow"))?;
        let offset = r.u64()?;
        if offset % alignment != 0 {
            return Err(Error::invalid("unaligned tensor offset"));
        }
        tensor_bytes = tensor_bytes
            .checked_add(bytes)
            .ok_or_else(|| Error::invalid("total tensor size overflow"))?;
        *encodings.entry(label.to_string()).or_insert(0) += 1;
        tensors.push(Tensor {
            name,
            dimensions,
            dtype,
            offset,
            bytes,
        });
    }
    let data_offset = r
        .pos
        .checked_add(alignment - 1)
        .ok_or_else(|| Error::invalid("data alignment overflow"))?
        / alignment
        * alignment;
    if data_offset > length {
        return Err(Error::invalid("tensor data starts beyond file"));
    }
    let mut ranges: Vec<_> = tensors.iter().map(|t| (t.offset, t.bytes)).collect();
    ranges.sort_unstable();
    let mut last_end = 0;
    for (offset, bytes) in ranges {
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| Error::invalid("tensor offset overflow"))?;
        if offset < last_end || end > length - data_offset {
            return Err(Error::invalid("overlapping or out-of-file tensor payload"));
        }
        last_end = end;
    }
    let architecture = metadata
        .get("general.architecture")
        .and_then(Value::string)
        .map(str::to_owned);
    let mut metadata_fingerprints = BTreeMap::new();
    for key in [
        "tokenizer.chat_template",
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.merges",
        "tokenizer.ggml.token_type",
    ] {
        if let Some(value) = metadata.get(key) {
            let mut writer = DigestWriter(Sha256::new());
            serde_json::to_writer(&mut writer, value).map_err(|e| Error::invalid(e.to_string()))?;
            metadata_fingerprints.insert(
                key.to_owned(),
                crate::storage::digest_hex(writer.0.finalize().into()),
            );
        }
    }
    Ok(Descriptor {
        version,
        byte_order: if r.big { "big" } else { "little" }.to_string(),
        file_bytes: length,
        data_offset,
        architecture,
        tensor_bytes,
        tensors,
        encodings,
        metadata_fingerprints,
        header_sha256: crate::storage::digest_hex(r.hasher.finalize().into()),
        metadata,
    })
}

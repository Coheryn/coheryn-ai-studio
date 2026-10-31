//! Bounded native tensor-file inspection and owned shard storage. No model code executes.
use crate::{Cancellation, Error, ErrorCode, Result, cpu, gguf::Tensor, memory, storage::Storage};
use serde::{
    Deserialize, Serialize,
    de::{MapAccess, Visitor},
};
use serde_json::value::RawValue;
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    marker::PhantomData,
    path::{Path, PathBuf},
};

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub header_bytes: usize,
    pub index_bytes: usize,
    pub aggregate_header_bytes: usize,
    pub tensors: usize,
    pub shards: usize,
    pub rank: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            header_bytes: 8 << 20,
            index_bytes: 8 << 20,
            aggregate_header_bytes: 32 << 20,
            tensors: 100_000,
            shards: 128,
            rank: 8,
        }
    }
}
impl Limits {
    fn validate(self) -> Result<()> {
        if self.header_bytes == 0
            || self.header_bytes > 32 << 20
            || self.index_bytes == 0
            || self.index_bytes > 32 << 20
            || self.aggregate_header_bytes == 0
            || self.aggregate_header_bytes > 128 << 20
            || self.tensors == 0
            || self.tensors > 100_000
            || self.shards == 0
            || self.shards > 128
            || self.rank == 0
            || self.rank > 8
        {
            return Err(Error::request("invalid finite Safetensors limits"));
        }
        Ok(())
    }
}
fn invalid(message: impl Into<String>) -> Error {
    Error::new(
        ErrorCode::InvalidSafetensors,
        format!("invalid Safetensors: {}", message.into()),
    )
}
fn unsupported(message: impl Into<String>) -> Error {
    Error::new(ErrorCode::UnsupportedFeature, message)
}
// Shared numeric helpers retain the existing GGUF error contract. Preserve their
// resource/I/O/cancellation errors, while identifying this source format correctly.
pub(crate) fn source_error(mut error: Error) -> Error {
    if error.code == ErrorCode::InvalidGguf {
        error.code = ErrorCode::InvalidSafetensors;
    }
    error
}

/// Reject duplicate keys rather than losing evidence through last-key-wins decoding.
struct UniqueMap<T>(BTreeMap<String, T>);
impl<'de, T: Deserialize<'de>> Deserialize<'de> for UniqueMap<T> {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        struct MapVisitor<T>(PhantomData<T>);
        impl<'de, T: Deserialize<'de>> Visitor<'de> for MapVisitor<T> {
            type Value = UniqueMap<T>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("object with unique keys")
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut access: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut entries = BTreeMap::new();
                while let Some((key, value)) = access.next_entry::<String, T>()? {
                    if entries.insert(key, value).is_some() {
                        return Err(serde::de::Error::custom("duplicate object key"));
                    }
                }
                Ok(UniqueMap(entries))
            }
        }
        deserializer.deserialize_map(MapVisitor(PhantomData))
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Index {
    weight_map: UniqueMap<String>,
    #[serde(default)]
    metadata: Option<UniqueMap<Box<RawValue>>>,
}
#[derive(Clone, Debug, Serialize)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: String,
    pub offset: u64,
    pub bytes: u64,
}
#[derive(Clone, Debug, Serialize)]
pub struct FileInfo {
    pub file: String,
    pub bytes: u64,
    pub header_bytes: usize,
    pub header_sha256: String,
    pub tensors: Vec<TensorInfo>,
}
struct TensorFile {
    info: FileInfo,
    storage: Arc<Storage>,
    tensors: BTreeMap<String, Tensor>,
}

fn regular_file(path: &Path) -> Result<File> {
    let before = std::fs::symlink_metadata(path)?;
    if !before.is_file() || before.file_type().is_symlink() {
        return Err(invalid("regular, non-symlink file required"));
    }
    let file = File::open(path)?;
    let after = file.metadata()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if before.dev() != after.dev() || before.ino() != after.ino() || after.nlink() != 1 {
            return Err(invalid("file identity/link policy failed"));
        }
    }
    if before.len() != after.len() || before.modified().ok() != after.modified().ok() {
        return Err(Error::new(
            ErrorCode::ModelChanged,
            "tensor file changed while opening",
        ));
    }
    Ok(file)
}
pub(crate) fn bounded_bytes(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let mut file = regular_file(path)?;
    let initial = file.metadata()?;
    let count = usize::try_from(initial.len()).map_err(|_| invalid("file length overflow"))?;
    if count > limit {
        return Err(Error::resource("tensor metadata file exceeds byte limit"));
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(count)
        .map_err(|_| Error::resource("metadata reservation failed"))?;
    file.by_ref()
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)?;
    let after = file.metadata()?;
    if bytes.len() != count
        || after.len() != initial.len()
        || after.modified().ok() != initial.modified().ok()
    {
        return Err(Error::new(
            ErrorCode::ModelChanged,
            "metadata changed during read",
        ));
    }
    Ok(bytes)
}

/// Configuration is inert, bounded JSON; duplicate semantic keys are errors.
pub(crate) fn configuration(bytes: &[u8]) -> Result<serde_json::Value> {
    fn decode(raw: &RawValue, depth: usize) -> Result<serde_json::Value> {
        if depth > 32 {
            return Err(invalid("configuration nesting exceeds limit"));
        }
        match raw.get().as_bytes().first() {
            Some(b'{') => {
                let map: UniqueMap<Box<RawValue>> =
                    serde_json::from_str(raw.get()).map_err(|e| invalid(e.to_string()))?;
                if map.0.len() > 4096 {
                    return Err(invalid("configuration object exceeds key limit"));
                }
                let mut result = serde_json::Map::new();
                for (key, value) in map.0 {
                    result.insert(key, decode(&value, depth + 1)?);
                }
                Ok(serde_json::Value::Object(result))
            }
            Some(b'[') => {
                let items: Vec<Box<RawValue>> =
                    serde_json::from_str(raw.get()).map_err(|e| invalid(e.to_string()))?;
                if items.len() > 4096 {
                    return Err(invalid("configuration array exceeds item limit"));
                }
                Ok(serde_json::Value::Array(
                    items
                        .iter()
                        .map(|v| decode(v, depth + 1))
                        .collect::<Result<_>>()?,
                ))
            }
            _ => serde_json::from_str(raw.get()).map_err(|e| invalid(e.to_string())),
        }
    }
    if bytes.len() > 1 << 20 {
        return Err(Error::resource("configuration exceeds one MiB"));
    }
    let raw: Box<RawValue> = serde_json::from_slice(bytes).map_err(|e| invalid(e.to_string()))?;
    decode(&raw, 0)
}
/// Inspect a bounded header without constructing storage or reading tensor payloads.
/// `file_bytes` is the caller's known complete length; this is metadata inspection,
/// not proof that a remote/untrusted declared payload exists or has valid values.
pub fn inspect_reader(
    mut file: impl Read,
    bytes: u64,
    limits: Limits,
    cancel: &Cancellation,
) -> Result<FileInfo> {
    limits.validate()?;
    cancel.check()?;
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)
        .map_err(|_| invalid("truncated header length"))?;
    let n = usize::try_from(u64::from_le_bytes(prefix))
        .map_err(|_| invalid("header length overflow"))?;
    let data = 8u64
        .checked_add(n as u64)
        .ok_or_else(|| invalid("data offset overflow"))?;
    if n == 0 || n > limits.header_bytes || data > bytes {
        return Err(invalid("header exceeds file or byte limit"));
    }
    let mut header = Vec::new();
    header
        .try_reserve_exact(n)
        .map_err(|_| Error::resource("header reservation failed"))?;
    header.resize(n, 0);
    file.read_exact(&mut header)
        .map_err(|_| invalid("truncated header"))?;

    if header.first() != Some(&b'{') {
        return Err(invalid("header must start with an object"));
    }
    let values: UniqueMap<Box<RawValue>> =
        serde_json::from_slice(&header).map_err(|e| invalid(e.to_string()))?;
    if values.0.len() > limits.tensors.saturating_add(1) {
        return Err(Error::resource("tensor count exceeds limit"));
    }
    let mut info = Vec::new();
    let mut ranges = Vec::new();
    for (name, value) in values.0 {
        cancel.check()?;
        if name == "__metadata__" {
            let metadata: UniqueMap<String> =
                serde_json::from_str(value.get()).map_err(|e| invalid(e.to_string()))?;
            if metadata.0.len() > 4096 {
                return Err(Error::resource("metadata count exceeds limit"));
            }
            continue;
        }
        if name.is_empty() || name.len() > 1024 || name.contains('\0') {
            return Err(invalid("invalid tensor name"));
        }
        let h: Header = serde_json::from_str(value.get()).map_err(|e| invalid(e.to_string()))?;
        if h.shape.len() > limits.rank {
            return Err(invalid("tensor rank exceeds limit"));
        }
        let width = match h.dtype.as_str() {
            "F32" => 4,
            "F16" => 2,
            "BF16" => 2,
            _ => {
                return Err(unsupported(format!(
                    "unsupported Safetensors dtype {}",
                    h.dtype
                )));
            }
        };
        let elements = h
            .shape
            .iter()
            .try_fold(1u64, |n, d| n.checked_mul(*d))
            .ok_or_else(|| invalid("shape overflow"))?;
        let size = elements
            .checked_mul(width)
            .ok_or_else(|| invalid("tensor byte overflow"))?;
        let [begin, end] = h.data_offsets;
        if end.checked_sub(begin) != Some(size) || end > bytes - data {
            return Err(invalid("tensor range or shape mismatch"));
        }
        ranges.push((begin, end));
        let offset = data
            .checked_add(begin)
            .ok_or_else(|| invalid("tensor offset overflow"))?;
        info.push(TensorInfo {
            name: name.clone(),
            shape: h.shape.clone(),
            dtype: h.dtype,
            offset,
            bytes: size,
        });
    }
    if info.len() > limits.tensors {
        return Err(Error::resource("tensor count exceeds limit"));
    }
    ranges.sort_unstable();
    let mut end = 0;
    for (begin, next) in ranges {
        if begin != end {
            return Err(invalid("tensor overlap or unindexed hole"));
        }
        end = next;
    }
    if end != bytes - data {
        return Err(invalid("unindexed tensor payload"));
    }
    let mut hash = Sha256::new();
    hash.update(prefix);
    hash.update(&header);
    Ok(FileInfo {
        file: "<reader>".into(),
        bytes,
        header_bytes: n,
        header_sha256: crate::storage::digest_hex(hash.finalize().into()),
        tensors: info,
    })
}
fn tensor_file(path: &Path, limits: Limits, cancel: &Cancellation) -> Result<TensorFile> {
    cancel.check()?;
    let mut file = regular_file(path)?;
    let initial = file.metadata()?;
    let bytes = initial.len();
    let mut info = inspect_reader(&mut file, bytes, limits, cancel)?;
    let after = file.metadata()?;
    if after.len() != bytes || after.modified().ok() != initial.modified().ok() {
        return Err(Error::new(
            ErrorCode::ModelChanged,
            "header changed during inspection",
        ));
    }
    info.file = path
        .file_name()
        .and_then(|v| v.to_str())
        .ok_or_else(|| invalid("file name must be UTF-8"))?
        .into();
    let tensors = info
        .tensors
        .iter()
        .map(|t| {
            let dtype = match t.dtype.as_str() {
                "F32" => 0,
                "F16" => 1,
                "BF16" => 30,
                _ => return Err(unsupported("unexpected inspected tensor encoding")),
            };
            Ok((
                t.name.clone(),
                Tensor {
                    name: t.name.clone(),
                    dimensions: t.shape.iter().rev().copied().collect(),
                    dtype,
                    offset: t.offset,
                    bytes: t.bytes,
                },
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    Ok(TensorFile {
        info,
        storage: Arc::new(Storage::from_tensor_file(file)?),
        tensors,
    })
}
fn safe_basename(name: &str) -> Result<()> {
    if name.is_empty()
        || name.len() > 255
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
        || name == "."
        || name == ".."
        || !name.ends_with(".safetensors")
    {
        return Err(invalid("unsafe shard basename"));
    }
    let stem = name.split('.').next().unwrap_or("").to_ascii_uppercase();
    if matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || (stem.len() == 4
            && (stem.starts_with("COM") || stem.starts_with("LPT"))
            && matches!(stem.as_bytes()[3], b'1'..=b'9'))
    {
        return Err(invalid("reserved shard basename"));
    }
    Ok(())
}

/// A whole-tensor shard set. Every tensor is tied to its owned file and checked range.
pub struct TensorSet {
    entrypoint: PathBuf,
    index_identity: Option<(usize, String)>,
    root: PathBuf,
    files: Vec<TensorFile>,
    lookup: BTreeMap<String, (usize, Tensor)>,
    owner: Arc<()>,
    views: Vec<Storage>,
    gguf: bool,
}
pub(crate) struct BoundTensor {
    file: usize,
    tensor: Tensor,
    owner: Arc<()>,
}
impl TensorSet {
    pub fn open(path: impl AsRef<Path>, limits: Limits, cancel: &Cancellation) -> Result<Self> {
        limits.validate()?;
        let path = path.as_ref();
        if std::fs::symlink_metadata(path)?.file_type().is_symlink() {
            return Err(invalid("checkpoint symlink rejected"));
        }
        let selected = if path.is_dir() {
            let index = path.join("model.safetensors.index.json");
            let single = path.join("model.safetensors");
            if index.exists() && single.exists() {
                return Err(invalid(
                    "ambiguous checkpoint directory; select an explicit entrypoint",
                ));
            }
            if index.exists() { index } else { single }
        } else {
            path.to_path_buf()
        };
        let root = selected
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."))
            .canonicalize()?;
        let mut index_identity = None;
        let mut files = Vec::new();
        let mut lookup = BTreeMap::new();
        if selected.extension().is_some_and(|x| x == "safetensors") {
            let f = tensor_file(&selected, limits, cancel)?;
            if f.info.header_bytes > limits.aggregate_header_bytes {
                return Err(Error::resource("aggregate header limit exceeded"));
            }
            for (name, tensor) in &f.tensors {
                lookup.insert(name.clone(), (0, tensor.clone()));
            }
            files.push(f);
        } else {
            if !selected
                .file_name()
                .is_some_and(|n| n == "model.safetensors.index.json")
            {
                return Err(invalid(
                    "explicit Safetensors file or model.safetensors.index.json required",
                ));
            }
            let index_data = bounded_bytes(&selected, limits.index_bytes)?;
            index_identity = Some((
                index_data.len(),
                crate::storage::digest_hex(Sha256::digest(&index_data).into()),
            ));
            let index: Index =
                serde_json::from_slice(&index_data).map_err(|e| invalid(e.to_string()))?;
            if index.weight_map.0.is_empty() || index.weight_map.0.len() > limits.tensors {
                return Err(invalid("index tensor count outside limits"));
            }
            let mut names = BTreeSet::new();
            let mut normalized = BTreeSet::new();
            for name in index.weight_map.0.values() {
                safe_basename(name)?;
                names.insert(name.clone());
            }
            if names.len() > limits.shards {
                return Err(Error::resource("shard count exceeds limit"));
            }
            let mut header_bytes = 0usize;
            let mut bytes = 0u64;
            for name in names {
                cancel.check()?;
                if !normalized.insert(name.to_lowercase()) {
                    return Err(invalid("colliding shard names"));
                }
                let f = tensor_file(&root.join(&name), limits, cancel)?;
                header_bytes = header_bytes
                    .checked_add(f.info.header_bytes)
                    .filter(|v| *v <= limits.aggregate_header_bytes)
                    .ok_or_else(|| Error::resource("aggregate header limit exceeded"))?;
                for (key, tensor) in &f.tensors {
                    if index.weight_map.0.get(key) != Some(&name)
                        || lookup
                            .insert(key.clone(), (files.len(), tensor.clone()))
                            .is_some()
                    {
                        return Err(invalid("shard/index tensor coverage mismatch"));
                    }
                    bytes = bytes
                        .checked_add(tensor.bytes)
                        .ok_or_else(|| invalid("set byte overflow"))?;
                }
                files.push(f);
            }
            if lookup.len() != index.weight_map.0.len() {
                return Err(invalid("missing indexed tensor"));
            }
            if let Some(metadata) = index.metadata
                && let Some(total) = metadata.0.get("total_size")
            {
                let expected: u64 =
                    serde_json::from_str(total.get()).map_err(|_| invalid("invalid total_size"))?;
                if expected != bytes {
                    return Err(invalid("index total_size mismatch"));
                }
            }
        }
        cancel.check()?;
        Ok(Self {
            entrypoint: root.join(
                selected
                    .file_name()
                    .ok_or_else(|| invalid("missing entrypoint filename"))?,
            ),
            index_identity,
            root,
            files,
            lookup,
            owner: Arc::new(()),
            views: Vec::new(),
            gguf: false,
        })
    }
    /// Bind architecture-owned aliases to checked GGUF ranges. Physical files stay
    /// independent; the optional temporal patch is a bounded byte view over one file.
    pub(crate) fn from_gguf(
        entrypoint: &Path,
        parts: &[(&Path, &crate::Descriptor)],
        aliases: Vec<(usize, Tensor)>,
        patch: Option<(usize, Tensor, Tensor, Tensor)>,
        cancel: &Cancellation,
    ) -> Result<Self> {
        let root = entrypoint
            .parent()
            .unwrap_or(Path::new("."))
            .canonicalize()?;
        let mut files = Vec::new();
        for (path, descriptor) in parts {
            cancel.check()?;
            if std::fs::symlink_metadata(path)?.file_type().is_symlink()
                || descriptor.byte_order != "little"
            {
                return Err(Error::invalid(
                    "GGUF adapter requires regular little-endian components",
                ));
            }
            let storage = Arc::new(Storage::open(path, descriptor)?);
            let tensors = descriptor
                .tensors
                .iter()
                .map(|t| {
                    Ok(TensorInfo {
                        name: t.name.clone(),
                        shape: t.dimensions.iter().rev().copied().collect(),
                        dtype: crate::gguf::encoding(t.dtype)
                            .ok_or_else(|| Error::invalid("unknown encoding"))?
                            .0
                            .into(),
                        offset: descriptor.data_offset + t.offset,
                        bytes: t.bytes,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            files.push(TensorFile {
                info: FileInfo {
                    file: path
                        .file_name()
                        .ok_or_else(|| Error::invalid("missing component name"))?
                        .to_string_lossy()
                        .into_owned(),
                    bytes: descriptor.file_bytes,
                    header_bytes: descriptor.data_offset as usize,
                    header_sha256: descriptor.header_sha256.clone(),
                    tensors,
                },
                storage,
                tensors: BTreeMap::new(),
            });
        }
        let mut lookup = BTreeMap::new();
        for (file, mut tensor) in aliases {
            let descriptor = parts
                .get(file)
                .ok_or_else(|| Error::invalid("invalid component index"))?
                .1;
            tensor.offset = descriptor
                .data_offset
                .checked_add(tensor.offset)
                .ok_or_else(|| Error::invalid("GGUF offset overflow"))?;
            if tensor
                .offset
                .checked_add(tensor.bytes)
                .is_none_or(|end| end > descriptor.file_bytes)
                || lookup.insert(tensor.name.clone(), (file, tensor)).is_some()
            {
                return Err(Error::invalid("invalid or duplicate bound GGUF tensor"));
            }
        }
        let mut views = Vec::new();
        if let Some((file, left, right, mut logical)) = patch {
            let descriptor = parts
                .get(file)
                .ok_or_else(|| Error::invalid("invalid patch component"))?
                .1;
            let rows = *left
                .dimensions
                .last()
                .ok_or_else(|| Error::invalid("empty patch shape"))?;
            if left.dimensions != right.dimensions
                || left.dimensions.len() != 4
                || left.dimensions[..3] != [16, 16, 3]
                || left.dtype != right.dtype
                || !matches!(left.dtype, 0 | 1 | 30)
                || left.bytes != right.bytes
            {
                return Err(Error::invalid("unsupported split temporal patch layout"));
            }
            views.push(Storage::temporal_patch(
                files[file].storage.clone(),
                [
                    descriptor.data_offset + left.offset,
                    descriptor.data_offset + right.offset,
                ],
                rows,
                256,
                3,
                if left.dtype == 0 { 4 } else { 2 },
            )?);
            logical.offset = 0;
            if lookup
                .insert(logical.name.clone(), (files.len(), logical))
                .is_some()
            {
                return Err(Error::invalid("duplicate patch alias"));
            }
        }
        Ok(Self {
            entrypoint: entrypoint.canonicalize()?,
            index_identity: None,
            root,
            files,
            lookup,
            owner: Arc::new(()),
            views,
            gguf: true,
        })
    }
    pub(crate) fn is_gguf(&self) -> bool {
        self.gguf
    }
    pub(crate) fn source_error(&self, error: Error) -> Error {
        if self.gguf {
            error
        } else {
            source_error(error)
        }
    }
    pub fn entrypoint(&self) -> &Path {
        &self.entrypoint
    }
    pub fn index_identity(&self) -> Option<&(usize, String)> {
        self.index_identity.as_ref()
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn files(&self) -> Vec<&FileInfo> {
        self.files.iter().map(|f| &f.info).collect()
    }
    pub fn tensor_bytes(&self) -> Result<u64> {
        self.lookup.values().try_fold(0u64, |sum, (_, tensor)| {
            sum.checked_add(tensor.bytes)
                .ok_or_else(|| Error::resource("tensor set byte overflow"))
        })
    }
    pub fn tensor_count(&self) -> usize {
        self.lookup.len()
    }
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.lookup.get(name).map(|(_, t)| t)
    }
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.lookup.keys().map(String::as_str)
    }
    pub fn unchanged(&self) -> Result<()> {
        for f in &self.files {
            f.storage.unchanged()?;
        }
        Ok(())
    }
    pub(crate) fn source(&self, name: &str) -> Result<(&Storage, &Tensor)> {
        let (file, tensor) = self
            .lookup
            .get(name)
            .ok_or_else(|| invalid(format!("missing tensor {name}")))?;
        Ok((self.storage(*file)?, tensor))
    }
    pub(crate) fn bind(&self, name: &str) -> Result<BoundTensor> {
        let (file, tensor) = self
            .lookup
            .get(name)
            .ok_or_else(|| invalid(format!("missing tensor {name}")))?;
        Ok(BoundTensor {
            file: *file,
            tensor: tensor.clone(),
            owner: self.owner.clone(),
        })
    }
    pub(crate) fn bound<'a>(
        &'a self,
        binding: &'a BoundTensor,
    ) -> Result<(&'a Storage, &'a Tensor)> {
        if !Arc::ptr_eq(&self.owner, &binding.owner) {
            return Err(invalid("tensor belongs to another storage set"));
        }
        Ok((self.storage(binding.file)?, &binding.tensor))
    }
    fn storage(&self, index: usize) -> Result<&Storage> {
        if index < self.files.len() {
            Ok(self.files[index].storage.as_ref())
        } else {
            self.views
                .get(index - self.files.len())
                .ok_or_else(|| invalid("invalid tensor storage view"))
        }
    }
    /// Explicit deep validation scans finite values in bounded tiles without a model/session.
    pub fn validate_payloads(&self, cancel: &Cancellation) -> Result<()> {
        let mut bytes = vec![0u8; 256 << 10];
        let mut floats = memory::zeros(bytes.len() / 2)?;
        for name in self.names() {
            let (storage, tensor) = self.source(name)?;
            let width = usize::try_from(tensor.dimensions.first().copied().unwrap_or(1))
                .map_err(|_| invalid("tensor row width overflow"))?;
            let row_bytes = crate::quant::row_bytes(tensor.dtype, width)?;
            let chunk_rows = (bytes.len() / row_bytes).min(floats.len() / width);
            if chunk_rows == 0 {
                return Err(Error::resource("tensor row exceeds deep-validation tile"));
            }
            let mut position = 0;
            while position < tensor.bytes {
                cancel.check()?;
                storage.unchanged()?;
                let count =
                    ((tensor.bytes - position).min((chunk_rows * row_bytes) as u64)) as usize;
                storage
                    .read(tensor.offset + position, &mut bytes[..count])
                    .map_err(|error| self.source_error(error))?;
                cpu::decode(
                    &bytes[..count],
                    tensor.dtype,
                    &mut floats[..count / row_bytes * width],
                )
                .map_err(|error| self.source_error(error))?;
                position += count as u64;
            }
        }
        cancel.check()?;
        self.unchanged()
    }
    pub fn read_vector(&self, name: &str) -> Result<Vec<f32>> {
        let (storage, tensor) = self.source(name)?;
        let count = tensor.dimensions.iter().try_fold(1usize, |a, b| {
            usize::try_from(*b)
                .ok()
                .and_then(|b| a.checked_mul(b))
                .ok_or_else(|| invalid("vector length overflow"))
        })?;
        if tensor.dimensions.len() > 3 || tensor.bytes > 256 << 10 {
            return Err(Error::resource("small-tensor read exceeds bound"));
        }
        let size = usize::try_from(tensor.bytes).map_err(|_| invalid("read overflow"))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size)
            .map_err(|_| Error::resource("small tensor allocation failed"))?;
        bytes.resize(size, 0);
        let mut out = memory::zeros(count)?;
        storage.unchanged()?;
        storage.read(tensor.offset, &mut bytes)?;
        cpu::decode(&bytes, tensor.dtype, &mut out).map_err(|error| self.source_error(error))?;
        Ok(out)
    }
}

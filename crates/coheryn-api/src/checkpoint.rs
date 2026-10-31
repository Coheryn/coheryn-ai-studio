//! Inert, manifest-first checkpoint transport and private managed source sets.
//! No archive extraction, remote filesystem paths, or model-provided code.
use crate::{
    auth::{private_directory, private_file},
    error::{ApiError, Result},
    registry::{ModelRecord, Registry, validate_hash},
    service::Service,
};
use coheryn_engine::{Cancellation, Error, ErrorCode, catalog};
use futures_util::{Stream, TryStreamExt};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
    time::Duration,
};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};

pub const MAGIC: &[u8; 12] = b"COHERYNCP1\r\n";
pub const MANIFEST_LIMIT: usize = 48 << 10;
pub const FILE_LIMIT: usize = 132;
const TILE: usize = 256 << 10;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CheckpointFile {
    pub name: String,
    pub bytes: u64,
    pub sha256: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CheckpointManifest {
    pub schema_version: u32,
    pub entrypoint: String,
    pub adapter: String,
    #[serde(deserialize_with = "crate::bounded::sequence::<_,_,132>")]
    pub files: Vec<CheckpointFile>,
}
fn portable(name: &str) -> bool {
    if name.is_empty()
        || name.len() > 128
        || name.starts_with('.')
        || name.ends_with('.')
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
    {
        return false;
    }
    let stem = name
        .split('.')
        .next()
        .unwrap_or_default()
        .to_ascii_uppercase();
    !matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        && !(stem.len() == 4
            && (stem.starts_with("COM") || stem.starts_with("LPT"))
            && matches!(stem.as_bytes()[3], b'1'..=b'9'))
}
fn allowed(name: &str) -> bool {
    portable(name)
        && (name.ends_with(".safetensors")
            || name.ends_with(".gguf")
            || matches!(
                name,
                "model.safetensors.index.json"
                    | "config.json"
                    | "tokenizer.json"
                    | "chat_template.jinja"
                    | "tokenizer_config.json"
                    | "preprocessor_config.json"
                    | "special_tokens_map.json"
                    | "LICENSE"
                    | "NOTICE"
            ))
}
impl CheckpointManifest {
    pub fn from_local(path: &Path, cancel: &Cancellation) -> Result<(Self, PathBuf)> {
        let selected = if path.is_dir() {
            if path.join("model.safetensors.index.json").is_file() {
                path.join("model.safetensors.index.json")
            } else {
                path.join("model.safetensors")
            }
        } else {
            path.to_owned()
        };
        let root = selected
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."))
            .to_owned();
        regular_directory(&root)?;
        let inspection =
            coheryn_engine::adapters::Registry::default().inspect(&selected, cancel)?;
        inspection.require_compatible()?;
        let d = inspection.descriptor();
        if d.format != catalog::Format::Safetensors
            && d.text_adapter
                .as_ref()
                .is_none_or(|a| a.adapter != "qwen35-gguf-cpu-v1")
        {
            return Err(ApiError::bad(
                "checkpoint transport requires a supported native source set",
            ));
        }
        let adapter = d
            .text_adapter
            .as_ref()
            .ok_or_else(|| ApiError::bad("checkpoint has no native text adapter"))?
            .adapter
            .to_owned();
        let mut names: BTreeSet<_> = d.source_files.iter().map(|f| f.name.clone()).collect();
        // Preserve relevant attribution/configuration without scanning or copying the source tree.
        for name in [
            "LICENSE",
            "NOTICE",
            "tokenizer_config.json",
            "preprocessor_config.json",
            "special_tokens_map.json",
        ] {
            if root.join(name).try_exists()? {
                names.insert(name.into());
            }
        }
        if names.len() > FILE_LIMIT {
            return Err(Error::resource("checkpoint file count exceeds limit").into());
        }
        let mut files = Vec::new();
        for name in names {
            if !allowed(&name) {
                return Err(ApiError::bad("unsupported checkpoint asset name"));
            }
            let asset = root.join(&name);
            let bytes = std::fs::symlink_metadata(&asset)?.len();
            files.push(CheckpointFile {
                name,
                bytes,
                sha256: "0".repeat(64),
            });
        }
        let mut manifest = Self {
            schema_version: 1,
            entrypoint: selected
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or_else(|| ApiError::bad("invalid checkpoint entrypoint"))?
                .into(),
            adapter,
            files,
        };
        manifest.validate()?;
        for file in &mut manifest.files {
            file.sha256 = hash_file(&root.join(&file.name), cancel)?;
        }
        Ok((manifest, root))
    }
    pub fn validate(&self) -> Result<u64> {
        if self.schema_version != 1
            || self.files.is_empty()
            || self.files.len() > FILE_LIMIT
            || self.adapter.is_empty()
            || self.adapter.len() > 128
            || !self
                .adapter
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
            || !allowed(&self.entrypoint)
            || !(self.entrypoint.ends_with(".safetensors")
                || self.entrypoint.ends_with(".gguf")
                || self.entrypoint == "model.safetensors.index.json")
        {
            return Err(ApiError::bad(
                "invalid checkpoint manifest version, entrypoint or adapter",
            ));
        }
        let mut previous = "";
        let mut folded = BTreeSet::new();
        let mut bytes = 0u64;
        for file in &self.files {
            if !allowed(&file.name)
                || file.name.as_str() <= previous
                || !folded.insert(file.name.to_ascii_lowercase())
                || file.bytes == 0
                || file.sha256.bytes().any(|b| b.is_ascii_uppercase())
            {
                return Err(ApiError::bad(
                    "checkpoint files must be sorted, unique, portable inert assets with lowercase hashes",
                ));
            }
            validate_hash(&file.sha256)?;
            let cap = match file.name.as_str() {
                "tokenizer.json" => 16 << 20,
                "chat_template.jinja" => 16 << 10,
                n if n.ends_with(".safetensors") || n.ends_with(".gguf") => u64::MAX,
                _ => 1 << 20,
            };
            if file.bytes > cap {
                return Err(Error::resource("checkpoint asset exceeds its byte limit").into());
            }
            previous = &file.name;
            bytes = bytes
                .checked_add(file.bytes)
                .ok_or_else(|| Error::resource("checkpoint byte overflow"))?;
        }
        if !self.files.iter().any(|f| f.name == self.entrypoint)
            || !self.files.iter().any(|f| f.name == "config.json")
        {
            return Err(ApiError::bad(
                "checkpoint entrypoint/configuration is missing",
            ));
        }
        if serde_json::to_vec(self)
            .map_err(|_| ApiError::bad("manifest serialization failed"))?
            .len()
            > MANIFEST_LIMIT
        {
            return Err(Error::resource("checkpoint manifest exceeds byte limit").into());
        }
        Ok(bytes)
    }
    /// Full payload identity. Unlike catalog metadata_identity, this binds every file byte.
    pub fn identity(&self) -> Result<String> {
        self.validate()?;
        let mut hash = Sha256::new();
        hash.update(b"coheryn-checkpoint-v1\0");
        hash.update(
            serde_json::to_vec(self).map_err(|_| ApiError::bad("manifest serialization failed"))?,
        );
        Ok(coheryn_engine::storage::digest_hex(hash.finalize().into()))
    }
    pub fn verify(&self, root: &Path, deep: bool, cancel: &Cancellation) -> Result<()> {
        self.validate()?;
        regular_directory(root)?;
        let expected: BTreeSet<_> = self.files.iter().map(|f| f.name.as_str()).collect();
        let mut found = BTreeSet::new();
        for entry in std::fs::read_dir(root)? {
            let entry = entry?;
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| ApiError::bad("non-UTF8 checkpoint asset"))?;
            if found.len() >= FILE_LIMIT || !expected.contains(name.as_str()) || !found.insert(name)
            {
                return Err(ApiError::bad("unexpected checkpoint directory entry"));
            }
        }
        if found.len() != expected.len() {
            return Err(ApiError::bad("incomplete checkpoint source set"));
        }
        for file in &self.files {
            cancel.check()?;
            let path = root.join(&file.name);
            let meta = std::fs::symlink_metadata(&path)?;
            if !meta.is_file() || meta.file_type().is_symlink() || meta.len() != file.bytes {
                return Err(Error::new(
                    ErrorCode::ModelChanged,
                    "managed checkpoint file type/size changed",
                )
                .into());
            }
            if deep && hash_file(&path, cancel)? != file.sha256 {
                return Err(Error::new(
                    ErrorCode::ModelChanged,
                    "managed checkpoint file SHA-256 mismatch",
                )
                .into());
            }
        }
        Ok(())
    }
    pub(crate) fn bind(&self, root: &Path, cancel: &Cancellation) -> Result<catalog::Inspection> {
        let inspection = coheryn_engine::adapters::Registry::default()
            .inspect(root.join(&self.entrypoint), cancel)?;
        inspection.require_compatible()?;
        let d = inspection.descriptor();
        if !(d.format == catalog::Format::Safetensors
            || (d.format == catalog::Format::Gguf && self.adapter == "qwen35-gguf-cpu-v1"))
            || d.text_adapter
                .as_ref()
                .is_none_or(|a| a.adapter != self.adapter)
        {
            return Err(ApiError::bad(
                "checkpoint native adapter does not match manifest",
            ));
        }
        // All execution inputs must be present; extra tensor/index files are never inert assets.
        let required: BTreeSet<_> = d.source_files.iter().map(|f| f.name.as_str()).collect();
        if required
            .iter()
            .any(|name| !self.files.iter().any(|f| f.name == *name))
            || self.files.iter().any(|f| {
                (f.name.ends_with(".safetensors")
                    || f.name.ends_with(".gguf")
                    || f.name.ends_with(".index.json"))
                    && !required.contains(f.name.as_str())
            })
        {
            return Err(ApiError::bad(
                "checkpoint manifest and selected native source set differ",
            ));
        }
        Ok(inspection)
    }
}
pub fn hash_file(path: &Path, cancel: &Cancellation) -> Result<String> {
    use std::io::Read;
    let meta = std::fs::symlink_metadata(path)?;
    if !meta.is_file() || meta.file_type().is_symlink() {
        return Err(ApiError::bad("checkpoint asset must be a regular file"));
    }
    let mut file = std::fs::File::open(path)?;
    let mut buffer = vec![0; TILE];
    let mut hash = Sha256::new();
    let mut read = 0u64;
    loop {
        cancel.check()?;
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        read = read
            .checked_add(n as u64)
            .filter(|n| *n <= meta.len())
            .ok_or_else(|| {
                Error::new(ErrorCode::ModelChanged, "checkpoint changed while hashing")
            })?;
        hash.update(&buffer[..n]);
    }
    let after = file.metadata()?;
    if read != meta.len()
        || after.len() != meta.len()
        || after.modified().ok() != meta.modified().ok()
    {
        return Err(Error::new(ErrorCode::ModelChanged, "checkpoint changed while hashing").into());
    }
    Ok(coheryn_engine::storage::digest_hex(hash.finalize().into()))
}
fn regular_directory(path: &Path) -> Result<()> {
    let meta = std::fs::symlink_metadata(path)?;
    if !meta.is_dir() || meta.file_type().is_symlink() {
        return Err(ApiError::bad("checkpoint root must be a regular directory"));
    }
    Ok(())
}
pub(crate) fn directory_bytes(path: &Path) -> Result<u64> {
    regular_directory(path)?;
    let mut total = 0u64;
    for (i, entry) in std::fs::read_dir(path)?.enumerate() {
        if i >= FILE_LIMIT {
            return Err(Error::resource("checkpoint file count exceeds limit").into());
        }
        let entry = entry?;
        let meta = std::fs::symlink_metadata(entry.path())?;
        if !meta.is_file() || meta.file_type().is_symlink() {
            return Err(ApiError::bad("checkpoint contains a non-regular file"));
        }
        total = total
            .checked_add(meta.len())
            .ok_or_else(|| Error::resource("checkpoint disk byte overflow"))?;
    }
    Ok(total)
}
pub(crate) fn remove_directory(path: &Path) -> Result<()> {
    directory_bytes(path)?; // bounded, flat and no links; never recursively follow paths.
    for entry in std::fs::read_dir(path)? {
        std::fs::remove_file(entry?.path())?;
    }
    std::fs::remove_dir(path)?;
    Ok(())
}
pub(crate) fn sync_directory(path: &Path) -> Result<()> {
    #[cfg(unix)]
    std::fs::File::open(path)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}
struct Staging(PathBuf);
impl Drop for Staging {
    fn drop(&mut self) {
        if self.0.exists()
            && remove_directory(&self.0).is_err()
            && let Ok(runtime) = tokio::runtime::Handle::try_current()
        {
            let path = self.0.clone();
            runtime.spawn(async move {
                for _ in 0..5 {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    if !path.exists() || remove_directory(&path).is_ok() {
                        break;
                    }
                }
            });
        }
    }
}
async fn read<R: AsyncRead + Unpin>(
    reader: &mut R,
    buffer: &mut [u8],
    service: &Service,
) -> Result<usize> {
    tokio::select! {
        _ = service.shutdown.cancelled() => Err(Error::new(ErrorCode::Cancelled,"checkpoint import cancelled by shutdown").into()),
        n = tokio::time::timeout(Duration::from_secs(30), reader.read(buffer)) => n.map_err(|_| ApiError::bad("checkpoint import idle timeout"))?.map_err(Into::into),
    }
}
async fn exact<R: AsyncRead + Unpin>(
    reader: &mut R,
    mut buffer: &mut [u8],
    service: &Service,
) -> Result<()> {
    while !buffer.is_empty() {
        let n = read(reader, buffer, service).await?;
        if n == 0 {
            return Err(ApiError::bad("truncated checkpoint stream"));
        }
        buffer = &mut buffer[n..];
    }
    Ok(())
}
impl Service {
    /// Caller holds the administration mutex, including through publication.
    pub async fn ingest_checkpoint<S, E>(
        self: &std::sync::Arc<Self>,
        name: String,
        expected: Option<String>,
        stream: S,
    ) -> Result<ModelRecord>
    where
        S: Stream<Item = std::result::Result<axum::body::Bytes, E>> + Send,
        E: std::fmt::Display,
    {
        self.registry.check_new(&name)?;
        if let Some(hash) = &expected {
            validate_hash(hash)?;
        }
        let _buffers = self
            .budget
            .reserve((2 * TILE + 2 * MANIFEST_LIMIT) as u64)?;
        let stream =
            Box::pin(stream.map_err(|_| std::io::Error::other("checkpoint transport interrupted")));
        let mut reader = tokio_util::io::StreamReader::new(stream);
        let mut prefix = [0; 16];
        exact(&mut reader, &mut prefix, self).await?;
        if &prefix[..12] != MAGIC {
            return Err(ApiError::bad("invalid checkpoint transport magic"));
        }
        let length = u32::from_le_bytes(prefix[12..].try_into().expect("four bytes")) as usize;
        if length == 0 || length > MANIFEST_LIMIT {
            return Err(Error::resource("checkpoint manifest byte limit exceeded").into());
        }
        let mut bytes = vec![0; length];
        exact(&mut reader, &mut bytes, self).await?;
        let manifest: CheckpointManifest = serde_json::from_slice(&bytes)
            .map_err(|_| ApiError::bad("invalid checkpoint manifest JSON"))?;
        let size = manifest.validate()?;
        let identity = manifest.identity()?;
        if expected.is_some_and(|v| !v.eq_ignore_ascii_case(&identity)) {
            return Err(ApiError::bad("checkpoint source-set SHA-256 mismatch"));
        }
        let available = self
            .registry
            .disk_quota
            .saturating_sub(self.registry.disk_bytes()?)
            .min(self.registry.max_model_bytes)
            .min(fs2::available_space(self.registry.root())?.saturating_sub(64 << 20));
        if size > available {
            return Err(Error::resource(
                "checkpoint exceeds model limit, available disk or disk quota",
            )
            .into());
        }
        let staging = Staging(
            self.registry
                .root()
                .join("models")
                .join(format!("{}.checkpoint-part", uuid::Uuid::new_v4().simple())),
        );
        private_directory(&staging.0)?;
        let mut buffer = vec![0; TILE];
        for asset in &manifest.files {
            let mut file = tokio::fs::File::from_std(private_file(&staging.0.join(&asset.name))?);
            let mut remaining = asset.bytes;
            let mut hash = Sha256::new();
            while remaining != 0 {
                let n = remaining.min(TILE as u64) as usize;
                exact(&mut reader, &mut buffer[..n], self).await?;
                file.write_all(&buffer[..n]).await?;
                hash.update(&buffer[..n]);
                remaining -= n as u64;
            }
            file.flush().await?;
            file.sync_all().await?;
            drop(file);
            if coheryn_engine::storage::digest_hex(hash.finalize().into()) != asset.sha256 {
                return Err(ApiError::bad("checkpoint file SHA-256 mismatch"));
            }
        }
        if read(&mut reader, &mut buffer[..1], self).await? != 0 {
            return Err(ApiError::bad("trailing checkpoint stream bytes"));
        }
        let reservation = self
            .budget
            .reserve(coheryn_engine::qwen35::RESIDENT_METADATA_BYTES)?;
        let validation = manifest.clone();
        // Move staging ownership into the job: timeout/disconnect cannot remove its inputs
        // until validation and any worker-held file handles have retired.
        let (staging, manifest) = tokio::task::spawn_blocking(move || -> Result<_> {
            let _reservation = reservation;
            validation.verify(&staging.0, false, &Default::default())?;
            validation.bind(&staging.0, &Default::default())?;
            Ok((staging, validation))
        })
        .await??;
        if self.shutdown.is_cancelled() {
            return Err(Error::new(
                ErrorCode::Cancelled,
                "checkpoint publication cancelled by shutdown",
            )
            .into());
        }
        let record = ModelRecord {
            schema_version: 2,
            id: name,
            sha256: identity,
            bytes: size,
            source: None,
            components: None,
            checkpoint: Some(manifest),
        };
        sync_directory(&staging.0)?;
        self.registry.commit_checkpoint(&staging.0, &record)?;
        Ok(record)
    }
}
impl Registry {
    pub fn checkpoint_root(&self, id: &str) -> Result<PathBuf> {
        crate::registry::validate_id(id)?;
        Ok(self
            .root()
            .join("models")
            .join(format!("model-{id}.checkpoint")))
    }
    pub fn checkpoint_path(&self, record: &ModelRecord) -> Result<PathBuf> {
        let manifest = record
            .checkpoint
            .as_ref()
            .ok_or_else(|| ApiError::bad("managed model is not a checkpoint"))?;
        manifest.validate()?;
        Ok(self.checkpoint_root(&record.id)?.join(&manifest.entrypoint))
    }
    pub(crate) fn commit_checkpoint(&self, staging: &Path, record: &ModelRecord) -> Result<()> {
        self.check_new(&record.id)?;
        let target = self.checkpoint_root(&record.id)?;
        std::fs::rename(staging, &target)?;
        let result =
            sync_directory(&self.root().join("models")).and_then(|_| self.write_record(record));
        // Once the record pointer exists, preserve its complete inputs even if directory
        // synchronization fails. Restart must never observe a published partial set.
        if result.is_err()
            && !self
                .root()
                .join("records")
                .join(format!("model-{}.json", record.id))
                .exists()
        {
            let _ = remove_directory(&target);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{auth::Credentials, config::ServerConfig};
    use axum::{
        body::{Body, Bytes, to_bytes},
        http::{Request, StatusCode},
    };
    use futures_util::StreamExt;
    use serde_json::json;
    use tower::ServiceExt;

    fn manifest() -> CheckpointManifest {
        CheckpointManifest {
            schema_version: 1,
            entrypoint: "model.safetensors".into(),
            adapter: "native-test".into(),
            files: vec![
                CheckpointFile {
                    name: "config.json".into(),
                    bytes: 2,
                    sha256: coheryn_engine::storage::digest_hex(Sha256::digest(b"{}").into()),
                },
                CheckpointFile {
                    name: "model.safetensors".into(),
                    bytes: 4,
                    sha256: coheryn_engine::storage::digest_hex(Sha256::digest(b"test").into()),
                },
            ],
        }
    }
    fn config(path: &Path) -> ServerConfig {
        ServerConfig {
            data_dir: path.join("private"),
            max_memory: 512 << 20,
            ..Default::default()
        }
    }
    fn stage(registry: &Registry) -> Staging {
        let staging = Staging(
            registry
                .root()
                .join("models")
                .join(format!("{}.checkpoint-part", uuid::Uuid::new_v4().simple())),
        );
        private_directory(&staging.0).unwrap();
        std::fs::write(staging.0.join("config.json"), b"{}").unwrap();
        std::fs::write(staging.0.join("model.safetensors"), b"test").unwrap();
        staging
    }
    fn record(m: CheckpointManifest) -> ModelRecord {
        ModelRecord {
            schema_version: 2,
            id: "checkpoint".into(),
            sha256: m.identity().unwrap(),
            bytes: m.validate().unwrap(),
            source: None,
            components: None,
            checkpoint: Some(m),
        }
    }
    fn bundle(m: &CheckpointManifest, payload: &[u8]) -> Vec<u8> {
        let json = serde_json::to_vec(m).unwrap();
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&(json.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&json);
        bytes.extend_from_slice(payload);
        bytes
    }
    #[test]
    fn manifest_rejects_paths_collisions_overflows_and_unbounded_assets() {
        let m = manifest();
        assert_eq!(m.validate().unwrap(), 6);
        for name in [
            "../config.json",
            "/model.safetensors",
            "CON.safetensors",
            "model.safetensors.",
            "model\\x.safetensors",
            "run.py",
            "a:b.safetensors",
        ] {
            let mut bad = m.clone();
            bad.files[1].name = name.into();
            bad.entrypoint = name.into();
            assert!(bad.validate().is_err(), "{name}");
        }
        let mut bad = m.clone();
        bad.files.reverse();
        assert!(bad.validate().is_err());
        let mut bad = m.clone();
        bad.files[1] = bad.files[0].clone();
        assert!(bad.validate().is_err());
        let mut bad = m.clone();
        bad.files[1].bytes = u64::MAX;
        assert!(bad.validate().is_err());
        let mut bad = m.clone();
        bad.files[0].bytes = (1 << 20) + 1;
        assert!(bad.validate().is_err());
        let mut bad = m.clone();
        bad.files[0].sha256 = "A".repeat(64);
        assert!(bad.validate().is_err());
        let mut bad = m.clone();
        bad.adapter = "../code".into();
        assert!(bad.validate().is_err());
        let mut other = m.clone();
        other.files[1].sha256 = "0".repeat(64);
        assert_ne!(m.identity().unwrap(), other.identity().unwrap());
        let mut value = serde_json::to_value(&m).unwrap();
        value["files"] = json!(vec![value["files"][0].clone(); FILE_LIMIT + 1]);
        assert!(serde_json::from_value::<CheckpointManifest>(value).is_err());
    }
    #[test]
    fn registry_roundtrip_checks_payloads_reclaims_orphans_and_preserves_commits() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = config(dir.path());
        let registry = Registry::open(&cfg).unwrap();
        let staging = stage(&registry);
        let m = manifest();
        m.verify(&staging.0, true, &Default::default()).unwrap();
        let r = record(m.clone());
        registry.commit_checkpoint(&staging.0, &r).unwrap();
        assert_eq!(registry.disk_bytes().unwrap(), 6);
        assert_eq!(registry.get("checkpoint").unwrap().sha256, r.sha256);
        assert!(registry.check_new("checkpoint").is_err());
        let orphan = stage(&registry);
        let orphan_path = orphan.0.clone();
        std::mem::forget(orphan);
        let promoted = stage(&registry);
        let promoted_path = registry.checkpoint_root("orphan").unwrap();
        std::fs::rename(&promoted.0, &promoted_path).unwrap();
        drop(registry);
        let registry = Registry::open(&cfg).unwrap();
        assert!(!orphan_path.exists());
        assert!(!promoted_path.exists());
        assert_eq!(registry.list().unwrap().len(), 1);
        let root = registry.checkpoint_root("checkpoint").unwrap();
        std::fs::write(root.join("model.safetensors"), b"evil").unwrap();
        // Length-only listing is cheap; loading/validation must request full verification.
        assert!(registry.get("checkpoint").is_ok());
        assert_eq!(
            m.verify(&root, true, &Default::default())
                .unwrap_err()
                .error
                .code,
            ErrorCode::ModelChanged
        );
        std::fs::write(root.join("extra.json"), b"x").unwrap();
        assert!(registry.get("checkpoint").is_err());
        std::fs::remove_file(root.join("extra.json")).unwrap();
        std::fs::write(root.join("model.safetensors"), b"x").unwrap();
        assert!(registry.get("checkpoint").is_err());
        registry.remove("checkpoint").unwrap();
        assert_eq!(registry.disk_bytes().unwrap(), 0);
        assert!(registry.list().unwrap().is_empty());
    }
    #[cfg(unix)]
    #[test]
    fn managed_sets_reject_symlinks_without_following_them() {
        let dir = tempfile::tempdir().unwrap();
        let registry = Registry::open(&config(dir.path())).unwrap();
        let staging = stage(&registry);
        let path = staging.0.join("model.safetensors");
        std::fs::remove_file(&path).unwrap();
        std::os::unix::fs::symlink(dir.path(), &path).unwrap();
        assert!(
            manifest()
                .verify(&staging.0, false, &Default::default())
                .is_err()
        );
        assert!(remove_directory(&staging.0).is_err());
        assert!(dir.path().exists());
        std::fs::remove_file(&path).unwrap();
    }
    #[tokio::test]
    async fn quota_shutdown_and_interrupted_uploads_reclaim_staging_and_credit() {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = config(dir.path());
        cfg.max_model_bytes = 4;
        let service = Service::open(
            cfg,
            Credentials {
                inference_token: "i".repeat(64),
                admin_token: "a".repeat(64),
            },
        )
        .unwrap();
        let bytes = bundle(&manifest(), b"{}test");
        let stream = futures_util::stream::iter([Ok::<_, std::io::Error>(Bytes::from(bytes))]);
        let error = service
            .ingest_checkpoint("quota".into(), None, stream)
            .await
            .unwrap_err();
        assert_eq!(error.error.code, ErrorCode::ResourceExhausted);
        assert_eq!(service.registry.disk_bytes().unwrap(), 0);
        drop(service);
        let service = Service::open(
            config(dir.path()),
            Credentials {
                inference_token: "i".repeat(64),
                admin_token: "a".repeat(64),
            },
        )
        .unwrap();
        let prefix = bundle(&manifest(), b"{}");
        let stream = futures_util::stream::iter([Ok::<_, std::io::Error>(Bytes::from(prefix))])
            .chain(futures_util::stream::pending());
        let worker_service = service.clone();
        let worker = tokio::spawn(async move {
            worker_service
                .ingest_checkpoint("cancel".into(), None, stream)
                .await
        });
        // Wait for a file to exist, rather than relying on a scheduling delay.
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if std::fs::read_dir(service.registry.root().join("models"))
                    .unwrap()
                    .flatten()
                    .any(|e| e.path().join("config.json").is_file())
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        service.shutdown.cancel();
        let error = worker.await.unwrap().unwrap_err();
        assert_eq!(error.error.code, ErrorCode::Cancelled);
        tokio::time::timeout(Duration::from_secs(5), async {
            while service.registry.disk_bytes().unwrap() != 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        assert!(service.registry.list().unwrap().is_empty());
        assert_eq!(service.budget.stats().reserved, 0);
    }
    #[tokio::test]
    async fn import_is_authenticated_bounded_and_transactional_on_bad_streams() {
        let dir = tempfile::tempdir().unwrap();
        let service = Service::open(
            config(dir.path()),
            Credentials {
                inference_token: "i".repeat(64),
                admin_token: "a".repeat(64),
            },
        )
        .unwrap();
        let app = crate::http::router(service.clone());
        let m = manifest();
        for (token, bytes, expected) in [
            ('i', bundle(&m, b"{}test"), StatusCode::FORBIDDEN),
            ('a', vec![0; 16], StatusCode::BAD_REQUEST),
            ('a', bundle(&m, b"{}tes"), StatusCode::BAD_REQUEST),
            ('a', bundle(&m, b"{}evil"), StatusCode::BAD_REQUEST),
            ('a', bundle(&m, b"{}test!"), StatusCode::BAD_REQUEST),
            // Complete transport with unsupported native contents must still roll back.
            ('a', bundle(&m, b"{}test"), StatusCode::BAD_REQUEST),
        ] {
            let request = Request::post("/api/v1/models/import-checkpoint?name=test")
                .header(
                    "authorization",
                    format!("Bearer {}", token.to_string().repeat(64)),
                )
                .header("content-type", "application/vnd.coheryn.checkpoint.v1")
                .body(Body::from(bytes))
                .unwrap();
            let response = app.clone().oneshot(request).await.unwrap();
            let status = response.status();
            let body = to_bytes(response.into_body(), 1 << 20).await.unwrap();
            assert_eq!(status, expected, "{}", String::from_utf8_lossy(&body));
            assert!(service.registry.list().unwrap().is_empty());
            assert_eq!(service.registry.disk_bytes().unwrap(), 0);
        }
        let mut too_large = MAGIC.to_vec();
        too_large.extend_from_slice(&((MANIFEST_LIMIT + 1) as u32).to_le_bytes());
        let stream = futures_util::stream::iter([Ok::<_, std::io::Error>(Bytes::from(too_large))]);
        assert_eq!(
            service
                .ingest_checkpoint("large".into(), None, stream)
                .await
                .unwrap_err()
                .error
                .code,
            ErrorCode::ResourceExhausted
        );
        assert_eq!(service.budget.stats().reserved, 0);
    }
}

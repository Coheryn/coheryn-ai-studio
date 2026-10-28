use crate::{
    auth::{private_directory, private_file},
    config::ServerConfig,
    error::{ApiError, Result},
};
use coheryn_engine::{Error, ErrorCode};
use serde::{Deserialize, Serialize};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRecord {
    pub schema_version: u32,
    pub id: String,
    pub sha256: String,
    pub bytes: u64,
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub components: Option<ModelPair>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ComponentRef {
    pub model: String,
    pub sha256: String,
    pub bytes: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelPair {
    pub language: ComponentRef,
    pub vision: ComponentRef,
}
impl ModelPair {
    pub fn identity(&self) -> String {
        use sha2::{Digest, Sha256};
        let bytes = serde_json::to_vec(self).expect("serializable model pair");
        coheryn_engine::storage::digest_hex(Sha256::digest(bytes).into())
    }
}
pub fn validate_id(id: &str) -> Result<()> {
    if id.is_empty()
        || id.len() > 64
        || !id
            .bytes()
            .next()
            .is_some_and(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        || !id
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"._-".contains(&b))
    {
        return Err(ApiError::bad(
            "model IDs must start with a lowercase letter/digit and contain at most 64 lowercase letters, digits, dots, underscores or hyphens",
        ));
    }
    Ok(())
}
pub fn validate_hash(hash: &str) -> Result<()> {
    if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(ApiError::bad("sha256 must contain 64 hexadecimal digits"));
    }
    Ok(())
}

pub struct Registry {
    root: PathBuf,
    _lock: File,
    pub max_model_bytes: u64,
    pub disk_quota: u64,
}
impl Registry {
    pub fn open(config: &ServerConfig) -> Result<Self> {
        private_directory(&config.data_dir)?;
        let root = config.data_dir.canonicalize()?;
        for name in ["models", "records"] {
            private_directory(&root.join(name))?;
        }
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(root.join("registry.lock"))?;
        fs2::FileExt::try_lock_exclusive(&lock).map_err(|_| {
            ApiError::conflict("this data directory is already owned by another daemon")
        })?;
        for entry in std::fs::read_dir(root.join("models"))? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.len() == 37
                && name.ends_with(".part")
                && name.as_bytes()[..32].iter().all(|b| b.is_ascii_hexdigit())
                && entry.file_type()?.is_file()
            {
                std::fs::remove_file(entry.path())?;
            }
        }
        Ok(Self {
            root,
            _lock: lock,
            max_model_bytes: config.max_model_bytes,
            disk_quota: config.disk_quota,
        })
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    fn record_path(&self, id: &str) -> Result<PathBuf> {
        validate_id(id)?;
        Ok(self.root.join("records").join(format!("model-{id}.json")))
    }
    pub fn model_path(&self, id: &str) -> Result<PathBuf> {
        validate_id(id)?;
        let path = self.root.join("models").join(format!("model-{id}.gguf"));
        if path.exists() {
            let meta = std::fs::symlink_metadata(&path)?;
            if !meta.is_file() || meta.file_type().is_symlink() {
                return Err(ApiError::bad(
                    "managed model must be a regular non-symlink file",
                ));
            }
        }
        Ok(path)
    }
    fn read_record(&self, id: &str) -> Result<ModelRecord> {
        let path = self.record_path(id)?;
        if !path.exists() {
            return Err(ApiError::missing("managed model not found"));
        }
        let meta = std::fs::symlink_metadata(&path)?;
        if !meta.is_file() || meta.file_type().is_symlink() || meta.len() > 65536 {
            return Err(ApiError::bad("invalid registry record"));
        }
        let file = File::open(path)?;
        let record: ModelRecord = serde_json::from_reader(file.take(65537))
            .map_err(|_| ApiError::bad("invalid registry record"))?;
        if record.id != id || record.schema_version != 1 {
            return Err(ApiError::bad("registry record identity/version mismatch"));
        }
        validate_hash(&record.sha256)?;
        Ok(record)
    }
    pub fn get(&self, id: &str) -> Result<ModelRecord> {
        let record = self.read_record(id)?;
        if let Some(pair) = &record.components {
            if pair.language.model == pair.vision.model
                || record.sha256 != pair.identity()
                || pair.language.bytes.checked_add(pair.vision.bytes) != Some(record.bytes)
            {
                return Err(ApiError::bad("invalid model pair identity"));
            }
            for component in [&pair.language, &pair.vision] {
                let child = self.read_record(&component.model)?;
                if child.components.is_some()
                    || child.sha256 != component.sha256
                    || child.bytes != component.bytes
                    || self.model_path(&component.model)?.metadata()?.len() != component.bytes
                {
                    return Err(Error::new(
                        ErrorCode::ModelChanged,
                        "model pair component changed",
                    )
                    .into());
                }
            }
        } else if self.model_path(id)?.metadata()?.len() != record.bytes {
            return Err(Error::new(ErrorCode::ModelChanged, "managed model size changed").into());
        }
        Ok(record)
    }
    pub(crate) fn pair_record(
        &self,
        id: &str,
        language: &str,
        vision: &str,
    ) -> Result<ModelRecord> {
        self.check_new(id)?;
        if language == vision {
            return Err(ApiError::bad("language and vision components must differ"));
        }
        let component = |id: &str| -> Result<ComponentRef> {
            let r = self.get(id)?;
            if r.components.is_some() {
                return Err(ApiError::bad("nested model pairs are unsupported"));
            }
            Ok(ComponentRef {
                model: r.id,
                sha256: r.sha256,
                bytes: r.bytes,
            })
        };
        let pair = ModelPair {
            language: component(language)?,
            vision: component(vision)?,
        };
        let record = ModelRecord {
            schema_version: 1,
            id: id.into(),
            sha256: pair.identity(),
            bytes: pair
                .language
                .bytes
                .checked_add(pair.vision.bytes)
                .ok_or_else(|| Error::resource("model pair byte overflow"))?,
            source: None,
            components: Some(pair),
        };
        Ok(record)
    }
    pub fn register_pair(&self, id: &str, language: &str, vision: &str) -> Result<ModelRecord> {
        let record = self.pair_record(id, language, vision)?;
        self.write_record(&record)?;
        Ok(record)
    }
    fn write_record(&self, record: &ModelRecord) -> Result<()> {
        let path = self.record_path(&record.id)?;
        let tmp = path.with_extension(format!("{}.part", uuid::Uuid::new_v4().simple()));
        let result = (|| -> Result<()> {
            let mut file = private_file(&tmp)?;
            file.write_all(
                &serde_json::to_vec(record)
                    .map_err(|_| ApiError::bad("record serialization failed"))?,
            )?;
            file.sync_all()?;
            std::fs::rename(&tmp, &path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(tmp);
        }
        result
    }
    pub fn list(&self) -> Result<Vec<ModelRecord>> {
        let mut ids = Vec::new();
        for entry in std::fs::read_dir(self.root.join("records"))? {
            let name = entry?.file_name();
            let Some(name) = name.to_str() else { continue };
            if let Some(id) = name
                .strip_prefix("model-")
                .and_then(|v| v.strip_suffix(".json"))
            {
                ids.push(id.to_owned());
            }
            if ids.len() > 128 {
                return Err(Error::resource("registry exceeds 128 models").into());
            }
        }
        ids.sort();
        ids.into_iter().map(|id| self.get(&id)).collect()
    }
    pub fn check_new(&self, id: &str) -> Result<()> {
        validate_id(id)?;
        if self.record_path(id)?.exists() || self.model_path(id)?.exists() {
            return Err(ApiError::conflict("model ID already exists"));
        }
        if self.list()?.len() >= 128 {
            return Err(Error::resource("registry is full").into());
        }
        Ok(())
    }
    pub fn disk_bytes(&self) -> Result<u64> {
        let mut total = 0u64;
        for (index, entry) in std::fs::read_dir(self.root.join("models"))?.enumerate() {
            if index >= 256 {
                return Err(Error::resource(
                    "managed storage exceeds file-count limit; remove orphaned models",
                )
                .into());
            }
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                return Err(ApiError::bad("managed storage contains a non-regular file"));
            }
            let meta = entry.metadata()?;
            total = total
                .checked_add(meta.len())
                .ok_or_else(|| Error::resource("disk usage overflow"))?;
        }
        Ok(total)
    }
    pub fn temporary(&self) -> Result<TemporaryModel> {
        let path = self
            .root
            .join("models")
            .join(format!("{}.part", uuid::Uuid::new_v4().simple()));
        let file = private_file(&path)?;
        Ok(TemporaryModel {
            path,
            file: Some(file),
        })
    }
    pub fn commit(&self, temporary: &mut TemporaryModel, record: &ModelRecord) -> Result<()> {
        self.check_new(&record.id)?;
        if let Some(file) = temporary.file.take() {
            file.sync_all()?;
        }
        std::fs::rename(&temporary.path, self.model_path(&record.id)?)?;
        let path = self.record_path(&record.id)?;
        let tmp = path.with_extension(format!("{}.part", uuid::Uuid::new_v4().simple()));
        let result = (|| -> Result<()> {
            let mut file = private_file(&tmp)?;
            file.write_all(
                &serde_json::to_vec(record)
                    .map_err(|_| ApiError::bad("record serialization failed"))?,
            )?;
            file.sync_all()?;
            std::fs::rename(&tmp, &path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&tmp);
            let _ = std::fs::remove_file(self.model_path(&record.id)?);
        }
        result
    }
    pub fn remove(&self, id: &str) -> Result<()> {
        if self.list()?.iter().any(|r| {
            r.components
                .as_ref()
                .is_some_and(|p| p.language.model == id || p.vision.model == id)
        }) {
            return Err(ApiError::conflict(
                "component is referenced by a model pair; remove the pair first",
            ));
        }
        let record = self.record_path(id)?;
        let model = self.model_path(id)?;
        if !record.exists() && !model.exists() {
            return Err(ApiError::missing("managed model not found"));
        }
        for path in [record, model] {
            match std::fs::remove_file(path) {
                Ok(()) => (),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
                Err(e) => return Err(e.into()),
            }
        }
        Ok(())
    }
}
pub struct TemporaryModel {
    pub path: PathBuf,
    pub file: Option<File>,
}
impl Drop for TemporaryModel {
    fn drop(&mut self) {
        self.file.take();
        if std::fs::remove_file(&self.path).is_err()
            && self.path.exists()
            && let Ok(runtime) = tokio::runtime::Handle::try_current()
        {
            let path = self.path.clone();
            runtime.spawn(async move {
                for _ in 0..5 {
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                    if tokio::fs::remove_file(&path).await.is_ok() || !path.exists() {
                        break;
                    }
                }
            });
        }
    }
}

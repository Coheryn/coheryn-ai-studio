use super::{Result, fixture, require, vision_fixture};
use axum::body::Bytes;
use coheryn_api::{
    checkpoint::{CheckpointManifest, MAGIC},
    service::Service,
};
use coheryn_engine::Cancellation;
use futures_util::{StreamExt, TryStreamExt, stream};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, io::Read, path::PathBuf, sync::Arc};
use tokio_util::io::ReaderStream;

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FileSource {
    pub path: PathBuf,
    pub sha256: String,
}
impl FileSource {
    fn validate(&self) -> Result<()> {
        coheryn_api::registry::validate_hash(&self.sha256)?;
        require(self.path.is_file(), "pinned source file is missing")
    }
    pub fn read_image(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        std::fs::File::open(&self.path)?
            .take((8 << 20) + 1)
            .read_to_end(&mut bytes)?;
        require(bytes.len() <= 8 << 20, "soak image exceeds 8 MiB")?;
        require(
            coheryn_engine::storage::digest_hex(Sha256::digest(&bytes).into()) == self.sha256,
            "pinned image changed",
        )?;
        Ok(bytes)
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointSource {
    pub path: PathBuf,
    pub pinned_files: BTreeMap<String, String>,
}
impl CheckpointSource {
    fn validate(&self) -> Result<()> {
        require(
            self.path.exists() && !self.pinned_files.is_empty() && self.pinned_files.len() <= 132,
            "invalid pinned checkpoint set",
        )?;
        for hash in self.pinned_files.values() {
            coheryn_api::registry::validate_hash(hash)?;
        }
        Ok(())
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Workload {
    pub text_prompt: String,
    pub backpressure_prompt: String,
    pub image_prompt: String,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum VisionSource {
    SharedText {
        image: FileSource,
    },
    Pair {
        language: FileSource,
        projector: FileSource,
        image: FileSource,
    },
    Checkpoint {
        path: PathBuf,
        pinned_files: BTreeMap<String, String>,
        image: FileSource,
    },
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Sources {
    pub schema_version: u32,
    pub kind: String,
    pub text: Option<FileSource>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_checkpoint: Option<CheckpointSource>,
    pub embedding: Option<FileSource>,
    pub vision: Option<VisionSource>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workload: Option<Workload>,
}
impl Sources {
    pub fn validate(&self) -> Result<()> {
        require(
            matches!(self.schema_version, 1 | 2),
            "unsupported sources schema",
        )?;
        require(
            self.schema_version == 2
                || (self.text_checkpoint.is_none()
                    && self.workload.is_none()
                    && !matches!(self.vision, Some(VisionSource::SharedText { .. }))),
            "modern source roles require schema 2",
        )?;
        if let Some(workload) = &self.workload {
            require(
                [
                    &workload.text_prompt,
                    &workload.backpressure_prompt,
                    &workload.image_prompt,
                ]
                .iter()
                .all(|p| !p.is_empty() && p.len() <= 4096),
                "invalid bounded workload prompts",
            )?;
        }
        match self.kind.as_str() {
            "synthetic" => require(
                self.text.is_none()
                    && self.text_checkpoint.is_none()
                    && self.embedding.is_none()
                    && self.vision.is_none()
                    && self.workload.is_none(),
                "synthetic mode must not hide pinned inputs",
            ),
            "pinned" => {
                match (&self.text, &self.text_checkpoint) {
                    (Some(file), None) => file.validate()?,
                    (None, Some(checkpoint)) => checkpoint.validate()?,
                    _ => return Err("exactly one text file or checkpoint is required".into()),
                }
                self.embedding
                    .as_ref()
                    .ok_or("missing embedding source")?
                    .validate()?;
                match self.vision.as_ref().ok_or("missing vision source")? {
                    VisionSource::SharedText { image } => {
                        require(
                            self.text_checkpoint.is_some(),
                            "shared vision requires a text checkpoint",
                        )?;
                        image.validate()?;
                    }
                    VisionSource::Pair {
                        language,
                        projector,
                        image,
                    } => {
                        language.validate()?;
                        projector.validate()?;
                        image.validate()?;
                    }
                    VisionSource::Checkpoint {
                        path,
                        pinned_files,
                        image,
                    } => {
                        CheckpointSource {
                            path: path.clone(),
                            pinned_files: pinned_files.clone(),
                        }
                        .validate()?;
                        image.validate()?;
                    }
                }
                Ok(())
            }
            _ => Err("unsupported sources mode".into()),
        }
    }
    pub fn ids(&self) -> Vec<&'static str> {
        if matches!(self.vision, Some(VisionSource::SharedText { .. })) {
            vec!["text", "embed"]
        } else {
            vec!["text", "embed", "image"]
        }
    }
    pub fn image_model(&self) -> &'static str {
        if matches!(self.vision, Some(VisionSource::SharedText { .. })) {
            "text"
        } else {
            "image"
        }
    }
    pub fn image(&self) -> Result<Vec<u8>> {
        match &self.vision {
            Some(
                VisionSource::Pair { image, .. }
                | VisionSource::Checkpoint { image, .. }
                | VisionSource::SharedText { image },
            ) => image.read_image(),
            None => Ok(include_bytes!("../../../../fixtures/vlm/images/minimum.png").to_vec()),
        }
    }
    pub async fn install(&self, service: &Arc<Service>) -> Result<()> {
        async fn checkpoint(
            s: &Arc<Service>,
            id: &str,
            path: &std::path::Path,
            pinned: &BTreeMap<String, String>,
        ) -> Result<()> {
            let (manifest, base) = CheckpointManifest::from_local(path, &Cancellation::default())?;
            let actual: BTreeMap<_, _> = manifest
                .files
                .iter()
                .map(|f| (f.name.clone(), f.sha256.clone()))
                .collect();
            require(
                &actual == pinned,
                "checkpoint source set differs from pinned files",
            )?;
            let metadata = serde_json::to_vec(&manifest)?;
            let mut prefix = MAGIC.to_vec();
            prefix.extend_from_slice(&(metadata.len() as u32).to_le_bytes());
            prefix.extend(metadata);
            let paths: Vec<_> = manifest.files.iter().map(|f| base.join(&f.name)).collect();
            let files = stream::iter(paths)
                .then(|p| async move { tokio::fs::File::open(p).await.map(ReaderStream::new) })
                .try_flatten();
            s.ingest_checkpoint(
                id.into(),
                Some(manifest.identity()?),
                stream::once(async { Ok::<_, std::io::Error>(Bytes::from(prefix)) }).chain(files),
            )
            .await?;
            Ok(())
        }
        async fn bytes(s: &Arc<Service>, id: &str, data: Vec<u8>) -> Result<()> {
            s.ingest(
                id.into(),
                None,
                None,
                stream::once(async { Ok::<_, std::io::Error>(Bytes::from(data)) }),
            )
            .await?;
            Ok(())
        }
        async fn file(s: &Arc<Service>, id: &str, source: &FileSource) -> Result<()> {
            s.ingest(
                id.into(),
                Some(source.sha256.clone()),
                None,
                ReaderStream::new(tokio::fs::File::open(&source.path).await?),
            )
            .await?;
            Ok(())
        }
        if self.kind == "synthetic" {
            bytes(service, "text", fixture::conversation_model()).await?;
            bytes(service, "embed", fixture::embedding_model(1)).await?;
            bytes(service, "language", fixture::tiny_vlm_model()).await?;
            let component = vision_fixture::component(256);
            bytes(service, "projector", std::fs::read(component.path())?).await?;
            service
                .register_pair("image".into(), "language".into(), "projector".into())
                .await?;
        } else {
            if let Some(source) = &self.text_checkpoint {
                checkpoint(service, "text", &source.path, &source.pinned_files).await?;
            } else {
                file(service, "text", self.text.as_ref().unwrap()).await?;
            }
            file(service, "embed", self.embedding.as_ref().unwrap()).await?;
            match self.vision.as_ref().unwrap() {
                VisionSource::SharedText { .. } => {}
                VisionSource::Pair {
                    language,
                    projector,
                    ..
                } => {
                    file(service, "language", language).await?;
                    file(service, "projector", projector).await?;
                    service
                        .register_pair("image".into(), "language".into(), "projector".into())
                        .await?;
                }
                VisionSource::Checkpoint {
                    path, pinned_files, ..
                } => {
                    checkpoint(service, "image", path, pinned_files).await?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn legacy_serialization_and_resident_roles_remain_unchanged() {
        let value: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../fixtures/soak/synthetic-models-v1.json"
        ))
        .unwrap();
        let source: Sources = serde_json::from_value(value.clone()).unwrap();
        source.validate().unwrap();
        assert_eq!(serde_json::to_value(&source).unwrap(), value);
        assert_eq!(source.ids(), ["text", "embed", "image"]);
        assert_eq!(source.image_model(), "image");
    }

    #[test]
    fn shared_text_vision_uses_one_resident_and_rejects_ambiguous_source_contracts() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.gguf");
        std::fs::write(&path, b"fixture").unwrap();
        let value = json!({"schema_version":2,"kind":"pinned","text":null,
            "text_checkpoint":{"path":path,"pinned_files":{"model.gguf":"0".repeat(64)}},
            "embedding":{"path":path,"sha256":"1".repeat(64)},
            "vision":{"kind":"shared_text","image":{"path":path,"sha256":"2".repeat(64)}}});
        let source: Sources = serde_json::from_value(value.clone()).unwrap();
        source.validate().unwrap();
        assert_eq!(source.ids(), ["text", "embed"]);
        assert_eq!(source.image_model(), "text");
        assert_eq!(serde_json::to_value(&source).unwrap(), value);
        let mut old_schema = value.clone();
        old_schema["schema_version"] = json!(1);
        assert!(
            serde_json::from_value::<Sources>(old_schema)
                .unwrap()
                .validate()
                .is_err()
        );
        let mut both = value.clone();
        both["text"] = both["embedding"].clone();
        assert!(
            serde_json::from_value::<Sources>(both)
                .unwrap()
                .validate()
                .is_err()
        );
        let mut missing = value;
        missing["text_checkpoint"] = serde_json::Value::Null;
        assert!(
            serde_json::from_value::<Sources>(missing)
                .unwrap()
                .validate()
                .is_err()
        );
    }

    #[tokio::test]
    async fn changed_pinned_file_is_rejected_instead_of_becoming_a_new_baseline() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("text.gguf");
        std::fs::write(&path, fixture::conversation_model()).unwrap();
        let file = FileSource {
            path,
            sha256: "0".repeat(64),
        };
        let source = Sources {
            schema_version: 1,
            kind: "pinned".into(),
            text: Some(file.clone()),
            text_checkpoint: None,
            embedding: Some(file.clone()),
            workload: None,
            vision: Some(VisionSource::Pair {
                language: file.clone(),
                projector: file.clone(),
                image: file,
            }),
        };
        let service = Service::open(
            coheryn_api::config::ServerConfig {
                data_dir: directory.path().join("vault"),
                ..Default::default()
            },
            coheryn_api::auth::Credentials {
                inference_token: "i".repeat(64),
                admin_token: "a".repeat(64),
            },
        )
        .unwrap();
        assert!(source.install(&service).await.is_err());
        assert!(service.loaded().unwrap().as_array().unwrap().is_empty());
        service.shutdown.cancel();
    }
}

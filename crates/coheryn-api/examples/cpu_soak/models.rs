use super::{Result, fixture, require, vision_fixture};
use axum::body::Bytes;
use coheryn_api::{
    checkpoint::{CheckpointManifest, MAGIC},
    service::Service,
};
use coheryn_engine::Cancellation;
use futures_util::{StreamExt, TryStreamExt, stream};
use serde::{Deserialize, Serialize};
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
            coheryn_api::checkpoint::hash_file(&self.path, &Cancellation::default())?
                == self.sha256,
            "pinned image changed",
        )?;
        Ok(bytes)
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum VisionSource {
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
    pub embedding: Option<FileSource>,
    pub vision: Option<VisionSource>,
}
impl Sources {
    pub fn validate(&self) -> Result<()> {
        require(self.schema_version == 1, "unsupported sources schema")?;
        match self.kind.as_str() {
            "synthetic" => require(
                self.text.is_none() && self.embedding.is_none() && self.vision.is_none(),
                "synthetic mode must not hide pinned inputs",
            ),
            "pinned" => {
                self.text
                    .as_ref()
                    .ok_or("missing text source")?
                    .validate()?;
                self.embedding
                    .as_ref()
                    .ok_or("missing embedding source")?
                    .validate()?;
                match self.vision.as_ref().ok_or("missing vision source")? {
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
                        require(
                            path.exists() && !pinned_files.is_empty() && pinned_files.len() <= 132,
                            "invalid pinned checkpoint set",
                        )?;
                        for hash in pinned_files.values() {
                            coheryn_api::registry::validate_hash(hash)?;
                        }
                        image.validate()?;
                    }
                }
                Ok(())
            }
            _ => Err("unsupported sources mode".into()),
        }
    }
    pub fn ids(&self) -> Vec<&'static str> {
        vec!["text", "embed", "image"]
    }
    pub fn image(&self) -> Result<Vec<u8>> {
        match &self.vision {
            Some(VisionSource::Pair { image, .. } | VisionSource::Checkpoint { image, .. }) => {
                image.read_image()
            }
            None => Ok(include_bytes!("../../../../fixtures/vlm/images/minimum.png").to_vec()),
        }
    }
    pub async fn install(&self, service: &Arc<Service>) -> Result<()> {
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
                None,
                Some(source.sha256.clone()),
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
            file(service, "text", self.text.as_ref().unwrap()).await?;
            file(service, "embed", self.embedding.as_ref().unwrap()).await?;
            match self.vision.as_ref().unwrap() {
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
                    let (manifest, base) =
                        CheckpointManifest::from_local(path, &Cancellation::default())?;
                    let actual: BTreeMap<_, _> = manifest
                        .files
                        .iter()
                        .map(|f| (f.name.clone(), f.sha256.clone()))
                        .collect();
                    require(
                        &actual == pinned_files,
                        "checkpoint source set differs from pinned files",
                    )?;
                    let metadata = serde_json::to_vec(&manifest)?;
                    let mut prefix = MAGIC.to_vec();
                    prefix.extend_from_slice(&(metadata.len() as u32).to_le_bytes());
                    prefix.extend(metadata);
                    let paths: Vec<_> = manifest.files.iter().map(|f| base.join(&f.name)).collect();
                    let files =
                        stream::iter(paths)
                            .then(|p| async move {
                                tokio::fs::File::open(p).await.map(ReaderStream::new)
                            })
                            .try_flatten();
                    service
                        .ingest_checkpoint(
                            "image".into(),
                            Some(manifest.identity()?),
                            stream::once(async { Ok::<_, std::io::Error>(Bytes::from(prefix)) })
                                .chain(files),
                        )
                        .await?;
                }
            }
        }
        Ok(())
    }
}

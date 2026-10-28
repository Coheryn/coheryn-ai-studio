//! Native multimodal contracts, bounded image preparation and cooperative execution.
pub mod image;
pub mod positions;
pub mod vision;
/// Fallible scheduler checkpoint, called only after owned CPU workers have joined.
pub type WorkYield<'a> = dyn FnMut() -> crate::Result<()> + 'a;
/// Unstable native image-to-text adapter shared by product and development entry points.
pub mod qwen2vl {
    pub use crate::inference::vlm::*;
}

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageUrl {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}
impl Content {
    /// Structural bounds only: this does not authorize a URL, decode an image or qualify a graph.
    pub fn validate_bounds(&self, max_bytes: usize, max_images: usize) -> Result<()> {
        let mut bytes = 0usize;
        let mut images = 0usize;
        let mut count = |n| -> Result<()> {
            bytes = bytes
                .checked_add(n)
                .filter(|n| *n <= max_bytes)
                .ok_or_else(|| Error::resource("multimodal content byte limit exceeded"))?;
            Ok(())
        };
        match self {
            Self::Text(text) => count(text.len())?,
            Self::Parts(parts) => {
                if parts.is_empty() || parts.len() > 256 {
                    return Err(Error::request("content requires 1..=256 ordered parts"));
                }
                for part in parts {
                    match part {
                        ContentPart::Text { text } => count(text.len())?,
                        ContentPart::ImageUrl { image_url } => {
                            if image_url.url.is_empty()
                                || image_url.detail.as_ref().is_some_and(|d| d.len() > 32)
                            {
                                return Err(Error::request("invalid image content descriptor"));
                            }
                            count(image_url.url.len())?;
                            count(image_url.detail.as_ref().map_or(0, |d| d.len()))?;
                            images += 1;
                            if images > max_images {
                                return Err(Error::resource("image count exceeds content limit"));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// A development selection manifest; structural validity never implies inference compatibility.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionManifest {
    pub schema_version: u32,
    pub stage: String,
    pub models: Vec<ModelSet>,
    pub runtime_dependency: bool,
    pub full_weight_downloaded: bool,
    pub oracle: Oracle,
    pub remaining_acceptance: Vec<String>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Oracle {
    pub source_record: String,
    pub source_record_sha256: String,
    pub revision: String,
    pub production_dependency: bool,
    pub vision_reference_outputs_captured: bool,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSet {
    pub id: String,
    pub repository: String,
    pub revision: String,
    pub declared_license: Option<String>,
    pub components: Vec<Component>,
    pub source_repository: String,
    pub source_revision: String,
    pub source_declared_license: Option<String>,
    pub source_asset_names: Vec<String>,
    pub source_assets: Vec<Asset>,
    pub qualification: String,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Component {
    pub role: String,
    pub file: String,
    pub bytes: u64,
    pub sha256: String,
    pub hash_status: String,
    pub prefix_bytes: u64,
    pub prefix_sha256: String,
    pub inventory: String,
    pub header_status: String,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Asset {
    pub file: String,
    pub bytes: u64,
    pub sha256: String,
}
fn hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}
fn basename(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 256
        && !matches!(value, "." | "..")
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c))
}
fn repository(value: &str) -> bool {
    let parts: Vec<_> = value.split('/').collect();
    parts.len() == 2 && parts.iter().all(|p| basename(p))
}
impl SelectionManifest {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > 1 << 20 {
            return Err(Error::resource("selection manifest exceeds one MiB"));
        }
        let manifest: Self = serde_json::from_slice(bytes)
            .map_err(|e| Error::invalid(format!("invalid selection manifest: {e}")))?;
        manifest.validate()?;
        Ok(manifest)
    }
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 1
            || self.models.is_empty()
            || self.models.len() > 8
            || self.runtime_dependency
            || self.oracle.production_dependency
            || !hex(&self.oracle.revision, 40)
            || !hex(&self.oracle.source_record_sha256, 64)
            || !self.oracle.source_record.starts_with("resources/")
            || self.oracle.source_record.len() > 1024
            || !self.oracle.source_record.split('/').all(basename)
            || self.remaining_acceptance.len() > 32
        {
            return Err(Error::invalid(
                "invalid selection manifest bounds/runtime boundary",
            ));
        }
        let mut ids = BTreeSet::new();
        for model in &self.models {
            if !basename(&model.id)
                || !ids.insert(&model.id)
                || !repository(&model.repository)
                || !repository(&model.source_repository)
                || !hex(&model.revision, 40)
                || !hex(&model.source_revision, 40)
                || model.components.len() != 2
                || model.source_assets.len() > 32
                || model.qualification != "not-executable-not-qualified"
            {
                return Err(Error::invalid("invalid/unqualified model-set selection"));
            }
            let mut roles = BTreeSet::new();
            let mut files = BTreeSet::new();
            for component in &model.components {
                if !matches!(component.role.as_str(), "language" | "vision-projector")
                    || !roles.insert(&component.role)
                    || !basename(&component.file)
                    || !files.insert(&component.file)
                    || component.bytes == 0
                    || !hex(&component.sha256, 64)
                    || !hex(&component.prefix_sha256, 64)
                    || component.prefix_bytes == 0
                    || component.prefix_bytes > 16 << 20
                    || component.prefix_bytes > component.bytes
                    || component.hash_status != "publisher-declared-not-full-file-verified"
                    || component.header_status != "bounded-prefix-parsed-payload-not-verified"
                {
                    return Err(Error::invalid(
                        "invalid component identity/extent/provenance",
                    ));
                }
                // Inventories are repository artifacts, never externally supplied loader paths.
                let expected = format!(
                    "fixtures/vlm/{}/{}.inventory.json",
                    model.id, component.role
                );
                if component.inventory != expected {
                    return Err(Error::invalid("unexpected component inventory path"));
                }
            }
            let mut assets = BTreeSet::new();
            for asset in &model.source_assets {
                if !basename(&asset.file)
                    || !assets.insert(&asset.file)
                    || asset.bytes == 0
                    || asset.bytes > 20 << 20
                    || !hex(&asset.sha256, 64)
                {
                    return Err(Error::invalid("invalid source asset identity/extent"));
                }
            }
            let names: BTreeSet<_> = model.source_asset_names.iter().collect();
            if model.source_asset_names.len() != assets.len()
                || names != assets
                || ![
                    "config.json",
                    "preprocessor_config.json",
                    "tokenizer_config.json",
                ]
                .iter()
                .all(|f| assets.contains(&f.to_string()))
            {
                return Err(Error::invalid("missing/duplicate model-set source assets"));
            }
        }
        Ok(())
    }
}

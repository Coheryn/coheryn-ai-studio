//! Private original-image assets. History references are authoritative; KV is never stored.
use crate::{
    auth,
    error::{ApiError, Result},
};
use coheryn_engine::{
    Cancellation, Error, ErrorCode,
    multimodal::image::{self, ImageInfo, ImageLimits},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
};

#[derive(Clone, Serialize)]
pub struct AssetLimits {
    pub total_bytes: u64,
    pub owner_bytes: u64,
    pub session_bytes: u64,
    pub session_items: usize,
}
impl Default for AssetLimits {
    fn default() -> Self {
        Self {
            total_bytes: 512 << 20,
            owner_bytes: 128 << 20,
            session_bytes: 64 << 20,
            session_items: 16,
        }
    }
}
impl AssetLimits {
    pub fn validate(&self) -> coheryn_engine::Result<()> {
        if self.total_bytes == 0
            || self.total_bytes > 2 << 30
            || self.owner_bytes == 0
            || self.owner_bytes > self.total_bytes
            || self.session_bytes == 0
            || self.session_bytes > self.owner_bytes
            || self.session_items == 0
            || self.session_items > 16
        {
            return Err(Error::request("invalid finite session image quotas"));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageAsset {
    pub id: String,
    pub sha256: String,
    pub bytes: u64,
    pub info: ImageInfo,
}
impl ImageAsset {
    pub fn validate(&self) -> Result<()> {
        asset_id(&self.id)?;
        crate::registry::validate_hash(&self.sha256)?;
        if self.bytes == 0
            || self.bytes > ImageLimits::default().max_encoded_bytes
            || self.bytes != self.info.encoded_bytes
        {
            return Err(ApiError::bad("invalid saved image byte count"));
        }
        image::plan(
            self.info,
            image::Processor::Qwen2VlStill,
            ImageLimits::default(),
        )?;
        Ok(())
    }
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExportImage {
    pub id: String,
    pub base64: String,
}
pub(crate) struct Assets {
    root: PathBuf,
}
fn asset_id(id: &str) -> Result<()> {
    if id.len() != 36
        || !id.starts_with("img-")
        || !id.as_bytes()[4..].iter().all(u8::is_ascii_hexdigit)
    {
        return Err(ApiError::bad("invalid session image ID"));
    }
    Ok(())
}
fn directory(path: &Path, create: bool) -> Result<()> {
    if create {
        auth::private_directory(path)?;
    }
    let m = std::fs::symlink_metadata(path)?;
    if !m.is_dir() || m.file_type().is_symlink() {
        return Err(ApiError::bad("unsafe image asset directory"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if m.permissions().mode() & 0o077 != 0 {
            return Err(ApiError::bad("image asset directory must be private"));
        }
    }
    Ok(())
}
fn regular(file: &File, expected: u64) -> Result<()> {
    let m = file.metadata()?;
    if !m.is_file() || m.len() != expected {
        return Err(ApiError::bad("session image asset size changed"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        if m.nlink() != 1 || m.permissions().mode() & 0o077 != 0 {
            return Err(ApiError::bad(
                "image asset must be private and have no extra links",
            ));
        }
    }
    Ok(())
}
impl Assets {
    pub fn new(root: &Path) -> Self {
        Self {
            root: root.join("image-assets"),
        }
    }
    fn session(&self, session: &str, create: bool) -> Result<PathBuf> {
        super::identifier(session)?;
        directory(&self.root, create)?;
        let path = self.root.join(session);
        directory(&path, create)?;
        Ok(path)
    }
    pub fn put(&self, session: &str, asset: &ImageAsset, bytes: &[u8]) -> Result<()> {
        asset.validate()?;
        if bytes.len() as u64 != asset.bytes
            || coheryn_engine::storage::digest_hex(Sha256::digest(bytes).into()) != asset.sha256
        {
            return Err(ApiError::bad("image asset content identity mismatch"));
        }
        let info = image::inspect(bytes, ImageLimits::default(), &Cancellation::default())?;
        if info.width != asset.info.width
            || info.height != asset.info.height
            || info.channels != asset.info.channels
            || info.orientation != asset.info.orientation
            || info.format != asset.info.format
        {
            return Err(ApiError::bad(
                "imported image descriptor differs from its bytes",
            ));
        }
        let dir = self.session(session, true)?;
        let path = dir.join(format!("{}.bin", asset.id));
        let tmp = dir.join(format!("{}.part", uuid::Uuid::new_v4().simple()));
        let result = (|| -> Result<()> {
            if path.exists() {
                return Err(ApiError::conflict("image asset already exists"));
            }
            let mut f = auth::private_file(&tmp)?;
            f.write_all(bytes)?;
            f.sync_all()?;
            drop(f);
            std::fs::rename(&tmp, &path)?;
            #[cfg(unix)]
            {
                File::open(&dir)?.sync_all()?;
                File::open(&self.root)?.sync_all()?;
                if let Some(parent) = self.root.parent() {
                    File::open(parent)?.sync_all()?;
                }
            }
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(tmp);
        }
        result
    }
    pub fn read(
        &self,
        session: &str,
        asset: &ImageAsset,
        cancel: &Cancellation,
    ) -> Result<Vec<u8>> {
        asset.validate()?;
        cancel.check()?;
        let path = self
            .session(session, false)?
            .join(format!("{}.bin", asset.id));
        if std::fs::symlink_metadata(&path)?.file_type().is_symlink() {
            return Err(ApiError::bad("image asset cannot be a symlink"));
        }
        let f = File::open(path)?;
        regular(&f, asset.bytes)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(asset.bytes as usize)
            .map_err(|_| Error::resource("image asset allocation failed"))?;
        (&f).take(asset.bytes + 1).read_to_end(&mut bytes)?;
        regular(&f, asset.bytes)?;
        cancel.check()?;
        if bytes.len() as u64 != asset.bytes
            || coheryn_engine::storage::digest_hex(Sha256::digest(&bytes).into()) != asset.sha256
        {
            return Err(Error::new(
                ErrorCode::ModelChanged,
                "session image asset content changed",
            )
            .into());
        }
        let info = image::inspect(&bytes, ImageLimits::default(), cancel)?;
        if info.width != asset.info.width
            || info.height != asset.info.height
            || info.format != asset.info.format
            || info.orientation != asset.info.orientation
            || info.channels != asset.info.channels
        {
            return Err(ApiError::bad(
                "saved image descriptor differs from original bytes",
            ));
        }
        Ok(bytes)
    }
    /// Counts all files, including abandoned writes, against the global disk quota.
    pub fn disk_bytes(&self) -> Result<u64> {
        if !self.root.exists() {
            return Ok(0);
        }
        directory(&self.root, false)?;
        let mut total = 0u64;
        for (n, entry) in std::fs::read_dir(&self.root)?.enumerate() {
            if n >= 128 {
                return Err(Error::resource("image asset session-directory limit exceeded").into());
            }
            let dir = entry?.path();
            directory(&dir, false)?;
            for (i, file) in std::fs::read_dir(dir)?.enumerate() {
                if i >= 32 {
                    return Err(Error::resource("image asset file-count limit exceeded").into());
                }
                let f = file?;
                let m = f.metadata()?;
                if !f.file_type()?.is_file() {
                    return Err(ApiError::bad("unsafe image asset entry"));
                }
                total = total
                    .checked_add(m.len())
                    .ok_or_else(|| Error::resource("asset disk accounting overflow"))?;
            }
        }
        Ok(total)
    }
    pub fn remove_session(&self, session: &str) -> Result<()> {
        if !self.root.exists() {
            return Ok(());
        }
        let path = self.root.join(session);
        if !path.exists() {
            return Ok(());
        }
        let dir = self.session(session, false)?;
        for (i, entry) in std::fs::read_dir(&dir)?.enumerate() {
            if i >= 32 {
                return Err(ApiError::bad("too many image asset files"));
            }
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                return Err(ApiError::bad("unsafe image asset entry"));
            }
            std::fs::remove_file(entry.path())?;
        }
        std::fs::remove_dir(dir)?;
        Ok(())
    }
    /// Reclaim ephemeral/abandoned assets after restart; never follow directory links.
    pub fn recover(
        &self,
        live: &std::collections::BTreeMap<String, BTreeSet<String>>,
    ) -> Result<()> {
        if !self.root.exists() {
            return Ok(());
        }
        directory(&self.root, false)?;
        for (n, entry) in std::fs::read_dir(&self.root)?.enumerate() {
            if n >= 128 {
                return Err(ApiError::bad("too many image asset directories"));
            }
            let entry = entry?;
            let id = entry.file_name().to_string_lossy().into_owned();
            let dir = self.session(&id, false)?;
            for (i, file) in std::fs::read_dir(&dir)?.enumerate() {
                if i >= 32 {
                    return Err(ApiError::bad("too many image asset files"));
                }
                let file = file?;
                if !file.file_type()?.is_file() {
                    return Err(ApiError::bad("unsafe image asset recovery entry"));
                }
                let name = file.file_name().to_string_lossy().into_owned();
                if !live.get(&id).is_some_and(|names| names.contains(&name)) {
                    std::fs::remove_file(file.path())?;
                }
            }
            if !live.contains_key(&id) {
                std::fs::remove_dir(dir)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct Attachment {
    pub session_id: String,
    pub revision: u64,
    pub image: ImageAsset,
}
impl super::Sessions {
    fn asset_quota(
        &self,
        state: &std::collections::BTreeMap<String, super::Session>,
        owner: &str,
        images: &[ImageAsset],
    ) -> Result<()> {
        let limits = &self.0.limits.images;
        let bytes = images.iter().try_fold(0u64, |n, a| {
            n.checked_add(a.bytes)
                .ok_or_else(|| Error::resource("image quota overflow"))
        })?;
        let owned = state
            .values()
            .filter(|s| s.owner == owner)
            .flat_map(|s| &s.view.image_assets)
            .try_fold(0u64, |n, a| {
                n.checked_add(a.bytes)
                    .ok_or_else(|| Error::resource("image owner quota overflow"))
            })?;
        let disk = self
            .0
            .assets
            .as_ref()
            .ok_or_else(|| ApiError::bad("image asset storage unavailable"))?
            .disk_bytes()?;
        if images.len() > limits.session_items
            || bytes > limits.session_bytes
            || owned
                .checked_add(bytes)
                .is_none_or(|n| n > limits.owner_bytes)
            || disk
                .checked_add(bytes)
                .is_none_or(|n| n > limits.total_bytes)
        {
            return Err(Error::resource("session/owner/global image disk quota exceeded").into());
        }
        Ok(())
    }
    pub fn image_stats(&self) -> Result<serde_json::Value> {
        let state = self.state()?;
        let assets = state
            .values()
            .map(|s| s.view.image_assets.len())
            .sum::<usize>();
        let referenced = state
            .values()
            .flat_map(|s| &s.view.image_assets)
            .map(|a| a.bytes)
            .sum::<u64>();
        let disk = self.0.assets.as_ref().map_or(Ok(0), Assets::disk_bytes)?;
        Ok(
            serde_json::json!({"assets":assets,"referenced_bytes":referenced,"disk_bytes":disk,"limits":self.0.limits.images,"storage":"private-original-bytes","decoded_pixel_cache":false,"persisted_kv":false}),
        )
    }
    pub fn attach_image(
        &self,
        owner: &str,
        id: &str,
        revision: u64,
        bytes: &[u8],
        limits: ImageLimits,
        cancel: &Cancellation,
    ) -> Result<Attachment> {
        let info = image::inspect(bytes, limits, cancel)?;
        image::plan(info, image::Processor::Qwen2VlStill, limits)?;
        let hash = coheryn_engine::storage::digest_hex(Sha256::digest(bytes).into());
        let mut state = self.state()?;
        let id = Self::resolve(&state, owner, id)?;
        self.hydrate(&mut state, &id)?;
        let session = state.get(&id).expect("resolved session");
        Self::mutable(session, revision)?;
        if session.view.task != super::SessionTask::ImageToText {
            return Err(ApiError::bad(
                "images require an image-to-text conversation",
            ));
        }
        let assets = self
            .0
            .assets
            .as_ref()
            .ok_or_else(|| ApiError::bad("image asset storage unavailable"))?;
        if let Some(existing) = session
            .view
            .image_assets
            .iter()
            .find(|a| a.sha256 == hash && a.bytes == bytes.len() as u64)
        {
            assets.read(&id, existing, cancel)?;
            return Ok(Attachment {
                session_id: id,
                revision: session.view.revision,
                image: existing.clone(),
            });
        }
        if session.view.image_assets.len() >= self.0.limits.images.session_items
            || session
                .view
                .image_assets
                .iter()
                .map(|a| a.bytes)
                .sum::<u64>()
                + bytes.len() as u64
                > self.0.limits.images.session_bytes
        {
            return Err(Error::resource("session image quota exceeded").into());
        }
        let asset = ImageAsset {
            id: format!("img-{}", uuid::Uuid::new_v4().simple()),
            sha256: hash,
            bytes: bytes.len() as u64,
            info,
        };
        self.asset_quota(&state, owner, std::slice::from_ref(&asset))?;
        assets.put(&id, &asset, bytes)?;
        let session = state.get_mut(&id).expect("resolved session");
        let old = session.view.clone();
        session.view.image_assets.push(asset.clone());
        session.view.revision += 1;
        session.view.updated = super::now();
        session.touched = std::time::Instant::now();
        if let Err(error) = self.persist(session, None) {
            session.view = old;
            let _ = assets.remove_image(&id, &asset.id);
            return Err(error);
        }
        Ok(Attachment {
            session_id: id,
            revision: session.view.revision,
            image: asset,
        })
    }
    pub(super) fn exported_assets(
        &self,
        session: &super::SessionView,
        current_bytes: usize,
    ) -> Result<Vec<ExportImage>> {
        use base64::{Engine, engine::general_purpose::STANDARD};
        let encoded = session
            .image_assets
            .iter()
            .try_fold(current_bytes as u64, |n, a| {
                n.checked_add(a.bytes.div_ceil(3) * 4 + 128)
                    .ok_or_else(|| Error::resource("image export size overflow"))
            })?;
        if encoded > crate::session_store::DOCUMENT_BYTES as u64 {
            return Err(Error::resource(
                "portable image export exceeds the bounded 8 MiB document limit",
            )
            .into());
        }
        let mut out = Vec::new();
        if let Some(assets) = &self.0.assets {
            for a in &session.image_assets {
                let bytes = assets.read(&session.id, a, &Cancellation::default())?;
                out.push(ExportImage {
                    id: a.id.clone(),
                    base64: STANDARD.encode(bytes),
                });
            }
        }
        Ok(out)
    }
    pub(super) fn import_assets(
        &self,
        state: &std::collections::BTreeMap<String, super::Session>,
        owner: &str,
        doc: &crate::session_store::ExportDocument,
    ) -> Result<()> {
        use base64::{Engine, engine::general_purpose::STANDARD};
        if doc.assets.len() != doc.session.image_assets.len() {
            return Err(ApiError::bad(
                "portable history is missing original image assets",
            ));
        }
        if doc.assets.is_empty() {
            return Ok(());
        }
        self.asset_quota(state, owner, &doc.session.image_assets)?;
        let assets = self
            .0
            .assets
            .as_ref()
            .ok_or_else(|| ApiError::bad("image asset storage unavailable"))?;
        let mut seen = BTreeSet::new();
        let result = (|| -> Result<()> {
            for payload in &doc.assets {
                if !seen.insert(&payload.id) {
                    return Err(ApiError::bad("duplicate exported image payload"));
                }
                let a = doc
                    .session
                    .image_assets
                    .iter()
                    .find(|a| a.id == payload.id)
                    .ok_or_else(|| ApiError::bad("unknown exported image payload"))?;
                if payload.base64.len() as u64 != a.bytes.div_ceil(3) * 4 {
                    return Err(ApiError::bad("exported image byte count mismatch"));
                }
                let bytes = STANDARD
                    .decode(&payload.base64)
                    .map_err(|_| ApiError::bad("invalid canonical exported image Base64"))?;
                image::inspect(&bytes, ImageLimits::default(), &Cancellation::default())?;
                assets.put(&doc.session.id, a, &bytes)?;
            }
            Ok(())
        })();
        if result.is_err() {
            let _ = assets.remove_session(&doc.session.id);
        }
        result
    }
}
impl Assets {
    fn remove_image(&self, session: &str, id: &str) -> Result<()> {
        asset_id(id)?;
        std::fs::remove_file(self.session(session, false)?.join(format!("{id}.bin")))?;
        Ok(())
    }
}

impl super::TurnLease {
    pub fn task(&self) -> super::SessionTask {
        self.task
    }
    pub fn vision_request(&self) -> crate::wire::GenerationRequest {
        let mut r = self
            .settings
            .request(self.model.clone(), self.context, "validation".into());
        r.prompt = None;
        r.messages = Some(vec![coheryn_engine::tokenizer::Message {
            role: "user".into(),
            content: "validation".into(),
            ..Default::default()
        }]);
        r
    }
    pub fn prepare_vision(
        &mut self,
        model: &coheryn_engine::multimodal::qwen2vl::Qwen2Vl,
        options: &coheryn_engine::multimodal::qwen2vl::Options,
        limits: ImageLimits,
        cancel: &Cancellation,
    ) -> Result<coheryn_engine::multimodal::qwen2vl::PreparedRequest> {
        use coheryn_engine::multimodal::qwen2vl::{ChatContent, ChatMessage, ChatPart};
        let mut selected: Vec<_> = self
            .messages
            .iter()
            .filter(|m| {
                m.status == super::TurnStatus::Completed
                    || m.turn_id.as_deref() == Some(&self.turn_id)
            })
            .cloned()
            .collect();
        let mut excluded: Vec<_> = self
            .messages
            .iter()
            .filter(|m| {
                m.status != super::TurnStatus::Completed
                    && m.turn_id.as_deref() != Some(&self.turn_id)
            })
            .map(|m| m.id.clone())
            .collect();
        loop {
            cancel.check()?;
            let result = (|| -> Result<_> {
                let mut messages = Vec::new();
                let mut images = Vec::new();
                if selected.len() > 64
                    || selected.iter().map(|m| m.content.len()).sum::<usize>() > 64 << 10
                {
                    return Err(Error::new(
                        ErrorCode::ContextLimitExceeded,
                        "selected history exceeds qualified image message/text limits",
                    )
                    .into());
                }
                let count = selected.iter().map(|m| m.images.len()).sum::<usize>();
                if count > limits.max_images {
                    return Err(Error::new(
                        ErrorCode::ImageInputLimitExceeded,
                        "selected history exceeds the model image limit",
                    )
                    .into());
                }
                let state = self.sessions.state()?;
                let session = state.get(&self.id).ok_or_else(super::internal)?;
                let descriptors: Vec<_> = selected
                    .iter()
                    .flat_map(|m| &m.images)
                    .map(|id| {
                        session
                            .view
                            .image_assets
                            .iter()
                            .find(|a| &a.id == id)
                            .cloned()
                            .ok_or_else(|| ApiError::bad("history image asset is missing"))
                    })
                    .collect::<Result<_>>()?;
                drop(state);
                let total = descriptors.iter().try_fold(0u64, |n, a| {
                    n.checked_add(a.bytes)
                        .filter(|n| *n <= limits.max_total_encoded_bytes)
                        .ok_or_else(|| {
                            ApiError::from(Error::new(
                                ErrorCode::ImageInputLimitExceeded,
                                "history images exceed aggregate byte limit",
                            ))
                        })
                })?;
                let _input = self.sessions.0.budget.reserve(total)?;
                let assets = self
                    .sessions
                    .0
                    .assets
                    .as_ref()
                    .ok_or_else(|| ApiError::bad("image asset storage unavailable"))?;
                let mut descriptors = descriptors.into_iter();
                let mut features = 0;
                for m in &selected {
                    let content = if m.images.is_empty() {
                        ChatContent::Text(m.content.clone())
                    } else {
                        let mut parts = Vec::new();
                        for _ in &m.images {
                            let a = descriptors.next().expect("checked image descriptors");
                            let bytes = assets.read(&self.id, &a, cancel)?;
                            let info = image::inspect(&bytes, limits, cancel)?;
                            let plan = image::plan(info, image::Processor::Qwen2VlStill, limits)?;
                            features += plan.image_feature_tokens;
                            if features > limits.max_total_image_tokens {
                                return Err(Error::new(
                                    ErrorCode::ImageInputLimitExceeded,
                                    "history image features exceed limit",
                                )
                                .into());
                            }
                            parts.push(ChatPart::Image {
                                index: images.len(),
                            });
                            images.push(bytes);
                        }
                        if !m.content.is_empty() {
                            parts.push(ChatPart::Text {
                                text: m.content.clone(),
                            });
                        }
                        ChatContent::Parts(parts)
                    };
                    messages.push(ChatMessage {
                        role: m.role.clone(),
                        content,
                    });
                }
                crate::vision::validate_controls(&self.vision_request())?;
                let prepared = model.prepare_owned(&messages, images, options, cancel)?;
                Ok(prepared)
            })();
            match result {
                Ok(prepared) => {
                    let history_revision = self.watch.snapshot()?.revision;
                    self.watch
                        .0
                        .lock()
                        .map_err(|_| super::internal())?
                        .receipt
                        .context = Some(super::ContextAccounting {
                        policy: self.policy,
                        history_revision,
                        context: self.context,
                        prompt_tokens: prepared.plan().physical_prompt_tokens,
                        reserved_output_tokens: self.settings.max_tokens,
                        included_message_ids: selected.iter().map(|m| m.id.clone()).collect(),
                        excluded_message_ids: excluded,
                    });
                    return Ok(prepared);
                }
                Err(error)
                    if self.policy == super::OverflowPolicy::OldestCompleteTurns
                        && matches!(
                            error.error.code,
                            ErrorCode::ContextLimitExceeded | ErrorCode::ImageInputLimitExceeded
                        ) => {}
                Err(error) => return Err(error),
            }
            let oldest=selected.iter().find(|m|m.role=="user" && m.turn_id.as_deref()!=Some(&self.turn_id)).and_then(|m|m.turn_id.clone()).ok_or_else(||ApiError::bad("protected instructions/current image turn cannot fit the context/image limits"))?;
            selected.retain(|m| {
                if m.turn_id.as_deref() == Some(&oldest) {
                    excluded.push(m.id.clone());
                    false
                } else {
                    true
                }
            });
        }
    }
}

impl Drop for super::Inner {
    fn drop(&mut self) {
        // Last ownership implies no worker lease can still be reading an asset.
        // On cleanup errors, preserve files for the next bounded recovery pass.
        if let Some(assets) = &self.assets
            && let Ok(state) = self.state.get_mut()
        {
            for (id, _) in state.iter().filter(|(_, s)| !s.view.persistent) {
                let _ = assets.remove_session(id);
            }
        }
    }
}

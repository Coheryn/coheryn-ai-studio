//! Opt-in embedded history storage. Model execution and KV tensors never enter it.
use crate::{
    auth,
    error::{ApiError, Result},
    sessions::{HistoryMessage, SessionView, TurnReceipt, TurnSettings},
};
use coheryn_engine::{
    Error, ErrorCode,
    memory::{Budget, Reservation},
};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::{path::Path, sync::Mutex};

pub const DOCUMENT_BYTES: usize = 8 << 20;
pub const TRANSFER_BYTES: usize = 16 << 20;
const APP_ID: i64 = 0x434f4853;

#[derive(Clone, Serialize)]
pub struct PersistenceLimits {
    pub enabled: bool,
    pub total_bytes: u64,
    pub owner_bytes: u64,
    pub retention_seconds: u64,
}
impl Default for PersistenceLimits {
    fn default() -> Self {
        Self {
            enabled: false,
            total_bytes: 1 << 30,
            owner_bytes: 512 << 20,
            retention_seconds: 2_592_000,
        }
    }
}
impl PersistenceLimits {
    pub fn validate(&self) -> coheryn_engine::Result<()> {
        if self.total_bytes < 1 << 20
            || self.total_bytes > 1 << 30
            || self.owner_bytes == 0
            || self.owner_bytes > self.total_bytes
            || self.retention_seconds < 60
            || self.retention_seconds > 315_360_000
        {
            return Err(Error::request(
                "invalid persistent history quotas/retention",
            ));
        }
        Ok(())
    }
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SavedTurn {
    pub key: String,
    pub fingerprint: [u8; 32],
    pub settings: TurnSettings,
    pub receipt: TurnReceipt,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExportDocument {
    pub version: u32,
    pub session: SessionView,
    #[serde(deserialize_with = "crate::bounded::sequence::<_, _, 256>")]
    pub messages: Vec<HistoryMessage>,
    #[serde(deserialize_with = "crate::bounded::sequence::<_, _, 128>")]
    pub turns: Vec<SavedTurn>,
}
pub struct Store {
    db: Mutex<Connection>,
    limits: PersistenceLimits,
    _reservation: Reservation,
}
fn failure(error: impl std::fmt::Display) -> ApiError {
    Error::new(
        ErrorCode::Io,
        format!("conversation store failure: {error}; preserve existing data before recovery"),
    )
    .into()
}
impl Store {
    pub fn open(root: &Path, limits: PersistenceLimits, budget: &Budget) -> Result<Self> {
        limits.validate()?;
        let reservation = budget.reserve(16 << 20)?;
        let directory = root.join("conversations");
        auth::private_directory(&directory)?;
        if std::fs::symlink_metadata(&directory)?
            .file_type()
            .is_symlink()
        {
            return Err(ApiError::bad("conversation directory cannot be a symlink"));
        }
        let path = directory.join("history.sqlite");
        let existed = path.exists();
        if existed {
            let metadata = std::fs::symlink_metadata(&path)?;
            if !metadata.is_file()
                || metadata.file_type().is_symlink()
                || metadata.len() > limits.total_bytes / 2
            {
                return Err(ApiError::bad(
                    "conversation file is unsafe or exceeds the physical quota",
                ));
            }
        } else {
            auth::private_file(&path)?.sync_all()?;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::{MetadataExt, PermissionsExt};
            let m = std::fs::metadata(&path)?;
            if m.nlink() != 1 || m.permissions().mode() & 0o077 != 0 {
                return Err(ApiError::bad(
                    "conversation file must be private and have no extra hard links",
                ));
            }
        }
        let mut db = Connection::open(&path).map_err(failure)?;
        db.busy_timeout(std::time::Duration::from_secs(2))
            .map_err(failure)?;
        db.set_limit(
            rusqlite::limits::Limit::SQLITE_LIMIT_LENGTH,
            TRANSFER_BYTES as i32,
        )
        .map_err(failure)?;
        let app: i64 = db
            .query_row("PRAGMA application_id", [], |r| r.get(0))
            .map_err(failure)?;
        let version: i64 = db
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .map_err(failure)?;
        let initialize = !existed
            || (app == 0
                && version == 0
                && db
                    .query_row("SELECT count(*) FROM sqlite_master", [], |r| {
                        r.get::<_, i64>(0)
                    })
                    .map_err(failure)?
                    == 0);
        if !initialize && (app != APP_ID || !matches!(version, 1 | 2)) {
            return Err(ApiError::bad(
                "unsupported conversation store format; preserve the original database",
            ));
        }
        let integrity: String = db
            .query_row("PRAGMA quick_check(1)", [], |r| r.get(0))
            .map_err(failure)?;
        if integrity != "ok" {
            return Err(ApiError::bad(
                "conversation store is corrupt; restore a validated export into a clean store",
            ));
        }
        if initialize {
            db.execute_batch("PRAGMA auto_vacuum=FULL;")
                .map_err(failure)?;
        }
        db.execute_batch("PRAGMA trusted_schema=OFF; PRAGMA foreign_keys=ON; PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL; PRAGMA cache_size=-2048; PRAGMA temp_store=MEMORY;").map_err(failure)?;
        let page_size: i64 = db
            .query_row("PRAGMA page_size", [], |r| r.get(0))
            .map_err(failure)?;
        db.pragma_update(
            None,
            "max_page_count",
            ((limits.total_bytes / 2) / (page_size as u64)) as i64,
        )
        .map_err(failure)?;
        if initialize {
            let tx = db.transaction().map_err(failure)?;
            tx.execute_batch("CREATE TABLE sessions(id TEXT PRIMARY KEY, owner TEXT NOT NULL, meta TEXT NOT NULL, bytes INTEGER NOT NULL CHECK(bytes>=0)); CREATE INDEX session_owners ON sessions(owner); CREATE TABLE records(session TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, kind INTEGER NOT NULL, ordinal INTEGER NOT NULL, value TEXT NOT NULL, PRIMARY KEY(session,kind,ordinal)); PRAGMA user_version=2;").map_err(failure)?;
            tx.pragma_update(None, "application_id", APP_ID)
                .map_err(failure)?;
            tx.commit().map_err(failure)?;
        }
        if !initialize && version == 1 {
            // Require the known table layout before committing the additive format marker.
            db.prepare("SELECT id,owner,meta,bytes FROM sessions LIMIT 0")
                .map_err(failure)?;
            db.prepare("SELECT session,kind,ordinal,value FROM records LIMIT 0")
                .map_err(failure)?;
            let tx = db.transaction().map_err(failure)?;
            tx.pragma_update(None, "user_version", 2i64)
                .map_err(failure)?;
            tx.commit().map_err(failure)?;
        }
        Ok(Self {
            db: Mutex::new(db),
            limits,
            _reservation: reservation,
        })
    }
    pub fn index(&self) -> Result<Vec<(String, SessionView)>> {
        let db = self
            .db
            .lock()
            .map_err(|_| failure("database lock unavailable"))?;
        let mut stmt = db
            .prepare("SELECT owner,meta FROM sessions ORDER BY id LIMIT 129")
            .map_err(failure)?;
        let mut rows = stmt.query([]).map_err(failure)?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().map_err(failure)? {
            let owner: String = row.get(0).map_err(failure)?;
            let raw: String = row.get(1).map_err(failure)?;
            if raw.len() > 65536 || owner.len() != 64 {
                return Err(ApiError::bad("invalid persistent session metadata"));
            }
            out.push((owner, serde_json::from_str(&raw).map_err(failure)?));
        }
        Ok(out)
    }
    pub fn bytes(&self, id: &str) -> Result<u64> {
        let db = self
            .db
            .lock()
            .map_err(|_| failure("database lock unavailable"))?;
        let (bytes,actual):(i64,i64)=db.query_row("SELECT bytes,length(CAST(meta AS BLOB))+(SELECT coalesce(sum(length(CAST(value AS BLOB))),0) FROM records WHERE session=?1) FROM sessions WHERE id=?1",[id],|r|Ok((r.get(0)?,r.get(1)?))).map_err(failure)?;
        if bytes < 0 || bytes > DOCUMENT_BYTES as i64 || bytes != actual {
            return Err(ApiError::bad("stored history byte accounting is corrupt"));
        }
        Ok(bytes as u64)
    }
    pub fn page(&self, id: &str, offset: usize, limit: usize) -> Result<Vec<HistoryMessage>> {
        self.bytes(id)?;
        let db = self
            .db
            .lock()
            .map_err(|_| failure("database lock unavailable"))?;
        let mut statement=db.prepare("SELECT value FROM records WHERE session=?1 AND kind=0 ORDER BY ordinal LIMIT ?2 OFFSET ?3").map_err(failure)?;
        let rows = statement
            .query_map(params![id, limit as i64, offset as i64], |r| {
                r.get::<_, String>(0)
            })
            .map_err(failure)?;
        let mut messages = Vec::new();
        for row in rows {
            messages.push(serde_json::from_str(&row.map_err(failure)?).map_err(failure)?);
        }
        Ok(messages)
    }
    pub fn load(&self, id: &str) -> Result<ExportDocument> {
        let db = self
            .db
            .lock()
            .map_err(|_| failure("database lock unavailable"))?;
        let (raw, bytes): (String, i64) = db
            .query_row("SELECT meta,bytes FROM sessions WHERE id=?1", [id], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .optional()
            .map_err(failure)?
            .ok_or_else(|| ApiError::missing("stored session missing"))?;
        if bytes < 0 || bytes > DOCUMENT_BYTES as i64 || raw.len() > 65536 {
            return Err(ApiError::bad("stored history exceeds bounded format"));
        }
        let view: SessionView = serde_json::from_str(&raw).map_err(failure)?;
        let mut messages = Vec::new();
        let mut turns = Vec::new();
        let mut read = raw.len();
        // Bounded record pages: never deserialize the complete database or an unbounded row.
        for kind in [0, 1] {
            let mut offset = 0;
            loop {
                let mut stmt=db.prepare("SELECT value FROM records WHERE session=?1 AND kind=?2 ORDER BY ordinal LIMIT 32 OFFSET ?3").map_err(failure)?;
                let rows = stmt
                    .query_map(params![id, kind, offset], |r| r.get::<_, String>(0))
                    .map_err(failure)?;
                let mut count = 0;
                for row in rows {
                    let row = row.map_err(failure)?;
                    read = read
                        .checked_add(row.len())
                        .filter(|n| *n <= DOCUMENT_BYTES)
                        .ok_or_else(|| ApiError::bad("stored history exceeds byte limit"))?;
                    if kind == 0 {
                        messages.push(serde_json::from_str(&row).map_err(failure)?);
                    } else {
                        turns.push(serde_json::from_str(&row).map_err(failure)?);
                    }
                    count += 1;
                    if messages.len() > 256 || turns.len() > 128 {
                        return Err(ApiError::bad("stored history exceeds record limits"));
                    }
                }
                if count < 32 {
                    break;
                }
                offset += 32;
            }
        }
        if read as i64 != bytes {
            return Err(ApiError::bad("unknown or malformed stored history records"));
        }
        Ok(ExportDocument {
            version: 2,
            session: view,
            messages,
            turns,
        })
    }
    pub fn save(&self, owner: &str, doc: &ExportDocument) -> Result<()> {
        crate::registry::validate_hash(owner)?;
        let mut metadata = doc.session.clone();
        metadata.messages = None;
        metadata.message_offset = 0;
        metadata.next_offset = None;
        metadata.cached_tokens = 0;
        metadata.cache_reserved_bytes = 0;
        let meta = serde_json::to_string(&metadata).map_err(failure)?;
        let mut rows = Vec::new();
        let mut bytes = meta.len();
        for (kind, records) in [
            (
                0,
                doc.messages
                    .iter()
                    .map(serde_json::to_string)
                    .collect::<Vec<_>>(),
            ),
            (1, doc.turns.iter().map(serde_json::to_string).collect()),
        ] {
            for (ordinal, value) in records.into_iter().enumerate() {
                let value = value.map_err(failure)?;
                bytes = bytes
                    .checked_add(value.len())
                    .filter(|n| *n <= DOCUMENT_BYTES)
                    .ok_or_else(|| {
                        Error::resource("persistent session exceeds serialized byte quota")
                    })?;
                rows.push((kind, ordinal, value));
            }
        }
        let mut db = self
            .db
            .lock()
            .map_err(|_| failure("database lock unavailable"))?;
        let tx = db.transaction().map_err(failure)?;
        let existing: Option<String> = tx
            .query_row(
                "SELECT owner FROM sessions WHERE id=?1",
                [&metadata.id],
                |r| r.get(0),
            )
            .optional()
            .map_err(failure)?;
        if existing.as_deref().is_some_and(|stored| stored != owner) {
            return Err(ApiError::conflict("persistent session owner mismatch"));
        }
        let (total,owned):(i64,i64)=tx.query_row("SELECT coalesce(sum(bytes),0),coalesce(sum(CASE WHEN owner=?1 THEN bytes ELSE 0 END),0) FROM sessions WHERE id<>?2",params![owner,metadata.id],|r|Ok((r.get(0)?,r.get(1)?))).map_err(failure)?;
        if total < 0
            || owned < 0
            || total.saturating_add(bytes as i64) > (self.limits.total_bytes / 2) as i64
            || owned.saturating_add(bytes as i64) > self.limits.owner_bytes as i64
        {
            return Err(Error::resource("persistent owner/global quota exhausted").into());
        }
        tx.execute("INSERT INTO sessions(id,owner,meta,bytes) VALUES(?1,?2,?3,?4) ON CONFLICT(id) DO UPDATE SET meta=excluded.meta,bytes=excluded.bytes WHERE sessions.owner=excluded.owner",params![metadata.id,owner,meta,bytes as i64]).map_err(failure)?;
        for (kind, ordinal, value) in rows {
            tx.execute("INSERT INTO records(session,kind,ordinal,value) VALUES(?1,?2,?3,?4) ON CONFLICT(session,kind,ordinal) DO UPDATE SET value=excluded.value WHERE records.value<>excluded.value",params![metadata.id,kind,ordinal as i64,value]).map_err(failure)?;
        }
        tx.execute("DELETE FROM records WHERE session=?1 AND ((kind=0 AND ordinal>=?2) OR (kind=1 AND ordinal>=?3))",params![metadata.id,doc.messages.len() as i64,doc.turns.len() as i64]).map_err(failure)?;
        tx.commit().map_err(failure)
    }
    pub fn delete(&self, id: &str, owner: &str) -> Result<()> {
        let db = self
            .db
            .lock()
            .map_err(|_| failure("database lock unavailable"))?;
        db.execute(
            "DELETE FROM sessions WHERE id=?1 AND owner=?2",
            params![id, owner],
        )
        .map_err(failure)?;
        Ok(())
    }
}

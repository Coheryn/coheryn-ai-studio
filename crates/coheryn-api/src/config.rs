use coheryn_engine::{Error, Result};
use serde::Serialize;
use std::{
    net::{IpAddr, Ipv4Addr},
    path::PathBuf,
    time::Duration,
};

#[derive(Clone, Serialize)]
pub struct ServerConfig {
    pub persistence: crate::session_store::PersistenceLimits,
    pub cache: crate::sessions::CacheLimits,
    pub sessions: crate::sessions::SessionLimits,
    pub embeddings: crate::embeddings::EmbeddingLimits,
    pub host: IpAddr,
    pub port: u16,
    pub data_dir: PathBuf,
    pub max_memory: u64,
    pub max_context: usize,
    pub queue_capacity: usize,
    pub max_concurrency: usize,
    pub max_loaded_models: usize,
    pub max_model_bytes: u64,
    pub disk_quota: u64,
    pub queue_timeout_seconds: u64,
    pub generation_timeout_seconds: u64,
    pub consumer_timeout_seconds: u64,
    pub embedding_timeout_seconds: u64,
}
pub fn default_data_dir() -> PathBuf {
    if cfg!(windows) {
        return std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."))
            .join("CoherynAIStudio");
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    if cfg!(target_os = "macos") {
        return home.join("Library/Application Support/CoherynAIStudio");
    }
    std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .unwrap_or_else(|| home.join(".local/share"))
        .join("coheryn")
}
impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            persistence: Default::default(),
            cache: Default::default(),
            sessions: Default::default(),
            embeddings: Default::default(),
            host: IpAddr::V4(Ipv4Addr::LOCALHOST),
            port: 11450,
            data_dir: default_data_dir(),
            max_memory: 1 << 30,
            max_context: 8192,
            queue_capacity: 8,
            max_concurrency: 1,
            max_loaded_models: 4,
            max_model_bytes: 4 << 30,
            disk_quota: 20 << 30,
            queue_timeout_seconds: 30,
            generation_timeout_seconds: 300,
            consumer_timeout_seconds: 30,
            embedding_timeout_seconds: 300,
        }
    }
}
impl ServerConfig {
    pub fn validate(&self) -> Result<()> {
        self.sessions.validate()?;
        self.cache.validate()?;
        self.persistence.validate()?;
        self.embeddings.validate()?;
        if self.embeddings.concurrency > self.max_concurrency
            || self.embedding_timeout_seconds == 0
            || self.embedding_timeout_seconds > 86400
        {
            return Err(Error::request(
                "embedding concurrency/timeout exceeds server policy",
            ));
        }
        if self.max_memory == 0
            || self.max_context == 0
            || self.max_context > 1_048_576
            || self.queue_capacity > 128
            || self.max_concurrency == 0
            || self.max_concurrency > 16
            || self.max_loaded_models == 0
            || self.max_loaded_models > 128
            || self.max_model_bytes == 0
            || self.max_model_bytes > self.disk_quota
            || self.queue_timeout_seconds == 0
            || self.generation_timeout_seconds == 0
            || self.consumer_timeout_seconds == 0
            || self.queue_timeout_seconds > 86400
            || self.generation_timeout_seconds > 86400
            || self.consumer_timeout_seconds > 86400
        {
            return Err(Error::request("invalid server resource limits"));
        }
        Ok(())
    }
    pub fn queue_timeout(&self) -> Duration {
        Duration::from_secs(self.queue_timeout_seconds)
    }
}

//! Headless service using the same native engine as standalone execution.
pub mod auth;
mod bounded;
pub mod checkpoint;
pub mod client;
pub mod config;
pub mod download;
pub mod embeddings;
pub mod error;
pub mod hardware;
pub mod http;
pub mod registry;
pub mod resident;
pub mod scheduler;
pub mod service;
mod session_http;
pub mod session_store;
pub mod sessions;
pub mod vision;
mod vision_planner;
pub mod wire;

#[cfg(test)]
#[path = "../../coheryn-engine/tests/common/mod.rs"]
pub(crate) mod test_model;

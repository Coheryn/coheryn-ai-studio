use crate::embeddings::EmbeddingSpace;
use coheryn_engine::{Descriptor, Model, embedding::EmbeddingModel};
use std::sync::Arc;

pub struct ResidentEmbedding {
    pub model: EmbeddingModel,
    pub space: EmbeddingSpace,
}
#[derive(Clone)]
pub enum Resident {
    Generation(Arc<Model>),
    Embedding(Arc<ResidentEmbedding>),
}
impl Resident {
    pub fn context(&self) -> usize {
        match self {
            Self::Generation(m) => m.context_limit(),
            Self::Embedding(m) => m.model.context_limit(),
        }
    }
    pub fn descriptor(&self) -> &Descriptor {
        match self {
            Self::Generation(m) => &m.descriptor,
            Self::Embedding(m) => m.model.descriptor(),
        }
    }
    pub fn pinned(&self) -> bool {
        match self {
            Self::Generation(m) => Arc::strong_count(m) > 1,
            Self::Embedding(m) => Arc::strong_count(m) > 1,
        }
    }
    pub fn task(&self) -> &'static str {
        match self {
            Self::Generation(_) => "text-generation",
            Self::Embedding(_) => "embedding",
        }
    }
}

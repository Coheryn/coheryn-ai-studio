use crate::embeddings::EmbeddingSpace;
use coheryn_engine::{Descriptor, Model, embedding::EmbeddingModel};
use std::sync::Arc;

/// Workspace contract for bounded history preparation, independent of KV/cache layout.
pub trait TextConversation: Send + Sync {
    fn context_limit(&self) -> usize;
    fn supports_history(&self) -> bool;
    fn render_history(
        &self,
        messages: &[coheryn_engine::tokenizer::Message],
        tools: &coheryn_engine::tools::ToolConfig,
    ) -> coheryn_engine::Result<String>;
    fn token_count(&self, prompt: &str) -> coheryn_engine::Result<usize>;
    fn validate_options(
        &self,
        options: &coheryn_engine::GenerationOptions,
    ) -> coheryn_engine::Result<()>;
    fn render_instructions(&self, system: &str) -> coheryn_engine::Result<String> {
        self.render_history(
            &[coheryn_engine::tokenizer::Message {
                role: "system".into(),
                content: system.into(),
                ..Default::default()
            }],
            &Default::default(),
        )
    }
}
impl TextConversation for Model {
    fn context_limit(&self) -> usize {
        self.context_limit()
    }
    fn supports_history(&self) -> bool {
        self.tokenizer.supports_conversations()
    }
    fn render_history(
        &self,
        messages: &[coheryn_engine::tokenizer::Message],
        tools: &coheryn_engine::tools::ToolConfig,
    ) -> coheryn_engine::Result<String> {
        self.tokenizer.render_tools(messages, tools)
    }
    fn token_count(&self, prompt: &str) -> coheryn_engine::Result<usize> {
        Ok(self.tokenizer.encode(prompt)?.len())
    }
    fn validate_options(
        &self,
        _: &coheryn_engine::GenerationOptions,
    ) -> coheryn_engine::Result<()> {
        Ok(())
    }
}
impl TextConversation for coheryn_engine::qwen35::Model {
    fn context_limit(&self) -> usize {
        self.context_limit()
    }
    fn supports_history(&self) -> bool {
        true
    }
    fn render_history(
        &self,
        messages: &[coheryn_engine::tokenizer::Message],
        tools: &coheryn_engine::tools::ToolConfig,
    ) -> coheryn_engine::Result<String> {
        if tools.enabled() {
            return Err(coheryn_engine::Error::new(
                coheryn_engine::ErrorCode::UnsupportedFeature,
                "Qwen3.5 history profile does not support tools",
            ));
        }
        if messages.len() > 256 {
            return Err(coheryn_engine::Error::resource(
                "Qwen3.5 history exceeds the template message bound",
            ));
        }
        self.render(messages)
    }
    fn token_count(&self, prompt: &str) -> coheryn_engine::Result<usize> {
        self.count_tokens(prompt)
    }
    fn validate_options(
        &self,
        options: &coheryn_engine::GenerationOptions,
    ) -> coheryn_engine::Result<()> {
        Self::check_options(options)
    }
    fn render_instructions(&self, system: &str) -> coheryn_engine::Result<String> {
        self.render(&[
            coheryn_engine::tokenizer::Message {
                role: "system".into(),
                content: system.into(),
                ..Default::default()
            },
            coheryn_engine::tokenizer::Message {
                role: "user".into(),
                content: " ".into(),
                ..Default::default()
            },
        ])
    }
}
#[derive(Clone)]
pub enum TextResident {
    Gguf(Arc<Model>),
    Qwen35(Arc<coheryn_engine::qwen35::Model>),
}
impl TextResident {
    pub fn conversation(&self) -> &dyn TextConversation {
        match self {
            Self::Gguf(m) => m.as_ref(),
            Self::Qwen35(m) => m.as_ref(),
        }
    }
}

pub struct ResidentEmbedding {
    pub model: EmbeddingModel,
    pub space: EmbeddingSpace,
}
#[derive(Clone)]
pub enum Resident {
    Generation(Arc<Model>),
    Qwen35(Arc<coheryn_engine::qwen35::Model>),
    Vision(Arc<coheryn_engine::multimodal::qwen2vl::Qwen2Vl>),
    Embedding(Arc<ResidentEmbedding>),
}
impl Resident {
    pub fn conversation_capabilities(&self) -> Option<serde_json::Value> {
        match self {
            Self::Generation(m) if m.tokenizer.supports_conversations() => Some(
                serde_json::json!({"histories":true,"prefix_kv_reuse":true,"execution":"prefix-cache"}),
            ),
            Self::Qwen35(model) => Some(
                serde_json::json!({"image_histories":model.vision_configuration().is_ok(),"histories":true,"prefix_kv_reuse":true,"recurrent_state_reuse":true,"execution":"bounded-hybrid-prefix-cache"}),
            ),
            Self::Vision(_) => Some(
                serde_json::json!({"histories":true,"prefix_kv_reuse":false,"execution":"full-prefill"}),
            ),
            _ => None,
        }
    }

    pub fn context(&self) -> usize {
        match self {
            Self::Generation(m) => m.context_limit(),
            Self::Qwen35(m) => m.context_limit(),
            Self::Vision(m) => m.context_limit(),
            Self::Embedding(m) => m.model.context_limit(),
        }
    }
    pub fn descriptor(&self) -> Option<&Descriptor> {
        match self {
            Self::Generation(m) => Some(&m.descriptor),
            Self::Qwen35(_) => None,
            Self::Vision(m) => Some(m.descriptor()),
            Self::Embedding(m) => Some(m.model.descriptor()),
        }
    }
    pub fn pinned(&self) -> bool {
        match self {
            Self::Generation(m) => Arc::strong_count(m) > 1,
            Self::Qwen35(m) => Arc::strong_count(m) > 1,
            Self::Vision(m) => Arc::strong_count(m) > 1,
            Self::Embedding(m) => Arc::strong_count(m) > 1,
        }
    }
    pub fn task(&self) -> &'static str {
        match self {
            Self::Generation(_) | Self::Qwen35(_) => "text-generation",
            Self::Vision(_) => "image-to-text",
            Self::Embedding(_) => "embedding",
        }
    }
}

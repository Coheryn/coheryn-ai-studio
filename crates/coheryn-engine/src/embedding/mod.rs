//! Independent GGUF encoder inference. No generation model or external runtime.
mod config;
mod execution;
mod plan;
mod tokenizer;

pub use config::EmbeddingConfig;
pub use execution::{
    EmbeddingAdmission, EmbeddingCheckpoint, EmbeddingItem, EmbeddingJob, EmbeddingModel,
    EmbeddingResult, EmbeddingUsage,
};
pub use plan::{EmbeddingPlan, EmbeddingPlanInput, estimate, model_bytes as estimate_model_bytes};
pub use tokenizer::EmbeddingTokenizer;

use crate::{Error, Result, planner::Profile};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EmbeddingOptions {
    pub max_input_tokens: usize,
    pub truncate: bool,
    pub batch_size: usize,
    pub max_batch_tokens: usize,
    pub profile: Profile,
    pub threads: Option<usize>,
}
impl Default for EmbeddingOptions {
    fn default() -> Self {
        Self {
            max_input_tokens: 512,
            truncate: false,
            batch_size: 8,
            max_batch_tokens: 2048,
            profile: Profile::Balanced,
            threads: None,
        }
    }
}
impl EmbeddingOptions {
    pub fn validate(&self, context: usize) -> Result<()> {
        if !(2..=context.min(512)).contains(&self.max_input_tokens)
            || !(1..=64).contains(&self.batch_size)
            || self.max_batch_tokens < self.max_input_tokens
            || self.max_batch_tokens > 4096
            || self.threads.is_some_and(|n| n == 0 || n > 16)
        {
            return Err(Error::request(
                "embedding limits require 2..=512 input tokens, 1..=64 batch items, and input_limit..=4096 batch tokens",
            ));
        }
        Ok(())
    }
}

pub(crate) fn compatibility(d: &crate::Descriptor) -> crate::CompatibilityReport {
    let mut diagnostics = Vec::new();
    let embedding_config = match config::configuration(d) {
        Ok(c) => Some(c),
        Err(e) => {
            diagnostics.push(e);
            None
        }
    };
    if let Err(e) = EmbeddingTokenizer::check_metadata(d) {
        diagnostics.push(e);
    }
    crate::CompatibilityReport {
        tool_protocol: None,
        container_valid: true,
        executable: diagnostics.is_empty(),
        qualification: "experimental-unqualified",
        backend: "cpu-reference",
        diagnostics,
        config: None,
        task: Some(crate::compat::ModelTask::Embedding),
        embedding_config,
    }
}

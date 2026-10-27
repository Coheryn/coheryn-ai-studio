use crate::error::{ApiError, Result};
use coheryn_engine::{Error, GenerationOptions, sampling::Sampling, tokenizer::Message};
use serde::{Deserialize, Serialize};

fn max_tokens() -> usize {
    128
}
fn top_p() -> f32 {
    1.
}
fn one() -> usize {
    1
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Stops {
    One(String),
    Many(Vec<String>),
}
#[derive(Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationRequest {
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<Message>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::bounded::optional_sequence::<_,_,16>"
    )]
    pub tools: Option<Vec<coheryn_engine::tools::Tool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<coheryn_engine::tools::ToolChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default = "max_tokens", alias = "max_completion_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub temperature: f32,
    #[serde(default = "top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub top_k: usize,
    #[serde(default)]
    pub seed: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<Stops>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "one")]
    pub n: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    /// Coheryn extension: explicit request capacity, never larger than loaded capacity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_memory: Option<u64>,
    #[serde(default = "balanced")]
    pub profile: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threads: Option<usize>,
}
impl GenerationRequest {
    pub fn completion(
        model: String,
        prompt: String,
        options: &GenerationOptions,
        context: usize,
        stream: bool,
    ) -> Self {
        Self {
            model,
            prompt: Some(prompt),
            messages: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            max_tokens: options.max_tokens,
            temperature: options.sampling.temperature,
            top_p: options.sampling.top_p,
            top_k: options.sampling.top_k,
            seed: options.sampling.seed,
            stop: Some(Stops::Many(options.stop.clone())),
            stream,
            n: 1,
            stream_options: None,
            context: Some(context),
            max_memory: None,
            profile: balanced(),
            threads: None,
        }
    }
    pub fn validate(&self) -> Result<()> {
        crate::registry::validate_id(&self.model)?;
        coheryn_engine::planner::Profile::parse(&self.profile)?;
        if self.threads.is_some_and(|n| n == 0 || n > 16) {
            return Err(ApiError::bad("threads must be in 1..=16"));
        }
        if self.max_tokens == 0 || self.temperature > 2.0 {
            return Err(ApiError::bad(
                "max_tokens must be positive and HTTP temperature must be in [0,2]",
            ));
        }
        if self.prompt.is_some() == self.messages.is_some() {
            return Err(ApiError::bad("supply exactly one of prompt or messages"));
        }
        if self.n != 1 {
            return Err(ApiError::bad("only n=1 is implemented"));
        }
        if self.stream_options.is_some() && !self.stream {
            return Err(ApiError::bad("stream_options requires stream=true"));
        }
        if self
            .prompt
            .as_ref()
            .is_some_and(|p| p.is_empty() || p.len() > 1 << 20)
        {
            return Err(ApiError::bad("prompt must contain 1..=1048576 bytes"));
        }
        self.tool_config().validate()?;
        if let Some(messages) = &self.messages {
            self.tool_config().validate_messages(messages, false)?;
        } else if self.tools.is_some()
            || self.tool_choice.is_some()
            || self.parallel_tool_calls.is_some()
        {
            return Err(ApiError::bad("tools require chat messages"));
        }
        let options = self.options();
        if self.tool_config().enabled() && !options.stop.is_empty() {
            return Err(Error::new(
                coheryn_engine::ErrorCode::UnsupportedFeature,
                "custom stop sequences are not qualified with tool generation",
            )
            .into());
        }
        coheryn_engine::sampling::Sampler::new(options.sampling)?;
        if options.stop.len() > 16 || options.stop.iter().any(|s| s.is_empty() || s.len() > 4096) {
            return Err(ApiError::bad("invalid stop sequences"));
        }
        Ok(())
    }
    pub fn tool_config(&self) -> coheryn_engine::tools::ToolConfig {
        coheryn_engine::tools::ToolConfig {
            tools: self.tools.clone().unwrap_or_default(),
            tool_choice: self.tool_choice.clone(),
            parallel_tool_calls: self.parallel_tool_calls,
        }
    }
    pub fn options(&self) -> GenerationOptions {
        GenerationOptions {
            tool_markers: self.tool_config().enabled(),
            max_tokens: self.max_tokens,
            sampling: Sampling {
                temperature: self.temperature,
                top_k: self.top_k,
                top_p: self.top_p,
                seed: self.seed,
            },
            stop: match &self.stop {
                None => vec![],
                Some(Stops::One(s)) => vec![s.clone()],
                Some(Stops::Many(s)) => s.clone(),
            },
        }
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRequest {
    pub model: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoadRequest {
    pub model: String,
    #[serde(default = "default_context")]
    pub context: usize,
}
fn default_context() -> usize {
    4096
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidateRequest {
    pub model: String,
    #[serde(default)]
    pub deep: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PullRequest {
    pub name: String,
    pub url: String,
    pub sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportQuery {
    pub name: String,
    pub sha256: Option<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EstimateRequest {
    #[serde(default)]
    pub task: Option<String>,
    #[serde(default)]
    pub input_bytes: Option<Vec<usize>>,
    #[serde(default)]
    pub max_input_tokens: Option<usize>,
    #[serde(default)]
    pub batch_size: Option<usize>,
    #[serde(default)]
    pub max_batch_tokens: Option<usize>,
    pub model: String,
    pub context: usize,
    #[serde(default = "balanced")]
    pub profile: String,
    #[serde(default = "one")]
    pub concurrency: usize,
    #[serde(default)]
    pub images: Option<coheryn_engine::planner::ImageWorkload>,
    #[serde(default)]
    pub threads: Option<usize>,
}
fn balanced() -> String {
    "balanced".into()
}

//! Shared embedding transport contracts, immutable space identity and bounded serialization.
use crate::error::{ApiError, Result};
use base64::{Engine, engine::general_purpose::STANDARD};
use coheryn_engine::{
    Descriptor, Error, ErrorCode,
    embedding::{EmbeddingConfig, EmbeddingOptions, EmbeddingResult},
    planner::Profile,
};
use serde::{
    Deserialize, Serialize,
    ser::{SerializeMap, SerializeSeq},
};
use sha2::{Digest, Sha256};
use std::io::{self, Write};

#[derive(Clone, Debug, Serialize)]
pub struct EmbeddingLimits {
    pub requests: usize,
    pub concurrency: usize,
    pub body_bytes: usize,
    pub items: usize,
    pub input_bytes: usize,
    pub request_bytes: usize,
    pub input_tokens: usize,
    pub total_tokens: usize,
    pub batch_size: usize,
    pub batch_tokens: usize,
    pub response_bytes: usize,
}
impl Default for EmbeddingLimits {
    fn default() -> Self {
        Self {
            requests: 4,
            concurrency: 1,
            body_bytes: 1 << 20,
            items: 256,
            input_bytes: 64 << 10,
            request_bytes: 8 << 20,
            input_tokens: 512,
            total_tokens: 32768,
            batch_size: 8,
            batch_tokens: 2048,
            response_bytes: 16 << 20,
        }
    }
}
impl EmbeddingLimits {
    pub fn validate(&self) -> coheryn_engine::Result<()> {
        if !(1..=128).contains(&self.requests)
            || !(1..=16).contains(&self.concurrency)
            || !(1024..=16 << 20).contains(&self.body_bytes)
            || !(1..=256).contains(&self.items)
            || !(1..=64 << 10).contains(&self.input_bytes)
            || !(1..=8 << 20).contains(&self.request_bytes)
            || !(2..=512).contains(&self.input_tokens)
            || !(2..=131072).contains(&self.total_tokens)
            || !(1..=64).contains(&self.batch_size)
            || self.batch_tokens < self.input_tokens
            || self.batch_tokens > 4096
            || !(1024..=32 << 20).contains(&self.response_bytes)
        {
            return Err(Error::request("invalid embedding resource limits"));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    One(String),
    Many(Vec<String>),
}
impl EmbeddingInput {
    pub fn texts(&self) -> &[String] {
        match self {
            Self::One(s) => std::slice::from_ref(s),
            Self::Many(v) => v,
        }
    }
}
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Encoding {
    #[default]
    Float,
    Base64,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
    #[serde(default)]
    pub encoding_format: Encoding,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_size: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_input_tokens: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_batch_tokens: Option<usize>,
    #[serde(default)]
    pub truncate: bool,
    #[serde(default)]
    pub profile: Profile,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threads: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_memory: Option<u64>,
}
impl EmbeddingRequest {
    pub fn new(model: String, inputs: Vec<String>, options: &EmbeddingOptions) -> Self {
        Self {
            model,
            input: EmbeddingInput::Many(inputs),
            encoding_format: Encoding::Float,
            dimensions: None,
            user: None,
            batch_size: Some(options.batch_size),
            max_input_tokens: Some(options.max_input_tokens),
            max_batch_tokens: Some(options.max_batch_tokens),
            truncate: options.truncate,
            profile: options.profile,
            threads: options.threads,
            max_memory: None,
        }
    }
    pub fn validate(&self, limits: &EmbeddingLimits) -> Result<()> {
        crate::registry::validate_id(&self.model)?;
        let texts = self.input.texts();
        if texts.is_empty() {
            return Err(ApiError::bad("embedding input must not be empty"));
        }
        if texts.len() > limits.items {
            return Err(batch_limit("embedding item count exceeds configured limit"));
        }
        let mut total = 0usize;
        for (index, text) in texts.iter().enumerate() {
            if text.is_empty() {
                return Err(ApiError::bad(format!(
                    "empty embedding input at index {index}"
                )));
            }
            if text.len() > limits.input_bytes {
                return Err(input_limit(format!(
                    "input at index {index} exceeds UTF-8 byte limit"
                )));
            }
            total = total
                .checked_add(text.len())
                .ok_or_else(|| batch_limit("input byte count overflow"))?;
        }
        if total > limits.request_bytes {
            return Err(batch_limit(
                "aggregate embedding input bytes exceed configured limit",
            ));
        }
        if self.user.as_ref().is_some_and(|s| s.len() > 256) {
            return Err(ApiError::bad("user identifier exceeds 256 bytes"));
        }
        Ok(())
    }
    pub fn options(&self, context: usize, limits: &EmbeddingLimits) -> Result<EmbeddingOptions> {
        let options = EmbeddingOptions {
            max_input_tokens: self
                .max_input_tokens
                .unwrap_or(context.min(limits.input_tokens)),
            truncate: self.truncate,
            batch_size: self.batch_size.unwrap_or(limits.batch_size),
            max_batch_tokens: self.max_batch_tokens.unwrap_or(limits.batch_tokens),
            profile: self.profile,
            threads: self.threads,
        };
        if options.max_input_tokens > limits.input_tokens
            || options.batch_size > limits.batch_size
            || options.max_batch_tokens > limits.batch_tokens
        {
            return Err(batch_limit(
                "requested embedding limits exceed operator policy",
            ));
        }
        options.validate(context)?;
        Ok(options)
    }
}
pub fn input_limit(message: impl Into<String>) -> ApiError {
    Error::new(ErrorCode::EmbeddingInputLimitExceeded, message).into()
}
pub fn batch_limit(message: impl Into<String>) -> ApiError {
    Error::new(ErrorCode::EmbeddingBatchLimitExceeded, message).into()
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingSpace {
    pub id: String,
    pub model_sha256: String,
    pub dimension: usize,
    pub tokenizer: String,
    pub pooling: String,
    pub normalization: String,
    pub instructions: String,
}
impl EmbeddingSpace {
    pub fn new(hash: &str, d: &Descriptor, c: &EmbeddingConfig) -> coheryn_engine::Result<Self> {
        crate::registry::validate_hash(hash).map_err(|e| e.error)?;
        let mut space = Self {
            id: String::new(),
            model_sha256: hash.to_owned(),
            dimension: c.hidden,
            tokenizer: "gguf-bert-first-nfd-mark-strip-v1;literal-specials;CLS/SEP;types=0".into(),
            pooling: "mean-real-tokens-including-CLS-SEP".into(),
            normalization: "L2-full-dimension".into(),
            instructions: "caller-supplied-literal-prefixes;no-implicit-instruction".into(),
        };
        let identity = serde_json::to_vec(&(&space, &d.header_sha256, &d.metadata_fingerprints))
            .map_err(|_| {
                Error::new(
                    ErrorCode::Internal,
                    "embedding identity serialization failed",
                )
            })?;
        space.id = format!(
            "coheryn-embedding-v1-{}",
            coheryn_engine::storage::digest_hex(Sha256::digest(identity).into())
        );
        Ok(space)
    }
}
/// Conservative upper bound, including JSON syntax and temporary per-vector Base64 bytes.
pub fn response_allowance(items: usize, dimension: usize, limit: usize) -> Result<u64> {
    if items == 0 || items > 256 || dimension == 0 || dimension > 4096 {
        return Err(batch_limit(
            "embedding response shape exceeds bounded transport dimensions",
        ));
    }
    let bytes = items
        .checked_mul(dimension)
        .and_then(|n| n.checked_mul(32))
        .and_then(|n| n.checked_add(items * 256 + 16384))
        .ok_or_else(|| batch_limit("embedding response estimate overflow"))?;
    if bytes > limit {
        return Err(batch_limit(
            "embedding response exceeds configured output limit; reduce input batch size",
        ));
    }
    Ok(bytes as u64)
}

pub struct BoundedWriter<W> {
    pub inner: W,
    pub written: u64,
    pub limit: u64,
    pub exceeded: bool,
}
impl<W> BoundedWriter<W> {
    pub fn new(inner: W, limit: u64) -> Self {
        Self {
            inner,
            written: 0,
            limit,
            exceeded: false,
        }
    }
}
impl<W: Write> Write for BoundedWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if (bytes.len() as u64) > self.limit.saturating_sub(self.written) {
            self.exceeded = true;
            return Err(io::Error::other("embedding output quota exceeded"));
        }
        let n = self.inner.write(bytes)?;
        self.written += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}
pub struct ItemView<'a> {
    pub index: usize,
    pub values: &'a [f32],
    pub encoding: Encoding,
}
impl Serialize for ItemView<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let mut map = s.serialize_map(Some(3))?;
        map.serialize_entry("object", "embedding")?;
        map.serialize_entry("index", &self.index)?;
        match self.encoding {
            Encoding::Float => map.serialize_entry("embedding", self.values)?,
            Encoding::Base64 => {
                let bytes: Vec<u8> = self.values.iter().flat_map(|v| v.to_le_bytes()).collect();
                map.serialize_entry("embedding", &STANDARD.encode(bytes))?;
            }
        }
        map.end()
    }
}
struct Items<'a>(&'a EmbeddingResult, Encoding);
impl Serialize for Items<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(self.0.data.len()))?;
        for item in &self.0.data {
            seq.serialize_element(&ItemView {
                index: item.index,
                values: &item.embedding,
                encoding: self.1,
            })?;
        }
        seq.end()
    }
}
pub fn serialize_response(
    result: &EmbeddingResult,
    model: &str,
    encoding: Encoding,
    allowance: u64,
) -> Result<Vec<u8>> {
    #[derive(Serialize)]
    struct Response<'a> {
        object: &'static str,
        data: Items<'a>,
        model: &'a str,
        usage: &'a coheryn_engine::embedding::EmbeddingUsage,
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(allowance as usize)
        .map_err(|_| Error::resource("embedding response allocation failed"))?;
    let mut out = BoundedWriter::new(bytes, allowance);
    serde_json::to_writer(
        &mut out,
        &Response {
            object: "list",
            data: Items(result, encoding),
            model,
            usage: &result.usage,
        },
    )
    .map_err(|_| {
        Error::resource("embedding response serialization failed or exceeded its reservation")
    })?;
    Ok(out.inner)
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Vector {
    Float(Vec<f32>),
    Base64(String),
}
impl Vector {
    pub fn floats(&self) -> coheryn_engine::Result<Vec<f32>> {
        match self {
            Self::Float(v) => Ok(v.clone()),
            Self::Base64(s) => {
                let bytes = STANDARD
                    .decode(s)
                    .map_err(|_| Error::request("invalid Base64 embedding"))?;
                if !bytes.len().is_multiple_of(4) {
                    return Err(Error::request("invalid embedding byte width"));
                }
                Ok(bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect())
            }
        }
    }
}
#[derive(Debug, Deserialize, Serialize)]
pub struct ResponseItem {
    pub object: String,
    pub index: usize,
    pub embedding: Vector,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub model: String,
    pub data: Vec<ResponseItem>,
    pub usage: Usage,
}

pub fn parser_allowance(body_bytes: usize) -> u64 {
    (body_bytes as u64) * 64 + (1 << 20)
}

use crate::{Descriptor, Error, ErrorCode, Result, gguf::Value};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, io::Write};
use tokenizers::{
    AddedToken, SplitDelimiterBehavior, Tokenizer,
    models::bpe::BPE,
    pre_tokenizers::{
        byte_level::ByteLevel,
        split::{Split, SplitPattern},
    },
};

// The Qwen2 byte-level BPE pretokenization expression, not generic GPT-2 splitting.
const QWEN_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

fn unsupported(message: impl Into<String>) -> Error {
    Error::new(ErrorCode::UnsupportedTokenizer, message)
}
pub fn check_metadata(d: &Descriptor) -> Result<()> {
    if d.text("tokenizer.ggml.model") != Some("gpt2")
        || d.text("tokenizer.ggml.pre") != Some("qwen2")
    {
        return Err(unsupported(
            "only GGUF gpt2/qwen2 byte-level BPE is implemented",
        ));
    }
    for key in [
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.merges",
        "tokenizer.ggml.token_type",
    ] {
        if d.get(key).and_then(Value::array).is_none() {
            return Err(unsupported(format!("missing {key}")));
        }
    }
    for key in [
        "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.add_eos_token",
        "tokenizer.ggml.add_space_prefix",
    ] {
        if matches!(d.get(key), Some(Value::Bool(true))) {
            return Err(unsupported(format!("{key}=true is not qualified")));
        }
    }
    let template = d.text("tokenizer.chat_template").ok_or_else(|| {
        Error::new(
            ErrorCode::UnsupportedChatTemplate,
            "missing tokenizer.chat_template",
        )
    })?;
    if template.len() > 16384 {
        return Err(Error::new(
            ErrorCode::UnsupportedChatTemplate,
            "template exceeds the G1 16 KiB source limit",
        ));
    }
    // Only known bounded programs execute; fuel alone cannot bound huge string expressions.
    let hash = crate::storage::digest_hex(
        Sha256::digest(serde_json::to_vec(template).map_err(|e| unsupported(e.to_string()))?)
            .into(),
    );
    if hash != "71672240b1a3b068067b6a332bc38d3fc05aaade9d41b4b27d7510ff4335504b"
        && template != "{{ messages[0].content }}"
    {
        return Err(Error::new(
            ErrorCode::UnsupportedChatTemplate,
            "template fingerprint has not been implemented; only the pinned Qwen text template and fixed passthrough fixture are accepted",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Message {
    pub role: String,
    #[serde(default, deserialize_with = "nullable_content")]
    pub content: String,
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        deserialize_with = "crate::tools::bounded::<_,_,8>"
    )]
    pub tool_calls: Vec<crate::tools::ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}
fn nullable_content<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<String, D::Error> {
    Ok(Option::<String>::deserialize(d)?.unwrap_or_default())
}

pub struct TextTokenizer {
    metadata_identity: String,
    inner: Tokenizer,
    template: String,
    pub eos: u32,
    pub eot: Option<u32>,
    end_of_text: Option<u32>,
    vocabulary: usize,
    token_bytes: Vec<Vec<u8>>,
    special: Vec<bool>,
}

fn byte_decoder() -> HashMap<char, u8> {
    let mut allowed: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let original = allowed.clone();
    let mut extra = 0;
    let mut bytes = allowed.clone();
    for b in 0..256 {
        if !original.contains(&b) {
            bytes.push(b);
            allowed.push(256 + extra);
            extra += 1;
        }
    }
    allowed
        .into_iter()
        .zip(bytes)
        .map(|(c, b)| (char::from_u32(c).unwrap(), b as u8))
        .collect()
}

impl TextTokenizer {
    pub fn from_descriptor(d: &Descriptor) -> Result<Self> {
        check_metadata(d)?;
        let tokens = d
            .get("tokenizer.ggml.tokens")
            .and_then(Value::array)
            .unwrap();
        let types = d
            .get("tokenizer.ggml.token_type")
            .and_then(Value::array)
            .unwrap();
        if types.len() != tokens.len() {
            return Err(unsupported("token type count differs from vocabulary"));
        }
        let mut vocab = HashMap::with_capacity(tokens.len());
        let mut special = Vec::with_capacity(tokens.len());
        for (i, (token, ty)) in tokens.iter().zip(types).enumerate() {
            let text = token
                .string()
                .ok_or_else(|| unsupported("vocabulary contains a non-string"))?;
            if text.len() > 4096 {
                return Err(unsupported("token text exceeds 4096-byte G1 limit"));
            }
            if vocab.insert(text.to_owned(), i as u32).is_some() {
                return Err(unsupported("duplicate token text"));
            }
            // GGUF NORMAL=1, CONTROL=3, USER_DEFINED=4, UNUSED=5, BYTE=6.
            let ty = ty
                .integer()
                .ok_or_else(|| unsupported("invalid token type"))?;
            if !matches!(ty, 1 | 3 | 4 | 5 | 6) {
                return Err(unsupported("unsupported token type"));
            }
            special.push(matches!(ty, 3 | 4));
        }
        let merges = d
            .get("tokenizer.ggml.merges")
            .and_then(Value::array)
            .unwrap()
            .iter()
            .map(|v| {
                let (a, b) = v
                    .string()
                    .and_then(|s| s.split_once(' '))
                    .ok_or_else(|| unsupported("invalid BPE merge"))?;
                Ok((a.to_string(), b.to_string()))
            })
            .collect::<Result<Vec<_>>>()?;
        let vocab: tokenizers::models::bpe::Vocab = vocab.into_iter().collect();
        let model = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .cache_capacity(0)
            .build()
            .map_err(|e| unsupported(e.to_string()))?;
        let mut inner = Tokenizer::new(model);
        let split = Split::new(
            SplitPattern::Regex(QWEN_PATTERN.to_owned()),
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .map_err(|e| unsupported(e.to_string()))?;
        inner.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::sequence::Sequence::new(
            vec![split.into(), ByteLevel::new(false, true, false).into()],
        )));
        for (i, t) in tokens.iter().enumerate() {
            if special[i] {
                inner
                    .add_special_tokens([AddedToken::from(t.string().unwrap().to_string(), true)])
                    .map_err(|e| unsupported(e.to_string()))?;
            }
        }
        let eos = d
            .uint("tokenizer.ggml.eos_token_id")
            .filter(|n| *n < tokens.len() as u64)
            .ok_or_else(|| unsupported("missing or invalid EOS token"))? as u32;
        let eot = match d.uint("tokenizer.ggml.eot_token_id") {
            Some(n) if n < tokens.len() as u64 => Some(n as u32),
            Some(_) => return Err(unsupported("invalid EOT token")),
            None => None,
        };
        let end_of_text = tokens
            .iter()
            .position(|t| t.string() == Some("<|endoftext|>"))
            .map(|i| i as u32);
        let decoder = byte_decoder();
        let token_bytes = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let text = t.string().unwrap();
                if special[i] {
                    Ok(text.as_bytes().to_vec())
                } else {
                    text.chars()
                        .map(|c| {
                            decoder
                                .get(&c)
                                .copied()
                                .ok_or_else(|| unsupported("invalid byte-level token character"))
                        })
                        .collect()
                }
            })
            .collect::<Result<Vec<Vec<u8>>>>()?;
        Ok(Self {
            metadata_identity: d.header_sha256.clone(),
            inner,
            template: d.text("tokenizer.chat_template").unwrap().to_owned(),
            eos,
            eot,
            end_of_text,
            vocabulary: tokens.len(),
            token_bytes,
            special,
        })
    }
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        if text.len() > 1 << 20 {
            return Err(Error::request("input exceeds one MiB"));
        }
        self.inner
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| unsupported(e.to_string()))
    }
    pub fn token_bytes(&self, id: u32) -> Result<&[u8]> {
        self.token_bytes
            .get(id as usize)
            .map(Vec::as_slice)
            .ok_or_else(|| Error::request("token ID outside vocabulary"))
    }
    pub fn is_special(&self, id: u32) -> bool {
        self.special.get(id as usize).copied().unwrap_or(true)
    }
    pub fn vocabulary(&self) -> usize {
        self.vocabulary
    }
    pub fn supports_conversations(&self) -> bool {
        // The passthrough numerical fixture deliberately ignores later messages.
        self.template != "{{ messages[0].content }}"
    }
    pub fn cache_identity(&self) -> &str {
        &self.metadata_identity
    }
    pub fn is_tool_marker(&self, id: u32) -> bool {
        self.supports_tools()
            && self
                .token_bytes
                .get(id as usize)
                .is_some_and(|b| b == b"<tool_call>" || b == b"</tool_call>")
    }
    pub fn is_eog(&self, id: u32) -> bool {
        id == self.eos || Some(id) == self.eot || Some(id) == self.end_of_text
    }
    pub fn supports_tools(&self) -> bool {
        self.supports_conversations()
    }
    pub fn render(&self, messages: &[Message]) -> Result<String> {
        self.render_tools(messages, &crate::tools::ToolConfig::default())
    }
    pub fn render_tools(
        &self,
        messages: &[Message],
        config: &crate::tools::ToolConfig,
    ) -> Result<String> {
        config.validate_messages(messages, false)?;
        if !self.supports_tools()
            && (config.enabled()
                || messages
                    .iter()
                    .any(|m| m.role == "tool" || !m.tool_calls.is_empty()))
        {
            return Err(Error::new(
                ErrorCode::UnsupportedChatTemplate,
                "this template is not qualified for tools",
            ));
        }
        let mut env = minijinja::Environment::new();
        env.set_fuel(Some(100_000));
        env.set_recursion_limit(32);
        env.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
        let source = if config.enabled()
            || messages
                .iter()
                .any(|m| m.role == "tool" || !m.tool_calls.is_empty())
        {
            include_str!("../../../fixtures/tools/qwen2.5-tools-v1.jinja")
        } else {
            &self.template
        };
        env.add_template("chat", source)
            .map_err(|e| Error::new(ErrorCode::UnsupportedChatTemplate, e.to_string()))?;
        env.add_filter(
            "tojson",
            |v: minijinja::Value| -> std::result::Result<String, minijinja::Error> {
                serde_json::to_value(v)
                    .map(crate::tools::template_json)
                    .map_err(|e| {
                        minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string())
                    })
            },
        );
        let mut output = LimitedWriter(Vec::new(), false);
        let template_messages: Vec<_> = messages.iter().map(|m| {
            let mut v=serde_json::to_value(m).expect("serializable chat message");
            v["tool_calls"]=serde_json::json!([]);
            for c in &m.tool_calls {
                v["tool_calls"].as_array_mut().unwrap().push(serde_json::json!({"id":c.id,"type":"function","function":{"name":c.function.name,"arguments":c.function.arguments}}));
            } v
        }).collect();
        let tools = config.effective_tools();
        let rendered = env.get_template("chat").unwrap().render_captured_to(minijinja::context!(messages=>minijinja::value::Serde(template_messages),tools=>minijinja::value::Serde(tools),add_generation_prompt=>true,bos_token=>"",eos_token=>"<|im_end|>"),&mut output);
        if output.1 {
            return Err(Error::resource("rendered chat exceeds one MiB"));
        }
        rendered.map_err(|e| Error::new(ErrorCode::UnsupportedChatTemplate, e.to_string()))?;
        String::from_utf8(output.0)
            .map_err(|_| Error::new(ErrorCode::Internal, "template returned invalid UTF-8"))
    }
}
struct LimitedWriter(Vec<u8>, bool);
impl Write for LimitedWriter {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        if self.0.len().saturating_add(b.len()) > 1 << 20 {
            self.1 = true;
            return Err(std::io::Error::other("template output limit exceeded"));
        }
        self.0.extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Keeps incomplete byte sequences between tokens, including Bengali/emoji splits.
#[derive(Default)]
pub struct Utf8Stream {
    pending: Vec<u8>,
}
impl Utf8Stream {
    pub fn push(&mut self, bytes: &[u8], finish: bool) -> String {
        self.pending.extend_from_slice(bytes);
        let mut output = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    output.push_str(s);
                    self.pending.clear();
                    break;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    output.push_str(std::str::from_utf8(&self.pending[..valid]).unwrap());
                    self.pending.drain(..valid);
                    match e.error_len() {
                        Some(n) => {
                            output.push('\u{fffd}');
                            self.pending.drain(..n);
                        }
                        None => {
                            if finish {
                                output.push_str(&String::from_utf8_lossy(&self.pending));
                                self.pending.clear();
                            }
                            break;
                        }
                    }
                }
            }
        }
        output
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn utf8_across_token_boundaries() {
        let bytes = "বাংলা🙂".as_bytes();
        let mut stream = Utf8Stream::default();
        let mut text = String::new();
        for b in bytes {
            text += &stream.push(&[*b], false);
        }
        text += &stream.push(&[], true);
        assert_eq!(text, "বাংলা🙂");
    }
    #[test]
    fn invalid_utf8_is_bounded_and_replaced() {
        let mut s = Utf8Stream::default();
        assert_eq!(s.push(&[255], false), "�");
        assert_eq!(s.push(&[0xe2], true), "�");
    }
}

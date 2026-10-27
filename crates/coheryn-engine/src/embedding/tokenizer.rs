use crate::{Cancellation, Descriptor, Error, ErrorCode, Result, gguf::Value};
use std::collections::HashMap;
use tokenizers::{
    NormalizedString, Normalizer, OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer,
    normalizers::{
        bert::BertNormalizer,
        replace::{Replace, ReplacePattern},
    },
    pre_tokenizers::bert::BertPreTokenizer,
};
use unicode_normalization::char::decompose_canonical;

pub const MAX_TEXT_BYTES: usize = 64 << 10;

pub struct EmbeddingTokenizer {
    vocab: HashMap<String, u32>,
    marks: Replace,
    longest: usize,
    bos: u32,
    eos: u32,
    unknown: u32,
}
fn unsupported(message: impl Into<String>) -> Error {
    Error::new(ErrorCode::UnsupportedTokenizer, message)
}
impl EmbeddingTokenizer {
    pub fn check_metadata(d: &Descriptor) -> Result<()> {
        if d.text("tokenizer.ggml.model") != Some("bert") {
            return Err(unsupported(
                "embedding adapter requires GGUF bert WordPiece",
            ));
        }
        for (key, expected) in [
            ("tokenizer.ggml.normalizer.lowercase", true),
            ("tokenizer.ggml.normalizer.strip_accents", true),
            ("tokenizer.ggml.normalizer.nfc", false),
            ("tokenizer.ggml.add_bos_token", true),
            ("tokenizer.ggml.add_eos_token", true),
        ] {
            if let Some(value) = d.get(key)
                && !matches!(value, Value::Bool(v) if *v == expected)
            {
                return Err(unsupported(format!(
                    "unqualified normalizer/special-token setting {key}"
                )));
            }
        }
        const NORMALIZER_KEYS: &[&str] = &[
            "tokenizer.ggml.normalizer.lowercase",
            "tokenizer.ggml.normalizer.strip_accents",
            "tokenizer.ggml.normalizer.nfc",
        ];
        if let Some(key) = d.metadata.keys().find(|key| {
            key.starts_with("tokenizer.ggml.normalizer.")
                && !NORMALIZER_KEYS.contains(&key.as_str())
        }) {
            return Err(unsupported(format!(
                "unqualified WordPiece normalizer metadata {key}"
            )));
        }
        if d.get("tokenizer.ggml.pre").is_some() {
            return Err(unsupported(
                "embedding pretokenizer variants require qualification",
            ));
        }
        let tokens = d
            .get("tokenizer.ggml.tokens")
            .and_then(Value::array)
            .filter(|t| !t.is_empty() && t.len() <= 100_000)
            .ok_or_else(|| unsupported("missing bounded WordPiece vocabulary"))?;
        if tokens
            .iter()
            .any(|t| t.string().is_none_or(|s| s.is_empty() || s.len() > 512))
        {
            return Err(unsupported("invalid WordPiece token"));
        }
        for (key, name) in [
            ("tokenizer.ggml.bos_token_id", "[CLS]"),
            ("tokenizer.ggml.eos_token_id", "[SEP]"),
            ("tokenizer.ggml.unknown_token_id", "[UNK]"),
            ("tokenizer.ggml.padding_token_id", "[PAD]"),
        ] {
            let id = d
                .uint(key)
                .filter(|id| *id < tokens.len() as u64)
                .ok_or_else(|| unsupported(format!("missing {key}")))?;
            if tokens[id as usize].string() != Some(name) {
                return Err(unsupported(format!("unqualified {key}")));
            }
        }
        Ok(())
    }
    pub fn from_descriptor(d: &Descriptor) -> Result<Self> {
        Self::check_metadata(d)?;
        let tokens = d
            .get("tokenizer.ggml.tokens")
            .and_then(Value::array)
            .filter(|t| !t.is_empty() && t.len() <= 100_000)
            .ok_or_else(|| unsupported("missing bounded WordPiece vocabulary"))?;
        let mut vocab = HashMap::new();
        vocab
            .try_reserve(tokens.len())
            .map_err(|_| Error::resource("WordPiece dictionary allocation failed"))?;
        let mut longest = 0;
        for (i, token) in tokens.iter().enumerate() {
            let text = token
                .string()
                .filter(|s| !s.is_empty() && s.len() <= 512)
                .ok_or_else(|| unsupported("invalid WordPiece token"))?;
            longest = longest.max(text.len());
            if vocab.insert(text.to_owned(), i as u32).is_some() {
                return Err(unsupported("duplicate WordPiece token"));
            }
        }
        let special = |key: &str, expected: &str| -> Result<u32> {
            let id = d
                .uint(key)
                .filter(|id| *id < tokens.len() as u64)
                .ok_or_else(|| unsupported(format!("missing {key}")))? as u32;
            if tokens[id as usize].string() != Some(expected) {
                return Err(unsupported(format!("unqualified {key}")));
            }
            Ok(id)
        };
        let bos = special("tokenizer.ggml.bos_token_id", "[CLS]")?;
        let eos = special("tokenizer.ggml.eos_token_id", "[SEP]")?;
        let unknown = special("tokenizer.ggml.unknown_token_id", "[UNK]")?;
        special("tokenizer.ggml.padding_token_id", "[PAD]")?;
        if d.uint("tokenizer.ggml.seperator_token_id")
            .is_some_and(|id| id != u64::from(eos))
        {
            return Err(unsupported("separator and EOS identities differ"));
        }
        Ok(Self {
            vocab,
            marks: Replace::new(ReplacePattern::Regex(r"\p{M}".into()), "")
                .map_err(|e| unsupported(e.to_string()))?,
            longest,
            bos,
            eos,
            unknown,
        })
    }
    /// Literal text; special-token-looking strings are ordinary input. Limits include CLS/SEP.
    pub fn encode(
        &self,
        text: &str,
        limit: usize,
        truncate: bool,
        cancel: &Cancellation,
    ) -> Result<(Vec<u32>, bool)> {
        if text.is_empty() || text.len() > MAX_TEXT_BYTES || !(2..=512).contains(&limit) {
            return Err(Error::request(
                "embedding input requires 1..=65536 bytes and a 2..=512 token limit",
            ));
        }
        cancel.check()?;
        let mut normalized = NormalizedString::from(text);
        // The qualified GGUF WordPiece contract maps each source code point to
        // the FIRST canonical decomposition code point, then removes all marks.
        // Applying full-string BERT NFD instead changes Korean token IDs. Keep
        // this rule model-specific; it must never become global text cleaning.
        normalized.map(|c| {
            let mut first = None;
            decompose_canonical(c, |part| {
                first.get_or_insert(part);
            });
            first.unwrap_or(c)
        });
        BertNormalizer::new(true, true, Some(false), true)
            .normalize(&mut normalized)
            .map_err(|e| unsupported(e.to_string()))?;
        // GGUF WordPiece strips every Unicode mark (Mn/Mc/Me); the general
        // Bert normalizer strips only Mn. This matters for Bengali vowel signs.
        self.marks
            .normalize(&mut normalized)
            .map_err(|e| unsupported(e.to_string()))?;
        let mut split = PreTokenizedString::from(normalized);
        BertPreTokenizer
            .pre_tokenize(&mut split)
            .map_err(|e| unsupported(e.to_string()))?;
        let mut ids = Vec::new();
        ids.try_reserve_exact(limit)
            .map_err(|_| Error::resource("embedding token allocation failed"))?;
        ids.push(self.bos);
        let mut truncated = false;
        let mut pieces = Vec::new();
        pieces
            .try_reserve_exact(limit)
            .map_err(|_| Error::resource("WordPiece scratch allocation failed"))?;
        for (word, _, _) in split.get_splits(OffsetReferential::Normalized, OffsetType::Byte) {
            cancel.check()?;
            if word.is_empty() {
                continue;
            }
            let word = format!("▁{word}");
            pieces.clear();
            let mut count = 0usize;
            let mut start = 0;
            while start < word.len() {
                if count.is_multiple_of(64) {
                    cancel.check()?;
                }
                let mut end = (start + self.longest).min(word.len());
                while end > start && !word.is_char_boundary(end) {
                    end -= 1;
                }
                let mut found = None;
                while end > start {
                    if let Some(id) = self.vocab.get(&word[start..end]) {
                        found = Some((*id, end));
                        break;
                    }
                    end -= 1;
                    while end > start && !word.is_char_boundary(end) {
                        end -= 1;
                    }
                }
                match found {
                    Some((id, end)) => {
                        if pieces.len() < limit {
                            pieces.push(id);
                        }
                        count += 1;
                        start = end;
                    }
                    None => {
                        pieces.clear();
                        pieces.push(self.unknown);
                        count = 1;
                        break;
                    }
                }
            }
            let remaining = limit - 1 - ids.len();
            if count > remaining {
                if !truncate {
                    return Err(Error::new(
                        ErrorCode::EmbeddingInputLimitExceeded,
                        "embedding input exceeds the token limit; enable truncation explicitly",
                    ));
                }
                ids.extend_from_slice(&pieces[..remaining]);
                truncated = true;
                break;
            }
            ids.extend_from_slice(&pieces);
        }
        ids.push(self.eos);
        Ok((ids, truncated))
    }
}

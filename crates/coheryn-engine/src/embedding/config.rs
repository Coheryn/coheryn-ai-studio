use crate::{Descriptor, Error, ErrorCode, Result, gguf::Value, quant};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingConfig {
    pub layers: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub vocabulary: usize,
    pub token_types: usize,
    pub model_context: usize,
    pub qualified_context: usize,
    pub rope_base: f32,
    pub norm_epsilon: f32,
}
impl EmbeddingConfig {
    pub(crate) fn check_bounds(&self) -> Result<()> {
        if !(1..=48).contains(&self.layers)
            || !(1..=4096).contains(&self.hidden)
            || !(1..=32768).contains(&self.intermediate)
            || !(1..=128).contains(&self.heads)
            || !self.hidden.is_multiple_of(self.heads)
            || self.head_dim != self.hidden / self.heads
            || !self.head_dim.is_multiple_of(2)
            || !(1..=100_000).contains(&self.vocabulary)
            || !(1..=16).contains(&self.token_types)
            || !(2..=2048).contains(&self.model_context)
            || self.qualified_context != self.model_context.min(512)
            || !self.rope_base.is_finite()
            || self.rope_base <= 0.
            || !self.norm_epsilon.is_finite()
            || self.norm_epsilon <= 0.
        {
            return Err(Error::invalid(
                "embedding configuration exceeds qualified bounds",
            ));
        }
        Ok(())
    }
}
fn unsupported(message: impl Into<String>) -> Error {
    Error::new(ErrorCode::UnsupportedFeature, message)
}
fn number(d: &Descriptor, key: &str, maximum: usize) -> Result<usize> {
    d.uint(key)
        .filter(|n| *n > 0 && *n <= maximum as u64)
        .map(|n| n as usize)
        .ok_or_else(|| Error::invalid(format!("missing or out-of-bounds {key}")))
}
pub fn configuration(d: &Descriptor) -> Result<EmbeddingConfig> {
    if d.architecture.as_deref() != Some("nomic-bert") {
        return Err(Error::new(
            ErrorCode::UnsupportedModelArchitecture,
            "embedding execution requires the qualified nomic-bert graph",
        ));
    }
    if d.byte_order != "little" {
        return Err(unsupported(
            "embedding execution requires little-endian GGUF",
        ));
    }
    if d.uint("split.count").is_some_and(|n| n > 1) {
        return Err(unsupported("sharded embedding models are unsupported"));
    }
    const KEYS: &[&str] = &[
        "nomic-bert.attention.causal",
        "nomic-bert.attention.head_count",
        "nomic-bert.attention.layer_norm_epsilon",
        "nomic-bert.block_count",
        "nomic-bert.context_length",
        "nomic-bert.embedding_length",
        "nomic-bert.feed_forward_length",
        "nomic-bert.pooling_type",
        "nomic-bert.rope.freq_base",
    ];
    if let Some(key) = d
        .metadata
        .keys()
        .find(|k| k.starts_with("nomic-bert.") && !KEYS.contains(&k.as_str()))
    {
        return Err(unsupported(format!(
            "unqualified embedding architecture metadata {key}"
        )));
    }
    if !matches!(
        d.get("nomic-bert.attention.causal"),
        Some(Value::Bool(false))
    ) || d.uint("nomic-bert.pooling_type") != Some(1)
    {
        return Err(unsupported(
            "only bidirectional attention with mean pooling is qualified",
        ));
    }
    let hidden = number(d, "nomic-bert.embedding_length", 4096)?;
    let heads = number(d, "nomic-bert.attention.head_count", 128)?;
    if !hidden.is_multiple_of(heads) || !(hidden / heads).is_multiple_of(2) {
        return Err(Error::invalid("invalid embedding head/RoPE dimensions"));
    }
    let vocabulary = d
        .get("tokenizer.ggml.tokens")
        .and_then(Value::array)
        .map(|a| a.len())
        .filter(|n| *n > 0 && *n <= 100_000)
        .ok_or_else(|| Error::invalid("invalid embedding vocabulary"))?;
    let float = |key: &str| {
        d.get(key)
            .and_then(Value::float)
            .map(|v| v as f32)
            .filter(|v| v.is_finite() && *v > 0.)
            .ok_or_else(|| Error::invalid(format!("invalid {key}")))
    };
    let model_context = number(d, "nomic-bert.context_length", 2048)?;
    let c = EmbeddingConfig {
        layers: number(d, "nomic-bert.block_count", 48)?,
        hidden,
        intermediate: number(d, "nomic-bert.feed_forward_length", 32768)?,
        heads,
        head_dim: hidden / heads,
        vocabulary,
        token_types: number(d, "tokenizer.ggml.token_type_count", 16)?,
        model_context,
        qualified_context: model_context.min(512),
        rope_base: float("nomic-bert.rope.freq_base")?,
        norm_epsilon: float("nomic-bert.attention.layer_norm_epsilon")?,
    };
    c.check_bounds()?;
    let expect = |name: &str, dims: &[usize]| -> Result<()> {
        let t = d
            .tensor(name)
            .ok_or_else(|| Error::invalid(format!("missing embedding tensor {name}")))?;
        if t.dimensions
            .iter()
            .copied()
            .ne(dims.iter().map(|n| *n as u64))
        {
            return Err(Error::invalid(format!(
                "incorrect embedding tensor shape {name}"
            )));
        }
        if (dims.len() == 1 && t.dtype != 0) || !matches!(t.dtype, 0 | 1 | 8) {
            return Err(unsupported(format!(
                "{name}: embedding execution supports F32/F16/Q8_0 matrices and F32 vectors"
            )));
        }
        if quant::row_bytes(t.dtype, dims[0])? > 256 << 10 {
            return Err(unsupported("embedding row exceeds the bounded I/O tile"));
        }
        Ok(())
    };
    expect("token_embd.weight", &[hidden, vocabulary])?;
    expect("token_types.weight", &[hidden, c.token_types])?;
    for suffix in ["weight", "bias"] {
        expect(&format!("token_embd_norm.{suffix}"), &[hidden])?;
    }
    for i in 0..c.layers {
        for (name, shape) in [
            ("attn_qkv.weight", vec![hidden, 3 * hidden]),
            ("attn_output.weight", vec![hidden, hidden]),
            ("ffn_up.weight", vec![hidden, c.intermediate]),
            ("ffn_gate.weight", vec![hidden, c.intermediate]),
            ("ffn_down.weight", vec![c.intermediate, hidden]),
        ] {
            expect(&format!("blk.{i}.{name}"), &shape)?;
        }
        for norm in ["attn_output_norm", "layer_output_norm"] {
            for suffix in ["weight", "bias"] {
                expect(&format!("blk.{i}.{norm}.{suffix}"), &[hidden])?;
            }
        }
    }
    if d.tensors.len() != 4 + 9 * c.layers {
        return Err(unsupported(
            "extra embedding tensors require a separately qualified graph adapter",
        ));
    }
    Ok(c)
}

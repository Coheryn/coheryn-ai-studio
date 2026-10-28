use crate::{Descriptor, Error, ErrorCode, Result};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelTask {
    TextGeneration,
    Embedding,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelConfig {
    pub layers: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocabulary: usize,
    pub context: usize,
    pub rope_base: f32,
    pub norm_epsilon: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompatibilityReport {
    pub task: Option<ModelTask>,
    pub embedding_config: Option<crate::embedding::EmbeddingConfig>,
    pub container_valid: bool,
    pub executable: bool,
    pub qualification: &'static str,
    pub backend: &'static str,
    pub diagnostics: Vec<Error>,
    pub config: Option<ModelConfig>,
    pub tool_protocol: Option<crate::tools::ToolCapabilities>,
}

fn positive(d: &Descriptor, key: &str, max: usize) -> Result<usize> {
    let n = d
        .uint(key)
        .ok_or_else(|| Error::invalid(format!("missing unsigned metadata {key}")))?;
    if n == 0 || n > max as u64 {
        return Err(Error::invalid(format!(
            "{key} outside supported dimension bounds"
        )));
    }
    Ok(n as usize)
}

pub fn configuration(d: &Descriptor) -> Result<ModelConfig> {
    configuration_for(d, "qwen2")
}

pub(crate) fn configuration_for(d: &Descriptor, architecture: &str) -> Result<ModelConfig> {
    if d.architecture.as_deref() != Some(architecture) {
        return Err(Error::new(
            ErrorCode::UnsupportedModelArchitecture,
            "only the Qwen2 decoder graph is implemented",
        ));
    }
    if !matches!(architecture, "qwen2" | "qwen2vl") {
        return Err(Error::invalid("unknown decoder configuration profile"));
    }
    let prefix = format!("{architecture}.");
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    const IMPLEMENTED_KEYS: &[&str] = &[
        "block_count",
        "context_length",
        "embedding_length",
        "feed_forward_length",
        "attention.head_count",
        "attention.head_count_kv",
        "attention.layer_norm_rms_epsilon",
        "rope.freq_base",
        "rope.dimension_count",
        "attention.key_length",
        "attention.value_length",
    ];
    if let Some(key) = d.metadata.keys().find(|k| {
        k.strip_prefix(&prefix).is_some_and(|suffix| {
            !IMPLEMENTED_KEYS.contains(&suffix)
                && !(architecture == "qwen2vl" && suffix == "rope.dimension_sections")
        })
    }) {
        return Err(Error::new(
            ErrorCode::UnsupportedFeature,
            format!("unimplemented architecture metadata {key}"),
        ));
    }
    let hidden = positive(d, &key("embedding_length"), 32768)?;
    let heads = positive(d, &key("attention.head_count"), 512)?;
    let kv_heads = positive(d, &key("attention.head_count_kv"), 512)?;
    if hidden % heads != 0 || heads % kv_heads != 0 || (hidden / heads) % 2 != 0 {
        return Err(Error::invalid("invalid Qwen2 GQA or RoPE dimensions"));
    }
    let vocabulary = d
        .get("tokenizer.ggml.tokens")
        .and_then(|v| v.array())
        .map(|a| a.len())
        .filter(|n| *n > 0 && *n <= 1_000_000)
        .ok_or_else(|| Error::invalid("missing or invalid tokenizer vocabulary"))?;
    let rope_base = d
        .get(&key("rope.freq_base"))
        .and_then(|v| v.float())
        .unwrap_or(10000.0) as f32;
    let norm_epsilon = d
        .get(&key("attention.layer_norm_rms_epsilon"))
        .and_then(|v| v.float())
        .ok_or_else(|| Error::invalid("missing RMSNorm epsilon"))? as f32;
    if !rope_base.is_finite()
        || rope_base <= 0.0
        || !norm_epsilon.is_finite()
        || norm_epsilon <= 0.0
    {
        return Err(Error::invalid("invalid RoPE base or norm epsilon"));
    }
    let head_dim = hidden / heads;
    for key in [
        "attention.key_length",
        "attention.value_length",
        "rope.dimension_count",
    ] {
        if d.uint(&format!("{architecture}.{key}"))
            .is_some_and(|n| n != head_dim as u64)
        {
            return Err(Error::new(
                ErrorCode::UnsupportedFeature,
                format!("nonstandard {key}"),
            ));
        }
    }
    if d.uint("split.count").is_some_and(|n| n > 1) {
        return Err(Error::new(
            ErrorCode::UnsupportedFeature,
            "multi-file GGUF shards are not implemented",
        ));
    }
    if d.metadata.keys().any(|key| {
        key.starts_with(&format!("{architecture}.rope.scaling."))
            || key == &format!("{architecture}.attention.sliding_window")
    }) {
        return Err(Error::new(
            ErrorCode::UnsupportedFeature,
            "RoPE scaling and sliding-window variants are not implemented",
        ));
    }
    Ok(ModelConfig {
        layers: positive(d, &key("block_count"), 256)?,
        hidden,
        intermediate: positive(d, &key("feed_forward_length"), 131072)?,
        heads,
        kv_heads,
        head_dim,
        vocabulary,
        context: positive(d, &key("context_length"), 1_048_576)?,
        rope_base,
        norm_epsilon,
    })
}

pub(crate) fn graph(d: &Descriptor, c: &ModelConfig) -> Result<()> {
    let expect = |name: &str, dims: &[usize]| -> Result<()> {
        let t = d
            .tensor(name)
            .ok_or_else(|| Error::invalid(format!("missing required tensor {name}")))?;
        if t.dimensions
            .iter()
            .copied()
            .ne(dims.iter().map(|v| *v as u64))
        {
            return Err(Error::invalid(format!(
                "incorrect shape for {name}: {:?}, expected {dims:?}",
                t.dimensions
            )));
        }
        Ok(())
    };
    expect("token_embd.weight", &[c.hidden, c.vocabulary])?;
    expect("output_norm.weight", &[c.hidden])?;
    if d.tensor("output.weight").is_some() {
        expect("output.weight", &[c.hidden, c.vocabulary])?;
    }
    for i in 0..c.layers {
        for suffix in ["attn_norm.weight", "ffn_norm.weight"] {
            expect(&format!("blk.{i}.{suffix}"), &[c.hidden])?;
        }
        for (suffix, rows) in [
            ("attn_q", c.hidden),
            ("attn_k", c.kv_heads * c.head_dim),
            ("attn_v", c.kv_heads * c.head_dim),
        ] {
            expect(&format!("blk.{i}.{suffix}.weight"), &[c.hidden, rows])?;
            expect(&format!("blk.{i}.{suffix}.bias"), &[rows])?;
        }
        expect(
            &format!("blk.{i}.attn_output.weight"),
            &[c.hidden, c.hidden],
        )?;
        expect(
            &format!("blk.{i}.ffn_gate.weight"),
            &[c.hidden, c.intermediate],
        )?;
        expect(
            &format!("blk.{i}.ffn_up.weight"),
            &[c.hidden, c.intermediate],
        )?;
        expect(
            &format!("blk.{i}.ffn_down.weight"),
            &[c.intermediate, c.hidden],
        )?;
    }
    // Do not silently ignore graph-changing tensors (e.g. additional biases).
    let expected = 2 + usize::from(d.tensor("output.weight").is_some()) + c.layers * 12;
    if d.tensors.len() != expected {
        return Err(Error::new(
            ErrorCode::UnsupportedFeature,
            "unexpected tensors in Qwen2 graph",
        ));
    }
    Ok(())
}

pub fn validate(d: &Descriptor) -> CompatibilityReport {
    if d.architecture.as_deref() == Some("nomic-bert") {
        return crate::embedding::compatibility(d);
    }
    let mut diagnostics = Vec::new();
    if d.byte_order != "little" {
        diagnostics.push(Error::new(
            ErrorCode::UnsupportedFeature,
            "big-endian GGUF metadata parses, but numeric execution is not qualified",
        ));
    }
    for t in &d.tensors {
        if !crate::quant::supported(t.dtype) {
            diagnostics.push(Error::new(
                ErrorCode::UnsupportedGgufTensorType,
                format!(
                    "{} uses dtype {}; native CPU supports F32/F16/Q8_0/Q4_0/Q5_0/Q4_K/Q6_K",
                    t.name, t.dtype
                ),
            ));
            break;
        }
        if crate::quant::row_bytes(t.dtype, t.dimensions[0] as usize)
            .map_or(true, |n| n > 256 << 10)
        {
            diagnostics.push(Error::new(
                ErrorCode::UnsupportedFeature,
                format!("{} row exceeds bounded native CPU kernel workspace", t.name),
            ));
            break;
        }
    }
    let config = match configuration(d) {
        Ok(c) => {
            if let Err(e) = graph(d, &c) {
                diagnostics.push(e);
            }
            Some(c)
        }
        Err(e) => {
            diagnostics.push(e);
            None
        }
    };
    if let Err(e) = crate::tokenizer::check_metadata(d) {
        diagnostics.push(e);
    }
    CompatibilityReport {
        tool_protocol: if diagnostics.is_empty() {
            crate::tools::capabilities(d)
        } else {
            None
        },
        task: (d.architecture.as_deref() == Some("qwen2")).then_some(ModelTask::TextGeneration),
        embedding_config: None,
        container_valid: true,
        executable: diagnostics.is_empty(),
        qualification: "experimental-unqualified",
        backend: "cpu-reference",
        diagnostics,
        config,
    }
}

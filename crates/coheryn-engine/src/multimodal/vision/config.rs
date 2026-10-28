use crate::{
    Descriptor, Error, ErrorCode, Result,
    gguf::{Tensor, Value},
    memory,
};
use serde::Serialize;
use std::collections::BTreeSet;

#[derive(Debug, Clone, Serialize)]
pub struct VisionConfig {
    pub layers: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub patch_size: usize,
    pub projector_hidden: usize,
    pub output_width: usize,
    pub norm_epsilon: f32,
    pub declared_intermediate: Option<u64>,
}
fn unsupported(message: impl Into<String>) -> Error {
    Error::new(ErrorCode::UnsupportedFeature, message)
}
fn number(d: &Descriptor, key: &str, max: usize) -> Result<usize> {
    d.get(key)
        .and_then(Value::integer)
        .filter(|n| *n > 0 && *n <= max as u64)
        .map(|n| n as usize)
        .ok_or_else(|| Error::invalid(format!("invalid vision metadata {key}")))
}
impl VisionConfig {
    /// Shape compatibility only; exact-file/backend qualification remains separate.
    pub fn from_descriptor(d: &Descriptor) -> Result<Self> {
        if d.tensors.len() > 1032 || d.metadata.len() > 512 {
            return Err(Error::resource(
                "vision component descriptor exceeds supported bounds",
            ));
        }
        if d.architecture.as_deref() != Some("clip")
            || d.text("clip.projector_type") != Some("qwen2vl_merger")
        {
            return Err(unsupported(
                "vision execution requires the supported Qwen2-VL merger graph",
            ));
        }
        if d.byte_order != "little"
            || !matches!(d.get("clip.has_vision_encoder"), Some(Value::Bool(true)))
            || d.metadata.keys().any(|key| key.starts_with("split."))
        {
            return Err(unsupported(
                "vision execution requires a single little-endian vision component",
            ));
        }
        const KEYS: &[&str] = &[
            "clip.has_vision_encoder",
            "clip.has_text_encoder",
            "clip.has_audio_encoder",
            "clip.projector_type",
            "clip.use_gelu",
            "clip.use_silu",
            "clip.vision.attention.head_count",
            "clip.vision.attention.layer_norm_epsilon",
            "clip.vision.block_count",
            "clip.vision.embedding_length",
            "clip.vision.feed_forward_length",
            "clip.vision.image_mean",
            "clip.vision.image_std",
            "clip.vision.image_size",
            "clip.vision.patch_size",
            "clip.vision.projection_dim",
        ];
        for (key, value) in &d.metadata {
            if key.starts_with("clip.") && !KEYS.contains(&key.as_str()) {
                return Err(unsupported(format!("unqualified vision metadata {key}")));
            }
            if matches!(
                key.as_str(),
                "clip.has_text_encoder"
                    | "clip.has_audio_encoder"
                    | "clip.use_gelu"
                    | "clip.use_silu"
            ) && !matches!(value, Value::Bool(false))
            {
                return Err(unsupported(format!("unqualified vision feature {key}")));
            }
        }
        for (key, expected) in [
            (
                "clip.vision.image_mean",
                [0.48145466f32, 0.4578275, 0.40821073],
            ),
            (
                "clip.vision.image_std",
                [0.26862954f32, 0.261_302_6, 0.275_777_1],
            ),
        ] {
            let values = d
                .get(key)
                .and_then(Value::array)
                .ok_or_else(|| Error::invalid("missing vision normalization"))?;
            if values.len() != 3
                || values
                    .iter()
                    .zip(expected)
                    .any(|(v, e)| v.float().map(|n| n as f32) != Some(e))
            {
                return Err(unsupported(
                    "vision normalization does not match the supported processor",
                ));
            }
        }
        let hidden = number(d, "clip.vision.embedding_length", 4096)?;
        let heads = number(d, "clip.vision.attention.head_count", 128)?;
        if !hidden.is_multiple_of(heads) || !(hidden / heads).is_multiple_of(4) {
            return Err(Error::invalid(
                "vision head dimensions must support spatial rotation",
            ));
        }
        if number(d, "clip.vision.patch_size", 64)? != 14 {
            return Err(unsupported("only patch size 14 is supported"));
        }
        number(d, "clip.vision.image_size", 16384)?;
        let up = d
            .tensor("v.blk.0.ffn_down.weight")
            .ok_or_else(|| Error::invalid("missing vision FFN tensor"))?;
        if up.dimensions.len() != 2
            || up.dimensions[0] != hidden as u64
            || !(1..=32768).contains(&up.dimensions[1])
        {
            return Err(Error::invalid("invalid actual vision FFN dimensions"));
        }
        let epsilon = d
            .get("clip.vision.attention.layer_norm_epsilon")
            .and_then(Value::float)
            .map(|v| v as f32)
            .filter(|v| v.is_finite() && *v > 0. && *v <= 1.)
            .ok_or_else(|| Error::invalid("invalid vision normalization epsilon"))?;
        let c = Self {
            layers: number(d, "clip.vision.block_count", 64)?,
            hidden,
            intermediate: up.dimensions[1] as usize,
            heads,
            head_dim: hidden / heads,
            patch_size: 14,
            projector_hidden: memory::product(&[hidden, 4])?,
            output_width: number(d, "clip.vision.projection_dim", 16384)?,
            norm_epsilon: epsilon,
            declared_intermediate: d
                .get("clip.vision.feed_forward_length")
                .and_then(Value::integer),
        };
        let mut names = BTreeSet::new();
        let mut tensor = |name: String, dims: Vec<u64>, dtype: u32| -> Result<()> {
            if !names.insert(name.clone()) {
                return Err(Error::invalid("duplicate expected vision tensor"));
            }
            let matches: Vec<&Tensor> = d.tensors.iter().filter(|t| t.name == name).collect();
            if matches.len() != 1 {
                return Err(Error::invalid(format!(
                    "missing or duplicate vision tensor {name}"
                )));
            }
            let t = matches[0];
            let bytes = memory::product(&dims.iter().map(|n| *n as usize).collect::<Vec<_>>())?
                .checked_mul(if dtype == 1 { 2 } else { 4 })
                .ok_or_else(|| Error::resource("vision tensor extent overflow"))?;
            if t.dimensions != dims || t.dtype != dtype || t.bytes != bytes as u64 {
                return Err(unsupported(format!(
                    "unsupported vision shape/encoding for {name}"
                )));
            }
            Ok(())
        };
        for name in ["v.patch_embd.weight", "v.patch_embd.weight.1"] {
            tensor(name.into(), vec![14, 14, 3, hidden as u64], 1)?;
        }
        for layer in 0..c.layers {
            for name in ["attn_q", "attn_k", "attn_v", "attn_out"] {
                tensor(
                    format!("v.blk.{layer}.{name}.weight"),
                    vec![hidden as u64, hidden as u64],
                    1,
                )?;
                tensor(format!("v.blk.{layer}.{name}.bias"), vec![hidden as u64], 0)?;
            }
            for name in ["ln1", "ln2"] {
                for suffix in ["weight", "bias"] {
                    tensor(
                        format!("v.blk.{layer}.{name}.{suffix}"),
                        vec![hidden as u64],
                        0,
                    )?;
                }
            }
            for (name, input, output) in [
                ("ffn_down", hidden, c.intermediate),
                ("ffn_up", c.intermediate, hidden),
            ] {
                tensor(
                    format!("v.blk.{layer}.{name}.weight"),
                    vec![input as u64, output as u64],
                    1,
                )?;
                tensor(format!("v.blk.{layer}.{name}.bias"), vec![output as u64], 0)?;
            }
        }
        for suffix in ["weight", "bias"] {
            tensor(format!("v.post_ln.{suffix}"), vec![hidden as u64], 0)?;
        }
        for (name, input, output) in [
            ("mm.0", c.projector_hidden, c.projector_hidden),
            ("mm.2", c.projector_hidden, c.output_width),
        ] {
            tensor(
                format!("{name}.weight"),
                vec![input as u64, output as u64],
                1,
            )?;
            tensor(format!("{name}.bias"), vec![output as u64], 0)?;
        }
        if names.len() != d.tensors.len() {
            return Err(unsupported("unqualified extra tensors in vision component"));
        }
        Ok(c)
    }
}

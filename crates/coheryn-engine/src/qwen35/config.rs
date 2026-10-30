use crate::{Error, ErrorCode, Result, safetensors::TensorSet};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeSet;

#[derive(Clone, Debug, Serialize)]
pub struct Config {
    pub hidden: usize,
    pub intermediate: usize,
    pub vocabulary: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub linear_key_heads: usize,
    pub linear_value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_kernel: usize,
    pub epsilon: f32,
    pub rope_base: f32,
    pub rotary_dim: usize,
    pub context: usize,
    pub eos: u32,
    pub tied: bool,
    pub layers: Vec<bool>,
}
fn unsupported(m: impl Into<String>) -> Error {
    Error::new(ErrorCode::UnsupportedModelArchitecture, m)
}
fn integer(v: &Value, name: &str, maximum: usize) -> Result<usize> {
    v.get(name)
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .filter(|n| *n > 0 && *n <= maximum)
        .ok_or_else(|| unsupported(format!("invalid Qwen3.5 {name}")))
}
impl Config {
    pub fn parse(root: &Value) -> Result<Self> {
        if root["model_type"] != "qwen3_5" {
            return Err(unsupported("unsupported source architecture"));
        }
        if root.get("quantization_config").is_some() {
            return Err(unsupported(
                "Qwen3.5 packed source schemes are not implemented",
            ));
        }
        let v = &root["text_config"];
        let allowed = [
            "model_type",
            "attention_bias",
            "attention_dropout",
            "attn_output_gate",
            "dtype",
            "eos_token_id",
            "bos_token_id",
            "pad_token_id",
            "full_attention_interval",
            "head_dim",
            "hidden_act",
            "hidden_size",
            "initializer_range",
            "intermediate_size",
            "layer_types",
            "linear_conv_kernel_dim",
            "linear_key_head_dim",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_value_head_dim",
            "max_position_embeddings",
            "mlp_only_layers",
            "mtp_num_hidden_layers",
            "mtp_use_dedicated_embeddings",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "rms_norm_eps",
            "tie_word_embeddings",
            "use_cache",
            "vocab_size",
            "mamba_ssm_dtype",
            "rope_parameters",
        ];
        let object = v
            .as_object()
            .ok_or_else(|| unsupported("missing Qwen3.5 text configuration"))?;
        if let Some(key) = object.keys().find(|k| !allowed.contains(&k.as_str())) {
            return Err(unsupported(format!(
                "unimplemented Qwen3.5 configuration {key}"
            )));
        }
        if v["hidden_act"] != "silu"
            || v["attention_bias"] != false
            || v["attn_output_gate"] != true
            || v["attention_dropout"].as_f64() != Some(0.)
            || v["model_type"] != "qwen3_5_text"
            || v["mamba_ssm_dtype"] != "float32"
            || !matches!(v["dtype"].as_str(), Some("bfloat16" | "float32"))
            || v.get("mlp_only_layers")
                .is_some_and(|x| x.as_array().is_none_or(|a| !a.is_empty()))
        {
            return Err(unsupported(
                "unimplemented Qwen3.5 activation/bias/attention configuration",
            ));
        }
        let rope = &v["rope_parameters"];
        if rope["rope_type"] != "default"
            || rope.as_object().is_none_or(|m| {
                m.keys().any(|k| {
                    ![
                        "rope_type",
                        "rope_theta",
                        "partial_rotary_factor",
                        "mrope_interleaved",
                        "mrope_section",
                    ]
                    .contains(&k.as_str())
                })
            })
        {
            return Err(unsupported("unimplemented Qwen3.5 rotary profile"));
        }
        let head_dim = integer(v, "head_dim", 512)?;
        let heads = integer(v, "num_attention_heads", 256)?;
        let kv_heads = integer(v, "num_key_value_heads", 256)?;
        let key_heads = integer(v, "linear_num_key_heads", 256)?;
        let value_heads = integer(v, "linear_num_value_heads", 256)?;
        let fraction = rope["partial_rotary_factor"]
            .as_f64()
            .filter(|x| x.is_finite() && *x > 0. && *x <= 1.)
            .ok_or_else(|| unsupported("invalid partial rotary fraction"))?;
        let dim = head_dim as f64 * fraction;
        if !heads.is_multiple_of(kv_heads)
            || !value_heads.is_multiple_of(key_heads)
            || dim.fract() != 0.
            || dim < 2.
            || !(dim as usize).is_multiple_of(2)
        {
            return Err(unsupported("incompatible Qwen3.5 head/rotary dimensions"));
        }
        let epsilon = v["rms_norm_eps"]
            .as_f64()
            .filter(|x| x.is_finite() && *x > 0. && *x < 1.)
            .ok_or_else(|| unsupported("invalid Qwen3.5 epsilon"))? as f32;
        let base = rope["rope_theta"]
            .as_f64()
            .filter(|x| x.is_finite() && *x > 1. && *x <= 1e12)
            .ok_or_else(|| unsupported("invalid Qwen3.5 rotary base"))? as f32;
        let layer_count = integer(v, "num_hidden_layers", 128)?;
        let layers = v["layer_types"]
            .as_array()
            .filter(|a| a.len() == layer_count)
            .ok_or_else(|| unsupported("layer-type count mismatch"))?
            .iter()
            .map(|x| match x.as_str() {
                Some("full_attention") => Ok(true),
                Some("linear_attention") => Ok(false),
                _ => Err(unsupported("unknown hybrid layer type")),
            })
            .collect::<Result<Vec<_>>>()?;
        let tied = v["tie_word_embeddings"]
            .as_bool()
            .ok_or_else(|| unsupported("missing tied-embedding policy"))?;
        let vocabulary = integer(v, "vocab_size", 1_000_000)?;
        let eos = integer(v, "eos_token_id", 1_000_000)?;
        if eos >= vocabulary
            || root
                .get("tie_word_embeddings")
                .is_some_and(|flag| flag.as_bool() != Some(tied))
        {
            return Err(unsupported(
                "inconsistent Qwen3.5 vocabulary/output configuration",
            ));
        }
        Ok(Self {
            hidden: integer(v, "hidden_size", 32768)?,
            intermediate: integer(v, "intermediate_size", 131072)?,
            vocabulary,
            heads,
            kv_heads,
            head_dim,
            linear_key_heads: key_heads,
            linear_value_heads: value_heads,
            key_dim: integer(v, "linear_key_head_dim", 512)?,
            value_dim: integer(v, "linear_value_head_dim", 512)?,
            conv_kernel: integer(v, "linear_conv_kernel_dim", 32)?,
            epsilon,
            rope_base: base,
            rotary_dim: dim as usize,
            context: integer(v, "max_position_embeddings", 4_194_304)?,
            eos: u32::try_from(eos).map_err(|_| unsupported("EOS overflow"))?,
            tied,
            layers,
        })
    }
    pub(crate) fn validate_tensors(&self, set: &TensorSet) -> Result<()> {
        if 2 * self.linear_key_heads * self.key_dim + self.linear_value_heads * self.value_dim
            > 65536
        {
            return Err(unsupported(
                "Qwen3.5 convolution channels exceed the native bound",
            ));
        }
        let mut required = BTreeSet::new();
        let mut require = |name: String, shape: &[usize]| -> Result<()> {
            let tensor = set
                .tensor(&name)
                .ok_or_else(|| unsupported(format!("missing Qwen3.5 tensor {name}")))?;
            let actual = tensor.dimensions.iter().rev().copied().collect::<Vec<_>>();
            if actual != shape.iter().map(|n| *n as u64).collect::<Vec<_>>()
                || !matches!(tensor.dtype, 0 | 30)
            {
                return Err(unsupported(format!(
                    "unsupported Qwen3.5 tensor shape/precision {name}"
                )));
            }
            let width = if tensor.dtype == 0 { 4u64 } else { 2 };
            if (shape.len() == 2 && shape[1] as u64 * width > 256 << 10)
                || (shape.len() != 2 && tensor.bytes > 256 << 10)
            {
                return Err(unsupported(format!(
                    "Qwen3.5 tensor exceeds bounded native read/workspace {name}"
                )));
            }
            required.insert(name);
            Ok(())
        };
        require(
            "model.language_model.embed_tokens.weight".into(),
            &[self.vocabulary, self.hidden],
        )?;
        require("model.language_model.norm.weight".into(), &[self.hidden])?;
        if !self.tied {
            require("lm_head.weight".into(), &[self.vocabulary, self.hidden])?;
        }
        for (i, full) in self.layers.iter().enumerate() {
            let prefix = format!("model.language_model.layers.{i}");
            for name in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
                require(format!("{prefix}.{name}"), &[self.hidden])?;
            }
            for name in ["gate_proj", "up_proj"] {
                require(
                    format!("{prefix}.mlp.{name}.weight"),
                    &[self.intermediate, self.hidden],
                )?;
            }
            require(
                format!("{prefix}.mlp.down_proj.weight"),
                &[self.hidden, self.intermediate],
            )?;
            if *full {
                for (name, width) in [
                    ("q_proj", 2 * self.heads * self.head_dim),
                    ("k_proj", self.kv_heads * self.head_dim),
                    ("v_proj", self.kv_heads * self.head_dim),
                ] {
                    require(
                        format!("{prefix}.self_attn.{name}.weight"),
                        &[width, self.hidden],
                    )?;
                }
                require(
                    format!("{prefix}.self_attn.o_proj.weight"),
                    &[self.hidden, self.heads * self.head_dim],
                )?;
                for name in ["q_norm", "k_norm"] {
                    require(
                        format!("{prefix}.self_attn.{name}.weight"),
                        &[self.head_dim],
                    )?;
                }
            } else {
                let keys = self.linear_key_heads * self.key_dim;
                let values = self.linear_value_heads * self.value_dim;
                let channels = 2 * keys + values;
                for (name, width) in [
                    ("in_proj_qkv", channels),
                    ("in_proj_z", values),
                    ("in_proj_a", self.linear_value_heads),
                    ("in_proj_b", self.linear_value_heads),
                ] {
                    require(
                        format!("{prefix}.linear_attn.{name}.weight"),
                        &[width, self.hidden],
                    )?;
                }
                require(
                    format!("{prefix}.linear_attn.out_proj.weight"),
                    &[self.hidden, values],
                )?;
                require(
                    format!("{prefix}.linear_attn.conv1d.weight"),
                    &[channels, 1, self.conv_kernel],
                )?;
                for name in ["A_log", "dt_bias"] {
                    require(
                        format!("{prefix}.linear_attn.{name}"),
                        &[self.linear_value_heads],
                    )?;
                }
                require(
                    format!("{prefix}.linear_attn.norm.weight"),
                    &[self.value_dim],
                )?;
            }
        }
        for name in set.names() {
            if !required.contains(name)
                && !name.starts_with("model.visual.")
                && !name.starts_with("mtp.")
            {
                return Err(unsupported(format!("unexpected Qwen3.5 tensor {name}")));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unsupported_graph_variants_and_inconsistent_dimensions_fail_before_binding() {
        let valid = serde_json::json!({"model_type":"qwen3_5","tie_word_embeddings":true,"text_config":{
            "model_type":"qwen3_5_text","hidden_act":"silu","attention_bias":false,"attn_output_gate":true,"attention_dropout":0.,
            "mamba_ssm_dtype":"float32","dtype":"float32","hidden_size":4,"intermediate_size":8,"vocab_size":16,"eos_token_id":3,
            "num_hidden_layers":2,"layer_types":["linear_attention","full_attention"],"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
            "linear_num_key_heads":2,"linear_num_value_heads":2,"linear_key_head_dim":2,"linear_value_head_dim":2,"linear_conv_kernel_dim":3,
            "rms_norm_eps":1e-6,"tie_word_embeddings":true,"max_position_embeddings":128,
            "rope_parameters":{"rope_type":"default","rope_theta":10000.,"partial_rotary_factor":0.5}
        }});
        let config = Config::parse(&valid).unwrap();
        assert_eq!(config.rotary_dim, 2);
        assert_eq!(config.layers, [false, true]);
        for (field, value) in [
            ("num_key_value_heads", serde_json::json!(3)),
            ("mamba_ssm_dtype", serde_json::json!("bfloat16")),
            ("eos_token_id", serde_json::json!(16)),
            ("hidden_act", serde_json::json!("gelu")),
            ("num_hidden_layers", serde_json::json!(3)),
            ("future_graph_setting", serde_json::json!(true)),
        ] {
            let mut source = valid.clone();
            source["text_config"][field] = value;
            assert!(Config::parse(&source).is_err(), "{field}");
        }
        let mut source = valid.clone();
        source["text_config"]["rope_parameters"]["rope_type"] = serde_json::json!("unknown");
        assert!(Config::parse(&source).is_err());
        let mut source = valid;
        source["tie_word_embeddings"] = serde_json::json!(false);
        assert!(Config::parse(&source).is_err());
    }
}

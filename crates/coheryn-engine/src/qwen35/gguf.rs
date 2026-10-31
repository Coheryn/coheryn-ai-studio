//! Native Qwen3.5 GGUF bindings into the shared hybrid/vision graph.
//! Qualified source sidecars retain tokenizer/template/processor semantics.
use super::Config;
use crate::{Cancellation, Descriptor, Error, ErrorCode, Result, safetensors::TensorSet};
use std::{collections::BTreeSet, path::Path};

fn bad(message: impl Into<String>) -> Error {
    Error::new(ErrorCode::UnsupportedModelArchitecture, message)
}
pub(crate) fn is_source(path: &Path) -> Result<bool> {
    use std::io::Read;
    if !path.is_file() {
        return Ok(false);
    }
    let mut file = std::fs::File::open(path)?;
    let mut magic = [0; 4];
    Ok(file.read(&mut magic)? == 4 && matches!(&magic, b"GGUF" | b"FUGG"))
}
pub(crate) fn configure(d: &Descriptor, json: &serde_json::Value) -> Result<Config> {
    let mut c = Config::parse(json)?;
    if d.architecture.as_deref() != Some("qwen35") || d.byte_order != "little" {
        return Err(bad("unsupported Qwen3.5 GGUF architecture/byte order"));
    }
    let keys = [
        "block_count",
        "context_length",
        "embedding_length",
        "feed_forward_length",
        "attention.head_count",
        "attention.head_count_kv",
        "rope.dimension_sections",
        "rope.freq_base",
        "attention.layer_norm_rms_epsilon",
        "attention.key_length",
        "attention.value_length",
        "ssm.conv_kernel",
        "ssm.state_size",
        "ssm.group_count",
        "ssm.time_step_rank",
        "ssm.inner_size",
        "full_attention_interval",
        "rope.dimension_count",
    ];
    if d.metadata.keys().any(|key| {
        key.strip_prefix("qwen35.")
            .is_some_and(|key| !keys.contains(&key))
            || key.starts_with("split.")
    }) || d.uint("general.quantization_version") != Some(2)
    {
        return Err(bad(
            "unsupported GGUF graph option, quantization contract or shard declaration",
        ));
    }
    for (key, expected) in [
        ("block_count", c.layers.len()),
        ("context_length", c.context),
        ("embedding_length", c.hidden),
        ("feed_forward_length", c.intermediate),
        ("attention.head_count", c.heads),
        ("attention.head_count_kv", c.kv_heads),
        ("attention.key_length", c.head_dim),
        ("attention.value_length", c.head_dim),
        ("ssm.conv_kernel", c.conv_kernel),
        ("ssm.state_size", c.key_dim),
        ("ssm.group_count", c.linear_key_heads),
        ("ssm.time_step_rank", c.linear_value_heads),
        ("ssm.inner_size", c.linear_value_heads * c.value_dim),
        ("rope.dimension_count", c.rotary_dim),
    ] {
        if d.uint(&format!("qwen35.{key}")) != Some(expected as u64) {
            return Err(bad(format!("GGUF/source configuration mismatch: {key}")));
        }
    }
    let interval = d
        .uint("qwen35.full_attention_interval")
        .filter(|v| *v > 0 && *v <= c.layers.len() as u64)
        .ok_or_else(|| bad("invalid full-attention interval"))? as usize;
    if c.layers
        .iter()
        .enumerate()
        .any(|(i, full)| *full != ((i + 1) % interval == 0))
        || d.get("qwen35.rope.freq_base")
            .and_then(crate::gguf::Value::float)
            .map(|v| v as f32)
            != Some(c.rope_base)
        || d.get("qwen35.attention.layer_norm_rms_epsilon")
            .and_then(crate::gguf::Value::float)
            .map(|v| v as f32)
            != Some(c.epsilon)
        || d.get("qwen35.rope.dimension_sections")
            .and_then(crate::gguf::Value::array)
            .map(|a| {
                a.iter()
                    .map(crate::gguf::Value::integer)
                    .collect::<Vec<_>>()
            })
            != Some(vec![Some(11), Some(11), Some(10), Some(0)])
    {
        return Err(bad("unsupported Qwen3.5 GGUF attention/rotary profile"));
    }
    if d.text("tokenizer.ggml.model") != Some("gpt2")
        || d.text("tokenizer.ggml.pre") != Some("qwen35")
        || d.uint("tokenizer.ggml.eos_token_id") != Some(248046)
        || d.uint("tokenizer.ggml.padding_token_id") != Some(c.eos as u64)
        || !matches!(
            d.get("tokenizer.ggml.add_bos_token"),
            Some(crate::gguf::Value::Bool(false))
        )
    {
        return Err(Error::new(
            ErrorCode::UnsupportedTokenizer,
            "unsupported Qwen3.5 GGUF tokenizer policy",
        ));
    }
    if d.metadata.keys().any(|key| {
        key.strip_prefix("tokenizer.ggml.").is_some_and(|key| {
            ![
                "model",
                "pre",
                "tokens",
                "merges",
                "token_type",
                "eos_token_id",
                "padding_token_id",
                "add_bos_token",
            ]
            .contains(&key)
        })
    }) {
        return Err(Error::new(
            ErrorCode::UnsupportedTokenizer,
            "unimplemented GGUF tokenizer option",
        ));
    }
    for (key, digest) in [
        (
            "tokens",
            "5ee0f927bcaa4b9fe85c244776ae9487468e427f83e053fc81f2a186f14936a3",
        ),
        (
            "merges",
            "7e299304d9ad9dc312acdbcb1f6ccf0dce1256bf1aa986d651f13814dfd27e7b",
        ),
        (
            "token_type",
            "f6fdca1063d1ae1cc77ba1f5087d259f044c2634e64b65e31bc844ec00e9acab",
        ),
    ] {
        if d.metadata_fingerprints
            .get(&format!("tokenizer.ggml.{key}"))
            .map(String::as_str)
            != Some(digest)
        {
            return Err(Error::new(
                ErrorCode::UnsupportedTokenizer,
                "unqualified GGUF tokenizer vocabulary/merge inventory",
            ));
        }
    }
    use sha2::Digest;
    let template = d
        .text("tokenizer.chat_template")
        .ok_or_else(|| bad("missing GGUF template"))?;
    if crate::storage::digest_hex(sha2::Sha256::digest(template.as_bytes()).into())
        != super::TEMPLATE_SHA256
    {
        return Err(Error::new(
            ErrorCode::UnsupportedChatTemplate,
            "unqualified GGUF template",
        ));
    }
    c.zero_centered_norms = false;
    c.ssm_pre_exponentiated = true;
    Ok(c)
}
fn language_name(name: &str) -> Option<String> {
    if name == "token_embd.weight" {
        return Some("model.language_model.embed_tokens.weight".into());
    }
    if name == "output_norm.weight" {
        return Some("model.language_model.norm.weight".into());
    }
    if name == "output.weight" {
        return Some("lm_head.weight".into());
    }
    let suffix = name.strip_prefix("blk.")?;
    let (layer, key) = suffix.split_once('.')?;
    layer.parse::<usize>().ok()?;
    let mapped = match key {
        "attn_norm.weight" => "input_layernorm.weight",
        "post_attention_norm.weight" => "post_attention_layernorm.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "attn_q_norm.weight" => "self_attn.q_norm.weight",
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        "attn_qkv.weight" => "linear_attn.in_proj_qkv.weight",
        "attn_gate.weight" => "linear_attn.in_proj_z.weight",
        "ssm_alpha.weight" => "linear_attn.in_proj_a.weight",
        "ssm_beta.weight" => "linear_attn.in_proj_b.weight",
        "ssm_conv1d.weight" => "linear_attn.conv1d.weight",
        "ssm_a" => "linear_attn.A_log",
        "ssm_dt.bias" => "linear_attn.dt_bias",
        "ssm_norm.weight" => "linear_attn.norm.weight",
        "ssm_out.weight" => "linear_attn.out_proj.weight",
        _ => return None,
    };
    Some(format!("model.language_model.layers.{layer}.{mapped}"))
}
fn vision_name(name: &str) -> Option<String> {
    let special = match name {
        "v.position_embd.weight" => Some("pos_embed.weight"),
        "v.patch_embd.bias" => Some("patch_embed.proj.bias"),
        "v.post_ln.weight" => Some("merger.norm.weight"),
        "v.post_ln.bias" => Some("merger.norm.bias"),
        "mm.0.weight" => Some("merger.linear_fc1.weight"),
        "mm.0.bias" => Some("merger.linear_fc1.bias"),
        "mm.2.weight" => Some("merger.linear_fc2.weight"),
        "mm.2.bias" => Some("merger.linear_fc2.bias"),
        _ => None,
    };
    if let Some(special) = special {
        return Some(format!("model.visual.{special}"));
    }
    let (layer, key) = name.strip_prefix("v.blk.")?.split_once('.')?;
    layer.parse::<usize>().ok()?;
    let (stem, suffix) = key.rsplit_once('.')?;
    if !matches!(suffix, "weight" | "bias") {
        return None;
    }
    let stem = match stem {
        "ln1" => "norm1",
        "ln2" => "norm2",
        "attn_qkv" => "attn.qkv",
        "attn_out" => "attn.proj",
        "ffn_up" => "mlp.linear_fc1",
        "ffn_down" => "mlp.linear_fc2",
        _ => return None,
    };
    Some(format!("model.visual.blocks.{layer}.{stem}.{suffix}"))
}
pub(crate) fn open(
    path: &Path,
    d: &Descriptor,
    json: &serde_json::Value,
    cancel: &Cancellation,
) -> Result<(Config, TensorSet)> {
    let config = configure(d, json)?;
    let root = path.parent().unwrap_or(Path::new("."));
    let mut aliases = Vec::new();
    for original in &d.tensors {
        let mut t = original.clone();
        t.name = language_name(&t.name)
            .ok_or_else(|| bad(format!("unsupported Qwen3.5 GGUF tensor {}", t.name)))?;
        if t.name.ends_with(".conv1d.weight") && t.dimensions.len() == 2 {
            t.dimensions.insert(1, 1);
        }
        aliases.push((0, t));
    }
    let vision_path = root.join("vision.gguf");
    let vision = if vision_path.try_exists()? {
        let descriptor = crate::inspect(&vision_path, crate::ParseLimits::default())?;
        let v = &json["vision_config"];
        for (key, source) in [
            ("projection_dim", "out_hidden_size"),
            ("embedding_length", "hidden_size"),
            ("feed_forward_length", "intermediate_size"),
            ("block_count", "depth"),
            ("attention.head_count", "num_heads"),
            ("patch_size", "patch_size"),
            ("spatial_merge_size", "spatial_merge_size"),
        ] {
            if descriptor.uint(&format!("clip.vision.{key}")) != v[source].as_u64() {
                return Err(bad(format!(
                    "GGUF/source vision configuration mismatch: {key}"
                )));
            }
        }
        let keys = [
            "has_vision_encoder",
            "vision.projection_dim",
            "vision.image_size",
            "vision.patch_size",
            "vision.embedding_length",
            "vision.feed_forward_length",
            "vision.block_count",
            "vision.attention.head_count",
            "vision.image_mean",
            "vision.image_std",
            "projector_type",
            "use_gelu",
            "vision.spatial_merge_size",
            "vision.attention.layer_norm_epsilon",
            "vision.is_deepstack_layers",
        ];
        if descriptor.metadata.keys().any(|key| {
            key.strip_prefix("clip.")
                .is_some_and(|key| !keys.contains(&key))
        }) || descriptor
            .get("clip.vision.attention.layer_norm_epsilon")
            .and_then(crate::gguf::Value::float)
            .map(|v| v as f32)
            != Some(1e-6)
            || !matches!(
                descriptor.get("clip.use_gelu"),
                Some(crate::gguf::Value::Bool(true))
            )
        {
            return Err(bad("unimplemented GGUF vision configuration option"));
        }
        for key in ["clip.vision.image_mean", "clip.vision.image_std"] {
            if descriptor
                .get(key)
                .and_then(crate::gguf::Value::array)
                .is_none_or(|a| {
                    a.len() != 3 || a.iter().any(|v| v.float().map(|v| v as f32) != Some(0.5))
                })
            {
                return Err(bad("unsupported GGUF vision normalization"));
            }
        }
        if descriptor.architecture.as_deref() != Some("clip")
            || descriptor.text("clip.projector_type") != Some("qwen3vl_merger")
            || !matches!(
                descriptor.get("clip.has_vision_encoder"),
                Some(crate::gguf::Value::Bool(true))
            )
            || descriptor
                .get("clip.vision.is_deepstack_layers")
                .and_then(crate::gguf::Value::array)
                .is_none_or(|a| {
                    a.len() != json["vision_config"]["depth"].as_u64().unwrap_or(0) as usize
                        || a.iter()
                            .any(|v| !matches!(v, crate::gguf::Value::Bool(false)))
                })
        {
            return Err(bad("unsupported Qwen3.5 GGUF vision component"));
        }
        Some(descriptor)
    } else {
        None
    };
    let mut parts = vec![(path, d)];
    let mut patch = None;
    if let Some(vision) = &vision {
        parts.push((&vision_path, vision));
        let mut consumed = BTreeSet::new();
        for original in &vision.tensors {
            if matches!(
                original.name.as_str(),
                "v.patch_embd.weight" | "v.patch_embd.weight.1"
            ) {
                continue;
            }
            let mut t = original.clone();
            t.name = vision_name(&t.name)
                .ok_or_else(|| bad(format!("unrecognized GGUF vision tensor {}", t.name)))?;
            if !matches!(t.dtype, 0 | 1 | 30) || !consumed.insert(t.name.clone()) {
                return Err(bad("unsupported/duplicate GGUF vision encoding"));
            }
            aliases.push((1, t));
        }
        let left = vision
            .tensor("v.patch_embd.weight")
            .ok_or_else(|| bad("missing temporal patch half"))?
            .clone();
        let right = vision
            .tensor("v.patch_embd.weight.1")
            .ok_or_else(|| bad("missing temporal patch half"))?
            .clone();
        let mut logical = left.clone();
        logical.name = "model.visual.patch_embed.proj.weight".into();
        logical.dimensions = vec![
            16,
            16,
            2,
            3,
            *left
                .dimensions
                .last()
                .ok_or_else(|| bad("invalid patch dimensions"))?,
        ];
        logical.bytes = left
            .bytes
            .checked_add(right.bytes)
            .ok_or_else(|| bad("patch size overflow"))?;
        patch = Some((1, left, right, logical));
    }
    let set = TensorSet::from_gguf(path, &parts, aliases, patch, cancel)?;
    config.validate_tensors(&set)?;
    Ok((config, set))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::Value;
    fn source() -> serde_json::Value {
        serde_json::json!({"model_type":"qwen3_5","tie_word_embeddings":true,"text_config":{
            "model_type":"qwen3_5_text","hidden_act":"silu","attention_bias":false,"attn_output_gate":true,"attention_dropout":0.,
            "mamba_ssm_dtype":"float32","dtype":"bfloat16","hidden_size":1024,"intermediate_size":3584,"vocab_size":248320,"eos_token_id":248044,
            "num_hidden_layers":24,"layer_types":(0..24).map(|i| if i%4==3 {"full_attention"} else {"linear_attention"}).collect::<Vec<_>>(),
            "num_attention_heads":8,"num_key_value_heads":2,"head_dim":256,"linear_num_key_heads":16,"linear_num_value_heads":16,
            "linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,"rms_norm_eps":1e-6,"tie_word_embeddings":true,
            "max_position_embeddings":262144,"rope_parameters":{"rope_type":"default","rope_theta":10000000.,"partial_rotary_factor":0.25}
        }})
    }
    fn descriptor() -> Descriptor {
        let values: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../fixtures/qwen35/gguf/language-metadata-v1.json"
        ))
        .unwrap();
        fn value(v: &serde_json::Value) -> Value {
            match v {
                serde_json::Value::String(s) => Value::String(s.clone()),
                serde_json::Value::Bool(b) => Value::Bool(*b),
                serde_json::Value::Array(a) => Value::Array(a.iter().map(value).collect()),
                n if n.is_u64() => Value::Unsigned(n.as_u64().unwrap()),
                n if n.is_i64() => Value::Signed(n.as_i64().unwrap()),
                n => Value::Float(n.as_f64().unwrap()),
            }
        }
        let inventory: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../fixtures/qwen35/gguf/f16.inventory.json"
        ))
        .unwrap();
        let mut d = Descriptor {
            version: 3,
            byte_order: "little".into(),
            file_bytes: 0,
            data_offset: 0,
            architecture: Some("qwen35".into()),
            tensor_bytes: 0,
            encodings: Default::default(),
            metadata_fingerprints: serde_json::from_value(
                inventory["metadata_fingerprints"].clone(),
            )
            .unwrap(),
            header_sha256: String::new(),
            tensors: Vec::new(),
            metadata: values
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), value(v)))
                .collect(),
        };
        // Match the GGUF signed integer array representation; do not rely on unsigned-only accessors.
        d.metadata.insert(
            "qwen35.rope.dimension_sections".into(),
            Value::Array([11, 11, 10, 0].map(Value::Signed).to_vec()),
        );
        d.metadata.insert(
            "tokenizer.chat_template".into(),
            Value::String("unqualified test template".into()),
        );
        d
    }
    #[test]
    fn incompatible_container_graph_and_tokenizer_options_are_rejected_before_weights() {
        let source = source();
        let d = descriptor();
        // The complete graph/tokenizer metadata passes, then the deliberately different template fails.
        assert_eq!(
            configure(&d, &source).unwrap_err().code,
            ErrorCode::UnsupportedChatTemplate
        );
        for (key, value) in [
            ("qwen35.embedding_length", Value::Unsigned(2048)),
            ("qwen35.full_attention_interval", Value::Unsigned(3)),
            ("qwen35.future_graph_option", Value::Bool(true)),
            ("split.count", Value::Unsigned(2)),
            ("general.quantization_version", Value::Unsigned(99)),
            (
                "qwen35.rope.dimension_sections",
                Value::Array([11, 11, 10, -1].map(Value::Signed).to_vec()),
            ),
        ] {
            let mut changed = d.clone();
            changed.metadata.insert(key.into(), value);
            assert_eq!(
                configure(&changed, &source).unwrap_err().code,
                ErrorCode::UnsupportedModelArchitecture,
                "{key}"
            );
        }
        let mut changed = d.clone();
        changed
            .metadata
            .insert("tokenizer.ggml.future_option".into(), Value::Bool(true));
        assert_eq!(
            configure(&changed, &source).unwrap_err().code,
            ErrorCode::UnsupportedTokenizer
        );
        changed = d.clone();
        changed
            .metadata_fingerprints
            .insert("tokenizer.ggml.merges".into(), "changed".into());
        assert_eq!(
            configure(&changed, &source).unwrap_err().code,
            ErrorCode::UnsupportedTokenizer
        );
        changed = d;
        changed.byte_order = "big".into();
        assert_eq!(
            configure(&changed, &source).unwrap_err().code,
            ErrorCode::UnsupportedModelArchitecture
        );
    }
    #[test]
    fn unknown_tensor_names_never_alias_supported_parameters() {
        assert!(language_name("blk.0.future.weight").is_none());
        assert!(language_name("blk.bad.attn_q.weight").is_none());
        assert!(vision_name("v.blk.0.future.weight").is_none());
        assert!(vision_name("v.blk.bad.ln1.weight").is_none());
        assert!(vision_name("v.blk.0.ln1.invalid").is_none());
    }
}

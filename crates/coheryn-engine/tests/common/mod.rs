//! Small deterministic complete GGUF used by engine and service tests.
use std::collections::BTreeMap;
fn string(out: &mut Vec<u8>, s: &str) {
    out.extend((s.len() as u64).to_le_bytes());
    out.extend(s.as_bytes());
}
fn text(s: &str) -> Vec<u8> {
    let mut b = 8u32.to_le_bytes().to_vec();
    string(&mut b, s);
    b
}
fn uint(n: u32) -> Vec<u8> {
    let mut b = 4u32.to_le_bytes().to_vec();
    b.extend(n.to_le_bytes());
    b
}
fn float(n: f32) -> Vec<u8> {
    let mut b = 6u32.to_le_bytes().to_vec();
    b.extend(n.to_le_bytes());
    b
}
fn array_text(a: &[String]) -> Vec<u8> {
    let mut b = 9u32.to_le_bytes().to_vec();
    b.extend(8u32.to_le_bytes());
    b.extend((a.len() as u64).to_le_bytes());
    for s in a {
        string(&mut b, s);
    }
    b
}
#[allow(dead_code)]
pub fn tiny_model() -> Vec<u8> {
    build_model(4, None)
}
#[allow(dead_code)]
pub fn conversation_model() -> Vec<u8> {
    let template: String = serde_json::from_str(include_str!(
        "../../../../fixtures/text/qwen-chat-template.json"
    ))
    .unwrap();
    build_with_template(4, None, &template)
}
#[allow(dead_code)]
pub fn packed_model(dtype: u32) -> Vec<u8> {
    build_model(256, Some(dtype))
}
fn build_model(hidden: usize, packed: Option<u32>) -> Vec<u8> {
    build_with_template(hidden, packed, "{{ messages[0].content }}")
}
#[allow(dead_code)]
pub fn tool_marker_model() -> Vec<u8> {
    let template: String = serde_json::from_str(include_str!(
        "../../../../fixtures/text/qwen-chat-template.json"
    ))
    .unwrap();
    build_with_options(4, None, &template, true)
}
fn build_with_template(hidden: usize, packed: Option<u32>, template: &str) -> Vec<u8> {
    build_with_options(hidden, packed, template, false)
}
fn build_with_options(
    hidden: usize,
    packed: Option<u32>,
    template: &str,
    tool_tokens: bool,
) -> Vec<u8> {
    build_profile(hidden, packed, template, tool_tokens, false)
}
#[allow(dead_code)]
pub fn tiny_vlm_model() -> Vec<u8> {
    let template: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../fixtures/vlm/qwen2-vl-2b/chat_template.json"
    ))
    .unwrap();
    build_profile(
        256,
        Some(1),
        template["chat_template"].as_str().unwrap(),
        false,
        true,
    )
}
fn build_profile(
    hidden: usize,
    packed: Option<u32>,
    template: &str,
    tool_tokens: bool,
    vision: bool,
) -> Vec<u8> {
    let mut metadata = BTreeMap::new();
    for (key, value) in [
        ("general.architecture", "qwen2"),
        ("tokenizer.ggml.model", "gpt2"),
        ("tokenizer.ggml.pre", "qwen2"),
        ("tokenizer.chat_template", template),
    ] {
        metadata.insert(key, text(value));
    }
    for (key, value) in [
        ("qwen2.block_count", 1),
        ("qwen2.embedding_length", hidden as u32),
        ("qwen2.feed_forward_length", (hidden * 2) as u32),
        ("qwen2.attention.head_count", 2),
        ("qwen2.attention.head_count_kv", 1),
        (
            "qwen2.context_length",
            if template == "{{ messages[0].content }}" {
                128
            } else {
                1024
            },
        ),
        ("tokenizer.ggml.eos_token_id", 258),
    ] {
        metadata.insert(key, uint(value));
    }
    metadata.insert("qwen2.attention.layer_norm_rms_epsilon", float(1e-6));
    metadata.insert("qwen2.rope.freq_base", float(1000000.));
    let mut bytes: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut alphabet = bytes.clone();
    let mut extra = 0;
    for b in 0..256 {
        if !bytes.contains(&b) {
            alphabet.push(256 + extra);
            extra += 1;
        }
    }
    // Alphabet is bijective and all 256 byte values are represented.
    bytes.clear();
    let mut tokens: Vec<String> = alphabet
        .into_iter()
        .map(|n| char::from_u32(n).unwrap().to_string())
        .collect();
    tokens.extend(if tool_tokens {
        [
            "<tool_call>".into(),
            "</tool_call>".into(),
            "<|im_end|>".into(),
        ]
    } else {
        [
            "<|endoftext|>".into(),
            "<|im_start|>".into(),
            "<|im_end|>".into(),
        ]
    });
    if vision {
        tokens.extend([
            "<|vision_start|>".into(),
            "<|vision_end|>".into(),
            "<|image_pad|>".into(),
        ]);
    }
    let vocabulary = tokens.len();
    metadata.insert("tokenizer.ggml.tokens", array_text(&tokens));
    metadata.insert("tokenizer.ggml.merges", array_text(&[]));
    let mut types = 9u32.to_le_bytes().to_vec();
    types.extend(4u32.to_le_bytes());
    types.extend((vocabulary as u64).to_le_bytes());
    for i in 0..vocabulary {
        types.extend((if i >= 256 { 3u32 } else { 1u32 }).to_le_bytes());
    }
    metadata.insert("tokenizer.ggml.token_type", types);
    let mut metadata: BTreeMap<String, Vec<u8>> = metadata
        .into_iter()
        .map(|(k, v)| {
            (
                if vision {
                    k.replace("qwen2.", "qwen2vl.")
                } else {
                    k.to_owned()
                },
                v,
            )
        })
        .collect();
    if vision {
        metadata.insert("general.architecture".into(), text("qwen2vl"));
        let mut sections = 9u32.to_le_bytes().to_vec();
        sections.extend(4u32.to_le_bytes());
        sections.extend(4u64.to_le_bytes());
        for n in [16u32, 24, 24, 0] {
            sections.extend(n.to_le_bytes());
        }
        metadata.insert("qwen2vl.rope.dimension_sections".into(), sections);
    }
    let mut tensors = vec![
        (
            "token_embd.weight".to_string(),
            vec![hidden, vocabulary],
            1f32,
        ),
        ("output_norm.weight".to_string(), vec![hidden], 1.),
        ("output.weight".to_string(), vec![hidden, vocabulary], 0.),
    ];
    for (name, shape, value) in [
        ("attn_norm.weight", vec![hidden], 1.),
        ("ffn_norm.weight", vec![hidden], 1.),
        ("attn_q.weight", vec![hidden, hidden], 0.),
        ("attn_k.weight", vec![hidden, hidden / 2], 0.),
        ("attn_v.weight", vec![hidden, hidden / 2], 0.),
        ("attn_q.bias", vec![hidden], 0.),
        ("attn_k.bias", vec![hidden / 2], 0.),
        ("attn_v.bias", vec![hidden / 2], 0.),
        ("attn_output.weight", vec![hidden, hidden], 0.),
        ("ffn_gate.weight", vec![hidden, hidden * 2], 0.),
        ("ffn_up.weight", vec![hidden, hidden * 2], 0.),
        ("ffn_down.weight", vec![hidden * 2, hidden], 0.),
    ] {
        tensors.push((format!("blk.0.{name}"), shape, value));
    }
    let mut b = b"GGUF".to_vec();
    b.extend(3u32.to_le_bytes());
    b.extend((tensors.len() as u64).to_le_bytes());
    b.extend((metadata.len() as u64).to_le_bytes());
    for (k, v) in metadata {
        string(&mut b, &k);
        b.extend(v);
    }
    let mut data = Vec::new();
    for (name, shape, value) in tensors {
        data.resize(data.len().div_ceil(32) * 32, 0);
        string(&mut b, &name);
        b.extend((shape.len() as u32).to_le_bytes());
        for d in &shape {
            b.extend((*d as u64).to_le_bytes());
        }
        let dtype = if shape.len() == 2 {
            packed.unwrap_or(0)
        } else {
            0
        };
        b.extend(dtype.to_le_bytes());
        b.extend((data.len() as u64).to_le_bytes());
        let elements = shape.iter().product::<usize>();
        if dtype == 0 {
            for _ in 0..elements {
                data.extend(value.to_le_bytes());
            }
        } else {
            let block = constant_block(dtype, value);
            let n = if dtype < 12 { 32 } else { 256 };
            for _ in 0..elements / n {
                data.extend(&block);
            }
        }
    }
    b.resize(b.len().div_ceil(32) * 32, 0);
    b.extend(data);
    b
}

fn constant_block(dtype: u32, value: f32) -> Vec<u8> {
    let one = 0x3c00u16.to_le_bytes();
    let scale = if value == 0. { [0, 0] } else { one };
    match dtype {
        1 => scale.repeat(32),
        8 => {
            let mut b = scale.to_vec();
            b.extend([1; 32]);
            b
        }
        2 => {
            let mut b = scale.to_vec();
            b.extend([0x99; 16]);
            b
        }
        6 => {
            let mut b = scale.to_vec();
            b.extend([255; 4]);
            b.extend([0x11; 16]);
            b
        }
        12 => {
            let mut b = scale.to_vec();
            b.extend([0, 0]);
            b.extend([1; 4]);
            b.extend([0; 4]);
            b.extend([1; 4]);
            b.extend([0x11; 128]);
            b
        }
        14 => {
            let mut b = vec![0x11; 128];
            b.extend([0xaa; 64]);
            b.extend([1; 16]);
            b.extend(scale);
            b
        }
        _ => panic!("unsupported fixture dtype"),
    }
}

/// Small nonconstant bidirectional encoder; exactly representable matrix weights.
#[allow(dead_code)]
pub fn embedding_model(dtype: u32) -> Vec<u8> {
    let hidden = 32usize;
    let tokens = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "▁hello", "▁world", "▁a", "a", "▁cafe",
        "▁中", "▁!", "▁[", "▁]", "▁cls",
    ];
    let mut metadata = BTreeMap::new();
    metadata.insert("general.architecture", text("nomic-bert"));
    metadata.insert("tokenizer.ggml.model", text("bert"));
    metadata.insert(
        "tokenizer.ggml.tokens",
        array_text(&tokens.map(str::to_owned)),
    );
    for (key, value) in [
        ("nomic-bert.block_count", 2),
        ("nomic-bert.embedding_length", 32),
        ("nomic-bert.feed_forward_length", 64),
        ("nomic-bert.attention.head_count", 2),
        ("nomic-bert.context_length", 512),
        ("nomic-bert.pooling_type", 1),
        ("tokenizer.ggml.token_type_count", 2),
        ("tokenizer.ggml.bos_token_id", 2),
        ("tokenizer.ggml.eos_token_id", 3),
        ("tokenizer.ggml.unknown_token_id", 1),
        ("tokenizer.ggml.padding_token_id", 0),
    ] {
        metadata.insert(key, uint(value));
    }
    metadata.insert(
        "nomic-bert.attention.causal",
        [7u32.to_le_bytes().as_slice(), &[0]].concat(),
    );
    metadata.insert("nomic-bert.attention.layer_norm_epsilon", float(1e-5));
    metadata.insert("nomic-bert.rope.freq_base", float(1000.));
    let mut tensors = vec![
        ("token_embd.weight".to_owned(), vec![hidden, tokens.len()]),
        ("token_types.weight".to_owned(), vec![hidden, 2]),
        ("token_embd_norm.weight".to_owned(), vec![hidden]),
        ("token_embd_norm.bias".to_owned(), vec![hidden]),
    ];
    for i in 0..2 {
        for (name, shape) in [
            ("attn_qkv.weight", vec![hidden, 3 * hidden]),
            ("attn_output.weight", vec![hidden, hidden]),
            ("ffn_up.weight", vec![hidden, 64]),
            ("ffn_gate.weight", vec![hidden, 64]),
            ("ffn_down.weight", vec![64, hidden]),
            ("attn_output_norm.weight", vec![hidden]),
            ("attn_output_norm.bias", vec![hidden]),
            ("layer_output_norm.weight", vec![hidden]),
            ("layer_output_norm.bias", vec![hidden]),
        ] {
            tensors.push((format!("blk.{i}.{name}"), shape));
        }
    }
    let mut b = b"GGUF".to_vec();
    b.extend(3u32.to_le_bytes());
    b.extend((tensors.len() as u64).to_le_bytes());
    b.extend((metadata.len() as u64).to_le_bytes());
    for (key, value) in metadata {
        string(&mut b, key);
        b.extend(value);
    }
    let mut data = Vec::new();
    for (name, shape) in tensors {
        data.resize(data.len().div_ceil(32) * 32, 0);
        string(&mut b, &name);
        b.extend((shape.len() as u32).to_le_bytes());
        for n in &shape {
            b.extend((*n as u64).to_le_bytes());
        }
        let encoding = if shape.len() == 1 { 0 } else { dtype };
        b.extend(encoding.to_le_bytes());
        b.extend((data.len() as u64).to_le_bytes());
        let values: Vec<f32> = (0..shape.iter().product())
            .map(|i: usize| {
                if shape.len() == 1 {
                    if name.ends_with("weight") {
                        1.
                    } else {
                        (i as f32 - 16.) / 128.
                    }
                } else {
                    (((i / shape[0] * 3 + i % shape[0] * 5) % 17) as f32 - 8.) / 64.
                }
            })
            .collect();
        match encoding {
            0 => {
                for v in values {
                    data.extend(v.to_le_bytes());
                }
            }
            1 => {
                for v in values {
                    data.extend({
                        let k = (v * 64.) as i32;
                        let magnitude: u16 = [
                            0, 0x2400, 0x2800, 0x2a00, 0x2c00, 0x2d00, 0x2e00, 0x2f00, 0x3000,
                        ][k.unsigned_abs() as usize];
                        (magnitude | if k < 0 { 0x8000 } else { 0 }).to_le_bytes()
                    });
                }
            }
            8 => {
                for block in values.as_chunks::<32>().0 {
                    data.extend(0x2400u16.to_le_bytes());
                    data.extend(block.iter().map(|v| (*v * 64.) as i8 as u8));
                }
            }
            _ => panic!("unsupported embedding fixture encoding"),
        }
    }
    b.resize(b.len().div_ceil(32) * 32, 0);
    b.extend(data);
    b
}

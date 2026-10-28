use std::{collections::BTreeMap, io::Write};
fn string(b: &mut Vec<u8>, s: &str) {
    b.extend((s.len() as u64).to_le_bytes());
    b.extend(s.as_bytes());
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
fn floats(v: &[f32]) -> Vec<u8> {
    let mut b = 9u32.to_le_bytes().to_vec();
    b.extend(6u32.to_le_bytes());
    b.extend((v.len() as u64).to_le_bytes());
    for x in v {
        b.extend(x.to_le_bytes());
    }
    b
}
/// Independently constructed small, nonzero graph; no publisher weights are embedded.
pub fn component(output_width: usize) -> tempfile::NamedTempFile {
    let mut meta: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    meta.insert("general.architecture".into(), text("clip"));
    meta.insert("clip.projector_type".into(), text("qwen2vl_merger"));
    meta.insert("clip.has_vision_encoder".into(), vec![7, 0, 0, 0, 1]);
    for (k, n) in [
        ("block_count", 2),
        ("embedding_length", 8),
        ("feed_forward_length", 17),
        ("attention.head_count", 2),
        ("patch_size", 14),
        ("image_size", 56),
        ("projection_dim", output_width as u32),
    ] {
        meta.insert(format!("clip.vision.{k}"), uint(n));
    }
    meta.insert(
        "clip.vision.attention.layer_norm_epsilon".into(),
        float(1e-6),
    );
    meta.insert(
        "clip.vision.image_mean".into(),
        floats(&[0.48145466, 0.4578275, 0.40821073]),
    );
    meta.insert(
        "clip.vision.image_std".into(),
        floats(&[0.26862954, 0.261_302_6, 0.275_777_1]),
    );
    let mut tensors: Vec<(String, Vec<usize>, u32)> = vec![];
    for name in ["v.patch_embd.weight", "v.patch_embd.weight.1"] {
        tensors.push((name.into(), vec![14, 14, 3, 8], 1));
    }
    for layer in 0..2 {
        for (name, input, output) in [
            ("attn_q", 8, 8),
            ("attn_k", 8, 8),
            ("attn_v", 8, 8),
            ("attn_out", 8, 8),
            ("ffn_down", 8, 32),
            ("ffn_up", 32, 8),
        ] {
            tensors.push((
                format!("v.blk.{layer}.{name}.weight"),
                vec![input, output],
                1,
            ));
            tensors.push((format!("v.blk.{layer}.{name}.bias"), vec![output], 0));
        }
        for name in ["ln1", "ln2"] {
            for suffix in ["weight", "bias"] {
                tensors.push((format!("v.blk.{layer}.{name}.{suffix}"), vec![8], 0));
            }
        }
    }
    for suffix in ["weight", "bias"] {
        tensors.push((format!("v.post_ln.{suffix}"), vec![8], 0));
    }
    for (name, input, output) in [("mm.0", 32, 32), ("mm.2", 32, output_width)] {
        tensors.push((format!("{name}.weight"), vec![input, output], 1));
        tensors.push((format!("{name}.bias"), vec![output], 0));
    }
    let mut b = b"GGUF".to_vec();
    b.extend(3u32.to_le_bytes());
    b.extend((tensors.len() as u64).to_le_bytes());
    b.extend((meta.len() as u64).to_le_bytes());
    for (k, v) in meta {
        string(&mut b, &k);
        b.extend(v);
    }
    let mut offset = 0usize;
    for (name, dims, dtype) in &tensors {
        string(&mut b, name);
        b.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
            b.extend((*d as u64).to_le_bytes());
        }
        b.extend(dtype.to_le_bytes());
        b.extend((offset as u64).to_le_bytes());
        offset = (offset + dims.iter().product::<usize>() * if *dtype == 1 { 2 } else { 4 })
            .next_multiple_of(32);
    }
    b.resize(b.len().next_multiple_of(32), 0);
    let data = b.len();
    for (index, (name, dims, dtype)) in tensors.iter().enumerate() {
        for i in 0..dims.iter().product::<usize>() {
            let value = if name.contains("ln") && name.ends_with("weight") {
                1.0
            } else {
                (((i * 13 + index * 7) % 23) as f32 - 11.) * 0.0078125
            };
            if *dtype == 1 {
                b.extend(half::f16::from_f32(value).to_le_bytes());
            } else {
                b.extend(value.to_le_bytes());
            }
        }
        b.resize(data + (b.len() - data).next_multiple_of(32), 0);
    }
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(&b).unwrap();
    f.flush().unwrap();
    f
}

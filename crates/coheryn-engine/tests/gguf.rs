use coheryn_engine::{
    ErrorCode,
    gguf::{ParseLimits, parse},
};
use std::io::Cursor;

fn string(out: &mut Vec<u8>, s: &str) {
    out.extend((s.len() as u64).to_le_bytes());
    out.extend(s.as_bytes());
}
fn file(offsets: &[u64], dtype: u32, dimensions: &[u64]) -> Vec<u8> {
    let mut b = b"GGUF".to_vec();
    b.extend(3u32.to_le_bytes());
    b.extend((offsets.len() as u64).to_le_bytes());
    b.extend(0u64.to_le_bytes());
    for (i, offset) in offsets.iter().enumerate() {
        string(&mut b, &format!("tensor{i}"));
        b.extend((dimensions.len() as u32).to_le_bytes());
        for d in dimensions {
            b.extend(d.to_le_bytes());
        }
        b.extend(dtype.to_le_bytes());
        b.extend(offset.to_le_bytes());
    }
    b.resize(b.len().div_ceil(32) * 32, 0);
    b.resize(b.len() + 128, 0);
    b
}
fn check(b: &[u8]) -> coheryn_engine::Result<coheryn_engine::Descriptor> {
    parse(Cursor::new(b), b.len() as u64, ParseLimits::default())
}
#[test]
fn valid_tensor_layout() {
    let d = check(&file(&[0, 32], 0, &[2, 2])).unwrap();
    assert_eq!(d.tensor_bytes, 32);
    assert_eq!(d.tensors.len(), 2);
}
#[test]
fn every_truncated_header_rejected() {
    let b = file(&[0], 0, &[2, 2]);
    let d = check(&b).unwrap();
    for n in 0..d.data_offset as usize + 16 {
        assert!(check(&b[..n]).is_err(), "accepted truncation {n}");
    }
}
#[test]
fn offset_alignment_range_and_overlap_rejected() {
    for offsets in [&[1][..], &[u64::MAX - 31], &[128], &[0, 0]] {
        assert!(check(&file(offsets, 0, &[2, 2])).is_err());
    }
}
#[test]
fn tensor_shape_and_encoding_rejected_safely() {
    for dims in [&[0][..], &[u64::MAX, 2], &[1, 1, 1, 1, 1]] {
        assert!(check(&file(&[0], 0, dims)).is_err());
    }
    assert_eq!(
        check(&file(&[0], 999, &[32])).unwrap_err().code,
        ErrorCode::UnsupportedGgufTensorType
    );
    assert!(check(&file(&[0], 2, &[31])).is_err());
    assert_eq!(check(&file(&[0], 2, &[32])).unwrap().tensor_bytes, 18);
}
#[test]
fn malicious_counts_and_string_lengths_rejected() {
    let mut b = file(&[0], 0, &[1]);
    b[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(check(&b).is_err());
    let mut b = file(&[0], 0, &[1]);
    b[24..32].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(check(&b).is_err());
}
#[test]
fn version_is_explicit() {
    let mut b = file(&[], 0, &[]);
    b[4..8].copy_from_slice(&2u32.to_le_bytes());
    assert_eq!(
        check(&b).unwrap_err().code,
        ErrorCode::UnsupportedGgufVersion
    );
}
#[test]
fn bounded_parser_mutation_corpus_never_panics() {
    let original = file(&[0, 32], 0, &[2, 2]);
    let mut state = 42u64;
    for _ in 0..2000 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut b = original.clone();
        let index = (state as usize) % b.len();
        b[index] ^= (state >> 32) as u8;
        let _ = check(&b);
    }
}
#[test]
fn metadata_limits_are_enforced_before_allocation() {
    let mut b = b"GGUF".to_vec();
    b.extend(3u32.to_le_bytes());
    b.extend(0u64.to_le_bytes());
    b.extend(1u64.to_le_bytes());
    string(&mut b, "huge");
    b.extend(9u32.to_le_bytes());
    b.extend(8u32.to_le_bytes());
    b.extend(u64::MAX.to_le_bytes());
    assert!(check(&b).is_err());
}
#[test]
fn big_endian_container_is_detected() {
    let mut b = b"GGUF".to_vec();
    b.extend(3u32.to_be_bytes());
    b.extend(0u64.to_be_bytes());
    b.extend(0u64.to_be_bytes());
    b.resize(32, 0);
    assert_eq!(check(&b).unwrap().byte_order, "big");
}

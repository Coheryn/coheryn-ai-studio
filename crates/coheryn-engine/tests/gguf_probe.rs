//! Routing consumes bounded metadata, while complete container validation stays mandatory.
use coheryn_engine::{
    Cancellation, ErrorCode, ParseLimits,
    gguf::{parse, probe_architecture},
};
use std::io::{Cursor, Read, Seek, SeekFrom};

struct Fixture {
    bytes: Vec<u8>,
    big: bool,
}
impl Fixture {
    fn new(big: bool, metadata: u64) -> Self {
        let mut value = Self {
            bytes: b"GGUF".to_vec(),
            big,
        };
        value.u32(3);
        value.u64(0);
        value.u64(metadata);
        value
    }
    fn u32(&mut self, n: u32) {
        self.bytes.extend(if self.big {
            n.to_be_bytes()
        } else {
            n.to_le_bytes()
        });
    }
    fn u64(&mut self, n: u64) {
        self.bytes.extend(if self.big {
            n.to_be_bytes()
        } else {
            n.to_le_bytes()
        });
    }
    fn string(&mut self, text: &str) {
        self.u64(text.len() as u64);
        self.bytes.extend(text.as_bytes());
    }
    fn key(&mut self, name: &str, ty: u32) {
        self.string(name);
        self.u32(ty);
    }
    fn architecture(&mut self) {
        self.key("general.architecture", 8);
        self.string("qwen35");
    }
    fn finish(mut self) -> Vec<u8> {
        self.bytes.resize(self.bytes.len().div_ceil(32) * 32, 0);
        self.bytes
    }
}
fn probe(bytes: &[u8], limits: ParseLimits) -> coheryn_engine::Result<Option<String>> {
    probe_architecture(
        Cursor::new(bytes),
        bytes.len() as u64,
        limits,
        &Cancellation::default(),
    )
}

#[test]
fn routing_stops_before_tokenizer_arrays_and_does_not_validate_the_remainder() {
    let mut file = Fixture::new(false, 2);
    file.architecture();
    let boundary = file.bytes.len();
    file.key("tokenizer.ggml.tokens", 9);
    file.u32(8);
    file.u64(u64::MAX);
    let bytes = file.finish();
    let mut reader = Cursor::new(&bytes);
    assert_eq!(
        probe_architecture(
            &mut reader,
            bytes.len() as u64,
            ParseLimits::default(),
            &Cancellation::default()
        )
        .unwrap()
        .as_deref(),
        Some("qwen35")
    );
    assert_eq!(reader.position(), boundary as u64);
    assert!(
        parse(
            Cursor::new(&bytes),
            bytes.len() as u64,
            ParseLimits::default()
        )
        .is_err()
    );
}

#[test]
fn late_architecture_matches_full_inspection_for_all_metadata_types_and_byte_orders() {
    for big in [false, true] {
        let mut file = Fixture::new(big, 14);
        for ty in 0..=12 {
            file.key(&format!("extra.{ty}"), ty);
            match ty {
                0 | 1 => file.bytes.push(123),
                2 | 3 => file.bytes.extend(if big {
                    77u16.to_be_bytes()
                } else {
                    77u16.to_le_bytes()
                }),
                4 | 5 => file.u32(12345),
                6 => file.u32(1.25f32.to_bits()),
                7 => file.bytes.push(1),
                8 => file.string(&"বাংলা 😀".repeat(2048)),
                9 => {
                    file.u32(9);
                    file.u64(1);
                    file.u32(8);
                    file.u64(2);
                    file.string("one");
                    file.string("two");
                }
                10 | 11 => file.u64(123456),
                12 => file.u64(2.5f64.to_bits()),
                _ => unreachable!(),
            }
        }
        file.architecture();
        let bytes = file.finish();
        let expected = parse(
            Cursor::new(&bytes),
            bytes.len() as u64,
            ParseLimits::default(),
        )
        .unwrap();
        assert_eq!(
            probe(&bytes, ParseLimits::default()).unwrap(),
            expected.architecture
        );
    }
}

#[test]
fn malformed_counts_types_boolean_and_nested_arrays_remain_bounded_errors() {
    for ty in [7, 9, 13] {
        let mut file = Fixture::new(false, 2);
        file.key("extra", ty);
        match ty {
            7 => file.bytes.push(2),
            9 => {
                file.u32(8);
                file.u64(u64::MAX);
            }
            _ => {}
        }
        file.architecture();
        assert!(probe(&file.finish(), ParseLimits::default()).is_err());
    }
    let mut file = Fixture::new(false, 2);
    file.key("deep", 9);
    for _ in 0..5 {
        file.u32(9);
        file.u64(1);
    }
    file.architecture();
    assert!(probe(&file.finish(), ParseLimits::default()).is_err());
    let file = Fixture::new(false, u64::MAX).finish();
    assert!(probe(&file, ParseLimits::default()).is_err());
}

#[test]
fn skipped_values_keep_the_same_aggregate_allocation_limit_as_full_inspection() {
    let mut file = Fixture::new(false, 2);
    file.key("extra", 9);
    file.u32(4);
    file.u64(1024);
    for _ in 0..1024 {
        file.u32(1);
    }
    file.architecture();
    let bytes = file.finish();
    let limits = ParseLimits {
        max_metadata_allocation: 4096,
        ..Default::default()
    };
    assert_eq!(
        probe(&bytes, limits).unwrap_err().code,
        ErrorCode::ResourceExhausted
    );
    assert_eq!(
        parse(Cursor::new(&bytes), bytes.len() as u64, limits)
            .unwrap_err()
            .code,
        ErrorCode::ResourceExhausted
    );
}

#[test]
fn every_truncated_routing_prefix_is_rejected_and_missing_architecture_is_explicit() {
    let mut file = Fixture::new(false, 1);
    file.architecture();
    for length in 0..file.bytes.len() {
        assert!(probe(&file.bytes[..length], ParseLimits::default()).is_err());
    }
    assert_eq!(
        probe(&Fixture::new(false, 0).finish(), ParseLimits::default()).unwrap(),
        None
    );
    let mut file = Fixture::new(false, 1);
    file.key("general.architecture", 4);
    file.u32(7);
    assert_eq!(probe(&file.finish(), ParseLimits::default()).unwrap(), None);
}

struct CancellingReader {
    inner: Cursor<Vec<u8>>,
    cancel: Cancellation,
    after: u64,
}
impl Read for CancellingReader {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        let count = self.inner.read(output)?;
        if self.inner.position() >= self.after {
            self.cancel.cancel();
        }
        Ok(count)
    }
}
impl Seek for CancellingReader {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(position)
    }
}
#[test]
fn cancellation_is_observed_before_during_and_at_completion_of_routing() {
    let cancel = Cancellation::default();
    cancel.cancel();
    assert_eq!(
        probe_architecture(Cursor::new([]), 0, ParseLimits::default(), &cancel)
            .unwrap_err()
            .code,
        ErrorCode::Cancelled
    );
    for large in [false, true] {
        let mut file = Fixture::new(false, if large { 2 } else { 1 });
        if large {
            file.key("extra", 8);
            file.string(&"x".repeat(20000));
        }
        file.architecture();
        let boundary = if large { 8192 } else { file.bytes.len() as u64 };
        let bytes = file.finish();
        let length = bytes.len() as u64;
        let cancel = Cancellation::default();
        let reader = CancellingReader {
            inner: Cursor::new(bytes),
            cancel: cancel.clone(),
            after: boundary,
        };
        assert_eq!(
            probe_architecture(reader, length, ParseLimits::default(), &cancel)
                .unwrap_err()
                .code,
            ErrorCode::Cancelled
        );
    }
}

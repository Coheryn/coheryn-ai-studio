use coheryn_engine::{
    Cancellation, ErrorCode,
    safetensors::{Limits, TensorSet},
};
use serde_json::json;
use std::path::Path;

fn write(path: &Path, header: &str, data: &[u8]) {
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(data);
    std::fs::write(path, bytes).unwrap();
}
#[test]
fn single_and_indexed_tensors_preserve_bytes_and_bf16_values() {
    let d = tempfile::tempdir().unwrap();
    let a = d.path().join("a.safetensors");
    let b = d.path().join("b.safetensors");
    write(
        &a,
        r#"{"a":{"dtype":"BF16","shape":[3],"data_offsets":[0,6]}}"#,
        &[0x80, 0x3f, 0x00, 0xc0, 0x80, 0x00],
    );
    write(
        &b,
        r#"{"b":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}"#,
        &3.5f32.to_le_bytes(),
    );
    let cancel = Cancellation::default();
    let single = TensorSet::open(&a, Limits::default(), &cancel).unwrap();
    assert_eq!(
        single.read_vector("a").unwrap(),
        [1.0, -2.0, f32::MIN_POSITIVE]
    );
    let index = d.path().join("model.safetensors.index.json");
    std::fs::write(&index,json!({"metadata":{"total_size":10},"weight_map":{"a":"a.safetensors","b":"b.safetensors"}}).to_string()).unwrap();
    let set = TensorSet::open(d.path(), Limits::default(), &cancel).unwrap();
    assert_eq!(set.tensor_count(), 2);
    assert_eq!(set.files().len(), 2);
    assert_eq!(
        set.read_vector("a").unwrap(),
        single.read_vector("a").unwrap()
    );
    assert_eq!(set.read_vector("b").unwrap(), [3.5]);
    std::fs::write(&a, b"changed").unwrap();
    assert!(set.unchanged().is_err());
}
#[test]
fn malformed_headers_ranges_dtypes_and_limits_fail_without_panics() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("x.safetensors");
    let cases = [
        r#"{"x":{"dtype":"F32","dtype":"F16","shape":[1],"data_offsets":[0,4]}}"#,
        r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        r#"{"x":{"dtype":"F32","shape":[18446744073709551615,2],"data_offsets":[0,4]}}"#,
        r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[1,5]}}"#,
        r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#,
        r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"y":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        r#"{"__metadata__":{"a":1}}"#,
    ];
    for header in cases {
        write(&p, header, &[0; 4]);
        assert!(
            TensorSet::open(&p, Limits::default(), &Cancellation::default()).is_err(),
            "{header}"
        );
    }
    write(
        &p,
        r#"{"x":{"dtype":"F8_E4M3","shape":[4],"data_offsets":[0,4]}}"#,
        &[0; 4],
    );
    assert_eq!(
        TensorSet::open(&p, Limits::default(), &Cancellation::default())
            .err()
            .unwrap()
            .code,
        ErrorCode::UnsupportedFeature
    );
    write(
        &p,
        r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        &[0; 4],
    );
    assert!(
        TensorSet::open(
            &p,
            Limits {
                header_bytes: 4,
                ..Limits::default()
            },
            &Cancellation::default()
        )
        .is_err()
    );
    let data = std::fs::read(&p).unwrap();
    for end in 0..data.len() {
        std::fs::write(&p, &data[..end]).unwrap();
        assert!(TensorSet::open(&p, Limits::default(), &Cancellation::default()).is_err());
    }
}
#[test]
fn hostile_index_paths_duplicates_coverage_and_cancellation_are_rejected() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("model.safetensors.index.json");
    write(
        &d.path().join("a.safetensors"),
        r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        &[0; 4],
    );
    for name in [
        "../a.safetensors",
        "/tmp/a.safetensors",
        "C:\\a.safetensors",
        "a/../a.safetensors",
        "https://example/a.safetensors",
        "CON.safetensors",
    ] {
        std::fs::write(&p, json!({"weight_map":{"a":name}}).to_string()).unwrap();
        assert!(TensorSet::open(&p, Limits::default(), &Cancellation::default()).is_err());
    }
    for index in [
        r#"{"weight_map":{"a":"a.safetensors","a":"a.safetensors"}}"#,
        r#"{"weight_map":{"b":"a.safetensors"}}"#,
        r#"{"metadata":{"total_size":5},"weight_map":{"a":"a.safetensors"}}"#,
    ] {
        std::fs::write(&p, index).unwrap();
        assert!(TensorSet::open(&p, Limits::default(), &Cancellation::default()).is_err());
    }
    let cancel = Cancellation::default();
    cancel.cancel();
    assert_eq!(
        TensorSet::open(d.path().join("a.safetensors"), Limits::default(), &cancel)
            .err()
            .unwrap()
            .code,
        ErrorCode::Cancelled
    );
}
#[cfg(unix)]
#[test]
fn symlink_and_hardlink_shards_are_rejected() {
    use std::os::unix::fs::symlink;
    let d = tempfile::tempdir().unwrap();
    let a = d.path().join("a.safetensors");
    write(
        &a,
        r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        &[0; 4],
    );
    let link = d.path().join("linked.safetensors");
    symlink(&a, &link).unwrap();
    assert!(TensorSet::open(&link, Limits::default(), &Cancellation::default()).is_err());
    std::fs::hard_link(&a, d.path().join("hard.safetensors")).unwrap();
    assert!(TensorSet::open(&a, Limits::default(), &Cancellation::default()).is_err());
}

#[test]
fn ambiguous_entrypoints_and_exact_tensor_and_header_limits_are_rejected() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("model.safetensors");
    let header = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#;
    write(&p, header, &[0; 8]);
    let cancel = Cancellation::default();
    assert!(
        TensorSet::open(
            &p,
            Limits {
                tensors: 1,
                ..Limits::default()
            },
            &cancel
        )
        .is_err()
    );
    assert!(
        TensorSet::open(
            &p,
            Limits {
                aggregate_header_bytes: 1,
                ..Limits::default()
            },
            &cancel
        )
        .is_err()
    );
    std::fs::write(
        d.path().join("model.safetensors.index.json"),
        r#"{"weight_map":{"a":"model.safetensors","b":"model.safetensors"}}"#,
    )
    .unwrap();
    assert!(
        TensorSet::open(d.path(), Limits::default(), &cancel)
            .err()
            .unwrap()
            .message
            .contains("ambiguous")
    );
    assert_eq!(
        TensorSet::open(&p, Limits::default(), &cancel)
            .unwrap()
            .tensor_count(),
        2
    );
}

#[test]
fn valid_container_with_nonfinite_payload_reports_its_own_format() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("nonfinite.safetensors");
    write(
        &p,
        r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        &f32::NAN.to_le_bytes(),
    );
    let set = TensorSet::open(&p, Limits::default(), &Cancellation::default()).unwrap();
    assert_eq!(
        set.read_vector("a").unwrap_err().code,
        ErrorCode::InvalidSafetensors
    );
}

#[test]
fn reader_inspection_is_bounded_and_leaves_payload_unread() {
    use coheryn_engine::safetensors::{Limits, inspect_reader};
    let header = br#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend([0; 8]);
    let mut reader = std::io::Cursor::new(&bytes);
    let info = inspect_reader(
        &mut reader,
        bytes.len() as u64,
        Limits::default(),
        &Default::default(),
    )
    .unwrap();
    assert_eq!(reader.position(), 8 + header.len() as u64);
    assert_eq!(info.tensors[0].bytes, 8);
    assert_eq!(info.tensors[0].offset, reader.position());
    for header_bytes in [0, usize::MAX] {
        let mut unread = std::io::Cursor::new(&bytes);
        let error = inspect_reader(
            &mut unread,
            bytes.len() as u64,
            Limits {
                header_bytes,
                ..Default::default()
            },
            &Default::default(),
        )
        .unwrap_err();
        assert_eq!(error.code, ErrorCode::InvalidRequest);
        assert_eq!(unread.position(), 0);
    }
    let limits = Limits {
        header_bytes: 1,
        ..Default::default()
    };
    assert!(
        inspect_reader(
            bytes.as_slice(),
            bytes.len() as u64,
            limits,
            &Default::default()
        )
        .is_err()
    );
    let cancelled = coheryn_engine::Cancellation::default();
    cancelled.cancel();
    assert!(
        inspect_reader(
            bytes.as_slice(),
            bytes.len() as u64,
            Limits::default(),
            &cancelled
        )
        .is_err()
    );
    for n in 0..8 + header.len() {
        assert!(
            inspect_reader(
                &bytes[..n],
                bytes.len() as u64,
                Limits::default(),
                &Default::default()
            )
            .is_err()
        );
    }
}

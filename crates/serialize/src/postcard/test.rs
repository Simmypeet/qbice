use super::*;
use crate::Plugin;

#[test]
fn varint_u32_roundtrip() {
    let plugin = Plugin::new();
    let test_values: &[u32] =
        &[0, 1, 127, 128, 255, 256, 16383, 16384, u32::MAX / 2, u32::MAX];

    for &value in test_values {
        let bytes = encode(&value, &plugin).unwrap();
        let decoded: u32 = decode(&bytes, &plugin).unwrap();
        assert_eq!(value, decoded, "roundtrip failed for {value}");
    }
}

#[test]
fn zigzag_i32_roundtrip() {
    let plugin = Plugin::new();
    let test_values: &[i32] =
        &[0, 1, -1, 63, -64, 64, -65, 127, -128, i32::MAX, i32::MIN];

    for &value in test_values {
        let bytes = encode(&value, &plugin).unwrap();
        let decoded: i32 = decode(&bytes, &plugin).unwrap();
        assert_eq!(value, decoded, "roundtrip failed for {value}");
    }
}

#[test]
fn varint_compactness() {
    let plugin = Plugin::new();

    // Small values should be compact
    let bytes = encode(&0u32, &plugin).unwrap();
    assert_eq!(bytes.len(), 1);

    let bytes = encode(&127u32, &plugin).unwrap();
    assert_eq!(bytes.len(), 1);

    let bytes = encode(&128u32, &plugin).unwrap();
    assert_eq!(bytes.len(), 2);
}

#[test]
fn string_roundtrip() {
    let plugin = Plugin::new();
    let value = "Hello, World!".to_string();

    let bytes = encode(&value, &plugin).unwrap();
    let decoded: String = decode(&bytes, &plugin).unwrap();
    assert_eq!(value, decoded);
}

#[test]
#[allow(clippy::float_cmp)]
fn float_roundtrip() {
    let plugin = Plugin::new();

    let f32_value: f32 = std::f32::consts::PI;
    let bytes = encode(&f32_value, &plugin).unwrap();
    assert_eq!(bytes.len(), 4); // f32 is always 4 bytes
    let decoded: f32 = decode(&bytes, &plugin).unwrap();
    assert_eq!(f32_value, decoded);

    let f64_value: f64 = std::f64::consts::PI;
    let bytes = encode(&f64_value, &plugin).unwrap();
    assert_eq!(bytes.len(), 8); // f64 is always 8 bytes
    let decoded: f64 = decode(&bytes, &plugin).unwrap();
    assert_eq!(f64_value, decoded);
}

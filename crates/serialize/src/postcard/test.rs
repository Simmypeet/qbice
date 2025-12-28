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

// =============================================================================
// Derive macro tests
// =============================================================================

use crate::{Decode, Encode};

// Unit struct
#[derive(Debug, Clone, PartialEq, Encode, Decode)]
#[serialize_crate(crate)]
struct UnitStruct;

// Tuple struct
#[derive(Debug, Clone, PartialEq, Encode, Decode)]
#[serialize_crate(crate)]
struct TupleStruct(u64, String);

// Named fields struct
#[derive(Debug, Clone, PartialEq, Encode, Decode)]
#[serialize_crate(crate)]
struct NamedStruct {
    x: i32,
    y: i32,
    name: String,
}

// Struct with skip
#[derive(Debug, Clone, PartialEq, Encode, Decode)]
#[serialize_crate(crate)]
struct StructWithSkip {
    value: u32,
    #[serialize(skip)]
    skipped: Vec<u8>,
}

// Generic struct
#[derive(Debug, Clone, PartialEq, Encode, Decode)]
#[serialize_crate(crate)]
struct GenericStruct<T> {
    inner: T,
}

// Enum with all variant types
#[derive(Debug, Clone, PartialEq, Encode, Decode)]
#[serialize_crate(crate)]
enum TestEnum {
    Unit,
    Tuple(u32, String),
    Named { a: i32, b: bool },
}

// Enum with skip on field
#[derive(Debug, Clone, PartialEq, Encode, Decode)]
#[serialize_crate(crate)]
enum EnumWithSkip {
    Variant {
        value: u32,
        #[serialize(skip)]
        skipped: String,
    },
}

// Generic enum
#[derive(Debug, Clone, PartialEq, Encode, Decode)]
#[serialize_crate(crate)]
enum GenericEnum<T> {
    None,
    Some(T),
}

#[test]
fn derive_unit_struct_roundtrip() {
    let plugin = Plugin::new();
    let value = UnitStruct;

    let bytes = encode(&value, &plugin).unwrap();
    assert!(bytes.is_empty()); // Unit struct should produce no bytes
    let decoded: UnitStruct = decode(&bytes, &plugin).unwrap();
    assert_eq!(value, decoded);
}

#[test]
fn derive_tuple_struct_roundtrip() {
    let plugin = Plugin::new();
    let value = TupleStruct(42, "hello".to_string());

    let bytes = encode(&value, &plugin).unwrap();
    let decoded: TupleStruct = decode(&bytes, &plugin).unwrap();
    assert_eq!(value, decoded);
}

#[test]
fn derive_named_struct_roundtrip() {
    let plugin = Plugin::new();
    let value = NamedStruct { x: -10, y: 20, name: "test".to_string() };

    let bytes = encode(&value, &plugin).unwrap();
    let decoded: NamedStruct = decode(&bytes, &plugin).unwrap();
    assert_eq!(value, decoded);
}

#[test]
fn derive_struct_with_skip() {
    let plugin = Plugin::new();
    let value = StructWithSkip { value: 123, skipped: vec![1, 2, 3] };

    let bytes = encode(&value, &plugin).unwrap();
    let decoded: StructWithSkip = decode(&bytes, &plugin).unwrap();

    assert_eq!(decoded.value, 123);
    assert!(decoded.skipped.is_empty()); // Should be Default::default()
}

#[test]
fn derive_generic_struct_roundtrip() {
    let plugin = Plugin::new();
    let value = GenericStruct { inner: "generic".to_string() };

    let bytes = encode(&value, &plugin).unwrap();
    let decoded: GenericStruct<String> = decode(&bytes, &plugin).unwrap();
    assert_eq!(value, decoded);
}

#[test]
fn derive_enum_unit_variant() {
    let plugin = Plugin::new();
    let value = TestEnum::Unit;

    let bytes = encode(&value, &plugin).unwrap();
    let decoded: TestEnum = decode(&bytes, &plugin).unwrap();
    assert_eq!(value, decoded);
}

#[test]
fn derive_enum_tuple_variant() {
    let plugin = Plugin::new();
    let value = TestEnum::Tuple(42, "world".to_string());

    let bytes = encode(&value, &plugin).unwrap();
    let decoded: TestEnum = decode(&bytes, &plugin).unwrap();
    assert_eq!(value, decoded);
}

#[test]
fn derive_enum_named_variant() {
    let plugin = Plugin::new();
    let value = TestEnum::Named { a: -5, b: true };

    let bytes = encode(&value, &plugin).unwrap();
    let decoded: TestEnum = decode(&bytes, &plugin).unwrap();
    assert_eq!(value, decoded);
}

#[test]
fn derive_enum_with_skip() {
    let plugin = Plugin::new();
    let value = EnumWithSkip::Variant {
        value: 999,
        skipped: "should be skipped".to_string(),
    };

    let bytes = encode(&value, &plugin).unwrap();
    let decoded: EnumWithSkip = decode(&bytes, &plugin).unwrap();

    match decoded {
        EnumWithSkip::Variant { value, skipped } => {
            assert_eq!(value, 999);
            assert!(skipped.is_empty()); // Should be Default::default()
        }
    }
}

#[test]
fn derive_generic_enum_roundtrip() {
    let plugin = Plugin::new();

    let none_value: GenericEnum<i32> = GenericEnum::None;
    let bytes = encode(&none_value, &plugin).unwrap();
    let decoded: GenericEnum<i32> = decode(&bytes, &plugin).unwrap();
    assert_eq!(none_value, decoded);

    let some_value = GenericEnum::Some(42);
    let bytes = encode(&some_value, &plugin).unwrap();
    let decoded: GenericEnum<i32> = decode(&bytes, &plugin).unwrap();
    assert_eq!(some_value, decoded);
}

#[test]
fn derive_invalid_variant_index() {
    let plugin = Plugin::new();

    // Manually create invalid data (variant index 99 doesn't exist)
    let invalid_bytes = encode(&99usize, &plugin).unwrap();

    let result: std::io::Result<TestEnum> = decode(&invalid_bytes, &plugin);
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("invalid variant index"));
}

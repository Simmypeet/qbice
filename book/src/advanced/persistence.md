# Persistence

QBICE automatically persists query results to disk, allowing computation state to survive across program restarts. This chapter explains how persistence works and how to use it effectively.

## How It Works

When you execute a query, QBICE:

1. Computes the result
2. Serializes the key and value
3. Stores them in the database
4. Records metadata (fingerprint, timestamp, etc.)

On subsequent runs:

1. Engine loads existing data from the database
2. Cached results are available immediately
3. Only changed queries need recomputation

## Database Backends

QBICE supports pluggable storage backends:

### RocksDB (Recommended)

Production-ready embedded database with excellent performance:

```rust
use qbice::storage::kv_database::rocksdb::RocksDB;

let mut engine = Engine::<DefaultConfig>::new_with(
    Plugin::default(),
    RocksDB::factory("/path/to/db"),
    hasher,
)?;
```

**Pros:**

- Battle-tested in production
- Excellent performance
- Good compression
- ACID guarantees

**Cons:**

- Larger binary size
- C++ dependency

### fjall

Rust-native alternative:

```rust
use qbice::storage::kv_database::fjall::Fjall;

let mut engine = Engine::<DefaultConfig>::new_with(
    Plugin::default(),
    Fjall::factory("/path/to/db"),
    hasher,
)?;
```

**Pros:**

- Pure Rust
- Smaller binary
- Simpler dependencies

**Cons:**

- Less mature
- Potentially lower performance (Could be tuned to this use case, if you are interested in helping please reach out :D )

## Database Location

### Development

Use temporary directories for testing:

```rust
use tempfile::tempdir;

let temp_dir = tempdir()?;
let mut engine = Engine::<DefaultConfig>::new_with(
    Plugin::default(),
    RocksDB::factory(temp_dir.path()),
    hasher,
)?;

// Database deleted when temp_dir is dropped
```

## Serialization

Query keys and values must implement `Encode` and `Decode`:

```rust
use qbice::{Encode, Decode};

#[derive(
    Debug, Clone, PartialEq, Eq, Hash,
    StableHash, Identifiable,
    Encode, Decode,  // Required for persistence
)]
pub struct MyQuery {
    pub id: u64,
    pub name: String,
}

#[derive(Debug, Clone, StableHash, Encode, Decode)]
pub struct MyValue {
    pub result: String,
    pub metadata: Metadata,
}
```

### Custom Serialization

For types that don't implement `Encode/Decode`, wrap them or use newtype pattern:

```rust
#[derive(Debug, Clone, StableHash)]
pub struct MyValue {
    pub data: ThirdPartyType,  // Doesn't implement Encode/Decode
}

// Implement manual serialization
impl Encode for MyValue {
    fn encode<E: Encoder + ?Sized>(
        &self,
        encoder: &mut E,
        plugin: &Plugin,
        session: &mut Session,
    ) -> std::io::Result<()> {
        // Custom encoding logic
    }
}

impl Decode for MyValue {
    fn decode<D: Decoder + ?Sized>(
        decoder: &mut D,
        plugin: &Plugin,
        session: &mut Session,
    ) -> std::io::Result<Self> {
        // Custom decoding logic
    }
}
```

## Schema Evolution

Generally, we provide no guarantees for schema between versions at all. If you
change the structure of your queries or values, you must clear the database to
avoid runtime panics.

## Garbage Collection

Currently, QBICE still doesn't provide built-in garbage collection for the
database. We plan to add this feature in the future.

We imagine that users will specify nodes where you normally use it, and QBICE
will periodically clean up unreachable entries from the database.

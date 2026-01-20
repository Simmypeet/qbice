# Setting Up the Engine

Now that we have queries and executors, let's create and configure the QBICE engine. The engine is the central component that manages query execution, caching, and dependency tracking.

## Creating an Engine

The basic setup requires three components:

1. **Plugin** - For serialization/deserialization
2. **Database Factory** - For persistent storage
3. **Hasher Builder** - For stable hashing

Here's a simple setup:

```rust
use std::sync::Arc;
use qbice::{
    Engine, DefaultConfig,
    serialize::Plugin,
    stable_hash::{SeededStableHasherBuilder, Sip128Hasher},
    storage::kv_database::rocksdb::RocksDB,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a temporary directory for the database
    let temp_dir = tempfile::tempdir()?;

    // Create the engine
    let mut engine = Engine::<DefaultConfig>::new_with(
        Plugin::default(),
        RocksDB::factory(temp_dir.path()),
        SeededStableHasherBuilder::<Sip128Hasher>::new(0),
    )?;

    // Engine is ready to use!
    Ok(())
}
```

Let's examine each component:

## Plugin - Serialization

```rust
Plugin::default()
```

The plugin handles serialization and deserialization of query keys and values.
For custom serialization needs, you can create your own plugin.

## Database Factory - Persistence

```rust
RocksDB::factory(temp_dir.path())
```

QBICE supports pluggable storage backends. RocksDB is the default and recommended choice for production use. The factory creates a database instance at the specified path.

Available backends:

- **RocksDB** (default) - Production-ready, embedded database
- **fjall** - Alternative key-value store

```rust
// Using RocksDB (requires "rocksdb" feature)
use qbice::storage::kv_database::rocksdb::RocksDB;
let factory = RocksDB::factory("/path/to/db");

// Using fjall (requires "fjall" feature)
use qbice::storage::kv_database::fjall::Fjall;
let factory = Fjall::factory("/path/to/db");
```

## Hasher Builder - Stable Hashing

```rust
SeededStableHasherBuilder::<Sip128Hasher>::new(0)
```

The hasher generates stable 128-bit hashes for queries. The seed (0 in this example) should be consistent across runs for the same project.

**Important**: Use the same seed when reloading a database, or cached results won't match!

## Registering Executors

After creating the engine, register all your executors:

```rust
use std::sync::Arc;

// Create executor instances
let divide_executor = Arc::new(DivideExecutor::new());
let safe_divide_executor = Arc::new(SafeDivideExecutor::new());

// Register with the engine
engine.register_executor(divide_executor.clone());
engine.register_executor(safe_divide_executor.clone());
```

A few important notes:

- Executors must be wrapped in `Arc` for shared ownership
- Each query type needs exactly one executor
- Registering the same query type twice will overwrite the first executor
- You can keep `Arc` clones to access executor state (like call counters)
- There's no executor for `Variable` since it's an input query
- It's expected to register the same executors again after reloading an engine
  from disk

## Creating a TrackedEngine

To actually execute queries and set inputs, first convert the engine to an `Arc`:

```rust
// Move engine into Arc for shared ownership
let engine = Arc::new(engine);
```

The `Arc` (Atomic Reference Count) enables shared ownership of the engine, which is required for both input sessions and query execution.

## Setting Input Values

Before querying, set the initial values for input queries:

```rust
// Create an input session (requires &Arc<Engine>)
{
    let mut input_session = engine.input_session();

    // Set variable values
    input_session.set_input(Variable::A, 42);
    input_session.set_input(Variable::B, 2);

} // Session is committed when dropped
```

The input session is a transaction-like mechanism:

- Changes are batched
- Dirty propagation happens when the session is dropped
- You can set many inputs efficiently

## Executing Queries

Create a `TrackedEngine` to execute queries:

```rust
// Create a tracked engine for querying
let tracked = engine.tracked();

// Now you can execute queries!
let result = tracked.query(&SafeDivide {
    numerator: Variable::A,
    denominator: Variable::B,
}).await;

assert_eq!(result, Some(21));
```

The `TrackedEngine` is a lightweight wrapper that:

- Tracks dependencies during query execution
- Provides the `query()` method
- Can be cloned cheaply

Each TrackedEngine is tied to a specific timestamp of the engine's state when
it's created.

Every time you update inputs, the engine's timestamp advances leaving the old
TrackedEngine stale.

Calling stale tracked engines will return a future that
never resolves, forcing you to drop the future.

## Complete Setup Example

Here's everything together:

```rust
use std::sync::Arc;
use qbice::{
    Engine, DefaultConfig,
    serialize::Plugin,
    stable_hash::{SeededStableHasherBuilder, Sip128Hasher},
    storage::kv_database::rocksdb::RocksDB,
};

// Import our types
use crate::{
    Variable, Divide, SafeDivide,
    VariableExecutor, DivideExecutor, SafeDivideExecutor,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Create the engine
    let temp_dir = tempfile::tempdir()?;
    let mut engine = Engine::<DefaultConfig>::new_with(
        Plugin::default(),
        RocksDB::factory(temp_dir.path()),
        SeededStableHasherBuilder::<Sip128Hasher>::new(0),
    )?;

    // 2. Register executors
    let divide_executor = Arc::new(DivideExecutor::new());
    let safe_divide_executor = Arc::new(SafeDivideExecutor::new());

    engine.register_executor(divide_executor.clone());
    engine.register_executor(safe_divide_executor.clone());

    // 3. Wrap in Arc for shared ownership
    let engine = Arc::new(engine);

    // 4. Set input values
    {
        let mut input_session = engine.input_session();
        input_session.set_input(Variable::A, 42);
        input_session.set_input(Variable::B, 2);
    }

    // 5. Create tracked engine for querying
    let tracked = engine.tracked();

    // 6. Ready to execute queries!
    println!("Setup complete!");

    Ok(())
}
```

## Configuration Options

QBICE supports custom configurations via the `Config` trait. We have provided
a sensible default configuration called `DefaultConfig`.

```rust
use qbice::DefaultConfig;
let mut engine = Engine::<DefaultConfig>::new_with(...)?;
```

For advanced use cases, you can implement your own `Config` type to customize behavior.

## Lifetime Management

Key points about engine lifetime:

- **Engine** - Owns the database and executor registry
- **Arc\<Engine\>** - Shared ownership, can be cloned and passed around
- **TrackedEngine** - Lightweight wrapper, cheap to clone
- **New TrackedEngine** - Create a new one after updating inputs

Typical pattern:

```rust
// The engine is wrapped in Arc, so no mutable access is needed
{
    let tracked = engine.tracked();
    // Use tracked for queries...
} // Drop tracked

// Create a new input session to update inputs
{
    let mut input_session = engine.input_session();
    // Update inputs...
}
```

## Key Takeaways

- The engine requires a plugin, database factory, and hasher builder
- Register executors with `register_executor()`
- Set input values via `input_session()`
- Convert to `Arc<Engine>` and call `tracked()` to execute queries
- `TrackedEngine` has an associated timestamp; create a new one after input
  updates

Next, we'll execute some queries and see incremental computation in action!

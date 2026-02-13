# QBICE - Query-Based Incremental Computation Engine

[![Crates.io](https://img.shields.io/crates/v/qbice.svg)](https://crates.io/crates/qbice)
[![Documentation](https://docs.rs/qbice/badge.svg)](https://docs.rs/qbice)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Simmypeet/qbice/blob/master/LICENSE)

QBICE is a high-performance, asynchronous incremental computation framework for Rust. Define your computation as a graph of queries, and QBICE automatically determines what needs to be recomputed when inputs change—minimizing redundant work through intelligent caching, dependency tracking, and advanced optimization techniques.

## Features

- **Incremental Computation** — Only recomputes what's necessary when inputs change
- **Async-First Design** — Built on Tokio for efficient concurrent execution
- **Cycle Detection** — Automatically detects and handles cyclic dependencies
- **Type-Safe** — Strongly-typed queries with associated value types
- **Thread-Safe** — Safely share the engine across multiple threads
- **Persistent Storage** — Pluggable key-value database backends (RocksDB, fjall) for caching query results
- **Visualization** — Generate interactive HTML dependency graphs to analyze computation structure

### Feature Flags

- `default` - Includes RocksDB backend
- `rocksdb` - Enable RocksDB storage backend
- `fjall` - Enable fjall storage backend

## Quick Start

Here's a simple example demonstrating safe division with incremental recomputation:

```rust
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use qbice::{
    Config, Decode, DefaultConfig, Encode, Engine, Executor,
    Identifiable, Query, StableHash, TrackedEngine,
    serialize::Plugin,
    stable_hash::{SeededStableHasherBuilder, Sip128Hasher},
    storage::{
        kv_database::rocksdb::RocksDB,
        storage_engine::db_backed::{Configuration, DbBacked, DbBackedFactory},
    },
};

// Define query types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, StableHash, Identifiable, Encode, Decode)]
pub enum Variable {
    A,
    B,
}

impl Query for Variable {
    type Value = i32;
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable, Encode, Decode)]
pub struct SafeDivide {
    pub numerator: Variable,
    pub denominator: Variable,
}

impl Query for SafeDivide {
    type Value = Option<i32>;
}

// Define executor
struct SafeDivideExecutor;

impl<C: Config> Executor<SafeDivide, C> for SafeDivideExecutor {
    async fn execute(
        &self,
        query: &SafeDivide,
        engine: &TrackedEngine<C>,
    ) -> Option<i32> {
        let num = engine.query(&query.numerator).await;
        let denom = engine.query(&query.denominator).await;

        if denom == 0 {
            return None;
        }

        Some(num / denom)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;

    // Create and configure the engine
    let mut engine = Engine::<DefaultConfig>::new_with(
        Plugin::default(),
        DbBackedFactory::builder()
            .configuration(Configuration::builder().build())
            .db_factory(RocksDB::factory(tempdir.path()))
            .build(),
        SeededStableHasherBuilder::<Sip128Hasher>::new(0),
    ).await?;

    // Register executor
    engine.register_executor(Arc::new(SafeDivideExecutor));

    // Set initial inputs
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable::A, 42).await;
        input_session.set_input(Variable::B, 2).await;
        input_session.commit().await;
    }

    // Execute query
    let tracked_engine = Arc::new(engine).tracked().await;
    let result = tracked_engine.query(&SafeDivide {
        numerator: Variable::A,
        denominator: Variable::B,
    }).await;

    assert_eq!(result, Some(21));

    Ok(())
}
```

## Core Concepts

### Queries

A **query** represents a unit of computation with an associated input (the query key) and output (the query value). Queries implement the `Query` trait and are identified by their type and a stable hash of their contents.

Required traits for queries:

- `StableHash` - For consistent hashing across program runs
- `Identifiable` - For stable type identification
- `Eq` + `Hash` - For use in hash maps
- `Clone` - For storing query keys
- `Debug` - For error messages and debugging
- `Send` + `Sync` - For thread-safe access

### Executors

An **executor** defines how to compute the value for a specific query type. Executors implement the `Executor` trait and can depend on other queries through the `TrackedEngine`.

### Engine

The `Engine` is the central database that stores computed values and manages the dependency graph. It tracks which queries depend on which other queries and handles cache invalidation when inputs change.

### Tracked Engine

The `TrackedEngine` is a wrapper around `Engine` that tracks dependencies during query execution. Use it to execute queries and build the dependency graph automatically.

## Advanced Features

### Firewall Queries

In large dependency graphs, a single input change can cause excessive dirty propagation through "chokepoint" queries that have many dependents. **Firewall queries** prevent dirty propagation from crossing them, limiting the scope of
dirty propagation.

A firewall query only propagates dirtiness to its dependents if its computed value actually changes. This is particularly useful for global analysis queries that produce large results. The firewall query also works best with projection queries
to minimize unnecessary invalidations.

```rust
impl Query for GlobalAnalysis {
    type Value = HashMap<String, Type>;

    fn execution_style() -> ExecutionStyle {
        ExecutionStyle::Firewall
    }
}
```

### Projection Queries

**Projection queries** work with firewall queries to provide fine-grained change detection. They read a small part of a firewall query's output and are very fast to compute (e.g., hash map lookup). When a firewall query changes, projection queries are re-executed to check if their specific slice of data changed, preventing unnecessary invalidation of downstream queries.

```rust
impl Query for GetFunctionType {
    type Value = Option<Type>;

    fn execution_style() -> ExecutionStyle {
        ExecutionStyle::Projection
    }
}
```

### Timestamp-Based Verification

QBICE uses a monotonic timestamp system to ensure each query is verified at most once per mutation. When inputs change, the engine's timestamp increments. During repairation, queries compare their last verification timestamp with the current timestamp to avoid redundant checks.

## Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                        Engine<C>                                │
│  ┌─────────────────────┐    ┌─────────────────────────────┐     │
│  │   Query Database    │    │    Executor Registry        │     │
│  │  - Cached values    │    │  - Query type → Executor    │     │
│  │  - Dependencies     │    └─────────────────────────────┘     │
│  │  - Dirty flags      │                                        │
│  │  - Fingerprints     │                                        │
│  └─────────────────────┘                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Arc::new(engine).tracked()
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TrackedEngine<C>                             │
│  - Reference to Engine                                          │
│  - Local query cache                                            │
│  - Caller tracking for dependencies                             │
└─────────────────────────────────────────────────────────────────┘
```

### Computation Process

1. **Dirty Propagation** (Eager) - When inputs change, dirty flags propagate upward through backward edges in the dependency graph
2. **Repairation** (Lazy) - When a query is requested, it checks and repairs its dirty dependencies before determining if it needs recomputation
3. **Fingerprinting** - Each query's result is hashed; recomputation only occurs if dependency fingerprints differ
4. **Cycle Detection** - Automatically detects cyclic dependencies and handles them appropriately

## Persistence

QBICE supports persistent storage of query results across program restarts. Computed values and metadata are stored in a pluggable key-value database:

- **RocksDB** (default) - Production-ready embedded database
- **fjall** - Alternative storage backend

```rust
// Results persist across program runs
let engine1 = Engine::<DefaultConfig>::new_with(...).await?;
// ... compute and store results ...

// Later, in a new process
let engine2 = Engine::<DefaultConfig>::new_with(...)?;
// Previous results are loaded automatically
```

## Documentation

For detailed documentation, examples, and API reference:

- **[docs.rs/qbice](https://docs.rs/qbice)** — Full API documentation with examples
- **[GitHub Repository](https://github.com/Simmypeet/qbice)** — Source code and issues
- **[Crates.io](https://crates.io/crates/qbice)** — Package information
- **[Tutorials and Guides](https://simmypeet.github.io/qbice/)** — Step-by-step tutorials and advanced topics

## Performance Considerations

QBICE is designed for scenarios where:

- Computations are expensive relative to cache lookup costs
- Input changes affect only a subset of the computation graph
- You want to avoid recomputing unchanged results

Best practices:

- Use cheap cloning types for query keys and values (e.g., `Arc`, small Copy types)
- Apply firewall queries at strategic points in dense graphs
- Use projection queries to extract specific data from large results
- If perfomance becomes an issue, try using our visualization tools to analyze the dependency graph

## Inspiration

QBICE is inspired by:

- [Salsa](https://github.com/salsa-rs/salsa) — A generic framework for on-demand, incrementalized computation
- [Adapton](https://github.com/Adapton/adapton.rust) — A library for incremental computing

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Simmypeet/qbice/blob/master/LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

Created by [Simmypeet](https://github.com/Simmypeet)

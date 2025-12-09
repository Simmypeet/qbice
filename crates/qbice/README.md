# QBICE

**Query-Based Incremental Computation Engine**

[![Crates.io](https://img.shields.io/crates/v/qbice.svg)](https://crates.io/crates/qbice)
[![Documentation](https://docs.rs/qbice/badge.svg)](https://docs.rs/qbice)
[![License](https://img.shields.io/crates/l/qbice.svg)](LICENSE)

QBICE is a high-performance, asynchronous incremental computation framework for Rust. Define your computation as a graph of queries, and QBICE automatically determines what needs to be recomputed when inputs change—minimizing redundant work through intelligent caching and dependency tracking.

## Features

- 🚀 **Incremental Computation** — Only recomputes what's necessary when inputs change
- ⚡ **Async-First Design** — Built on Tokio for efficient concurrent execution
- 🔄 **Cycle Detection** — Automatically detects and handles cyclic dependencies
- 🔒 **Type-Safe** — Strongly-typed queries with associated value types
- 🧵 **Thread-Safe** — Safely share the engine across multiple threads
- 📊 **Visualization** — Generate interactive HTML dependency graphs

## Quick Start

```rust
use std::sync::Arc;
use qbice::{
    Identifiable, StableHash,
    config::DefaultConfig,
    engine::{Engine, TrackedEngine},
    executor::{CyclicError, Executor},
    query::Query,
};

// Define an input query
#[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable)]
struct Variable(u64);

impl Query for Variable {
    type Value = i64;
}

// Define a computation query
#[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable)]
struct Sum {
    a: Variable,
    b: Variable,
}

impl Query for Sum {
    type Value = i64;
}

// Define the executor
struct SumExecutor;

impl<C: qbice::config::Config> Executor<Sum, C> for SumExecutor {
    async fn execute(
        &self,
        query: &Sum,
        engine: &TrackedEngine<C>,
    ) -> Result<i64, CyclicError> {
        let a = engine.query(&query.a).await?;
        let b = engine.query(&query.b).await?;
        Ok(a + b)
    }
}

#[tokio::main]
async fn main() {
    // Create and configure the engine
    let mut engine = Engine::<DefaultConfig>::new();
    engine.register_executor::<Sum, _>(Arc::new(SumExecutor));

    // Set input values
    {
        let mut session = engine.input_session();
        session.set_input(Variable(0), 10);
        session.set_input(Variable(1), 20);
    }

    // Query the engine
    let engine = Arc::new(engine);
    let tracked = engine.clone().tracked();
    let result = tracked.query(&Sum {
        a: Variable(0),
        b: Variable(1),
    }).await;

    assert_eq!(result, Ok(30));
}
```

## Core Concepts

### Queries

A **query** represents a unit of computation with an input key and an output value. Queries are identified by their type and content hash, ensuring stable identification across program runs.

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable)]
struct FileContents {
    path: Arc<str>,  // Use Arc for cheap cloning
}

impl Query for FileContents {
    type Value = Arc<[u8]>;  // Use Arc for cheap cloning
}
```

> **Performance Tip:** Both query types and their values are cloned frequently internally. Use `Arc<T>`, `Arc<str>`, or `Arc<[T]>` for heap-allocated data to ensure O(1) cloning.

### Executors

**Executors** define how to compute values for queries. They must behave as **pure functions**—given the same inputs, they must always produce the same output.

```rust
// A division query that computes dividend / divisor
#[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable)]
struct Division {
    dividend: Variable,
    divisor: Variable,
}

impl Query for Division {
    type Value = i64;
}

struct DivisionExecutor;

impl<C: Config> Executor<Division, C> for DivisionExecutor {
    async fn execute(
        &self,
        query: &Division,
        engine: &TrackedEngine<C>,
    ) -> Result<i64, CyclicError> {
        let dividend = engine.query(&query.dividend).await?;
        let divisor = engine.query(&query.divisor).await?;
        Ok(dividend / divisor)
    }
}

// A safe division that returns None for division by zero.
// Demonstrates queries depending on other queries.
#[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable)]
struct SafeDivision {
    dividend: Variable,
    divisor: Variable,
}

impl Query for SafeDivision {
    type Value = Option<i64>;
}

struct SafeDivisionExecutor;

impl<C: Config> Executor<SafeDivision, C> for SafeDivisionExecutor {
    async fn execute(
        &self,
        query: &SafeDivision,
        engine: &TrackedEngine<C>,
    ) -> Result<Option<i64>, CyclicError> {
        let divisor = engine.query(&query.divisor).await?;

        if divisor == 0 {
            Ok(None)  // Avoid division by zero
        } else {
            // Delegate to the Division query
            let result = engine.query(&Division {
                dividend: query.dividend,
                divisor: query.divisor,
            }).await?;
            Ok(Some(result))
        }
    }
}
```

> **Important:** Executors should not read from global mutable state, system time, or external sources without modeling them as query dependencies. This ensures correct incremental behavior.

### Engine Lifecycle

```rust
// 1. Create and configure
let mut engine = Engine::<DefaultConfig>::new();
engine.register_executor::<MyQuery, _>(Arc::new(MyExecutor));

// 2. Set inputs
{
    let mut session = engine.input_session();
    session.set_input(InputQuery(0), value);
}

// 3. Wrap in Arc and query
let engine = Arc::new(engine);
let tracked = engine.clone().tracked();
let result = tracked.query(&MyQuery { ... }).await?;

// 4. Update inputs (requires dropping TrackedEngine first)
drop(tracked);
{
    let engine_mut = Arc::get_mut(&mut engine).unwrap();
    let mut session = engine_mut.input_session();
    session.set_input(InputQuery(0), new_value);
}

// 5. Query again - only affected computations rerun
let tracked = engine.tracked();
let result = tracked.query(&MyQuery { ... }).await?;
```

## Incremental Updates

QBICE tracks dependencies automatically. When you update inputs, only queries that depend on changed values are recomputed:

```rust
// Initial computation
let tracked = engine.clone().tracked();
assert_eq!(tracked.query(&Sum { a: Variable(0), b: Variable(1) }).await, Ok(300));
drop(tracked);

// Update one input
{
    let engine_mut = Arc::get_mut(&mut engine).unwrap();
    let mut session = engine_mut.input_session();
    session.set_input(Variable(0), 150);  // Changed!
}

// Only Sum is recomputed, not unrelated queries
let tracked = engine.tracked();
assert_eq!(tracked.query(&Sum { a: Variable(0), b: Variable(1) }).await, Ok(350));
```

## Handling Cycles

QBICE detects cyclic dependencies and returns `CyclicError`. For intentional cycles (e.g., fixed-point computations), implement `scc_value`:

```rust
impl<C: Config> Executor<Reachable, C> for ReachableExecutor {
    async fn execute(&self, query: &Reachable, engine: &TrackedEngine<C>) -> Result<bool, CyclicError> {
        if query.from == query.to {
            return Ok(true);
        }
        // Check reachability through neighbors...
        Ok(false)
    }

    fn scc_value() -> bool {
        false  // Default value when cycle detected
    }
}
```

## Visualization

Generate interactive HTML visualizations of your dependency graph:

```rust
engine.visualize_html(&my_query, "dependency_graph.html")?;
```

This creates an interactive page with:

- Pan and zoom navigation
- Click-to-inspect node details
- Search and filtering
- Color-coded dependency edges

## Execution Styles

For advanced optimization, queries can have different execution styles:

- **Normal** — Standard dependency tracking (default)
- **Projection** — Fast extractors for parts of other queries
- **Firewall** — Boundaries that limit dirty propagation

```rust
impl<C: Config> Executor<MyQuery, C> for MyExecutor {
    async fn execute(&self, ...) -> Result<T, CyclicError> { ... }

    fn execution_style() -> ExecutionStyle {
        ExecutionStyle::Firewall  // Limits dirty propagation
    }
}
```

## Requirements

- Rust 1.88.0 or later (Edition 2024)
- Tokio runtime

## License

This project is licensed under [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

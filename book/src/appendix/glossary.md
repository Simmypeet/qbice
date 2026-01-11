# Glossary

## Core Concepts

### Engine

The central database that manages query execution, result caching, and dependency tracking. The `Engine` stores computed values, tracks dependencies between queries, and coordinates executor execution.

### TrackedEngine

A lightweight wrapper around the `Engine` that tracks dependencies during query execution. Created from `Arc<Engine>` via the `tracked()` method. Provides the `query()` method for executing queries.

### Query

A type representing a computation's input (key) and output (value). Implements the `Query` trait. Each unique query instance represents a distinct computation. The query itself only defines _what_ to compute, not _how_.

### Executor

Implements the `Executor` trait to define _how_ to compute a value for a specific query type. Executors contain the actual computation logic and can depend on other queries through the `TrackedEngine`.

### Query Key

The query instance itself, which serves as the input to the computation. Different field values create different query keys, representing different computations.

### Query Value

The output of a query computation. Defined by the `Value` associated type in the `Query` trait implementation.

### Query ID

A unique identifier for each query instance, combining a stable type identifier and a 128-bit hash of the query's contents.

## Dependency Tracking

### Dependency Graph

The directed graph of query dependencies, where nodes are queries and edges represent "depends on" relationships. QBICE automatically builds this graph during query execution.

### Forward Edge

An edge from a query to its dependencies (callees). Stores the last seen fingerprint of the dependency for change detection.

### Backward Edge

An edge from a query to its dependents (callers). Used for dirty propagation.

### Dirty Flag

A boolean flag indicating whether a query's result may be invalid due to changes in its dependencies. Queries marked dirty need verification before their cached result can be used.

### Dirty Propagation

The process of marking queries as dirty when their dependencies change. Propagates upward through backward edges from changed inputs to dependent queries.

### Fingerprint

A 128-bit hash of a query's result, used for change detection. When a dependency is recomputed, its fingerprint is compared to the previously seen fingerprint to determine if the dependent needs recomputation.

### Verification Timestamp

A timestamp indicating when a query was last verified to be up-to-date. Used to avoid redundant verification checks.

## Execution Styles

### Normal Query

The standard execution style with full dependency tracking. Most queries should use this style.

### External Input

Marks a query as an input value set via input sessions. These queries should never be executed—their values are provided directly.

### Firewall Query

An advanced optimization that prevents dirty propagation from crossing the query boundary. Only propagates dirtiness to dependents if the firewall's computed value actually changes. Used to limit excessive dirty propagation at chokepoints.

### Projection Query

A fast query that extracts a specific slice of data from a firewall or projection query result. Must be very fast to execute (e.g., hash lookup, field access). Provides fine-grained change detection for large firewall results.

### Transitive Firewall Edge

A special dependency edge that allows queries to directly check firewall queries they transitively depend on. Ensures correctness when firewall queries block normal dirty propagation.

## Execution Flow

### Repairation

The lazy process of verifying and recomputing queries when they're requested. Checks dependencies, compares fingerprints, and re-executes only if necessary.

### Incremental Computation

The process of recomputing only what's necessary when inputs change, by tracking dependencies and using cached results for unchanged queries.

### Cache Hit

When a query's cached result can be returned without re-execution because nothing has changed.

### Cache Miss

When a query must be re-executed because it's dirty or has never been computed.

### Cycle

A situation where a query depends on itself, either directly or transitively. QBICE automatically detects cycles and reports an error.

## Storage

### Persistence

The automatic saving of query results to disk, allowing computation state to survive across program restarts.

### Database Backend

The pluggable key-value store used for persisting query results. QBICE supports RocksDB and fjall.

### Serialization

The process of converting query keys and values to bytes for storage. Implemented via the `Encode` and `Decode` traits.

### Stable Hash

A hash function that produces consistent results across program runs. Required for queries to ensure persistence works correctly.

### Hasher Seed

A value used to initialize the stable hasher. Must be consistent across runs for cached results to match.

## Configuration

### Config Trait

A trait for customizing QBICE's behavior. Most users should use `DefaultConfig`.

### Plugin

Handles serialization and deserialization of query keys and values. The default plugin works for types implementing `Encode` and `Decode`.

### Input Session

A transaction-like mechanism for setting input query values. Changes are batched and committed when the session is dropped, triggering dirty propagation.

## Performance

### Chokepoint

A query with many dependents that acts as a bottleneck for dirty propagation. Good candidates for firewall queries.

### Over-Invalidation

When queries are marked dirty even though their specific dependencies haven't changed. Solved by projection queries.

### Local Cache

A cache maintained by each `TrackedEngine` instance for the current execution context. Ensures repeated queries within the same context return instantly.

### Global Cache

The engine's persistent cache of computed query results, shared across all `TrackedEngine` instances.

## Related Concepts

### Memoization

Caching function results based on their inputs. QBICE provides automatic, incremental memoization with dependency tracking.

### Dataflow Programming

A programming paradigm where programs are modeled as directed graphs of data flowing between operations. QBICE enables dataflow-style programming with automatic incremental updates.

### Salsa

An inspiration for QBICE. A Rust framework for on-demand, incrementalized computation.

### Adapton

Another inspiration for QBICE. A library for incremental computing with explicit dependency tracking.

## See Also

- [Introduction](../introduction.md) - Overview of QBICE
- [Getting Started](../tutorial/getting-started.md) - Tutorial introduction
- [Engine](../reference/engine.md) - Engine reference
- [Query](../reference/query.md) - Query trait reference

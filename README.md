# QBICE

**Query-Based Incremental Computation Engine**

[![Crates.io](https://img.shields.io/crates/v/qbice.svg)](https://crates.io/crates/qbice)
[![Documentation](https://docs.rs/qbice/badge.svg)](https://docs.rs/qbice)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

QBICE is an asynchronous incremental computation framework for Rust. Define your computation as a graph of queries, and QBICE automatically determines what needs to be recomputed when inputs change—minimizing redundant work through intelligent caching and dependency tracking.

## Documentation

For detailed documentation, examples, and API reference, please visit:

- **[docs.rs/qbice](https://docs.rs/qbice)** — Full API documentation with examples
- **[crates.io/crates/qbice](https://crates.io/crates/qbice)** — Package information
- **[Tutorials and Guides](https://simmypeet.github.io/qbice/)** — Step-by-step tutorials and advanced topics

## Features

- **Incremental Computation** — Only recomputes what's necessary when inputs change
- **Async-First Design** — Built on Tokio for efficient concurrent execution
- **Cycle Detection** — Automatically detects and handles cyclic dependencies
- **Type-Safe** — Strongly-typed queries with associated value types
- **Thread-Safe** — Safely share the engine across multiple threads
- **Visualization** — Generate interactive HTML dependency graphs

## Installation

Add QBICE to your `Cargo.toml`:

```toml
[dependencies]
qbice = "0.1.0"
tokio = { version = "1", features = ["full"] }
```

## Requirements

- Rust 1.88.0 or later (Edition 2024)
- Tokio runtime for async execution

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

QBICE is inspired by:

- [Salsa](https://github.com/salsa-rs/salsa) — A generic framework for on-demand, incrementalized computation
- [Adapton](https://github.com/Adapton/adapton.rust) — A library for incremental computing

# Introduction

Welcome to the QBICE (Query-Based Incremental Computation Engine) book! This guide will teach you how to build efficient incremental computation systems using QBICE.

## What is QBICE?

QBICE is a high-performance, asynchronous incremental computation framework for Rust. It allows you to define computations as a graph of queries, where changes to inputs automatically propagate through the system—recomputing only what's necessary.

## Why Incremental Computation?

Consider a compiler that needs to type-check a large codebase. When a developer changes a single file, recompiling everything from scratch would be wasteful. Instead, an incremental system:

1. **Tracks dependencies** - Knows which computations depend on the changed file
2. **Invalidates selectively** - Marks only affected computations as needing updates
3. **Recomputes minimally** - Recalculates only what's necessary

This approach can reduce compilation times from minutes to seconds.

## Key Benefits

- **Automatic Dependency Tracking** - QBICE automatically records which queries depend on which others
- **Minimal Recomputation** - Only recomputes queries whose inputs have changed
- **Async-First** - Built on Tokio for efficient concurrent execution
- **Persistent** - Query results survive program restarts

## Use Cases

QBICE is ideal for scenarios where:

- Computations are expensive relative to cache lookups
- Inputs change incrementally
- You want to avoid redundant recomputation
- You need fine-grained dependency tracking

Common applications include:

- **Compilers** - Incremental compilation and analysis
- **Build Systems** - Smart rebuilding of artifacts
- **IDEs** - Real-time code analysis and diagnostics

## Prior Art

QBICE builds on decades of research and practical work in incremental computation. Understanding these foundational systems helps appreciate QBICE's design choices.

### Adapton

[Adapton](http://adapton.org/) is a pioneering research that introduced many core concepts used in QBICE:

- **Demanded Computation Graphs (DCG)** - The idea that computations form a graph where dependencies are tracked automatically
- **Dirty Propagation** - Marking dependency edges as dirty when inputs change

The SafeDivide example in the tutorial is adapted from Adapton's classic examples, demonstrating how incremental computation handles error cases elegantly.

### Salsa

[Salsa](https://github.com/salsa-rs/salsa) is a Rust framework for incremental computation used in rust-analyzer. It provides:

- Query-based incremental computation
- Automatic dependency tracking

QBICE shares similar goals but differs in:

- **Fine-Grained Invalidation**: Salsa uses timestamp based to determine whether
  the query needs to be reverified, while QBICE uses fine-grained dirty
  propagation through dependency edges.
- **Optimizations**: Salsa achieves optimized graph traversal through the
  conceopt of durability, while QBICE provides firewall and projection queries
  to optimize dirty propagation and recomputation.

## How This Book Is Organized

This book is divided into three main sections:

### Tutorial

A hands-on guide that walks you through building your first QBICE application. You'll learn the fundamentals by creating a simple calculator that performs incremental computation.

### Advanced Topics

Deep dives into optimization techniques like firewall queries, projection queries, and performance tuning strategies.

## Prerequisites

This book assumes you're familiar with:

- **Rust** - Basic to intermediate knowledge
- **Async/Await** - Understanding of async Rust and Tokio
- **Traits** - How Rust traits work

If you're new to these topics, we recommend:

- [The Rust Book](https://doc.rust-lang.org/book/)
- [Async Book](https://rust-lang.github.io/async-book/)

## Getting Help

- **API Documentation** - [docs.rs/qbice](https://docs.rs/qbice)
- **GitHub** - [github.com/Simmypeet/qbice](https://github.com/Simmypeet/qbice)
- **Issues** - Report bugs or request features on GitHub

Let's get started!

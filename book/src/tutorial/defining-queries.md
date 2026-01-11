# Defining Queries

A **query** in QBICE represents a computation with an input (the query key) and an output (the query value). Let's define the queries for our calculator.

## Required Traits

Every query type must implement several traits:

- `Query` - The main trait defining the output type
- `StableHash` - For consistent hashing across runs
- `Identifiable` - For stable type identification
- `Encode` / `Decode` - For persistence
- `Debug`, `Clone`, `PartialEq`, `Eq`, `Hash` - Standard Rust traits

Fortunately, most of these can be derived automatically!

## Variable Query

First, let's define a query for variables. Variables are inputs to our system—they don't compute anything, they just hold values. We'll use an enum for simplicity:

```rust
use qbice::{
    Decode, Encode, Identifiable, Query, StableHash,
};

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    StableHash,
    Identifiable,
    Encode,
    Decode,
)]
pub enum Variable {
    A,
    B,
}

impl Query for Variable {
    type Value = i32;
}
```

Let's break this down:

### The Enum

```rust
pub enum Variable {
    A,
    B,
}
```

The enum itself is the query **key**. Different variants represent different queries. `Variable::A` and `Variable::B` are distinct queries. Using an enum is simpler than using strings for our example.

### The Derives

```rust
#[derive(
    Debug,              // For error messages
    Clone,              // Queries must be cloneable
    Copy,               // Cheap to copy (for enums)
    PartialEq, Eq,      // For comparing queries
    PartialOrd, Ord,    // For ordering (useful for sorted collections)
    Hash,               // For hash maps
    StableHash,         // For consistent hashing
    Identifiable,       // For type identification
    Encode, Decode,     // For persistence
)]
```

These derived traits enable QBICE to:

- Store queries in hash maps
- Generate stable identifiers
- Persist results to disk
- Display debug information

### The Query Trait

```rust
impl Query for Variable {
    type Value = i32;
}
```

The `Value` associated type defines what this query produces. Variables produce `i32` values.

## Divide Query

Now let's define a query that divides two variables:

```rust
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    StableHash,
    Identifiable,
    Encode,
    Decode,
)]
pub struct Divide {
    pub numerator: Variable,
    pub denominator: Variable,
}

impl Query for Divide {
    type Value = i32;
}
```

The `Divide` query takes two variables and produces their quotient. The key insight: `Divide` doesn't actually perform the division—that's the executor's job. The query just describes **what** to compute.

Note: This version will panic if the denominator is zero (we'll handle that with `SafeDivide`).

## SafeDivide Query

Now for the safe version that handles division by zero:

```rust
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    StableHash,
    Identifiable,
    Encode,
    Decode,
)]
pub struct SafeDivide {
    pub numerator: Variable,
    pub denominator: Variable,
}

impl Query for SafeDivide {
    type Value = Option<i32>;  // Returns None for division by zero!
}
```

Notice that `SafeDivide` returns `Option<i32>` instead of `i32`. This allows us to return `None` when dividing by zero, making our computation safe and preventing panics.

## Why Separate Queries from Execution?

You might wonder: why not just put the computation logic in the query itself?

The separation provides several benefits:

1. **Decoupling**: This might sound cliche, but separating the _what_ (query)
   from the _how_ (executor) can sometime be beneficial. For example, in a large
   scale query system if you want to do unit testing, you can mock out executors
   without changing the query definitions.
2. **External Effects**: Sometimes you can perform some side-effects in the executor
   like logging, metrics, etc. NOTE: Be careful with side-effects in executors, some
   side-effects that doesn't influence the output value are usually okay, but anything that influences the output value will definitely break the semantics of incremental computation.

## Complete Code

Here's our complete query module:

```rust
use qbice::{
    Decode, Encode, Identifiable, Query, StableHash,
};

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    StableHash,
    Identifiable,
    Encode,
    Decode,
)]
pub enum Variable {
    A,
    B,
}

impl Query for Variable {
    type Value = i32;
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    StableHash,
    Identifiable,
    Encode,
    Decode,
)]
pub struct Divide {
    pub numerator: Variable,
    pub denominator: Variable,
}

impl Query for Divide {
    type Value = i32;
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    StableHash,
    Identifiable,
    Encode,
    Decode,
)]
pub struct SafeDivide {
    pub numerator: Variable,
    pub denominator: Variable,
}

impl Query for SafeDivide {
    type Value = Option<i32>;
}
```

## Key Takeaways

- Queries define **what** to compute, not **how**
- Query types are the **keys**; the associated `Value` type is the **result**
- Most required traits can be derived automatically
- Each unique query instance (different field values) represents a distinct computation

Next, we'll implement the executors that actually perform these computations!

# Implementing Executors

Now that we've defined our queries, let's implement the **executors** that actually perform the computations. Executors implement the `Executor` trait, which defines how to compute a value for a specific query type.

## The Executor Trait

The `Executor` trait has one required method:

```rust
pub trait Executor<Q: Query, C: Config>: Send + Sync {
    async fn execute(
        &self,
        query: &Q,
        engine: &TrackedEngine<C>,
    ) -> Q::Value;
}
```

Let's break this down:

- `&self` - The executor instance (you can store state here)
- `query: &Q` - The specific query being executed
- `engine: &TrackedEngine<C>` - Used to query other dependencies
- Returns `Q::Value` - The computed result

## Divide Executor

Let's start with the `Divide` executor:

```rust
pub struct DivideExecutor;

impl<C: Config> Executor<Divide, C> for DivideExecutor {
    async fn execute(
        &self,
        query: &Divide,
        engine: &TrackedEngine<C>,
    ) -> i32 {
        // Query the numerator
        let num = engine.query(&query.numerator).await;

        // Query the denominator
        let denom = engine.query(&query.denominator).await;

        // Assert denominator is not zero
        assert!(denom != 0, "denominator should not be zero");

        // Return the quotient
        num / denom
    }
}
```

This is where QBICE's magic happens! Notice:

1. **We query other queries** - `engine.query()` executes dependencies
2. **Dependencies are tracked automatically** - QBICE records that `Divide` depends on two `Variable` queries
3. **It's async** - We can await other queries without blocking
4. **We assert safety** - This version panics if denominator is zero

## SafeDivide Executor

Now for the safe version that handles division by zero:

```rust
pub struct SafeDivideExecutor;

impl<C: Config> Executor<SafeDivide, C> for SafeDivideExecutor {
    async fn execute(
        &self,
        query: &SafeDivide,
        engine: &TrackedEngine<C>,
    ) -> Option<i32> {
        // Query the denominator first
        let denom = engine.query(&query.denominator).await;

        // Check for division by zero
        if denom == 0 {
            return None;
        }

        // Safe to divide - delegate to Divide query
        Some(
            engine.query(&Divide {
                numerator: query.numerator,
                denominator: query.denominator,
            }).await
        )
    }
}
```

This executor demonstrates an important pattern:

- **Early return** - We check for division by zero first
- **Query composition** - SafeDivide depends on Divide
- **Error handling** - Returns `None` instead of panicking

## Adding State to Executors

Executors can maintain state. This is useful for tracking metrics,
like call counts. Maintaining state is ok as long as it doesn't affect
the correctness of computations.

```rust
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct DivideExecutor {
    call_count: AtomicUsize,
}

impl DivideExecutor {
    pub fn new() -> Self {
        Self {
            call_count: AtomicUsize::new(0),
        }
    }

    pub fn call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

impl<C: Config> Executor<Divide, C> for DivideExecutor {
    async fn execute(
        &self,
        query: &Divide,
        engine: &TrackedEngine<C>,
    ) -> i32 {
        // Increment counter
        self.call_count.fetch_add(1, Ordering::SeqCst);

        let num = engine.query(&query.numerator).await;
        let denom = engine.query(&query.denominator).await;

        assert!(denom != 0, "denominator should not be zero");

        num / denom
    }
}
```

Now we can verify that QBICE is actually performing incremental computation by checking how many times each executor was called!

## Complete Code

Here's our complete executor module:

```rust
use std::sync::atomic::{AtomicUsize, Ordering};
use qbice::{Config, Executor, TrackedEngine};

// Import our query types
use crate::{Divide, SafeDivide};

pub struct DivideExecutor {
    pub call_count: AtomicUsize,
}

impl DivideExecutor {
    pub fn new() -> Self {
        Self {
            call_count: AtomicUsize::new(0),
        }
    }
}

impl<C: Config> Executor<Divide, C> for DivideExecutor {
    async fn execute(
        &self,
        query: &Divide,
        engine: &TrackedEngine<C>,
    ) -> i32 {
        self.call_count.fetch_add(1, Ordering::SeqCst);

        let num = engine.query(&query.numerator).await;
        let denom = engine.query(&query.denominator).await;

        assert!(denom != 0, "denominator should not be zero");

        num / denom
    }
}

pub struct SafeDivideExecutor {
    pub call_count: AtomicUsize,
}

impl SafeDivideExecutor {
    pub fn new() -> Self {
        Self {
            call_count: AtomicUsize::new(0),
        }
    }
}

impl<C: Config> Executor<SafeDivide, C> for SafeDivideExecutor {
    async fn execute(
        &self,
        query: &SafeDivide,
        engine: &TrackedEngine<C>,
    ) -> Option<i32> {
        self.call_count.fetch_add(1, Ordering::SeqCst);

        let denom = engine.query(&query.denominator).await;
        if denom == 0 {
            return None;
        }

        Some(
            engine.query(&Divide {
                numerator: query.numerator,
                denominator: query.denominator,
            }).await
        )
    }
}
```

## Key Takeaways

- Executors define **how** to compute query results
- Use `engine.query()` to depend on other queries
- Dependencies are tracked automatically by QBICE
- Executors can maintain state (like metrics) if it doesn't affect correctness

Next, we'll set up the engine and register these executors!

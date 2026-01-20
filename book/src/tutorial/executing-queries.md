# Executing Queries

Now that we've set up the engine, let's execute some queries and see QBICE's incremental computation in action!

## Basic Query Execution

Executing a query is straightforward—call `query()` on a `TrackedEngine`:

```rust
let tracked = engine.tracked();

let result = tracked.query(&SafeDivide {
    numerator: Variable::A,
    denominator: Variable::B,
}).await;

println!("SafeDivide(A, B) = {:?}", result); // Output: SafeDivide(A, B) = Some(21)
```

Remember, we set `Variable::A = 42` and `Variable::B = 2` in our input session, so `42 / 2 = 21`.

## Verifying Incremental Computation

Let's prove QBICE is actually caching results. Remember we added call counters to our executors:

```rust
// Execute some queries
let tracked = engine.tracked();

tracked.query(&Divide {
    numerator: Variable::A,
    denominator: Variable::B,
}).await;

tracked.query(&SafeDivide {
    numerator: Variable::A,
    denominator: Variable::B,
}).await;

// Check call counts
println!("Divide called: {} times", divide_executor.call_count.load(Ordering::SeqCst));
println!("SafeDivide called: {} times", safe_divide_executor.call_count.load(Ordering::SeqCst));
```

Expected output:

```
Divide called: 1 times
SafeDivide called: 1 times
```

The counts didn't increase! QBICE returned cached results because nothing changed.

## Querying with Different Keys

Each unique query key is tracked separately:

```rust
let tracked = engine.tracked();

// These are different queries (different keys)
let ab = tracked.query(&Divide {
    numerator: Variable::A,
    denominator: Variable::B,
}).await;

let aa = tracked.query(&Divide {
    numerator: Variable::A,
    denominator: Variable::A,
}).await;

println!("Divide(A, B) = {}", ab); // 21
println!("Divide(A, A) = {}", aa); // 1

// Both queries were executed (check the call count)
println!("Divide called: {} times", divide_executor.call_count.load(Ordering::SeqCst));
```

Expected output:

```
Divide called: 2 times
```

QBICE distinguishes between `Divide { numerator: A, denominator: A }` and `Divide { numerator: A, denominator: B }` because they have different keys.

## Async Concurrent Execution

Since queries are async, you can execute multiple independent queries concurrently:

```rust
use tokio::join;

let tracked = engine.tracked();

// Execute multiple queries in parallel
let (result1, result2) = join!(
    tracked.query(&Divide { numerator: Variable::A, denominator: Variable::B }),
    tracked.query(&SafeDivide { numerator: Variable::A, denominator: Variable::B }),
);

println!("Divide(A, B) = {}", result1);     // 21
println!("SafeDivide(A, B) = {:?}", result2); // Some(21)
```

QBICE can safely execute these in parallel, handling any shared dependencies automatically.

## Complete Example

Here's a complete example putting it all together:

```rust
use std::sync::Arc;
use std::sync::atomic::Ordering;
use qbice::{
    Engine, DefaultConfig,
    serialize::Plugin,
    stable_hash::{SeededStableHasherBuilder, Sip128Hasher},
    storage::kv_database::rocksdb::RocksDB,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let mut engine = Engine::<DefaultConfig>::new_with(
        Plugin::default(),
        RocksDB::factory(temp_dir.path()),
        SeededStableHasherBuilder::<Sip128Hasher>::new(0),
    )?;

    let divide_executor = Arc::new(DivideExecutor::new());
    let safe_divide_executor = Arc::new(SafeDivideExecutor::new());

    engine.register_executor(Arc::new(VariableExecutor));
    engine.register_executor(divide_executor.clone());
    engine.register_executor(safe_divide_executor.clone());

    // Wrap engine in Arc
    let engine = Arc::new(engine);

    // Set initial values
    {
        let mut input_session = engine.input_session();
        input_session.set_input(Variable::A, 42);
        input_session.set_input(Variable::B, 2);
    }

    let tracked = engine.tracked();

    // Execute queries
    println!("=== First Execution ===");
    let result = tracked.query(&SafeDivide {
        numerator: Variable::A,
        denominator: Variable::B,
    }).await;
    println!("SafeDivide(A, B) = {:?}", result);
    println!("Divide called: {} times", divide_executor.call_count.load(Ordering::SeqCst));
    println!("SafeDivide called: {} times", safe_divide_executor.call_count.load(Ordering::SeqCst));

    // Execute again (should use cache)
    println!("\n=== Second Execution (cached) ===");
    let result2 = tracked.query(&SafeDivide {
        numerator: Variable::A,
        denominator: Variable::B,
    }).await;
    println!("SafeDivide(A, B) = {:?}", result2);
    println!("Divide called: {} times", divide_executor.call_count.load(Ordering::SeqCst));
    println!("SafeDivide called: {} times", safe_divide_executor.call_count.load(Ordering::SeqCst));

    Ok(())
}
```

Expected output:

```
=== First Execution ===
SafeDivide(A, B) = Some(21)
Divide called: 1 times
SafeDivide called: 1 times

=== Second Execution (cached) ===
SafeDivide(A, B) = Some(21)
Divide called: 1 times
SafeDivide called: 1 times
```

## Key Takeaways

- Use `tracked.query(&query_key)` to execute queries
- Results are automatically cached
- Same query keys return cached results without re-execution
- Different query keys are tracked separately
- Async execution allows for concurrent query processing
- Dependencies are tracked automatically

Next, we'll learn how to update inputs and observe incremental recomputation!

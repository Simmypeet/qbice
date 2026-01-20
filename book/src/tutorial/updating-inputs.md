# Updating Inputs

The real power of incremental computation shines when inputs change. QBICE automatically determines what needs to be recomputed and what can remain cached. Let's see how this works!

## Updating Input Values

To update inputs, create a new input session. Note that the engine must be wrapped in `Arc` to call `input_session()`:

```rust
// First execution
{
    let tracked = engine.tracked();
    let result = tracked.query(&SafeDivide {
        numerator: Variable::A,
        denominator: Variable::B,
    }).await;
    println!("SafeDivide(A, B) = {:?}", result); // Some(21)
} // Drop tracked

// Update input (engine is already in Arc)
{
    let mut input_session = engine.input_session();
    input_session.set_input(Variable::A, 84); // Changed from 42 to 84
} // Changes committed when dropped

// Query again
{
    let tracked = engine.tracked();
    let result = tracked.query(&SafeDivide {
        numerator: Variable::A,
        denominator: Variable::B,
    }).await;
    println!("SafeDivide(A, B) = {:?}", result); // Some(42)
}
```

## Observing Incremental Recomputation

Let's verify QBICE's incremental computation with an interesting example that
tracks executor call counts.

```rust
use std::sync::atomic::Ordering;

// Reset call counters
divide_executor.call_count.store(0, Ordering::SeqCst);
safe_divide_executor.call_count.store(0, Ordering::SeqCst);

// Wrap engine in Arc (required for input_session)
let engine = Arc::new(engine);

// Set up initial state: A=42, B=2
{
    let mut input_session = engine.input_session();
    input_session.set_input(Variable::A, 42);
    input_session.set_input(Variable::B, 2);
}

// Execute SafeDivide query
{
    let tracked = engine.tracked();

    let result = tracked.query(&SafeDivide {
        numerator: Variable::A,
        denominator: Variable::B,
    }).await;

    println!("SafeDivide(A, B) = {:?}", result); // Some(21)
}

println!("Initial execution (A=42, B=2):");
println!("  Divide called: {}", divide_executor.call_count.load(Ordering::SeqCst));
println!("  SafeDivide called: {}", safe_divide_executor.call_count.load(Ordering::SeqCst));
// Output: Divide: 1, SafeDivide: 1
```

Now let's change B to 0 to trigger division by zero:

```rust
// Change B to 0 (engine is already in Arc)
{
    let mut input_session = engine.input_session();
    input_session.set_input(Variable::B, 0);
}

// Query again
{
    let tracked = engine.tracked();

    let result = tracked.query(&SafeDivide {
        numerator: Variable::A,
        denominator: Variable::B,
    }).await;

    println!("SafeDivide(A, B) = {:?}", result); // None
}

println!("After changing B to 0:");
println!("  Divide called: {}", divide_executor.call_count.load(Ordering::SeqCst));
println!("  SafeDivide called: {}", safe_divide_executor.call_count.load(Ordering::SeqCst));
// Output: Divide: 1, SafeDivide: 2
```

Notice that:

- `SafeDivide` was executed again (count increased from 1 to 2)
- `Divide` was **NOT** executed (count stayed at 1)
- This is because `SafeDivide` returns early when denominator is 0

Now let's change B back to 2:

```rust
// Change B back to 2
{
    let mut input_session = engine.input_session();
    input_session.set_input(Variable::B, 2);
}

// Query again
{
    let tracked = engine.tracked();

    let result = tracked.query(&SafeDivide {
        numerator: Variable::A,
        denominator: Variable::B,
    }).await;

    println!("SafeDivide(A, B) = {:?}", result); // Some(21)
}

println!("After changing B back to 2:");
println!("  Divide called: {}", divide_executor.call_count.load(Ordering::SeqCst));
println!("  SafeDivide called: {}", safe_divide_executor.call_count.load(Ordering::SeqCst));
// Output: Divide: 1, SafeDivide: 3
```

This demonstrates QBICE's incremental recomputation:

- `SafeDivide` executed again (count: 2 → 3)
- `Divide` **still didn't execute** (count stayed at 1)
- Even though B changed, its value is back to 2, same as the original
- `Divide`'s cached result from the first execution is still valid!
- QBICE detects this via fingerprint comparison and reuses the cached value

## Understanding Dirty Propagation

When you change an input, QBICE performs **dirty propagation**:

1. The changed input is marked dirty
2. All queries that depend on it are marked dirty (transitively)
3. When a dirty query is requested, it checks its dependencies
4. If a dependency's value hasn't actually changed (via fingerprint comparison), recomputation may stop

In our example above, when we changed B from 0 back to 2:

- B was marked dirty
- `Divide(A, B)` was marked dirty (depends on B)
- `SafeDivide(A, B)` was marked dirty (depends on Divide)
- When `SafeDivide` executed, it sees that `Divide` is dirty
- `Divides` checks its dependencies with what they were before:
  - `Variable::A` is unchanged (42)
  - `Variable::B` is unchanged (2)
- Since both inputs are the same as before, `Divide` reuses its cached result
- Thus, `Divide`'s call count remains at 1 throughout

## Batched Updates

You can update multiple inputs at once:

```rust
{
    let mut input_session = engine.input_session();
    input_session.set_input(Variable::A, 100);
    input_session.set_input(Variable::B, 5);
} // All changes committed atomically
```

## Complete Incremental Example

Here's a complete example demonstrating incremental computation:

```rust
use std::sync::Arc;
use std::sync::atomic::Ordering;

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

    // Initial setup
    {
        let mut input_session = engine.input_session();
        input_session.set_input(Variable::A, 42);
        input_session.set_input(Variable::B, 2);
    }

    // Computation 1: Initial A=42, B=2
    println!("=== Initial Computation (A=42, B=2) ===");
    {
        let tracked = engine.tracked();
        let result = tracked.query(&SafeDivide {
            numerator: Variable::A,
            denominator: Variable::B,
        }).await;
        println!("SafeDivide(A, B) = {:?}", result);
    }
    println!("Divide executions: {}", divide_executor.call_count.load(Ordering::SeqCst));
    println!("SafeDivide executions: {}", safe_divide_executor.call_count.load(Ordering::SeqCst));

    // Update B to 0 (division by zero!)
    println!("\n=== Update B to 0 ===");
    {
        let mut input_session = engine.input_session();
        input_session.set_input(Variable::B, 0);
    }

    // Computation 2: SafeDivide returns None
    {
        let tracked = engine.tracked();
        let result = tracked.query(&SafeDivide {
            numerator: Variable::A,
            denominator: Variable::B,
        }).await;
        println!("SafeDivide(A, B) = {:?}", result);
    }
    println!("Divide executions: {}", divide_executor.call_count.load(Ordering::SeqCst));
    println!("SafeDivide executions: {}", safe_divide_executor.call_count.load(Ordering::SeqCst));

    // Update B back to 2 (original value!)
    println!("\n=== Update B back to 2 ===");
    {
        let mut input_session = engine.input_session();
        input_session.set_input(Variable::B, 2);
    }

    // Computation 3: SafeDivide returns Some(21) again
    {
        let tracked = engine.tracked();
        let result = tracked.query(&SafeDivide {
            numerator: Variable::A,
            denominator: Variable::B,
        }).await;
        println!("SafeDivide(A, B) = {:?}", result);
    }
    println!("Divide executions: {}", divide_executor.call_count.load(Ordering::SeqCst));
    println!("SafeDivide executions: {}", safe_divide_executor.call_count.load(Ordering::SeqCst));

    Ok(())
}
```

Expected output:

```
=== Initial Computation (A=42, B=2) ===
SafeDivide(A, B) = Some(21)
Divide executions: 1
SafeDivide executions: 1

=== Update B to 0 ===
SafeDivide(A, B) = None
Divide executions: 1
SafeDivide executions: 2

=== Update B back to 2 ===
SafeDivide(A, B) = Some(21)
Divide executions: 1
SafeDivide executions: 3
```

## Graph Illustration

To illustrate, this is how compoutation graph looks like on the first run:

```txt
                  Some(21)
                     |
               SafeDivide(A, B)---------+
                     |                  |
                     21                 2
                     |                  |
           +-----Divide(A, B)------+    |
           |                       |    |
           42                      2    |
           |                       |    |
        Variable::A             Variable::B
           |                         |
           42                        2
```

Note that each edge represents a dependency and it records the value that it saw
at that time.

Next, when we change B to 0:

```txt
                  Some(21)
                     |
               SafeDivide(A, B)---------+
                     |                  |
                     21*                2*
                     |                  |
           +-----Divide(A, B)------+    |
           |                       |    |
           42                      2*   |
           |                       |    |
        Variable::A             Variable::B
           |                         |
           42                        0
```

Here we mark the dirtied edges with a `*`. When we changed B to 0, all
transitive edges were marked dirty.

Here when we query for `SafeDivide(A, B)`, it sees that its dependency
`Divide(A, B)` and `Variable::B` are dirty, so it recomputes.

```txt
                   None
                     |
               SafeDivide(A, B)---------+
                                        |
                                        0
                                        |
           +-----Divide(A, B)------+    |
           |                       |    |
           42                      2*   |
           |                       |    |
        Variable::A             Variable::B
           |                         |
           42                        0
```

Here, `SafeDivide` returns early because denominator is 0, so `Divide` is never executed. Note that the edge from `Divide(A, B)` to `Variable::B` remains dirty.

Finally, when we change B back to 2:

```txt
                  Some(21)
                     |
               SafeDivide(A, B)---------+
                                        |
                                        2*
                                        |
           +-----Divide(A, B)------+    |
           |                       |    |
           42                      2*   |
           |                       |    |
        Variable::A             Variable::B
           |                         |
           42                        2
```

When `SafeDivide` executes, it sees that `Variable::B` is dirty, so it
recomputes, which means `Divide` is also invoked.

However, according to above graph, the last time `Divide` executed, both its
inputs (`Variable::A` and `Variable::B`) had the same values (42 and 2
respectively). Since nothing has changed, `Divide` reuses its cached result and
does not execute again.

Resulting in the final graph:

```txt
                  Some(21)
                     |
               SafeDivide(A, B)---------+
                     |                  |
                     21*                2*
                     |                  |
           +-----Divide(A, B)------+    |
           |                       |    |
           42                      2*   |
           |                       |    |
        Variable::A             Variable::B
           |                         |
           42                        2
```

## Key Takeaways

- Drop the old `TrackedEngine` and create a new one after updating inputs
- Create a new input session to update values
- QBICE automatically tracks which queries are affected
- Only queries that depend on changed inputs are recomputed
- Fingerprint-based comparison prevents unnecessary recomputation
- Multiple inputs can be updated in a single session

## What's Next?

You've completed the tutorial! You now know how to:

- Define queries and implement executors
- Set up the engine and register components
- Execute queries and build dependency graphs
- Update inputs and leverage incremental computation

For deeper understanding, continue to the **Reference** section to learn about each component in detail, or explore **Advanced Topics** for optimization techniques like firewall and projection queries.

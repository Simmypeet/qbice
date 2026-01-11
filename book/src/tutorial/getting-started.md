# Getting Started

In this tutorial, we'll build a simple incremental calculator that can compute arithmetic expressions. This will teach you the fundamentals of QBICE: queries, executors, and the engine.

## Project Setup

First, create a new Rust project:

```bash
cargo new qbice-calculator
cd qbice-calculator
```

Add QBICE to your `Cargo.toml`:

```toml
[dependencies]
qbice = "0.2.0"
tokio = { version = "1", features = ["full"] }
tempfile = "3"
```

## What We'll Build

Our calculator will support:

- **Variables** - Named numeric values (A and B)
- **Division** - Dividing two values (with assertion that denominator != 0)
- **Safe Division** - Division that returns `None` for division by zero

The key feature: when we change a variable's value, only dependent computations will be recalculated. Plus, we'll see how queries can depend on other queries (SafeDivide depends on Divide).

## The Problem

Imagine we have these computations:

```
fn Divide(Variable::A, Variable::B) -> i32 {
    let a = read Variable::A;
    let b = read Variable::B;

    panic if b == 0;

    a / b
}

fn SafeDivide(Variable::A, Variable::B) -> Option<i32> {
    let b = read Variable::B;

    if b == 0 {
        return None;
    }

    Some(
        Divide(Variable::A, Variable::B)
    )
}
```

If we start with `A = 42` and `B = 2`, and compute `SafeDivide(A, B)`, we get
`Some(21)`.

Now, if we change `B` to `0` and recompute `SafeDivide(A, B)`, we should get `None`.

Notice how `SafeDivide` depends on `Divide`, which in turn depends on `Variable::A` and `Variable::B`.

Finally, if we change `B` back to `2` and recompute, we should get `Some(21)` again
but without re-executing `Divide` since its result was cached.

QBICE automatically tracks these dependencies and handles the incremental updates.

## Project Structure

We'll build this in stages:

1. **Define Queries** - Create types representing our computations
2. **Implement Executors** - Write the logic for each computation
3. **Set Up the Engine** - Configure QBICE to manage our queries
4. **Execute Queries** - Run computations and see results
5. **Update Inputs** - Change values and observe incremental updates

Let's start by defining our queries!

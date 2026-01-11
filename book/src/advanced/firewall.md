# Firewall and Projection Queries

**Firewall queries** and **projection queries** are advanced optimization techniques that work together to provide fine-grained change detection and prevent unnecessary dirty propagation in dense dependency graphs. They're particularly useful for expensive global analysis queries that have many dependents.

## The Problem: Chokepoints

Consider a compiler that performs global type analysis. This analysis:

- Depends on many input files
- Produces a large type table
- Is depended upon by hundreds of downstream queries

When any input file changes:

1. The global analysis is marked dirty
2. **All** downstream queries are marked dirty (transitively)
3. On next execution, hundreds of queries need verification

This creates a **chokepoint** where a single change causes excessive dirty propagation.

## The Solution: Firewalls

A firewall query acts as a boundary that limits dirty propagation:

```
Input Changes
    ↓
Firewall Query (checked first)
    ↓ (only if result changed)
Downstream Queries
```

When an input changes:

1. Dirty propagation stops at the firewall
2. When the firewall is requested, it's recomputed
3. **Only if the firewall's output changes** does dirty propagation continue

## Defining a Firewall Query

Mark a query as a firewall by overriding `execution_style()`:

```rust
use qbice::{Query, ExecutionStyle};

#[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable, Encode, Decode)]
pub struct GlobalTypeAnalysis {
    // Input parameters
}

impl Query for GlobalTypeAnalysis {
    type Value = TypeTable;  // Large result

    fn execution_style() -> ExecutionStyle {
        ExecutionStyle::Firewall
    }
}
```

## How Firewalls Work

### Without Firewall

```
Input → Analysis → [100 downstream queries all marked dirty]
```

When input changes:

- Analysis marked dirty
- 100 queries marked dirty
- On next query, must verify all 100 queries

### With Firewall

```
Input → Analysis [FIREWALL] → [100 downstream queries]
```

When input changes:

- Analysis marked dirty
- Downstream queries **NOT** marked dirty
- On next query:
  1. Analysis is recomputed
  2. If result unchanged, return cached value
  3. Downstream queries remain clean!

## Example: Global Analysis

```rust
use std::collections::HashMap;

#[derive(Debug, Clone, StableHash)]
pub struct TypeTable {
    pub types: HashMap<ID, Type>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable, Encode, Decode)]
pub struct GlobalTypeAnalysis;

impl Query for GlobalTypeAnalysis {
    type Value = Arc<TypeTable>;

    fn execution_style() -> ExecutionStyle {
        ExecutionStyle::Firewall
    }
}

pub struct GlobalTypeAnalysisExecutor;

impl<C: Config> Executor<GlobalTypeAnalysis, C> for GlobalTypeAnalysisExecutor {
    async fn execute(
        &self,
        _query: &GlobalTypeAnalysis,
        engine: &TrackedEngine<C>,
    ) -> TypeTable {
        // This is expensive and depends on many inputs
        let files = engine.query(&GetAllSourceFiles).await;

        let mut types = HashMap::new();
        for file in files {
            let file_types = engine.query(&ParseFile {
                path: file.path.clone(),
            }).await;
            types.extend(file_types);
        }

        TypeTable { types }
    }
}
```

### Downstream Query

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable, Encode, Decode)]
pub struct GetFunctionType {
    pub function_name: ID,
}

impl Query for GetFunctionType {
    type Value = Option<Type>;
}

impl<C: Config> Executor<GetFunctionType, C> for GetFunctionTypeExecutor {
    async fn execute(
        &self,
        query: &GetFunctionType,
        engine: &TrackedEngine<C>,
    ) -> Option<Type> {
        // Depends on the firewall
        let table = engine.query(&GlobalTypeAnalysis).await;
        table.types.get(&query.function_name).cloned()
    }
}
```

## Transitive Firewall Edges

Queries that transitively depend on a firewall maintain special **transitive firewall edges**:

```
Firewall
    ↓
  Query A
    ↓
  Query B  ← has transitive edge to Firewall
```

When Query B is requested:

1. It first checks the Firewall directly
2. If Firewall changed, dirty flags propagate from Firewall → A → B
3. Then normal verification continues

This ensures correctness while limiting dirty propagation.

## When to Use Firewalls

Use firewalls when a query is:

1. **Expensive to compute** - So recomputation should be avoided
2. **Has many dependents** - Creates a chokepoint
3. **Changes infrequently** - Most input changes don't affect the result
4. **Produces large results** - But downstream queries read small parts

## When NOT to Use Firewalls

Avoid firewalls when:

1. **Query is cheap** - Overhead of firewall logic isn't worth it
2. **Few dependents** - No chokepoint exists
3. **Frequently changes** - Firewall won't prevent much dirty propagation
4. **Small result** - Downstream queries read everything anyway

## Trade-offs

### Benefits

- **Reduced dirty propagation** - Potentially thousands of queries stay clean
- **Fewer verifications** - Less work during repairation
- **Selective invalidation** - Only truly affected queries recompute

### Costs

- **Extra recomputation** - Firewall recomputed even if result doesn't change
- **Increased complexity** - More complex dependency tracking
- **Memory overhead** - Transitive firewall edges stored

The trade-off is worth it when:

```
Cost of extra firewall recomputation < Cost of verifying many downstream queries
```

# Projection Queries

**Projection queries** work together with firewall queries to provide fine-grained change detection. They solve the over-invalidation problem by reading specific slices of large firewall results.

## The Problem: Over-Invalidation

Consider a firewall query that produces a large result:

```rust
impl Query for GlobalTypeTable {
    type Value = Arc<HashMap<ID, Type>>;  // 1000+ entries
    fn execution_style() -> ExecutionStyle {
        ExecutionStyle::Firewall
    }
}
```

Downstream queries typically read just one entry:

```rust
impl<C: Config> Executor<GetFunctionType, C> for GetFunctionTypeExecutor {
    async fn execute(&self, query: &GetFunctionType, engine: &TrackedEngine<C>) -> Option<Type> {
        let table = engine.query(&GlobalTypeTable).await;
        table.get(&query.function_name).cloned()  // Only reads ONE entry!
    }
}
```

The problem:

- A change affects one entry in the table
- The firewall result "changed" (different hash)
- **All** downstream queries are marked dirty
- In reality, only small subset of downstream queries are actually affected

## The Solution: Projections

Projection queries extract specific data from firewall/projection results:

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable, Encode, Decode)]
pub struct GetFunctionType {
    pub function_name: ID,
}

impl Query for GetFunctionType {
    type Value = Option<Type>;

    fn execution_style() -> ExecutionStyle {
        ExecutionStyle::Projection  // Mark as projection!
    }
}
```

Key properties:

- **Must only depend on Firewall**
- **Should be very fast** (memory access, hash lookup, field access)
- **Extracts a small piece** of a larger result

## How Projections Work

### Normal Flow (Without Projection)

```
Input changes
  ↓
Firewall recomputes (hash changes)
  ↓
All downstream marked dirty
```

### With Projection

```
Input changes
  ↓
Firewall recomputes (hash changes)
  ↓
Projection recomputes (checks specific slice)
  ↓
Only if projection's result changed → mark downstream dirty
```

The projection is re-executed immediately when the firewall changes. If the projection's specific slice hasn't changed, dirty propagation stops.

## Example: Type Table

### Firewall Query

```rust
use std::collections::HashMap;

#[derive(Debug, Clone, StableHash)]
pub struct TypeTable {
    pub types: HashMap<ID, Type>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable, Encode, Decode)]
pub struct GlobalTypeTable;

impl Query for GlobalTypeTable {
    type Value = Arc<TypeTable>;  // Use Arc for cheap cloning

    fn execution_style() -> ExecutionStyle {
        ExecutionStyle::Firewall
    }
}
```

### Projection Query

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable, Encode, Decode)]
pub struct GetFunctionType {
    pub function_name: ID,
}

impl Query for GetFunctionType {
    type Value = Option<Type>;

    fn execution_style() -> ExecutionStyle {
        ExecutionStyle::Projection
    }
}

impl<C: Config> Executor<GetFunctionType, C> for GetFunctionTypeExecutor {
    async fn execute(
        &self,
        query: &GetFunctionType,
        engine: &TrackedEngine<C>,
    ) -> Option<Type> {
        // Very fast: just a hash lookup
        let table = engine.query(&GlobalTypeTable).await;
        table.types.get(&query.function_name).cloned()
    }
}
```

### Downstream Query

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable, Encode, Decode)]
pub struct TypeCheckFunction {
    pub function_name: ID,
}

impl Query for TypeCheckFunction {
    type Value = Result<(), TypeError>;
}

impl<C: Config> Executor<TypeCheckFunction, C> for TypeCheckFunctionExecutor {
    async fn execute(
        &self,
        query: &TypeCheckFunction,
        engine: &TrackedEngine<C>,
    ) -> Result<(), TypeError> {
        // Depends on projection
        let function_type = engine.query(&GetFunctionType {
            function_name: query.function_name.clone(),
        }).await;

        // Type check using the function's type
        type_check(function_type)
    }
}
```

### Behavior

When a source file changes:

1. `GlobalTypeTable` (firewall) recomputes
2. Only functions in that file have new types
3. `GetFunctionType` (projection) re-executes for each dependent
4. If a specific function's type **didn't** change, its projection returns the same value
5. Most of the downstream `TypeCheckFunction` queries remain clean

Result: Only functions with changed types are re-checked.

## Projection Rules

### 1. Only Depend on Firewall

```rust
// ✓ GOOD - depends on firewall
impl<C: Config> Executor<MyProjection, C> for MyProjectionExecutor {
    async fn execute(&self, query: &MyProjection, engine: &TrackedEngine<C>) -> Value {
        let firewall_result = engine.query(&MyFirewall).await;
        firewall_result.get_field()
    }
}

// ✗ BAD - depends on normal query
impl<C: Config> Executor<BadProjection, C> for BadProjectionExecutor {
    async fn execute(&self, query: &BadProjection, engine: &TrackedEngine<C>) -> Value {
        let normal = engine.query(&NormalQuery).await;  // NOT ALLOWED!
        normal.field
    }
}
```

### 2. Keep Projections Fast

Projections are re-executed during dirty propagation, so they must be very fast:

```rust
// ✓ GOOD - O(1) lookup
impl<C: Config> Executor<FastProjection, C> for FastProjectionExecutor {
    async fn execute(&self, query: &FastProjection, engine: &TrackedEngine<C>) -> Value {
        let table = engine.query(&FirewallTable).await;
        table.get(&query.key).cloned()  // Hash lookup
    }
}

// ✓ GOOD - field access
impl<C: Config> Executor<FieldProjection, C> for FieldProjectionExecutor {
    async fn execute(&self, query: &FieldProjection, engine: &TrackedEngine<C>) -> Value {
        let data = engine.query(&FirewallData).await;
        data.field.clone()  // Direct field access
    }
}

// ✗ BAD - expensive operation
impl<C: Config> Executor<ExpensiveProjection, C> for ExpensiveProjectionExecutor {
    async fn execute(&self, query: &ExpensiveProjection, engine: &TrackedEngine<C>) -> Value {
        let data = engine.query(&FirewallData).await;
        data.items.iter()
            .filter(|item| expensive_check(item))  // TOO SLOW!
            .collect()
    }
}
```

### 3. Extract Small Slices

Projections should return small portions of the firewall result:

```rust
// ✓ GOOD - returns one item
impl Query for GetItem {
    type Value = Option<Item>;
    fn execution_style() -> ExecutionStyle {
        ExecutionStyle::Projection
    }
}

// ✓ GOOD - returns small subset
impl Query for GetItemSubset {
    type Value = Vec<Item>;  // Small vec
    fn execution_style() -> ExecutionStyle {
        ExecutionStyle::Projection
    }
}

// ✗ BAD - returns entire result
impl Query for GetAllItems {
    type Value = Vec<Item>;  // Entire table!
    fn execution_style() -> ExecutionStyle {
        ExecutionStyle::Projection  // Pointless!
    }
}
```

## Common Patterns

### Map Lookup

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable, Encode, Decode)]
pub struct GetValue {
    pub key: ID,
}

impl Query for GetValue {
    type Value = Option<Value>;
    fn execution_style() -> ExecutionStyle {
        ExecutionStyle::Projection
    }
}

impl<C: Config> Executor<GetValue, C> for GetValueExecutor {
    async fn execute(&self, query: &GetValue, engine: &TrackedEngine<C>) -> Option<Value> {
        let map = engine.query(&GlobalMap).await;
        map.get(&query.key).cloned()
    }
}
```

### Field Access

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable, Encode, Decode)]
pub struct GetConfig {
    pub field: ConfigField,
}

impl Query for GetConfig {
    type Value = ConfigValue;
    fn execution_style() -> ExecutionStyle {
        ExecutionStyle::Projection
    }
}

impl<C: Config> Executor<GetConfig, C> for GetConfigExecutor {
    async fn execute(&self, query: &GetConfig, engine: &TrackedEngine<C>) -> ConfigValue {
        let config = engine.query(&GlobalConfig).await;
        match query.field {
            ConfigField::Timeout => config.timeout,
            ConfigField::MaxRetries => config.max_retries,
            // ...
        }
    }
}
```

### Index Access

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash, StableHash, Identifiable, Encode, Decode)]
pub struct GetArrayElement {
    pub index: usize,
}

impl Query for GetArrayElement {
    type Value = Option<Element>;
    fn execution_style() -> ExecutionStyle {
        ExecutionStyle::Projection
    }
}

impl<C: Config> Executor<GetArrayElement, C> for GetArrayElementExecutor {
    async fn execute(&self, query: &GetArrayElement, engine: &TrackedEngine<C>) -> Option<Element> {
        let array = engine.query(&GlobalArray).await;
        array.get(query.index).cloned()
    }
}
```

## Performance Trade-offs

### Benefits

- **Fine-grained invalidation** - Only truly affected queries recompute
- **Reduced recomputation** - Could save thousands of verifications
- **Composable** - Build chains of projections

### Costs

- **Extra executions** - Projections run during dirty propagation
- **More queries** - Need projection queries in addition to normal queries
- **Increased complexity** - More query types to manage

The trade-off is worth it when:

```
Cost of projection re-execution < Cost of recomputing downstream queries
```

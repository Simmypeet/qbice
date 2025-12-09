# Firewall

To recap, the QBICE incremental dependency graph is made up of queries, each
**query represented a node** in the graph and edges **representing**
**dependencies**. When a query depends on another query, an edge is created both
ways: a forward edge from the caller to the callee, and a backward edge from the
callee to the caller.

The structure looks similar to this:

```rust
pub struct ForwardEdge {
    pub callee: QueryID,
    pub seen_fingerprint: CompactHash128,
    pub dirty: bool,
}

pub struct BackwardEdge {
    pub caller: QueryID,
}

pub struct QueryNode {
    pub forward_edges: Vec<ForwardEdge>, // edges to callees
    pub backward_edges: Vec<BackwardEdge>, // edges to callers

    pub fingerprint: CompactHash128, // current fingerprint
    pub verified_at: Timestamp, // last verified timestamp
}

pub struct QueryGraph {
    pub nodes: HashMap<QueryID, QueryNode>,
}
```

When the leaf nodes (inputs) are mutated, the **dirty flags are propagated**
**upwards** through the backward edges to mark dependent queries as dirty
**transitively**.

> Note that the dirty flag is stored in the forward edge, not the query node
> itself.

The process above is called **dirty propagation**, which traverses the graph
using the **backward edges**.

When a query is requested, if it is marked dirty, it will recursively check its
forward edges to see if any of its callees are dirty, and **repair them first**
**before determining whether it needs to recompute itself** by comparing
fingerprints of its callees with the seen fingerprints stored in the forward
edges.

The process above is called **repairation**, which traverses the graph using the
**forward edges**.

These two processes have some important properties:

- **Dirty propagation is done eagerly** when mutations happen, while
  **repairation is done lazily** when queries are requested.
- **Dirty propagation can be operated in unordered manner**, meaning it's possible
  to use multiple threads to **propagate dirtiness concurrently**.
- When a node is repaired, **it recursively repairs its dirty callees first**,
  then checks whether it needs to recompute itself.
- When a node is repairing its callees, **it needs to do so in the order they**
  **were called**, to ensure the correctness of the computation. This is
  inherently a property of the **dynamic call graph** since one callee's output
  may affect another callee's input.
- When determining whether a node needs to recompute itself: for each dirty
  callee (in order), **it repairs that particular callee first, then compares**
  **the callee's fingerprint after repairation with the seen fingerprint**
  **stored in the forward edge**. If they differ, the checking short-circuits,
  unwires the old edges (including forward edges of itself and backward edges
  of its callees), and recomputes itself.
- The amount of work in **repairation is proportional to the number of dirty**
  **nodes** in the transitive closure of the requested query.
- The **dirty propagation** is much cheaper than **repairation**, if we can
  significantly reduce the number of **dirty edges** in the graph, we can save
  a lot of heavy time spent in repairation.

## Problem: Chokepoints

In a large dense dependency graph, a small mutation at the leaf nodes can cause
**a large number of edges to be marked dirty**, due to the transitive nature of
dirty propagation.

Imagine a scenario where a program has to do some **global analysis** which
**depends on many queries and is called by many other queries**. Think of a
global analysis query as a node in the graph that has many incoming and outgoing
edges, **resembling a chokepoint**.

If a leaf node is mutated, the **global analysis query will be marked dirty**,
which in turn causes **all queries depending on it to be marked dirty as well**.
This can lead to a **large portion of the graph being marked dirty**, even if
only a small part of the graph is actually affected by the mutation.

## Solution: Firewall

We mitigate this problem by marking a certain query as a **firewall**.

A **firewall** is a query that **prevents dirty propagation from crossing it**.
When a mutation occurs, if the dirty propagation reaches a firewall query, it
**marks the firewall query's callees as dirty, but does not propagate the**
**dirty flag to its callers**.

When the firewall query is requested later, it will perform a repairation as
usual, repairing its dirty callees first, then checking whether it needs to
recompute itself.

The difference is that when the firewall query happens to be recomputed, it will
**check against its previous fingerprint**. If it differs, it means that the
**firewall has changed**, and thus dirty flags will need to be propagated from
it to its callers.

This can be seen as an incremental dirty propagation where the firewall query
**acts as a new source of dirtiness** if it recomputes to a different value.

## Firewall Awareness

**Now this raises a question**: what happens to the queries that **transitively
depend on the firewall** query whwn a mutation occurs? Since the **dirty flag**
**will never reach to them** when the mutation happens, how will they know that
they need to repair themselves when requested later?

**The answer is**: all of the transitive dependents of a firewall query will
have a special transitive forward edge to the firewall query itself. These
edges are called **Transitive Firewall Edges**.

Recall that during repairation, a query will repair its dirty callees first,
then determine whether it needs to recompute itself by comparing fingerprints.

When a query has transitive firewall edges to firewall queries, during its
repairation, it will **directly repair those firewall queries first**
(regardless of whether they are dirty or not), then proceed to check its other
callees as usual.

The key observation here is that by repairing the firewall queries first,
if the firewall queries have changed, **eventually the dirty flags will**
**propagate from the firewall queries to their callers**.

## Problem: Over-Invalidation

Recall the earlier scenario where a global analysis query acts as a chokepoint
in the graph.

Imagine that this global analysis query produces a large table as its output
(e.g., a hash map from function names to their types). Many queries depend on
this global analysis query, reading **only one entry** from the table.

For example, that particular global analysis query may produce 200 entries.
But in a next mutation, only one entry is changed. However, **this still**
**counts as the global analysis query being changed**, causing all its dependents
to be marked dirty.

In practice, this could only affect a small portion of the downstream queries,
but due to the coarse granularity of dirty propagation, **all downstream**
**queries are marked dirty and need to be repaired**.

## Solution: Projections

To mitigate the over-invalidation problem, we introduce a new kind of query
that works together with firewall queries, called **Projection Query**.

**Projection Queries** are special queries that depend on only either
**Firewall** or **Projection** queries. Their main purpose is to read a small
part of the output of a firewall or projection query. The projection query
**should be very fast to compute**, for example, memory access, hash map lookup,
etc.

Recall that when the Firewall query is recomputed, if it changes, it will
**propagate dirtiness to its callers**. However, if a caller of the firewall
query is a projection query, it will **run that particular projection query again**
to read the relevant part of the firewall's output. **If the relevant part has**
**not changed, the dirty flag will not propagate further**.

Here we trade off some extra recomputation in the projection query for reducing
the amount of dirty propagation in the graph, which is often worth it since
projection queries are usually much cheaper than propagating dirtiness through
many queries.

## Implementation Details

### Timestamp

To ensure that the engine do the least amount of work during recomputation, we
introduce another subtle but important detail: the **Timestamp**.

In order to ensure the correctness of the incremental computation, everytime
the query is requested, **it needs to ensure that all of its callees are not**
**dirty** by iterating through each calle. This can still be costly if a query
has many callees, even if all of them are clean.

To optimize this, we introduce a **monotonic timestamp** in the engine that
goes up everytime a mutation happens. Beside that, each query node also stores a
**timestamp of when it was last verified**. During repairation, if a query sees
that its **verified at timestamp is smaller than the engine's current**
**timestamp**, it knows that callees may have changed since the last time it was
verified, so it needs to iterate through all of its callees to ensure they are
not dirty.

This ensures that each **query is only repaired at most once per mutation**,
avoiding unnecessary re-verification of clean queries.

In the current setup, the **mutation and query executions cannot be done**
**concurrently**, thus a single global timestamp is sufficient. In the future,
if we want to support concurrent mutations and queries, we may need to introduce
**Multi-Version Concurrency Control (MVCC) to manage multiple timestamps**.

### Transitive Firewall Edge Archetype

The transitive dependents of a firewall query must **maintain a list of**
**firewall** it transitively depends on. In a naive implementation, this can be
done by having each query a copy of the list of firewall queries it depends on.

However, this can lead to high memory usage if many queries depend on the same
firewall. To optimize this, we can introduce an archetype for transitive
firewall edges.

Instead of each query storing its own list of firewall queries, we can have a
shared archetype that stores the list of firewall queries, and each query can
just store a reference to that archetype. This way, multiple queries that depend
on the same firewall can share the same archetype, reducing memory usage.

## Tradeoffs

Nothing comes for free. By introducing firewalls and projections, we are making
some tradeoffs.

- **Increased Checking Overhead**: During repairation, queries now need to
  **repair firewall queries first**, which adds some overhead to the repairation
  process. However, if the users keep the number of firewalls low, this overhead
  should be minimal compared to the savings from reduced dirty propagation.
- **Firewall Repairation**: Firewall queries will always be repaired if the
  dirty flag reaches them during dirty propagation. This means that firewalls
  is expected to be executed. Therefore, **firewalls should be designed to be**
  **readily computable and not panic if they are happened to be recomputed at**
  **any time.**
- **Projection Overhead**: When the firewall changes and is about to propagate
  dirtiness to its callers, **projection queries will need to be executed**
  **again** to check whether the relevant part has changed. This adds some
  overhead to the dirty propagation process.

By acknowledging these tradeoffs, these are the general guidelines for using
firewalls and projections effectively:

- **Minimize the Number of Firewalls**: Benchmark and identify chokepoints in
  the graph, and only place firewalls at those chokepoints. Avoid placing too
  many firewalls, as this can lead to increased checking overhead during
  repairation.
- **Design Firewalls to be readily Computable**: Ensure that firewall queries
  **are executable at any time without panicking**, as they may be recomputed
  during repairation regardless of whether they are actually used by downstream
  queries.
- **Projections should be Fast**: Design projection queries to be fast to
  compute, as they will be executed again during dirty propagation when the
  firewall changes.

## Summary

We have introduced two new kinds of queries: **Firewall** and **Projection**.
These queries help mitigate the problems of **chokepoints** and
**over-invalidation** in the QBICE incremental dependency graph. By using
firewalls to stop dirty propagation at certain points and projections to read
small parts of outputs, we can significantly reduce the amount of unnecessary
recomputation in the graph, leading to reduced number of graph traversals during
repairation and improved overall performance of the incremental compiler.

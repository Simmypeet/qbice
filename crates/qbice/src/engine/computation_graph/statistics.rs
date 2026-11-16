//! Counters of the work the engine does, by query type and by the edges that
//! dirty propagation marks.
//!
//! They let a client assert how much an edit costs, such as that an edit to
//! one function body recomputes the same number of queries however large the
//! program is, without parsing the engine's logs. [`Observability`], set when
//! the engine is built, chooses what they count and when they start again
//! from zero.

use std::{
    collections::BTreeMap,
    fmt,
    sync::atomic::{AtomicUsize, Ordering},
};

use bon::Builder;
use dashmap::DashMap;
use fxhash::FxBuildHasher;
use qbice_stable_type_id::StableTypeID;

use crate::{
    Engine, config::Config, engine::computation_graph::ComputationGraph,
};

/// What the engine records about its own work, set once when it is built:
/// see [`crate::engine::EngineOptions`].
///
/// The engine also emits [`tracing`] events, such as those of the
/// `qbice::dirty` target for dirty propagation; the subscriber filters them,
/// so they need no setting here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Builder)]
pub struct Observability {
    /// What [`Engine::statistics`] counts.
    #[builder(default = StatisticsLevel::Off)]
    pub statistics: StatisticsLevel,

    /// When the counters of [`Engine::statistics`] start again from zero.
    #[builder(default = StatisticsReset::OnCommit)]
    pub reset: StatisticsReset,
}

impl Default for Observability {
    fn default() -> Self { Self::builder().build() }
}

/// What [`Engine::statistics`] counts. Each level counts what the levels
/// before it do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StatisticsLevel {
    /// Nothing. Recording costs a branch per event.
    Off,

    /// The work done for each query type: see [`QueryStatistics`]. Recording
    /// costs a map lookup and an atomic increment per event.
    ByQueryType,

    /// Also the edges that dirty propagation marks, by the query types of
    /// the callee and the caller: see [`Statistics::dirtied_edges_between`].
    /// It tells which dependency carries a change to the queries it reaches,
    /// at the cost of another map lookup per edge marked.
    ByEdge,
}

/// When the counters of [`Engine::statistics`] start again from zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StatisticsReset {
    /// When an input session commits: the counters hold the work of the
    /// current timestamp.
    OnCommit,

    /// Never: the counters hold the work of the engine's lifetime, and
    /// [`Statistics::since`] gives the work between two reads of them.
    Never,
}

/// Records the events of the engine's work in the counters that
/// [`Observability`] chose; [`Engine::statistics`] reads them as
/// [`Statistics`].
#[derive(Debug)]
pub struct StatisticsRecorder {
    observability: Observability,

    /// The counters of each query type that had an event.
    by_type: DashMap<StableTypeID, Counters, FxBuildHasher>,

    /// The dirtied edges by the query types of the callee and the caller,
    /// at [`StatisticsLevel::ByEdge`].
    by_edge: DashMap<(StableTypeID, StableTypeID), AtomicUsize, FxBuildHasher>,
}

/// The counters of one query type.
#[derive(Debug, Default)]
struct Counters {
    fresh: AtomicUsize,
    recomputed_changed: AtomicUsize,
    recomputed_unchanged: AtomicUsize,
    verified_clean: AtomicUsize,
    edges_checked: AtomicUsize,
    dirtied_edges: AtomicUsize,
    propagation_stops: AtomicUsize,
}

impl Counters {
    /// The counter that `event` increments.
    const fn of(&self, event: Event) -> &AtomicUsize {
        match event {
            Event::Fresh => &self.fresh,
            Event::RecomputedChanged => &self.recomputed_changed,
            Event::RecomputedUnchanged => &self.recomputed_unchanged,
            Event::VerifiedClean => &self.verified_clean,
            Event::EdgeChecked => &self.edges_checked,
            Event::DirtiedEdge { .. } => &self.dirtied_edges,
            Event::PropagationStopped => &self.propagation_stops,
        }
    }
}

/// Something the engine did for a query, which [`StatisticsRecorder::record`]
/// counts for its type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Event {
    /// The query had no stored value and was executed.
    Fresh,

    /// The query had a stored value, was executed again, and its value
    /// changed.
    RecomputedChanged,

    /// The query had a stored value, was executed again, and its value
    /// stayed the same.
    RecomputedUnchanged,

    /// The query had a stored value and was repaired without executing it,
    /// since none of its callees changed.
    VerifiedClean,

    /// The repair of the query checked whether an edge to one of its callees
    /// was dirty.
    EdgeChecked,

    /// Dirty propagation marked the edge from the query to its callee dirty.
    DirtiedEdge {
        /// The query type of the callee.
        callee: StableTypeID,
    },

    /// Dirty propagation reached the query, a firewall or a projection,
    /// through a dirtied edge and stopped there.
    PropagationStopped,
}

impl StatisticsRecorder {
    pub(super) fn new(observability: Observability) -> Self {
        Self {
            observability,
            by_type: DashMap::default(),
            by_edge: DashMap::default(),
        }
    }

    /// Whether any counter is recorded.
    pub(super) fn is_enabled(&self) -> bool {
        self.observability.statistics != StatisticsLevel::Off
    }

    /// Records `event` for the query type `query_type`, if [`Observability`]
    /// chose to count it.
    pub(super) fn record(&self, query_type: StableTypeID, event: Event) {
        match self.observability.statistics {
            StatisticsLevel::Off => return,
            StatisticsLevel::ByQueryType => {}
            StatisticsLevel::ByEdge => {
                if let Event::DirtiedEdge { callee } = event {
                    increment(&self.by_edge, (callee, query_type), |count| {
                        count
                    });
                }
            }
        }

        increment(&self.by_type, query_type, |counters| counters.of(event));
    }
}

/// Increments the counter that `counter` picks from the entry `key` of `map`,
/// creating the entry if it has none.
fn increment<K: Copy + Eq + std::hash::Hash, V: Default>(
    map: &DashMap<K, V, FxBuildHasher>,
    key: K,
    counter: impl Fn(&V) -> &AtomicUsize,
) {
    // most events are for an entry that already exists, which only needs a
    // shared lock of its shard; it is released before taking the exclusive
    // lock that creates the entry
    if let Some(entry) = map.get(&key) {
        counter(&entry).fetch_add(1, Ordering::Relaxed);
        return;
    }

    counter(&map.entry(key).or_default()).fetch_add(1, Ordering::Relaxed);
}

impl<C: Config> ComputationGraph<C> {
    /// Sets the counters back to zero for a new timestamp, unless
    /// [`Observability`] keeps them for the engine's lifetime.
    pub(super) fn reset_statistics_on_commit(&self) {
        match self.statistics_recorder.observability.reset {
            StatisticsReset::OnCommit => {
                self.statistics_recorder.by_type.clear();
                self.statistics_recorder.by_edge.clear();
            }
            StatisticsReset::Never => {}
        }
    }
}

/// The work the engine did for the queries of one type.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QueryStatistics {
    fresh: usize,
    recomputed_changed: usize,
    recomputed_unchanged: usize,
    verified_clean: usize,
    edges_checked: usize,
    dirtied_edges: usize,
    propagation_stops: usize,
}

impl QueryStatistics {
    /// The number of queries executed with no stored value, which the engine
    /// had never computed.
    #[must_use]
    pub const fn fresh(&self) -> usize { self.fresh }

    /// The number of queries executed again whose value changed.
    #[must_use]
    pub const fn recomputed_changed(&self) -> usize { self.recomputed_changed }

    /// The number of queries executed again whose value stayed the same: the
    /// work that a firewall or a projection could have saved.
    #[must_use]
    pub const fn recomputed_unchanged(&self) -> usize {
        self.recomputed_unchanged
    }

    /// The number of queries executed again, whether or not their value
    /// changed.
    #[must_use]
    pub const fn recomputed(&self) -> usize {
        self.recomputed_changed + self.recomputed_unchanged
    }

    /// The number of queries executed, fresh or again.
    #[must_use]
    pub const fn executed(&self) -> usize { self.fresh + self.recomputed() }

    /// The number of queries with a stored value that a repair found clean,
    /// without executing them.
    #[must_use]
    pub const fn verified_clean(&self) -> usize { self.verified_clean }

    /// The number of edges to callees whose dirtiness the repairs of these
    /// queries checked.
    ///
    /// A repair that checks the callees of an unordered group in parallel
    /// stops checking once one of them requires recomputing, so this count
    /// can differ between runs.
    #[must_use]
    pub const fn edges_checked(&self) -> usize { self.edges_checked }

    /// The number of edges from these queries to their callees that dirty
    /// propagation marked dirty.
    ///
    /// Each edge is marked once per timestamp, so a query reached through
    /// several of its callees counts once for each.
    #[must_use]
    pub const fn dirtied_edges(&self) -> usize { self.dirtied_edges }

    /// The number of dirtied edges through which dirty propagation reached
    /// these queries, firewalls or projections, and stopped: such a query is
    /// executed again eagerly instead, and dirties its callers only if its
    /// value changed.
    #[must_use]
    pub const fn propagation_stops(&self) -> usize { self.propagation_stops }

    /// The names and values of the counters.
    const fn counters(&self) -> [(&'static str, usize); 7] {
        [
            ("fresh", self.fresh),
            ("recomputed changed", self.recomputed_changed),
            ("recomputed unchanged", self.recomputed_unchanged),
            ("verified clean", self.verified_clean),
            ("edges checked", self.edges_checked),
            ("dirtied edges", self.dirtied_edges),
            ("propagation stops", self.propagation_stops),
        ]
    }

    /// The counters of `self` and `other`, combined by `combine`.
    fn zip_with(
        &self,
        other: &Self,
        combine: impl Fn(usize, usize) -> usize,
    ) -> Self {
        Self {
            fresh: combine(self.fresh, other.fresh),
            recomputed_changed: combine(
                self.recomputed_changed,
                other.recomputed_changed,
            ),
            recomputed_unchanged: combine(
                self.recomputed_unchanged,
                other.recomputed_unchanged,
            ),
            verified_clean: combine(self.verified_clean, other.verified_clean),
            edges_checked: combine(self.edges_checked, other.edges_checked),
            dirtied_edges: combine(self.dirtied_edges, other.dirtied_edges),
            propagation_stops: combine(
                self.propagation_stops,
                other.propagation_stops,
            ),
        }
    }
}

/// The work the engine did, by query type, since its counters last started
/// from zero: see [`Observability`].
///
/// Query types are identified by their [`StableTypeID`]: see
/// [`Engine::get_type_name_from_stable_type_id`] and
/// [`Engine::get_execution_style_from_stable_type_id`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Statistics {
    by_type: BTreeMap<StableTypeID, QueryStatistics>,

    /// The dirtied edges by the query types of the callee and the caller.
    by_edge: BTreeMap<(StableTypeID, StableTypeID), usize>,
}

impl Statistics {
    /// The work done for the queries of the type `query_type`, or `None` if
    /// there was none.
    #[must_use]
    pub fn get(&self, query_type: StableTypeID) -> Option<&QueryStatistics> {
        self.by_type.get(&query_type)
    }

    /// The work done for each query type that had any.
    pub fn iter(
        &self,
    ) -> impl Iterator<Item = (StableTypeID, &QueryStatistics)> {
        self.by_type
            .iter()
            .map(|(query_type, statistics)| (*query_type, statistics))
    }

    /// The work done for all query types together.
    #[must_use]
    pub fn total(&self) -> QueryStatistics {
        self.by_type
            .values()
            .fold(QueryStatistics::default(), |total, statistics| {
                total.zip_with(statistics, |a, b| a + b)
            })
    }

    /// The number of edges from queries of the type `caller` to callees of
    /// the type `callee` that dirty propagation marked dirty. Counted only at
    /// [`StatisticsLevel::ByEdge`].
    #[must_use]
    #[allow(clippy::similar_names)] // callee and caller are the terms
    pub fn dirtied_edges_between(
        &self,
        callee: StableTypeID,
        caller: StableTypeID,
    ) -> usize {
        self.by_edge.get(&(callee, caller)).copied().unwrap_or(0)
    }

    /// The dirtied edges as `(callee, caller, count)` by the query types of
    /// their ends, for each pair with any. Counted only at
    /// [`StatisticsLevel::ByEdge`].
    pub fn dirtied_edges(
        &self,
    ) -> impl Iterator<Item = (StableTypeID, StableTypeID, usize)> {
        self.by_edge
            .iter()
            .map(|(&(callee, caller), &count)| (callee, caller, count))
    }

    /// The work done between `earlier`, an earlier read of the counters, and
    /// this one, with [`StatisticsReset::Never`].
    ///
    /// With [`StatisticsReset::OnCommit`], a counter that started from zero
    /// in between and has not caught up counts as zero.
    #[must_use]
    pub fn since(&self, earlier: &Self) -> Self {
        let empty = QueryStatistics::default();
        let by_type = self
            .by_type
            .iter()
            .map(|(query_type, statistics)| {
                let before = earlier.by_type.get(query_type).unwrap_or(&empty);
                (
                    *query_type,
                    statistics.zip_with(before, usize::saturating_sub),
                )
            })
            .filter(|(_, statistics)| *statistics != empty)
            .collect::<BTreeMap<_, _>>();

        let by_edge = self
            .by_edge
            .iter()
            .map(|(edge, count)| {
                let before = earlier.by_edge.get(edge).copied().unwrap_or(0);
                (*edge, count.saturating_sub(before))
            })
            .filter(|(_, count)| *count != 0)
            .collect();

        Self { by_type, by_edge }
    }

    /// Displays one line for each query type with its counters that are not
    /// zero, by name, then one for each pair of query types with dirtied
    /// edges, the most first.
    ///
    /// `type_name` names the query types, such as
    /// [`Engine::get_type_name_from_stable_type_id`] does; a type it does not
    /// name is shown by its ID.
    pub fn display<'a>(
        &'a self,
        type_name: impl Fn(StableTypeID) -> Option<&'static str> + 'a,
    ) -> impl fmt::Display + 'a {
        DisplayStatistics { statistics: self, type_name }
    }
}

/// See [`Statistics::display`].
struct DisplayStatistics<'a, F> {
    statistics: &'a Statistics,
    type_name: F,
}

impl<F: Fn(StableTypeID) -> Option<&'static str>> DisplayStatistics<'_, F> {
    fn name(&self, query_type: StableTypeID) -> String {
        (self.type_name)(query_type)
            .map_or_else(|| format!("{query_type:?}"), str::to_owned)
    }
}

impl<F: Fn(StableTypeID) -> Option<&'static str>> fmt::Display
    for DisplayStatistics<'_, F>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut by_type = self
            .statistics
            .iter()
            .map(|(query_type, statistics)| (self.name(query_type), statistics))
            .collect::<Vec<_>>();
        by_type.sort_by(|(a, _), (b, _)| a.cmp(b));

        for (name, statistics) in by_type {
            let counters = statistics
                .counters()
                .into_iter()
                .filter(|(_, value)| *value != 0)
                .map(|(counter, value)| format!("{value} {counter}"))
                .collect::<Vec<_>>();
            writeln!(f, "{name}: {}", counters.join(", "))?;
        }

        let mut edges = self
            .statistics
            .dirtied_edges()
            .map(|(callee, caller, count)| {
                (std::cmp::Reverse(count), self.name(callee), self.name(caller))
            })
            .collect::<Vec<_>>();
        edges.sort();
        for (std::cmp::Reverse(count), callee, caller) in edges {
            writeln!(f, "{count} dirtied edges {callee} -> {caller}")?;
        }

        Ok(())
    }
}

impl<C: Config> Engine<C> {
    /// The work the engine did, by query type, since its counters last
    /// started from zero, as [`Observability`] chose to count it.
    #[must_use]
    pub fn statistics(&self) -> Statistics {
        let recorder = &self.computation_graph.statistics_recorder;
        let load = |counter: &AtomicUsize| counter.load(Ordering::Relaxed);

        let mut by_type = BTreeMap::new();
        for entry in &recorder.by_type {
            let counters = entry.value();
            by_type.insert(*entry.key(), QueryStatistics {
                fresh: load(&counters.fresh),
                recomputed_changed: load(&counters.recomputed_changed),
                recomputed_unchanged: load(&counters.recomputed_unchanged),
                verified_clean: load(&counters.verified_clean),
                edges_checked: load(&counters.edges_checked),
                dirtied_edges: load(&counters.dirtied_edges),
                propagation_stops: load(&counters.propagation_stops),
            });
        }

        let by_edge = recorder
            .by_edge
            .iter()
            .map(|entry| (*entry.key(), load(entry.value())))
            .collect();

        Statistics { by_type, by_edge }
    }
}

use crate::{
    Engine, ExecutionStyle, config::Config,
    engine::computation_graph::ComputationGraph, query::QueryID,
};

/// A drop guard for undoing the registration of a callee query.
///
/// This aims to ensure cancelation safety in case of the task being yielded and
/// canceled mid query.
pub struct UndoRegisterCallee<'d, C: Config> {
    graph: &'d ComputationGraph<C>,
    caller_source: Option<QueryID>,
    callee_target: QueryID,
    defused: bool,
}

impl<'d, C: Config> UndoRegisterCallee<'d, C> {
    /// Creates a new [`UndoRegisterCallee`] instance.
    pub const fn new(
        graph: &'d ComputationGraph<C>,
        caller_source: Option<QueryID>,
        callee_target: QueryID,
    ) -> Self {
        Self { graph, caller_source, callee_target, defused: false }
    }

    /// Don't undo the registration when dropped.
    pub fn defuse(mut self) { self.defused = true; }
}

impl<C: Config> Drop for UndoRegisterCallee<'_, C> {
    fn drop(&mut self) {
        if self.defused {
            return;
        }

        if let Some(caller) = self.caller_source.as_ref() {
            let caller_meta = self.graph.lock.get_lock(caller);

            caller_meta.abort_callee(&self.callee_target);
        }
    }
}
impl<C: Config> Engine<C> {
    pub(super) fn register_callee(
        &self,
        caller_source: Option<&QueryID>,
        calee_target: &QueryID,
    ) -> Option<UndoRegisterCallee<'_, C>> {
        // record the dependency first, don't necessary need to figure out
        // the observed value fingerprint yet
        caller_source.map_or_else(
            || None,
            |caller| {
                let computing = self.computation_graph.lock.get_lock(caller);

                assert!(
                    !computing.query_kind().is_external_input(),
                    "`ExternalInput` queries cannot call other queries"
                );

                // Invariant Check: projection query can only requires firewall
                // queries.
                if computing.query_kind().is_projection() {
                    // get the kind of query about to be registerd by looking
                    // up from the executor registry
                    let entry =
                        self.executor_registry.get_executor_entry_by_type_id(
                            &calee_target.stable_type_id(),
                        );
                    let exec_style = entry.obtain_execution_style();

                    assert!(
                        matches!(exec_style, ExecutionStyle::Firewall),
                        "Projection query can only depend on firewall queries"
                    );
                }

                computing.register_calee(calee_target);

                Some(UndoRegisterCallee::new(
                    &self.computation_graph,
                    Some(*caller),
                    *calee_target,
                ))
            },
        )
    }
}

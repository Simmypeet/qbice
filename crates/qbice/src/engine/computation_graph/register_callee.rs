use std::sync::Arc;

use crate::{
    Engine, ExecutionStyle,
    config::Config,
    engine::computation_graph::{CallerInformation, lock::Computing},
    query::QueryID,
};

/// A drop guard for undoing the registration of a callee query.
///
/// This aims to ensure cancelation safety in case of the task being yielded and
/// canceled mid query.
pub struct UndoRegisterCallee {
    caller_computing: Arc<Computing>,
    callee_target: QueryID,
    defused: bool,
}

impl UndoRegisterCallee {
    /// Creates a new [`UndoRegisterCallee`] instance.
    pub const fn new(
        caller_computing: Arc<Computing>,
        callee_target: QueryID,
    ) -> Self {
        Self { caller_computing, callee_target, defused: false }
    }

    /// Don't undo the registration when dropped.
    pub fn defuse(mut self) { self.defused = true; }
}

impl Drop for UndoRegisterCallee {
    fn drop(&mut self) {
        if self.defused {
            return;
        }

        self.caller_computing.abort_callee(&self.callee_target);
    }
}
impl<C: Config> Engine<C> {
    pub(super) fn register_callee(
        &self,
        caller: &CallerInformation,
        calee_target: &QueryID,
    ) -> Option<UndoRegisterCallee> {
        // record the dependency first, don't necessary need to figure out
        // the observed value fingerprint yet
        caller.get_query_caller().map_or_else(
            || None,
            |caller| {
                let computing = caller.computing();

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

                Some(UndoRegisterCallee::new(computing.clone(), *calee_target))
            },
        )
    }
}

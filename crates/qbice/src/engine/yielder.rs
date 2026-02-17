use std::sync::atomic::AtomicUsize;

pub struct EveryNQueryYielder {
    counter: AtomicUsize,
    n: usize,
}

impl EveryNQueryYielder {
    /// Creates a new `EveryNQueryYielder` that will yield to the async runtime
    /// every `n + 1` queries.
    pub const fn new(n: usize) -> Self {
        Self { counter: AtomicUsize::new(0), n: n + 1 }
    }

    /// Checks if the engine should yield to the async runtime based on the
    /// number of queries executed since the last yield.
    pub async fn tick(&self) {
        let count =
            self.counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        if count.is_multiple_of(self.n) {
            // Yield to the async runtime to allow other tasks to make progress.
            tokio::task::yield_now().await;
        }
    }
}

pub enum Yielder {
    Never,
    EveryNQuery(EveryNQueryYielder),
}

impl Yielder {
    /// Creates a new `Yielder` that will yield to the async runtime every `n`
    /// queries.
    pub const fn every_n_query(n: usize) -> Self {
        Self::EveryNQuery(EveryNQueryYielder::new(n))
    }

    /// Creates a new `Yielder` that will never yield to the async runtime.
    pub const fn never() -> Self { Self::Never }

    /// Checks if the engine should yield to the async runtime based on the
    /// configured yielding strategy.
    pub async fn tick(&self) {
        match self {
            Self::Never => {}
            Self::EveryNQuery(yielder) => yielder.tick().await,
        }
    }
}

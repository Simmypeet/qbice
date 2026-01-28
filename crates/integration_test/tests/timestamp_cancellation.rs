#![allow(missing_docs)]

use std::{
    sync::{Arc, atomic::AtomicUsize},
    time::Duration,
};

use qbice::{Config, Decode, Encode, Executor, Identifiable, StableHash};
use qbice_integration_test::{Variable, create_test_engine};
use tempfile::tempdir;
use tokio::sync::{Notify, mpsc::UnboundedSender};
use tokio_util::sync::CancellationToken;

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    StableHash,
    Identifiable,
    Hash,
    Encode,
    Decode,
)]
pub struct HangingQuery(pub Variable);

impl qbice::Query for HangingQuery {
    type Value = i64;
}

#[derive(Debug)]
pub struct HangingQueryExecutor {
    query_start_sender: UnboundedSender<()>,
    query_done_sender: UnboundedSender<i64>,
    notify: Arc<Notify>,
    call_count: AtomicUsize,
}

impl<C: Config> Executor<HangingQuery, C> for HangingQueryExecutor {
    async fn execute(
        &self,
        query: &HangingQuery,
        engine: &qbice::TrackedEngine<C>,
    ) -> i64 {
        let variable = engine.query(&query.0).await;

        if variable == 0 {
            let notified = self.notify.notified();
            let _ = self.query_start_sender.send(());

            notified.await;
        }

        self.call_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let result = variable * 2;

        let _ = self.query_done_sender.send(result);

        result
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn basic_timestamp_cancellation() {
    let tempdir = tempdir().unwrap();

    let notify = Arc::new(Notify::new());

    let (query_start_sender, mut query_start_recv) =
        tokio::sync::mpsc::unbounded_channel();
    let (query_done_sender, mut query_done_recv) =
        tokio::sync::mpsc::unbounded_channel();

    let mut engine = create_test_engine(&tempdir);

    let hanging_executor = Arc::new(HangingQueryExecutor {
        query_start_sender,
        notify: notify.clone(),
        query_done_sender,
        call_count: AtomicUsize::new(0),
    });

    engine.register_executor(hanging_executor.clone());

    let cancellation_token = CancellationToken::new();

    let engine = Arc::new(engine);

    // start with zero
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(0), 0).await;
    }

    let new_session_handle = {
        let engine = engine.clone();
        let cancellation_token = cancellation_token.clone();

        tokio::spawn(async move {
            // wait for signal to start new timestamp
            let _ = query_start_recv.recv().await;

            // start new timestamp
            {
                let mut input_session = engine.input_session().await;
                input_session.set_input(Variable(0), 2).await;
                drop(input_session);
            }

            notify.notify_waiters();

            // spawn a background thread signaling cancellation after 1 second
            tokio::spawn(async move {
                loop {
                    if query_done_recv.recv().await == Some(0) {
                        tokio::time::sleep(Duration::from_secs(1)).await;
                        cancellation_token.cancel();
                    }
                }
            });

            let tracked_engine = engine.tracked().await;
            let result = tracked_engine.query(&HangingQuery(Variable(0))).await;

            assert_eq!(result, 4);
        })
    };

    let tracked_engine = engine.clone().tracked().await;
    let result = tokio::select! {
        () = cancellation_token.cancelled() => {
            // cancelled
            None
        }

        res = tracked_engine.query(&HangingQuery(Variable(0))) => {
            Some(res)
        }
    };

    assert!(result.is_none());

    let _ = new_session_handle.await;

    // should've been called twice in total
    assert_eq!(
        hanging_executor.call_count.load(std::sync::atomic::Ordering::Relaxed),
        2
    );

    // query again to ensure engine is still functional
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(0), 3).await;
    }

    let tracked_engine = engine.tracked().await;

    assert_eq!(tracked_engine.query(&HangingQuery(Variable(0))).await, 6);
}

#[tokio::test(flavor = "multi_thread")]
async fn multiple_concurrent_queries_cancelled() {
    let tempdir = tempdir().unwrap();

    let notify = Arc::new(Notify::new());

    let (query_start_sender, mut query_start_recv) =
        tokio::sync::mpsc::unbounded_channel();
    let (query_done_sender, mut query_done_recv) =
        tokio::sync::mpsc::unbounded_channel();

    let mut engine = create_test_engine(&tempdir);

    let hanging_executor = Arc::new(HangingQueryExecutor {
        query_start_sender,
        notify: notify.clone(),
        query_done_sender,
        call_count: AtomicUsize::new(0),
    });

    engine.register_executor(hanging_executor.clone());

    let cancellation_token = CancellationToken::new();
    let engine = Arc::new(engine);

    // start with zero
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(0), 0).await;
        input_session.set_input(Variable(1), 0).await;
        input_session.set_input(Variable(2), 0).await;
    }

    let new_session_handle = {
        let engine = engine.clone();
        let cancellation_token = cancellation_token.clone();

        tokio::spawn(async move {
            // wait for three queries to start
            for _ in 0..3 {
                let _ = query_start_recv.recv().await;
            }

            // increment timestamp
            {
                let mut input_session = engine.input_session().await;
                input_session.set_input(Variable(0), 2).await;
                drop(input_session);
            }

            notify.notify_waiters();

            // signal cancellation after seeing the first query complete
            tokio::spawn(async move {
                // wait for all three queries to complete
                let mut completed = 0;
                while completed < 3 {
                    if query_done_recv.recv().await == Some(0) {
                        completed += 1;
                    }
                }

                tokio::time::sleep(Duration::from_secs(1)).await;
                cancellation_token.cancel();
            });

            let tracked_engine = engine.tracked().await;
            let result = tracked_engine.query(&HangingQuery(Variable(0))).await;

            assert_eq!(result, 4);
        })
    };

    let tracked_engine = engine.clone().tracked().await;

    // start multiple concurrent queries
    let handles: Vec<_> = (0..3)
        .map(|i| {
            let tracked_engine = tracked_engine.clone();
            let cancellation_token = cancellation_token.clone();
            let variable = HangingQuery(Variable(i));

            tokio::spawn(async move {
                tokio::select! {
                    () = cancellation_token.cancelled() => None,
                    res = tracked_engine.query(&variable) => Some(res),
                }
            })
        })
        .collect();

    // collect results
    for handle in handles {
        let result = handle.await.unwrap();

        // all should be cancelled
        assert!(result.is_none());
    }

    let _ = new_session_handle.await;

    // verify engine is still functional
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(0), 5).await;
    }

    let tracked_engine = engine.tracked().await;
    assert_eq!(tracked_engine.query(&HangingQuery(Variable(0))).await, 10);
}

#[tokio::test(flavor = "multi_thread")]
async fn rapid_timestamp_increments() {
    let tempdir = tempdir().unwrap();

    let notify = Arc::new(Notify::new());

    let (query_start_sender, mut query_start_recv) =
        tokio::sync::mpsc::unbounded_channel();
    let (query_done_sender, mut query_done_recv) =
        tokio::sync::mpsc::unbounded_channel();

    let mut engine = create_test_engine(&tempdir);

    let hanging_executor = Arc::new(HangingQueryExecutor {
        query_start_sender,
        notify: notify.clone(),
        query_done_sender,
        call_count: AtomicUsize::new(0),
    });

    engine.register_executor(hanging_executor.clone());

    let cancellation_token = CancellationToken::new();
    let engine = Arc::new(engine);

    // start with zero
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(0), 0).await;
    }

    let increment_handle = {
        let engine = engine.clone();
        let cancellation_token = cancellation_token.clone();

        tokio::spawn(async move {
            // wait for first query to start
            let _ = query_start_recv.recv().await;

            // rapidly increment timestamp multiple times
            for i in 1..=5 {
                {
                    let mut input_session = engine.input_session().await;
                    input_session.set_input(Variable(0), i * 2).await;
                    drop(input_session);
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }

            notify.notify_waiters();

            // cancel after first result
            if query_done_recv.recv().await == Some(0) {
                tokio::time::sleep(Duration::from_millis(100)).await;
                cancellation_token.cancel();
            }

            // query with latest timestamp should succeed
            let tracked_engine = engine.tracked().await;
            let result = tracked_engine.query(&HangingQuery(Variable(0))).await;

            assert_eq!(result, 20); // 10 * 2
        })
    };

    let tracked_engine = engine.clone().tracked().await;
    let result = tokio::select! {
        () = cancellation_token.cancelled() => None,
        res = tracked_engine.query(&HangingQuery(Variable(0))) => Some(res),
    };

    assert!(result.is_none());

    let _ = increment_handle.await;
}

#[tokio::test(flavor = "multi_thread")]
async fn stale_tracked_engine_queries_timeout() {
    let tempdir = tempdir().unwrap();

    let notify = Arc::new(Notify::new());

    let (query_start_sender, mut query_start_recv) =
        tokio::sync::mpsc::unbounded_channel();
    let (query_done_sender, _query_done_recv) =
        tokio::sync::mpsc::unbounded_channel();

    let mut engine = create_test_engine(&tempdir);

    let hanging_executor = Arc::new(HangingQueryExecutor {
        query_start_sender,
        notify: notify.clone(),
        query_done_sender,
        call_count: AtomicUsize::new(0),
    });

    engine.register_executor(hanging_executor.clone());

    let engine = Arc::new(engine);

    // start with zero
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(0), 0).await;
    }

    // create tracked engine before timestamp increment
    let stale_tracked_engine = engine.clone().tracked().await;

    let _stale_query_handle = tokio::spawn({
        let stale_tracked_engine = stale_tracked_engine.clone();
        async move { stale_tracked_engine.query(&HangingQuery(Variable(0))).await }
    });

    // wait for query to start hanging
    let _ = query_start_recv.recv().await;

    // increment timestamp - this makes stale_tracked_engine stale
    {
        let mut input_session = engine.input_session().await;
        input_session.set_input(Variable(0), 5).await;
        drop(input_session);
    }

    // unblock the hanging query
    notify.notify_waiters();

    // try to query again with the stale tracked engine
    // this should hang indefinitely since the engine is stale
    let stale_query_result = tokio::time::timeout(
        Duration::from_millis(500),
        stale_tracked_engine.query(&HangingQuery(Variable(0))),
    )
    .await;

    // should timeout because stale queries get stuck
    assert!(stale_query_result.is_err(), "Stale query should timeout");
}

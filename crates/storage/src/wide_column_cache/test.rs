use std::sync::Arc;

#[tokio::test(flavor = "multi_thread")]
pub async fn insert_overwrite_pending_get_init() {
    let cache = Arc::new(moka::future::Cache::<i32, i32>::builder().build());

    let (done_outer_insert_tx, done_outer_insert_rx) =
        tokio::sync::oneshot::channel::<()>();

    let (start_get_init_tx, start_get_init_rx) =
        tokio::sync::oneshot::channel::<()>();

    tokio::spawn({
        let cache = cache.clone();

        async move {
            cache
                .optionally_get_with(1, async {
                    start_get_init_tx.send(()).unwrap();

                    // Wait until we are notified to proceed.
                    let _ = done_outer_insert_rx.await;
                    None
                })
                .await;
        }
    });

    // wait until get_with has started its init future
    let _ = start_get_init_rx.await;

    // we can now insert while the init future is still pending
    cache.insert(1, 7).await;
    assert_eq!(cache.get(&1).await, Some(7));

    // now allow the get_with init future to complete
    let _ = done_outer_insert_tx.send(());

    assert_eq!(cache.get(&1).await, Some(7));
}

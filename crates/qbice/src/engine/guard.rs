use std::pin::Pin;

/// A guard created from [`GuardExt::guarded`].
pub struct Guard<T: Send + 'static> {
    future: Option<Pin<Box<dyn Future<Output = T> + Send>>>,
}

impl<T: Send + 'static> Future for Guard<T> {
    type Output = T;

    fn poll(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if let Some(future) = self.future.as_mut() {
            match future.as_mut().poll(cx) {
                std::task::Poll::Ready(value) => {
                    // Mark the future as completed.
                    self.future = None;

                    std::task::Poll::Ready(value)
                }
                std::task::Poll::Pending => std::task::Poll::Pending,
            }
        } else {
            panic!("Guard polled after completion");
        }
    }
}

impl<T: Send + 'static> Drop for Guard<T> {
    fn drop(&mut self) {
        // If the future is still present, spawn it to ensure it completes.
        if let Some(future) = self.future.take() {
            tokio::spawn(future);
        }
    }
}

/// Allows creating a guarded future from any future.
pub trait GuardExt: Future
where
    Self::Output: Send + 'static,
{
    /// Wraps the future in a `Guard`, ensuring it completes even if dropped.
    ///
    /// This works by spawning the future onto the Tokio runtime if it is
    /// dropped before completion.
    ///
    /// This is useful for "transactional" operations that must complete to
    /// maintain system integrity.
    fn guarded(self) -> Guard<Self::Output>;
}

impl<F: Future + Send + 'static> GuardExt for F
where
    F::Output: Send + 'static,
{
    fn guarded(self) -> Guard<F::Output> {
        Guard { future: Some(Box::pin(self)) }
    }
}

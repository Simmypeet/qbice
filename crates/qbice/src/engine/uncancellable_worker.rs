use std::pin::Pin;

pub struct Guard {
    future: Option<Pin<Box<dyn Future<Output = ()> + Send>>>,
}

impl Future for Guard {
    type Output = ();

    fn poll(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if let Some(future) = self.future.as_mut() {
            match future.as_mut().poll(cx) {
                std::task::Poll::Ready(()) => {
                    // Mark the future as completed.
                    self.future = None;

                    std::task::Poll::Ready(())
                }
                std::task::Poll::Pending => std::task::Poll::Pending,
            }
        } else {
            panic!("Guard polled after completion");
        }
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        // If the future is still present, spawn it to ensure it completes.
        if let Some(future) = self.future.take() {
            tokio::spawn(future);
        }
    }
}

pub trait GuardExt {
    fn guarded(self) -> Guard;
}

impl<F: Future<Output = ()> + Send + 'static> GuardExt for F {
    fn guarded(self) -> Guard { Guard { future: Some(Box::pin(self)) } }
}

use crate::{Config, Engine};

impl<C: Config> Engine<C> {
    pub(super) async fn cancel(&self) -> ! { futures::future::pending().await }
}

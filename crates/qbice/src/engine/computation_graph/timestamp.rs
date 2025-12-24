use std::sync::atomic::AtomicU64;

use qbice_serialize::{Decode, Encode};
use qbice_stable_type_id::Identifiable;
use qbice_storage::kv_database::{
    Column, KvDatabase, Normal, WriteTransaction,
};

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Encode,
    Decode,
    Identifiable,
)]
pub struct Timestamp(u64);

#[derive(Identifiable)]
struct TimestampColumn;

impl Column for TimestampColumn {
    type Key = ();
    type Value = Timestamp;
    type Mode = Normal;
}

pub struct TimestampManager {
    current_timestamp: AtomicU64,
}

impl TimestampManager {
    pub fn new(db: &impl KvDatabase) -> Self {
        let current_timestamp = db.get::<TimestampColumn>(&());

        current_timestamp.map_or_else(
            || {
                let tx = db.write_transaction();

                tx.put::<TimestampColumn>(&(), &Timestamp(0));

                tx.commit();

                Self { current_timestamp: AtomicU64::new(0) }
            },
            |current_timestamp| Self {
                current_timestamp: AtomicU64::new(current_timestamp.0),
            },
        )
    }

    pub fn increment(&self, tx: &impl WriteTransaction) {
        let new_timestamp = self
            .current_timestamp
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;

        let timestamp = Timestamp(new_timestamp);

        tx.put::<TimestampColumn>(&(), &timestamp);
    }

    pub fn get_current(&self) -> Timestamp {
        Timestamp(
            self.current_timestamp.load(std::sync::atomic::Ordering::SeqCst),
        )
    }
}

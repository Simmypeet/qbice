pub trait DynQuery: Any {
    fn stable_type_id(&self) -> StableTypeID;
    fn hash_128(&self) -> u128;
}

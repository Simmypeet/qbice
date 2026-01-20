//! Tests for ExternalInput executor style and refresh functionality.

#![allow(missing_docs)]

use std::sync::{
    Arc, RwLock,
    atomic::{AtomicUsize, Ordering},
};

use qbice::{
    Decode, Encode, Executor, Identifiable, Query, StableHash, TrackedEngine,
    config::Config, query::ExecutionStyle,
};
use qbice_integration_test::{Variable, create_test_engine};
use tempfile::tempdir;

// ============================================================================
// File Read Query - Simulates reading from external files
// ============================================================================

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Identifiable,
    StableHash,
    Encode,
    Decode,
)]
pub struct FileRead(pub u32);

impl Query for FileRead {
    type Value = String;
}

#[derive(Debug)]
pub struct FileReadExecutor {
    pub call_count: AtomicUsize,
    pub file_contents: RwLock<std::collections::HashMap<u32, String>>,
}

impl Default for FileReadExecutor {
    fn default() -> Self { Self::new() }
}

impl FileReadExecutor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            call_count: AtomicUsize::new(0),
            file_contents: RwLock::new(std::collections::HashMap::new()),
        }
    }

    pub fn set_file_content(&self, id: u32, content: String) {
        self.file_contents.write().unwrap().insert(id, content);
    }

    pub fn get_call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

impl<C: Config> Executor<FileRead, C> for FileReadExecutor {
    fn execution_style() -> ExecutionStyle { ExecutionStyle::ExternalInput }

    async fn execute(
        &self,
        query: &FileRead,
        _engine: &TrackedEngine<C>,
    ) -> String {
        self.call_count.fetch_add(1, Ordering::SeqCst);

        // Simulate reading from a file
        self.file_contents
            .read()
            .unwrap()
            .get(&query.0)
            .cloned()
            .unwrap_or_else(|| format!("default-{}", query.0))
    }
}

// ============================================================================
// Dependent Query - Depends on FileRead
// ============================================================================

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Identifiable,
    StableHash,
    Encode,
    Decode,
)]
pub struct UpperCase(pub FileRead);

impl Query for UpperCase {
    type Value = String;
}

#[derive(Debug, Default)]
pub struct UpperCaseExecutor {
    pub call_count: AtomicUsize,
}

impl<C: Config> Executor<UpperCase, C> for UpperCaseExecutor {
    async fn execute(
        &self,
        query: &UpperCase,
        engine: &TrackedEngine<C>,
    ) -> String {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        let content = engine.query(&query.0).await;
        content.to_uppercase()
    }
}

// ============================================================================
// Network Request Query - Another ExternalInput example
// ============================================================================

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Identifiable,
    StableHash,
    Encode,
    Decode,
)]
pub struct NetworkRequest(pub u32);

impl Query for NetworkRequest {
    type Value = i64;
}

#[derive(Debug)]
pub struct NetworkRequestExecutor {
    pub call_count: AtomicUsize,
    pub responses: RwLock<std::collections::HashMap<u32, i64>>,
}

impl Default for NetworkRequestExecutor {
    fn default() -> Self { Self::new() }
}

impl NetworkRequestExecutor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            call_count: AtomicUsize::new(0),
            responses: RwLock::new(std::collections::HashMap::new()),
        }
    }

    pub fn set_response(&self, id: u32, value: i64) {
        self.responses.write().unwrap().insert(id, value);
    }
}

impl<C: Config> Executor<NetworkRequest, C> for NetworkRequestExecutor {
    fn execution_style() -> ExecutionStyle { ExecutionStyle::ExternalInput }

    async fn execute(
        &self,
        query: &NetworkRequest,
        _engine: &TrackedEngine<C>,
    ) -> i64 {
        self.call_count.fetch_add(1, Ordering::SeqCst);

        // Simulate network request
        self.responses.read().unwrap().get(&query.0).copied().unwrap_or(0)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[tokio::test]
async fn external_input_basic_execution() {
    let tempdir = tempdir().unwrap();
    let mut engine = create_test_engine(&tempdir);

    let file_executor = Arc::new(FileReadExecutor::new());
    file_executor.set_file_content(1, "hello world".to_string());

    engine.register_executor(file_executor.clone());

    let engine = Arc::new(engine);
    let tracked = engine.tracked();

    // First execution
    let result = tracked.query(&FileRead(1)).await;
    assert_eq!(result, "hello world");
    assert_eq!(file_executor.get_call_count(), 1);

    // Second execution should use cache
    let result = tracked.query(&FileRead(1)).await;
    assert_eq!(result, "hello world");
    assert_eq!(file_executor.get_call_count(), 1);
}

#[tokio::test]
async fn refresh_re_executes_external_input() {
    let tempdir = tempdir().unwrap();
    let mut engine = create_test_engine(&tempdir);

    let file_executor = Arc::new(FileReadExecutor::new());
    file_executor.set_file_content(1, "initial content".to_string());

    engine.register_executor(file_executor.clone());

    let engine = Arc::new(engine);

    // First execution
    {
        let tracked = engine.clone().tracked();
        let result = tracked.query(&FileRead(1)).await;
        assert_eq!(result, "initial content");
        assert_eq!(file_executor.get_call_count(), 1);
    }

    // Simulate file content change in the external world
    file_executor.set_file_content(1, "updated content".to_string());

    // Refresh the query - should re-execute
    {
        let mut input_session = engine.input_session();
        input_session.refresh::<FileRead>().await;
    }

    assert_eq!(file_executor.get_call_count(), 2);

    // Query again - should get new content
    {
        let tracked = engine.tracked();
        let result = tracked.query(&FileRead(1)).await;
        assert_eq!(result, "updated content");
        // Should not execute again, using cached result from refresh
        assert_eq!(file_executor.get_call_count(), 2);
    }
}

#[tokio::test]
async fn refresh_only_dirties_if_result_changed() {
    let tempdir = tempdir().unwrap();
    let mut engine = create_test_engine(&tempdir);

    let file_executor = Arc::new(FileReadExecutor::new());
    let upper_executor = Arc::new(UpperCaseExecutor::default());

    file_executor.set_file_content(1, "hello".to_string());

    engine.register_executor(file_executor.clone());
    engine.register_executor(upper_executor.clone());

    let engine = Arc::new(engine);

    // First execution
    {
        let tracked = engine.clone().tracked();
        let result = tracked.query(&UpperCase(FileRead(1))).await;
        assert_eq!(result, "HELLO");
        assert_eq!(file_executor.get_call_count(), 1);
        assert_eq!(upper_executor.call_count.load(Ordering::SeqCst), 1);
    }

    // Refresh but content hasn't actually changed
    {
        let mut input_session = engine.input_session();
        input_session.refresh::<FileRead>().await;
    }

    // FileRead was re-executed
    assert_eq!(file_executor.get_call_count(), 2);

    // Query UpperCase - should NOT re-execute since FileRead result didn't
    // change
    {
        let tracked = engine.tracked();
        let result = tracked.query(&UpperCase(FileRead(1))).await;
        assert_eq!(result, "HELLO");
        assert_eq!(upper_executor.call_count.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn refresh_propagates_when_result_changes() {
    let tempdir = tempdir().unwrap();
    let mut engine = create_test_engine(&tempdir);

    let file_executor = Arc::new(FileReadExecutor::new());
    let upper_executor = Arc::new(UpperCaseExecutor::default());

    file_executor.set_file_content(1, "hello".to_string());

    engine.register_executor(file_executor.clone());
    engine.register_executor(upper_executor.clone());

    let engine = Arc::new(engine);

    // First execution
    {
        let tracked = engine.clone().tracked();
        let result = tracked.query(&UpperCase(FileRead(1))).await;
        assert_eq!(result, "HELLO");
        assert_eq!(file_executor.get_call_count(), 1);
        assert_eq!(upper_executor.call_count.load(Ordering::SeqCst), 1);
    }

    // Change the file content
    file_executor.set_file_content(1, "goodbye".to_string());

    // Refresh
    {
        let mut input_session = engine.input_session();
        input_session.refresh::<FileRead>().await;
    }

    // FileRead was re-executed
    assert_eq!(file_executor.get_call_count(), 2);

    // Query UpperCase - should re-execute since FileRead result changed
    {
        let tracked = engine.tracked();
        let result = tracked.query(&UpperCase(FileRead(1))).await;
        assert_eq!(result, "GOODBYE");
        assert_eq!(upper_executor.call_count.load(Ordering::SeqCst), 2);
    }
}

#[tokio::test]
async fn refresh_multiple_queries_of_same_type() {
    let tempdir = tempdir().unwrap();
    let mut engine = create_test_engine(&tempdir);

    let file_executor = Arc::new(FileReadExecutor::new());

    file_executor.set_file_content(1, "file1".to_string());
    file_executor.set_file_content(2, "file2".to_string());
    file_executor.set_file_content(3, "file3".to_string());

    engine.register_executor(file_executor.clone());

    let engine = Arc::new(engine);

    // Execute multiple FileRead queries
    {
        let tracked = engine.clone().tracked();
        assert_eq!(tracked.query(&FileRead(1)).await, "file1");
        assert_eq!(tracked.query(&FileRead(2)).await, "file2");
        assert_eq!(tracked.query(&FileRead(3)).await, "file3");
        assert_eq!(file_executor.get_call_count(), 3);
    }

    // Change only file 2
    file_executor.set_file_content(2, "file2-updated".to_string());

    // Refresh all FileRead queries
    {
        let mut input_session = engine.input_session();
        input_session.refresh::<FileRead>().await;
    }

    // All 3 should be re-executed
    assert_eq!(file_executor.get_call_count(), 6);

    // Query all again - file 2 should have new content
    {
        let tracked = engine.clone().tracked();
        assert_eq!(tracked.query(&FileRead(1)).await, "file1");
        assert_eq!(tracked.query(&FileRead(2)).await, "file2-updated");
        assert_eq!(tracked.query(&FileRead(3)).await, "file3");
        // Should use cached results from refresh
        assert_eq!(file_executor.get_call_count(), 6);
    }
}

#[tokio::test]
async fn refresh_different_query_types_independent() {
    let tempdir = tempdir().unwrap();
    let mut engine = create_test_engine(&tempdir);

    let file_executor = Arc::new(FileReadExecutor::new());
    let network_executor = Arc::new(NetworkRequestExecutor::new());

    file_executor.set_file_content(1, "file".to_string());
    network_executor.set_response(1, 100);

    engine.register_executor(file_executor.clone());
    engine.register_executor(network_executor.clone());

    let engine = Arc::new(engine);

    // Execute both types
    {
        let tracked = engine.clone().tracked();
        assert_eq!(tracked.query(&FileRead(1)).await, "file");
        assert_eq!(tracked.query(&NetworkRequest(1)).await, 100);
        assert_eq!(file_executor.get_call_count(), 1);
        assert_eq!(network_executor.call_count.load(Ordering::SeqCst), 1);
    }

    // Refresh only FileRead
    {
        let mut input_session = engine.input_session();
        input_session.refresh::<FileRead>().await;
    }

    // Only FileRead should be re-executed
    assert_eq!(file_executor.get_call_count(), 2);
    assert_eq!(network_executor.call_count.load(Ordering::SeqCst), 1);

    // Refresh only NetworkRequest
    {
        let mut input_session = engine.input_session();
        input_session.refresh::<NetworkRequest>().await;
    }

    // Now NetworkRequest should be re-executed
    assert_eq!(file_executor.get_call_count(), 2);
    assert_eq!(network_executor.call_count.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn refresh_with_no_previous_queries() {
    let tempdir = tempdir().unwrap();
    let mut engine = create_test_engine(&tempdir);

    let file_executor = Arc::new(FileReadExecutor::new());
    engine.register_executor(file_executor.clone());

    let engine = Arc::new(engine);

    // Refresh without ever executing any FileRead queries
    {
        let mut input_session = engine.input_session();
        input_session.refresh::<FileRead>().await;
    }

    // Should be a no-op
    assert_eq!(file_executor.get_call_count(), 0);
}

#[tokio::test]
async fn external_input_not_affected_by_normal_refresh() {
    let tempdir = tempdir().unwrap();
    let mut engine = create_test_engine(&tempdir);

    let file_executor = Arc::new(FileReadExecutor::new());
    let upper_executor = Arc::new(UpperCaseExecutor::default());

    file_executor.set_file_content(1, "test".to_string());

    engine.register_executor(file_executor.clone());
    engine.register_executor(upper_executor.clone());

    let engine = Arc::new(engine);

    // First execution
    {
        let tracked = engine.clone().tracked();
        let result = tracked.query(&UpperCase(FileRead(1))).await;
        assert_eq!(result, "TEST");
        assert_eq!(file_executor.get_call_count(), 1);
        assert_eq!(upper_executor.call_count.load(Ordering::SeqCst), 1);
    }

    // Create a new input session but don't refresh
    {
        let _input_session = engine.input_session();
        // Just drop it without calling refresh
    }

    // Query again - ExternalInput should still be cached
    {
        let tracked = engine.tracked();
        let result = tracked.query(&FileRead(1)).await;
        assert_eq!(result, "test");
        assert_eq!(file_executor.get_call_count(), 1);
    }
}

#[tokio::test]
async fn refresh_multiple_times() {
    let tempdir = tempdir().unwrap();
    let mut engine = create_test_engine(&tempdir);

    let file_executor = Arc::new(FileReadExecutor::new());
    file_executor.set_file_content(1, "v1".to_string());

    engine.register_executor(file_executor.clone());

    let engine = Arc::new(engine);

    // First execution
    {
        let tracked = engine.clone().tracked();
        assert_eq!(tracked.query(&FileRead(1)).await, "v1");
        assert_eq!(file_executor.get_call_count(), 1);
    }

    // Refresh 1
    file_executor.set_file_content(1, "v2".to_string());
    {
        let mut input_session = engine.input_session();
        input_session.refresh::<FileRead>().await;
    }
    assert_eq!(file_executor.get_call_count(), 2);

    // Refresh 2
    file_executor.set_file_content(1, "v3".to_string());
    {
        let mut input_session = engine.input_session();
        input_session.refresh::<FileRead>().await;
    }
    assert_eq!(file_executor.get_call_count(), 3);

    // Refresh 3
    file_executor.set_file_content(1, "v4".to_string());
    {
        let mut input_session = engine.input_session();
        input_session.refresh::<FileRead>().await;
    }
    assert_eq!(file_executor.get_call_count(), 4);

    // Query - should get latest
    {
        let tracked = engine.tracked();
        assert_eq!(tracked.query(&FileRead(1)).await, "v4");
        assert_eq!(file_executor.get_call_count(), 4);
    }
}

#[tokio::test]
async fn refresh_in_same_session_as_other_inputs() {
    let tempdir = tempdir().unwrap();
    let mut engine = create_test_engine(&tempdir);

    let file_executor = Arc::new(FileReadExecutor::new());
    file_executor.set_file_content(1, "file".to_string());

    engine.register_executor(file_executor.clone());

    let engine = Arc::new(engine);

    // Set a regular input and refresh external input in same session
    {
        let mut input_session = engine.input_session();
        input_session.set_input(Variable(0), 42);
        input_session.refresh::<FileRead>().await;
        input_session.set_input(Variable(1), 100);
    }

    // Both should work
    {
        let tracked = engine.tracked();
        assert_eq!(tracked.query(&Variable(0)).await, 42);
        assert_eq!(tracked.query(&Variable(1)).await, 100);
    }
}

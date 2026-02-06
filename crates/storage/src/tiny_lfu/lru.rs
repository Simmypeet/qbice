use fxhash::FxHashMap;

#[derive(Debug)]
struct Entry<K> {
    val: K,
    prev: Option<usize>,
    next: Option<usize>,
}

/// A simple LRU (Least Recently Used) list implementation.
#[derive(Debug)]
pub struct Lru<K> {
    free_list: Vec<usize>,

    head: Option<usize>,
    tail: Option<usize>,

    entries: Vec<Option<Entry<K>>>,
    index_map: FxHashMap<K, usize>,
}

impl Default for Lru<String> {
    fn default() -> Self { Self::new() }
}

impl<K> Lru<K> {
    /// Creates a new, empty LRU list.
    #[must_use]
    pub fn new() -> Self {
        Self {
            free_list: Vec::new(),
            head: None,
            tail: None,
            entries: Vec::new(),
            index_map: FxHashMap::default(),
        }
    }
}

impl<K: std::hash::Hash + Eq + Clone> Lru<K> {
    /// Inserts or updates the key to be the most recently used item (moves to
    /// head).
    pub fn hit(&mut self, key: &K) {
        if let Some(&idx) = self.index_map.get(key) {
            // Key exists, move it to the head
            self.move_to_head(idx);
        } else {
            // Key doesn't exist, insert it at the head
            self.insert_at_head(key.clone());
        }
    }

    /// Peeks at the least recently used item (from tail) without removing it.
    pub fn peek_lru(&self) -> Option<&K> {
        let tail_idx = self.tail?;
        let entry = self.entries[tail_idx].as_ref()?;
        Some(&entry.val)
    }

    /// Pops the least recently used item (from tail).
    pub fn pop(&mut self) -> Option<K> {
        let tail_idx = self.tail?;

        let entry = self.entries[tail_idx].take()?;
        self.index_map.remove(&entry.val);
        self.free_list.push(tail_idx);

        // Update tail
        self.tail = entry.prev;
        if let Some(new_tail) = self.tail {
            self.entries[new_tail].as_mut().unwrap().next = None;
        } else {
            // List is now empty
            self.head = None;
        }

        Some(entry.val)
    }

    fn move_to_head(&mut self, idx: usize) {
        if self.head == Some(idx) {
            return; // Already at head
        }

        let entry = self.entries[idx].as_mut().unwrap();
        let prev = entry.prev;
        let next = entry.next;

        // Unlink from current position
        if let Some(prev_idx) = prev {
            self.entries[prev_idx].as_mut().unwrap().next = next;
        }
        if let Some(next_idx) = next {
            self.entries[next_idx].as_mut().unwrap().prev = prev;
        }

        // Update tail if this was the tail
        if self.tail == Some(idx) {
            self.tail = prev;
        }

        // Move to head
        let old_head = self.head;
        self.head = Some(idx);

        let entry = self.entries[idx].as_mut().unwrap();
        entry.prev = None;
        entry.next = old_head;

        if let Some(old_head_idx) = old_head {
            self.entries[old_head_idx].as_mut().unwrap().prev = Some(idx);
        }
    }

    fn insert_at_head(&mut self, key: K) {
        let idx = if let Some(free_idx) = self.free_list.pop() {
            self.entries[free_idx] =
                Some(Entry { val: key.clone(), prev: None, next: self.head });
            free_idx
        } else {
            let idx = self.entries.len();
            self.entries.push(Some(Entry {
                val: key.clone(),
                prev: None,
                next: self.head,
            }));
            idx
        };

        if let Some(old_head) = self.head {
            self.entries[old_head].as_mut().unwrap().prev = Some(idx);
        } else {
            // List was empty, this is also the tail
            self.tail = Some(idx);
        }

        self.head = Some(idx);
        self.index_map.insert(key, idx);
    }

    /// Removes the specified key from the LRU list.
    pub fn remove(&mut self, key: &K) {
        if let Some(&idx) = self.index_map.get(key) {
            let entry = self.entries[idx].take().unwrap();
            self.index_map.remove(key);
            self.free_list.push(idx);

            // Unlink from list
            if let Some(prev_idx) = entry.prev {
                self.entries[prev_idx].as_mut().unwrap().next = entry.next;
            } else {
                // This was head
                self.head = entry.next;
            }

            if let Some(next_idx) = entry.next {
                self.entries[next_idx].as_mut().unwrap().prev = entry.prev;
            } else {
                // This was tail
                self.tail = entry.prev;
            }
        }
    }

    pub fn len(&self) -> usize { self.index_map.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_pop() {
        let mut lru: Lru<i32> = Lru::new();
        assert_eq!(lru.pop(), None);
    }

    #[test]
    fn test_single_insert_and_pop() {
        let mut lru = Lru::new();
        lru.hit(&1);
        assert_eq!(lru.pop(), Some(1));
        assert_eq!(lru.pop(), None);
    }

    #[test]
    fn test_lru_order() {
        let mut lru = Lru::new();
        lru.hit(&1);
        lru.hit(&2);
        lru.hit(&3);

        // Order: head -> 3 -> 2 -> 1 -> tail
        // Pop should return 1 (least recently used)
        assert_eq!(lru.pop(), Some(1));
        assert_eq!(lru.pop(), Some(2));
        assert_eq!(lru.pop(), Some(3));
        assert_eq!(lru.pop(), None);
    }

    #[test]
    fn test_hit_moves_to_head() {
        let mut lru = Lru::new();
        lru.hit(&1);
        lru.hit(&2);
        lru.hit(&3);

        // Now hit 1 again, moving it to head
        lru.hit(&1);

        // Order: head -> 1 -> 3 -> 2 -> tail
        assert_eq!(lru.pop(), Some(2));
        assert_eq!(lru.pop(), Some(3));
        assert_eq!(lru.pop(), Some(1));
    }

    #[test]
    fn test_hit_existing_head() {
        let mut lru = Lru::new();
        lru.hit(&1);
        lru.hit(&2);

        // Hit 2 (already head), should not change order
        lru.hit(&2);

        assert_eq!(lru.pop(), Some(1));
        assert_eq!(lru.pop(), Some(2));
    }

    #[test]
    fn test_hit_middle_element() {
        let mut lru = Lru::new();
        lru.hit(&1);
        lru.hit(&2);
        lru.hit(&3);

        // Hit 2 (middle element)
        lru.hit(&2);

        // Order: head -> 2 -> 3 -> 1 -> tail
        assert_eq!(lru.pop(), Some(1));
        assert_eq!(lru.pop(), Some(3));
        assert_eq!(lru.pop(), Some(2));
    }

    #[test]
    fn test_reuse_free_slots() {
        let mut lru = Lru::new();
        lru.hit(&1);
        lru.hit(&2);

        // Pop and insert new element
        assert_eq!(lru.pop(), Some(1));
        lru.hit(&3);

        // Free slot from 1 should be reused
        assert!(lru.entries.len() <= 2);

        assert_eq!(lru.pop(), Some(2));
        assert_eq!(lru.pop(), Some(3));
    }

    #[test]
    fn test_string_keys() {
        let mut lru = Lru::new();
        lru.hit(&"hello".to_string());
        lru.hit(&"world".to_string());

        lru.hit(&"hello".to_string());

        assert_eq!(lru.pop(), Some("world".to_string()));
        assert_eq!(lru.pop(), Some("hello".to_string()));
    }

    #[test]
    fn test_many_operations() {
        let mut lru = Lru::new();

        for i in 0..100 {
            lru.hit(&i);
        }

        // Hit some elements to move them to head
        lru.hit(&0);
        lru.hit(&50);

        // Pop all, first 98 should be in reverse insertion order (excluding 0
        // and 50)
        let mut popped = Vec::new();
        while let Some(val) = lru.pop() {
            popped.push(val);
        }

        assert_eq!(popped.len(), 100);
        // Last two should be 50 and 0 (most recently hit)
        assert_eq!(popped[98], 50);
        assert_eq!(popped[99], 0);
    }
}

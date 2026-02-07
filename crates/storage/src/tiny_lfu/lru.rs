use std::hash::Hash;

use fxhash::FxHashMap;

/// A `usize` that uses `usize::MAX` as a sentinel for "no value",
/// allowing `Entry` to store linked-list pointers without the extra
/// byte of `Option<usize>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NonMax(usize);

impl NonMax {
    const NONE: Self = Self(usize::MAX);

    fn some(val: usize) -> Self {
        debug_assert_ne!(val, usize::MAX, "NonMax cannot hold usize::MAX");
        Self(val)
    }

    const fn get(self) -> Option<usize> {
        if self.0 == usize::MAX { None } else { Some(self.0) }
    }
}

#[derive(Debug)]
struct Entry<K> {
    val: K,
    prev: NonMax,
    next: NonMax,
}

pub struct Cursor<'x, K> {
    lru: &'x mut Lru<K>,
    current: NonMax,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(unused)]
pub enum MoveTo {
    MoreRecent,
    LessRecent,
}

impl<K: Eq + Hash + Clone> Cursor<'_, K> {
    pub fn move_to(&mut self, direction: MoveTo) {
        if let Some(current_idx) = self.current.get() {
            let entry = self.lru.entries[current_idx].as_ref().unwrap();
            self.current = match direction {
                MoveTo::MoreRecent => entry.prev,
                MoveTo::LessRecent => entry.next,
            };
        }
    }

    pub fn get(&self) -> Option<&K> {
        let current_idx = self.current.get()?;
        Some(&self.lru.entries[current_idx].as_ref().unwrap().val)
    }

    pub fn remove(&mut self, move_to: MoveTo) -> Option<K> {
        let current_idx = self.current.get()?;

        let entry = self.lru.entries[current_idx].take().unwrap();

        self.current = match move_to {
            MoveTo::MoreRecent => entry.prev,
            MoveTo::LessRecent => entry.next,
        };

        self.lru.index_map.remove(&entry.val);
        self.lru.free_list.push(current_idx);

        if let Some(prev_idx) = entry.prev.get() {
            self.lru.entries[prev_idx].as_mut().unwrap().next = entry.next;
        } else {
            self.lru.head = entry.next;
        }

        if let Some(next_idx) = entry.next.get() {
            self.lru.entries[next_idx].as_mut().unwrap().prev = entry.prev;
        } else {
            self.lru.tail = entry.prev;
        }

        Some(entry.val)
    }
}

/// A simple LRU (Least Recently Used) list implementation.
#[derive(Debug)]
pub struct Lru<K> {
    free_list: Vec<usize>,

    head: NonMax,
    tail: NonMax,

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
            head: NonMax::NONE,
            tail: NonMax::NONE,
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

    /// Moves the key to the head if it exists. Returns true if it was found.
    pub fn hit_if_exists(&mut self, key: &K) -> bool {
        if let Some(&idx) = self.index_map.get(key) {
            self.move_to_head(idx);
            true
        } else {
            false
        }
    }

    /// Checks if the key exists in the list.
    pub fn contains(&self, key: &K) -> bool { self.index_map.contains_key(key) }

    /// Pops the least recently used item (from tail).
    pub fn pop_least_recent(&mut self) -> Option<K> {
        let tail_idx = self.tail.get()?;

        let entry = self.entries[tail_idx].take()?;
        self.index_map.remove(&entry.val);
        self.free_list.push(tail_idx);

        // Update tail
        self.tail = entry.prev;
        if let Some(new_tail) = self.tail.get() {
            self.entries[new_tail].as_mut().unwrap().next = NonMax::NONE;
        } else {
            // List is now empty
            self.head = NonMax::NONE;
        }

        Some(entry.val)
    }

    pub fn peek_least_recent(&self) -> Option<&K> {
        let tail_idx = self.tail.get()?;
        Some(&self.entries[tail_idx].as_ref().unwrap().val)
    }

    /// Inserts or updates the key to be the least recently used item (moves to
    /// tail).
    #[allow(unused)]
    pub fn push_to_least_recently_used(&mut self, key: &K) {
        if let Some(&idx) = self.index_map.get(key) {
            self.move_to_tail(idx);
        } else {
            self.insert_at_tail(key.clone());
        }
    }

    fn move_to_head(&mut self, idx: usize) {
        if self.head == NonMax::some(idx) {
            return; // Already at head
        }

        let entry = self.entries[idx].as_mut().unwrap();
        let prev = entry.prev;
        let next = entry.next;

        // Unlink from current position
        if let Some(prev_idx) = prev.get() {
            self.entries[prev_idx].as_mut().unwrap().next = next;
        }
        if let Some(next_idx) = next.get() {
            self.entries[next_idx].as_mut().unwrap().prev = prev;
        }

        // Update tail if this was the tail
        if self.tail == NonMax::some(idx) {
            self.tail = prev;
        }

        // Move to head
        let old_head = self.head;
        self.head = NonMax::some(idx);

        let entry = self.entries[idx].as_mut().unwrap();
        entry.prev = NonMax::NONE;
        entry.next = old_head;

        if let Some(old_head_idx) = old_head.get() {
            self.entries[old_head_idx].as_mut().unwrap().prev =
                NonMax::some(idx);
        }
    }

    fn insert_at_head(&mut self, key: K) {
        let idx = if let Some(free_idx) = self.free_list.pop() {
            self.entries[free_idx] = Some(Entry {
                val: key.clone(),
                prev: NonMax::NONE,
                next: self.head,
            });
            free_idx
        } else {
            let idx = self.entries.len();
            self.entries.push(Some(Entry {
                val: key.clone(),
                prev: NonMax::NONE,
                next: self.head,
            }));
            idx
        };

        if let Some(old_head) = self.head.get() {
            self.entries[old_head].as_mut().unwrap().prev = NonMax::some(idx);
        } else {
            // List was empty, this is also the tail
            self.tail = NonMax::some(idx);
        }

        self.head = NonMax::some(idx);
        self.index_map.insert(key, idx);
    }

    fn move_to_tail(&mut self, idx: usize) {
        if self.tail == NonMax::some(idx) {
            return; // Already at tail
        }

        let entry = self.entries[idx].as_mut().unwrap();
        let prev = entry.prev;
        let next = entry.next;

        // Unlink from current position
        if let Some(prev_idx) = prev.get() {
            self.entries[prev_idx].as_mut().unwrap().next = next;
        }
        if let Some(next_idx) = next.get() {
            self.entries[next_idx].as_mut().unwrap().prev = prev;
        }

        // Update head if this was head
        if self.head == NonMax::some(idx) {
            self.head = next;
        }

        // Move to tail
        let old_tail = self.tail;
        self.tail = NonMax::some(idx);

        let entry = self.entries[idx].as_mut().unwrap();
        entry.next = NonMax::NONE;
        entry.prev = old_tail;

        if let Some(old_tail_idx) = old_tail.get() {
            self.entries[old_tail_idx].as_mut().unwrap().next =
                NonMax::some(idx);
        } else {
            // List was empty (or this was the only element, but we checked
            // tail==idx) If the list is empty, then head must also
            // be None.
            self.head = NonMax::some(idx);
        }
    }

    fn insert_at_tail(&mut self, key: K) {
        let idx = if let Some(free_idx) = self.free_list.pop() {
            self.entries[free_idx] = Some(Entry {
                val: key.clone(),
                prev: self.tail,
                next: NonMax::NONE,
            });
            free_idx
        } else {
            let idx = self.entries.len();
            self.entries.push(Some(Entry {
                val: key.clone(),
                prev: self.tail,
                next: NonMax::NONE,
            }));
            idx
        };

        if let Some(old_tail) = self.tail.get() {
            self.entries[old_tail].as_mut().unwrap().next = NonMax::some(idx);
        } else {
            // List was empty, this is also the head
            self.head = NonMax::some(idx);
        }

        self.tail = NonMax::some(idx);
        self.index_map.insert(key, idx);
    }

    /// Removes the specified key from the LRU list.
    pub fn remove(&mut self, key: &K) -> Option<K> {
        if let Some(&idx) = self.index_map.get(key) {
            let entry = self.entries[idx].take().unwrap();
            self.index_map.remove(key);
            self.free_list.push(idx);

            // Unlink from list
            if let Some(prev_idx) = entry.prev.get() {
                self.entries[prev_idx].as_mut().unwrap().next = entry.next;
            } else {
                // This was head
                self.head = entry.next;
            }

            if let Some(next_idx) = entry.next.get() {
                self.entries[next_idx].as_mut().unwrap().prev = entry.prev;
            } else {
                // This was tail
                self.tail = entry.prev;
            }

            Some(entry.val)
        } else {
            None
        }
    }

    pub fn len(&self) -> usize { self.index_map.len() }

    pub const fn least_recent_cursor(&mut self) -> Cursor<'_, K> {
        let tail = self.tail;
        Cursor { lru: self, current: tail }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_pop() {
        let mut lru: Lru<i32> = Lru::new();
        assert_eq!(lru.pop_least_recent(), None);
    }

    #[test]
    fn test_single_insert_and_pop() {
        let mut lru = Lru::new();
        lru.hit(&1);
        assert_eq!(lru.pop_least_recent(), Some(1));
        assert_eq!(lru.pop_least_recent(), None);
    }

    #[test]
    fn test_lru_order() {
        let mut lru = Lru::new();
        lru.hit(&1);
        lru.hit(&2);
        lru.hit(&3);

        // Order: head -> 3 -> 2 -> 1 -> tail
        // Pop should return 1 (least recently used)
        assert_eq!(lru.pop_least_recent(), Some(1));
        assert_eq!(lru.pop_least_recent(), Some(2));
        assert_eq!(lru.pop_least_recent(), Some(3));
        assert_eq!(lru.pop_least_recent(), None);
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
        assert_eq!(lru.pop_least_recent(), Some(2));
        assert_eq!(lru.pop_least_recent(), Some(3));
        assert_eq!(lru.pop_least_recent(), Some(1));
    }

    #[test]
    fn test_hit_existing_head() {
        let mut lru = Lru::new();
        lru.hit(&1);
        lru.hit(&2);

        // Hit 2 (already head), should not change order
        lru.hit(&2);

        assert_eq!(lru.pop_least_recent(), Some(1));
        assert_eq!(lru.pop_least_recent(), Some(2));
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
        assert_eq!(lru.pop_least_recent(), Some(1));
        assert_eq!(lru.pop_least_recent(), Some(3));
        assert_eq!(lru.pop_least_recent(), Some(2));
    }

    #[test]
    fn test_reuse_free_slots() {
        let mut lru = Lru::new();
        lru.hit(&1);
        lru.hit(&2);

        // Pop and insert new element
        assert_eq!(lru.pop_least_recent(), Some(1));
        lru.hit(&3);

        // Free slot from 1 should be reused
        assert!(lru.entries.len() <= 2);

        assert_eq!(lru.pop_least_recent(), Some(2));
        assert_eq!(lru.pop_least_recent(), Some(3));
    }

    #[test]
    fn test_string_keys() {
        let mut lru = Lru::new();
        lru.hit(&"hello".to_string());
        lru.hit(&"world".to_string());

        lru.hit(&"hello".to_string());

        assert_eq!(lru.pop_least_recent(), Some("world".to_string()));
        assert_eq!(lru.pop_least_recent(), Some("hello".to_string()));
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
        while let Some(val) = lru.pop_least_recent() {
            popped.push(val);
        }

        assert_eq!(popped.len(), 100);
        // Last two should be 50 and 0 (most recently hit)
        assert_eq!(popped[98], 0);
        assert_eq!(popped[99], 50);
    }

    #[test]
    fn test_push_to_least_recently_used() {
        let mut lru = Lru::new();
        lru.hit(&1);
        lru.hit(&2);
        lru.hit(&3);

        // Insert new element to LRU: should become 0
        lru.push_to_least_recently_used(&0);

        // Order: head -> 3 -> 2 -> 1 -> 0 -> tail

        assert_eq!(lru.pop_least_recent(), Some(0));
        assert_eq!(lru.pop_least_recent(), Some(1));
        assert_eq!(lru.pop_least_recent(), Some(2));
        assert_eq!(lru.pop_least_recent(), Some(3));
    }

    #[test]
    fn test_push_to_least_recent_moves_existing() {
        let mut lru = Lru::new();
        lru.hit(&1);
        lru.hit(&2);
        lru.hit(&3);
        // Order: head -> 3 -> 2 -> 1 -> tail

        // Move 3 (head) to tail
        lru.push_to_least_recently_used(&3);
        // Order: head -> 2 -> 1 -> 3 -> tail

        assert_eq!(lru.pop_least_recent(), Some(3));
        assert_eq!(lru.pop_least_recent(), Some(1));
        assert_eq!(lru.pop_least_recent(), Some(2));

        // Test middle move
        let mut lru = Lru::new();
        lru.hit(&1);
        lru.hit(&2);
        lru.hit(&3);
        // Order: head -> 3 -> 2 -> 1 -> tail

        // Move 2 (middle) to tail
        lru.push_to_least_recently_used(&2);
        // Order: head -> 3 -> 1 -> 2 -> tail
        assert_eq!(lru.pop_least_recent(), Some(2));
        assert_eq!(lru.pop_least_recent(), Some(1));
        assert_eq!(lru.pop_least_recent(), Some(3));

        // Test tail move (no-op logically, but should remain tail)
        let mut lru = Lru::new();
        lru.hit(&1);
        lru.hit(&2);
        lru.hit(&3);
        // Order: head -> 3 -> 2 -> 1 -> tail

        lru.push_to_least_recently_used(&1);
        // Order: head -> 3 -> 2 -> 1 -> tail
        assert_eq!(lru.pop_least_recent(), Some(1));
        assert_eq!(lru.pop_least_recent(), Some(2));
        assert_eq!(lru.pop_least_recent(), Some(3));
    }
}

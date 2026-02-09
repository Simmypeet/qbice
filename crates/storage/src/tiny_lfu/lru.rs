use std::{collections::HashMap, ptr::NonNull};

use fxhash::FxBuildHasher;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Region {
    Window = 0,
    Probation = 1,
    Protected = 2,
    Pinned = 3,
}

struct Node<K> {
    key: K,
    prev: Option<NonNull<Self>>,
    next: Option<NonNull<Self>>,
}

impl<K> Node<K> {
    #[allow(clippy::unnecessary_box_returns)]
    fn new(key: K) -> Box<Self> {
        Box::new(Self { key, prev: None, next: None })
    }
}

struct LruList<K> {
    // Windows, Protected, Probation, Pinned
    heads: [Option<NonNull<Node<K>>>; 4],
    tails: [Option<NonNull<Node<K>>>; 4],
    lens: [usize; 4],
}

// Safety: We properly manage the Node pointers and ensure no data races.
// Nodes are only accessed through &mut self methods, ensuring exclusive access.
unsafe impl<K: Send> Send for LruList<K> {}
unsafe impl<K: Sync> Sync for LruList<K> {}

impl<K> LruList<K> {
    const fn new() -> Self {
        Self { heads: [None; 4], tails: [None; 4], lens: [0; 4] }
    }

    // Connects a node to the head of a specific region
    const fn push_head(
        &mut self,
        mut node_ptr: NonNull<Node<K>>,
        region: Region,
    ) {
        let r_idx = region as usize;

        unsafe {
            let node = node_ptr.as_mut();
            node.prev = None;
            node.next = self.heads[r_idx];

            // Update the old head's prev pointer
            if let Some(mut old_head) = self.heads[r_idx] {
                old_head.as_mut().prev = Some(node_ptr);
            }

            // Update the struct's pointers
            self.heads[r_idx] = Some(node_ptr);
            if self.tails[r_idx].is_none() {
                self.tails[r_idx] = Some(node_ptr);
            }
        }
    }

    // Removes a node from its current region
    const fn unlink(&mut self, node_ptr: NonNull<Node<K>>, region: Region) {
        let r_idx = region as usize;

        unsafe {
            let node = node_ptr.as_ref();
            let prev = node.prev;
            let next = node.next;

            // Update the previous node's next pointer
            if let Some(mut prev_ptr) = prev {
                prev_ptr.as_mut().next = next;
            } else {
                // We're removing the head
                self.heads[r_idx] = next;
            }

            // Update the next node's prev pointer
            if let Some(mut next_ptr) = next {
                next_ptr.as_mut().prev = prev;
            } else {
                // We're removing the tail
                self.tails[r_idx] = prev;
            }
        }
    }

    fn move_to_head(&mut self, mut node_ptr: NonNull<Node<K>>, region: Region) {
        let r_idx = region as usize;

        // Check if already at head
        if self.heads[r_idx] == Some(node_ptr) {
            return;
        }

        unsafe {
            let node = node_ptr.as_mut();
            let prev = node.prev;
            let next = node.next;

            // Unlink from current position
            if let Some(mut prev_ptr) = prev {
                prev_ptr.as_mut().next = next;
            }

            if let Some(mut next_ptr) = next {
                next_ptr.as_mut().prev = prev;
            }

            // Update tail if this was the tail
            if self.tails[r_idx] == Some(node_ptr) {
                self.tails[r_idx] = prev;
            }

            // Move to head
            node.prev = None;
            node.next = self.heads[r_idx];

            if let Some(mut old_head) = self.heads[r_idx] {
                old_head.as_mut().prev = Some(node_ptr);
            }

            self.heads[r_idx] = Some(node_ptr);
        }
    }

    fn pop_least_recent(&mut self, region: Region) -> Option<Box<Node<K>>> {
        let r_idx = region as usize;
        let tail_ptr = self.tails[r_idx]?;

        unsafe {
            let node = tail_ptr.as_ref();
            let prev = node.prev;

            if let Some(mut prev_ptr) = prev {
                prev_ptr.as_mut().next = None;
                self.tails[r_idx] = Some(prev_ptr);
            } else {
                // We're removing the only node
                self.heads[r_idx] = None;
                self.tails[r_idx] = None;
            }

            // Decrease length
            self.lens[r_idx] -= 1;

            // Convert back to Box to properly deallocate
            Some(Box::from_raw(tail_ptr.as_ptr()))
        }
    }
}

impl<K> Drop for LruList<K> {
    fn drop(&mut self) {
        // Clean up all nodes in all regions
        for region in [
            Region::Window,
            Region::Probation,
            Region::Protected,
            Region::Pinned,
        ] {
            while self.pop_least_recent(region).is_some() {}
        }
    }
}

pub struct Lru<K> {
    list: LruList<K>,
    map: HashMap<K, (NonNull<Node<K>>, Region), FxBuildHasher>,
}

// Safety: We properly manage the Node pointers through Box allocation.
// Access is controlled through &mut self methods, ensuring no data races.
unsafe impl<K: Send> Send for Lru<K> {}
unsafe impl<K: Sync> Sync for Lru<K> {}

impl<K> Lru<K> {
    pub fn new() -> Self {
        Self { map: HashMap::default(), list: LruList::new() }
    }
}

impl<K: std::hash::Hash + Eq + Clone> Lru<K> {
    pub fn hit(&mut self, key: &K, protected_capacity: usize) -> bool {
        if let Some((node_ptr, region)) = self.map.get_mut(key) {
            match region {
                // On a hit, move the node to the head of its region
                Region::Window => self.list.move_to_head(*node_ptr, *region),

                // promote to protected
                Region::Probation => {
                    // move to protected
                    self.list.unlink(*node_ptr, Region::Probation);
                    self.list.push_head(*node_ptr, Region::Protected);

                    // update lengths and region
                    self.list.lens[Region::Probation as usize] -= 1;
                    self.list.lens[Region::Protected as usize] += 1;
                    *region = Region::Protected;

                    // if protected over capacity, demote LRU to probation
                    if self.list.lens[Region::Protected as usize]
                        > protected_capacity
                        && let Some(lru_ptr) =
                            self.list.tails[Region::Protected as usize]
                    {
                        self.list.unlink(lru_ptr, Region::Protected);
                        self.list.push_head(lru_ptr, Region::Probation);

                        // Get the key of the LRU node
                        let lru_key = unsafe { &lru_ptr.as_ref().key };
                        let (_, lru_region) =
                            self.map.get_mut(lru_key).unwrap();

                        // update lengths
                        self.list.lens[Region::Protected as usize] -= 1;
                        self.list.lens[Region::Probation as usize] += 1;
                        *lru_region = Region::Probation;
                    }
                }

                Region::Protected => {
                    self.list.move_to_head(*node_ptr, *region);
                }

                Region::Pinned => {
                    // do nothing
                }
            }

            true
        } else {
            false
        }
    }

    pub fn new_entry(&mut self, key: K, region: Region) {
        let node = Node::new(key.clone());
        let node_ptr = NonNull::from(Box::leak(node));

        assert!(
            self.map.insert(key, (node_ptr, region)).is_none(),
            "Key already exists in LRU map"
        );

        self.list.push_head(node_ptr, region);
        self.list.lens[region as usize] += 1;
    }

    pub const fn window_len(&self) -> usize {
        self.list.lens[Region::Window as usize]
    }

    pub const fn probation_len(&self) -> usize {
        self.list.lens[Region::Probation as usize]
    }

    pub const fn protected_len(&self) -> usize {
        self.list.lens[Region::Protected as usize]
    }

    pub const fn pinned_len(&self) -> usize {
        self.list.lens[Region::Pinned as usize]
    }

    pub fn peek_least_recent(&self, region: Region) -> Option<&K> {
        let tail_ptr = self.list.tails[region as usize]?;
        unsafe { Some(&tail_ptr.as_ref().key) }
    }

    pub fn pop_least_recent(&mut self, region: Region) -> Option<K> {
        let node_box = self.list.pop_least_recent(region)?;
        let key = node_box.key;

        self.map.remove(&key);

        // Shrink map size
        self.try_shrink_map();

        Some(key)
    }

    pub fn move_least_recent_of_to_new_region(
        &mut self,
        from_region: Region,
        to_region: Region,
    ) {
        let Some(tail_ptr) = self.list.tails[from_region as usize] else {
            return;
        };

        self.list.unlink(tail_ptr, from_region);
        self.list.push_head(tail_ptr, to_region);

        // Update lengths
        self.list.lens[from_region as usize] -= 1;
        self.list.lens[to_region as usize] += 1;

        let key = unsafe { &tail_ptr.as_ref().key };
        let (_, region) = self.map.get_mut(key).unwrap();
        *region = to_region;
    }

    #[allow(clippy::cast_precision_loss)]
    pub fn remove(&mut self, key: &K) -> Option<K> {
        let (node_ptr, region) = self.map.remove(key)?;

        self.list.unlink(node_ptr, region);
        self.list.lens[region as usize] -= 1;

        // Convert back to Box to properly deallocate
        let node_box = unsafe { Box::from_raw(node_ptr.as_ptr()) };

        // Shrink map size
        self.try_shrink_map();

        Some(node_box.key)
    }

    #[allow(clippy::cast_precision_loss)]
    fn try_shrink_map(&mut self) {
        let ratio = self.map.len() as f32 / self.map.capacity() as f32;
        if ratio < 0.25 {
            self.map.shrink_to(self.map.capacity() / 2);
        }
    }

    pub fn check_is_in_region(&self, key: &K, region: Region) -> bool {
        if let Some((_, found)) = self.map.get(key) {
            *found == region
        } else {
            false
        }
    }

    pub fn shuffle_tail_to_head(&mut self, region: Region) {
        if let Some(tail_ptr) = self.list.tails[region as usize] {
            self.list.move_to_head(tail_ptr, region);
        }
    }

    pub fn move_key_to_head_of_region(&mut self, key: &K, new_region: Region) {
        let (node_ptr, region) = self.map.get_mut(key).unwrap();

        assert!(
            *region != new_region,
            "Key is already in the specified region"
        );

        self.list.unlink(*node_ptr, *region);
        self.list.push_head(*node_ptr, new_region);

        self.list.lens[new_region as usize] += 1;
        self.list.lens[*region as usize] -= 1;

        *region = new_region;
    }
}

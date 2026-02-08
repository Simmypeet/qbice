use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Region {
    Window = 0,
    Probation = 1,
    Protected = 2,
    Pinned = 3,
}

pub type Index = u32;
pub const NULL: Index = u32::MAX;

pub struct Entry<K> {
    pub key: K,
    pub prev: Index,
    pub next: Index,
}

struct LruList<K> {
    nodes: slab::Slab<Entry<K>>,

    // Windows, Protected, Probation, Pinned
    heads: [Index; 4],
    tails: [Index; 4],
    lens: [usize; 4],
}

impl<K> LruList<K> {
    pub const fn new() -> Self {
        Self {
            nodes: slab::Slab::new(),
            heads: [NULL; 4],
            tails: [NULL; 4],
            lens: [0; 4],
        }
    }

    // Connects a node to the head of a specific region
    fn push_head(&mut self, index: u32, region: Region) {
        let r_idx = region as usize;
        let index_usize = index as usize;

        let old_head = self.heads[r_idx];

        // Update the node's pointers
        let node = &mut self.nodes[index_usize];
        node.next = old_head;
        node.prev = NULL;

        // Update the old head's prev pointer
        if old_head != NULL {
            self.nodes[old_head as usize].prev = index;
        }

        // Update the struct's pointers
        self.heads[r_idx] = index;
        if self.tails[r_idx] == NULL {
            self.tails[r_idx] = index;
        }
    }

    // Removes a node from its current region
    fn unlink(&mut self, index: u32, region: Region) {
        let index_usize = index as usize;
        let node = &self.nodes[index_usize];

        let prev = node.prev;
        let next = node.next;

        // Update the previous node's next pointer
        if prev == NULL {
            // We're removing the head
            self.heads[region as usize] = next;
        } else {
            self.nodes[prev as usize].next = next;
        }

        // Update the next node's prev pointer
        if next == NULL {
            // We're removing the tail
            self.tails[region as usize] = prev;
        } else {
            self.nodes[next as usize].prev = prev;
        }
    }

    fn move_to_head(&mut self, index: u32, region: Region) {
        let idx = index as usize;
        let region_idx = region as usize;
        if self.heads[region_idx] == index {
            return; // Already at head
        }

        let entry = &mut self.nodes[idx];
        let prev = entry.prev;
        let next = entry.next;

        // Unlink from current position
        if prev != NULL {
            self.nodes[prev as usize].next = next;
        }

        if next != NULL {
            self.nodes[next as usize].prev = prev;
        }

        // Update tail if this was the tail
        if self.tails[region_idx] == index {
            self.tails[region_idx] = prev;
        }

        // Move to head
        let old_head = self.heads[region_idx];
        self.heads[region_idx] = index;

        let entry = &mut self.nodes[idx];
        entry.prev = NULL;
        entry.next = old_head;

        if old_head != NULL {
            self.nodes[old_head as usize].prev = index;
        }
    }

    fn pop_least_recent(&mut self, region: Region) -> Option<K> {
        let tail_index = self.tails[region as usize];
        if tail_index == NULL {
            return None;
        }

        let entry = self.nodes.remove(tail_index as usize);
        let key = entry.key;

        if entry.prev == NULL {
            // We're removing the only node
            self.heads[region as usize] = NULL;
            self.tails[region as usize] = NULL;
        } else {
            self.nodes[entry.prev as usize].next = NULL;
            self.tails[region as usize] = entry.prev;
        }

        // Decrease length
        self.lens[region as usize] -= 1;

        Some(key)
    }
}

pub struct Lru<K> {
    list: LruList<K>,
    map: HashMap<K, (Index, Region)>,
}

impl<K> Entry<K> {
    pub const fn new(key: K) -> Self { Self { key, prev: NULL, next: NULL } }
}

impl<K> Lru<K> {
    pub fn new() -> Self { Self { map: HashMap::new(), list: LruList::new() } }
}

impl<K: std::hash::Hash + Eq + Clone> Lru<K> {
    pub fn hit(&mut self, key: &K, protected_capacity: usize) -> bool {
        if let Some((index, region)) = self.map.get_mut(key) {
            match region {
                // On a hit, move the node to the head of its region
                Region::Window => self.list.move_to_head(*index, *region),

                // promote to
                Region::Probation => {
                    // move to protected
                    self.list.unlink(*index, Region::Probation);
                    self.list.push_head(*index, Region::Protected);

                    // update lengths and region
                    self.list.lens[Region::Probation as usize] -= 1;
                    self.list.lens[Region::Protected as usize] += 1;
                    *region = Region::Protected;

                    // if protected over capacity, demote LRU to probation
                    if self.list.lens[Region::Protected as usize]
                        > protected_capacity
                    {
                        let lru_index =
                            self.list.tails[Region::Protected as usize];

                        if self.list.tails[Region::Protected as usize] != NULL {
                            self.list.unlink(lru_index, Region::Protected);
                            self.list.push_head(lru_index, Region::Probation);

                            let (_, lru_region) = self
                                .map
                                .get_mut(
                                    &self.list.nodes[lru_index as usize].key,
                                )
                                .unwrap();

                            // update lengths
                            self.list.lens[Region::Protected as usize] -= 1;
                            self.list.lens[Region::Probation as usize] += 1;
                            *lru_region = Region::Probation;
                        }
                    }
                }

                Region::Protected => {
                    self.list.move_to_head(*index, *region);
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

    #[allow(clippy::cast_possible_truncation)]
    pub fn new_entry(&mut self, key: K, region: Region) {
        let entry = Entry::new(key.clone());
        let index = self.list.nodes.insert(entry) as u32;

        assert!(
            self.map.insert(key, (index, region)).is_none(),
            "Key already exists in LRU map"
        );

        self.list.push_head(index, region);
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

    pub fn peek_least_recent(&self, region: Region) -> Option<&K> {
        let tail_index = self.list.tails[region as usize];
        if tail_index == NULL {
            None
        } else {
            Some(&self.list.nodes[tail_index as usize].key)
        }
    }

    pub fn pop_least_recent(&mut self, region: Region) -> Option<K> {
        let entry = self.list.pop_least_recent(region);

        if let Some(ref key) = entry {
            self.map.remove(key);
        }

        entry
    }

    pub fn move_least_recent_of_to_new_region(
        &mut self,
        from_region: Region,
        to_region: Region,
    ) {
        let tail_index = self.list.tails[from_region as usize];
        if tail_index == NULL {
            return;
        }

        self.list.unlink(tail_index, from_region);
        self.list.push_head(tail_index, to_region);

        // Update lengths
        self.list.lens[from_region as usize] -= 1;
        self.list.lens[to_region as usize] += 1;

        let key = &self.list.nodes[tail_index as usize].key;
        let (_, region) = self.map.get_mut(key).unwrap();

        *region = to_region;
    }

    pub fn remove(&mut self, key: &K) -> Option<K> {
        if let Some((index, region)) = self.map.remove(key) {
            self.list.unlink(index, region);
            self.list.lens[region as usize] -= 1;
            let entry = self.list.nodes.remove(index as usize);
            Some(entry.key)
        } else {
            None
        }
    }
}

//Tree's cache cut into shards by the key's top bits, each behind its own lock; an answer is copied out, not borrowed
use crate::search::Cache;
use std::sync::{Mutex, MutexGuard};

const SHARD_BITS: u32 = 6;

pub struct SharedCache {
    shards: Box<[Mutex<Cache>]>,
}

impl SharedCache {
    //room for `positions` answers in all
    pub fn new(positions: usize) -> SharedCache {
        let shards = 1usize << SHARD_BITS;
        SharedCache { shards: (0..shards).map(|_| Mutex::new(Cache::new(positions.div_ceil(shards)))).collect() }
    }

    #[cfg(test)]
    pub fn from_cache(cache: &Cache, positions: usize) -> SharedCache {
        let shared = SharedCache::new(positions);
        for generation in [&cache.older, &cache.newer] {
            for (&key, &(start, count, value)) in &generation.places {
                shared.insert(key, &generation.priors[start..start + usize::from(count)], value);
            }
        }
        shared
    }

    fn shard(&self, key: u64) -> MutexGuard<'_, Cache> {
        self.shards[(key >> (64 - SHARD_BITS)) as usize].lock().expect("a cache shard")
    }

    //copies the priors into `priors`, returns the value
    pub fn get(&self, key: u64, priors: &mut Vec<f32>) -> Option<f32> {
        let shard = self.shard(key);
        let answer = shard.get(key)?;
        priors.clear();
        priors.extend_from_slice(answer.priors);
        Some(answer.value)
    }

    pub fn insert(&self, key: u64, priors: &[f32], value: f32) {
        self.shard(key).insert(key, priors, value);
    }

    pub fn clear(&self) {
        for shard in &self.shards {
            shard.lock().expect("a cache shard").clear();
        }
    }
}

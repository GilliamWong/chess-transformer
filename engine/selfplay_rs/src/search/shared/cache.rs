//the network's answers by position for many threads: Tree's cache cut into shards by the key's top bits, each behind a
//lock of its own, and an answer copied out rather than borrowed
use crate::search::Cache;
use std::sync::Mutex;

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

    //a Tree's cache, answer for answer, with room for `positions`
    pub fn from_cache(cache: &Cache, positions: usize) -> SharedCache {
        let shared = SharedCache::new(positions);
        for generation in [&cache.older, &cache.newer] {
            for (&key, &(start, count, value)) in &generation.places {
                shared.insert(key, &generation.priors[start..start + usize::from(count)], value);
            }
        }
        shared
    }

    fn shard(&self, key: u64) -> std::sync::MutexGuard<'_, Cache> {
        self.shards[(key >> (64 - SHARD_BITS)) as usize].lock().expect("a cache shard")
    }

    //the position's priors, copied into `priors`, and its value
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

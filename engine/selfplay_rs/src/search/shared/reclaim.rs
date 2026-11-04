//what a move leaves dead, freed beside the next search (see the references in shared.rs)
use super::*;
use std::sync::{Arc, Weak};

//the nodes a sweeper or drain frees before it looks again, and the garbage it takes at a time
const SWEEP_BATCH: usize = 256;
const GARBAGE_TAKEN: usize = 256;

#[derive(Default)]
pub struct Garbage {
    //nodes with no reference left, whose children still hold the counts of their moves
    pub(super) dead: Vec<u32>,
    //the first links of lists that no node has any more
    pub(super) lists: Vec<u32>,
    //threads in the middle of freeing some. They are counted under the garbage's lock, as they take and leave their batch, so that
    //"none left and none being freed" is one look
    working: usize,
}

//what a batch frees, given back to the pools together
#[derive(Default)]
struct Freed {
    nodes: Vec<u32>,
    links: Vec<u32>,
}

impl SharedTree {
    //a move's node is no longer named by it
    pub(super) fn drop_reference(&self, id: usize) {
        if self.nodes.get(id).release() {
            self.garbage.lock().expect("the garbage").dead.push(id as u32);
        }
    }

    pub fn has_garbage(&self) -> bool {
        let garbage = self.garbage.lock().expect("the garbage");
        !garbage.dead.is_empty() || !garbage.lists.is_empty()
    }

    //no garbage is left and no thread is freeing any: all that was freed is given back to the pools
    pub fn idle(&self) -> bool {
        let garbage = self.garbage.lock().expect("the garbage");
        garbage.dead.is_empty() && garbage.lists.is_empty() && garbage.working == 0
    }

    //a batch of the garbage for the thread that takes it, who then counts as working; none if there is none
    pub fn take_garbage(&self) -> Option<Garbage> {
        let mut garbage = self.garbage.lock().expect("the garbage");
        if garbage.dead.is_empty() && garbage.lists.is_empty() {
            return None;
        }
        garbage.working += 1;
        let dead_from = garbage.dead.len().saturating_sub(GARBAGE_TAKEN);
        let lists_from = garbage.lists.len().saturating_sub(GARBAGE_TAKEN);
        Some(Garbage { dead: garbage.dead.split_off(dead_from), lists: garbage.lists.split_off(lists_from), working: 0 })
    }

    //frees up to `budget` nodes of the batch taken and returns what it freed. What is left of the batch goes back to the
    //garbage, and the thread stops working, once the room is given back to the pools
    pub fn free_garbage(&self, taken: Garbage, budget: usize) -> usize {
        let Garbage { mut dead, lists, .. } = taken;
        let mut freed = Freed::default();
        let mut nodes = 0;
        while nodes < budget
            && let Some(id) = dead.pop()
        {
            self.free_node(id, &mut dead, &mut freed);
            nodes += 1;
        }
        for &head in &lists {
            self.links.collect_list(head, &mut freed.links);
        }
        self.give_back(&mut freed);
        self.counts.reclaimed.fetch_add(nodes as u64, Ordering::Relaxed);
        let mut garbage = self.garbage.lock().expect("the garbage");
        garbage.dead.append(&mut dead);
        garbage.working -= 1;
        nodes + lists.len()
    }

    //frees up to `budget` nodes of the garbage and returns what it freed: none when there is none for this thread, though
    //another may be freeing some
    pub fn reclaim(&self, budget: usize) -> usize {
        self.take_garbage().map_or(0, |taken| self.free_garbage(taken, budget))
    }

    fn give_back(&self, freed: &mut Freed) {
        self.nodes.free(&mut freed.nodes);
        self.links.free(&mut freed.links);
    }

    //a dead node's table entry, children's counts and room, in this order, the node itself with the batch's rest. Children
    //that die with it go on `dead`
    fn free_node(&self, id: u32, dead: &mut Vec<u32>, freed: &mut Freed) {
        let node = self.nodes.get(id as usize);
        if let Some(graph) = &self.graph {
            graph.remove_if(graph_key(node.key(), u32::from(node.halfmoves())), id);
        }
        for (_, link) in self.links.linked(node) {
            let child = self.links.child(link);
            if self.nodes.get(child as usize).release() {
                dead.push(child);
            }
        }
        let edges = node.edges();
        match node.children() {
            Children::List(head) => self.links.collect_list(head, &mut freed.links),
            Children::Solid(first) => self.links.free_slots(first, edges.len()),
        }
        if !edges.is_empty() {
            self.edges.free(edges.start, edges.len());
        }
        freed.nodes.push(id);
    }

    //frees all the garbage, and waits for the other threads freeing some
    pub fn drain(&self) {
        while !self.idle() {
            if self.reclaim(SWEEP_BATCH) == 0 {
                std::thread::yield_now();
            }
        }
    }

    //threads that free the garbage until it is gone or the tree is closed, each leaving as soon as the tree has no other owner
    pub fn start_sweepers(self: &Arc<Self>, threads: usize) {
        if !self.has_garbage() {
            return;
        }
        let mut sweepers = self.sweepers.lock().expect("the sweepers");
        for _ in 0..threads {
            let tree = Arc::downgrade(self);
            if let Ok(sweeper) = std::thread::Builder::new().name("sweeper".to_string()).spawn(move || sweep(tree)) {
                sweepers.push(sweeper);
            }
        }
    }

    //the sweepers, for a test to wait on after the tree is dropped
    #[cfg(test)]
    pub fn take_sweepers(&self) -> Vec<JoinHandle<()>> {
        std::mem::take(&mut *self.sweepers.lock().expect("the sweepers"))
    }

    //waits for the sweepers to leave: they do once the tree is closed
    pub fn join_sweepers(&self) {
        let sweepers = std::mem::take(&mut *self.sweepers.lock().expect("the sweepers"));
        for sweeper in sweepers {
            sweeper.join().expect("a sweeper");
        }
    }

    //the tree's lists made solid since the last move are no longer read
    pub(super) fn release_retired_lists(&mut self) {
        let retired = std::mem::take(self.retired.get_mut().expect("the retired lists"));
        self.garbage.get_mut().expect("the garbage").lists.extend(retired);
    }
}

fn sweep(tree: Weak<SharedTree>) {
    while let Some(tree) = tree.upgrade() {
        if tree.closed.load(Ordering::Acquire) {
            return;
        }
        if tree.reclaim(SWEEP_BATCH) == 0 {
            if tree.idle() {
                return;
            }
            std::thread::yield_now();
        }
    }
}

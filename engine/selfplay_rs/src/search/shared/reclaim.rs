//what a move leaves dead, freed beside the next search (see the references in shared.rs)
use super::*;
use std::sync::{Arc, Weak};

//the nodes a sweeper or drain frees before it looks again, and the garbage it takes at a time
const SWEEP_BATCH: usize = 256;
const GARBAGE_TAKEN: usize = 256;
//a root is held only by a tree that takes no more than one part in this many of the memory limit
const HOLDING_SHARE: usize = 2;

#[derive(Default)]
pub struct Garbage {
    //nodes with no reference left, whose children still hold the counts of their moves
    pub(super) dead: Vec<u32>,
    //the first links of lists that no node has any more
    pub(super) lists: Vec<u32>,
    //threads freeing some, counted under this lock as they take and leave their batch: "none left and none freeing" is one look
    working: usize,
    //the root the last line was played from, if it was held (hold_root): it has the count it had as the root, and is no garbage yet
    pub(super) held: Option<u32>,
    //the root is held when a line is next played from it
    pub(super) holds_next: bool,
}

impl Garbage {
    fn idle(&self) -> bool {
        self.dead.is_empty() && self.lists.is_empty() && self.working == 0
    }
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

    //Tree::hold_root, with transpositions and before a line is played from the root: the root keeps its count when the line is
    //played (move_root), and with it the rest of the tree stays alive and in the table, where it would be dead: a move that comes
    //to one of its positions takes the node with its visits, as it takes any node other paths made. It is held for one move:
    //the next line played gives it up (release_held), and so does a descent short of room, sooner. What the search took of the
    //rest stays, as tree: so under `memory_limit` (bytes, usize::MAX for none) nothing is held by a tree that takes more than
    //one part in HOLDING_SHARE of it, which leaves the next search the room of that tree again whatever it takes, nor once the
    //arenas have grown to the limit: the room a move frees is then the search's
    pub fn hold_root(&mut self, memory_limit: usize) {
        let fits = memory_limit == usize::MAX || self.allocated_bytes() <= memory_limit / HOLDING_SHARE;
        let holds = self.graph.is_some() && fits && !self.gives_out_freed_room(memory_limit);
        self.garbage.get_mut().expect("the garbage").holds_next = holds;
    }

    //the held root's count dropped: it is dead with what it alone reaches, for the sweepers and descents to free. False if none
    //was held
    pub fn release_held(&self) -> bool {
        self.give_up_held(&mut self.garbage.lock().expect("the garbage"))
    }

    //release_held under the garbage's lock: the root is held no more and dead in one step, for whoever looks at the garbage
    fn give_up_held(&self, garbage: &mut Garbage) -> bool {
        let Some(id) = garbage.held.take() else { return false };
        if self.nodes.get(id as usize).release() {
            garbage.dead.push(id);
        }
        true
    }

    pub fn has_garbage(&self) -> bool {
        let garbage = self.garbage.lock().expect("the garbage");
        !garbage.dead.is_empty() || !garbage.lists.is_empty()
    }

    //no garbage is left and no thread is freeing any: all that was freed is given back to the pools
    pub(super) fn idle(&self) -> bool {
        self.garbage.lock().expect("the garbage").idle()
    }

    //for a descent short of room that found no garbage to take: whether some is still to be freed. Garbage is left or a thread
    //is freeing some; or a root is held, which is given up here for its room. One look under the garbage's lock: in two, a
    //thread could find the garbage idle before another gave up the held root and no root held after, and call the tree full
    //beside all that the root kept
    pub(super) fn room_still_to_free(&self) -> bool {
        let mut garbage = self.garbage.lock().expect("the garbage");
        !garbage.idle() || self.give_up_held(&mut garbage)
    }

    //a batch of the garbage for the thread that takes it, who then counts as working; none if there is none
    fn take_garbage(&self) -> Option<(Vec<u32>, Vec<u32>)> {
        let mut garbage = self.garbage.lock().expect("the garbage");
        if garbage.dead.is_empty() && garbage.lists.is_empty() {
            return None;
        }
        garbage.working += 1;
        let dead_from = garbage.dead.len().saturating_sub(GARBAGE_TAKEN);
        let lists_from = garbage.lists.len().saturating_sub(GARBAGE_TAKEN);
        Some((garbage.dead.split_off(dead_from), garbage.lists.split_off(lists_from)))
    }

    //frees up to `budget` nodes of the batch and returns what it freed; the rest goes back to the garbage, once the room is given back
    fn free_garbage(&self, (mut dead, lists): (Vec<u32>, Vec<u32>), budget: usize) -> usize {
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

    //frees up to `budget` nodes of the garbage: none when there is none to take, though another thread may be freeing some
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

    //frees all the garbage, and waits for the other threads freeing some: what the sweepers would free
    pub fn sweep_now(&self) {
        while !self.idle() {
            if self.reclaim(SWEEP_BATCH) == 0 {
                std::thread::yield_now();
            }
        }
    }

    //frees all that the root does not reach: the garbage, and what a held root kept
    #[cfg(test)]
    pub fn drain(&self) {
        self.release_held();
        self.sweep_now();
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

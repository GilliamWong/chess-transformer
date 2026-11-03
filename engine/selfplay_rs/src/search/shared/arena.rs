use crate::rules::PackedMove;
use crate::search::{NO_LINK, NO_NODE, Outcome, Proof, SOLID, listed};
use shakmaty::Move;
use std::ops::Range;
use std::ptr::{null_mut, slice_from_raw_parts_mut};
use std::sync::atomic::{AtomicPtr, AtomicU16, AtomicU32, AtomicU64, AtomicUsize, Ordering};

const NODE_CHUNK_BITS: u32 = if cfg!(test) { 6 } else { 16 };
const NODE_CHUNKS: usize = 1 << 14;
const EDGE_CHUNK_BITS: u32 = if cfg!(test) { 10 } else { 20 };
const EDGE_CHUNKS: usize = 1 << 15;
//a chunk holds a solid node's slots (218 moves at most); links and slots stay below 1 << 31, which SOLID takes
const LINK_CHUNK_BITS: u32 = if cfg!(test) { 8 } else { 18 };
const LINK_CHUNKS: usize = 1 << 13;

const IN_FLIGHT: u64 = 1 << 32;

//the orderings are the tree's protocol (shared.rs)
#[repr(align(64))]
#[derive(Default)]
pub struct Node {
    count: AtomicU64, //visits in the low 32 bits, simulations in flight in the high 32
    value_sum: AtomicU64,
    visited_policy_sqrt: AtomicU64,
    edges_at: AtomicU64, //the first edge above the low 8 bits, the number of edges in them; 0 until expanded
    proof_value: AtomicU64,
    proof_meta: AtomicU64, //0 without a proof, else the plies above the low 2 bits and the outcome in them (1 loss, 2 draw, 3 win)
    key: AtomicU64,
    refreshes: AtomicU16, //begun, wrapping
    halfmoves: AtomicU16, //with transpositions the node's fifty-move count
    children: AtomicU32,  //a list's first link, or once solid its first slot (SOLID set): see Children
}

//a node's moves with a node: a list of links in the moves' order, or once solid one slot a move
#[derive(Clone, Copy)]
pub enum Children {
    List(u32),
    Solid(u32),
}

impl Node {
    pub fn counts(&self) -> (u32, u32) {
        let count = self.count.load(Ordering::Acquire);
        (count as u32, (count >> 32) as u32)
    }

    pub fn visits(&self) -> u32 {
        self.count.load(Ordering::Acquire) as u32
    }

    //one more simulation in flight; returns the (visits, in flight) before
    pub fn bump(&self) -> (u32, u32) {
        let before = self.count.fetch_add(IN_FLIGHT, Ordering::AcqRel);
        (before as u32, (before >> 32) as u32)
    }

    pub fn take_off(&self, in_flight: u32) {
        let before = self.count.fetch_sub(u64::from(in_flight) << 32, Ordering::Release);
        debug_assert!((before >> 32) as u32 >= in_flight, "more simulations taken off than were in flight");
    }

    //one more visit, with `in_flight` simulations off the count; returns the visits before
    pub fn visit(&self, in_flight: u32) -> u32 {
        let before = self.count.fetch_add(1u64.wrapping_sub(u64::from(in_flight) << 32), Ordering::AcqRel);
        debug_assert!((before >> 32) as u32 >= in_flight, "more simulations backed up than were in flight");
        before as u32
    }

    //for a node of a tree being copied, before anything else sees it
    #[cfg(test)]
    pub fn set_counts(&self, visits: u32, in_flight: u32, value_sum: f64) {
        self.count.store(u64::from(in_flight) << 32 | u64::from(visits), Ordering::Relaxed);
        self.value_sum.store(value_sum.to_bits(), Ordering::Relaxed);
    }

    pub fn take_visits(&self, visits: u32) {
        self.count.fetch_sub(u64::from(visits), Ordering::Relaxed);
    }

    //the node as if never visited, its key kept
    pub fn reset(&self) {
        for field in [&self.count, &self.value_sum, &self.visited_policy_sqrt, &self.edges_at, &self.proof_value, &self.proof_meta] {
            field.store(0, Ordering::Relaxed);
        }
        self.children.store(NO_LINK, Ordering::Relaxed);
    }

    //the numbers, key and proof (not the edges or links) copied to a node of a tree being made
    pub fn copy_to(&self, target: &Node) {
        target.count.store(self.count.load(Ordering::Relaxed), Ordering::Relaxed);
        target.value_sum.store(self.value_sum.load(Ordering::Relaxed), Ordering::Relaxed);
        target.visited_policy_sqrt.store(self.visited_policy_sqrt.load(Ordering::Relaxed), Ordering::Relaxed);
        target.proof_value.store(self.proof_value.load(Ordering::Relaxed), Ordering::Relaxed);
        target.proof_meta.store(self.proof_meta.load(Ordering::Relaxed), Ordering::Relaxed);
        target.key.store(self.key.load(Ordering::Relaxed), Ordering::Relaxed);
        target.halfmoves.store(self.halfmoves.load(Ordering::Relaxed), Ordering::Relaxed);
    }

    pub fn value_sum(&self) -> f64 {
        f64::from_bits(self.value_sum.load(Ordering::Relaxed))
    }

    pub fn add_value(&self, delta: f64) {
        let _ = self.value_sum.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |seen| Some((f64::from_bits(seen) + delta).to_bits()));
    }

    pub fn mean_score(&self, visits: u32) -> f64 {
        self.value_sum() / f64::from(visits)
    }

    pub fn visited_policy_sqrt(&self) -> f64 {
        f64::from_bits(self.visited_policy_sqrt.load(Ordering::Relaxed))
    }

    //sequentially consistent with the refresh counter: a store buffered past the counter's check would be lost
    pub fn set_visited_policy_sqrt(&self, value: f64) {
        self.visited_policy_sqrt.store(value.to_bits(), Ordering::SeqCst);
    }

    //returns the refreshes begun, this one included
    pub fn begin_refresh(&self) -> u16 {
        self.refreshes.fetch_add(1, Ordering::SeqCst).wrapping_add(1)
    }

    pub fn refreshes(&self) -> u16 {
        self.refreshes.load(Ordering::SeqCst)
    }

    pub fn key(&self) -> u64 {
        self.key.load(Ordering::Relaxed)
    }

    pub fn halfmoves(&self) -> u16 {
        self.halfmoves.load(Ordering::Relaxed)
    }

    //by the thread that makes the node, before its index is given out
    pub fn set_halfmoves(&self, halfmoves: u16) {
        self.halfmoves.store(halfmoves, Ordering::Relaxed);
    }

    pub fn children(&self) -> Children {
        let children = self.children.load(Ordering::Acquire);
        if listed(children) { Children::List(children) } else { Children::Solid(children) }
    }

    //the list's head, which a new first link is swapped into
    pub fn list_head(&self) -> &AtomicU32 {
        &self.children
    }

    //the moves from now on the solid slots from `first` (Links::solid's); only with the tree to itself or before it is published
    pub fn make_solid(&self, first: u32) {
        self.children.store(first, Ordering::Release);
    }

    pub fn edges(&self) -> Range<usize> {
        let edges_at = self.edges_at.load(Ordering::Acquire);
        let first = (edges_at >> 8) as usize;
        first..first + (edges_at & 255) as usize
    }

    pub fn has_edges(&self) -> bool {
        self.edges_at.load(Ordering::Acquire) & 255 > 0
    }

    //the edges, written before, become visible to those that read them
    pub fn publish_edges(&self, first: usize, count: usize) {
        self.edges_at.store((first as u64) << 8 | count as u64, Ordering::Release);
    }

    pub fn proof(&self) -> Option<Proof> {
        let meta = self.proof_meta.load(Ordering::Acquire);
        let outcome = match meta & 3 {
            0 => return None,
            1 => Outcome::Loss,
            2 => Outcome::Draw,
            _ => Outcome::Win,
        };
        Some(Proof { outcome, value: f64::from_bits(self.proof_value.load(Ordering::Relaxed)), plies: (meta >> 2) as u32 })
    }

    pub fn proves_a_loss(&self) -> bool {
        self.proof_meta.load(Ordering::Acquire) & 3 == 1
    }

    pub fn set_proof(&self, proof: Proof) {
        let outcome = match proof.outcome {
            Outcome::Loss => 1,
            Outcome::Draw => 2,
            Outcome::Win => 3,
        };
        self.proof_value.store(proof.value.to_bits(), Ordering::Relaxed);
        self.proof_meta.store(u64::from(proof.plies) << 2 | outcome, Ordering::Release);
    }

    pub fn clear_proof(&self) {
        self.proof_meta.store(0, Ordering::Release);
    }
}

//chunks of `1 << bits` elements, each made on first use by whichever thread needs it first and freed only with the table
struct Table<T> {
    chunks: Box<[AtomicPtr<T>]>,
    bits: u32,
}

impl<T> Table<T> {
    fn new(chunks: usize, bits: u32) -> Table<T> {
        Table { chunks: (0..chunks).map(|_| AtomicPtr::new(null_mut())).collect(), bits }
    }

    //every index given out is in a made chunk (the allocators make it first)
    fn chunk_of(&self, index: usize) -> *mut T {
        let chunk = self.chunks[index >> self.bits].load(Ordering::Acquire);
        assert!(!chunk.is_null(), "an index in a chunk that was never made");
        chunk
    }

    fn get(&self, index: usize) -> &T {
        &self.slice(index, 1)[0]
    }

    fn slice(&self, index: usize, len: usize) -> &[T] {
        let offset = index & ((1 << self.bits) - 1);
        assert!(offset + len <= 1 << self.bits, "elements across a chunk's end");
        // SAFETY: the chunk never moves or goes before the table does, and the assert keeps the elements inside it
        unsafe { std::slice::from_raw_parts(self.chunk_of(index).add(offset), len) }
    }

    fn ensure_chunk(&self, index: usize, make: impl FnOnce() -> Box<[T]>) {
        let slot = &self.chunks[index >> self.bits];
        if !slot.load(Ordering::Acquire).is_null() {
            return;
        }
        let fresh = Box::into_raw(make());
        assert_eq!(fresh.len(), 1 << self.bits);
        if slot.compare_exchange(null_mut(), fresh.cast(), Ordering::AcqRel, Ordering::Acquire).is_err() {
            // SAFETY: fresh came from Box::into_raw just now, and the other thread's chunk won the slot
            drop(unsafe { Box::from_raw(fresh) });
        }
    }
}

impl<T> Drop for Table<T> {
    fn drop(&mut self) {
        let len = 1 << self.bits;
        for slot in self.chunks.iter_mut() {
            let chunk = *slot.get_mut();
            if !chunk.is_null() {
                // SAFETY: a made chunk is a Box<[T]> of len elements that only this table owns, and no reference into it
                // outlives the table
                drop(unsafe { Box::from_raw(slice_from_raw_parts_mut(chunk, len)) });
            }
        }
    }
}

pub struct Nodes {
    table: Table<Node>,
    next: AtomicUsize,
}

impl Nodes {
    //below NO_NODE, and the table's
    pub const CAPACITY: usize = (NODE_CHUNKS << NODE_CHUNK_BITS) - 1;

    pub fn new() -> Nodes {
        Nodes { table: Table::new(NODE_CHUNKS, NODE_CHUNK_BITS), next: AtomicUsize::new(0) }
    }

    pub fn len(&self) -> usize {
        self.next.load(Ordering::Relaxed)
    }

    pub fn get(&self, index: usize) -> &Node {
        self.table.get(index)
    }

    //a new node with its position's key; its index is given to others only once it is published
    pub fn push(&self, key: u64) -> usize {
        let index = self.next.fetch_add(1, Ordering::Relaxed);
        assert!(index < Self::CAPACITY, "the tree is past its nodes");
        self.table.ensure_chunk(index, || (0..1 << NODE_CHUNK_BITS).map(|_| Node::default()).collect());
        self.table.get(index).key.store(key, Ordering::Relaxed);
        index
    }
}

//the first of `count` elements (at most a chunk of `1 << bits`) in one chunk, from `next` and below `capacity`
fn alloc_run(next: &AtomicUsize, count: usize, bits: u32, capacity: usize) -> usize {
    debug_assert!(count > 0 && count <= 1 << bits);
    let mut at = next.load(Ordering::Relaxed);
    let first = loop {
        let chunk_end = (at | ((1 << bits) - 1)) + 1;
        let first = if at + count <= chunk_end { at } else { chunk_end };
        match next.compare_exchange_weak(at, first + count, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break first,
            Err(now) => at = now,
        }
    };
    assert!(first + count <= capacity, "the tree is past its room");
    first
}

//a list's move with a node; written before it is swapped into its list (release) and read after its predecessor's `next` (acquire)
pub struct Link {
    child: AtomicU32,
    visits: AtomicU32, //the move's own visits (with transpositions)
    next: AtomicU32,
    place: AtomicU32, //its place among the node's moves
}

impl Link {
    fn new() -> Link {
        Link { child: AtomicU32::new(NO_NODE), visits: AtomicU32::new(0), next: AtomicU32::new(NO_LINK), place: AtomicU32::new(0) }
    }

    pub fn child(&self) -> u32 {
        self.child.load(Ordering::Acquire)
    }

    pub fn visits(&self) -> &AtomicU32 {
        &self.visits
    }

    pub fn next(&self) -> u32 {
        self.next.load(Ordering::Acquire)
    }

    pub fn place(&self) -> usize {
        self.place.load(Ordering::Relaxed) as usize
    }
}

//a solid node's move: its node (NO_NODE until it has one, which is swapped in) and its own visits
pub struct Slot {
    child: AtomicU32,
    visits: AtomicU32,
}

impl Slot {
    fn new() -> Slot {
        Slot { child: AtomicU32::new(NO_NODE), visits: AtomicU32::new(0) }
    }

    pub fn child(&self) -> u32 {
        self.child.load(Ordering::Acquire)
    }

    pub fn visits(&self) -> &AtomicU32 {
        &self.visits
    }

    //sets the move's node if it has none: Ok(child), else Err(the node it has)
    pub fn claim(&self, child: u32) -> Result<u32, u32> {
        self.child.compare_exchange(NO_NODE, child, Ordering::AcqRel, Ordering::Acquire).map(|_| child)
    }
}

pub struct Links {
    table: Table<Link>,
    next: AtomicUsize,
    slots: Table<Slot>,
    next_slot: AtomicUsize,
}

impl Links {
    const CAPACITY: usize = LINK_CHUNKS << LINK_CHUNK_BITS;

    //link 0 is NO_LINK's, never given out
    pub fn new() -> Links {
        let links = Links {
            table: Table::new(LINK_CHUNKS, LINK_CHUNK_BITS),
            next: AtomicUsize::new(1),
            slots: Table::new(LINK_CHUNKS, LINK_CHUNK_BITS),
            next_slot: AtomicUsize::new(0),
        };
        links.make(0);
        links
    }

    //the links given out so far, the gaps at chunk ends included
    pub fn len(&self) -> usize {
        self.next.load(Ordering::Relaxed)
    }

    //the slots given out so far, the gaps included
    pub fn slots(&self) -> usize {
        self.next_slot.load(Ordering::Relaxed)
    }

    fn make(&self, first: usize) {
        self.table.ensure_chunk(first, || (0..1 << LINK_CHUNK_BITS).map(|_| Link::new()).collect());
    }

    pub fn get(&self, link: u32) -> &Link {
        debug_assert!(listed(link), "a solid slot read as a link");
        self.table.get(link as usize)
    }

    pub fn slot(&self, link: u32) -> &Slot {
        self.slots.get((link & !SOLID) as usize)
    }

    pub fn solid_slots(&self, first: u32, moves: usize) -> &[Slot] {
        if moves == 0 { &[] } else { self.slots.slice((first & !SOLID) as usize, moves) }
    }

    //the move's node: NO_NODE for a solid slot without one
    pub fn child(&self, link: u32) -> u32 {
        if listed(link) { self.get(link).child() } else { self.slot(link).child() }
    }

    pub fn visits(&self, link: u32) -> &AtomicU32 {
        if listed(link) { &self.get(link).visits } else { &self.slot(link).visits }
    }

    //a list link for the move at `place` with its node, not yet in a list
    pub fn push(&self, child: u32, place: usize) -> u32 {
        let link = alloc_run(&self.next, 1, LINK_CHUNK_BITS, Self::CAPACITY);
        self.make(link);
        let made = self.table.get(link);
        made.child.store(child, Ordering::Relaxed);
        made.place.store(place as u32, Ordering::Relaxed);
        link as u32
    }

    //`count` solid slots, none with a node: returns the first, SOLID with its index
    pub fn solid(&self, count: usize) -> u32 {
        let first = alloc_run(&self.next_slot, count, LINK_CHUNK_BITS, Self::CAPACITY);
        self.slots.ensure_chunk(first, || (0..1 << LINK_CHUNK_BITS).map(|_| Slot::new()).collect());
        first as u32 | SOLID
    }

    //the link of the move at `place` in the list from `head`, or Err of the pointer a new link for it goes into (the head or
    //a link's next) and the link that pointer holds now
    pub fn find<'a>(&'a self, head: &'a AtomicU32, place: usize) -> Result<u32, (&'a AtomicU32, u32)> {
        let mut pointer = head;
        loop {
            let link = pointer.load(Ordering::Acquire);
            if link == NO_LINK {
                return Err((pointer, link));
            }
            let record = self.get(link);
            let at = record.place();
            if at > place {
                return Err((pointer, link));
            }
            if at == place {
                return Ok(link);
            }
            pointer = &record.next;
        }
    }

    //puts `link` (Links::push's) into its list where `find` said, between `pointer` and the link `at` it held: false if the
    //pointer holds another link now
    pub fn insert(&self, pointer: &AtomicU32, at: u32, link: u32) -> bool {
        self.get(link).next.store(at, Ordering::Relaxed);
        pointer.compare_exchange(at, link, Ordering::AcqRel, Ordering::Acquire).is_ok()
    }

    //for a list made whole (compact, from_tree), before its node is published
    pub fn set_next(&self, link: u32, next: u32) {
        self.get(link).next.store(next, Ordering::Relaxed);
    }

    //a solid slot's node and visits, before the node is published or with the tree to itself
    pub fn set_solid(&self, link: u32, child: u32, visits: u32) {
        let slot = self.slot(link);
        slot.child.store(child, Ordering::Relaxed);
        slot.visits.store(visits, Ordering::Relaxed);
    }

    pub fn linked<'a>(&'a self, node: &Node) -> Linked<'a> {
        let children = node.children();
        let (at, solid) = match children {
            Children::List(head) => (head, &[][..]),
            Children::Solid(first) => (0, self.solid_slots(first, node.edges().len())),
        };
        Linked { links: self, children, at, solid }
    }
}

//a node's moves with a node in their order, each as (its place among the node's moves, its name): the list's links, or the
//solid slots that have a node. A list's link put in meanwhile may be passed over, as a column read before it would be
pub struct Linked<'a> {
    links: &'a Links,
    children: Children,
    at: u32, //a list's next link, a solid node's next place
    solid: &'a [Slot],
}

impl Iterator for Linked<'_> {
    type Item = (usize, u32);

    fn next(&mut self) -> Option<(usize, u32)> {
        match self.children {
            Children::List(_) => {
                if self.at == NO_LINK {
                    return None;
                }
                let (link, record) = (self.at, self.links.get(self.at));
                self.at = record.next();
                Some((record.place(), link))
            }
            Children::Solid(first) => {
                while let Some(slot) = self.solid.get(self.at as usize) {
                    let (place, link) = (self.at, first + self.at);
                    self.at += 1;
                    if slot.child() != NO_NODE {
                        return Some((place as usize, link));
                    }
                }
                None
            }
        }
    }
}

pub struct Edges {
    priors: Table<AtomicU32>,
    moves: Table<AtomicU16>,
    next: AtomicUsize,
}

impl Edges {
    const CAPACITY: usize = EDGE_CHUNKS << EDGE_CHUNK_BITS;

    pub fn new() -> Edges {
        Edges {
            priors: Table::new(EDGE_CHUNKS, EDGE_CHUNK_BITS),
            moves: Table::new(EDGE_CHUNKS, EDGE_CHUNK_BITS),
            next: AtomicUsize::new(0),
        }
    }

    //the edges given out so far, the gaps at chunk ends included
    pub fn len(&self) -> usize {
        self.next.load(Ordering::Relaxed)
    }

    //room for a node's `count` edges (at most a chunk), in one chunk; returns the first
    pub fn alloc(&self, count: usize) -> usize {
        let first = alloc_run(&self.next, count, EDGE_CHUNK_BITS, Self::CAPACITY);
        self.priors.ensure_chunk(first, || (0..1 << EDGE_CHUNK_BITS).map(|_| AtomicU32::new(0)).collect());
        self.moves.ensure_chunk(first, || (0..1 << EDGE_CHUNK_BITS).map(|_| AtomicU16::new(0)).collect());
        first
    }

    pub fn priors(&self, edges: Range<usize>) -> &[AtomicU32] {
        if edges.is_empty() {
            return &[];
        }
        self.priors.slice(edges.start, edges.len())
    }

    pub fn prior(&self, edge: usize) -> f32 {
        f32::from_bits(self.priors.get(edge).load(Ordering::Relaxed))
    }

    pub fn chess_move(&self, edge: usize) -> PackedMove {
        PackedMove::from_bits(self.moves.get(edge).load(Ordering::Relaxed))
    }

    //a node's moves and priors from edge `first`, by the thread that owns the node, before its edges are published
    pub fn write(&self, first: usize, moves: &[Move], priors: &[f32]) {
        if moves.is_empty() {
            return;
        }
        let slots = self.moves.slice(first, moves.len()).iter().zip(self.priors.slice(first, moves.len()));
        for ((move_slot, prior_slot), (&chess_move, &prior)) in slots.zip(moves.iter().zip(priors)) {
            move_slot.store(PackedMove::new(chess_move).bits(), Ordering::Relaxed);
            prior_slot.store(prior.to_bits(), Ordering::Relaxed);
        }
    }

    pub fn set(&self, edge: usize, chess_move: PackedMove, prior: f32) {
        self.moves.get(edge).store(chess_move.bits(), Ordering::Relaxed);
        self.priors.get(edge).store(prior.to_bits(), Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edges_are_given_out_whole_and_once() {
        let edges = Edges::new();
        let chunk = 1usize << EDGE_CHUNK_BITS;
        let run_len = |worker: usize, turn: usize| 1 + (turn * 7 + worker * 13) % 218;
        let runs: Vec<Vec<usize>> = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..4)
                .map(|worker| {
                    let edges = &edges;
                    scope.spawn(move || (0..2_000).map(|turn| edges.alloc(run_len(worker, turn))).collect::<Vec<_>>())
                })
                .collect();
            workers.into_iter().map(|worker| worker.join().expect("a thread")).collect()
        });
        let mut taken = vec![false; edges.len()];
        for (worker, firsts) in runs.iter().enumerate() {
            for (turn, &first) in firsts.iter().enumerate() {
                let count = run_len(worker, turn);
                assert_eq!(first / chunk, (first + count - 1) / chunk, "a run across a chunk");
                for edge in first..first + count {
                    assert!(!std::mem::replace(&mut taken[edge], true), "edge {edge} given out twice");
                }
                assert_eq!(edges.priors(first..first + count).len(), count);
            }
        }
    }

    #[test]
    fn a_list_keeps_one_link_a_move_in_order() {
        let (links, head) = (Links::new(), AtomicU32::new(NO_LINK));
        let made: Vec<Vec<(usize, u32)>> = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..4u32)
                .map(|worker| {
                    let (links, head) = (&links, &head);
                    scope.spawn(move || {
                        (0..200usize)
                            .map(|turn| {
                                let place = (turn * 37 + worker as usize * 11) % 150;
                                let link = links.push(worker * 1000 + turn as u32, place);
                                loop {
                                    match links.find(head, place) {
                                        Ok(found) => break (place, links.child(found)),
                                        Err((pointer, at)) if links.insert(pointer, at, link) => break (place, links.child(link)),
                                        Err(_) => {}
                                    }
                                }
                            })
                            .collect()
                    })
                })
                .collect();
            workers.into_iter().map(|worker| worker.join().expect("a thread")).collect()
        });
        let mut list = Vec::new();
        let mut link = head.load(Ordering::Relaxed);
        while link != NO_LINK {
            list.push((links.get(link).place(), links.child(link)));
            link = links.get(link).next();
        }
        assert!(list.windows(2).all(|pair| pair[0].0 < pair[1].0), "a list out of order or with a move twice");
        for (place, child) in made.into_iter().flatten() {
            assert_eq!(list.iter().find(|&&(at, _)| at == place).map(|&(_, child)| child), Some(child));
        }
    }
}

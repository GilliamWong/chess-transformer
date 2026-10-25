//the storage of a tree that many threads grow at once: chunks made on first use and never moved or freed before the tree
//is, so a reference into one stays valid while other threads allocate. A node is one cache line of atomics and an edge
//a prior and a move (PackedMove). As lc0 has them, a move gets a Link once a simulation goes through it: its node, its
//own visits (with transpositions) and its place among the node's moves, in a list in the moves' order; a node with
//SOLID_TREE_THRESHOLD visits has its list turned into an array of links, one a move (lc0's solid tree)
use crate::rules::PackedMove;
use crate::search::{NO_NODE, Outcome, Proof};
use shakmaty::Move;
use std::ops::Range;
use std::ptr::{null_mut, slice_from_raw_parts_mut};
use std::sync::atomic::{AtomicPtr, AtomicU16, AtomicU32, AtomicU64, AtomicUsize, Ordering};

const NODE_CHUNK_BITS: u32 = if cfg!(test) { 6 } else { 16 };
const NODE_CHUNKS: usize = 1 << 14;
const EDGE_CHUNK_BITS: u32 = if cfg!(test) { 10 } else { 20 };
const EDGE_CHUNKS: usize = 1 << 15;
//a chunk holds a solid node's links (at most 218 moves); below 1 << 31, which the SOLID flag of a node's children takes
const LINK_CHUNK_BITS: u32 = if cfg!(test) { 8 } else { 18 };
const LINK_CHUNKS: usize = 1 << 13;

//a node's children: none, the first link of its list, or with SOLID its solid links' first
pub const NO_LINK: u32 = 0;
const SOLID: u32 = 1 << 31;

//one in-flight simulation in a node's count, which holds the visits below it
pub const IN_FLIGHT: u64 = 1 << 32;

//a node's numbers; every access is atomic, and the orderings are the tree's protocol (see shared.rs)
#[repr(align(64))]
#[derive(Default)]
pub struct Node {
    count: AtomicU64, //visits in the low 32 bits, simulations in flight in the high 32
    value_sum: AtomicU64,
    visited_policy_sqrt: AtomicU64,
    edges_at: AtomicU64, //Tree's encoding: the first edge above the low 8 bits, the number of edges in them; 0 until expanded
    proof_value: AtomicU64,
    proof_meta: AtomicU64, //0 without a proof, else the plies above the low 2 bits and the outcome in them (1 loss, 2 draw, 3 win)
    key: AtomicU64,
    //the refreshes of the visited policy begun, wrapping: a refresh compares two counts a few loads apart
    refreshes: AtomicU16,
    //with transpositions the node's fifty-move count (search.rs Graph's halfmoves)
    halfmoves: AtomicU16,
    //the links of its moves with a node (Children)
    children: AtomicU32,
}

//a node's links: a list in the moves' order, or once solid one a move from the first
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

    //counts one more simulation in flight through the node; returns the (visits, in flight) before
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

    //a node of a tree being copied, before anything else sees it
    #[cfg(test)]
    pub fn set_counts(&self, visits: u32, in_flight: u32, value_sum: f64) {
        self.count.store(u64::from(in_flight) << 32 | u64::from(visits), Ordering::Relaxed);
        self.value_sum.store(value_sum.to_bits(), Ordering::Relaxed);
    }

    pub fn take_visits(&self, visits: u32) {
        self.count.fetch_sub(u64::from(visits), Ordering::Relaxed);
    }

    //the node as if never visited, its position's key kept
    pub fn reset(&self) {
        for field in [&self.count, &self.value_sum, &self.visited_policy_sqrt, &self.edges_at, &self.proof_value, &self.proof_meta] {
            field.store(0, Ordering::Relaxed);
        }
        self.children.store(NO_LINK, Ordering::Relaxed);
    }

    //the numbers, key and proof of a node (not its edges or links) copied to a node of a tree being made
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
        let mut seen = self.value_sum.load(Ordering::Relaxed);
        while let Err(now) =
            self.value_sum.compare_exchange_weak(seen, (f64::from_bits(seen) + delta).to_bits(), Ordering::Relaxed, Ordering::Relaxed)
        {
            seen = now;
        }
    }

    pub fn mean_score(&self, visits: u32) -> f64 {
        self.value_sum() / f64::from(visits)
    }

    pub fn visited_policy_sqrt(&self) -> f64 {
        f64::from_bits(self.visited_policy_sqrt.load(Ordering::Relaxed))
    }

    //with the refresh counter below this is sequentially consistent: a store buffered past the counter's check is a lost one
    pub fn set_visited_policy_sqrt(&self, value: f64) {
        self.visited_policy_sqrt.store(value.to_bits(), Ordering::SeqCst);
    }

    //a refresh of the visited policy begins; returns the number of refreshes begun, itself included
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

    //set by the thread that makes the node, before its index is given out
    pub fn set_halfmoves(&self, halfmoves: u16) {
        self.halfmoves.store(halfmoves, Ordering::Relaxed);
    }

    pub fn children(&self) -> Children {
        let children = self.children.load(Ordering::Acquire);
        if children & SOLID != 0 { Children::Solid(children & !SOLID) } else { Children::List(children) }
    }

    //the head of a list node's links, which a new first link is swapped into (Links::insert)
    pub fn list_head(&self) -> &AtomicU32 {
        &self.children
    }

    //the node's links from now on the solid ones from `first`; with the tree to itself (or one not yet published)
    pub fn make_solid(&self, first: u32) {
        self.children.store(first | SOLID, Ordering::Release);
    }

    pub fn edges(&self) -> Range<usize> {
        let edges_at = self.edges_at.load(Ordering::Acquire);
        let first = (edges_at >> 8) as usize;
        first..first + (edges_at & 255) as usize
    }

    pub fn has_edges(&self) -> bool {
        self.edges_at.load(Ordering::Acquire) & 255 > 0
    }

    //makes the node's edges, written before, visible to those that read them
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

//chunks of `columns` times `1 << bits` elements made on first use, a pointer to each. A chunk is allocated once, by whichever
//thread needs it first, and freed only with the table, so a reference into one is valid for the table's life. Elements are
//found by an index that counts `1 << bits` to a chunk, a column of a chunk by its number
struct Table<T> {
    chunks: Box<[AtomicPtr<T>]>,
    bits: u32,
    columns: usize,
}

impl<T> Table<T> {
    fn new(chunks: usize, bits: u32, columns: usize) -> Table<T> {
        Table { chunks: (0..chunks).map(|_| AtomicPtr::new(null_mut())).collect(), bits, columns }
    }

    //the chunk's first element; every index given out is in a chunk that is made (the allocators make it before they give
    //the index out)
    fn chunk_of(&self, index: usize) -> *mut T {
        let chunk = self.chunks[index >> self.bits].load(Ordering::Acquire);
        assert!(!chunk.is_null(), "an index in a chunk that was never made");
        chunk
    }

    fn get(&self, index: usize) -> &T {
        &self.slice(index, 0, 1)[0]
    }

    //`len` elements of a column from `index`, all in its chunk
    fn slice(&self, index: usize, column: usize, len: usize) -> &[T] {
        let offset = index & ((1 << self.bits) - 1);
        assert!(column < self.columns && offset + len <= 1 << self.bits, "elements across a chunk's end");
        // SAFETY: the chunk holds columns << bits elements and never moves or goes before the table does, and the assert keeps
        // the elements inside the column
        unsafe { std::slice::from_raw_parts(self.chunk_of(index).add((column << self.bits) + offset), len) }
    }

    //makes the chunk of the index (columns << bits elements, from `make`) if no thread has
    fn made(&self, index: usize, make: impl FnOnce() -> Box<[T]>) {
        let slot = &self.chunks[index >> self.bits];
        if !slot.load(Ordering::Acquire).is_null() {
            return;
        }
        let fresh = Box::into_raw(make());
        assert_eq!(fresh.len(), self.columns << self.bits);
        if slot.compare_exchange(null_mut(), fresh.cast(), Ordering::AcqRel, Ordering::Acquire).is_err() {
            // SAFETY: fresh came from Box::into_raw just now, and the other thread's chunk won the slot
            drop(unsafe { Box::from_raw(fresh) });
        }
    }
}

impl<T> Drop for Table<T> {
    fn drop(&mut self) {
        let len = self.columns << self.bits;
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
    //the nodes a tree may have: below NO_NODE, and the table's
    pub const CAPACITY: usize = (NODE_CHUNKS << NODE_CHUNK_BITS) - 1;

    pub fn new() -> Nodes {
        Nodes { table: Table::new(NODE_CHUNKS, NODE_CHUNK_BITS, 1), next: AtomicUsize::new(0) }
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
        self.table.made(index, || (0..1 << NODE_CHUNK_BITS).map(|_| Node::default()).collect());
        self.table.get(index).key.store(key, Ordering::Relaxed);
        index
    }
}

//`count` elements (at most a chunk of `1 << bits`) in one chunk from the allocator's next, below `capacity`: the first
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

//a move with a node: the node, the move's own visits (with transpositions), the next link of its node's list, and its place
//among the node's moves. A list's link is written before it is swapped in (release) and read after its predecessor's
//`next` (acquire); a solid link's node is swapped in alone
pub struct Link {
    child: AtomicU32,
    visits: AtomicU32,
    next: AtomicU32,
    place: AtomicU32,
}

impl Link {
    fn new(place: u32) -> Link {
        Link { child: AtomicU32::new(NO_NODE), visits: AtomicU32::new(0), next: AtomicU32::new(NO_LINK), place: AtomicU32::new(place) }
    }

    //NO_NODE until a solid link's move has one
    pub fn child(&self) -> u32 {
        self.child.load(Ordering::Acquire)
    }

    //the move's node for a solid link, which may have one already: the node it has
    pub fn claim(&self, child: u32) -> Result<u32, u32> {
        self.child.compare_exchange(NO_NODE, child, Ordering::AcqRel, Ordering::Acquire).map(|_| child)
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

pub struct Links {
    table: Table<Link>,
    next: AtomicUsize,
}

impl Links {
    pub const CAPACITY: usize = LINK_CHUNKS << LINK_CHUNK_BITS;

    //link 0 is NO_LINK's, never given out
    pub fn new() -> Links {
        let links = Links { table: Table::new(LINK_CHUNKS, LINK_CHUNK_BITS, 1), next: AtomicUsize::new(1) };
        links.make(0);
        links
    }

    //the links given out so far, the gaps at chunk ends included
    pub fn len(&self) -> usize {
        self.next.load(Ordering::Relaxed)
    }

    fn make(&self, first: usize) {
        self.table.made(first, || (0..1 << LINK_CHUNK_BITS).map(|_| Link::new(0)).collect());
    }

    pub fn get(&self, link: u32) -> &Link {
        self.table.get(link as usize)
    }

    //a list link for the move at `place` with its node, not yet in a list
    pub fn push(&self, child: u32, place: usize) -> u32 {
        let link = alloc_run(&self.next, 1, LINK_CHUNK_BITS, Self::CAPACITY);
        self.make(link);
        let made = self.table.get(link);
        made.child.store(child, Ordering::Relaxed);
        made.visits.store(0, Ordering::Relaxed);
        made.place.store(place as u32, Ordering::Relaxed);
        link as u32
    }

    //a solid node's links, one a move from the first returned: none has a node yet
    pub fn solid(&self, count: usize) -> u32 {
        let first = alloc_run(&self.next, count, LINK_CHUNK_BITS, Self::CAPACITY);
        self.make(first);
        for place in 0..count {
            let link = self.table.get(first + place);
            link.child.store(NO_NODE, Ordering::Relaxed);
            link.visits.store(0, Ordering::Relaxed);
            link.next.store(NO_LINK, Ordering::Relaxed);
            link.place.store(place as u32, Ordering::Relaxed);
        }
        first as u32
    }

    //the link of the move at `place` in the list from `head`, or Err of the pointer a new link for it goes into (the head or
    //a link's next) and the link that pointer holds now
    pub fn find<'a>(&'a self, head: &'a AtomicU32, place: usize) -> Result<u32, (&'a AtomicU32, u32)> {
        let mut pointer = head;
        loop {
            let link = pointer.load(Ordering::Acquire);
            if link == NO_LINK || self.get(link).place() > place {
                return Err((pointer, link));
            }
            if self.get(link).place() == place {
                return Ok(link);
            }
            pointer = &self.get(link).next;
        }
    }

    //puts the link (Links::push's, of its place) into its list between the pointer found and the link it held: false if
    //the pointer holds another link now
    pub fn insert(&self, pointer: &AtomicU32, at: u32, link: u32) -> bool {
        self.get(link).next.store(at, Ordering::Relaxed);
        pointer.compare_exchange(at, link, Ordering::AcqRel, Ordering::Acquire).is_ok()
    }

    //a link's next when a list is made whole, as compact and from_tree make them, before the node is published
    pub fn set_next(&self, link: u32, next: u32) {
        self.get(link).next.store(next, Ordering::Relaxed);
    }

    //a solid link's node and visits, before the node is published or with the tree to itself
    pub fn set_solid(&self, link: u32, child: u32, visits: u32) {
        let link = self.get(link);
        link.child.store(child, Ordering::Relaxed);
        link.visits.store(visits, Ordering::Relaxed);
    }

    //the node's moves with a node, in their order
    pub fn linked<'a>(&'a self, node: &Node) -> Linked<'a> {
        let children = node.children();
        let at = match children {
            Children::List(head) => head,
            Children::Solid(_) => 0,
        };
        Linked { links: self, children, at, moves: node.edges().len() as u32 }
    }
}

//a node's moves with a node in their order, each as (its place among the node's moves, its link): the list's links, or the
//solid links that have a node. A list's link put in meanwhile may be passed over, as a column read before it would be
pub struct Linked<'a> {
    links: &'a Links,
    children: Children,
    at: u32, //a list's next link, a solid node's next place
    moves: u32,
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
                while self.at < self.moves {
                    let (place, link) = (self.at, first + self.at);
                    self.at += 1;
                    if self.links.get(link).child() != NO_NODE {
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
    pub const CAPACITY: usize = EDGE_CHUNKS << EDGE_CHUNK_BITS;

    pub fn new() -> Edges {
        Edges { priors: Table::new(EDGE_CHUNKS, EDGE_CHUNK_BITS, 1), moves: Table::new(EDGE_CHUNKS, EDGE_CHUNK_BITS, 1), next: AtomicUsize::new(0) }
    }

    //the edges given out so far, the gaps at chunk ends included
    pub fn len(&self) -> usize {
        self.next.load(Ordering::Relaxed)
    }

    //room for a node's `count` edges (at most a chunk), in one chunk; returns the first
    pub fn alloc(&self, count: usize) -> usize {
        let first = alloc_run(&self.next, count, EDGE_CHUNK_BITS, Self::CAPACITY);
        self.priors.made(first, || (0..1 << EDGE_CHUNK_BITS).map(|_| AtomicU32::new(0)).collect());
        self.moves.made(first, || (0..1 << EDGE_CHUNK_BITS).map(|_| AtomicU16::new(0)).collect());
        first
    }

    pub fn priors(&self, edges: Range<usize>) -> &[AtomicU32] {
        if edges.is_empty() {
            return &[];
        }
        debug_assert_eq!(edges.start >> EDGE_CHUNK_BITS, (edges.end - 1) >> EDGE_CHUNK_BITS, "a node's edges lie in one chunk");
        self.priors.slice(edges.start, 0, edges.len())
    }

    pub fn prior(&self, edge: usize) -> f32 {
        f32::from_bits(self.priors.get(edge).load(Ordering::Relaxed))
    }

    //the edge's move, as the position it is played from reads it (PackedMove::to_move)
    pub fn chess_move(&self, edge: usize) -> PackedMove {
        PackedMove::from_bits(self.moves.get(edge).load(Ordering::Relaxed))
    }

    //a node's moves and priors from edge `first`, written by the thread that owns the node before its edges are published
    pub fn write(&self, first: usize, moves: &[Move], priors: &[f32]) {
        if moves.is_empty() {
            return;
        }
        let edges = first..first + moves.len();
        let slots = self.moves.slice(first, 0, moves.len()).iter().zip(self.priors(edges));
        for ((move_slot, prior_slot), (&chess_move, &prior)) in slots.zip(moves.iter().zip(priors)) {
            move_slot.store(PackedMove::new(chess_move).bits(), Ordering::Relaxed);
            prior_slot.store(prior.to_bits(), Ordering::Relaxed);
        }
    }

    //an edge's move and prior
    pub fn set(&self, edge: usize, chess_move: PackedMove, prior: f32) {
        self.moves.get(edge).store(chess_move.bits(), Ordering::Relaxed);
        self.priors.get(edge).store(prior.to_bits(), Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    //edges are given out in runs that never cross a chunk, to threads that never get the same edge twice
    #[test]
    fn edges_are_given_out_whole_and_once() {
        let edges = Edges::new();
        let chunk = 1usize << EDGE_CHUNK_BITS;
        let runs: Vec<Vec<(usize, usize)>> = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..4)
                .map(|worker| {
                    let edges = &edges;
                    scope.spawn(move || (0..2_000).map(|turn| (edges.alloc(1 + (turn * 7 + worker * 13) % 218), 0)).collect::<Vec<_>>())
                })
                .collect();
            workers.into_iter().map(|worker| worker.join().expect("a thread")).collect()
        });
        let mut taken = vec![false; edges.len()];
        for (worker, worker_runs) in runs.iter().enumerate() {
            for (turn, &(first, _)) in worker_runs.iter().enumerate() {
                let count = 1 + (turn * 7 + worker * 13) % 218;
                assert_eq!(first / chunk, (first + count - 1) / chunk, "a run across a chunk");
                for edge in first..first + count {
                    assert!(!std::mem::replace(&mut taken[edge], true), "edge {edge} given out twice");
                }
                //its priors are there
                assert_eq!(edges.priors(first..first + count).len(), count);
            }
        }
    }

    //a list's links stay in their moves' order and each move gets one link, however many threads put them in at once
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
                                        Ok(found) => break (place, links.get(found).child()),
                                        Err((pointer, at)) if links.insert(pointer, at, link) => break (place, links.get(link).child()),
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
            list.push((links.get(link).place(), links.get(link).child()));
            link = links.get(link).next();
        }
        assert!(list.windows(2).all(|pair| pair[0].0 < pair[1].0), "a list out of order or with a move twice");
        //every thread found the node its move's link has
        for (place, child) in made.into_iter().flatten() {
            assert_eq!(list.iter().find(|&&(at, _)| at == place).map(|&(_, child)| child), Some(child));
        }
    }
}

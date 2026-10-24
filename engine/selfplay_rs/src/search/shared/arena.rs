//the storage of a tree that many threads grow at once: chunks made on first use and never moved or freed before the tree
//is, so a reference into one stays valid while other threads allocate. A node is one cache line of atomics, an edge three
//u32 columns (prior, child, move)
use crate::search::{NO_NODE, Outcome, Proof};
use shakmaty::{Move, Role, Square};
use std::ops::Range;
use std::ptr::{null_mut, slice_from_raw_parts_mut};
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};

const NODE_CHUNK_BITS: u32 = if cfg!(test) { 6 } else { 16 };
const NODE_CHUNKS: usize = 1 << 14;
const EDGE_CHUNK_BITS: u32 = if cfg!(test) { 10 } else { 20 };
const EDGE_CHUNKS: usize = 1 << 15;

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
    refreshes: AtomicU32,
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
    }

    //the numbers, key and proof of a node (not its edges) copied to a node of a tree being made
    pub fn copy_to(&self, target: &Node) {
        target.count.store(self.count.load(Ordering::Relaxed), Ordering::Relaxed);
        target.value_sum.store(self.value_sum.load(Ordering::Relaxed), Ordering::Relaxed);
        target.visited_policy_sqrt.store(self.visited_policy_sqrt.load(Ordering::Relaxed), Ordering::Relaxed);
        target.proof_value.store(self.proof_value.load(Ordering::Relaxed), Ordering::Relaxed);
        target.proof_meta.store(self.proof_meta.load(Ordering::Relaxed), Ordering::Relaxed);
        target.key.store(self.key.load(Ordering::Relaxed), Ordering::Relaxed);
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

    pub fn set_visited_policy_sqrt(&self, value: f64) {
        self.visited_policy_sqrt.store(value.to_bits(), Ordering::Relaxed);
    }

    //a refresh of the visited policy begins; returns the number of refreshes begun, itself included
    pub fn begin_refresh(&self) -> u32 {
        self.refreshes.fetch_add(1, Ordering::AcqRel) + 1
    }

    pub fn refreshes(&self) -> u32 {
        self.refreshes.load(Ordering::Acquire)
    }

    pub fn key(&self) -> u64 {
        self.key.load(Ordering::Relaxed)
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

//the edges' columns of a chunk, in one slice: priors, children, moves
const PRIORS: usize = 0;
const CHILDREN: usize = 1;
const MOVES: usize = 2;

pub struct Edges {
    table: Table<AtomicU32>,
    next: AtomicUsize,
}

impl Edges {
    pub const CAPACITY: usize = EDGE_CHUNKS << EDGE_CHUNK_BITS;

    pub fn new() -> Edges {
        Edges { table: Table::new(EDGE_CHUNKS, EDGE_CHUNK_BITS, 3), next: AtomicUsize::new(0) }
    }

    //the edges given out so far, the gaps at chunk ends included
    pub fn len(&self) -> usize {
        self.next.load(Ordering::Relaxed)
    }

    //room for a node's `count` edges (at most a chunk), in one chunk; returns the first
    pub fn alloc(&self, count: usize) -> usize {
        debug_assert!(count > 0 && count <= 1 << EDGE_CHUNK_BITS);
        let mut at = self.next.load(Ordering::Relaxed);
        let first = loop {
            let chunk_end = (at | ((1 << EDGE_CHUNK_BITS) - 1)) + 1;
            let first = if at + count <= chunk_end { at } else { chunk_end };
            match self.next.compare_exchange_weak(at, first + count, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break first,
                Err(now) => at = now,
            }
        };
        assert!(first + count <= Self::CAPACITY, "the tree is past its edges");
        self.table.made(first, || {
            let chunk = 1usize << EDGE_CHUNK_BITS;
            (0..3 * chunk).map(|slot| AtomicU32::new(if slot / chunk == CHILDREN { NO_NODE } else { 0 })).collect()
        });
        first
    }

    fn column(&self, edges: Range<usize>, column: usize) -> &[AtomicU32] {
        if edges.is_empty() {
            return &[];
        }
        debug_assert_eq!(edges.start >> EDGE_CHUNK_BITS, (edges.end - 1) >> EDGE_CHUNK_BITS, "a node's edges lie in one chunk");
        self.table.slice(edges.start, column, edges.len())
    }

    pub fn priors(&self, edges: Range<usize>) -> &[AtomicU32] {
        self.column(edges, PRIORS)
    }

    pub fn children(&self, edges: Range<usize>) -> &[AtomicU32] {
        self.column(edges, CHILDREN)
    }

    pub fn prior(&self, edge: usize) -> f32 {
        f32::from_bits(self.priors(edge..edge + 1)[0].load(Ordering::Relaxed))
    }

    pub fn child(&self, edge: usize) -> &AtomicU32 {
        &self.children(edge..edge + 1)[0]
    }

    pub fn chess_move(&self, edge: usize) -> Move {
        decode(self.column(edge..edge + 1, MOVES)[0].load(Ordering::Relaxed))
    }

    //a node's moves and priors from edge `first`, written by the thread that owns the node before its edges are published
    pub fn write(&self, first: usize, moves: &[Move], priors: &[f32]) {
        let edges = first..first + moves.len();
        let slots = self.column(edges.clone(), MOVES).iter().zip(self.priors(edges));
        for ((move_slot, prior_slot), (&chess_move, &prior)) in slots.zip(moves.iter().zip(priors)) {
            move_slot.store(encode(chess_move), Ordering::Relaxed);
            prior_slot.store(prior.to_bits(), Ordering::Relaxed);
        }
    }

    //an edge's move and prior
    pub fn set(&self, edge: usize, chess_move: Move, prior: f32) {
        self.write(edge, &[chess_move], &[prior]);
    }
}

//a move in 23 bits: kind (normal, en passant, castle) 21..23, promotion 18..21, capture 15..18, role 12..15, to 6..12, from 0..6
fn encode(chess_move: Move) -> u32 {
    let role = |role: Option<Role>| role.map_or(0, u32::from);
    match chess_move {
        Move::Normal { role: moved, from, capture, to, promotion } => {
            from.to_u32() | to.to_u32() << 6 | u32::from(moved) << 12 | role(capture) << 15 | role(promotion) << 18
        }
        Move::EnPassant { from, to } => from.to_u32() | to.to_u32() << 6 | 1 << 21,
        Move::Castle { king, rook } => king.to_u32() | rook.to_u32() << 6 | 2 << 21,
        Move::Put { .. } => unreachable!("no drops in chess"),
    }
}

fn decode(code: u32) -> Move {
    let role = |bits: u32| (bits != 0).then(|| Role::ALL[bits as usize - 1]);
    let (from, to) = (Square::new(code & 63), Square::new(code >> 6 & 63));
    match code >> 21 {
        0 => Move::Normal {
            role: Role::ALL[(code >> 12 & 7) as usize - 1],
            from,
            capture: role(code >> 15 & 7),
            to,
            promotion: role(code >> 18 & 7),
        },
        1 => Move::EnPassant { from, to },
        _ => Move::Castle { king: from, rook: to },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::fen::Fen;
    use shakmaty::{CastlingMode, Chess, Position};

    //a position with every kind of move: castles, en passant, promotions with and without captures, checks
    const POSITIONS: [&str; 6] = [
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3",
        "1k6/6P1/8/8/8/8/8/K7 w - - 0 1",
        "r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ];

    //every legal move of positions with all kinds of moves comes back from its code, and no two share one
    #[test]
    fn a_move_comes_back_from_its_code() {
        let mut codes = std::collections::HashSet::new();
        let mut kinds = [0; 3];
        for fen in POSITIONS {
            let board: Chess = fen.parse::<Fen>().expect("a fen").into_position(CastlingMode::Standard).expect("a position");
            for chess_move in board.legal_moves() {
                assert_eq!(decode(encode(chess_move)), chess_move);
                kinds[(encode(chess_move) >> 21) as usize] += 1;
                if !codes.insert((fen, encode(chess_move))) {
                    panic!("{chess_move:?} shares its code");
                }
            }
        }
        assert!(kinds.iter().all(|&kind| kind > 0), "{kinds:?}");
    }

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
                //its columns are there, and its children are free
                assert!(edges.children(first..first + count).iter().all(|child| child.load(Ordering::Relaxed) == NO_NODE));
            }
        }
    }
}

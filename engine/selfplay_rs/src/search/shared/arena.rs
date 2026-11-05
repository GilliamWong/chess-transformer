use crate::rules::PackedMove;
use crate::search::{MOST_MOVES, NO_LINK, NO_NODE, Outcome, Proof, SOLID, by_prior, listed};
use shakmaty::Move;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::ptr::{null_mut, slice_from_raw_parts_mut};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU16, AtomicU32, AtomicU64, AtomicUsize, Ordering};

const NODE_CHUNK_BITS: u32 = if cfg!(test) { 6 } else { 16 };
const NODE_CHUNKS: usize = 1 << 14;
const EDGE_CHUNK_BITS: u32 = if cfg!(test) { 10 } else { 20 };
const EDGE_CHUNKS: usize = 1 << 15;
//a chunk holds a solid node's slots (218 moves at most); links and slots stay below 1 << 31, which SOLID takes
const LINK_CHUNK_BITS: u32 = if cfg!(test) { 8 } else { 18 };
const LINK_CHUNKS: usize = 1 << 13;

const IN_FLIGHT: u64 = 1 << 32;
//a freed node's count in a debug build, for a descent that reaches it to find
const FREED: u64 = u64::MAX;
//proof_meta keeps a proof's plies in the bits above its outcome's two
const PROOF_PLIES_BITS: u32 = 30;

//a counter that every thread writes, on a line of its own: the fields beside it are read by every descent
#[repr(align(64))]
pub struct Alone<T>(pub T);

impl<T: Default> Default for Alone<T> {
    fn default() -> Alone<T> {
        Alone(T::default())
    }
}

impl<T> std::ops::Deref for Alone<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

//the orderings are the tree's protocol (shared.rs)
#[repr(align(64))]
#[derive(Default)]
pub struct Node {
    count: AtomicU64, //visits in the low 32 bits, simulations in flight in the high 32
    value_sum: AtomicU64,
    visited_policy_sqrt: AtomicU64,
    edges_at: AtomicU64, //the first edge above the low 8 bits, the number of edges in them; 0 until expanded
    proof_value: AtomicU64,
    key: AtomicU64,
    proof_meta: AtomicU32, //0 without a proof, else the plies above the low 2 bits and the outcome in them (1 loss, 2 draw, 3 win)
    references: AtomicU32, //the moves (links and slots) that name the node, and 1 for the root; 0 once it is dead: see SharedTree
    refreshes: AtomicU16,  //begun, wrapping
    halfmoves: AtomicU16,  //with transpositions the node's fifty-move count
    children: AtomicU32,   //a list's first link, or once solid its first slot (SOLID set): see Children
}

const _: () = assert!(size_of::<Node>() == 64);

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
        debug_assert!(before != FREED, "a descent reached a freed node");
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

    //the node as if never visited, its key and references kept
    pub fn reset(&self) {
        for field in [&self.count, &self.value_sum, &self.visited_policy_sqrt, &self.edges_at, &self.proof_value] {
            field.store(0, Ordering::Relaxed);
        }
        self.proof_meta.store(0, Ordering::Relaxed);
        self.children.store(NO_LINK, Ordering::Relaxed);
    }

    //a reference to a node that has one: false for a dead node, which no one revives
    pub fn acquire(&self) -> bool {
        let mut seen = self.references.load(Ordering::Relaxed);
        while seen > 0 {
            match self.references.compare_exchange_weak(seen, seen + 1, Ordering::AcqRel, Ordering::Relaxed) {
                Ok(_) => return true,
                Err(now) => seen = now,
            }
        }
        false
    }

    //a reference gone; true for the last, whose thread then owns the node's death
    pub fn release(&self) -> bool {
        let before = self.references.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(before > 0, "a reference released from a dead node");
        before == 1
    }

    //the node as new, with a reference: for a node out of the pool or the table's end, before its index is given out
    fn recycle(&self, key: u64) {
        self.reset();
        self.refreshes.store(0, Ordering::Relaxed);
        self.halfmoves.store(0, Ordering::Relaxed);
        self.key.store(key, Ordering::Relaxed);
        self.references.store(1, Ordering::Release);
    }

    //for a node of a tree being copied, before anything else sees it
    #[cfg(test)]
    pub fn set_references(&self, references: u32) {
        self.references.store(references, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub fn references(&self) -> u32 {
        self.references.load(Ordering::Acquire)
    }

    #[cfg(test)]
    pub fn is_freed(&self) -> bool {
        self.count.load(Ordering::Relaxed) == FREED
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

    //the list's head, which a new first link is swapped into: for threads that hold the gate, not for a reader (children() gives its head)
    pub(super) fn list_head(&self) -> &AtomicU32 {
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
        Some(Proof { outcome, value: f64::from_bits(self.proof_value.load(Ordering::Relaxed)), plies: meta >> 2 })
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
        assert!(proof.plies < 1 << PROOF_PLIES_BITS, "a proof of {} plies", proof.plies);
        self.proof_value.store(proof.value.to_bits(), Ordering::Relaxed);
        self.proof_meta.store(proof.plies << 2 | outcome, Ordering::Release);
    }

    pub fn clear_proof(&self) {
        self.proof_meta.store(0, Ordering::Release);
    }

    //the node as `from` is, with its edges from `first_edge` and its moves' nodes from `children`, for a copy nothing reads yet; its
    //references are the copy's moves' (add_reference)
    pub fn copy_of(&self, from: &Node, first_edge: usize, children: u32) {
        let copied = [&self.count, &self.value_sum, &self.visited_policy_sqrt, &self.proof_value, &self.key];
        for (to, from) in copied.into_iter().zip([&from.count, &from.value_sum, &from.visited_policy_sqrt, &from.proof_value, &from.key]) {
            to.store(from.load(Ordering::Relaxed), Ordering::Relaxed);
        }
        let moves = from.edges().len();
        self.edges_at.store(if moves == 0 { 0 } else { (first_edge as u64) << 8 | moves as u64 }, Ordering::Relaxed);
        self.proof_meta.store(from.proof_meta.load(Ordering::Relaxed), Ordering::Relaxed);
        self.halfmoves.store(from.halfmoves(), Ordering::Relaxed);
        self.children.store(children, Ordering::Relaxed);
    }

    //a move of a copy nothing reads yet names the node
    pub fn add_reference(&self) {
        self.references.fetch_add(1, Ordering::Relaxed);
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

    //a hint to the cpu that element `index` is read soon; it reads nothing the program sees, a chunk not made included
    fn prefetch(&self, index: usize) {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            let chunk = self.chunks[index >> self.bits].load(Ordering::Relaxed);
            // SAFETY: sse is part of x86_64, and a prefetch of an address that is none is no read
            unsafe { _mm_prefetch::<_MM_HINT_T0>(chunk.wrapping_add(index & ((1 << self.bits) - 1)).cast()) };
        }
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

//indices given back, to be given out again. `idle` is how many it holds, looked at without the lock: a pool never used leaves
//the tables to hand out in the order of their ends
#[derive(Default)]
struct Pool {
    indices: Mutex<Vec<u32>>,
    idle: AtomicUsize,
    //the indices are kept, not given out: what is made lies together, at the table's end
    held_back: Alone<AtomicBool>,
}

impl Pool {
    fn take(&self) -> Option<u32> {
        if self.held_back.load(Ordering::Relaxed) || self.idle.load(Ordering::Relaxed) == 0 {
            return None;
        }
        let mut indices = self.indices.lock().expect("a pool");
        let index = indices.pop();
        self.idle.store(indices.len(), Ordering::Relaxed);
        index
    }

    fn give(&self, index: u32) {
        let mut indices = self.indices.lock().expect("a pool");
        indices.push(index);
        self.idle.store(indices.len(), Ordering::Relaxed);
    }

    //the indices come out of the pool lowest first within a batch, so that nodes made together lie together
    fn give_all(&self, given: &mut Vec<u32>) {
        given.sort_unstable_by(|one, other| other.cmp(one));
        let mut indices = self.indices.lock().expect("a pool");
        indices.append(given);
        self.idle.store(indices.len(), Ordering::Relaxed);
    }

    fn idle(&self) -> usize {
        self.idle.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    fn held(&self) -> Vec<u32> {
        self.indices.lock().expect("a pool").clone()
    }
}

//runs given back, joined with the runs next to them, kept by their first element and by their length; no run crosses a chunk. A
//take of up to PIECE_LENGTHS - 1 elements pops a piece of that length; when there is none, the shortest run that is long enough is
//cut into as many as PIECE_BATCH, so that the runs' sets are visited once in that many takes and the pieces come side by side
struct FreeRuns {
    pieces: Box<[Mutex<Vec<usize>>]>, //by length
    runs: Mutex<RunSets>,
    idle: AtomicUsize, //the elements in the pieces and the runs, looked at without a lock
    chunk_mask: usize,
    //the runs are kept, not given out
    held_back: Alone<AtomicBool>,
}

const PIECE_LENGTHS: usize = 256;
const PIECE_BATCH: usize = 32;

#[derive(Default)]
struct RunSets {
    by_first: BTreeMap<usize, usize>,
    by_length: BTreeSet<(usize, usize)>, //(length, first)
}

impl RunSets {
    fn insert(&mut self, first: usize, length: usize) {
        self.by_first.insert(first, length);
        self.by_length.insert((length, first));
    }

    fn remove(&mut self, first: usize, length: usize) {
        self.by_first.remove(&first);
        self.by_length.remove(&(length, first));
    }
}

impl FreeRuns {
    fn new(chunk_bits: u32) -> FreeRuns {
        FreeRuns {
            pieces: (0..PIECE_LENGTHS).map(|_| Mutex::new(Vec::new())).collect(),
            runs: Mutex::new(RunSets::default()),
            idle: AtomicUsize::new(0),
            chunk_mask: (1 << chunk_bits) - 1,
            held_back: Alone::default(),
        }
    }

    fn idle(&self) -> usize {
        self.idle.load(Ordering::Relaxed)
    }

    fn give(&self, first: usize, count: usize) {
        assert!((1..PIECE_LENGTHS).contains(&count), "a run of {count}");
        let mut runs = self.runs.lock().expect("the runs given back");
        let (mut first, mut length) = (first, count);
        //a run before or after, unless the chunk starts or ends between them
        if first & self.chunk_mask != 0
            && let Some((&before, &before_length)) = runs.by_first.range(..first).next_back()
            && before + before_length == first
        {
            runs.remove(before, before_length);
            (first, length) = (before, length + before_length);
        }
        let end = first + length;
        if end & self.chunk_mask != 0
            && let Some(&after_length) = runs.by_first.get(&end)
        {
            runs.remove(end, after_length);
            length += after_length;
        }
        runs.insert(first, length);
        self.idle.fetch_add(count, Ordering::Relaxed);
    }

    //a run of `count` elements
    fn take(&self, count: usize) -> Option<usize> {
        assert!((1..PIECE_LENGTHS).contains(&count), "a run of {count}");
        if self.held_back.load(Ordering::Relaxed) || self.idle() == 0 {
            return None;
        }
        if let Some(first) = self.pieces[count].lock().expect("the pieces").pop() {
            self.idle.fetch_sub(count, Ordering::Relaxed);
            return Some(first);
        }
        let mut runs = self.runs.lock().expect("the runs given back");
        let &(length, first) = runs.by_length.range((count, 0)..).next()?;
        runs.remove(first, length);
        let cut = (length / count).min(PIECE_BATCH);
        if length > cut * count {
            runs.insert(first + cut * count, length - cut * count);
        }
        if cut > 1 {
            self.pieces[count].lock().expect("the pieces").extend((1..cut).rev().map(|piece| first + piece * count));
        }
        self.idle.fetch_sub(count, Ordering::Relaxed);
        Some(first)
    }

    #[cfg(test)]
    fn held(&self) -> Vec<(usize, usize)> {
        let runs = self.runs.lock().expect("the runs given back");
        let mut held: Vec<(usize, usize)> = runs.by_first.iter().map(|(&first, &length)| (first, length)).collect();
        for (length, pieces) in self.pieces.iter().enumerate() {
            held.extend(pieces.lock().expect("the pieces").iter().map(|&first| (first, length)));
        }
        held.sort_unstable();
        held
    }
}

pub struct Nodes {
    table: Table<Node>,
    next: Alone<AtomicUsize>,
    freed: Pool,
}

impl Nodes {
    pub fn hold_back(&self, held: bool) {
        self.freed.held_back.store(held, Ordering::Relaxed);
    }

    pub fn holds_back(&self) -> bool {
        self.freed.held_back.load(Ordering::Relaxed)
    }

    //below NO_NODE, and the table's
    pub const CAPACITY: usize = (NODE_CHUNKS << NODE_CHUNK_BITS) - 1;

    pub fn new() -> Nodes {
        Nodes { table: Table::new(NODE_CHUNKS, NODE_CHUNK_BITS), next: Alone(AtomicUsize::new(0)), freed: Pool::default() }
    }

    //the nodes made so far, those given back included
    pub fn len(&self) -> usize {
        self.next.load(Ordering::Relaxed)
    }

    pub fn used(&self) -> usize {
        self.len() - self.freed.idle()
    }

    pub fn get(&self, index: usize) -> &Node {
        self.table.get(index)
    }

    //a new node with its position's key and a reference; its index is given to others only once it is published
    pub fn push(&self, key: u64) -> usize {
        let index = match self.freed.take() {
            Some(freed) => freed as usize,
            None => {
                let index = self.next.fetch_add(1, Ordering::Relaxed);
                assert!(index < Self::CAPACITY, "the tree is past its nodes");
                self.table.ensure_chunk(index, || (0..1 << NODE_CHUNK_BITS).map(|_| Node::default()).collect());
                index
            }
        };
        self.table.get(index).recycle(key);
        index
    }

    //dead nodes whose children and room were given back: the last thing done with them
    pub fn free(&self, indices: &mut Vec<u32>) {
        for &index in indices.iter() {
            let node = self.table.get(index as usize);
            debug_assert_eq!(node.references.load(Ordering::Relaxed), 0, "a node freed with references");
            if cfg!(debug_assertions) {
                node.count.store(FREED, Ordering::Relaxed);
            }
        }
        self.freed.give_all(indices);
    }

    #[cfg(test)]
    pub fn freed(&self) -> Vec<u32> {
        self.freed.held()
    }
}

//the first of `count` elements (at most a chunk of `1 << bits`) in one chunk, from `next` and below `capacity`, and the
//elements skipped at the chunk's end before it
fn alloc_run(next: &AtomicUsize, count: usize, bits: u32, capacity: usize) -> (usize, usize) {
    debug_assert!(count > 0 && count <= 1 << bits);
    let mut at = next.load(Ordering::Relaxed);
    let (first, skipped) = loop {
        let first = run_start(at, count, bits);
        match next.compare_exchange_weak(at, first + count, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break (first, first - at),
            Err(now) => at = now,
        }
    };
    assert!(first + count <= capacity, "the tree is past its room");
    (first, skipped)
}

//where a run of `count` elements given out from `at` starts: at the next chunk if it would cross this one's end
fn run_start(at: usize, count: usize, bits: u32) -> usize {
    let chunk_end = (at | ((1 << bits) - 1)) + 1;
    if at + count <= chunk_end { at } else { chunk_end }
}

//runs of elements, each in one chunk: from the runs given back, else from the end of those given out so far
struct Runs {
    next: Alone<AtomicUsize>,
    #[cfg(test)]
    skipped: AtomicUsize, //the elements left unused at chunk ends
    freed: FreeRuns,
    bits: u32,
    capacity: usize,
}

impl Runs {
    fn new(bits: u32, capacity: usize) -> Runs {
        Runs {
            next: Alone(AtomicUsize::new(0)),
            #[cfg(test)]
            skipped: AtomicUsize::new(0),
            freed: FreeRuns::new(bits),
            bits,
            capacity,
        }
    }

    fn alloc(&self, count: usize) -> usize {
        self.freed.take(count).unwrap_or_else(|| {
            let (first, _skipped) = alloc_run(&self.next, count, self.bits, self.capacity);
            #[cfg(test)]
            self.skipped.fetch_add(_skipped, Ordering::Relaxed);
            first
        })
    }

    //the elements given out so far, the gaps at chunk ends and the runs given back included
    fn len(&self) -> usize {
        self.next.load(Ordering::Relaxed)
    }

    //the elements and gaps in use: as much memory
    fn taken(&self) -> usize {
        self.len() - self.freed.idle()
    }

    //the elements in the runs in use
    #[cfg(test)]
    fn used(&self) -> usize {
        self.taken() - self.skipped.load(Ordering::Relaxed)
    }
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
    next: Alone<AtomicUsize>,
    freed: Pool,
    slots: Table<Slot>,
    slot_runs: Runs,
}

impl Links {
    const CAPACITY: usize = LINK_CHUNKS << LINK_CHUNK_BITS;

    pub fn hold_back(&self, held: bool) {
        self.freed.held_back.store(held, Ordering::Relaxed);
        self.slot_runs.freed.held_back.store(held, Ordering::Relaxed);
    }

    //link 0 is NO_LINK's, never given out
    pub fn new() -> Links {
        let links = Links::empty();
        links.make(0);
        links
    }

    //without link 0's chunk
    fn empty() -> Links {
        Links {
            table: Table::new(LINK_CHUNKS, LINK_CHUNK_BITS),
            next: Alone(AtomicUsize::new(1)),
            freed: Pool::default(),
            slots: Table::new(LINK_CHUNKS, LINK_CHUNK_BITS),
            slot_runs: Runs::new(LINK_CHUNK_BITS, Self::CAPACITY),
        }
    }

    //the links given out so far, those given back included
    pub fn len(&self) -> usize {
        self.next.load(Ordering::Relaxed)
    }

    //the links in use
    pub fn used(&self) -> usize {
        self.len() - self.freed.idle()
    }

    //the slots given out so far, the gaps at chunk ends and the runs given back included
    pub fn slots(&self) -> usize {
        self.slot_runs.len()
    }

    //the slots and gaps in use
    pub fn slots_taken(&self) -> usize {
        self.slot_runs.taken()
    }

    //the slots in the runs in use
    #[cfg(test)]
    pub fn slots_used(&self) -> usize {
        self.slot_runs.used()
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

    //a hint that a node's first link or slot is read soon
    pub fn prefetch(&self, children: Children) {
        match children {
            Children::List(head) if head != NO_LINK => self.table.prefetch(head as usize),
            Children::List(_) => {}
            Children::Solid(first) => self.slots.prefetch((first & !SOLID) as usize),
        }
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
        let link = match self.freed.take() {
            Some(freed) => freed as usize,
            None => {
                let link = self.next.fetch_add(1, Ordering::Relaxed);
                assert!(link < Self::CAPACITY, "the tree is past its links");
                self.make(link);
                link
            }
        };
        let made = self.table.get(link);
        made.child.store(child, Ordering::Relaxed);
        made.visits.store(0, Ordering::Relaxed);
        made.next.store(NO_LINK, Ordering::Relaxed);
        made.place.store(place as u32, Ordering::Relaxed);
        link as u32
    }

    //a link that was never put into its list, given back
    pub fn free_link(&self, link: u32) {
        self.freed.give(link);
    }

    //links of lists, given back
    pub fn free(&self, links: &mut Vec<u32>) {
        self.freed.give_all(links);
    }

    //the links of the list from `head`, added to `links` to give back (Links::free)
    pub fn collect_list(&self, head: u32, links: &mut Vec<u32>) {
        let mut link = head;
        while link != NO_LINK {
            links.push(link);
            link = self.get(link).next();
        }
    }

    //`count` solid slots, none with a node: returns the first, SOLID with its index
    pub fn solid(&self, count: usize) -> u32 {
        let first = self.slot_runs.alloc(count);
        self.make_slots(first);
        for slot in self.slots.slice(first, count) {
            slot.child.store(NO_NODE, Ordering::Relaxed);
            slot.visits.store(0, Ordering::Relaxed);
        }
        first as u32 | SOLID
    }

    fn make_slots(&self, first: usize) {
        self.slots.ensure_chunk(first, || (0..1 << LINK_CHUNK_BITS).map(|_| Slot::new()).collect());
    }

    #[cfg(test)]
    pub fn freed_links(&self) -> Vec<u32> {
        self.freed.held()
    }

    #[cfg(test)]
    pub fn freed_slot_runs(&self) -> Vec<(usize, usize)> {
        self.slot_runs.freed.held()
    }

    //a solid node's `count` slots from `first`, given back
    pub fn free_slots(&self, first: u32, count: usize) {
        self.slot_runs.freed.give((first & !SOLID) as usize, count);
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

    //the link of the move at `place` in the list from `head`, which the caller loaded; a list made solid since stays readable
    pub fn find_from(&self, head: u32, place: usize) -> Option<u32> {
        let mut link = head;
        while link != NO_LINK {
            let record = self.get(link);
            match record.place().cmp(&place) {
                std::cmp::Ordering::Equal => return Some(link),
                std::cmp::Ordering::Greater => return None,
                std::cmp::Ordering::Less => link = record.next(),
            }
        }
        None
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

    //a list's link of a copy nothing reads yet
    pub fn write(&self, link: u32, child: u32, visits: u32, place: usize, next: u32) {
        let written = self.get(link);
        written.child.store(child, Ordering::Relaxed);
        written.visits.store(visits, Ordering::Relaxed);
        written.place.store(place as u32, Ordering::Relaxed);
        written.next.store(next, Ordering::Relaxed);
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
    runs: Runs,
}

impl Edges {
    const CAPACITY: usize = EDGE_CHUNKS << EDGE_CHUNK_BITS;

    pub fn hold_back(&self, held: bool) {
        self.runs.freed.held_back.store(held, Ordering::Relaxed);
    }

    pub fn new() -> Edges {
        Edges {
            priors: Table::new(EDGE_CHUNKS, EDGE_CHUNK_BITS),
            moves: Table::new(EDGE_CHUNKS, EDGE_CHUNK_BITS),
            runs: Runs::new(EDGE_CHUNK_BITS, Self::CAPACITY),
        }
    }

    //the edges given out so far, the gaps at chunk ends and the runs given back included
    pub fn len(&self) -> usize {
        self.runs.len()
    }

    //the edges and gaps in use
    pub fn taken(&self) -> usize {
        self.runs.taken()
    }

    //the edges in the runs in use
    #[cfg(test)]
    pub fn used(&self) -> usize {
        self.runs.used()
    }

    //room for a node's `count` edges (at most 255), in one chunk; returns the first
    pub fn alloc(&self, count: usize) -> usize {
        let first = self.runs.alloc(count);
        self.make(first);
        first
    }

    fn make(&self, first: usize) {
        self.priors.ensure_chunk(first, || (0..1 << EDGE_CHUNK_BITS).map(|_| AtomicU32::new(0)).collect());
        self.moves.ensure_chunk(first, || (0..1 << EDGE_CHUNK_BITS).map(|_| AtomicU16::new(0)).collect());
    }

    #[cfg(test)]
    pub fn freed_runs(&self) -> Vec<(usize, usize)> {
        self.runs.freed.held()
    }

    //a dead node's `count` edges from `first`, given back
    pub fn free(&self, first: usize, count: usize) {
        self.runs.freed.give(first, count);
    }

    //a hint that the priors from edge `first` are read soon
    pub fn prefetch(&self, first: usize) {
        self.priors.prefetch(first);
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

    //a node's moves and priors, which come in the order of the moves, as edges from `first` in the order of by_prior; by the thread
    //that owns the node, before its edges are published
    pub fn write(&self, first: usize, moves: &[Move], priors: &[f32]) {
        let slots = self.moves.slice(first, moves.len()).iter().zip(self.priors.slice(first, moves.len()));
        for ((move_slot, prior_slot), key) in slots.zip(by_prior(priors, &mut [0; MOST_MOVES])) {
            let place = (key & 255) as usize;
            move_slot.store(PackedMove::new(moves[place]).bits(), Ordering::Relaxed);
            prior_slot.store(priors[place].to_bits(), Ordering::Relaxed);
        }
    }

    //for a tree being copied
    #[cfg(test)]
    pub fn set(&self, edge: usize, chess_move: PackedMove, prior: f32) {
        self.moves.get(edge).store(chess_move.bits(), Ordering::Relaxed);
        self.priors.get(edge).store(prior.to_bits(), Ordering::Relaxed);
    }

    //`from`'s edges `edges` written from `first`, for a copy nothing reads yet
    pub fn copy_from(&self, first: usize, from: &Edges, edges: Range<usize>) {
        let count = edges.len();
        for (to, from) in self.priors.slice(first, count).iter().zip(from.priors.slice(edges.start, count)) {
            to.store(from.load(Ordering::Relaxed), Ordering::Relaxed);
        }
        for (to, from) in self.moves.slice(first, count).iter().zip(from.moves.slice(edges.start, count)) {
            to.store(from.load(Ordering::Relaxed), Ordering::Relaxed);
        }
    }
}

//where a copy puts its nodes' edges, links and slots, node after node from the arenas' starts as the allocators would give them
//out; the fields are the room given out so far
#[derive(Clone, Copy)]
pub struct Layout {
    edges: usize,
    edges_in_runs: usize,
    links: usize,
    slots: usize,
    slots_in_runs: usize,
}

impl Layout {
    pub fn new() -> Layout {
        Layout { edges: 0, edges_in_runs: 0, links: 1, slots: 0, slots_in_runs: 0 }
    }

    //the next node, with `edges` edges and `links` moves with a node, solid (a slot a move) or listed (a link each); returns its first edge
    //and its children as Node::children has them
    pub fn place(&mut self, edges: usize, links: usize, solid: bool) -> (usize, u32) {
        if edges == 0 {
            return (0, NO_LINK);
        }
        let first_edge = run_start(self.edges, edges, EDGE_CHUNK_BITS);
        (self.edges, self.edges_in_runs) = (first_edge + edges, self.edges_in_runs + edges);
        if solid {
            let first_slot = run_start(self.slots, edges, LINK_CHUNK_BITS);
            (self.slots, self.slots_in_runs) = (first_slot + edges, self.slots_in_runs + edges);
            return (first_edge, first_slot as u32 | SOLID);
        }
        let first_link = if links == 0 { NO_LINK } else { self.links as u32 };
        self.links += links;
        (first_edge, first_link)
    }

    //the edges, links and slots given out so far
    pub fn room(&self) -> (usize, usize, usize) {
        (self.edges, self.links, self.slots)
    }
}

//a copy's arenas, its nodes and the room of its Layout given out from their starts; each chunk is made by one make_chunk
pub struct Fresh {
    pub nodes: Nodes,
    pub edges: Edges,
    pub links: Links,
}

impl Fresh {
    pub fn new(nodes: usize, layout: &Layout) -> Fresh {
        assert!(nodes <= Nodes::CAPACITY && layout.edges <= Edges::CAPACITY, "a copy past the arenas' room");
        assert!(layout.links <= Links::CAPACITY && layout.slots <= Links::CAPACITY, "a copy past the arenas' room");
        let fresh = Fresh { nodes: Nodes::new(), edges: Edges::new(), links: Links::empty() };
        fresh.nodes.next.store(nodes, Ordering::Relaxed);
        fresh.edges.runs.next.store(layout.edges, Ordering::Relaxed);
        fresh.links.next.store(layout.links, Ordering::Relaxed);
        fresh.links.slot_runs.next.store(layout.slots, Ordering::Relaxed);
        #[cfg(test)]
        {
            fresh.edges.runs.skipped.store(layout.edges - layout.edges_in_runs, Ordering::Relaxed);
            fresh.links.slot_runs.skipped.store(layout.slots - layout.slots_in_runs, Ordering::Relaxed);
        }
        fresh
    }

    //the chunks of nodes, edges, links and slots that the room given out lies in
    fn chunk_counts(&self) -> [usize; 4] {
        [
            self.nodes.len().div_ceil(1 << NODE_CHUNK_BITS),
            self.edges.len().div_ceil(1 << EDGE_CHUNK_BITS),
            self.links.len().div_ceil(1 << LINK_CHUNK_BITS),
            self.links.slots().div_ceil(1 << LINK_CHUNK_BITS),
        ]
    }

    pub fn chunks(&self) -> usize {
        self.chunk_counts().iter().sum()
    }

    //the chunk-th of chunks()
    pub fn make_chunk(&self, chunk: usize) {
        let [nodes, edges, links, _] = self.chunk_counts();
        if chunk < nodes {
            self.nodes.table.ensure_chunk(chunk << NODE_CHUNK_BITS, || (0..1 << NODE_CHUNK_BITS).map(|_| Node::default()).collect());
        } else if chunk < nodes + edges {
            self.edges.make((chunk - nodes) << EDGE_CHUNK_BITS);
        } else if chunk < nodes + edges + links {
            self.links.make((chunk - nodes - edges) << LINK_CHUNK_BITS);
        } else {
            self.links.make_slots((chunk - nodes - edges - links) << LINK_CHUNK_BITS);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    //what is out of the pools as (first, count): room given out twice is found at once
    #[derive(Default)]
    struct Out(Mutex<std::collections::BTreeMap<usize, usize>>);

    impl Out {
        fn take(&self, first: usize, count: usize, what: &str) {
            let mut out = self.0.lock().expect("what is out");
            if let Some((&before, &length)) = out.range(..=first).next_back() {
                assert!(before + length <= first, "{what}: {first}+{count} is out while {before}+{length} is");
            }
            if let Some((&after, &length)) = out.range(first..).next() {
                assert!(first + count <= after, "{what}: {first}+{count} is out while {after}+{length} is");
            }
            out.insert(first, count);
        }

        fn give(&self, first: usize, count: usize) {
            assert_eq!(self.0.lock().expect("what is out").remove(&first), Some(count));
        }
    }

    //a draw's turn: take one more of something held, or give one back
    fn churn<T>(held: &mut Vec<T>, draw: u64, make: impl FnOnce() -> T, give: impl FnOnce(T)) {
        if held.len() < 300 && (draw >> 8) % 5 < 3 || held.is_empty() {
            held.push(make());
        } else {
            give(held.swap_remove((draw >> 16) as usize % held.len()));
        }
    }

    fn run_in_one_chunk(first: usize, count: usize, bits: u32) {
        assert_eq!(first >> bits, (first + count - 1) >> bits, "a run across a chunk");
    }

    //a pool or runs that are held back give out nothing, and what they hold is given out once they are not
    #[test]
    fn room_held_back_is_given_out_once_it_is_not() {
        let (pool, runs) = (Pool::default(), FreeRuns::new(8));
        pool.give_all(&mut vec![5, 3]);
        runs.give(100, 40);
        pool.held_back.store(true, Ordering::Relaxed);
        runs.held_back.store(true, Ordering::Relaxed);
        assert_eq!((pool.take(), runs.take(5), pool.idle(), runs.idle()), (None, None, 2, 40));
        pool.held_back.store(false, Ordering::Relaxed);
        runs.held_back.store(false, Ordering::Relaxed);
        assert_eq!((pool.take(), runs.take(5)), (Some(3), Some(100)));
    }

    #[test]
    fn pools_hand_out_each_slot_once() {
        let (nodes, links, edges) = (Nodes::new(), Links::new(), Edges::new());
        let (out_nodes, out_links, out_edges, out_slots) = (Out::default(), Out::default(), Out::default(), Out::default());
        std::thread::scope(|scope| {
            for worker in 0..8u64 {
                let (nodes, links, edges) = (&nodes, &links, &edges);
                let (out_nodes, out_links, out_edges, out_slots) = (&out_nodes, &out_links, &out_edges, &out_slots);
                scope.spawn(move || {
                    let mut state = 0x2545_f491_4f6c_dd1d ^ (worker + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                    let mut draw = move || {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        state
                    };
                    //runs of one to a few, and of any length: exact hits, splits and the table's end
                    let length = |draw: u64| if draw >> 40 & 1 == 0 { 1 + (draw >> 41) as usize % 6 } else { 1 + (draw >> 41) as usize % 218 };
                    let free_node = |index: usize| {
                        out_nodes.give(index, 1);
                        assert!(nodes.get(index).release());
                        nodes.free(&mut vec![index as u32]);
                    };
                    let free_link = |link: usize| {
                        out_links.give(link, 1);
                        links.free_link(link as u32);
                    };
                    let free_edges = |(first, count): (usize, usize)| {
                        out_edges.give(first, count);
                        edges.free(first, count);
                    };
                    let free_slots = |(first, count): (u32, usize)| {
                        out_slots.give((first & !SOLID) as usize, count);
                        links.free_slots(first, count);
                    };
                    let (mut held_nodes, mut held_links, mut held_edges, mut held_slots) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                    for _ in 0..20_000 {
                        let turn = draw();
                        match turn % 4 {
                            0 => churn(
                                &mut held_nodes,
                                turn,
                                || {
                                    let index = nodes.push(turn);
                                    out_nodes.take(index, 1, "nodes");
                                    index
                                },
                                free_node,
                            ),
                            1 => churn(
                                &mut held_links,
                                turn,
                                || {
                                    let link = links.push(NO_NODE, 0) as usize;
                                    out_links.take(link, 1, "links");
                                    link
                                },
                                free_link,
                            ),
                            2 => churn(
                                &mut held_edges,
                                turn,
                                || {
                                    let count = length(turn);
                                    let first = edges.alloc(count);
                                    run_in_one_chunk(first, count, EDGE_CHUNK_BITS);
                                    out_edges.take(first, count, "edges");
                                    (first, count)
                                },
                                free_edges,
                            ),
                            _ => churn(
                                &mut held_slots,
                                turn,
                                || {
                                    let count = length(turn);
                                    let first = links.solid(count);
                                    run_in_one_chunk((first & !SOLID) as usize, count, LINK_CHUNK_BITS);
                                    out_slots.take((first & !SOLID) as usize, count, "slots");
                                    (first, count)
                                },
                                free_slots,
                            ),
                        }
                    }
                    held_nodes.into_iter().for_each(free_node);
                    held_links.into_iter().for_each(free_link);
                    held_edges.into_iter().for_each(free_edges);
                    held_slots.into_iter().for_each(free_slots);
                });
            }
        });
        assert_eq!((nodes.used(), links.used(), edges.used(), links.slots_used()), (0, 1, 0, 0), "room not given back");
    }

    #[test]
    fn freed_runs_join_their_neighbours_but_not_across_a_chunk() {
        //chunks of 16
        let runs = FreeRuns::new(4);
        runs.give(7, 5);
        runs.give(3, 4);
        runs.give(12, 4);
        runs.give(16, 8);
        assert_eq!(runs.held(), [(3, 13), (16, 8)]);
        assert_eq!(runs.idle(), 21);
        assert_eq!(runs.take(14), None);
        assert_eq!(runs.take(9), Some(3));
        assert_eq!(runs.take(4), Some(12));
        assert_eq!(runs.take(8), Some(16));
        assert_eq!((runs.take(1), runs.idle()), (None, 0));
    }

    #[test]
    fn a_run_is_cut_into_pieces_that_come_side_by_side() {
        let runs = FreeRuns::new(8);
        runs.give(100, 40);
        runs.give(150, 3);
        assert_eq!([runs.take(5), runs.take(5), runs.take(5)], [Some(100), Some(105), Some(110)]);
        assert_eq!(runs.idle(), 43 - 15);
        assert_eq!(runs.held(), [(115, 5), (120, 5), (125, 5), (130, 5), (135, 5), (150, 3)]);
        assert_eq!(runs.take(4), None);
    }

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

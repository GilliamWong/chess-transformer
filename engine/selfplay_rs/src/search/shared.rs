//the protocol between threads: edges are published (release) after they are written, a visit is counted (release) after the
//node's value is added, and readers load both with acquire. A move's link is written before it is put into its node's list
//(release); of two threads that put one in for a move, one link is kept. A reader may see a half backed up path, never a torn
//field. A proof rewrites a node's sum and its ancestors', which cannot happen beside backups, so proving a node with visits
//holds the gate exclusively.
//With transpositions a move's visits and the mean it holds are one word (arena's MoveMean), which a thread changes in one
//step: a backup to one more visit at the mean its node has now, a descent that passes to that mean alone. The thread whose
//step it was then adds to the node the move is a move of what the step changed: the move's visits at 1 - its mean after, less
//those before. So every step is in that node's sum once, whatever the threads' order, and when no descent is in progress and
//no leaf waits, an unproven node's sum is its own visits' scores and each of its moves' visits at the mean the move holds. A
//backup counts a node's visit only once the move above it has taken its step, and a move's first visit, which stops the
//descent above a node that has moves and visits, is a step of the descent itself: of the threads that come to the move
//together one stops, the node's visits it found are other moves' or the node's own, and a move never has more visits than its
//node.
//Selection reads a node's moves and none of their nodes: beside its visits and mean a move holds, in a second word (arena's
//MoveState), its node, the descents in flight through it, whether its node is a proven loss and a filter for the repeats. A
//descent that goes on into a move's node counts itself in flight at the move, and at a node only where it may end to evaluate
//it (a node without moves or without visits: arrive), which is what makes one descent the leaf's owner and the others'
//arrivals collisions. Its backup, or the end of the call whose collision it was, takes it off both: with nothing in progress
//and no leaf waiting every count is 0.
//A node keeps a count of the moves (list links and solid slots) that name it, and 1 if it is the root or the root held over the
//last move (reclaim.rs's hold_root). The thread that makes a
//move's node, or takes it from the table (only a node with a count can be taken), gives the move that count. The thread that
//drops the last count owns the node's death (a count never rises from 0) and puts it on the garbage, which sweepers and descents
//short of room free: its table entry, then its children's counts, links, slots and edges, the node last. A descent reaches a
//node only through a move or a table hit that takes a count, so a dead node is read by no one but whoever frees it
//
//Descents in runs (gather_in, walk_moves). A call's descents are still one descent from the root after another, each choosing
//at a node by what the node's moves hold when it looks. But where one look says that the call's next k descents would all
//take the same move (each finding one more in flight there and nothing else changed: more_in_a_row), they go down as a run:
//the thread adds k to the move's count of descents in flight in one step, walks on from the move's node with k descents to
//place, and when the walk comes back takes off the count of those it did not get to. On one thread this is the tree of the
//single descents, number for number (Tree::walk_moves says why; the tests hold it to Tree, which is held to
//gather_by_descents). Beside other threads:
//- What a thread may assume while it walks a run: nothing of the others. Its k are in the move's count until it takes the
//  unused ones off itself, and no one else takes them off. What it read of the node is a snapshot, as a descent's always
//  was; it is now up to k descents old where it was one. Another thread's backup or run that comes between two descents of
//  the run is seen at this thread's next look at the node (after the run, or sooner if the walk comes back early), where
//  single descents saw it at the next descent. The choice below the move is not fixed by the run: every node the run comes
//  to is looked at when it comes, and again for each move after the first.
//- What another thread sees meanwhile: the move's count with the whole run in it from that one step (a count is one word,
//  never half written), so it weighs the move as it would behind k descents on their way. The count may be more than the
//  descents that end up waiting below, until the walk takes the rest off; it is never less than those that wait. A run is
//  never longer than the leaves its round was short of when the walk left the root (gather_in), so a call shows the others
//  no more than it meant to gather then; the round's other calls may take some of those leaves meanwhile.
//- Why every count is exact when all have finished: each one that a run adds to a move has one owner at any moment, who
//  takes it off once. The walk's frame at the move owns the run. Each descent that the walk below reports as over passes
//  from the frame to what a single descent's count always went to: the waiting leaf, whose backup takes one off every move
//  of its path (back_up_means); the call's collisions, taken off when the call ends (end_call); or the backup made at once
//  for a result. The frame takes off the rest, k less those over, under the move's name of that moment (current_link). The
//  same holds at every move above, each of which counted the descent in a run of its own. A node's own count (arrive) is as
//  before: one for each descent that came to end at it, taken off by the same three.
//- Why visits and sums are exact: they are written by backups and by the step into a move's node (meet) alone, and neither
//  has changed. A run makes the step once where its descents made it k times; on one thread the later ones would have found
//  the mean the first left, and where they would not (the move took a new mean, so the node's own mean moved for the move
//  above) the run is one descent and the walk above looks again. Beside other threads a later descent of the run might have
//  taken a mean that another thread's backup made meanwhile; the move then takes it at the next descent that passes, or
//  at the next backup through it, as any move that no descent passes for a while does. No step is lost or made twice.
//- The gate: a thread lets a writer in only where it asks for its next descent (may_start), between two descents as before,
//  and a walk asks before every descent. A walk that has let one in, or has made lists solid itself, goes back up and starts
//  that descent at the root, as a single descent that let a writer in would: a writer may have proven a node the walk is in,
//  below which no descent goes, and no frame decides by what it read before the writer. What the frames hold on the way up
//  are nodes' indices, links' names and counts, nothing read from the tree; a list made solid meanwhile is found by
//  current_link, as a waiting leaf's is, and the solid copy has the move's count with the run in it (make_solid). A search
//  closed meanwhile (close) gives no descent leave to start, and the walk goes back up taking its runs off.
//- Compaction, a moved or held root and the sweepers work on a tree no call is in, or on what the root no longer reaches:
//  a run does not outlive its call, so they find every count as single descents left it.

mod arena;
mod cache;
mod compact;
mod reclaim;
mod reuse;
#[cfg(test)]
mod audit;
#[cfg(test)]
mod descents;

use arena::{Alone, Children, Edges, Link, Links, MoveRecord, MoveState, Node, Nodes, Slot, units};
pub use cache::SharedCache;
use reclaim::Garbage;
use std::sync::atomic::{AtomicU32, AtomicU64};
use std::sync::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::thread::JoinHandle;

use super::*;

pub struct Hold<'a> {
    tree: &'a SharedTree,
    guard: Option<RwLockReadGuard<'a, ()>>,
}

impl Hold<'_> {
    //takes the share again only once the waiting thread has had the tree: the gate is not fair, a share taken at once passes it.
    //Returns whether a writer was let in
    pub fn give_way(&mut self) -> bool {
        let waited_for = self.tree.writers.load(Ordering::Relaxed) > 0;
        if waited_for {
            self.tree.counts.gave_way.fetch_add(1, Ordering::Relaxed);
            self.guard = None;
            while self.tree.writers.load(Ordering::Acquire) > 0 {
                std::thread::yield_now();
            }
            self.guard = Some(self.tree.gate.read().expect("the gate"));
        }
        waited_for
    }
}

const NODE_BYTES: usize = size_of::<Node>();
//an edge store's unit (arena::Edges)
const EDGE_BYTES: usize = size_of::<u16>();
const TABLE_ENTRY_BYTES: usize = 32;
const LINK_BYTES: usize = size_of::<Link>();
const SLOT_BYTES: usize = size_of::<Slot>();
//the most a descent allocates: a node, its move's link and a leaf's edges
pub const DESCENT_BYTES: usize = NODE_BYTES + LINK_BYTES + units(MOST_MOVES) * EDGE_BYTES;

const SOLID_BATCH: usize = 32;

//the most calls that descend at once, which the room left is shared out by
const MOST_CALLS: usize = 64;

//the leaves a call has room for from the start: rounds are of a few dozen
const LEAVES_AT_ONCE: usize = 64;

//the nodes a descent short of room frees at a time
const ASSIST_BATCH: usize = 64;

const TABLE_SHARDS: usize = 256;

//under a memory limit, the room freed is kept until the arenas have grown to the limit: what a search makes then lies together at
//their ends, which is faster to search than it scattered over the holes a move leaves. Without a limit it is given out at once, and
//so it is once the node table is three quarters full, which a tree of few moves a node reaches before a large limit
pub(super) fn holds_back_freed_room(memory_limit: usize, footprint_bytes: usize, nodes: usize) -> bool {
    memory_limit != usize::MAX && footprint_bytes < memory_limit && nodes < Nodes::CAPACITY / 4 * 3
}

struct SharedGraph {
    shards: Box<[Mutex<KeyMap<u32>>]>,
    entries: Alone<AtomicUsize>,
}

impl SharedGraph {
    fn new() -> SharedGraph {
        SharedGraph { shards: (0..TABLE_SHARDS).map(|_| Mutex::new(KeyMap::default())).collect(), entries: Alone(AtomicUsize::new(0)) }
    }

    fn shard(&self, entry: u64) -> &Mutex<KeyMap<u32>> {
        &self.shards[(entry >> 56) as usize % TABLE_SHARDS]
    }

    //the entry's node is `id`'s no more; an entry made over meanwhile by a new node stays
    fn remove_if(&self, entry: u64, id: u32) {
        let mut shard = self.shard(entry).lock().expect("a table shard");
        if shard.get(&entry) == Some(&id) {
            shard.remove(&entry);
            self.entries.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

//how a walk left the descents it was given (SharedTree::walk_moves), as Tree's Walk
enum Walk {
    //all of them wait, for the network or as collisions
    Quiet,
    //those it got to wait, and a move on their way took a new mean in passing: the node above looks again before the next
    Stirred,
    //one was finished at once: the next descent starts at the root
    Interrupted,
    //the call has what it may gather
    Ended,
}

//how a descent ended at a node without moves (SharedTree::evaluate_leaf)
enum Evaluated {
    //it is the round's leaf, waiting for the network
    Waits,
    //the node is another descent's to evaluate: a collision
    AnothersLeaf,
    //a game's result or a cached answer, backed up
    AtOnce,
    //the round's last leaf went to another of its calls meanwhile: the descent leaves no trace and the call ends
    NotTaken,
}

//a gathering call's walk (SharedTree::gather_in)
struct Walker<'a, 'c, 'h> {
    hold: &'c mut Hold<'a>,
    round: Round,
    together: &'c Together,
    settings: SearchSettings,
    tables: Option<&'c Tables>,
    cache: Option<&'c SharedCache>,
    boards: &'c mut Vec<u8>,
    call: &'c mut Call<'h>,
    root_covered: bool,
    root_keys: usize,
    //the walk's place: the nodes from the root, and with transpositions the links between them
    path: Vec<usize>,
    edges: Vec<usize>,
    //the moves from the root to the walk's place (edges), and the position after the first `played` of them: a position is
    //played out only where one is needed (a move's node to make, a pass of known moves, a leaf to evaluate). STALE once the
    //walk has left the line the position was played along
    line: Vec<usize>,
    board: Chess,
    played: usize,
    //child_node's moves to play: none, the position is played before
    no_moves: Vec<usize>,
    priors: Vec<f32>,
    //the descents the call may still make without reserving their room (free_descents)
    free_descents: usize,
    //whether the descent now on its way was let start (may_start), and whether it reserved its room
    started: bool,
    reserved: bool,
    //whether a writer has had the tree since the walk left the root (may_start): the descent on its way starts there again
    writer_let_in: bool,
}

const STALE: usize = usize::MAX;

impl Walker<'_, '_, '_> {
    //the walk starts again at the root
    fn forget_line(&mut self) {
        self.line.clear();
        if self.played != 0 {
            self.played = STALE;
        }
    }

    //the walk goes back up over its last move
    fn drop_move(&mut self) {
        self.line.pop();
        if self.played > self.line.len() {
            self.played = STALE;
        }
    }

    //the position at the walk's place, in `board`
    fn play_line(&mut self, tree: &SharedTree) {
        if self.played == STALE {
            (self.board, self.played) = (tree.board.clone(), 0);
        }
        for &edge in &self.line[self.played..] {
            let chess_move = tree.edges.chess_move(edge).to_move(&self.board);
            self.board.play_unchecked(chess_move);
        }
        self.played = self.line.len();
    }
}

enum Arrival {
    Inner,
    Proven(Proof),
    //a node that waits for the network or is being expanded
    Collision,
    //a node to evaluate. Owned: a new leaf, or a node without moves that no other descent is at. Not owned: a node without
    //moves that has visits, with other descents at it. That is a game's end or repeat reached again, or else a collision: a
    //node that waits for the network, with transpositions one that paths' repeats have visited
    Leaf { owned: bool },
    //with transpositions, above the node the path's last move leads to (not counted in flight there): a repeat of a
    //position on the path, or a move's first visit of a node other paths made (Tree::stops_at), which the descent has
    //counted (meet)
    Repeat,
    FirstVisit,
}

struct Call<'a> {
    simulations: u32,
    collisions: usize,
    //the nodes that a collision's descent counted itself in flight at (SharedTree::in_flight_nodes), and with
    //transpositions its path's moves (each as its node and link), which are in flight until the call ends: once for each of
    //the descents that collided there in a row (`collided_runs`: where the run's nodes and moves end, and its descents)
    collided: Vec<usize>,
    collided_moves: Vec<(usize, u32)>,
    collided_runs: Vec<(usize, usize, u32)>,
    leaves: Vec<PendingLeaf>,
    //for the call that began a round others may help gather: once it has made that many simulations, it calls for them
    help: Option<(u32, &'a dyn Fn())>,
}

impl<'a> Call<'a> {
    fn new(leaves: Vec<PendingLeaf>, help: Option<(u32, &'a dyn Fn())>) -> Call<'a> {
        Call { simulations: 0, collisions: 0, collided: Vec::new(), collided_moves: Vec::new(), collided_runs: Vec::new(), leaves, help }
    }

    //a descent that ended on a node another one owns stays in flight until the call ends, at the nodes `in_flight` and at its
    //path's moves (`links`). `run`: the run of the descent before it, if that one collided on the same path with nothing
    //between: the descent is then one more of that run
    fn collide(&mut self, path: &[usize], links: &[usize], in_flight: &[usize], run: &mut Option<usize>) {
        self.collisions += 1;
        if let Some(run) = run.filter(|&run| run + 1 == self.collided_runs.len()) {
            self.collided_runs[run].2 += 1;
            return;
        }
        self.collided.extend_from_slice(in_flight);
        self.collided_moves.extend(path.iter().zip(links).map(|(&node, &link)| (node, link as u32)));
        *run = Some(self.collided_runs.len());
        self.collided_runs.push((self.collided.len(), self.collided_moves.len(), 1));
    }
}

//what the calls that gather one round together count in all: the round's leaves and simulations are theirs together, and the others
//stop once the call that began the round has ended it
#[derive(Default)]
pub struct Together {
    leaves: AtomicU32,
    simulations: AtomicU32,
    ended: AtomicBool,
}

impl Together {
    pub fn end(&self) {
        self.ended.store(true, Ordering::Release);
    }
}

#[derive(Default)]
pub struct Counts {
    pub gave_way: AtomicU64,
    pub exclusive: AtomicU64,
    pub orphans: AtomicU64,
    //nodes freed, and the times a descent short of room freed some
    pub reclaimed: AtomicU64,
    pub assists: AtomicU64,
}

pub struct SharedTree {
    pub board: Chess,
    pub keys: Vec<u64>, //the position_key of every position of the game, the root's last
    root: usize,
    nodes: Nodes,
    edges: Edges,
    links: Links,
    to_solidify: Mutex<Vec<u32>>,
    queued: AtomicUsize,
    //what is dead and not yet freed
    garbage: Mutex<Garbage>,
    //the first links of lists that were made solid since the last move: waiting leaves may still read them
    retired: Mutex<Vec<u32>>,
    sweepers: Mutex<Vec<JoinHandle<()>>>,
    solid_threshold: u32,
    solid_batch: usize,
    pending: Mutex<HashMap<u64, Vec<PendingLeaf>>>,
    //network leaves gathered and not yet backed up
    waiting: Alone<AtomicUsize>,
    //bytes that the descents in progress that reserved room may still allocate (under a memory limit); a waiting leaf's are
    //counted by `waiting`
    reserved: AtomicUsize,
    //no room for another simulation: the search ends as at its simulations, until a move is played
    full: AtomicBool,
    closed: AtomicBool,
    //calls hold the gate shared, whatever needs the tree to itself holds it exclusive; writers counts the threads waiting
    gate: RwLock<()>,
    writers: AtomicUsize,
    pub counts: Alone<Counts>,
    graph: Option<SharedGraph>,
    //the leaves sent to the network since the search began (begin_search)
    evaluations: Alone<AtomicU64>,
}

//the tests that look for races set it, and jitter then yields now and then between a protocol's steps to widen the windows
#[cfg(test)]
pub static JITTER: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
fn jitter() {
    use std::cell::Cell;
    if !JITTER.load(Ordering::Relaxed) {
        return;
    }
    thread_local!(static STATE: Cell<u64> = const { Cell::new(0x2545_f491_4f6c_dd1d) });
    let draw = STATE.with(|state| {
        let mut x = state.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        state.set(x);
        x
    });
    if draw % 4 == 0 {
        std::thread::yield_now();
    }
}

#[cfg(not(test))]
#[inline(always)]
fn jitter() {}

//a test's writer, let in where a call asks for its next descent (may_start) as another thread's would be: it returns whether it
//has had the tree
#[cfg(test)]
thread_local!(pub static WRITER_AT_START: std::cell::Cell<Option<fn(&SharedTree) -> bool>> = const { std::cell::Cell::new(None) });

//a test's step between a reader's load of a node's children and what it does with them, to change the node as another thread could
#[cfg(test)]
thread_local!(pub static AFTER_CHILDREN_LOADED: std::cell::Cell<Option<fn(&SharedTree, &Node)>> = const { std::cell::Cell::new(None) });

impl SharedTree {
    pub fn new(board: Chess, keys: Vec<u64>) -> SharedTree {
        let root_key = *keys.last().expect("the root's key");
        let nodes = Nodes::new();
        let root = nodes.push(root_key);
        SharedTree::with_arenas(board, keys, root, nodes, Edges::new(), Links::new(), None)
    }

    fn with_arenas(
        board: Chess,
        keys: Vec<u64>,
        root: usize,
        nodes: Nodes,
        edges: Edges,
        links: Links,
        graph: Option<SharedGraph>,
    ) -> SharedTree {
        SharedTree {
            board,
            keys,
            root,
            nodes,
            edges,
            links,
            to_solidify: Mutex::new(Vec::new()),
            queued: AtomicUsize::new(0),
            garbage: Mutex::new(Garbage::default()),
            retired: Mutex::new(Vec::new()),
            sweepers: Mutex::new(Vec::new()),
            solid_threshold: SOLID_TREE_THRESHOLD,
            solid_batch: SOLID_BATCH,
            pending: Mutex::new(HashMap::new()),
            waiting: Alone(AtomicUsize::new(0)),
            reserved: AtomicUsize::new(0),
            full: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            gate: RwLock::new(()),
            writers: AtomicUsize::new(0),
            counts: Alone(Counts::default()),
            graph,
            evaluations: Alone(AtomicU64::new(0)),
        }
    }

    //one node a position, from a tree's start
    pub fn with_transpositions(mut self, transpositions: bool) -> SharedTree {
        assert_eq!(self.nodes.len(), 1, "transpositions from a tree's start");
        if transpositions {
            let graph = SharedGraph::new();
            self.root().set_halfmoves(halfmove_count(self.board.halfmoves()));
            let entry = graph_key(self.root().key(), u32::from(self.root().halfmoves()));
            graph.shard(entry).lock().expect("a table shard").insert(entry, self.root as u32);
            graph.entries.fetch_add(1, Ordering::Relaxed);
            self.graph = Some(graph);
        }
        self
    }

    //a search begins in the tree: its evaluations are counted from here
    pub fn begin_search(&mut self) {
        *self.evaluations.0.get_mut() = 0;
    }

    //the leaves sent to the network since the search began, those waiting for their answers included
    pub fn evaluations(&self) -> u64 {
        self.evaluations.load(Ordering::Relaxed)
    }

    pub fn root(&self) -> &Node {
        self.nodes.get(self.root)
    }

    pub fn has_transpositions(&self) -> bool {
        self.graph.is_some()
    }

    pub fn is_full(&self) -> bool {
        self.full.load(Ordering::Relaxed)
    }

    //the bytes in use: what is dead and not yet freed counts, what was freed does not
    pub fn allocated_bytes(&self) -> usize {
        let graph = self.graph.as_ref().map_or(0, |graph| graph.entries.load(Ordering::Relaxed) * TABLE_ENTRY_BYTES);
        self.nodes.used() * NODE_BYTES
            + self.edges.taken() * EDGE_BYTES
            + self.links.used() * LINK_BYTES
            + self.links.slots_taken() * SLOT_BYTES
            + graph
    }

    //the bytes the arenas have been grown to: freed room stays, to be given out again
    pub fn footprint_bytes(&self) -> usize {
        self.nodes.len() * NODE_BYTES
            + self.edges.len() * EDGE_BYTES
            + self.links.len() * LINK_BYTES
            + self.links.slots() * SLOT_BYTES
    }

    //whether a search gives out the room a move freed, scattered among what the root reaches: under a memory limit, once the arenas have
    //grown to it (holds_back_freed_room)
    pub fn gives_out_freed_room(&self, memory_limit: usize) -> bool {
        memory_limit != usize::MAX && !holds_back_freed_room(memory_limit, self.footprint_bytes(), self.nodes.len())
    }

    pub fn pending(&self) -> usize {
        self.waiting.load(Ordering::Acquire)
    }

    //the bytes that descents in progress and waiting leaves may still allocate: the most a leaf's moves take
    pub fn reserved_bytes(&self) -> usize {
        self.reserved.load(Ordering::Acquire) + self.pending() * units(MOST_MOVES) * EDGE_BYTES
    }

    pub fn hold(&self) -> Hold<'_> {
        Hold { tree: self, guard: Some(self.gate.read().expect("the gate")) }
    }

    fn writer(&self) -> RwLockWriteGuard<'_, ()> {
        self.writers.fetch_add(1, Ordering::AcqRel);
        let writer = self.gate.write().expect("the gate");
        self.writers.fetch_sub(1, Ordering::AcqRel);
        writer
    }

    //runs `work` with the tree to itself, once every call has left it. The caller's share is given up first and taken again
    //after: a thread waiting for the gate while holding a share would never get it
    fn exclusive<'a>(&'a self, hold: &mut Hold<'a>, work: impl FnOnce()) {
        hold.guard = None;
        let writer = self.writer();
        self.counts.exclusive.fetch_add(1, Ordering::Relaxed);
        work();
        drop(writer);
        hold.guard = Some(self.gate.read().expect("the gate"));
    }

    //the tree to itself for a caller that holds no share
    pub fn quiet<R>(&self, work: impl FnOnce() -> R) -> R {
        let _writer = self.writer();
        work()
    }

    pub fn chess_move(&self, edge: usize) -> PackedMove {
        self.edges.chess_move(edge)
    }

    //the link of the parent's move `edge`, NO_LINK until a simulation has gone through it
    fn link_of(&self, parent: &Node, edge: usize) -> u32 {
        let place = edge - parent.edges().start;
        //loaded once: a list made solid meanwhile stays readable, and this is its head
        let children = parent.children();
        #[cfg(test)]
        if let Some(step) = AFTER_CHILDREN_LOADED.get() {
            step(self, parent);
        }
        match children {
            Children::Solid(first) => Some(first + place as u32).filter(|&link| self.links.child(link) != NO_NODE).unwrap_or(NO_LINK),
            Children::List(head) => self.links.find_from(head, place).unwrap_or(NO_LINK),
        }
    }

    pub fn child_index(&self, parent: &Node, edge: usize) -> Option<usize> {
        let link = self.link_of(parent, edge);
        (link != NO_LINK).then(|| self.links.child(link) as usize)
    }

    pub fn child(&self, parent: &Node, edge: usize) -> Option<&Node> {
        self.child_index(parent, edge).map(|child| self.nodes.get(child))
    }

    //a move's visits by its link: its node's, or with transpositions the move's own (its node counts every path's)
    fn link_visits(&self, link: u32) -> u32 {
        match (link, &self.graph) {
            (NO_LINK, _) => 0,
            (_, Some(_)) => self.links.slot(link).held().load().0,
            (_, None) => self.nodes.get(self.links.child(link) as usize).visits(),
        }
    }

    pub fn visits(&self, parent: &Node, edge: usize) -> u32 {
        self.link_visits(self.link_of(parent, edge))
    }

    //a move whose child is a proven loss for the side making it
    fn proven_lost(&self, parent: &Node, edge: usize) -> bool {
        self.child(parent, edge).is_some_and(|child| child.proves_a_loss())
    }

    pub fn two_most_visits(&self) -> (u32, u32) {
        let (mut most, mut second) = (0, 0);
        let root = self.root();
        for visits in root.edges().filter(|&edge| !self.proven_lost(root, edge)).map(|edge| self.visits(root, edge)) {
            if visits > most {
                (most, second) = (visits, most);
            } else if visits > second {
                second = visits;
            }
        }
        (most, second)
    }

    fn most_visited_edge(&self, node: usize) -> usize {
        let parent = self.nodes.get(node);
        let preference = |edge: usize| {
            let visits = self.visits(parent, edge);
            let child = self.child(parent, edge);
            //the node's own mean: with transpositions a move's visits are not its node's
            let mean_score = child.filter(|_| visits > 0).map_or(0.0, |child| child.mean_score(child.visits()));
            let proven = match child.and_then(|child| child.proof()) {
                Some(proof) if proof.outcome != Outcome::Draw => proof_rank(&proof),
                _ => (Outcome::Draw, 0),
            };
            (proven, visits, mean_score, f64::from(self.edges.prior(parent.edges(), edge)))
        };
        let mut edges = parent.edges();
        let mut best = edges.next().expect("a searched node has children");
        for edge in edges {
            if preference(edge) > preference(best) {
                best = edge;
            }
        }
        best
    }

    pub fn principal_variation(&self) -> Vec<usize> {
        let (mut line, mut node) = (Vec::new(), self.root);
        while self.nodes.get(node).has_edges() {
            let edge = self.most_visited_edge(node);
            let parent = self.nodes.get(node);
            if !line.is_empty() && self.visits(parent, edge) == 0 {
                break;
            }
            line.push(edge);
            let Some(child) = self.child_index(parent, edge) else { break };
            node = child;
        }
        line
    }

    //the root's edges, the most visited first and ties in their order
    pub fn edges_by_visits(&self) -> Vec<usize> {
        let root = self.root();
        let mut edges: Vec<usize> = root.edges().collect();
        edges.sort_by(|&a, &b| self.visits(root, b).cmp(&self.visits(root, a)));
        edges
    }

    fn proof_from_moves(&self, node: usize) -> Option<Proof> {
        let node = self.nodes.get(node);
        let moves = node.edges().len();
        if moves == 0 {
            return None;
        }
        let (mut best, mut proven): (Option<Proof>, usize) = (None, 0);
        for (_, link) in self.links.linked(node) {
            if let Some(proof) = self.nodes.get(self.links.child(link) as usize).proof() {
                proven += 1;
                if best.is_none_or(|best| proof_rank(&proof) > proof_rank(&best)) {
                    best = Some(proof);
                }
            }
        }
        let best = best?;
        let outcome = match best.outcome {
            Outcome::Win => Outcome::Loss,
            _ if proven != moves => return None,
            Outcome::Draw => Outcome::Draw,
            Outcome::Loss => Outcome::Win,
        };
        Some(Proof { outcome, value: 1.0 - best.value, plies: best.plies + 1 })
    }

    pub fn root_decided(&self) -> bool {
        self.proof_from_moves(self.root).is_some()
    }

    //whether at most one of the root's moves is not a proven loss (it may be unproven); false before the root has moves
    pub fn one_move_left(&self) -> bool {
        let root = self.root();
        root.has_edges() && root.edges().filter(|&edge| !self.proven_lost(root, edge)).take(2).count() < 2
    }

    //Tree::select_edge beside other threads: with transpositions the node's moves are weighed by what they hold (move_value),
    //and no move's node is read. Returns the edge and its link (NO_LINK for a list's move without one, a solid slot whether
    //it has a node or not)
    fn select_edge(&self, id: usize, settings: &SearchSettings, repeats: Option<&Repeats>) -> (usize, u32) {
        let parent = self.nodes.get(id);
        let scan = self.scan(parent, settings);
        let (mut best, mut best_u, mut best_link) = (0, f64::NEG_INFINITY, NO_LINK);
        self.each_weighed(id, |index, link, prior, state, slot| {
            if let Some(u) = self.move_value(&scan, repeats, prior, state, slot)
                && u > best_u
            {
                (best, best_u, best_link) = (index, u, link);
                self.prefetch_move(&scan, state.child());
            }
        });
        (parent.edges().start + best, best_link)
    }

    //Tree::select_with_rivals: select_edge for a node that several descents in a row come to (walk_moves), with the scan the
    //moves were weighed by and the highest values among the moves weighed before the chosen one and among those after it
    fn select_with_rivals(&self, id: usize, settings: &SearchSettings, repeats: Option<&Repeats>) -> (usize, u32, Scan, (f64, f64)) {
        let parent = self.nodes.get(id);
        let scan = self.scan(parent, settings);
        let (mut best, mut best_u, mut best_link) = (0, f64::NEG_INFINITY, NO_LINK);
        let (mut before, mut after) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        self.each_weighed(id, |index, link, prior, state, slot| {
            let Some(u) = self.move_value(&scan, repeats, prior, state, slot) else { return };
            if u > best_u {
                (before, after) = (before.max(best_u).max(after), f64::NEG_INFINITY);
                (best, best_u, best_link) = (index, u, link);
                self.prefetch_move(&scan, state.child());
            } else {
                after = after.max(u);
            }
        });
        (parent.edges().start + best, best_link, scan, (before, after))
    }

    //Tree::rivals: select_with_rivals' two values for the move at `place`, None if selection does not weigh the move
    fn rivals(&self, id: usize, scan: &Scan, repeats: Option<&Repeats>, place: usize) -> Option<(f64, f64)> {
        let (mut before, mut after, mut met) = (f64::NEG_INFINITY, f64::NEG_INFINITY, false);
        self.each_weighed(id, |index, _, prior, state, slot| {
            if index == place {
                met = true;
            } else if let Some(u) = self.move_value(scan, repeats, prior, state, slot) {
                if met { after = after.max(u) } else { before = before.max(u) }
            }
        });
        met.then_some((before, after))
    }

    //Tree::more_in_a_row: how many more descents in a row select_edge would send down the node's move at `place` (its name
    //`link`) after one that is not counted there yet, at most `most`, by what the move holds now: each finds one more in
    //flight there, and the move is taken while its value stays above the best before it and not below the best after it
    fn more_in_a_row(&self, id: usize, scan: &Scan, repeats: Option<&Repeats>, (place, link): (usize, u32), rivals: (f64, f64), most: u32) -> u32 {
        let slot = self.links.slot(link);
        let (prior, state) = (self.edges.priors(self.nodes.get(id).edges()).get(place), slot.state());
        let holds = |more: u32| {
            self.move_value_after(scan, repeats, prior, state, slot, more).is_some_and(|u| u > rivals.0 && u >= rivals.1)
        };
        if most == 0 || !holds(1) {
            return 0;
        }
        if holds(most) {
            return most;
        }
        let (mut held, mut failed) = (1, most);
        while held * 2 < failed && holds(held * 2) {
            held *= 2;
        }
        failed = failed.min(held * 2);
        while failed - held > 1 {
            let middle = held + (failed - held) / 2;
            if holds(middle) { held = middle } else { failed = middle }
        }
        held
    }

    //the node's moves that selection weighs, in the order it weighs them: `weigh` is given each one's place among the node's
    //moves, its link (NO_LINK for a list's move without one), its prior, its word and its slot (a slot without a node for a
    //move that has none)
    #[inline(always)]
    fn each_weighed(&self, id: usize, mut weigh: impl FnMut(usize, u32, f64, MoveState, &Slot)) {
        let parent = self.nodes.get(id);
        let edges = parent.edges();
        //weighs the move at `index`; returns its node, NO_NODE for none
        let mut consider = |index: usize, link: u32, prior: f64, slot: &Slot| {
            let state = slot.state();
            weigh(index, link, prior, state, slot);
            state.child()
        };
        let priors = self.edges.priors(edges.clone());
        let heads = priors.heads.len();
        //the head's moves, then the tail's: of each one's moves without a node the first has the highest prior. A tail move is looked at
        //only with a prior above the head's first move without a node: moves without one score alike but for their priors, and a tail
        //move got its node so (the head's first prior without a node only falls)
        let mut head_unvisited = f64::NEG_INFINITY;
        match parent.children() {
            Children::Solid(first) => {
                let slots = self.links.solid_slots(first, edges.len());
                //the head's moves the scan passes over after its first move without a node (from skipped_from to the head's
                //end), and where it stops in the tail
                let (mut skipped_from, mut stopped_at) = (usize::MAX, edges.len());
                for (index, (code, slot)) in priors.heads.iter().zip(slots).enumerate() {
                    let prior = f64::from(head_prior(code.load(Ordering::Relaxed)));
                    if consider(index, first + index as u32, prior, slot) == NO_NODE {
                        (head_unvisited, skipped_from) = (prior, index + 1);
                        break;
                    }
                }
                for index in heads..edges.len() {
                    let prior = priors.tail(index - heads);
                    if prior <= head_unvisited {
                        stopped_at = index;
                        break;
                    }
                    if consider(index, first + index as u32, prior, &slots[index]) == NO_NODE {
                        stopped_at = index + 1;
                        break;
                    }
                }
                //the moves link_known_moves gave a node may stand among those the scan passed over: each is looked at too
                //(Tree::select_edge's). A move without a node there is behind one of a higher prior that was weighed
                let known_to = parent.known_to();
                for index in (skipped_from..heads.min(known_to)).chain(stopped_at.max(heads)..known_to) {
                    let slot = &slots[index];
                    if slot.child() != NO_NODE {
                        consider(index, first + index as u32, priors.get(index), slot);
                    }
                }
            }
            //every link is weighed, in the moves' order: a move that link_known_moves gave a node may stand anywhere among
            //those without one (Tree::select_edge's). Of the moves without a node between the links, from `from` up to the
            //next link's, the head's first and the tail's first are weighed, the tail's only with a prior above the head's
            Children::List(head) => {
                let unvisited = self.links.no_slot();
                let (mut link, mut from) = (head, 0);
                let (mut head_weighed, mut tail_weighed) = (false, false);
                loop {
                    let record = (link != NO_LINK).then(|| self.links.get(link));
                    let to = record.map_or(edges.len(), |record| record.place());
                    if from < to.min(heads) && !head_weighed {
                        (head_weighed, head_unvisited) = (true, priors.get(from));
                        consider(from, NO_LINK, head_unvisited, unvisited);
                    }
                    let tail = from.max(heads);
                    if tail < to && !tail_weighed {
                        tail_weighed = true;
                        let prior = priors.tail(tail - heads);
                        if prior > head_unvisited {
                            consider(tail, NO_LINK, prior, unvisited);
                        }
                    }
                    let Some(record) = record else { break };
                    consider(to, link, priors.get(to), record.slot());
                    (link, from) = (record.next(), to + 1);
                }
            }
        }
    }

    //asks for the lines a descent through the move's node `child` reads first, while the node's other moves are weighed:
    //without transpositions its edges and moves (its own line was read to weigh it), with them its own line
    #[inline(always)]
    fn prefetch_move(&self, scan: &Scan, child: u32) {
        if child == NO_NODE {
            return;
        }
        if scan.transpositions {
            return self.nodes.prefetch(child as usize);
        }
        let child = self.nodes.get(child as usize);
        let edges = child.edges();
        if !edges.is_empty() {
            self.edges.prefetch(edges);
            self.links.prefetch(child.children());
        }
    }

    //what select_edge's scores of the node's moves are made of
    fn scan(&self, parent: &Node, settings: &SearchSettings) -> Scan {
        let (parent_visits, _) = parent.counts();
        let c_puct =
            settings.c_puct + settings.c_puct_factor * ((f64::from(parent_visits) + settings.c_puct_base) / settings.c_puct_base).ln();
        Scan {
            c_puct,
            sqrt_children_visits: f64::from(parent_visits.saturating_sub(1).max(1)).sqrt(),
            first_play_urgency: (1.0 - parent.mean_score(parent_visits)) - settings.fpu_reduction * parent.visited_policy_sqrt(),
            noised: false,
            transpositions: self.graph.is_some(),
            draw: settings.draw,
        }
    }

    //Tree::move_value: select_edge's value of a move from its prior, its word `state` and its slot (a slot without a node for
    //a move that has none); None for a proven loss, never chosen. Without transpositions from the node's own numbers. With
    //them from what the move holds, its node not read: the mean it holds, or as a move without a node while it holds none,
    //over its own visits and the descents in flight through it. Only a move whose filter says its node may repeat a position
    //of the path has the node's key read
    #[inline(always)]
    fn move_value(&self, scan: &Scan, repeats: Option<&Repeats>, prior: f64, state: MoveState, slot: &Slot) -> Option<f64> {
        self.move_value_after(scan, repeats, prior, state, slot, 0)
    }

    //move_value of a move that `more` descents have gone down since its counts were read, each in flight there
    //(more_in_a_row): of a move with a node, as only such a move has been gone down
    #[inline(always)]
    fn move_value_after(&self, scan: &Scan, repeats: Option<&Repeats>, prior: f64, state: MoveState, slot: &Slot, more: u32) -> Option<f64> {
        let id = state.child();
        if id == NO_NODE {
            //sqrt_children_visits / 1.0 is sqrt_children_visits: no division
            return Some(scan.first_play_urgency + (scan.c_puct * prior * scan.sqrt_children_visits));
        }
        if !scan.transpositions {
            let child = self.nodes.get(id as usize);
            if child.proves_a_loss() {
                return None;
            }
            let (visits, in_flight) = child.counts();
            let q = if visits > 0 { child.mean_score(visits) } else { scan.first_play_urgency };
            return Some(q + (scan.c_puct * prior * (scan.sqrt_children_visits / f64::from(1 + visits + in_flight + more))));
        }
        if state.lost() {
            return None;
        }
        let (visits, mean) = slot.held().load();
        let repeat = repeats.is_some_and(|repeats| {
            repeats.may_repeat(state.repeats()) && {
                let child = self.nodes.get(id as usize);
                repeats.repeats(child.key(), child.halfmoves())
            }
        });
        let q = if repeat {
            1.0 - scan.draw
        } else if mean == NO_MEAN {
            scan.first_play_urgency
        } else {
            f64::from(mean)
        };
        Some(q + (scan.c_puct * prior * (scan.sqrt_children_visits / f64::from(1 + visits + state.pending() + more))))
    }

    //plays the moves of the path (edges) that `board` has not seen yet: a descent plays them only where it needs the position
    fn play_path(&self, board: &mut Chess, unplayed: &mut Vec<usize>) {
        for edge in unplayed.drain(..) {
            let chess_move = self.edges.chess_move(edge).to_move(board);
            board.play_unchecked(chess_move);
        }
    }

    //the node and link of the parent's move `edge` (select_edge's link), made if the move has none yet; the node's position is
    //board with the path's moves `unplayed` played. Of two threads that make one, one node is kept and the other's stays
    //unused. With transpositions it is the table's node of the position, made by one thread under the shard's lock
    fn child_node(&self, parent: usize, edge: usize, link: u32, board: &mut Chess, unplayed: &mut Vec<usize>) -> (usize, u32) {
        //select_edge names no move when it chose none (every move a proven loss, the first taken): its node if it has one
        let link = if link == NO_LINK { self.link_of(self.nodes.get(parent), edge) } else { link };
        if link != NO_LINK && self.links.child(link) != NO_NODE {
            return (self.links.child(link) as usize, link);
        }
        self.play_path(board, unplayed);
        let (key, halfmoves) = (position_key(board), halfmove_count(board.halfmoves()));
        let repeats = repeat_filter(key, halfmoves);
        let made = match &self.graph {
            None => self.nodes.push(key),
            Some(graph) => {
                let entry = graph_key(key, u32::from(halfmoves));
                let mut shard = graph.shard(entry).lock().expect("a table shard");
                //a node with no reference left is dead: it is made again, and the entry now names the new one
                match shard.get(&entry).copied().filter(|&shared| self.nodes.get(shared as usize).acquire()) {
                    Some(shared) => shared as usize,
                    None => {
                        let made = self.nodes.push(key);
                        self.nodes.get(made).set_halfmoves(halfmoves);
                        if shard.insert(entry, made as u32).is_none() {
                            graph.entries.fetch_add(1, Ordering::Relaxed);
                        }
                        made
                    }
                }
            }
        };
        jitter();
        let parent = self.nodes.get(parent);
        let place = edge - parent.edges().start;
        //the node's reference becomes the move's if the move is kept
        let kept = match parent.children() {
            Children::Solid(first) => {
                let link = first + place as u32;
                self.links.slot(link).claim(made as u32, repeats).map(|_| (made, link)).map_err(|other| (other as usize, link))
            }
            Children::List(_) => self.insert_link(parent, place, MoveRecord::new(made as u32, repeats)),
        };
        kept.unwrap_or_else(|other| {
            self.counts.orphans.fetch_add(1, Ordering::Relaxed);
            self.drop_reference(made);
            other
        })
    }

    //a link in the parent's list for its move at `place`, with what the move is to hold, its node first of all: Ok of the node
    //and the link if it is kept, else Err of the node and link the move has. The link is whole before the list has it
    fn insert_link(&self, parent: &Node, place: usize, record: MoveRecord) -> Result<(usize, u32), (usize, u32)> {
        let mut mine = NO_LINK;
        loop {
            match self.links.find(parent.list_head(), place) {
                Ok(found) => {
                    if mine != NO_LINK {
                        self.links.free_link(mine);
                    }
                    return Err((self.links.child(found) as usize, found));
                }
                Err((pointer, at)) => {
                    if mine == NO_LINK {
                        mine = self.links.push(record, place);
                    }
                    if self.links.insert(pointer, at, mine) {
                        return Ok((record.state.child() as usize, mine));
                    }
                    jitter();
                }
            }
        }
    }

    //the link a waiting leaf's path holds for a move of the node, in case the node's list was made solid since
    fn current_link(&self, id: usize, link: u32) -> u32 {
        if !listed(link) {
            return link;
        }
        match self.nodes.get(id).children() {
            Children::Solid(first) => first + self.links.get(link).place() as u32,
            Children::List(_) => link,
        }
    }

    //the node's list is queued to be made solid by the visit that takes it to the threshold from `visits_before`
    fn queue_solid(&self, id: usize, visits_before: u32) {
        if visits_before + 1 == self.solid_threshold && matches!(self.nodes.get(id).children(), Children::List(_)) {
            self.queue_to_solidify(id);
        }
    }

    fn queue_to_solidify(&self, id: usize) {
        let mut queue = self.to_solidify.lock().expect("the solid queue");
        queue.push(id as u32);
        self.queued.store(queue.len(), Ordering::Release);
    }

    //makes the queued lists solid once there is a batch of them, with the tree to itself; returns whether it did, the caller's
    //share of the gate given up meanwhile
    fn solidify_queued<'a>(&'a self, hold: &mut Hold<'a>) -> bool {
        if self.queued.load(Ordering::Acquire) < self.solid_batch {
            return false;
        }
        let queued = {
            let mut queue = self.to_solidify.lock().expect("the solid queue");
            if queue.len() < self.solid_batch {
                return false;
            }
            self.queued.store(0, Ordering::Release);
            std::mem::take(&mut *queue)
        };
        self.exclusive(hold, || {
            for id in queued {
                self.make_solid(id as usize);
            }
        });
        true
    }

    //the node's list copied into one slot a move, with all that each move holds (the descents in flight through it too); needs
    //the tree to itself. The list stays readable until the next move: a waiting leaf may name one of its links
    fn make_solid(&self, id: usize) {
        let node = self.nodes.get(id);
        let moves = node.edges().len();
        let Children::List(head) = node.children() else { return };
        if moves == 0 {
            return;
        }
        let children = self.links.linked(node).map(|(place, link)| (place, self.links.record(link)));
        set_links(&self.links, node, moves, true, children);
        if head != NO_LINK {
            self.retired.lock().expect("the retired lists").push(head);
        }
    }

    //what a descent finds at the node. Without transpositions it counts itself in flight there, whatever it finds: selection
    //reads the nodes' counts. With them the moves count the descents (walk_moves), and a node only those that may end at it: a
    //proven node and one with moves and visits are passed uncounted, so that a descent writes nothing to a node on its way
    //down. The count decides whose leaf a node without moves or visits is
    fn arrive(&self, id: usize) -> Arrival {
        let node = self.nodes.get(id);
        let passes = self.graph.is_some();
        if passes {
            if let Some(proof) = node.proof() {
                return Arrival::Proven(proof);
            }
            if node.has_edges() && node.visits() > 0 {
                return Arrival::Inner;
            }
        }
        let (visits, in_flight) = node.bump();
        jitter();
        let found = match node.proof() {
            Some(proof) => Arrival::Proven(proof),
            None if visits == 0 && in_flight == 0 => return Arrival::Leaf { owned: true },
            None if visits == 0 => return Arrival::Collision,
            None if node.has_edges() => Arrival::Inner,
            None => return Arrival::Leaf { owned: in_flight == 0 },
        };
        //proven or given its moves and visits since the look above: no node to end at
        if passes {
            node.take_off(1);
        }
        found
    }

    //the nodes of a descent's path that the descent counted itself in flight at (arrive): all without transpositions, with
    //them the last, if the descent ended at a leaf (`at_leaf`) and not at a proven node
    fn in_flight_nodes<'p>(&self, path: &'p [usize], at_leaf: bool) -> &'p [usize] {
        match (&self.graph, at_leaf) {
            (None, _) => path,
            (Some(_), true) => &path[path.len() - 1..],
            (Some(_), false) => &[],
        }
    }

    //with transpositions, for a descent that came to the expanded, unproven `child` by `link`, a move of the node `id`: Tree's
    //stops_at and take_mean as one step of the move. A move without visits has its first, at its node's mean, and the descent
    //stops (true): the node `id` has it in its sum from here, and the backup goes on from that node's visit. A move with
    //visits takes its node's mean, the node `id` what that changed (unless it holds the mean already), and the descent goes
    //on. A node whose evaluation is not backed up yet has no mean to take: the descent goes on to it, which is a collision
    fn meet(&self, id: usize, link: u32, child: &Node) -> bool {
        if child.visits() == 0 {
            return false;
        }
        let word = self.links.slot(link).held();
        let mut seen = word.load();
        loop {
            let (visits, old) = seen;
            let mean = held_mean(child, child.visits());
            if visits == 0 {
                if word.change(seen, (1, mean)).is_ok() {
                    self.add_move_visit(id, visits, old, mean);
                    return true;
                }
            } else if mean.to_bits() == old.to_bits() {
                return false;
            } else if word.change(seen, (visits, mean)).is_ok() {
                let parent = self.nodes.get(id);
                if parent.proof().is_none() {
                    parent.add_value(f64::from(visits) * (f64::from(old) - f64::from(mean)));
                }
                return false;
            }
            jitter();
            seen = word.load();
        }
    }

    //Tree::known_pass_due and the pass's beginning: whether the node, of `visits` visits, is due a pass of link_known_moves
    //that this thread is to make (Node::begin_known_pass)
    fn begins_known_pass(&self, node: &Node, visits: u32) -> bool {
        let passes = node.known_passes();
        passes < KNOWN_PASSES && u64::from(visits) >= known_pass_due_at(passes) && node.begin_known_pass(passes)
    }

    //Tree::link_known_moves beside other threads, for the thread that began the pass of the node `id`, of `visits` visits and
    //with its moves as `children`: each of its moves without a node takes the node its position has in the table, if that node
    //has a visit, and then holds that node's mean and whether it is a proven loss. In a list the move gets a link, written
    //whole before it is put in at its place (insert_link): a selecting thread sees the move without a node or as given. A
    //solid node's slot takes one node (Slot::claim), then its mean, which is the move's only while no descent has given it
    //one. Either way a descent that makes or finds the move's node meanwhile and this thread end with the same one, the
    //other's count given back. `board` is the node's position. The passes the node has had and how far a solid node's
    //selection looks for the moves given a node (Node::known_to) are published last: until then it looks as far as the pass
    //before left it
    fn link_known_moves(&self, id: usize, children: Children, visits: u32, board: &Chess) {
        let (node, graph) = (self.nodes.get(id), self.graph.as_ref().expect("a tree with transpositions"));
        let edges = node.edges();
        let mut with_node = [false; MOST_MOVES];
        for (place, _) in self.links.linked(node) {
            with_node[place] = true;
        }
        let mut known_to = node.known_to();
        for (place, edge) in edges.enumerate() {
            if with_node[place] {
                continue;
            }
            let mut after = board.clone();
            after.play_unchecked(self.edges.chess_move(edge).to_move(board));
            let (key, halfmoves) = (position_key(&after), halfmove_count(after.halfmoves()));
            let entry = graph_key(key, u32::from(halfmoves));
            //the move's count of the node is taken under the shard's lock, as child_node takes it: a dead node is nobody's
            let known = {
                let shard = graph.shard(entry).lock().expect("a table shard");
                let visited = |known: &u32| self.nodes.get(*known as usize).visits() > 0 && self.nodes.get(*known as usize).acquire();
                shard.get(&entry).copied().filter(visited)
            };
            if let Some(known) = known {
                let found = self.nodes.get(known as usize);
                //of a proven node its proven score, as learn_proofs gives a move
                let mean = found.proof().map_or_else(|| held_mean(found, found.visits()), |proof| proof.value as f32);
                let repeats = repeat_filter(key, halfmoves);
                let given = match children {
                    Children::List(_) => {
                        let state = MoveState::new(known, repeats, found.proves_a_loss());
                        self.insert_link(node, place, MoveRecord { visits: 0, mean, state }).is_ok()
                    }
                    Children::Solid(first) => {
                        let slot = self.links.slot(first + place as u32);
                        let claimed = slot.claim(known, repeats).is_ok();
                        if claimed {
                            let _ = slot.held().change((0, NO_MEAN), (0, mean));
                            if found.proves_a_loss() {
                                slot.learn_loss();
                            }
                        }
                        claimed
                    }
                };
                if !given {
                    self.drop_reference(known as usize);
                }
                known_to = known_to.max(place + 1);
            }
        }
        node.end_known_pass(known_passes_due(visits), known_to);
    }

    //with transpositions the path's moves (`links`) learn their nodes' proofs, from the path's last node up as far as the nodes
    //are proven (Tree::learn_proofs): a move holds its node's proven score from then on, and that its node is a proven loss
    //for the side that makes it
    fn learn_proofs(&self, path: &[usize], links: &[usize]) {
        for (depth, &link) in links.iter().enumerate().rev() {
            let Some(proof) = self.nodes.get(path[depth + 1]).proof() else { break };
            let slot = self.links.slot(self.current_link(path[depth], link as u32));
            self.hold_mean(path[depth], slot, proof.value as f32);
            if proof.outcome == Outcome::Loss && !slot.state().lost() {
                slot.learn_loss();
            }
        }
    }

    //Tree::hold_mean beside other threads: the move of the node `id` whose slot this is holds `mean` from now on, by a step
    //of the move as a descent's that passes it (meet): the node takes what the step changed, unless the move holds the mean
    //already
    fn hold_mean(&self, id: usize, slot: &Slot, mean: f32) {
        let word = slot.held();
        let mut seen = word.load();
        while seen.1.to_bits() != mean.to_bits() {
            let (visits, old) = seen;
            match word.change(seen, (visits, mean)) {
                Ok(()) => {
                    let parent = self.nodes.get(id);
                    if parent.proof().is_none() {
                        parent.add_value(f64::from(visits) * (f64::from(old) - f64::from(mean)));
                    }
                    return;
                }
                Err(now) => seen = now,
            }
            jitter();
        }
    }

    //the node's moves as edges, without nodes. The node is the caller's: nothing else reads or writes its edges until they
    //are published
    fn expand(&self, id: usize, moves: &MoveList, priors: &[f32]) {
        let first = self.edges.alloc(moves.len());
        self.edges.write(first, moves, priors);
        jitter();
        let node = self.nodes.get(id);
        node.publish_edges(first, moves.len());
        //with transpositions paths' repeats may have given the node a solid one's visits before it had moves: its list is made
        //solid from its start (Tree::make_solid_once_visited)
        if self.graph.is_some() && node.visits() >= self.solid_threshold {
            self.queue_to_solidify(id);
        }
    }

    //with transpositions the simulation ended at the path's last node and is a visit of every node and move (`links`, the
    //walk's `edges`) above it; `at_leaf`: the node is a leaf the descent counted itself in flight at, and not a proven node
    //it only met
    fn back_up(&self, path: &[usize], links: &[usize], value: f64, at_leaf: bool) {
        if self.graph.is_none() {
            return self.backup(path, value);
        }
        let (&leaf, above) = path.split_last().expect("the root at least");
        self.add_own_score(leaf, 1.0 - value);
        self.back_up_means(above, links, leaf, (u32::from(at_leaf), 1));
    }

    //Tree::count_own_visit's score: a simulation ended at the node, with its score for the side that moved into it. The
    //visit is back_up_means' to count
    fn add_own_score(&self, id: usize, score: f64) {
        let node = self.nodes.get(id);
        node.add_value(node.proof().map_or(score, |proof| proof.value));
    }

    //the node's sum after a step of one of its moves to one more visit (the protocol): the move had `visits` at the mean `old`
    //and has one more at `mean`. The node's visit for it is back_up_means' to count
    fn add_move_visit(&self, id: usize, visits: u32, old: f32, mean: f32) {
        let node = self.nodes.get(id);
        jitter();
        match node.proof() {
            //Tree's sum of a proven node is its visits at the proven score, written whole: so it is here unless other
            //threads' visits are half counted, and with those but for the rounding
            Some(proof) => {
                let whole = f64::from(node.visits() + 1) * proof.value;
                node.change_value(|sum| if (sum + proof.value - whole).abs() < proof.value / 2.0 { whole } else { sum + proof.value });
            }
            //a node's score is for the side that moved into it: one less its move's
            None => {
                let (after, before) = (f64::from(visits + 1) * (1.0 - f64::from(mean)), f64::from(visits) * (1.0 - f64::from(old)));
                node.change_value(|sum| sum + after - before);
            }
        }
        if visits == 0 {
            self.refresh_visited_policy(id);
        }
    }

    //Tree::back_up_means beside other threads' (see the protocol), up from the node `below`: its sum has the simulation's
    //score or its move's step already, its visit is not counted yet, and the descent counted itself in flight `in_flight.0`
    //times at it and `in_flight.1` times at the move to it. `path` are the nodes above it from the root and `links` their
    //moves on the way down, each move in flight once and no node. A move takes the mean its node has with that visit, and
    //the node counts the visit after; a step is tried again with what the move then holds, if another thread's came first
    fn back_up_means(&self, path: &[usize], links: &[usize], below: usize, in_flight: (u32, u32)) {
        debug_assert_eq!(path.len(), links.len(), "a move for every node above");
        let (mut below, (mut in_flight, mut pending)) = (below, in_flight);
        for (&id, &link) in path.iter().zip(links).rev() {
            let link = self.current_link(id, link as u32);
            debug_assert_eq!(self.links.child(link) as usize, below, "the path's move to the node below");
            let (slot, child) = (self.links.slot(link), self.nodes.get(below));
            let word = slot.held();
            let mut seen = word.load();
            let (visits, old, mean) = loop {
                let ((visits, old), mean) = (seen, held_mean(child, child.visits() + 1));
                match word.change(seen, (visits + 1, mean)) {
                    Ok(()) => break (visits, old, mean),
                    Err(now) => seen = now,
                }
                jitter();
            };
            jitter();
            if pending > 0 {
                slot.take_pending(pending);
            }
            self.queue_solid(below, child.visit(in_flight));
            self.add_move_visit(id, visits, old, mean);
            (below, in_flight, pending) = (id, 0, 1);
        }
        jitter();
        self.queue_solid(below, self.nodes.get(below).visit(in_flight));
    }

    //back_up without transpositions: a node's visit is counted by the node, after its value
    fn backup(&self, path: &[usize], value: f64) {
        let mut score = 1.0 - value;
        for (depth, &id) in path.iter().enumerate().rev() {
            let node = self.nodes.get(id);
            //from a node proven while the simulation was in flight, the proven score
            if let Some(proof) = node.proof() {
                score = proof.value;
            }
            node.add_value(score);
            jitter();
            let visits_before = node.visit(1);
            self.queue_solid(id, visits_before);
            if visits_before == 0 && depth > 0 {
                self.refresh_visited_policy(path[depth - 1]);
            }
            score = 1.0 - score;
        }
    }

    //the node's visited policy again, in edge order and python's summation; refreshers repeat until none began meanwhile
    fn refresh_visited_policy(&self, id: usize) {
        let node = self.nodes.get(id);
        let mut seen = node.begin_refresh();
        loop {
            let priors = self.edges.priors(node.edges());
            let visited = self.links.linked(node).filter(|&(_, link)| self.link_visits(link) > 0);
            let sum = python_sum(visited.map(|(place, _)| priors.get(place)));
            jitter();
            node.set_visited_policy_sqrt(sum.sqrt());
            let now = node.refreshes();
            if now == seen {
                break;
            }
            seen = now;
        }
    }

    //proves the path's last node: its visits so far count the proven score instead of theirs, in its ancestors too (up to a
    //proven one). A node already proven keeps its proof. With transpositions only the node's own. Needs the tree to itself
    fn prove(&self, path: &[usize], proof: Proof) {
        let (&id, ancestors) = path.split_last().expect("a node to prove");
        let node = self.nodes.get(id);
        if node.proof().is_some() {
            return;
        }
        let change = f64::from(node.visits()) * proof.value - node.value_sum();
        jitter();
        node.add_value(change);
        node.set_proof(proof);
        if self.graph.is_some() {
            return;
        }
        for (distance, &ancestor) in ancestors.iter().rev().enumerate() {
            let ancestor = self.nodes.get(ancestor);
            //a simulation that descended before an ancestor was proven: its visits all count the proven score, and
            //what is above it does not see this proof
            if ancestor.proof().is_some() {
                break;
            }
            ancestor.add_value(if distance % 2 == 0 { -change } else { change });
        }
    }

    //after the path's last node was proven, its parent may be too, and so on up. Needs the tree to itself
    fn propagate_proofs(&self, path: &[usize]) {
        for depth in (1..path.len().saturating_sub(1)).rev() {
            match self.proof_from_moves(path[depth]) {
                Some(proof) => self.prove(&path[..=depth], proof),
                None => break,
            }
        }
    }

    fn proves_parent(&self, path: &[usize]) -> bool {
        path.len() >= 3 && self.proof_from_moves(path[path.len() - 2]).is_some()
    }

    //room for one more descent: the root's own visits and the leaves waiting are within the simulations, the leaves read
    //first (a finished leaf adds its visit before it stops waiting, so it is never missed). A search by time (UNLIMITED
    //simulations) counts neither
    fn has_room(&self, visit_limit: u32) -> bool {
        let within_visits = visit_limit == u32::MAX || self.pending() + (self.root().visits() as usize) < visit_limit as usize;
        within_visits && !self.closed.load(Ordering::Acquire)
    }

    //no call gathers any more and the sweepers leave; the caller has the tree to itself (quiet)
    pub fn close(&self) {
        self.closed.store(true, Ordering::Release);
    }

    pub fn reopen(&mut self) {
        *self.closed.get_mut() = false;
    }

    fn hold_back_freed_room(&self, memory_limit: usize) {
        let held = holds_back_freed_room(memory_limit, self.footprint_bytes(), self.nodes.len());
        if held != self.nodes.holds_back() {
            self.nodes.hold_back(held);
            self.links.hold_back(held);
            self.edges.hold_back(held);
        }
    }

    //the descents a call may make without `reserve`: its share of a quarter of the room left (nodes, and bytes under a memory
    //limit), on the thought that at most MOST_CALLS calls are descending. Far from the limit that is more than a call descends
    pub fn free_descents(&self, memory_limit: usize) -> usize {
        //a descent takes a node and a link at most: a graph may have more links than nodes
        let nodes = Nodes::CAPACITY.saturating_sub(self.nodes.len()).min(self.links.room());
        let bytes = match memory_limit {
            usize::MAX => usize::MAX,
            _ => memory_limit.saturating_sub(self.allocated_bytes() + self.reserved_bytes()),
        };
        nodes.min(bytes / DESCENT_BYTES) / (4 * MOST_CALLS)
    }

    //room for a descent's node and its leaf's moves, under the memory limit; true if it is kept for the descent (`release`
    //gives it back). Short of room, the thread frees garbage until it has some or there is none and none being freed. Without
    //`limited` the descent is let through whatever the room. A leaf that waits has its moves' room counted by `waiting`
    pub fn reserve(&self, memory_limit: usize, limited: bool) -> bool {
        if limited && (self.nodes.len() + 64 >= Nodes::CAPACITY || self.links.room() <= 64) {
            return false;
        }
        if memory_limit == usize::MAX {
            return true;
        }
        self.reserved.fetch_add(DESCENT_BYTES, Ordering::AcqRel);
        while limited && self.allocated_bytes() + self.reserved_bytes() > memory_limit {
            if self.reclaim(ASSIST_BATCH) > 0 {
                self.counts.assists.fetch_add(1, Ordering::Relaxed);
            } else if self.room_still_to_free() {
                std::thread::yield_now();
            } else if self.allocated_bytes() + self.reserved_bytes() > memory_limit {
                //nothing is left to free, but the last batch's thread may have freed it since the loop's look: the room is looked at again
                self.reserved.fetch_sub(DESCENT_BYTES, Ordering::AcqRel);
                return false;
            }
        }
        true
    }

    //the room of a descent back, if it reserved some: its node is allocated and its leaf's moves are counted as waiting or allocated
    fn release(&self, memory_limit: usize, reserved: bool) {
        if reserved && memory_limit != usize::MAX {
            self.reserved.fetch_sub(DESCENT_BYTES, Ordering::AcqRel);
        }
    }

    //the network leaves of the round, in the order they were gathered in, for finish_leaf
    pub fn queue(&self, round: u64, leaves: Vec<PendingLeaf>) {
        if leaves.is_empty() {
            return;
        }
        let mut pending = self.pending.lock().expect("the pending leaves");
        let queued = pending.entry(round).or_default();
        //a round queued once is the caller's vector, not a copy of it
        if queued.is_empty() {
            *queued = leaves;
        } else {
            queued.extend(leaves);
        }
    }

    pub fn take_round(&self, round: u64) -> Vec<PendingLeaf> {
        self.pending.lock().expect("the pending leaves").remove(&round).unwrap_or_default()
    }

    //Tree::gather for one thread of several: descents until the round has its leaves or its simulations, the leaves' boards
    //added to boards, a leaf that needs no network finished at once. Returns the leaves, which the caller queues (`queue`)
    //under the round. A collision keeps its simulation in flight until the call ends. Several threads may overrun the root's
    //simulations by one each: a check and a leaf are not one step
    pub fn gather(
        &self,
        round: &Round,
        settings: &SearchSettings,
        tables: Option<&Tables>,
        cache: Option<&SharedCache>,
        boards: &mut Vec<u8>,
    ) -> Vec<PendingLeaf> {
        self.gather_together(round, &Together::default(), None, settings, tables, cache, boards)
    }

    //gather as one of the calls that gather a round together, each into boards of its own; the caller queues all their leaves under
    //the round, in the order of their boards. help: for the call that began the round, (simulations, call): once it has made that
    //many itself with the round still short of leaves it calls for the others, and it ends the round (Together::end) when it returns
    #[allow(clippy::too_many_arguments)]
    pub fn gather_together(
        &self,
        round: &Round,
        together: &Together,
        help: Option<(u32, &dyn Fn())>,
        settings: &SearchSettings,
        tables: Option<&Tables>,
        cache: Option<&SharedCache>,
        boards: &mut Vec<u8>,
    ) -> Vec<PendingLeaf> {
        let mut hold = self.hold();
        let leaves = Vec::with_capacity(LEAVES_AT_ONCE.min(round.leaves as usize));
        let mut call = Call::new(leaves, help);
        self.gather_in(&mut hold, round, together, settings, tables, cache, boards, &mut call);
        self.end_call(call)
    }

    //a call's collisions are in flight no more, and its leaves are the search's evaluations
    fn end_call(&self, call: Call) -> Vec<PendingLeaf> {
        let (mut nodes, mut moves) = (0, 0);
        for &(nodes_end, moves_end, descents) in &call.collided_runs {
            for &node in &call.collided[nodes..nodes_end] {
                self.nodes.get(node).take_off(descents);
            }
            for &(node, link) in &call.collided_moves[moves..moves_end] {
                self.links.slot(self.current_link(node, link)).take_pending(descents);
            }
            (nodes, moves) = (nodes_end, moves_end);
        }
        self.evaluations.fetch_add(call.leaves.len() as u64, Ordering::Relaxed);
        call.leaves
    }

    //Tree::gather for one thread of several, in Tree's way: the call's descents are those of one descent from the root after
    //another (the tests' gather_by_descents), but a node that several of them pass in a row is passed once (walk_moves).
    //What the threads share is at the head of this file ("descents in runs")
    #[allow(clippy::too_many_arguments)]
    fn gather_in<'a>(
        &'a self,
        hold: &mut Hold<'a>,
        round: &Round,
        together: &Together,
        settings: &SearchSettings,
        tables: Option<&Tables>,
        cache: Option<&SharedCache>,
        boards: &mut Vec<u8>,
        call: &mut Call,
    ) {
        let mut keys = self.keys.clone();
        let root_keys = keys.len();
        let free_descents = self.free_descents(round.memory_limit);
        self.hold_back_freed_room(round.memory_limit);
        let mut walker = Walker {
            hold,
            round: *round,
            together,
            settings: *settings,
            tables,
            cache,
            boards,
            call,
            root_covered: tables.is_some_and(|tables| tables.covers(&self.board)),
            root_keys,
            path: Vec::new(),
            edges: Vec::new(),
            line: Vec::new(),
            board: self.board.clone(),
            played: 0,
            no_moves: Vec::new(),
            priors: Vec::new(),
            free_descents,
            started: false,
            reserved: false,
            writer_let_in: false,
        };
        //the most descents a walk from the root is given (Tree::gather's): twice those that waited in the last walk, and one.
        //Without transpositions every descent goes down alone: selection reads the nodes' counts there, and a descent counts
        //itself at every node it passes
        let mut most = if self.graph.is_some() { u32::MAX } else { 1 };
        //a descent that a walk sent back to the root (a writer was let in) is started already
        while walker.started || self.may_start(&mut walker) {
            walker.writer_let_in = false;
            //the descents that may follow one another with no result between them: those the round and the search still
            //have room for. The round's collisions are not among them, so that a run is never longer than the leaves the
            //round is short of now: the other threads read a run as descents on their way
            let leaves = round.leaves.saturating_sub(together.leaves.load(Ordering::Acquire));
            let simulations = round.simulations.saturating_sub(together.simulations.load(Ordering::Acquire));
            let room = match round.visit_limit {
                u32::MAX => u32::MAX,
                limit => (limit as usize).saturating_sub(self.pending() + self.root().visits() as usize).min(u32::MAX as usize) as u32,
            };
            let descents = leaves.min(simulations).min(room).min(most).max(1);
            let waited_before = walker.call.leaves.len() + walker.call.collisions;
            keys.truncate(root_keys);
            walker.path.clear();
            walker.path.push(self.root);
            walker.edges.clear();
            walker.forget_line();
            if matches!(self.arrive_run(&mut walker, &mut keys, descents).0, Walk::Ended) {
                break;
            }
            if self.graph.is_some() {
                most = ((walker.call.leaves.len() + walker.call.collisions - waited_before) as u32).saturating_mul(2).saturating_add(1);
            }
        }
    }

    //whether the call may start another descent, as gather_by_descents asks before each one: it lets a waiting writer in, the
    //round is short of leaves and simulations, the search of visits and still open, and there is room for the descent
    fn may_start<'a>(&'a self, walker: &mut Walker<'a, '_, '_>) -> bool {
        //a writer let in here may have ended the search and closed the tree (end's quiet), so the room is looked at
        //after: a leaf taken in a closed tree belongs to a search that is gone
        let waited_for = walker.hold.give_way();
        walker.writer_let_in |= waited_for | self.solidify_queued(walker.hold);
        #[cfg(test)]
        if let Some(writer) = WRITER_AT_START.get() {
            walker.writer_let_in |= writer(self);
        }
        let (round, together) = (&walker.round, walker.together);
        let short = together.leaves.load(Ordering::Acquire) < round.leaves
            && together.simulations.load(Ordering::Acquire) < round.simulations
            && !together.ended.load(Ordering::Acquire);
        if !(short && self.has_room(round.visit_limit)) {
            return false;
        }
        if let Some((alone, call_for_help)) = walker.call.help
            && walker.call.simulations >= alone
        {
            walker.call.help = None;
            call_for_help();
        }
        let reserved = walker.free_descents == 0;
        if reserved {
            //the root's own evaluation always fits, so that a full search still has moves to report
            if !self.reserve(round.memory_limit, self.root().has_edges()) {
                self.full.store(true, Ordering::Relaxed);
                return false;
            }
        } else {
            walker.free_descents -= 1;
        }
        (walker.started, walker.reserved) = (true, reserved);
        true
    }

    //for a walk below the root that has no descent on its way: whether the next one may start where the walk is. Else the
    //walk goes back up as `Err` says: the call has what it may gather (Ended), or a writer has had the tree, and the descent,
    //which is started, starts at the root (Interrupted)
    fn starts_here<'a>(&'a self, walker: &mut Walker<'a, '_, '_>) -> Result<(), Walk> {
        if walker.started {
            return Ok(());
        }
        if !self.may_start(walker) {
            return Err(Walk::Ended);
        }
        if walker.writer_let_in { Err(Walk::Interrupted) } else { Ok(()) }
    }

    //the descent on its way has ended: its room is given back, and the next one is asked for anew
    fn descent_over(&self, walker: &mut Walker) {
        self.release(walker.round.memory_limit, walker.reserved);
        (walker.started, walker.reserved) = (false, false);
    }

    //the descent on its way is one of the round's simulations
    fn count_simulation(&self, walker: &mut Walker) {
        walker.call.simulations += 1;
        walker.together.simulations.fetch_add(1, Ordering::AcqRel);
    }

    //the descent on its way has ended without waiting for the network
    fn finished_at_once(&self, walker: &mut Walker) {
        self.descent_over(walker);
        self.count_simulation(walker);
    }

    //`descents` descents in a row have come to the path's last node, each of which has counted itself in flight at every move
    //above (walk_moves): what gather_by_descents does with each once it is at the node. Returns how the walk left them, and how
    //many of them wait or were finished; the others' counts at the moves above are the callers' to take back
    fn arrive_run<'a>(&'a self, walker: &mut Walker<'a, '_, '_>, keys: &mut Vec<u64>, descents: u32) -> (Walk, u32) {
        let id = *walker.path.last().expect("the root at least");
        let (mut over, mut collided_here) = (0, None);
        while over < descents {
            if let Err(walk) = self.starts_here(walker) {
                return (walk, over);
            }
            let owned = match self.arrive(id) {
                Arrival::Inner => {
                    let (walk, walked) = self.walk_moves(walker, keys, descents - over);
                    over += walked;
                    match walk {
                        Walk::Quiet => continue,
                        _ => return (walk, over),
                    }
                }
                Arrival::Proven(proof) => {
                    self.finished_at_once(walker);
                    self.back_up(&walker.path, &walker.edges, 1.0 - proof.value, false);
                    //with transpositions this parent may not have met the proof, made on another path
                    if self.graph.is_some() && walker.settings.certainty && self.proves_parent(&walker.path) {
                        self.exclusive(walker.hold, || self.propagate_proofs(&walker.path));
                    }
                    self.learn_proofs(&walker.path, &walker.edges);
                    return (Walk::Interrupted, over + 1);
                }
                Arrival::Repeat | Arrival::FirstVisit => unreachable!("a descent stops above a node in walk_moves"),
                Arrival::Collision => None,
                Arrival::Leaf { owned } => Some(owned),
            };
            let waits = match owned {
                None => false,
                Some(owned) => match self.evaluate_leaf(walker, keys, owned) {
                    Evaluated::Waits => true,
                    Evaluated::AnothersLeaf => false,
                    Evaluated::AtOnce => return (Walk::Interrupted, over + 1),
                    Evaluated::NotTaken => return (Walk::Ended, over),
                },
            };
            //the descent ended on a node another one owns: it stays in flight until the call ends, at the nodes it counted
            //itself at and at its path's moves, if the round has a collision left; else it leaves no trace and ends the round
            if !waits {
                self.descent_over(walker);
                let in_flight = self.in_flight_nodes(&walker.path, true);
                if walker.call.collisions == walker.round.collisions as usize {
                    for &node in in_flight {
                        self.nodes.get(node).take_off(1);
                    }
                    return (Walk::Ended, over);
                }
                walker.call.collide(&walker.path, &walker.edges, in_flight, &mut collided_here);
            }
            over += 1;
        }
        (Walk::Quiet, over)
    }

    //`descents` descents in a row have come to the path's last node, which has moves: Tree::walk_moves beside other threads.
    //Each look is one descent's selection at the node, by what the moves hold at that moment, and the step into the move's
    //node. The descents that select_edge would send down the same move after it, each finding one more in flight there
    //(more_in_a_row), go down with it: the move counts them all in flight in one step, before the walk goes on, and what the
    //walk below did not use of them is taken off when it comes back. A descent finished at once sends the walk back to the root
    //(Interrupted); a move that took a new mean in passing has its descent go down alone and the node above look again
    //(Stirred). Returns how the walk left the descents and how many wait or were finished
    fn walk_moves<'a>(&'a self, walker: &mut Walker<'a, '_, '_>, keys: &mut Vec<u64>, descents: u32) -> (Walk, u32) {
        let id = *walker.path.last().expect("the root at least");
        let node = self.nodes.get(id);
        let mut over = 0;
        loop {
            if let Err(walk) = self.starts_here(walker) {
                return (walk, over);
            }
            //a descent that comes to a node due a pass gives its moves the nodes their positions have; a list's moves, which
            //take links for them, only if the descent has room to give: near the memory limit, where every descent reserves
            //its room, a pass's links are in no reservation, and only while half the links are left (free_descents)
            if self.graph.is_some() {
                let (children, visits) = (node.children(), node.visits());
                let has_room = matches!(children, Children::Solid(_)) || !walker.reserved && self.links.room() > Links::CAPACITY / 2;
                if has_room && self.begins_known_pass(node, visits) {
                    walker.play_line(self);
                    self.link_known_moves(id, children, visits, &walker.board);
                }
            }
            let left = descents - over;
            let repeats = self.graph.as_ref().map(|_| Repeats::new(keys, walker.root_keys - 1, u32::from(node.halfmoves())));
            //one descent is select_edge's alone; several want to know how long its move stays the choice
            let (edge, selected, looked) = if left == 1 {
                let (edge, selected) = self.select_edge(id, &walker.settings, repeats.as_ref());
                (edge, selected, None)
            } else {
                let (edge, selected, scan, rivals) = self.select_with_rivals(id, &walker.settings, repeats.as_ref());
                (edge, selected, Some((scan, rivals)))
            };
            walker.line.push(edge);
            //a move without a node gets one, which takes the position
            let had_node = selected != NO_LINK && self.links.child(selected) != NO_NODE;
            if !had_node {
                walker.play_line(self);
            }
            let (child_id, link) = self.child_node(id, edge, selected, &mut walker.board, &mut walker.no_moves);
            walker.path.push(child_id);
            let child = self.nodes.get(child_id);
            let (mut stop, mut new_mean) = (None, false);
            if let Some(repeats) = &repeats {
                walker.edges.push(link as usize);
                //a repeat of this path's position, whatever the node is to other paths, or a node that knows more than its move
                if repeats.repeats(child.key(), child.halfmoves()) {
                    stop = Some(Arrival::Repeat);
                } else if child.has_edges() && child.proof().is_none() {
                    let held = self.links.slot(link).held().load().1;
                    stop = self.meet(id, link, child).then_some(Arrival::FirstVisit);
                    new_mean = stop.is_none() && self.links.slot(link).held().load().1.to_bits() != held.to_bits();
                }
            }
            if let Some(stop) = stop {
                self.finished_at_once(walker);
                let shared = walker.path.pop().expect("the shared node");
                match stop {
                    //the move's first visit is in its node's sum since the descent stopped (meet)
                    Arrival::FirstVisit => {
                        let above = &walker.path[..walker.path.len() - 1];
                        self.back_up_means(above, &walker.edges[..above.len()], id, (0, 1));
                    }
                    //a repeat is a visit of the node too, at a draw's score for the side that moved into it (Tree::gather's)
                    _ => {
                        self.add_own_score(shared, 1.0 - walker.settings.draw);
                        self.back_up_means(&walker.path, &walker.edges, shared, (0, 0));
                    }
                }
                walker.edges.pop();
                walker.drop_move();
                return (Walk::Interrupted, over + 1);
            }
            //the descents that take this move in a row. After a new mean the next descent's step changes this node again: one
            let run = match &looked {
                Some((scan, rivals)) if !new_mean => {
                    let place = edge - node.edges().start;
                    //the look made before the step holds unless the step gave the move its node: the next move without one
                    //is weighed from then on
                    let rivals = if had_node { Some(*rivals) } else { self.rivals(id, scan, repeats.as_ref(), place) };
                    1 + rivals.map_or(0, |rivals| self.more_in_a_row(id, scan, repeats.as_ref(), (place, link), rivals, left - 1))
                }
                _ => 1,
            };
            keys.push(child.key());
            if self.graph.is_some() {
                self.links.slot(link).add_pending(run);
            }
            let (walk, walked) = self.arrive_run(walker, keys, run);
            keys.pop();
            walker.path.pop();
            walker.drop_move();
            if self.graph.is_some() {
                walker.edges.pop();
                //the descents of the run that the walk did not get to are in flight at the move no more. The node's list may
                //have been made solid since
                if walked < run {
                    self.links.slot(self.current_link(id, link)).take_pending(run - walked);
                }
            }
            over += walked;
            match walk {
                Walk::Quiet | Walk::Stirred => {}
                Walk::Interrupted | Walk::Ended => return (walk, over),
            }
            if new_mean {
                return (Walk::Stirred, over);
            }
            if over == descents {
                return (Walk::Quiet, over);
            }
        }
    }

    //gather_by_descents' end of a descent at the path's last node, which has no moves and at which the descent has counted
    //itself in flight (arrive); `owned`: no other descent is at it
    fn evaluate_leaf<'a>(&'a self, walker: &mut Walker<'a, '_, '_>, keys: &[u64], owned: bool) -> Evaluated {
        let node = *walker.path.last().expect("the root at least");
        let settings = walker.settings;
        walker.play_line(self);
        let mut moves = walker.board.legal_moves();
        let node_tables = walker.tables.filter(|_| node != self.root && (!walker.root_covered || walker.board.halfmoves() == 0));
        //the root is searched whatever the rules say of it: Searches.start refuses only a mate or stalemate
        let game_over = if node == self.root { None } else { game_result(&walker.board, keys, &moves, settings.draw, node_tables) };
        //a repeat inside the search is a draw, but not a proven one: the game need not repeat
        let (&leaf_key, path_keys) = keys[walker.root_keys - 1..].split_last().expect("the root's key");
        //with transpositions a leaf is no repeat inside the search (Tree::gather's)
        let value = game_over.or_else(|| path_keys.contains(&leaf_key).then_some(settings.draw));
        //no game's end, and the node is another descent's to evaluate: Tree::gather's collision
        if value.is_none() && !owned {
            return Evaluated::AnothersLeaf;
        }
        self.count_simulation(walker);
        if let Some(value) = value {
            self.descent_over(walker);
            let (path, edges) = (&walker.path, &walker.edges);
            match game_over {
                //the result is for the side to move, the proof for the side that moved here. A win or loss with moves
                //left is the tables', after any mate
                Some(result) if settings.certainty && node != self.root => {
                    let (outcome, plies) = if result == 0.0 {
                        (Outcome::Win, if moves.is_empty() { 0 } else { TABLEBASE_PLIES })
                    } else if result == 1.0 {
                        (Outcome::Loss, TABLEBASE_PLIES)
                    } else {
                        (Outcome::Draw, 0)
                    };
                    let proof = Proof { outcome, value: 1.0 - result, plies };
                    let leaf = self.nodes.get(node);
                    //with transpositions a path's repeat may visit the leaf beside this, at a draw's score: that is a drawn
                    //leaf's proven score, and the position right after a capture or pawn move repeats none
                    let scored_alike = self.graph.is_none() || outcome == Outcome::Draw || leaf.halfmoves() == 0;
                    if leaf.visits() == 0 && scored_alike {
                        //a leaf nobody has visited, and nobody can before its proof is seen: it rewrites no sum
                        leaf.set_proof(proof);
                        self.back_up(path, edges, value, true);
                        if self.proves_parent(path) {
                            self.exclusive(walker.hold, || self.propagate_proofs(path));
                        }
                    } else {
                        self.exclusive(walker.hold, || {
                            self.prove(path, proof);
                            self.back_up(path, edges, value, true);
                            self.propagate_proofs(path);
                        });
                    }
                }
                _ => {
                    self.back_up(path, edges, value, true);
                    if settings.certainty && self.proves_parent(path) {
                        self.exclusive(walker.hold, || self.propagate_proofs(path));
                    }
                }
            }
            if settings.certainty {
                self.learn_proofs(path, edges);
            }
            return Evaluated::AtOnce;
        }
        sort_moves_in_python_order(&walker.board, &mut moves);
        //a hash collision would give another number of moves
        let cached = walker.cache.and_then(|cache| cache.get(leaf_key, &mut walker.priors)).filter(|_| walker.priors.len() == moves.len());
        match cached {
            Some(value) => {
                self.descent_over(walker);
                self.expand(node, &moves, &walker.priors);
                self.back_up(&walker.path, &walker.edges, f64::from(value), true);
                Evaluated::AtOnce
            }
            None => {
                //the round's last leaf may have gone to another of its calls meanwhile: this one is left for a later round,
                //and the descent in flight no more at the node (at the moves above the walk takes it off)
                if walker.together.leaves.fetch_add(1, Ordering::AcqRel) >= walker.round.leaves {
                    walker.together.leaves.fetch_sub(1, Ordering::AcqRel);
                    for &node in self.in_flight_nodes(&walker.path, true) {
                        self.nodes.get(node).take_off(1);
                    }
                    self.descent_over(walker);
                    return Evaluated::NotTaken;
                }
                encode_board(&walker.board, walker.boards);
                self.waiting.fetch_add(1, Ordering::AcqRel);
                let (path, edges) = (walker.path.clone(), walker.edges.clone());
                walker.call.leaves.push(PendingLeaf { path, edges, key: leaf_key, moves, turn: walker.board.turn() });
                self.descent_over(walker);
                Evaluated::Waits
            }
        }
    }

    //`answer` gives the move logits, then the expected score; the cache keeps it sharpened. The caller holds a share of the
    //gate (`hold`) for the leaves of its round
    #[allow(clippy::too_many_arguments)]
    pub fn finish_leaf(
        &self,
        leaf: PendingLeaf,
        answer: impl Fn(usize) -> f32,
        answer_width: usize,
        move_ids: &MoveIds,
        cache: Option<&SharedCache>,
        settings: &SearchSettings,
        logits: &mut Vec<f32>,
    ) {
        logits.clear();
        logits.extend(leaf.moves.iter().map(|&chess_move| answer(move_ids.id(chess_move, leaf.turn))));
        self.finish_leaf_in_order(leaf, logits, answer(answer_width - 1), cache, settings);
    }

    //finish_leaf for the leaf's move logits and value, the logits in the order of its moves as a lean answer has them; they
    //become the priors
    pub fn finish_leaf_in_order(
        &self,
        leaf: PendingLeaf,
        logits: &mut [f32],
        value: f32,
        cache: Option<&SharedCache>,
        settings: &SearchSettings,
    ) {
        assert_eq!(logits.len(), leaf.moves.len(), "a logit a move");
        let (value, kept) = leaf_answer(logits, value, settings);
        let node = *leaf.path.last().expect("the root at least");
        self.expand(node, &leaf.moves, logits);
        self.back_up(&leaf.path, &leaf.edges, f64::from(value), true);
        if let Some(cache) = cache.filter(|_| kept) {
            cache.insert(leaf.key, logits, value);
        }
        self.waiting.fetch_sub(1, Ordering::AcqRel);
    }
}

//a node's mean over `visits` visits, for a move to hold. Read while other threads back up through the node it may be off by
//their values: it is kept a score
fn held_mean(node: &Node, visits: u32) -> f32 {
    node.mean_score(visits).clamp(0.0, 1.0) as f32
}

//makes a node's links before it is published, from its moves with a node as (place, what the move holds) in order
fn set_links(links: &Links, node: &Node, moves: usize, solid: bool, children: impl Iterator<Item = (usize, MoveRecord)>) {
    if solid {
        let first = links.solid(moves);
        for (place, record) in children {
            links.set_solid(first + place as u32, record);
        }
        node.make_solid(first);
        return;
    }
    let mut last = NO_LINK;
    for (place, record) in children {
        let link = links.push(record, place);
        if last == NO_LINK {
            node.list_head().store(link, Ordering::Relaxed);
        } else {
            links.set_next(last, link);
        }
        last = link;
    }
}

//what a Tree's move holds, as a SharedTree's
#[cfg(test)]
fn record_of(slot: &super::Slot) -> MoveRecord {
    let state = MoveState::new(slot.child, slot.repeats, slot.lost).with_pending(u32::from(slot.pending));
    MoveRecord { visits: slot.visits, mean: slot.mean, state }
}

//what a SharedTree's move holds, as a Tree's
#[cfg(test)]
fn slot_of(record: MoveRecord) -> super::Slot {
    let state = record.state;
    super::Slot {
        child: state.child(),
        visits: record.visits,
        mean: record.mean,
        pending: state.pending() as u16,
        lost: state.lost(),
        repeats: state.repeats(),
    }
}

#[cfg(test)]
impl SharedTree {
    pub fn queued_rounds(&self) -> usize {
        self.pending.lock().expect("the pending leaves").len()
    }

    pub fn from_tree(tree: &Tree) -> SharedTree {
        assert_eq!(tree.pending(), 0, "a tree to copy has nothing waiting");
        let (nodes, edges, links) = (Nodes::new(), Edges::new(), Links::new());
        for (id, node) in tree.nodes.iter().enumerate() {
            assert_eq!(nodes.push(tree.node_keys[id]), id);
            let copy = nodes.get(id);
            copy.end_known_pass(u32::from(node.known_passes), usize::from(node.known_to));
            copy.set_counts(node.visit_count, node.in_flight, node.value_sum);
            copy.set_visited_policy_sqrt(node.visited_policy_sqrt);
            if let Some(graph) = &tree.graph {
                copy.set_halfmoves(graph.halfmoves[id]);
            }
            if let Some(proof) = node.proof {
                copy.set_proof(proof);
            }
            if node.has_edges() {
                let (first, count) = (edges.alloc(node.edges().len()), node.edges().len());
                for (rank, edge) in node.edges().enumerate() {
                    edges.set(first, count, rank, tree.moves[edge], tree.priors[edge]);
                }
                let children = tree.linked(id).map(|(place, link)| (place, record_of(tree.slot(link))));
                set_links(&links, copy, node.edges().len(), node.solid_in_a_copy(), children);
                copy.publish_edges(first, node.edges().len());
            }
        }
        let graph = tree.graph.as_ref().map(|graph| {
            let copy = SharedGraph::new();
            for (&entry, &id) in &graph.table {
                copy.shard(entry).lock().expect("a table shard").insert(entry, id);
            }
            copy.entries.store(graph.table.len(), Ordering::Relaxed);
            copy
        });
        let shared = SharedTree {
            root: tree.root_index(),
            nodes,
            edges,
            links,
            full: AtomicBool::new(tree.full),
            graph,
            ..SharedTree::new(tree.board.clone(), tree.keys.clone())
        };
        shared.count_references();
        shared
    }

    pub fn node(&self, id: usize) -> &Node {
        self.nodes.get(id)
    }

    //whether the arenas hold much more than the memory limit allows in use, which freed room that nothing fits does
    pub fn sprawls(&self, memory_limit: usize) -> bool {
        memory_limit != usize::MAX && self.footprint_bytes() > memory_limit + memory_limit / 4
    }

    //what a compaction would leave: the part the root reaches copied in breadth first order
    pub fn compacted(&self) -> SharedTree {
        SharedTree::from_tree(&self.to_reachable_tree())
    }

    //select_edge's pick for a node nothing is working in, and the pick of a look at every move: the first of the best scores
    pub fn picked_and_best(&self, id: usize, settings: &SearchSettings) -> (usize, usize) {
        let parent = self.nodes.get(id);
        let scan = self.scan(parent, settings);
        let mut best = (parent.edges().start, f64::NEG_INFINITY);
        for edge in parent.edges() {
            let slot = match self.link_of(parent, edge) {
                NO_LINK => self.links.no_slot(),
                link => self.links.slot(link),
            };
            let prior = f64::from(self.edges.prior(parent.edges(), edge));
            if let Some(u) = self.move_value(&scan, None, prior, slot.state(), slot)
                && u > best.1
            {
                best = (edge, u);
            }
        }
        (self.select_edge(id, settings, None).0, best.0)
    }

    pub fn with_solid_links(mut self, threshold: u32, batch: usize) -> SharedTree {
        (self.solid_threshold, self.solid_batch) = (threshold, batch);
        self
    }

    pub fn lists_and_solids(&self) -> (usize, usize) {
        let linked = (0..self.nodes.len()).map(|id| self.nodes.get(id)).filter(|node| self.links.linked(node).next().is_some());
        linked.fold((0, 0), |(lists, solids), node| match node.children() {
            Children::List(_) => (lists + 1, solids),
            Children::Solid(_) => (lists, solids + 1),
        })
    }

    //the nodes with moves and the visits of a solid node whose moves with a node are still a list
    pub fn lists_with_a_solid_nodes_visits(&self) -> usize {
        let kept_its_list =
            |node: &&Node| node.has_edges() && node.visits() >= self.solid_threshold && matches!(node.children(), Children::List(_));
        (0..self.nodes.len()).map(|id| self.nodes.get(id)).filter(kept_its_list).count()
    }

    pub fn prior(&self, parent: &Node, edge: usize) -> f64 {
        f64::from(self.edges.prior(parent.edges(), edge))
    }

    pub fn root_index(&self) -> usize {
        self.root
    }

    //the node made solid at once, as a thread with the tree to itself would
    pub fn solidify(&self, node: &Node) {
        let id = (0..self.nodes.len()).find(|&id| std::ptr::eq(self.nodes.get(id), node)).expect("a node of the tree");
        self.make_solid(id);
    }

    //edges numbered as a one thread search's Tree has them, in the order the nodes were expanded in
    pub fn to_tree(&self) -> Tree {
        assert_eq!(self.nodes.used(), self.nodes.len(), "nodes given back: to_reachable_tree numbers what is left");
        let count = self.nodes.len();
        let mut expanded: Vec<(usize, usize)> = (0..count)
            .filter_map(|id| Some(self.nodes.get(id).edges()).filter(|edges| !edges.is_empty()).map(|edges| (edges.start, id)))
            .collect();
        expanded.sort_unstable();
        let mut first_edge = vec![0; count];
        let (mut moves, mut priors) = (Vec::new(), Vec::new());
        for &(_, id) in &expanded {
            first_edge[id] = moves.len();
            let edges = self.nodes.get(id).edges();
            for edge in edges.clone() {
                moves.push(self.edges.chess_move(edge));
                priors.push(self.edges.prior(edges.clone(), edge));
            }
        }
        let graph = self.graph.as_ref().map(|_| {
            let halfmoves: Vec<u16> = (0..count).map(|id| self.nodes.get(id).halfmoves()).collect();
            let table = (0..count).map(|id| (graph_key(self.nodes.get(id).key(), u32::from(halfmoves[id])), id as u32)).collect();
            Graph { table, halfmoves, held: None }
        });
        let (mut links, mut slots) = (vec![super::UNLINKED], Vec::new());
        let nodes = (0..count)
            .map(|id| {
                let node = self.nodes.get(id);
                let (visit_count, in_flight) = node.counts();
                let edges = node.edges();
                let mut copy = super::Node {
                    visit_count,
                    in_flight,
                    value_sum: node.value_sum(),
                    visited_policy_sqrt: node.visited_policy_sqrt(),
                    edges_at: if edges.is_empty() { 0 } else { (first_edge[id] as u64) << 8 | edges.len() as u64 },
                    proof: node.proof(),
                    children: NO_LINK,
                    known_passes: node.known_passes() as u8,
                    known_to: node.known_to() as u8,
                };
                let children = self.links.linked(node).map(|(place, link)| (place, slot_of(self.links.record(link))));
                let solid = copy.solid_in_a_copy();
                super::link_up(&mut copy, &mut links, &mut slots, solid, children);
                copy
            })
            .collect();
        Tree {
            board: self.board.clone(),
            keys: self.keys.clone(),
            nodes,
            root: self.root,
            node_keys: (0..count).map(|id| self.nodes.get(id).key()).collect(),
            moves,
            priors,
            noised_priors: Vec::new(),
            links,
            slots,
            memory_limit: usize::MAX,
            pending: Vec::new(),
            logits: Vec::new(),
            full: self.is_full(),
            graph,
        }
    }
}

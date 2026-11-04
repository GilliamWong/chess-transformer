//the protocol between threads: edges are published (release) after they are written, a visit is counted (release) after the
//node's value is added, and readers load both with acquire. A move's link is written before it is put into its node's list
//(release); of two threads that put one in for a move, one link is kept. A reader may see a half backed up path, never a torn
//field. A proof rewrites a node's sum and its ancestors', which cannot happen beside backups, so proving a node with visits
//holds the gate exclusively.
//A node keeps a count of the moves (list links and solid slots) that name it, and 1 if it is the root. The thread that makes a
//move's node, or takes it from the table (only a node with a count can be taken), gives the move that count. The thread that
//drops the last count owns the node's death (a count never rises from 0) and puts it on the garbage, which sweepers and descents
//short of room free: its table entry, then its children's counts, links, slots and edges, the node last. A descent reaches a
//node only through a move or a table hit that takes a count, so a dead node is read by no one but whoever frees it

mod arena;
mod cache;
mod reclaim;
mod reuse;
#[cfg(test)]
mod audit;

use arena::{Alone, Children, Edges, Link, Links, Node, Nodes, Slot};
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
    //takes the share again only once the waiting thread has had the tree: the gate is not fair, a share taken at once passes it
    pub fn give_way(&mut self) {
        if self.tree.writers.load(Ordering::Relaxed) > 0 {
            self.tree.counts.gave_way.fetch_add(1, Ordering::Relaxed);
            self.guard = None;
            while self.tree.writers.load(Ordering::Acquire) > 0 {
                std::thread::yield_now();
            }
            self.guard = Some(self.tree.gate.read().expect("the gate"));
        }
    }
}

const NODE_BYTES: usize = size_of::<Node>();
const EDGE_BYTES: usize = size_of::<f32>() + size_of::<PackedMove>();
const LINK_BYTES: usize = size_of::<Link>();
const SLOT_BYTES: usize = size_of::<Slot>();
const TABLE_ENTRY_BYTES: usize = 32;
const DESCENT_NODE_BYTES: usize = NODE_BYTES + LINK_BYTES;
pub const DESCENT_BYTES: usize = DESCENT_NODE_BYTES + MOST_MOVES * EDGE_BYTES;

const SOLID_BATCH: usize = 32;

//the most calls that descend at once, which the room left is shared out by
const MOST_CALLS: usize = 64;

//the leaves a call has room for from the start: rounds are of a few dozen
const LEAVES_AT_ONCE: usize = 64;

//the nodes a descent short of room frees at a time
const ASSIST_BATCH: usize = 64;

const TABLE_SHARDS: usize = 256;

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

enum Arrival {
    Inner,
    Proven(Proof),
    //a node that waits for the network or is being expanded
    Collision,
    //a node to evaluate: a new leaf the descent owns, or a game's end or repeat reached again
    Leaf,
    //with transpositions, above the node the path's last move leads to (not counted in flight there): a repeat of a
    //position on the path, or a move with fewer visits than its node
    Repeat,
    CatchUp,
}

struct Call {
    simulations: u32,
    collisions: usize,
    //the nodes of every collision's path, which are in flight until the call ends
    collided: Vec<usize>,
    leaves: Vec<PendingLeaf>,
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

//a test's step between a reader's load of a node's children and what it does with them, to change the node as another thread could
#[cfg(test)]
thread_local!(pub static AFTER_CHILDREN_LOADED: std::cell::Cell<Option<fn(&SharedTree, &Node)>> = const { std::cell::Cell::new(None) });

impl SharedTree {
    pub fn new(board: Chess, keys: Vec<u64>) -> SharedTree {
        let root_key = *keys.last().expect("the root's key");
        let nodes = Nodes::new();
        let root = nodes.push(root_key);
        SharedTree {
            board,
            keys,
            root,
            nodes,
            edges: Edges::new(),
            links: Links::new(),
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
            graph: None,
        }
    }

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
        self.nodes.len() * NODE_BYTES + self.edges.len() * EDGE_BYTES + self.links.len() * LINK_BYTES + self.links.slots() * SLOT_BYTES
    }

    //whether the arenas hold much more than the memory limit allows in use, which freed room that nothing fits does
    pub fn sprawls(&self, memory_limit: usize) -> bool {
        memory_limit != usize::MAX && self.footprint_bytes() > memory_limit + memory_limit / 4
    }

    pub fn pending(&self) -> usize {
        self.waiting.load(Ordering::Acquire)
    }

    //the bytes that descents in progress and waiting leaves may still allocate: the most a leaf's moves take
    pub fn reserved_bytes(&self) -> usize {
        self.reserved.load(Ordering::Acquire) + self.pending() * MOST_MOVES * EDGE_BYTES
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

    fn link_visits(&self, link: u32) -> u32 {
        match (link, &self.graph) {
            (NO_LINK, _) => 0,
            (_, Some(_)) => self.links.visits(link).load(Ordering::Acquire),
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
            (proven, visits, mean_score, f64::from(self.edges.prior(edge)))
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

    //returns the edge and its link (NO_LINK for a list's move without one, a solid slot whether it has a node or not)
    fn select_edge(&self, id: usize, settings: &SearchSettings, repeats: Option<&Repeats>) -> (usize, u32) {
        let parent = self.nodes.get(id);
        let edges = parent.edges();
        let scan = self.scan(parent, settings);
        let value = |prior: &AtomicU32, child: u32, own_visits: &AtomicU32| self.move_value(&scan, repeats, prior, child, own_visits);
        let (mut best, mut best_u, mut best_link) = (0, f64::NEG_INFINITY, NO_LINK);
        let priors = self.edges.priors(edges.clone());
        match parent.children() {
            Children::Solid(first) => {
                for (index, (prior, slot)) in priors.iter().zip(self.links.solid_slots(first, edges.len())).enumerate() {
                    let child = slot.child();
                    if let Some((u, node)) = value(prior, child, slot.visits())
                        && u > best_u
                    {
                        (best, best_u, best_link) = (index, u, first + index as u32);
                        self.prefetch_children(node);
                    }
                    //the first move without a node is the best of those that follow it: their priors are no higher
                    if child == NO_NODE {
                        break;
                    }
                }
            }
            //the links come in their moves' order as the moves do: the next one's place is known ahead
            Children::List(head) => {
                let unvisited = AtomicU32::new(0);
                let place_of = |link: u32| if link == NO_LINK { usize::MAX } else { self.links.get(link).place() };
                let (mut next_link, mut next_place) = (head, place_of(head));
                for (index, prior) in priors.iter().enumerate() {
                    let (link, found) = if index == next_place {
                        let record = self.links.get(next_link);
                        let found = (next_link, value(prior, record.child(), record.visits()));
                        next_link = record.next();
                        next_place = place_of(next_link);
                        found
                    } else {
                        (NO_LINK, value(prior, NO_NODE, &unvisited))
                    };
                    if let Some((u, child)) = found
                        && u > best_u
                    {
                        (best, best_u, best_link) = (index, u, link);
                        self.prefetch_children(child);
                    }
                    if link == NO_LINK {
                        break;
                    }
                }
            }
        }
        (edges.start + best, best_link)
    }

    //the lines a descent through the move's `child` reads first, asked for while the node's other moves are scored
    #[inline(always)]
    fn prefetch_children(&self, child: Option<&Node>) {
        if let Some(child) = child {
            let edges = child.edges();
            if !edges.is_empty() {
                self.edges.prefetch(edges.start);
                self.links.prefetch(child.children());
            }
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

    //select_edge's value of a move with its node; None for a proven loss, never chosen
    #[inline(always)]
    fn move_value(
        &self,
        scan: &Scan,
        repeats: Option<&Repeats>,
        prior: &AtomicU32,
        child: u32,
        own_visits: &AtomicU32,
    ) -> Option<(f64, Option<&Node>)> {
        let prior = f64::from(f32::from_bits(prior.load(Ordering::Relaxed)));
        if child == NO_NODE {
            //sqrt_children_visits / 1.0 is sqrt_children_visits: no division
            return Some((scan.first_play_urgency + (scan.c_puct * prior * scan.sqrt_children_visits), None));
        }
        let child = self.nodes.get(child as usize);
        if child.proves_a_loss() {
            return None;
        }
        let (visits, in_flight) = child.counts();
        let q = if repeats.is_some_and(|repeats| repeats.repeats(child.key(), child.halfmoves())) {
            1.0 - scan.draw
        } else if visits > 0 {
            child.mean_score(visits)
        } else {
            scan.first_play_urgency
        };
        //with transpositions a move's own visits, its node's mean (every path's)
        let own = if scan.transpositions { own_visits.load(Ordering::Acquire) } else { visits };
        Some((q + (scan.c_puct * prior * (scan.sqrt_children_visits / f64::from(1 + own + in_flight))), Some(child)))
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
        let key = position_key(board);
        let made = match &self.graph {
            None => self.nodes.push(key),
            Some(graph) => {
                let halfmoves = halfmove_count(board.halfmoves());
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
                self.links.slot(link).claim(made as u32).map(|_| (made, link)).map_err(|other| (other as usize, link))
            }
            Children::List(_) => self.insert_link(parent, place, made),
        };
        kept.unwrap_or_else(|other| {
            self.counts.orphans.fetch_add(1, Ordering::Relaxed);
            self.drop_reference(made);
            other
        })
    }

    //a link for the node `made` in the parent's list, at the place of its move: Ok if it is kept, else Err of the node and link
    //the move has
    fn insert_link(&self, parent: &Node, place: usize, made: usize) -> Result<(usize, u32), (usize, u32)> {
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
                        mine = self.links.push(made as u32, place);
                    }
                    if self.links.insert(pointer, at, mine) {
                        return Ok((made, mine));
                    }
                    jitter();
                }
            }
        }
    }

    //the link a waiting leaf's path holds for a move of the node, in case the node's list was made solid since
    fn current_link(&self, id: usize, link: u32) -> u32 {
        match self.nodes.get(id).children() {
            Children::Solid(first) if listed(link) => first + self.links.get(link).place() as u32,
            _ => link,
        }
    }

    fn queue_solid(&self, id: usize, visits_before: u32) {
        if visits_before + 1 == self.solid_threshold && matches!(self.nodes.get(id).children(), Children::List(_)) {
            let mut queue = self.to_solidify.lock().expect("the solid queue");
            queue.push(id as u32);
            self.queued.store(queue.len(), Ordering::Release);
        }
    }

    fn solidify_queued<'a>(&'a self, hold: &mut Hold<'a>) {
        if self.queued.load(Ordering::Acquire) < self.solid_batch {
            return;
        }
        let queued = {
            let mut queue = self.to_solidify.lock().expect("the solid queue");
            if queue.len() < self.solid_batch {
                return;
            }
            self.queued.store(0, Ordering::Release);
            std::mem::take(&mut *queue)
        };
        self.exclusive(hold, || {
            for id in queued {
                self.make_solid(id as usize);
            }
        });
    }

    //the node's list copied into one slot a move, each move's count with it; needs the tree to itself. The list stays readable
    //until the next move: a waiting leaf may name one of its links
    fn make_solid(&self, id: usize) {
        let node = self.nodes.get(id);
        let moves = node.edges().len();
        let Children::List(head) = node.children() else { return };
        if moves == 0 {
            return;
        }
        let children =
            self.links.linked(node).map(|(place, link)| (place, self.links.child(link), self.links.visits(link).load(Ordering::Relaxed)));
        set_links(&self.links, node, moves, true, children);
        if head != NO_LINK {
            self.retired.lock().expect("the retired lists").push(head);
        }
    }

    fn arrive(&self, id: usize) -> Arrival {
        let node = self.nodes.get(id);
        let (visits, in_flight) = node.bump();
        jitter();
        if let Some(proof) = node.proof() {
            return Arrival::Proven(proof);
        }
        if visits == 0 {
            return if in_flight == 0 { Arrival::Leaf } else { Arrival::Collision };
        }
        if node.has_edges() { Arrival::Inner } else { Arrival::Leaf }
    }

    //one simulation down to a leaf, every node on the way counted in flight; the path ends on the node it reached. With
    //transpositions `edges` gets the links of the moves between the path's nodes, and the descent may stop above a shared
    //node (a repeat or a transposition visit, as Tree::gather's), that node the path's last but not counted in flight. The moves
    //are not played on `board` but left in `unplayed`, for where the position is needed
    fn descend(
        &self,
        settings: &SearchSettings,
        board: &mut Chess,
        unplayed: &mut Vec<usize>,
        keys: &mut Vec<u64>,
        path: &mut Vec<usize>,
        edges: &mut Vec<usize>,
    ) -> Arrival {
        let path_start = keys.len() - 1;
        let mut node = self.root;
        path.push(node);
        loop {
            match self.arrive(node) {
                Arrival::Inner => {
                    let repeats = self.graph.as_ref().map(|_| Repeats::new(keys, path_start, u32::from(self.nodes.get(node).halfmoves())));
                    let (edge, selected) = self.select_edge(node, settings, repeats.as_ref());
                    unplayed.push(edge);
                    let (child, link) = self.child_node(node, edge, selected, board, unplayed);
                    node = child;
                    path.push(node);
                    let child = self.nodes.get(node);
                    let mut stop = None;
                    if let Some(repeats) = &repeats {
                        edges.push(link as usize);
                        //a repeat of this path's position, whatever the node is to other paths, or a node that knows
                        //more than its move
                        if repeats.repeats(child.key(), child.halfmoves()) {
                            stop = Some(Arrival::Repeat);
                        } else if child.has_edges()
                            && child.proof().is_none()
                            && self.links.visits(link).load(Ordering::Acquire) < child.visits()
                        {
                            stop = Some(Arrival::CatchUp);
                        }
                    }
                    keys.push(child.key());
                    if let Some(stop) = stop {
                        return stop;
                    }
                }
                other => return other,
            }
        }
    }

    fn take_off(&self, path: &[usize]) {
        for &node in path {
            self.nodes.get(node).take_off(1);
        }
    }

    //the node's moves as edges, without nodes. The node is the caller's: nothing else reads or writes its edges until they
    //are published
    fn expand(&self, id: usize, moves: &MoveList, priors: &[f32]) {
        let first = self.edges.alloc(moves.len());
        self.edges.write(first, moves, priors);
        jitter();
        self.nodes.get(id).publish_edges(first, moves.len());
    }

    //with transpositions every move's link (`links`, descend's `edges`) counts the visit, and a node's visited policy is
    //refreshed when one of its moves has its first
    fn back_up(&self, path: &[usize], links: &[usize], value: f64) {
        if self.graph.is_none() {
            return self.backup(path, value);
        }
        let mut score = 1.0 - value;
        for (depth, &id) in path.iter().enumerate().rev() {
            let node = self.nodes.get(id);
            if let Some(proof) = node.proof() {
                score = proof.value;
            }
            node.add_value(score);
            jitter();
            self.queue_solid(id, node.visit(1));
            if let Some(&link) = links.get(depth)
                && self.links.visits(self.current_link(id, link as u32)).fetch_add(1, Ordering::AcqRel) == 0
            {
                self.refresh_visited_policy(id);
            }
            score = 1.0 - score;
        }
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
            let sum = python_sum(visited.map(|(place, _)| f64::from(f32::from_bits(priors[place].load(Ordering::Relaxed)))));
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

    //the descents a call may make without `reserve`: its share of a quarter of the room left (nodes, and bytes under a memory
    //limit), on the thought that at most MOST_CALLS calls are descending. Far from the limit that is more than a call descends
    pub fn free_descents(&self, memory_limit: usize) -> usize {
        let nodes = Nodes::CAPACITY.saturating_sub(self.nodes.len());
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
        if limited && self.nodes.len() + 64 >= Nodes::CAPACITY {
            return false;
        }
        if memory_limit == usize::MAX {
            return true;
        }
        self.reserved.fetch_add(DESCENT_BYTES, Ordering::AcqRel);
        while limited && self.allocated_bytes() + self.reserved_bytes() > memory_limit {
            if self.reclaim(ASSIST_BATCH) > 0 {
                self.counts.assists.fetch_add(1, Ordering::Relaxed);
            } else if !self.idle() {
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
        let mut hold = self.hold();
        let leaves = Vec::with_capacity(LEAVES_AT_ONCE.min(round.leaves as usize));
        let mut call = Call { simulations: 0, collisions: 0, collided: Vec::new(), leaves };
        self.gather_in(&mut hold, round, settings, tables, cache, boards, &mut call);
        self.take_off(&call.collided);
        call.leaves
    }

    #[allow(clippy::too_many_arguments)]
    fn gather_in<'a>(
        &'a self,
        hold: &mut Hold<'a>,
        round: &Round,
        settings: &SearchSettings,
        tables: Option<&Tables>,
        cache: Option<&SharedCache>,
        boards: &mut Vec<u8>,
        call: &mut Call,
    ) {
        let root_covered = tables.is_some_and(|tables| tables.covers(&self.board));
        let mut keys = self.keys.clone();
        let root_keys = keys.len();
        let mut path = Vec::new();
        let mut edges = Vec::new();
        let mut unplayed = Vec::new();
        let mut priors = Vec::new();
        let mut free_descents = self.free_descents(round.memory_limit);
        loop {
            //a writer let in here may have ended the search and closed the tree (end's quiet), so the room is looked at
            //after: a leaf taken in a closed tree belongs to a search that is gone
            hold.give_way();
            self.solidify_queued(hold);
            if !(call.leaves.len() < round.leaves as usize && call.simulations < round.simulations && self.has_room(round.visit_limit)) {
                break;
            }
            let reserved = free_descents == 0;
            if reserved {
                //the root's own evaluation always fits, so that a full search still has moves to report
                if !self.reserve(round.memory_limit, self.root().has_edges()) {
                    self.full.store(true, Ordering::Relaxed);
                    break;
                }
            } else {
                free_descents -= 1;
            }
            let mut board = self.board.clone();
            keys.truncate(root_keys);
            path.clear();
            edges.clear();
            unplayed.clear();
            let arrival = self.descend(settings, &mut board, &mut unplayed, &mut keys, &mut path, &mut edges);
            let give_back = || self.release(round.memory_limit, reserved);
            let node = *path.last().expect("the root at least");
            match arrival {
                Arrival::Inner => unreachable!("a descent ends on a leaf"),
                Arrival::Proven(proof) => {
                    give_back();
                    call.simulations += 1;
                    self.back_up(&path, &edges, 1.0 - proof.value);
                    continue;
                }
                //a repeat is a draw for this path, a transposition visit the shared node's mean: the visit is the
                //move's, the node's numbers stay as they are
                Arrival::Repeat | Arrival::CatchUp => {
                    give_back();
                    call.simulations += 1;
                    let shared = self.nodes.get(path.pop().expect("the shared node"));
                    //a mean read while other threads back up through the node may be off by their values: kept a score
                    let value = match arrival {
                        Arrival::Repeat => 1.0 - settings.draw,
                        _ => shared.mean_score(shared.visits()).clamp(0.0, 1.0),
                    };
                    self.back_up(&path, &edges, value);
                    continue;
                }
                Arrival::Collision => {
                    give_back();
                    if call.collisions == round.collisions as usize {
                        self.take_off(&path);
                        break;
                    }
                    call.collisions += 1;
                    call.collided.extend_from_slice(&path);
                    continue;
                }
                Arrival::Leaf => {}
            }

            self.play_path(&mut board, &mut unplayed);
            let mut moves = board.legal_moves();
            let node_tables = tables.filter(|_| node != self.root && (!root_covered || board.halfmoves() == 0));
            //the root is searched whatever the rules say of it: Searches.start refuses only a mate or stalemate
            let game_over = if node == self.root { None } else { game_result(&board, &keys, &moves, settings.draw, node_tables) };
            //a repeat inside the search is a draw, but not a proven one: the game need not repeat
            let (&leaf_key, path_keys) = keys[root_keys - 1..].split_last().expect("the root's key");
            //with transpositions a leaf is no repeat inside the search (Tree::gather's)
            let value = game_over.or_else(|| path_keys.contains(&leaf_key).then_some(settings.draw));
            call.simulations += 1;
            if let Some(value) = value {
                give_back();
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
                        if self.nodes.get(node).visits() == 0 {
                            //a leaf nobody has visited, and nobody can before its proof is seen: it rewrites no sum
                            self.nodes.get(node).set_proof(proof);
                            self.back_up(&path, &edges, value);
                            if self.proves_parent(&path) {
                                self.exclusive(hold, || self.propagate_proofs(&path));
                            }
                        } else {
                            self.exclusive(hold, || {
                                self.prove(&path, proof);
                                self.back_up(&path, &edges, value);
                                self.propagate_proofs(&path);
                            });
                        }
                    }
                    _ => {
                        self.back_up(&path, &edges, value);
                        if settings.certainty && self.proves_parent(&path) {
                            self.exclusive(hold, || self.propagate_proofs(&path));
                        }
                    }
                }
                continue;
            }
            sort_moves_in_python_order(&board, &mut moves);
            //a hash collision would give another number of moves
            let cached = cache.and_then(|cache| cache.get(leaf_key, &mut priors)).filter(|_| priors.len() == moves.len());
            match cached {
                Some(value) => {
                    give_back();
                    self.expand(node, &moves, &priors);
                    self.back_up(&path, &edges, f64::from(value));
                }
                None => {
                    encode_board(&board, boards);
                    self.waiting.fetch_add(1, Ordering::AcqRel);
                    call.leaves.push(PendingLeaf { path: path.clone(), edges: edges.clone(), key: leaf_key, moves, turn: board.turn() });
                    give_back();
                }
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
        self.back_up(&leaf.path, &leaf.edges, f64::from(value));
        if let Some(cache) = cache.filter(|_| kept) {
            cache.insert(leaf.key, logits, value);
        }
        self.waiting.fetch_sub(1, Ordering::AcqRel);
    }
}

//makes a node's links before it is published, from its moves with a node as (place, node, the move's visits) in order
fn set_links(links: &Links, node: &Node, moves: usize, solid: bool, children: impl Iterator<Item = (usize, u32, u32)>) {
    if solid {
        let first = links.solid(moves);
        for (place, child, move_visits) in children {
            links.set_solid(first + place as u32, child, move_visits);
        }
        node.make_solid(first);
        return;
    }
    let mut last = NO_LINK;
    for (place, child, move_visits) in children {
        let link = links.push(child, place);
        links.visits(link).store(move_visits, Ordering::Relaxed);
        if last == NO_LINK {
            node.list_head().store(link, Ordering::Relaxed);
        } else {
            links.set_next(last, link);
        }
        last = link;
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
            copy.set_counts(node.visit_count, node.in_flight, node.value_sum);
            copy.set_visited_policy_sqrt(node.visited_policy_sqrt);
            if let Some(proof) = node.proof {
                copy.set_proof(proof);
            }
            if node.has_edges() {
                let first = edges.alloc(node.edges().len());
                for (index, edge) in node.edges().enumerate() {
                    edges.set(first + index, tree.moves[edge], tree.priors[edge]);
                }
                let children = tree.linked(id).map(|(place, link)| (place, tree.link_child(link), tree.own_visits(link)));
                set_links(&links, copy, node.edges().len(), node.visit_count >= SOLID_TREE_THRESHOLD, children);
                copy.publish_edges(first, node.edges().len());
            }
        }
        let shared = SharedTree {
            root: tree.root_index(),
            nodes,
            edges,
            links,
            full: AtomicBool::new(tree.full),
            ..SharedTree::new(tree.board.clone(), tree.keys.clone())
        };
        shared.count_references();
        shared
    }

    pub fn node(&self, id: usize) -> &Node {
        self.nodes.get(id)
    }

    //select_edge's pick for a node nothing is working in, and the pick of a look at every move: the first of the best scores
    pub fn picked_and_best(&self, id: usize, settings: &SearchSettings) -> (usize, usize) {
        let parent = self.nodes.get(id);
        let scan = self.scan(parent, settings);
        let unvisited = AtomicU32::new(0);
        let mut best = (parent.edges().start, f64::NEG_INFINITY);
        for edge in parent.edges() {
            let (child, visits) = match self.link_of(parent, edge) {
                NO_LINK => (NO_NODE, &unvisited),
                link => (self.links.child(link), self.links.visits(link)),
            };
            let prior = &AtomicU32::new(self.edges.prior(edge).to_bits());
            if let Some((u, _)) = self.move_value(&scan, None, prior, child, visits)
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

    pub fn prior(&self, edge: usize) -> f64 {
        f64::from(self.edges.prior(edge))
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
            for edge in self.nodes.get(id).edges() {
                moves.push(self.edges.chess_move(edge));
                priors.push(self.edges.prior(edge));
            }
        }
        let graph = self.graph.as_ref().map(|_| {
            let halfmoves: Vec<u16> = (0..count).map(|id| self.nodes.get(id).halfmoves()).collect();
            let table = (0..count).map(|id| (graph_key(self.nodes.get(id).key(), u32::from(halfmoves[id])), id as u32)).collect();
            Graph { table, halfmoves }
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
                };
                let children = self
                    .links
                    .linked(node)
                    .map(|(place, link)| (place, self.links.child(link), self.links.visits(link).load(Ordering::Relaxed)));
                super::link_up(&mut copy, &mut links, &mut slots, visit_count >= SOLID_TREE_THRESHOLD, children);
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

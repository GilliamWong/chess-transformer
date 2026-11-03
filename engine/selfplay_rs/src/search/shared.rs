//the protocol between threads: edges are published (release) after they are written, a visit is counted (release) after the
//node's value is added, and readers load both with acquire. A move's link is written before it is put into its node's list
//(release); of two threads that put one in for a move, one link is kept. A reader may see a half backed up path, never a torn
//field. A proof rewrites a node's sum and its ancestors', which cannot happen beside backups, so proving a node with visits
//holds the gate exclusively

mod arena;
mod cache;
mod reuse;

use arena::{Children, Edges, Link, Links, Node, Nodes, Slot};
pub use cache::SharedCache;
use std::sync::atomic::{AtomicU32, AtomicU64};
use std::sync::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};

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
const DESCENT_BYTES: usize = DESCENT_NODE_BYTES + MOST_MOVES * EDGE_BYTES;

const SOLID_BATCH: usize = 32;

const TABLE_SHARDS: usize = 256;

struct SharedGraph {
    shards: Box<[Mutex<KeyMap<u32>>]>,
    entries: AtomicUsize,
}

impl SharedGraph {
    fn new() -> SharedGraph {
        SharedGraph { shards: (0..TABLE_SHARDS).map(|_| Mutex::new(KeyMap::default())).collect(), entries: AtomicUsize::new(0) }
    }

    fn shard(&self, entry: u64) -> &Mutex<KeyMap<u32>> {
        &self.shards[(entry >> 56) as usize % TABLE_SHARDS]
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
    collisions: Vec<Vec<usize>>,
    leaves: Vec<PendingLeaf>,
}

#[derive(Default)]
pub struct Counts {
    pub gave_way: AtomicU64,
    pub exclusive: AtomicU64,
    pub orphans: AtomicU64,
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
    solid_threshold: u32,
    solid_batch: usize,
    pending: Mutex<HashMap<u64, Vec<PendingLeaf>>>,
    //network leaves gathered and not yet backed up
    waiting: AtomicUsize,
    //bytes that descents in progress and waiting leaves may still allocate (under a memory limit)
    reserved: AtomicUsize,
    //no room for another simulation: the search ends as at its simulations, until a move is played
    full: AtomicBool,
    closed: AtomicBool,
    //calls hold the gate shared, whatever needs the tree to itself holds it exclusive; writers counts the threads waiting
    gate: RwLock<()>,
    writers: AtomicUsize,
    pub counts: Counts,
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
            solid_threshold: SOLID_TREE_THRESHOLD,
            solid_batch: SOLID_BATCH,
            pending: Mutex::new(HashMap::new()),
            waiting: AtomicUsize::new(0),
            reserved: AtomicUsize::new(0),
            full: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            gate: RwLock::new(()),
            writers: AtomicUsize::new(0),
            counts: Counts::default(),
            graph: None,
        }
    }

    pub fn with_transpositions(mut self, transpositions: bool) -> SharedTree {
        assert_eq!(self.nodes.len(), 1, "transpositions from a tree's start");
        if transpositions {
            let graph = SharedGraph::new();
            self.root().set_halfmoves(halfmove_count(self.board.halfmoves()));
            let entry = graph_key(self.root().key(), self.board.halfmoves());
            graph.shard(entry).lock().expect("a table shard").insert(entry, self.root as u32);
            graph.entries.fetch_add(1, Ordering::Relaxed);
            self.graph = Some(graph);
        }
        self
    }

    pub fn root(&self) -> &Node {
        self.nodes.get(self.root)
    }

    pub fn is_full(&self) -> bool {
        self.full.load(Ordering::Relaxed)
    }

    pub fn allocated_bytes(&self) -> usize {
        let graph = self.graph.as_ref().map_or(0, |graph| graph.entries.load(Ordering::Relaxed) * TABLE_ENTRY_BYTES);
        self.nodes.len() * NODE_BYTES
            + self.edges.len() * EDGE_BYTES
            + self.links.len() * LINK_BYTES
            + self.links.slots() * SLOT_BYTES
            + graph
    }

    pub fn pending(&self) -> usize {
        self.waiting.load(Ordering::Acquire)
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
        match parent.children() {
            Children::Solid(first) => Some(first + place as u32).filter(|&link| self.links.child(link) != NO_NODE).unwrap_or(NO_LINK),
            Children::List(_) => self.links.find(parent.list_head(), place).unwrap_or(NO_LINK),
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
        let (parent_visits, _) = parent.counts();
        let c_puct =
            settings.c_puct + settings.c_puct_factor * ((f64::from(parent_visits) + settings.c_puct_base) / settings.c_puct_base).ln();
        let edges = parent.edges();
        let first_play_urgency = (1.0 - parent.mean_score(parent_visits)) - settings.fpu_reduction * parent.visited_policy_sqrt();
        let sqrt_children_visits = f64::from(parent_visits.saturating_sub(1).max(1)).sqrt();

        let scan = Scan {
            c_puct,
            sqrt_children_visits,
            first_play_urgency,
            noised: false,
            transpositions: self.graph.is_some(),
            draw: settings.draw,
        };
        let value = |prior: &AtomicU32, child: u32, own_visits: &AtomicU32| self.move_value(&scan, repeats, prior, child, own_visits);
        let (mut best, mut best_u, mut best_link) = (0, f64::NEG_INFINITY, NO_LINK);
        let priors = self.edges.priors(edges.clone());
        match parent.children() {
            Children::Solid(first) => {
                for (index, (prior, slot)) in priors.iter().zip(self.links.solid_slots(first, edges.len())).enumerate() {
                    if let Some(u) = value(prior, slot.child(), slot.visits())
                        && u > best_u
                    {
                        (best, best_u, best_link) = (index, u, first + index as u32);
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
                    if let Some(u) = found
                        && u > best_u
                    {
                        (best, best_u, best_link) = (index, u, link);
                    }
                }
            }
        }
        (edges.start + best, best_link)
    }

    //select_edge's value of a move; None for a proven loss, never chosen
    #[inline(always)]
    fn move_value(&self, scan: &Scan, repeats: Option<&Repeats>, prior: &AtomicU32, child: u32, own_visits: &AtomicU32) -> Option<f64> {
        let prior = f64::from(f32::from_bits(prior.load(Ordering::Relaxed)));
        if child == NO_NODE {
            //sqrt_children_visits / 1.0 is sqrt_children_visits: no division
            return Some(scan.first_play_urgency + (scan.c_puct * prior * scan.sqrt_children_visits));
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
        Some(q + (scan.c_puct * prior * (scan.sqrt_children_visits / f64::from(1 + own + in_flight))))
    }

    //the node and link of the parent's move `edge` (select_edge's link), made if the move has none yet; board is the node's
    //position. Of two threads that make one, one node is kept and the other's stays unused. With transpositions it is the
    //table's node of the position, made by one thread under the shard's lock
    fn child_node(&self, parent: usize, edge: usize, link: u32, board: &Chess) -> (usize, u32) {
        //select_edge names no move when it chose none (every move a proven loss, the first taken): its node if it has one
        let link = if link == NO_LINK { self.link_of(self.nodes.get(parent), edge) } else { link };
        if link != NO_LINK && self.links.child(link) != NO_NODE {
            return (self.links.child(link) as usize, link);
        }
        let key = position_key(board);
        let made = match &self.graph {
            None => self.nodes.push(key),
            Some(graph) => {
                let entry = graph_key(key, board.halfmoves());
                let mut shard = graph.shard(entry).lock().expect("a table shard");
                match shard.get(&entry) {
                    Some(&shared) => shared as usize,
                    None => {
                        let made = self.nodes.push(key);
                        self.nodes.get(made).set_halfmoves(halfmove_count(board.halfmoves()));
                        shard.insert(entry, made as u32);
                        graph.entries.fetch_add(1, Ordering::Relaxed);
                        made
                    }
                }
            }
        };
        jitter();
        let parent = self.nodes.get(parent);
        let place = edge - parent.edges().start;
        let kept = match parent.children() {
            Children::Solid(first) => {
                let link = first + place as u32;
                self.links.slot(link).claim(made as u32).map(|_| (made, link)).map_err(|other| (other as usize, link))
            }
            Children::List(_) => {
                let mut mine = NO_LINK;
                loop {
                    match self.links.find(parent.list_head(), place) {
                        Ok(found) => break Err((self.links.child(found) as usize, found)),
                        Err((pointer, at)) => {
                            if mine == NO_LINK {
                                mine = self.links.push(made as u32, place);
                            }
                            if self.links.insert(pointer, at, mine) {
                                break Ok((made, mine));
                            }
                            jitter();
                        }
                    }
                }
            }
        };
        kept.unwrap_or_else(|other| {
            self.counts.orphans.fetch_add(1, Ordering::Relaxed);
            other
        })
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

    //the node's list copied into one slot a move; needs the tree to itself
    fn make_solid(&self, id: usize) {
        let node = self.nodes.get(id);
        let moves = node.edges().len();
        if moves == 0 || !matches!(node.children(), Children::List(_)) {
            return;
        }
        let children =
            self.links.linked(node).map(|(place, link)| (place, self.links.child(link), self.links.visits(link).load(Ordering::Relaxed)));
        set_links(&self.links, node, moves, true, children);
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
    //node (a repeat or a transposition visit, as Tree::gather's), that node the path's last but not counted in flight
    fn descend(
        &self,
        settings: &SearchSettings,
        board: &mut Chess,
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
                    let repeats = self.graph.as_ref().map(|_| Repeats::new(keys, path_start, board.halfmoves()));
                    let (edge, selected) = self.select_edge(node, settings, repeats.as_ref());
                    let chess_move = self.edges.chess_move(edge).to_move(board);
                    board.play_unchecked(chess_move);
                    let (child, link) = self.child_node(node, edge, selected, board);
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
    //first (a finished leaf adds its visit before it stops waiting, so it is never missed)
    fn has_room(&self, visit_limit: u32) -> bool {
        self.pending() + (self.root().visits() as usize) < visit_limit as usize && !self.closed.load(Ordering::Acquire)
    }

    //no call gathers any more; the caller has the tree to itself (quiet)
    pub fn close(&self) {
        self.closed.store(true, Ordering::Release);
    }

    pub fn reopen(&mut self) {
        *self.closed.get_mut() = false;
    }

    //room for a descent's node and its leaf's moves, under the memory limit; true if it is kept for the descent
    fn reserve(&self, memory_limit: usize) -> bool {
        if self.nodes.len() + 64 >= Nodes::CAPACITY {
            return false;
        }
        if memory_limit == usize::MAX {
            return true;
        }
        let reserved = self.reserved.fetch_add(DESCENT_BYTES, Ordering::AcqRel) + DESCENT_BYTES;
        if self.allocated_bytes() + reserved <= memory_limit {
            return true;
        }
        self.reserved.fetch_sub(DESCENT_BYTES, Ordering::AcqRel);
        false
    }

    //a descent's reservation back, but the moves of the leaf it leaves waiting
    fn release(&self, memory_limit: usize, waiting: bool) {
        if memory_limit != usize::MAX {
            self.reserved.fetch_sub(if waiting { DESCENT_NODE_BYTES } else { DESCENT_BYTES }, Ordering::AcqRel);
        }
    }

    //the network leaves of the round, in the order they were gathered in, for finish_leaf
    pub fn queue(&self, round: u64, leaves: Vec<PendingLeaf>) {
        if !leaves.is_empty() {
            self.pending.lock().expect("the pending leaves").entry(round).or_default().extend(leaves);
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
    ) -> Result<Vec<PendingLeaf>, SyzygyError> {
        let mut hold = self.hold();
        let mut call = Call { simulations: 0, collisions: Vec::new(), leaves: Vec::new() };
        let gathered = self.gather_in(&mut hold, round, settings, tables, cache, boards, &mut call);
        for path in &call.collisions {
            self.take_off(path);
        }
        gathered.map(|()| call.leaves)
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
    ) -> Result<(), SyzygyError> {
        let root_covered = tables.is_some_and(|tables| tables.covers(&self.board));
        let mut keys = self.keys.clone();
        let root_keys = keys.len();
        let mut path = Vec::new();
        let mut edges = Vec::new();
        let mut priors = Vec::new();
        loop {
            //a writer let in here may have ended the search and closed the tree (end's quiet), so the room is looked at
            //after: a leaf taken in a closed tree belongs to a search that is gone
            hold.give_way();
            self.solidify_queued(hold);
            if !(call.leaves.len() < round.leaves as usize && call.simulations < round.simulations && self.has_room(round.visit_limit)) {
                break;
            }
            //the root's own evaluation always fits, so that a full search still has moves to report
            let reserved = self.root().has_edges();
            if reserved && !self.reserve(round.memory_limit) {
                self.full.store(true, Ordering::Relaxed);
                break;
            }
            let mut board = self.board.clone();
            keys.truncate(root_keys);
            path.clear();
            edges.clear();
            let arrival = self.descend(settings, &mut board, &mut keys, &mut path, &mut edges);
            let give_back = |waiting: bool| {
                if reserved {
                    self.release(round.memory_limit, waiting);
                }
            };
            let node = *path.last().expect("the root at least");
            match arrival {
                Arrival::Inner => unreachable!("a descent ends on a leaf"),
                Arrival::Proven(proof) => {
                    give_back(false);
                    call.simulations += 1;
                    self.back_up(&path, &edges, 1.0 - proof.value);
                    continue;
                }
                //a repeat is a draw for this path, a transposition visit the shared node's mean: the visit is the
                //move's, the node's numbers stay as they are
                Arrival::Repeat | Arrival::CatchUp => {
                    give_back(false);
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
                    give_back(false);
                    if call.collisions.len() == round.collisions as usize {
                        self.take_off(&path);
                        break;
                    }
                    call.collisions.push(path.clone());
                    continue;
                }
                Arrival::Leaf => {}
            }

            let mut moves = board.legal_moves();
            let node_tables = tables.filter(|_| node != self.root && (!root_covered || board.halfmoves() == 0));
            //the root is searched whatever the rules say of it: Searches.start refuses only a mate or stalemate
            let game_over = if node == self.root {
                None
            } else {
                match game_result(&board, &keys, &moves, settings.draw, settings.repetitions_for_a_draw, node_tables) {
                    Ok(game_over) => game_over,
                    Err(error) => {
                        give_back(false);
                        self.take_off(&path);
                        return Err(error);
                    }
                }
            };
            //a repeat inside the search is a draw, but not a proven one: the game need not repeat
            let (&leaf_key, path_keys) = keys[root_keys - 1..].split_last().expect("the root's key");
            //with transpositions a leaf is no repeat inside the search (Tree::gather's)
            let value = game_over.or_else(|| path_keys.contains(&leaf_key).then_some(settings.draw));
            call.simulations += 1;
            if let Some(value) = value {
                give_back(false);
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
                    give_back(false);
                    self.expand(node, &moves, &priors);
                    self.back_up(&path, &edges, f64::from(value));
                }
                None => {
                    encode_board(&board, boards);
                    self.waiting.fetch_add(1, Ordering::AcqRel);
                    call.leaves.push(PendingLeaf { path: path.clone(), edges: edges.clone(), key: leaf_key, moves, turn: board.turn() });
                    give_back(true);
                }
            }
        }
        Ok(())
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
        memory_limit: usize,
        logits: &mut Vec<f32>,
    ) {
        logits.clear();
        logits.extend(leaf.moves.iter().map(|&chess_move| answer(move_ids.id(chess_move, leaf.turn))));
        softmax(logits);
        let value = sharpen(answer(answer_width - 1), settings.value_sharpness);
        let node = *leaf.path.last().expect("the root at least");
        self.expand(node, &leaf.moves, logits);
        self.back_up(&leaf.path, &leaf.edges, f64::from(value));
        if let Some(cache) = cache {
            cache.insert(leaf.key, logits, value);
        }
        if memory_limit != usize::MAX {
            self.reserved.fetch_sub(MOST_MOVES * EDGE_BYTES, Ordering::AcqRel);
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
        SharedTree {
            root: tree.root_index(),
            nodes,
            edges,
            links,
            full: AtomicBool::new(tree.full),
            ..SharedTree::new(tree.board.clone(), tree.keys.clone())
        }
    }

    pub fn node(&self, id: usize) -> &Node {
        self.nodes.get(id)
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

    //edges numbered as a one thread search's Tree has them, in the order the nodes were expanded in
    pub fn to_tree(&self) -> Tree {
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

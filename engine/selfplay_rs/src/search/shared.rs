//a tree that many threads search at once (Searches' concurrent_descents): Tree's search with the counters in atomics and the
//storage in chunks that never move, so that the threads descend, expand and back up in the same tree at the same time. A
//simulation counts itself in flight on every node it passes on its way down, as Leela Zero's virtual loss does, so that
//the threads steer around each other as one thread's rounds do. One thread makes exactly Tree::gather's search.
//
//Protocol of a node's counts: a descent's bump returns the node's state before it. A node with no visits and nothing in
//flight is a leaf the descent now owns; with no visits and something in flight it is a collision, whether it waits for
//the network or is being expanded (its edges are published before its first visit). Edges are published (release) after
//they are written, a visit is counted (release) after the node's value is added, and readers load both with acquire.
//A reader may see a half backed up path, never a torn field. What a proof rewrites (a node's sum and its ancestors')
//cannot be done beside backups, so proofs are made with the gate held exclusively
mod arena;
mod cache;
mod reuse;

use arena::{Edges, Node, Nodes};
pub use cache::SharedCache;
use std::sync::atomic::AtomicU64;
use std::sync::{Mutex, RwLock, RwLockReadGuard};

use super::*;

//the share of the gate a call holds while it works
type Hold<'a> = RwLockReadGuard<'a, ()>;

//the bytes a node and an edge take, for the memory limit
const NODE_BYTES: usize = size_of::<Node>();
const EDGE_BYTES: usize = 3 * size_of::<u32>();
//what a descent may need: its node, and the moves of its leaf
const DESCENT_BYTES: usize = NODE_BYTES + MOST_MOVES * EDGE_BYTES;

//where a descent ended
enum Arrival {
    Inner,
    Proven(Proof),
    //a node that waits for the network or is being expanded
    Collision,
    //a node to evaluate: a new leaf the descent owns, or a game's end or repeat reached again
    Leaf,
}

//what one call of gather has so far
struct Call {
    simulations: u32,
    collisions: Vec<Vec<usize>>,
    leaves: Vec<PendingLeaf>,
}

//what the tree counts of itself, for the benches
#[derive(Default)]
pub struct Counts {
    pub exclusive: AtomicU64,
    pub orphans: AtomicU64,
}

pub struct SharedTree {
    pub board: Chess,   //the root's position
    pub keys: Vec<u64>, //the position_key of every position of the game, the root's last
    root: usize,
    nodes: Nodes,
    edges: Edges,
    //the network leaves of each round, queued by the call that gathered them
    pending: Mutex<HashMap<u64, Vec<PendingLeaf>>>,
    //network leaves gathered and not yet backed up
    waiting: AtomicUsize,
    //bytes that descents in progress and waiting leaves may still allocate (under a memory limit)
    reserved: AtomicUsize,
    //no room for another simulation: the search ends as at its simulations, until a move is played
    full: AtomicBool,
    //calls hold it shared, whatever needs the tree still holds it exclusive
    gate: RwLock<()>,
    pub counts: Counts,
}

//yields the thread now and then between a protocol's steps, to widen the windows the tests look for races in
#[cfg(test)]
fn jitter() {
    use std::cell::Cell;
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
            pending: Mutex::new(HashMap::new()),
            waiting: AtomicUsize::new(0),
            reserved: AtomicUsize::new(0),
            full: AtomicBool::new(false),
            gate: RwLock::new(()),
            counts: Counts::default(),
        }
    }

    //a copy of a Tree whose search is done (nothing waits for the network), nodes numbered as they are in the Tree
    pub fn from_tree(tree: &Tree) -> SharedTree {
        assert_eq!(tree.pending(), 0, "a tree to copy has nothing waiting");
        let nodes = Nodes::new();
        let edges = Edges::new();
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
                    edges.child(first + index).store(tree.children[edge], Ordering::Relaxed);
                }
                copy.publish_edges(first, node.edges().len());
            }
        }
        SharedTree {
            board: tree.board.clone(),
            keys: tree.keys.clone(),
            root: tree.root_index(),
            nodes,
            edges,
            pending: Mutex::new(HashMap::new()),
            waiting: AtomicUsize::new(0),
            reserved: AtomicUsize::new(0),
            full: AtomicBool::new(tree.full),
            gate: RwLock::new(()),
            counts: Counts::default(),
        }
    }

    pub fn root(&self) -> &Node {
        self.nodes.get(self.root)
    }

    pub fn root_index(&self) -> usize {
        self.root
    }

    pub fn is_full(&self) -> bool {
        self.full.load(Ordering::Relaxed)
    }

    //the bytes the tree's nodes and edges take
    pub fn allocated_bytes(&self) -> usize {
        self.nodes.len() * NODE_BYTES + self.edges.len() * EDGE_BYTES
    }

    //network leaves gathered and not backed up yet
    pub fn pending(&self) -> usize {
        self.waiting.load(Ordering::Acquire)
    }

    //a share of the gate, held while a call works
    pub fn hold(&self) -> Hold<'_> {
        self.gate.read().expect("the gate")
    }

    //the tree to itself: `work` runs once every call has left it and none enters. `hold` is the caller's share, given
    //up first and taken again after (nothing may wait for the gate while holding a share and another thread waits for it)
    fn exclusive<'a, R>(&'a self, hold: &mut Option<Hold<'a>>, work: impl FnOnce() -> R) -> R {
        drop(hold.take());
        let writer = self.gate.write().expect("the gate");
        self.counts.exclusive.fetch_add(1, Ordering::Relaxed);
        let result = work();
        drop(writer);
        *hold = Some(self.hold());
        result
    }

    //the tree to itself for a caller that holds no share
    pub fn quiet<R>(&self, work: impl FnOnce() -> R) -> R {
        let _writer = self.gate.write().expect("the gate");
        work()
    }

    pub fn chess_move(&self, edge: usize) -> Move {
        self.edges.chess_move(edge)
    }

    pub fn node(&self, id: usize) -> &Node {
        self.nodes.get(id)
    }

    //the edge's node, None until a simulation has gone through it
    pub fn child_index(&self, edge: usize) -> Option<usize> {
        let child = self.edges.child(edge).load(Ordering::Acquire);
        (child != NO_NODE).then_some(child as usize)
    }

    pub fn prior(&self, edge: usize) -> f64 {
        f64::from(self.edges.prior(edge))
    }

    //the edge's node, None until a simulation has gone through it
    pub fn child(&self, edge: usize) -> Option<&Node> {
        let child = self.edges.child(edge).load(Ordering::Acquire);
        (child != NO_NODE).then(|| self.nodes.get(child as usize))
    }

    pub fn visits(&self, edge: usize) -> u32 {
        self.child(edge).map_or(0, |child| child.visits())
    }

    //a move whose child is a proven loss for the side making it
    fn proven_lost(&self, edge: usize) -> bool {
        self.child(edge).is_some_and(|child| child.proves_a_loss())
    }

    //the visits of the root's two most visited moves, a proven loss left out (Tree::two_most_visits)
    pub fn two_most_visits(&self) -> (u32, u32) {
        let (mut most, mut second) = (0, 0);
        for visits in self.root().edges().filter(|&edge| !self.proven_lost(edge)).map(|edge| self.visits(edge)) {
            if visits > most {
                (most, second) = (visits, most);
            } else if visits > second {
                second = visits;
            }
        }
        (most, second)
    }

    pub fn moves_not_lost(&self) -> usize {
        self.root().edges().filter(|&edge| !self.proven_lost(edge)).count()
    }

    //Tree::most_visited_edge
    pub fn most_visited_edge(&self, node: usize) -> usize {
        let preference = |edge: usize| {
            let visits = self.visits(edge);
            let mean_score = self.child(edge).filter(|_| visits > 0).map_or(0.0, |child| child.mean_score(visits));
            let proven = match self.child(edge).and_then(|child| child.proof()) {
                Some(proof) if proof.outcome != Outcome::Draw => proof_rank(&proof),
                _ => (Outcome::Draw, 0),
            };
            (proven, visits, mean_score, f64::from(self.edges.prior(edge)))
        };
        let mut edges = self.nodes.get(node).edges();
        let mut best = edges.next().expect("a searched node has children");
        for edge in edges {
            if preference(edge) > preference(best) {
                best = edge;
            }
        }
        best
    }

    //Tree::principal_variation, as edges
    pub fn principal_variation(&self) -> Vec<usize> {
        let (mut line, mut node) = (Vec::new(), self.root);
        while self.nodes.get(node).has_edges() {
            let edge = self.most_visited_edge(node);
            if !line.is_empty() && self.visits(edge) == 0 {
                break;
            }
            line.push(edge);
            match self.edges.child(edge).load(Ordering::Acquire) {
                NO_NODE => break,
                child => node = child as usize,
            }
        }
        line
    }

    //the root's edges, the most visited first and ties in their order
    pub fn edges_by_visits(&self) -> Vec<usize> {
        let mut edges: Vec<usize> = self.root().edges().collect();
        edges.sort_by(|&a, &b| self.visits(b).cmp(&self.visits(a)));
        edges
    }

    //a node's result from its moves' proofs (Tree::proof_from_moves)
    fn proof_from_moves(&self, node: usize) -> Option<Proof> {
        let edges = self.nodes.get(node).edges();
        if edges.is_empty() {
            return None;
        }
        let mut best: Option<Proof> = None;
        let mut all_proven = true;
        for edge in edges {
            match self.child(edge).and_then(|child| child.proof()) {
                Some(proof) if best.is_none_or(|best| proof_rank(&proof) > proof_rank(&best)) => best = Some(proof),
                Some(_) => {}
                None => all_proven = false,
            }
        }
        let best = best?;
        let outcome = match best.outcome {
            Outcome::Win => Outcome::Loss,
            _ if !all_proven => return None,
            Outcome::Draw => Outcome::Draw,
            Outcome::Loss => Outcome::Win,
        };
        Some(Proof { outcome, value: 1.0 - best.value, plies: best.plies + 1 })
    }

    pub fn root_decided(&self) -> bool {
        self.proof_from_moves(self.root).is_some()
    }

    //Tree::select_edge: the exploration term counts the simulations in flight below a child as visits. Returns the edge
    fn select_edge(&self, id: usize, settings: &SearchSettings) -> usize {
        let parent = self.nodes.get(id);
        let (parent_visits, _) = parent.counts();
        let c_puct = settings.c_puct + settings.c_puct_factor * ((f64::from(parent_visits) + settings.c_puct_base) / settings.c_puct_base).ln();
        let children_visits = parent_visits.saturating_sub(1).max(1);
        let edges = parent.edges();
        let first_play_urgency = (1.0 - parent.mean_score(parent_visits)) - settings.fpu_reduction * parent.visited_policy_sqrt();
        let sqrt_children_visits = f64::from(children_visits).sqrt();

        let (mut best, mut best_u) = (0, f64::NEG_INFINITY);
        for (index, (prior, child)) in self.edges.priors(edges.clone()).iter().zip(self.edges.children(edges.clone())).enumerate() {
            let prior = f64::from(f32::from_bits(prior.load(Ordering::Relaxed)));
            let u = match child.load(Ordering::Acquire) {
                //sqrt_children_visits / 1.0 is sqrt_children_visits: no division
                NO_NODE => first_play_urgency + (c_puct * prior * sqrt_children_visits),
                child => {
                    let child = self.nodes.get(child as usize);
                    if child.proves_a_loss() {
                        continue;
                    }
                    let (visits, in_flight) = child.counts();
                    let q = if visits > 0 { child.mean_score(visits) } else { first_play_urgency };
                    let started = visits + in_flight;
                    q + (c_puct * prior * (sqrt_children_visits / f64::from(1 + started)))
                }
            };
            if u > best_u {
                (best, best_u) = (index, u);
            }
        }
        edges.start + best
    }

    //the edge's node, made if the edge has none yet; board is the node's position. Of two threads that make one, one
    //node is kept and the other's stays unused
    fn child_node(&self, edge: usize, board: &Chess) -> usize {
        let slot = self.edges.child(edge);
        let existing = slot.load(Ordering::Acquire);
        if existing != NO_NODE {
            return existing as usize;
        }
        let made = self.nodes.push(position_key(board));
        jitter();
        match slot.compare_exchange(NO_NODE, made as u32, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => made,
            Err(other) => {
                self.counts.orphans.fetch_add(1, Ordering::Relaxed);
                other as usize
            }
        }
    }

    //counts the simulation in flight on the node, and says what it has reached
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

    //one simulation down to a leaf, every node on the way counted in flight; the path ends on the node it reached
    fn descend(&self, settings: &SearchSettings, board: &mut Chess, keys: &mut Vec<u64>, path: &mut Vec<usize>) -> Arrival {
        let mut node = self.root;
        path.push(node);
        loop {
            match self.arrive(node) {
                Arrival::Inner => {
                    let edge = self.select_edge(node, settings);
                    board.play_unchecked(self.edges.chess_move(edge));
                    node = self.child_node(edge, board);
                    path.push(node);
                    keys.push(self.nodes.get(node).key());
                }
                other => return other,
            }
        }
    }

    //takes the simulations of a path off the nodes' in flight counts
    fn take_off(&self, path: &[usize]) {
        for &node in path {
            self.nodes.get(node).take_off(1);
        }
    }

    //mcts.expand: the moves as edges, without nodes. The node is the caller's: nothing else reads or writes its edges
    //until they are published
    fn expand(&self, id: usize, moves: &MoveList, priors: &[f32]) {
        let first = self.edges.alloc(moves.len());
        for (index, (&chess_move, &prior)) in moves.iter().zip(priors).enumerate() {
            self.edges.set(first + index, chess_move, prior);
        }
        jitter();
        self.nodes.get(id).publish_edges(first, moves.len());
    }

    //mcts.backup along the path, for a simulation that was in flight there. A node counts the visit after its value
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
            if node.visit(1) == 0 && depth > 0 {
                self.refresh_visited_policy(path[depth - 1]);
            }
            score = 1.0 - score;
        }
    }

    //the parent's visited policy again, in edge order and python's summation. Threads that refresh one node at once
    //repeat until none began another during their scan: the last store follows a scan that saw every visit
    fn refresh_visited_policy(&self, id: usize) {
        let node = self.nodes.get(id);
        let mut seen = node.begin_refresh();
        loop {
            let sum = python_sum(node.edges().filter(|&edge| self.visits(edge) > 0).map(|edge| f64::from(self.edges.prior(edge))));
            jitter();
            node.set_visited_policy_sqrt(sum.sqrt());
            let now = node.refreshes();
            if now == seen {
                break;
            }
            seen = now;
        }
    }

    //proves the path's last node: its visits so far count the proven score instead of theirs, in its ancestors too (up
    //to a proven one). A node already proven keeps its proof. Needs the tree to itself
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

    //whether propagate_proofs would prove the leaf's parent
    fn proves_parent(&self, path: &[usize]) -> bool {
        path.len() >= 3 && self.proof_from_moves(path[path.len() - 2]).is_some()
    }

    //room for one more descent: the root's own visits and the leaves waiting are within the simulations
    fn has_room(&self, visit_limit: u32) -> bool {
        self.root().visits() as usize + self.pending() < visit_limit as usize
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
            self.reserved.fetch_sub(if waiting { NODE_BYTES } else { DESCENT_BYTES }, Ordering::AcqRel);
        }
    }

    //the network leaves of the round, in the order they were gathered in, for finish_leaf
    pub fn queue(&self, round: u64, leaves: Vec<PendingLeaf>) {
        self.pending.lock().expect("the pending leaves").entry(round).or_default().extend(leaves);
    }

    pub fn take_round(&self, round: u64) -> Vec<PendingLeaf> {
        self.pending.lock().expect("the pending leaves").remove(&round).unwrap_or_default()
    }

    //Tree::gather for one thread of several: descents until the round has its leaves or its simulations, the leaves'
    //boards added to boards, a leaf that needs no network finished at once. Returns the leaves, which the caller queues
    //(`queue`) under the round once it has what it needs of them. A collision keeps its simulation in flight until the
    //call ends. Several threads may overrun the root's simulations by one each: a check and a leaf are not one step
    pub fn gather(
        &self,
        round: &Round,
        settings: &SearchSettings,
        tables: Option<&Tables>,
        cache: Option<&SharedCache>,
        boards: &mut Vec<u8>,
    ) -> Result<Vec<PendingLeaf>, SyzygyError> {
        let mut hold = Some(self.hold());
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
        hold: &mut Option<Hold<'a>>,
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
        let mut priors = Vec::new();
        while call.leaves.len() < round.leaves as usize && call.simulations < round.simulations && self.has_room(round.visit_limit) {
            //the root's own evaluation always fits, so that a full search still has moves to report
            let reserved = self.root().has_edges();
            if reserved && !self.reserve(round.memory_limit) {
                self.full.store(true, Ordering::Relaxed);
                break;
            }
            let mut board = self.board.clone();
            keys.truncate(root_keys);
            path.clear();
            let arrival = self.descend(settings, &mut board, &mut keys, &mut path);
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
                    self.backup(&path, 1.0 - proof.value);
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

            let mut legal_moves = board.legal_moves();
            let node_tables = tables.filter(|_| node != self.root && (!root_covered || board.halfmoves() == 0));
            //the root is searched whatever the rules say of it: Searches.start refuses only a mate or stalemate
            let game_over = if node == self.root {
                None
            } else {
                match game_result(&board, &keys, &legal_moves, settings.draw, settings.repetitions_for_a_draw, node_tables) {
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
            let value = game_over.or_else(|| path_keys.contains(&leaf_key).then_some(settings.draw));
            call.simulations += 1;
            if let Some(value) = value {
                give_back(false);
                match game_over {
                    //the result is for the side to move, the proof for the side that moved here. A win or loss with moves
                    //left is the tables', after any mate
                    Some(result) if settings.certainty && node != self.root => {
                        let (outcome, plies) = if result == 0.0 {
                            (Outcome::Win, if legal_moves.is_empty() { 0 } else { TABLEBASE_PLIES })
                        } else if result == 1.0 {
                            (Outcome::Loss, TABLEBASE_PLIES)
                        } else {
                            (Outcome::Draw, 0)
                        };
                        self.exclusive(hold, || {
                            self.prove(&path, Proof { outcome, value: 1.0 - result, plies });
                            self.backup(&path, value);
                            self.propagate_proofs(&path);
                        });
                    }
                    _ => {
                        self.backup(&path, value);
                        if settings.certainty && self.proves_parent(&path) {
                            self.exclusive(hold, || self.propagate_proofs(&path));
                        }
                    }
                }
                continue;
            }
            sort_moves_in_python_order(&board, &mut legal_moves);
            let moves = legal_moves;
            //a hash collision would give another number of moves
            let cached = cache.and_then(|cache| cache.get(leaf_key, &mut priors)).filter(|_| priors.len() == moves.len());
            match cached {
                Some(value) => {
                    give_back(false);
                    self.expand(node, &moves, &priors);
                    self.backup(&path, f64::from(value));
                }
                None => {
                    encode_board(&board, boards);
                    self.waiting.fetch_add(1, Ordering::AcqRel);
                    call.leaves.push(PendingLeaf { path: path.clone(), key: leaf_key, moves, turn: board.turn() });
                    give_back(true);
                }
            }
        }
        Ok(())
    }

    //mcts.finish for one leaf of a round; answer is the move logits, then the expected score. The cache keeps it,
    //sharpened. The caller holds a share of the gate (`hold`) for the leaves of its round
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
        self.backup(&leaf.path, f64::from(value));
        if let Some(cache) = cache {
            cache.insert(leaf.key, logits, value);
        }
        if memory_limit != usize::MAX {
            self.reserved.fetch_sub(MOST_MOVES * EDGE_BYTES, Ordering::AcqRel);
        }
        self.waiting.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
impl SharedTree {
    //a Tree of the same numbers with its edges numbered as Tree's vectors have them (those of a one thread search, in the
    //order the nodes were expanded in), to compare a search with Tree's number for number
    pub fn to_tree(&self) -> Tree {
        let count = self.nodes.len();
        let mut expanded: Vec<(usize, usize)> =
            (0..count).filter_map(|id| Some(self.nodes.get(id).edges()).filter(|edges| !edges.is_empty()).map(|edges| (edges.start, id))).collect();
        expanded.sort_unstable();
        let mut first_edge = vec![0; count];
        let (mut moves, mut priors, mut children) = (Vec::new(), Vec::new(), Vec::new());
        for &(_, id) in &expanded {
            first_edge[id] = moves.len();
            for edge in self.nodes.get(id).edges() {
                moves.push(self.edges.chess_move(edge));
                priors.push(self.edges.prior(edge));
                children.push(self.edges.child(edge).load(Ordering::Relaxed));
            }
        }
        let nodes = (0..count)
            .map(|id| {
                let node = self.nodes.get(id);
                let (visit_count, in_flight) = node.counts();
                let edges = node.edges();
                super::Node {
                    visit_count,
                    in_flight,
                    value_sum: node.value_sum(),
                    visited_policy_sqrt: node.visited_policy_sqrt(),
                    edges_at: if edges.is_empty() { 0 } else { (first_edge[id] as u64) << 8 | edges.len() as u64 },
                    proof: node.proof(),
                }
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
            children,
            pending: Vec::new(),
            logits: Vec::new(),
            full: self.is_full(),
        }
    }
}

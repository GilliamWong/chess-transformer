//mcts.py's search with the same order of operations and arithmetic, so that it visits the same nodes. A round may also
//gather many leaves for one network call (gather), as lc0 does; one leaf a round is mcts.py's search exactly
use crate::rules::{MoveIds, Tables, encode_board, game_result, position_key, sort_moves_in_python_order};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Gamma};
use shakmaty::{Chess, Color, Move, MoveList, Position};
use shakmaty_syzygy::SyzygyError;
use std::collections::{HashMap, VecDeque};
use std::hash::{BuildHasherDefault, Hasher};
use std::mem::MaybeUninit;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

//mcts.py's constants
pub struct SearchSettings {
    pub c_puct: f64,
    pub c_puct_factor: f64,
    pub c_puct_base: f64,
    pub fpu_reduction: f64,
    pub draw: f64,
    pub repetitions_for_a_draw: usize,
    pub value_sharpness: f64, //the network's value through sharpen; 1 is mcts.py's search
    //certainty propagation (lc0's sticky endgames): a game's end in the tree (mate, a draw by the rules, the tables) is
    //a proven result, and so is a node with a winning move or with every move's result proven; a two-fold repetition
    //stays an unproven draw. A proven node's visits all count its result, and descents stop there. mcts.py has none
    pub certainty: bool,
}

//a result the search has proven, for the side that played into the node (as the node's value_sum)
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Outcome {
    Loss,
    Draw,
    Win,
}

#[derive(Clone, Copy, Debug)]
pub struct Proof {
    pub outcome: Outcome,
    pub value: f64, //the node's score under the proof, a draw's as the tree values draws
    pub plies: u32, //to the game's end: a quicker win or a slower loss is the better one
}

//a tablebase win or loss counts as this many plies from its position, so that a mate comes before it
const TABLEBASE_PLIES: u32 = 1000;

//how good a move's proven result is for the side choosing it: a win, the quicker the better, then a draw, then a
//loss, the slower the better
fn proof_rank(proof: &Proof) -> (Outcome, i64) {
    let plies = i64::from(proof.plies);
    (proof.outcome, match proof.outcome {
        Outcome::Win => -plies,
        Outcome::Draw => 0,
        Outcome::Loss => plies,
    })
}

//mcts.Node's numbers, in one vector per tree with the root first. An expanded node's moves (its edges) are contiguous
//in the tree's edge vectors, and a move gets a node only once a simulation goes through it, as lc0's edges do: a node
//for each of a position's ~37 moves took 1.8 KB a simulation, most of them never visited
#[derive(Clone, Copy)]
pub struct Node {
    pub visit_count: u32,
    //simulations through this node waiting for the network, and while a round is gathered its collisions
    in_flight: u32,
    pub value_sum: f64,
    visited_policy_sqrt: f64, //sqrt of the sum of visited children's priors, refreshed when that set or the priors change
    //the node's first edge above the low 8 bits, its number of edges in them (a position has at most 218 moves), 0
    //until the node is expanded: edges outnumber nodes ~37 to 1, and u32 edge indices would end at ~115M nodes
    edges_at: u64,
    pub proof: Option<Proof>, //only with SearchSettings' certainty, and never at the root
}

//an edge without a node
const NO_NODE: u32 = u32::MAX;

impl Node {
    fn new() -> Node {
        Node { visit_count: 0, in_flight: 0, value_sum: 0.0, visited_policy_sqrt: 0.0, edges_at: 0, proof: None }
    }

    //the node's moves, as indices into the tree's edges
    pub fn edges(&self) -> Range<usize> {
        let first = (self.edges_at >> 8) as usize;
        first..first + (self.edges_at & 255) as usize
    }

    pub fn has_edges(&self) -> bool {
        self.edges_at & 255 > 0
    }

    pub fn mean_score(&self) -> f64 {
        self.value_sum / f64::from(self.visit_count)
    }
}

//python's sum() of floats (3.12+) is neumaier's compensated sum, which differs from a plain loop in the last bits
fn python_sum(values: impl Iterator<Item = f64>) -> f64 {
    let mut values = values;
    let Some(mut total) = values.next() else { return 0.0 };
    let mut compensation = 0.0;
    for value in values {
        let sum = total + value;
        compensation += if total.abs() >= value.abs() { (total - sum) + value } else { (value - sum) + total };
        total = sum;
    }
    if compensation != 0.0 && compensation.is_finite() { total + compensation } else { total }
}

//mcts.evaluate_many's softmax over the legal moves' logits, in place
pub fn softmax(logits: &mut [f32]) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0;
    for logit in logits.iter_mut() {
        *logit = (*logit - max).exp();
        total += *logit;
    }
    for weight in logits.iter_mut() {
        *weight /= total;
    }
}

//the value's logit times sharpness, E' = 1 / (1 + ((1 - E) / E)^sharpness); 1 leaves it exactly as it was
pub fn sharpen(value: f32, sharpness: f64) -> f32 {
    if sharpness == 1.0 {
        return value;
    }
    let value = f64::from(value);
    (1.0 / (1.0 + ((1.0 - value) / value).powf(sharpness))) as f32
}

//a simulation waiting for the network: its nodes from the root down, and its leaf's position and legal moves
struct PendingLeaf {
    path: Vec<usize>,
    key: u64,
    moves: MoveList,
    turn: Color,
}

//the network's answer for a position: its legal moves' priors in python-chess's order, and its expected score
struct Answer<'a> {
    priors: &'a [f32],
    value: f32,
}

//a hasher for zobrist keys, which are uniform already: one multiply spreads them over the table, where std's SipHash
//would hash them again
#[derive(Default)]
struct KeyHasher(u64);

impl Hasher for KeyHasher {
    fn write(&mut self, _: &[u8]) {
        unreachable!("only u64 keys")
    }

    fn write_u64(&mut self, key: u64) {
        self.0 = key.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

type KeyMap<V> = HashMap<u64, V, BuildHasherDefault<KeyHasher>>;

//the priors a cached position is given room for, above the ~37 legal moves of a middlegame's
const PRIORS_A_POSITION: usize = 40;

//a generation of the cache: by position, its priors' place in one vector and its value
#[derive(Default)]
struct Generation {
    places: KeyMap<(usize, u16, f32)>,
    priors: Vec<f32>,
}

impl Generation {
    fn get(&self, key: u64) -> Option<Answer<'_>> {
        let &(start, count, value) = self.places.get(&key)?;
        Some(Answer { priors: &self.priors[start..start + usize::from(count)], value })
    }

    fn clear(&mut self) {
        self.places.clear();
        self.priors.clear();
    }
}

//the network's answers by position (lc0's nn cache): a position reached again by another order of moves (a third of a
//d512 search's leaves at 1 s a move) needs no network call. The network sees no history, so a position's answer is the
//same however it was reached. In two generations: when the newer is full the older is cleared and becomes the newer,
//keeping the latest. A generation keeps its priors in one vector, so that turning one over frees nothing
pub struct Cache {
    newer: Generation,
    older: Generation,
    generation: usize, //positions the newer generation holds before it becomes the older
}

impl Cache {
    pub fn new(positions: usize) -> Cache {
        Cache { newer: Generation::default(), older: Generation::default(), generation: positions.div_ceil(2) }
    }

    fn get(&self, key: u64) -> Option<Answer<'_>> {
        self.newer.get(key).or_else(|| self.older.get(key))
    }

    fn insert(&mut self, key: u64, priors: &[f32], value: f32) {
        if self.newer.places.len() >= self.generation {
            std::mem::swap(&mut self.older, &mut self.newer);
            self.newer.clear();
        }
        //a generation makes its room at its first answer, so that filling it never regrows its vectors (a regrowth
        //copies all they hold)
        if self.newer.places.capacity() == 0 {
            self.newer.places.reserve(self.generation);
            self.newer.priors.reserve(self.generation * PRIORS_A_POSITION);
        }
        self.newer.places.insert(key, (self.newer.priors.len(), priors.len() as u16, value));
        self.newer.priors.extend_from_slice(priors);
    }

    pub fn clear(&mut self) {
        self.newer.clear();
        self.older.clear();
    }
}

//how many leaves a round may gather for the network (lc0's minibatch), how many simulations it may do in all (those
//finished at once too: lc0's out of order evaluations), and how many of its descents may end on a leaf already waiting
//(collisions). The simulations waiting for the network count as visits in the exploration term, so a round spreads out
pub struct Round {
    pub id: u64, //its leaves wait under it until finish_leaf takes them: rounds may come back in any order
    pub leaves: u32,
    pub simulations: u32,
    pub collisions: u32,
    pub visit_limit: u32,    //the search's simulations, those already done and waiting included
    pub memory_limit: usize, //bytes the tree's vectors may take, a growth's copy included; usize::MAX for none
}

//the most moves a position may have room for (MoveList's capacity): a waiting leaf's expansion needs that many edges
const MOST_MOVES: usize = 256;

//whether the nodes' u32 indices (the edges' children) have room for one more, below NO_NODE
fn node_fits(nodes: usize) -> bool {
    nodes < NO_NODE as usize
}

//the capacity a vector of `capacity` elements of `bytes` each grows to for `needed`: doubling, as std's vectors grow,
//but no further than a new allocation of `spare` bytes, made while the old one is still held. None if `needed` does
//not fit
fn grown_capacity(capacity: usize, needed: usize, bytes: usize, spare: usize) -> Option<usize> {
    if needed <= capacity {
        return Some(capacity);
    }
    let capacity = needed.max(2 * capacity).min(spare / bytes);
    (capacity >= needed).then_some(capacity)
}

fn reserve_to<T>(vector: &mut Vec<T>, capacity: usize) {
    if capacity > vector.capacity() {
        vector.reserve_exact(capacity - vector.len());
    }
}

//a tree's node and edge vectors, as compact returns the old ones to free
pub type Vectors = (Vec<Node>, Vec<u64>, Vec<Move>, Vec<f32>, Vec<u32>);

//nodes a walk or copy that may be stopped (an abort flag) goes through between looks at the flag
const ABORT_CHECKS: usize = 4096;

//fewer items than this (a level's nodes, the order's) are copied on one thread: more would not pay for the threads. The
//tests' small trees take several
const PARALLEL_ITEMS: usize = if cfg!(test) { 64 } else { 16384 };

//parts of the tree a thread of stale_candidates gets on the mean, so that a big part at the end leaves none idle long
const PARTS_A_THREAD: usize = 16;

//a part of the tree for one of stale_candidates' threads: a subtree (its node, and the places of the moves down to it)
//or a candidate leaf
enum Part {
    Subtree(usize, Vec<u16>),
    Leaf(Vec<u16>),
}

pub struct Tree {
    pub board: Chess,   //the root's position
    pub keys: Vec<u64>, //the position_key of every position of the game, the root's last
    pub nodes: Vec<Node>,
    root: usize,         //the root's node: 0 but after move_root, until compact
    node_keys: Vec<u64>, //each node's position_key, made with the node
    //the edges: every expanded node's moves in python-chess's order with their priors, and each move's node once it
    //has one (NO_NODE before)
    moves: Vec<Move>,
    priors: Vec<f32>,        //as the network gives them, read as f64s
    noised_priors: Vec<f64>, //the root's once add_root_noise has mixed its noise in (no longer f32s); empty without
    children: Vec<u32>,
    //the simulations waiting for the network by round (Round's id), each round's in the order of its boards. A round
    //whose leaves are all back keeps its room for a later one
    pending: Vec<(u64, VecDeque<PendingLeaf>)>,
    logits: Vec<f32>,
    //no room for another simulation (make_room): the search ends as at its simulations, until a move is played
    pub full: bool,
}

impl Tree {
    pub fn new(board: Chess, keys: Vec<u64>) -> Tree {
        let root_key = *keys.last().expect("the root's key");
        Tree {
            board,
            keys,
            nodes: vec![Node::new()],
            root: 0,
            node_keys: vec![root_key],
            moves: Vec::new(),
            priors: Vec::new(),
            noised_priors: Vec::new(),
            children: Vec::new(),
            pending: Vec::new(),
            logits: Vec::new(),
            full: false,
        }
    }

    //room for one more simulation: its node, and the moves of every leaf that may be expanded before the next look
    //(those waiting for the network and this one), within the nodes' u32 indices and the memory limit. Under a limit
    //the vectors are grown here (grown_capacity), so that no push grows them past it
    fn make_room(&mut self, memory_limit: usize) -> bool {
        if !node_fits(self.nodes.len()) {
            return false;
        }
        if memory_limit == usize::MAX {
            return true;
        }
        let spare = memory_limit.saturating_sub(self.allocated_bytes());
        let node_bytes = size_of::<Node>() + size_of::<u64>();
        let Some(nodes) = grown_capacity(self.nodes.capacity().min(self.node_keys.capacity()), self.nodes.len() + 1, node_bytes, spare)
        else {
            return false;
        };
        reserve_to(&mut self.nodes, nodes);
        reserve_to(&mut self.node_keys, nodes);
        let spare = memory_limit.saturating_sub(self.allocated_bytes());
        let edge_bytes = size_of::<Move>() + size_of::<f32>() + size_of::<u32>();
        let capacity = self.moves.capacity().min(self.priors.capacity()).min(self.children.capacity());
        let Some(edges) = grown_capacity(capacity, self.moves.len() + (self.pending() + 1) * MOST_MOVES, edge_bytes, spare) else {
            return false;
        };
        reserve_to(&mut self.moves, edges);
        reserve_to(&mut self.priors, edges);
        reserve_to(&mut self.children, edges);
        true
    }

    //the bytes the tree's node and edge vectors have allocated
    pub fn allocated_bytes(&self) -> usize {
        self.nodes.capacity() * size_of::<Node>()
            + self.node_keys.capacity() * size_of::<u64>()
            + self.moves.capacity() * size_of::<Move>()
            + self.priors.capacity() * size_of::<f32>()
            + self.children.capacity() * size_of::<u32>()
    }

    pub fn root(&self) -> &Node {
        &self.nodes[self.root]
    }

    pub fn root_index(&self) -> usize {
        self.root
    }

    //an edge's prior, a root edge's with its noise
    fn prior(&self, edge: usize) -> f64 {
        let root = self.root().edges();
        if !self.noised_priors.is_empty() && root.contains(&edge) {
            self.noised_priors[edge - root.start]
        } else {
            f64::from(self.priors[edge])
        }
    }

    pub fn chess_move(&self, edge: usize) -> Move {
        self.moves[edge]
    }

    //the edge's node, None until a simulation has gone through it
    pub fn child(&self, edge: usize) -> Option<&Node> {
        let child = self.children[edge];
        (child != NO_NODE).then(|| &self.nodes[child as usize])
    }

    pub fn visits(&self, edge: usize) -> u32 {
        self.child(edge).map_or(0, |child| child.visit_count)
    }

    //simulations waiting for the network
    pub fn pending(&self) -> usize {
        self.pending.iter().map(|(_, leaves)| leaves.len()).sum()
    }

    //the leaves of the round waiting for the network, in the room of a round whose leaves are all back if it has none
    fn round_leaves(&mut self, round: u64) -> &mut VecDeque<PendingLeaf> {
        let waiting = self.pending.iter().position(|(id, leaves)| *id == round && !leaves.is_empty());
        let slot = match waiting.or_else(|| self.pending.iter().position(|(_, leaves)| leaves.is_empty())) {
            Some(slot) => slot,
            None => {
                self.pending.push((round, VecDeque::new()));
                self.pending.len() - 1
            }
        };
        self.pending[slot].0 = round;
        &mut self.pending[slot].1
    }

    //a move whose child is a proven loss for the side making it
    fn proven_lost(&self, edge: usize) -> bool {
        self.child(edge).and_then(|child| child.proof).is_some_and(|proof| proof.outcome == Outcome::Loss)
    }

    //the visits of the root's two most visited moves, 0 for a move it does not have. A proven loss is left out: the
    //move choice puts it last whatever its visits (most_visited_edge)
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

    //the root's moves not proven lost: with one left the move choice is known (most_visited_edge puts losses last)
    pub fn moves_not_lost(&self) -> usize {
        self.root().edges().filter(|&edge| !self.proven_lost(edge)).count()
    }

    //mcts.select_child, with the simulations under a child that wait for the network counted in its visits for the
    //exploration term (lc0's n started); an edge without a node is a child without visits. A proven loss (certainty)
    //is passed over while any other move is left: its result is known, and its visits would only drag the node's mean,
    //and with it the first play urgency of the moves not yet tried, down. Returns the edge
    fn select_edge(&self, node: usize, settings: &SearchSettings) -> usize {
        let parent = &self.nodes[node];
        let c_puct =
            settings.c_puct + settings.c_puct_factor * ((f64::from(parent.visit_count) + settings.c_puct_base) / settings.c_puct_base).ln();
        let children_visits = parent.visit_count.saturating_sub(1).max(1);
        let edges = parent.edges();
        let first_play_urgency = (1.0 - parent.mean_score()) - settings.fpu_reduction * parent.visited_policy_sqrt;
        let sqrt_children_visits = f64::from(children_visits).sqrt();

        let (mut best, mut best_u) = (0, f64::NEG_INFINITY);
        let noised = node == self.root && !self.noised_priors.is_empty();
        for (index, (&prior, &child)) in self.priors[edges.clone()].iter().zip(&self.children[edges.clone()]).enumerate() {
            let prior = if noised { self.noised_priors[index] } else { f64::from(prior) };
            let (q, started) = match child {
                NO_NODE => (first_play_urgency, 0),
                child => {
                    let child = &self.nodes[child as usize];
                    if child.proof.is_some_and(|proof| proof.outcome == Outcome::Loss) {
                        continue;
                    }
                    let q = if child.visit_count > 0 { child.mean_score() } else { first_play_urgency };
                    (q, child.visit_count + child.in_flight)
                }
            };
            let u = q + (c_puct * prior * (sqrt_children_visits / f64::from(1 + started)));
            if u > best_u {
                (best, best_u) = (index, u);
            }
        }
        edges.start + best
    }

    //the edge's node, made if the edge has none yet; board is the node's position
    fn child_node(&mut self, edge: usize, board: &Chess) -> usize {
        if self.children[edge] == NO_NODE {
            self.children[edge] = self.nodes.len() as u32;
            self.nodes.push(Node::new());
            self.node_keys.push(position_key(board));
        }
        self.children[edge] as usize
    }

    //mcts.descend and finish until the round has its leaves or its simulations, the leaves' boards added to boards: a
    //leaf that needs no network (the game's result, or an answer in the cache) is finished at once. A descent that ends
    //on a leaf already waiting (a collision) counts as waiting along its path, so the next descents turn elsewhere; the
    //collisions are taken off again at the end. Returns the leaves gathered
    pub fn gather(
        &mut self,
        round: &Round,
        settings: &SearchSettings,
        tables: Option<&Tables>,
        cache: Option<&Cache>,
        boards: &mut Vec<u8>,
    ) -> Result<u32, SyzygyError> {
        let (mut simulations, mut gathered) = (0, 0);
        let mut collisions: Vec<Vec<usize>> = Vec::new();
        let mut path = Vec::new();
        //the tables finish the nodes they cover below the root. Below a root they cover (the engine searches a size it
        //has no dtz tables for), only those right after a capture or pawn move, as lc0 probes: the network steers the
        //moves between, and the tables' result, which starts the fifty move count there, is exact
        let root_covered = tables.is_some_and(|tables| tables.covers(&self.board));
        while gathered < round.leaves
            && simulations < round.simulations
            && self.root().visit_count as usize + self.pending() < round.visit_limit as usize
        {
            //the root's own evaluation always fits, so that a full search still has moves to report
            if self.root().has_edges() && !self.make_room(round.memory_limit) {
                self.full = true;
                break;
            }
            let mut board = self.board.clone();
            let root_keys = self.keys.len();
            path.clear();
            path.push(self.root);
            let mut node = self.root;
            while self.nodes[node].has_edges() {
                let edge = self.select_edge(node, settings);
                board.play_unchecked(self.moves[edge]);
                //a descent never ends without a visit or a simulation waiting, so the node is needed
                node = self.child_node(edge, &board);
                path.push(node);
                self.keys.push(self.node_keys[node]);
                if self.nodes[node].proof.is_some() {
                    break;
                }
            }

            //a proven result is known without going further
            if let Some(proof) = self.nodes[node].proof {
                self.keys.truncate(root_keys);
                simulations += 1;
                self.backup(&path, 1.0 - proof.value, 0);
                continue;
            }

            if self.nodes[node].in_flight > 0 {
                self.keys.truncate(root_keys);
                if collisions.len() == round.collisions as usize {
                    break;
                }
                self.add_in_flight(&path);
                collisions.push(path.clone());
                continue;
            }

            let mut legal_moves = board.legal_moves();
            let node_tables = tables.filter(|_| node != self.root && (!root_covered || board.halfmoves() == 0));
            //the root is searched whatever the rules say of it: Searches.start refuses only a mate or stalemate
            let game_over = if node == self.root {
                None
            } else {
                game_result(&board, &self.keys, &legal_moves, settings.draw, settings.repetitions_for_a_draw, node_tables)?
            };
            //a repeat inside the search is a draw, but not a proven one: the game need not repeat (lc0 gives two-fold
            //draws no bounds either)
            let (&leaf_key, path_keys) = self.keys[root_keys - 1..].split_last().expect("the root's key");
            let value = game_over.or_else(|| path_keys.contains(&leaf_key).then_some(settings.draw));
            self.keys.truncate(root_keys);

            simulations += 1;
            if let Some(value) = value {
                //the root is never proven: the rules and the tables never score it
                if let Some(result) = game_over
                    && settings.certainty
                    && node != self.root
                {
                    //the result is for the side to move, the proof for the side that moved here. A win or loss with
                    //moves left is the tables', after any mate
                    let (outcome, plies) = if result == 0.0 {
                        (Outcome::Win, if legal_moves.is_empty() { 0 } else { TABLEBASE_PLIES })
                    } else if result == 1.0 {
                        (Outcome::Loss, TABLEBASE_PLIES)
                    } else {
                        (Outcome::Draw, 0)
                    };
                    self.prove(&path, Proof { outcome, value: 1.0 - result, plies });
                }
                self.backup(&path, value, 0);
                if settings.certainty {
                    self.propagate_proofs(&path);
                }
                continue;
            }
            sort_moves_in_python_order(&board, &mut legal_moves);
            let moves = legal_moves;
            //a hash collision would give another number of moves
            match cache.and_then(|cache| cache.get(leaf_key)).filter(|answer| answer.priors.len() == moves.len()) {
                Some(answer) => {
                    self.expand(node, &moves, answer.priors);
                    self.backup(&path, f64::from(answer.value), 0);
                }
                None => {
                    encode_board(&board, boards);
                    self.add_in_flight(&path);
                    self.round_leaves(round.id).push_back(PendingLeaf { path: path.clone(), key: leaf_key, moves, turn: board.turn() });
                    gathered += 1;
                }
            }
        }
        for path in collisions {
            for node in path {
                self.nodes[node].in_flight -= 1;
            }
        }
        Ok(gathered)
    }

    //mcts.finish for the round's oldest waiting leaf; answer is the move logits, then the expected score. The cache keeps
    //it, sharpened
    pub fn finish_leaf(
        &mut self,
        round: u64,
        answer: impl Fn(usize) -> f32,
        answer_width: usize,
        move_ids: &MoveIds,
        cache: Option<&mut Cache>,
        settings: &SearchSettings,
    ) {
        let leaf = self
            .pending
            .iter_mut()
            .find(|(id, leaves)| *id == round && !leaves.is_empty())
            .and_then(|(_, leaves)| leaves.pop_front())
            .expect("a waiting simulation of the round");
        self.logits.clear();
        self.logits.extend(leaf.moves.iter().map(|&chess_move| answer(move_ids.id(chess_move, leaf.turn))));
        softmax(&mut self.logits);
        let value = sharpen(answer(answer_width - 1), settings.value_sharpness);

        let node = *leaf.path.last().expect("the root at least");
        let priors = std::mem::take(&mut self.logits);
        self.expand(node, &leaf.moves, &priors);
        self.backup(&leaf.path, f64::from(value), 1);
        if let Some(cache) = cache {
            cache.insert(leaf.key, &priors, value);
        }
        self.logits = priors;
    }

    //mcts.expand: the moves as edges, without nodes
    fn expand(&mut self, node: usize, moves: &MoveList, priors: &[f32]) {
        let first_edge = self.moves.len();
        for (&chess_move, &prior) in moves.iter().zip(priors) {
            self.moves.push(chess_move);
            self.priors.push(prior);
        }
        self.children.resize(self.moves.len(), NO_NODE);
        self.nodes[node].edges_at = (first_edge as u64) << 8 | (self.moves.len() - first_edge) as u64;
    }

    fn add_in_flight(&mut self, path: &[usize]) {
        for &node in path {
            self.nodes[node].in_flight += 1;
        }
    }

    //mcts.backup along the path, for a simulation that had in_flight visits waiting there
    fn backup(&mut self, path: &[usize], value: f64, in_flight: u32) {
        let mut score = 1.0 - value;
        for (depth, &node) in path.iter().enumerate().rev() {
            //from a node proven while the simulation waited for the network, the proven score (lc0's backup does the same)
            if let Some(proof) = self.nodes[node].proof {
                score = proof.value;
            }
            self.nodes[node].visit_count += 1;
            self.nodes[node].in_flight -= in_flight;
            self.nodes[node].value_sum += score;
            if self.nodes[node].visit_count == 1 && depth > 0 {
                self.refresh_visited_policy(path[depth - 1]);
            }
            score = 1.0 - score;
        }
    }

    //proves the path's last node: its visits so far count the proven score instead of theirs, in its ancestors too
    //(lc0's AdjustForTerminal). A visit's score s goes to the node and 1 - s to the parent, so the change alternates
    fn prove(&mut self, path: &[usize], proof: Proof) {
        let (&node, ancestors) = path.split_last().expect("a node to prove");
        let change = f64::from(self.nodes[node].visit_count) * proof.value - self.nodes[node].value_sum;
        self.nodes[node].value_sum += change;
        self.nodes[node].proof = Some(proof);
        for (distance, &ancestor) in ancestors.iter().rev().enumerate() {
            self.nodes[ancestor].value_sum += if distance % 2 == 0 { -change } else { change };
        }
    }

    //after the path's last node was proven, its parent may be too, and so on up to the root's children
    fn propagate_proofs(&mut self, path: &[usize]) {
        for depth in (1..path.len().saturating_sub(1)).rev() {
            match self.proof_from_moves(path[depth]) {
                Some(proof) => self.prove(&path[..=depth], proof),
                None => break,
            }
        }
    }

    //an expanded node's result from its moves' proofs: it is lost for the side that moved into it if a move wins (the
    //quickest), and otherwise known once every move's result is, the best of them
    fn proof_from_moves(&self, node: usize) -> Option<Proof> {
        let edges = self.nodes[node].edges();
        if edges.is_empty() {
            return None;
        }
        let mut best: Option<Proof> = None;
        let mut all_proven = true;
        for edge in edges {
            match self.child(edge).and_then(|child| child.proof) {
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

    //the root's result is proven: a move wins, or every move's result is known, and searching on changes nothing
    pub fn root_decided(&self) -> bool {
        self.proof_from_moves(self.root).is_some()
    }

    //a kept tree under a new root (Searches' tree reuse; self-play keeps mcts.py's tree as it is): a leaf scored by a
    //rule that held only from the old root comes out of the tree, its visits taken off its ancestors, as lc0 drops a
    //two-fold draw whose first position is no longer in the search. That is a repeat inside the search whose first
    //position is now before the root, where the game needs a third, and a tables' result the new root's probing would
    //not look up. Leaves below a proven node stay: no descent reaches them
    pub fn drop_stale_leaves(&mut self, settings: &SearchSettings, tables: Option<&Tables>) -> Result<(), SyzygyError> {
        let root_covered = tables.is_some_and(|tables| tables.covers(&self.board));
        let mut stale = Vec::new();
        self.find_stale_leaves(&mut vec![self.root], &mut Vec::new(), settings, tables, root_covered, &mut stale)?;
        for path in stale {
            self.take_out(&path);
        }
        Ok(())
    }

    //the paths to the stale leaves under the path's last node, which the line's moves (its edges) reach from the root;
    //self.keys has the path's keys
    fn find_stale_leaves(
        &mut self,
        path: &mut Vec<usize>,
        line: &mut Vec<usize>,
        settings: &SearchSettings,
        tables: Option<&Tables>,
        root_covered: bool,
        stale: &mut Vec<Vec<usize>>,
    ) -> Result<(), SyzygyError> {
        let node = *path.last().expect("the root at least");
        let edges = self.nodes[node].edges();
        if edges.is_empty() {
            if node != self.root && self.nodes[node].visit_count > 0 && self.is_stale(path, line, settings, tables, root_covered)? {
                stale.push(path.clone());
            }
            return Ok(());
        }
        for edge in edges {
            let child = self.children[edge];
            if child == NO_NODE || self.nodes[child as usize].proof.is_some() {
                continue;
            }
            path.push(child as usize);
            line.push(edge);
            self.keys.push(self.node_keys[child as usize]);
            let found = self.find_stale_leaves(path, line, settings, tables, root_covered, stale);
            self.keys.pop();
            line.pop();
            path.pop();
            found?;
        }
        Ok(())
    }

    //whether the leaf at the end of path, which the line's moves (edges) reach from the root, is no longer what gather
    //scored it as; self.keys ends with the path's keys. Only a leaf's position is played out: one for every node took
    //half the walk's time
    fn is_stale(
        &self,
        path: &[usize],
        line: &[usize],
        settings: &SearchSettings,
        tables: Option<&Tables>,
        root_covered: bool,
    ) -> Result<bool, SyzygyError> {
        //as gather scores a leaf
        let mut board = self.board.clone();
        for &edge in line {
            board.play_unchecked(self.moves[edge]);
        }
        let legal_moves = board.legal_moves();
        let node_tables = tables.filter(|_| !root_covered || board.halfmoves() == 0);
        let over = game_result(&board, &self.keys, &legal_moves, settings.draw, settings.repetitions_for_a_draw, node_tables)?;
        let (&leaf_key, path_keys) = self.keys[self.keys.len() - path.len()..].split_last().expect("the leaf's key");
        Ok(over.is_none() && !path_keys.contains(&leaf_key))
    }

    //the leaves drop_stale_leaves may find stale once the root has moved further down, found before the moves are
    //known (Searches' advance, while the opponent thinks): every leaf below the root with visits, no moves and no
    //proof of its own, in the walk's order, each as the places of its line's moves among their nodes' moves, which hold
    //across compact. Leaves below a node proven now are kept too, as a later root's own proof is dropped. The walk
    //runs on `threads` threads, each taking the next of the tree's parts (split_for_walk), their candidates joined in
    //the parts' order. None once abort is set
    pub fn stale_candidates(&self, threads: usize, abort: &AtomicBool) -> Option<Vec<Vec<u16>>> {
        let share = (self.root().visit_count / (threads * PARTS_A_THREAD).max(1) as u32).max(1);
        let mut parts = Vec::new();
        self.split_for_walk(self.root, &mut Vec::new(), share, &mut parts);
        let next = AtomicUsize::new(0);
        //the parts a thread walked, with their candidates, or None once stopped
        let walk = || {
            let (mut walked, mut looked) = (Vec::new(), 0);
            loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some(part) = parts.get(index) else { return Some(walked) };
                let mut candidates = Vec::new();
                match part {
                    Part::Leaf(places) => candidates.push(places.clone()),
                    Part::Subtree(node, places) => {
                        if !self.find_candidates(*node, &mut places.clone(), &mut candidates, &mut looked, abort) {
                            return None;
                        }
                    }
                }
                walked.push((index, candidates));
            }
        };
        let walked: Vec<Option<_>> = std::thread::scope(|scope| {
            let workers: Vec<_> = (1..threads).map(|_| scope.spawn(walk)).collect();
            let mut walked = vec![walk()];
            walked.extend(workers.into_iter().map(|worker| worker.join().expect("a walk's thread")));
            walked
        });
        let mut walked: Vec<(usize, Vec<Vec<u16>>)> = walked.into_iter().collect::<Option<Vec<_>>>()?.into_iter().flatten().collect();
        walked.sort_unstable_by_key(|&(index, _)| index);
        Some(walked.into_iter().flat_map(|(_, candidates)| candidates).collect())
    }

    //the parts of the tree below node (its line's moves at `places`) for stale_candidates' threads, in the walk's
    //order: a subtree of at most `share` visits is one part, a bigger one is split into its moves'; a leaf met on the
    //way that is a candidate is a part of its own
    fn split_for_walk(&self, node: usize, places: &mut Vec<u16>, share: u32, parts: &mut Vec<Part>) {
        if node != self.root && self.nodes[node].visit_count <= share {
            parts.push(Part::Subtree(node, places.clone()));
            return;
        }
        let edges = self.nodes[node].edges();
        for edge in edges.clone() {
            let child = self.children[edge];
            if child == NO_NODE {
                continue;
            }
            places.push((edge - edges.start) as u16);
            let child = child as usize;
            if self.nodes[child].has_edges() {
                self.split_for_walk(child, places, share, parts);
            } else if self.nodes[child].visit_count > 0 && self.nodes[child].proof.is_none() {
                parts.push(Part::Leaf(places.clone()));
            }
            places.pop();
        }
    }

    fn find_candidates(
        &self,
        node: usize,
        places: &mut Vec<u16>,
        candidates: &mut Vec<Vec<u16>>,
        looked: &mut usize,
        abort: &AtomicBool,
    ) -> bool {
        *looked += 1;
        if looked.is_multiple_of(ABORT_CHECKS) && abort.load(Ordering::Relaxed) {
            return false;
        }
        let edges = self.nodes[node].edges();
        for edge in edges.clone() {
            let child = self.children[edge];
            if child == NO_NODE {
                continue;
            }
            places.push((edge - edges.start) as u16);
            let child = child as usize;
            if self.nodes[child].has_edges() {
                if !self.find_candidates(child, places, candidates, looked, abort) {
                    return false;
                }
            } else if self.nodes[child].visit_count > 0 && self.nodes[child].proof.is_none() {
                candidates.push(places.clone());
            }
            places.pop();
        }
        true
    }

    //drop_stale_leaves from the candidates found at the root the moves at `line`'s places moved it down from: of those
    //below the root, the leaves its walk would find stale, taken out in its order
    pub fn drop_stale_candidates(
        &mut self,
        line: &[u16],
        candidates: &[Vec<u16>],
        settings: &SearchSettings,
        tables: Option<&Tables>,
    ) -> Result<(), SyzygyError> {
        let root_covered = tables.is_some_and(|tables| tables.covers(&self.board));
        let root_keys = self.keys.len();
        let mut stale = Vec::new();
        'candidates: for places in candidates.iter().filter(|places| places.len() > line.len() && places.starts_with(line)) {
            let (mut path, mut edges) = (vec![self.root], Vec::new());
            for &place in &places[line.len()..] {
                let edge = self.nodes[*path.last().expect("the root at least")].edges().start + usize::from(place);
                let child = self.children[edge] as usize;
                //the walk passes over a proven node and what is below it
                if self.nodes[child].proof.is_some() {
                    continue 'candidates;
                }
                path.push(child);
                edges.push(edge);
            }
            let leaf = *path.last().expect("a leaf below the root");
            if !self.nodes[leaf].has_edges() && self.nodes[leaf].visit_count > 0 {
                self.keys.extend(path[1..].iter().map(|&node| self.node_keys[node]));
                let found = self.is_stale(&path, &edges, settings, tables, root_covered);
                self.keys.truncate(root_keys);
                if found? {
                    stale.push(path);
                }
            }
        }
        for path in stale {
            self.take_out(&path);
        }
        Ok(())
    }

    //takes a leaf's visits out of the tree: a visit's score s went to the leaf, 1 - s to its parent and so on up, and
    //the leaf is as if never visited
    fn take_out(&mut self, path: &[usize]) {
        let (&leaf, ancestors) = path.split_last().expect("a leaf below the root");
        let (visits, value_sum) = (self.nodes[leaf].visit_count, self.nodes[leaf].value_sum);
        for (distance, &ancestor) in ancestors.iter().rev().enumerate() {
            self.nodes[ancestor].visit_count -= visits;
            self.nodes[ancestor].value_sum -= if distance % 2 == 0 { f64::from(visits) - value_sum } else { value_sum };
        }
        self.nodes[leaf] = Node::new();
        self.refresh_visited_policy(*ancestors.last().expect("a leaf below the root"));
    }

    fn refresh_visited_policy(&mut self, node: usize) {
        //Keep Python's summation order and compensation, including after noise on a reused root.
        let edges = self.nodes[node].edges();
        self.nodes[node].visited_policy_sqrt = python_sum(edges.filter(|&edge| self.visits(edge) > 0).map(|edge| self.prior(edge))).sqrt();
    }

    //mcts.add_root_noise: dirichlet noise as gamma draws scaled to add up to 1, mixed in f64 as mcts.py mixes it
    pub fn add_root_noise(&mut self, random: &mut ChaCha8Rng, alpha: f64, share: f64) {
        let gamma = Gamma::new(alpha, 1.0).expect("a positive noise alpha");
        let edges = self.root().edges();
        let draws: Vec<f64> = edges.clone().map(|_| gamma.sample(random)).collect();
        let total: f64 = draws.iter().sum();
        self.noised_priors = edges.zip(draws).map(|(edge, draw)| (1.0 - share) * self.prior(edge) + share * (draw / total)).collect();
        self.refresh_visited_policy(self.root);
    }

    //mcts.most_visited_move: ties to the better mean score, then the prior, then the first. Returns the edge. A proven
    //win comes before every other move (the quickest first) and a proven loss after (the slowest first), as lc0 has it
    pub fn most_visited_edge(&self, node: usize) -> usize {
        let preference = |edge: usize| {
            let visits = self.visits(edge);
            let mean_score = if visits > 0 { self.nodes[self.children[edge] as usize].mean_score() } else { 0.0 };
            let proven = match self.child(edge).and_then(|child| child.proof) {
                Some(proof) if proof.outcome != Outcome::Draw => proof_rank(&proof),
                _ => (Outcome::Draw, 0),
            };
            (proven, visits, mean_score, self.prior(edge))
        };
        let mut edges = self.nodes[node].edges();
        let mut best = edges.next().expect("a searched node has children");
        for edge in edges {
            if preference(edge) > preference(best) {
                best = edge;
            }
        }
        best
    }

    //mcts.principal_variation, as edges
    pub fn principal_variation(&self) -> Vec<usize> {
        let (mut line, mut node) = (Vec::new(), self.root);
        while self.nodes[node].has_edges() {
            let edge = self.most_visited_edge(node);
            if !line.is_empty() && self.visits(edge) == 0 {
                break;
            }
            line.push(edge);
            //an edge without a node leads to a node without children
            match self.children[edge] {
                NO_NODE => break,
                child => node = child as usize,
            }
        }
        line
    }

    //the root's edges, the most visited first and ties in their order (a stable sort)
    pub fn edges_by_visits(&self) -> Vec<usize> {
        let mut edges: Vec<usize> = self.root().edges().collect();
        edges.sort_by(|&a, &b| self.visits(b).cmp(&self.visits(a)));
        edges
    }

    //plays the root's move and makes its node the root, with its subtree (tree reuse) or without
    pub fn play(&mut self, edge: usize, keep_subtree: bool) {
        assert_eq!(self.pending(), 0, "a move is played once the search's leaves are back");
        if keep_subtree && self.children[edge] != NO_NODE {
            self.move_root(&[edge]);
            self.compact(1, None);
        } else {
            self.board.play_unchecked(self.moves[edge]);
            self.keys.push(position_key(&self.board));
            (self.nodes, self.moves, self.priors, self.children) = (vec![Node::new()], Vec::new(), Vec::new(), Vec::new());
            self.node_keys = vec![*self.keys.last().expect("the root's key")];
            self.noised_priors.clear();
            self.full = false;
            self.root = 0;
        }
    }

    //plays a line of moves from the root (edges, each of the node the last one leads to) and makes the node it ends on
    //the root, with its subtree: the subtree is copied once, where play would copy it at every move
    pub fn play_line(&mut self, line: &[usize]) {
        self.move_root(line);
        let old = self.compact(1, None);
        //the old tree is freed on a thread of its own, off the move's clock (1.6 ms at 100k nodes); here if none starts
        let _ = std::thread::Builder::new().spawn(move || drop(old));
    }

    //plays a line of moves from the root (edges, each of the node the last one leads to) and makes the node it ends on
    //the root where it is, without a copy: the rest of the tree stays in the vectors until compact. Every edge of the
    //line has its node. The new root's proof is dropped, as a root is never proven (root_decided reads its moves'), and
    //so is the old root's noise
    pub fn move_root(&mut self, line: &[usize]) {
        assert_eq!(self.pending(), 0, "a move is played once the search's leaves are back");
        for &edge in line {
            assert_ne!(self.children[edge], NO_NODE, "a line through the tree's nodes");
            self.board.play_unchecked(self.moves[edge]);
            self.keys.push(position_key(&self.board));
            self.root = self.children[edge] as usize;
        }
        if !line.is_empty() {
            self.nodes[self.root].proof = None;
            self.noised_priors.clear();
            self.full = false;
        }
    }

    //the root's part of the tree copied out of the rest (left there by move_root) as a tree of its own, in breadth-first
    //order from the root. Its proofs still hold: a proof comes from the game's end by its rules or the tables, which a
    //new root leaves as it was (a two-fold repetition proves nothing). On one thread in one pass; on more, the order is
    //found a level at a time in chunks, then each stretch of it is copied into its own part of the new vectors, whose
    //pages that thread is the first to touch: the same tree. Returns the old tree's vectors to free, or None once abort
    //is set (the tree then left as it was)
    pub fn compact(&mut self, threads: usize, abort: Option<&AtomicBool>) -> Option<Vectors> {
        if threads <= 1 {
            return self.compact_on_one_thread(abort);
        }
        let tree: &Tree = self;
        //the kept nodes in breadth-first order (their numbers in the new tree), and of each its moves with a node and
        //its moves
        let (mut order, mut counts) = (vec![tree.root as u32], Vec::new());
        let mut level = 0..1;
        while !level.is_empty() {
            let chunks: Vec<&[u32]> = order[level.clone()].chunks(chunk_size(level.len(), threads)).collect();
            let found = each_in_parallel(&chunks, |chunk| tree.next_level(chunk, abort));
            let found: Vec<Level> = found.into_iter().collect::<Option<_>>()?;
            for (next, chunk_counts) in found {
                order.extend(next);
                counts.extend(chunk_counts);
            }
            level = level.end..order.len();
        }

        //the order in stretches, one a thread, each with its edges and where they and its children's numbers start
        let size = chunk_size(order.len(), threads);
        let positions: Vec<Range<usize>> = (0..order.len()).step_by(size).map(|start| start..(start + size).min(order.len())).collect();
        let (mut first_edge, mut first_child) = (0, 1);
        let starts: Vec<(usize, usize, usize)> = positions
            .iter()
            .map(|positions| {
                let edges = counts[positions.clone()].iter().map(|&(_, edges)| usize::from(edges)).sum();
                let start = (edges, first_edge, first_child);
                first_edge += edges;
                first_child += counts[positions.clone()].iter().map(|&(with_node, _)| usize::from(with_node)).sum::<usize>();
                start
            })
            .collect();

        //room for the whole tree, so the copy never regrows (that took twice the time); unwritten room takes no memory
        let (node_room, edge_room) = (tree.nodes.len(), tree.moves.len());
        let (mut nodes, mut node_keys) = (Vec::with_capacity(node_room), Vec::with_capacity(node_room));
        let (mut moves, mut priors, mut children) =
            (Vec::with_capacity(edge_room), Vec::with_capacity(edge_room), Vec::with_capacity(edge_room));
        {
            let mut nodes_left = &mut nodes.spare_capacity_mut()[..order.len()];
            let mut keys_left = &mut node_keys.spare_capacity_mut()[..order.len()];
            let mut moves_left = &mut moves.spare_capacity_mut()[..first_edge];
            let mut priors_left = &mut priors.spare_capacity_mut()[..first_edge];
            let mut children_left = &mut children.spare_capacity_mut()[..first_edge];
            let mut stretches = Vec::with_capacity(positions.len());
            for (positions, &(edges, first_edge, first_child)) in positions.iter().zip(&starts) {
                let (nodes, node_keys, moves, priors, children);
                (nodes, nodes_left) = std::mem::take(&mut nodes_left).split_at_mut(positions.len());
                (node_keys, keys_left) = std::mem::take(&mut keys_left).split_at_mut(positions.len());
                (moves, moves_left) = std::mem::take(&mut moves_left).split_at_mut(edges);
                (priors, priors_left) = std::mem::take(&mut priors_left).split_at_mut(edges);
                (children, children_left) = std::mem::take(&mut children_left).split_at_mut(edges);
                stretches.push(Stretch {
                    positions: positions.clone(),
                    first_edge,
                    first_child,
                    nodes,
                    node_keys,
                    moves,
                    priors,
                    children,
                });
            }
            let order = &order;
            let copied = match stretches.len() {
                1 => tree.copy_stretch(stretches.pop().expect("a stretch"), order, abort),
                _ => std::thread::scope(|scope| {
                    let workers: Vec<_> =
                        stretches.into_iter().map(|stretch| scope.spawn(move || tree.copy_stretch(stretch, order, abort))).collect();
                    //the scope joins any thread a failed copy leaves running
                    workers.into_iter().all(|worker| worker.join().expect("a thread of the copy"))
                }),
            };
            if !copied {
                return None;
            }
        }
        //SAFETY: the stretches split the first order.len() nodes and first_edge edges between them, and each wrote all
        //of its own
        unsafe {
            nodes.set_len(order.len());
            node_keys.set_len(order.len());
            moves.set_len(first_edge);
            priors.set_len(first_edge);
            children.set_len(first_edge);
        }
        self.root = 0;
        Some((
            std::mem::replace(&mut self.nodes, nodes),
            std::mem::replace(&mut self.node_keys, node_keys),
            std::mem::replace(&mut self.moves, moves),
            std::mem::replace(&mut self.priors, priors),
            std::mem::replace(&mut self.children, children),
        ))
    }

    //the nodes the moves of a chunk of a level lead to (that part of the next level), in order, and of each node in the
    //chunk its moves with a node and its moves; None once abort is set
    fn next_level(&self, chunk: &[u32], abort: Option<&AtomicBool>) -> Option<Level> {
        let (mut next, mut counts) = (Vec::new(), Vec::with_capacity(chunk.len()));
        for (index, &node) in chunk.iter().enumerate() {
            if index.is_multiple_of(ABORT_CHECKS) && abort.is_some_and(|abort| abort.load(Ordering::Relaxed)) {
                return None;
            }
            let (before, edges) = (next.len(), self.nodes[node as usize].edges());
            next.extend(self.children[edges.clone()].iter().filter(|&&child| child != NO_NODE));
            counts.push(((next.len() - before) as u8, edges.len() as u8));
        }
        Some((next, counts))
    }

    //compact on one thread, in one pass: the nodes in breadth-first order, each one's moves as it comes, and the moves
    //with a node getting theirs copied at the end
    fn compact_on_one_thread(&mut self, abort: Option<&AtomicBool>) -> Option<Vectors> {
        //room for the whole tree, so the copy never regrows (that took twice the time); unwritten room takes no memory
        let (node_room, edge_room) = (self.nodes.len(), self.moves.len());
        let mut nodes = Vec::with_capacity(node_room);
        nodes.push(Node { proof: None, ..self.nodes[self.root] });
        let mut node_keys = Vec::with_capacity(node_room);
        node_keys.push(self.node_keys[self.root]);
        let (mut moves, mut priors, mut children) =
            (Vec::with_capacity(edge_room), Vec::with_capacity(edge_room), Vec::with_capacity(edge_room));
        let mut index = 0;
        while index < nodes.len() {
            if index.is_multiple_of(ABORT_CHECKS) && abort.is_some_and(|abort| abort.load(Ordering::Relaxed)) {
                return None;
            }
            let edges = nodes[index].edges();
            let first_edge = moves.len();
            //a node without moves keeps the first edge Node::new gives it
            if !edges.is_empty() {
                nodes[index].edges_at = (first_edge as u64) << 8 | edges.len() as u64;
            }
            moves.extend_from_slice(&self.moves[edges.clone()]);
            priors.extend_from_slice(&self.priors[edges.clone()]);
            children.extend_from_slice(&self.children[edges]);
            for child in &mut children[first_edge..] {
                if *child != NO_NODE {
                    nodes.push(self.nodes[*child as usize]);
                    node_keys.push(self.node_keys[*child as usize]);
                    *child = (nodes.len() - 1) as u32;
                }
            }
            index += 1;
        }
        self.root = 0;
        Some((
            std::mem::replace(&mut self.nodes, nodes),
            std::mem::replace(&mut self.node_keys, node_keys),
            std::mem::replace(&mut self.moves, moves),
            std::mem::replace(&mut self.priors, priors),
            std::mem::replace(&mut self.children, children),
        ))
    }

    //compact's copy of a stretch of the breadth-first order: each node with its moves at their new edges, and the moves'
    //nodes numbered as the order has them. False once abort is set
    fn copy_stretch(&self, stretch: Stretch, order: &[u32], abort: Option<&AtomicBool>) -> bool {
        let (mut edge, mut child) = (stretch.first_edge, stretch.first_child);
        for (slot, position) in stretch.positions.clone().enumerate() {
            if slot.is_multiple_of(ABORT_CHECKS) && abort.is_some_and(|abort| abort.load(Ordering::Relaxed)) {
                return false;
            }
            let old = order[position] as usize;
            let mut node = self.nodes[old];
            if position == 0 {
                node.proof = None;
            }
            let edges = node.edges();
            //a node without moves keeps the first edge Node::new gives it
            if !edges.is_empty() {
                node.edges_at = (edge as u64) << 8 | edges.len() as u64;
            }
            stretch.nodes[slot].write(node);
            stretch.node_keys[slot].write(self.node_keys[old]);
            for old_edge in edges {
                let at = edge - stretch.first_edge;
                stretch.moves[at].write(self.moves[old_edge]);
                stretch.priors[at].write(self.priors[old_edge]);
                stretch.children[at].write(match self.children[old_edge] {
                    NO_NODE => NO_NODE,
                    _ => {
                        child += 1;
                        (child - 1) as u32
                    }
                });
                edge += 1;
            }
        }
        true
    }
}

//a part of a level of compact's breadth-first order: the nodes its chunk's moves lead to, and of each node of the chunk
//its moves with a node and its moves
type Level = (Vec<u32>, Vec<(u8, u8)>);

//a stretch of the breadth-first order for one of compact's threads: its positions, where its edges and the numbers of
//its moves' nodes start, and its parts of the new vectors
struct Stretch<'a> {
    positions: Range<usize>,
    first_edge: usize,
    first_child: usize,
    nodes: &'a mut [MaybeUninit<Node>],
    node_keys: &'a mut [MaybeUninit<u64>],
    moves: &'a mut [MaybeUninit<Move>],
    priors: &'a mut [MaybeUninit<f32>],
    children: &'a mut [MaybeUninit<u32>],
}

//the items in a chunk for compact's threads: a `threads`th of them each, or all in one below PARALLEL_ITEMS
fn chunk_size(items: usize, threads: usize) -> usize {
    if items < PARALLEL_ITEMS { items.max(1) } else { items.div_ceil(threads.max(1)) }
}

//f of each item, in order, each on a scoped thread of its own (the first here); items is not empty
fn each_in_parallel<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    std::thread::scope(|scope| {
        let workers: Vec<_> = items[1..].iter().map(|item| scope.spawn(|| f(item))).collect();
        let mut done = vec![f(&items[0])];
        done.extend(workers.into_iter().map(|worker| worker.join().expect("a thread of the copy")));
        done
    })
}

#[cfg(test)]
mod tests;

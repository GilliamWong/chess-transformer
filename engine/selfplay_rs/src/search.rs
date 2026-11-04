//mcts.py's search operation for operation: a round of one leaf is exactly mcts.py's
use crate::rules::{MoveIds, PackedMove, Tables, encode_board, game_result, position_key, sort_moves_in_python_order};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Gamma};
use shakmaty::{Chess, Color, Move, MoveList, Position};
use std::collections::{HashMap, VecDeque};
use std::hash::{BuildHasherDefault, Hasher};
use std::mem::MaybeUninit;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

pub struct SearchSettings {
    pub c_puct: f64,
    pub c_puct_factor: f64,
    pub c_puct_base: f64,
    pub fpu_reduction: f64,
    pub draw: f64,
    pub value_sharpness: f64, //1 is mcts.py's search
    //proven results end descents; mcts.py has none
    pub certainty: bool,
    //one node a position; mcts.py has none
    pub transpositions: bool,
}

//a proven result, for the side that played into the node
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Outcome {
    Loss,
    Draw,
    Win,
}

#[derive(Clone, Copy)]
pub struct Proof {
    pub outcome: Outcome,
    pub value: f64, //the node's score under the proof
    pub plies: u32, //to the game's end
}

//plies a tablebase win or loss counts, so that a mate comes first
const TABLEBASE_PLIES: u32 = 1000;

fn proof_rank(proof: &Proof) -> (Outcome, i64) {
    let plies = i64::from(proof.plies);
    (proof.outcome, match proof.outcome {
        Outcome::Win => -plies,
        Outcome::Draw => 0,
        Outcome::Loss => plies,
    })
}

#[derive(Clone, Copy)]
#[repr(align(64))]
pub struct Node {
    pub visit_count: u32,
    //simulations waiting for the network through this node, and a gathering round's collisions
    in_flight: u32,
    pub value_sum: f64,
    visited_policy_sqrt: f64, //sqrt of the sum of visited children's priors
    //first edge above the low 8 bits, number of edges in them (at most 218); 0 until expanded
    edges_at: u64,
    pub proof: Option<Proof>, //only with SearchSettings' certainty, and never at the root
    children: u32,            //its moves with a node (NO_LINK)
}

const NO_NODE: u32 = u32::MAX;

//a move with a node has a link in its node's list, or a slot once the node has SOLID_TREE_THRESHOLD visits. A move is named by its
//link, or by SOLID with its slot; a node's children is NO_LINK, its first link or SOLID with its first slot. Link 0 is never given out
pub const NO_LINK: u32 = 0;
pub const SOLID: u32 = 1 << 31;
pub const SOLID_TREE_THRESHOLD: u32 = 100;

pub fn listed(link: u32) -> bool {
    link & SOLID == 0
}

#[derive(Clone, Copy)]
pub struct Link {
    child: u32,
    visits: u32,
    next: u32,
    place: u32,
}

#[derive(Clone, Copy)]
pub struct Slot {
    child: u32,
    visits: u32,
}

const UNVISITED_SLOT: Slot = Slot { child: NO_NODE, visits: 0 };
const UNLINKED: Link = Link { child: NO_NODE, visits: 0, next: NO_LINK, place: 0 };

impl Node {
    fn new() -> Node {
        Node { visit_count: 0, in_flight: 0, value_sum: 0.0, visited_policy_sqrt: 0.0, edges_at: 0, proof: None, children: NO_LINK }
    }

    pub fn edges(&self) -> Range<usize> {
        let first = (self.edges_at >> 8) as usize;
        first..first + (self.edges_at & 255) as usize
    }

    //a node without moves keeps edges_at 0
    fn set_edges(&mut self, first: usize, count: usize) {
        if count > 0 {
            self.edges_at = (first as u64) << 8 | count as u64;
        }
    }

    pub fn has_edges(&self) -> bool {
        self.edges_at & 255 > 0
    }

    pub fn mean_score(&self) -> f64 {
        self.value_sum / f64::from(self.visit_count)
    }
}

//python's sum() of floats (3.12+) is neumaier's compensated sum, which differs from a plain loop in the last bits
fn python_sum(mut values: impl Iterator<Item = f64>) -> f64 {
    let Some(mut total) = values.next() else { return 0.0 };
    let mut compensation = 0.0;
    for value in values {
        let sum = total + value;
        compensation += if total.abs() >= value.abs() { (total - sum) + value } else { (value - sum) + total };
        total = sum;
    }
    if compensation != 0.0 && compensation.is_finite() { total + compensation } else { total }
}

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

//the value's logit times sharpness
pub fn sharpen(value: f32, sharpness: f64) -> f32 {
    if sharpness == 1.0 {
        return value;
    }
    let value = f64::from(value);
    (1.0 / (1.0 + ((1.0 - value) / value).powf(sharpness))) as f32
}

//a simulation waiting for the network: its path, the links between its nodes (transpositions only), and its leaf's moves
pub struct PendingLeaf {
    pub path: Vec<usize>,
    pub edges: Vec<usize>,
    pub key: u64,
    pub moves: MoveList,
    pub turn: Color,
}

struct Answer<'a> {
    priors: &'a [f32],
    value: f32,
}

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

//position and fifty-move count as one key: positions differing in the count are not shared, so a fifty-move draw holds for every path
pub fn graph_key(key: u64, halfmoves: u32) -> u64 {
    key ^ u64::from(halfmoves + 1).wrapping_mul(0xc2b2_ae3d_27d4_eb4f)
}

//the fifty-move count as Graph keeps it, saturated: the rule's draw ends the game long before
pub fn halfmove_count(halfmoves: u32) -> u16 {
    halfmoves.min(u32::from(u16::MAX)) as u16
}

//the positions of a descent's path that a move from its last node could repeat (a two-fold repetition inside the search): same side
//to move, four plies or more back, no capture or pawn move between. `keys` is the game's positions then the path's from `path_start`,
//the last node's last; `halfmoves` its count
pub struct Repeats<'a> {
    candidates: &'a [u64],
    filter: u128,
}

impl<'a> Repeats<'a> {
    pub fn new(keys: &'a [u64], path_start: usize, halfmoves: u32) -> Repeats<'a> {
        let position = keys.len();
        let lowest = position.saturating_sub(halfmoves as usize + 1).max(path_start);
        let candidates = if position >= lowest + 4 { &keys[lowest..position - 3] } else { &[] };
        let filter = candidates.iter().rev().step_by(2).fold(0, |filter, &key| filter | Self::bit(key));
        Repeats { candidates, filter }
    }

    fn bit(key: u64) -> u128 {
        1 << (key >> 57)
    }

    pub fn repeats(&self, key: u64, halfmoves: u16) -> bool {
        halfmoves != 0 && self.filter & Self::bit(key) != 0 && self.candidates.iter().rev().step_by(2).any(|&earlier| earlier == key)
    }
}

//the node of every position by graph_key, and each node's fifty-move count. The count only grows along a path, so the graph has no
//cycles; a repeat of a position on a path is a draw on that path only (Repeats), never in the numbers of the shared node
#[derive(Clone)]
struct Graph {
    table: KeyMap<u32>,
    halfmoves: Vec<u16>,
}

impl Graph {
    fn new(root_key: u64, root_halfmoves: u32) -> Graph {
        let mut table = KeyMap::default();
        table.insert(graph_key(root_key, root_halfmoves), 0);
        Graph { table, halfmoves: vec![halfmove_count(root_halfmoves)] }
    }

    fn bytes(&self) -> usize {
        self.table.capacity() * TABLE_ENTRY_BYTES + self.halfmoves.capacity() * size_of::<u16>()
    }
}

//a table entry's key, node and control byte, in a table at most 7/8 full
const TABLE_ENTRY_BYTES: usize = 16;

//why a descent stopped above a node: a repeat on its path, or a move with fewer visits than its node (a transposition visit)
enum Stop {
    Repeat,
    CatchUp,
}

const PRIORS_A_POSITION: usize = 40;

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

//the network's answers by position: it sees no history, so an answer is the same however the position was reached. In two
//generations: when the newer is full the older is cleared and becomes the newer
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

//a round (lc0's minibatch): leaves to gather for the network, simulations in all (those finished at once too), and descents that may
//end on a leaf already waiting (collisions). Waiting simulations count as visits in the exploration term
pub struct Round {
    pub id: u64, //its leaves wait under it until finish_leaf takes them; rounds may come back in any order
    pub leaves: u32,
    pub simulations: u32,
    pub collisions: u32,
    pub visit_limit: u32,    //the search's simulations, those done and waiting included
    pub memory_limit: usize, //bytes the tree's vectors may take, a growth's copy included; usize::MAX for none
}

//MoveList's capacity: a waiting leaf's expansion needs that many edges
const MOST_MOVES: usize = 256;

fn node_fits(nodes: usize) -> bool {
    nodes < NO_NODE as usize
}

//the capacity a vector grows to for `needed` elements of `bytes` each: doubling, but no further than a new allocation of `spare`
//bytes (made while the old one is held) allows; None if `needed` does not fit
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

pub type Vectors = (Vec<Node>, Vec<u64>, Vec<PackedMove>, Vec<f32>, Vec<Link>, Vec<Slot>);

const ABORT_CHECKS: usize = 4096;

fn aborted(item: usize, abort: Option<&AtomicBool>) -> bool {
    item.is_multiple_of(ABORT_CHECKS) && abort.is_some_and(|abort| abort.load(Ordering::Relaxed))
}

//fewer items than this are copied on one thread; the tests' small trees take several
const PARALLEL_ITEMS: usize = if cfg!(test) { 64 } else { 16384 };

const PARTS_A_THREAD: usize = 16;

//a subtree (its node, the places of the moves down to it) or a candidate leaf, for one of stale_candidates' threads
enum Part {
    Subtree(usize, Vec<u16>),
    Leaf(Vec<u16>),
}

pub struct Tree {
    pub board: Chess,   //the root's position
    pub keys: Vec<u64>, //the position_key of every position of the game, the root's last
    pub nodes: Vec<Node>,
    root: usize, //the root's node: 0 but after move_root, until compact
    node_keys: Vec<u64>,
    //the edges: every expanded node's moves in python-chess's order with their priors
    moves: Vec<PackedMove>,
    priors: Vec<f32>,        //as the network gives them, read as f64s
    noised_priors: Vec<f64>, //the root's once add_root_noise has mixed its noise in; empty without
    links: Vec<Link>,
    slots: Vec<Slot>,
    //the last round's memory limit, which make_solid stays within
    memory_limit: usize,
    //the simulations waiting for the network by round (Round's id), in the order of the round's boards
    pending: Vec<(u64, VecDeque<PendingLeaf>)>,
    logits: Vec<f32>,
    //no room for another simulation (make_room): the search ends as at its simulations, until a move is played
    pub full: bool,
    graph: Option<Graph>,
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
            links: vec![UNLINKED],
            slots: Vec::new(),
            memory_limit: usize::MAX,
            pending: Vec::new(),
            logits: Vec::new(),
            full: false,
            graph: None,
        }
    }

    pub fn with_transpositions(mut self, transpositions: bool) -> Tree {
        assert_eq!(self.nodes.len(), 1, "transpositions from a tree's start");
        let root_key = self.node_keys[0];
        self.graph = transpositions.then(|| Graph::new(root_key, self.board.halfmoves()));
        self
    }

    //room for one more simulation: its node, and the moves of every leaf that may be expanded before the next look, within the nodes' u32
    //indices and the memory limit. Under a limit the vectors are grown here (grown_capacity), so that no push grows them past it
    fn make_room(&mut self, memory_limit: usize) -> bool {
        self.memory_limit = memory_limit;
        if !node_fits(self.nodes.len()) || self.links.len() >= SOLID as usize {
            return false;
        }
        if memory_limit == usize::MAX {
            return true;
        }
        let graph_node_bytes = if self.graph.is_some() { size_of::<u16>() } else { 0 };
        let node_bytes = size_of::<Node>() + size_of::<u64>() + graph_node_bytes;
        let Some(nodes) =
            grown_capacity(self.nodes.capacity().min(self.node_keys.capacity()), self.nodes.len() + 1, node_bytes, self.spare_bytes())
        else {
            return false;
        };
        reserve_to(&mut self.nodes, nodes);
        reserve_to(&mut self.node_keys, nodes);
        //a link for this simulation's move and one for each waiting leaf's
        let Some(links) =
            grown_capacity(self.links.capacity(), self.links.len() + self.pending() + 1, size_of::<Link>(), self.spare_bytes())
        else {
            return false;
        };
        reserve_to(&mut self.links, links);
        if let Some(graph) = &mut self.graph {
            reserve_to(&mut graph.halfmoves, nodes);
        }
        //the table doubles when full, its new room made while the old is held
        if let Some(graph) = &self.graph
            && graph.table.len() == graph.table.capacity()
        {
            let more = graph.table.capacity().max(1024);
            if self.spare_bytes() < 2 * more * TABLE_ENTRY_BYTES {
                return false;
            }
            self.graph.as_mut().expect("the graph").table.reserve(more);
        }
        let edge_bytes = size_of::<PackedMove>() + size_of::<f32>();
        let capacity = self.moves.capacity().min(self.priors.capacity());
        let needed = self.moves.len() + (self.pending() + 1) * MOST_MOVES;
        let Some(edges) = grown_capacity(capacity, needed, edge_bytes, self.spare_bytes()) else {
            return false;
        };
        reserve_to(&mut self.moves, edges);
        reserve_to(&mut self.priors, edges);
        //one more node's moves of slots, for make_solid
        if let Some(slots) = grown_capacity(self.slots.capacity(), self.slots.len() + MOST_MOVES, size_of::<Slot>(), self.spare_bytes()) {
            reserve_to(&mut self.slots, slots);
        }
        true
    }

    fn spare_bytes(&self) -> usize {
        self.memory_limit.saturating_sub(self.allocated_bytes())
    }

    pub fn allocated_bytes(&self) -> usize {
        self.nodes.capacity() * size_of::<Node>()
            + self.node_keys.capacity() * size_of::<u64>()
            + self.moves.capacity() * size_of::<PackedMove>()
            + self.priors.capacity() * size_of::<f32>()
            + self.links.capacity() * size_of::<Link>()
            + self.slots.capacity() * size_of::<Slot>()
            + self.graph.as_ref().map_or(0, Graph::bytes)
    }

    pub fn root(&self) -> &Node {
        &self.nodes[self.root]
    }

    pub fn root_index(&self) -> usize {
        self.root
    }

    fn prior(&self, edge: usize) -> f64 {
        let root = self.root().edges();
        if !self.noised_priors.is_empty() && root.contains(&edge) {
            self.noised_priors[edge - root.start]
        } else {
            f64::from(self.priors[edge])
        }
    }

    pub fn chess_move(&self, edge: usize) -> PackedMove {
        self.moves[edge]
    }

    //a move's node by its name (NO_NODE for a solid slot without one)
    fn link_child(&self, link: u32) -> u32 {
        if listed(link) { self.links[link as usize].child } else { self.slots[(link & !SOLID) as usize].child }
    }

    fn link_visits(&mut self, link: u32) -> &mut u32 {
        if listed(link) { &mut self.links[link as usize].visits } else { &mut self.slots[(link & !SOLID) as usize].visits }
    }

    fn link_of(&self, node: usize, edge: usize) -> u32 {
        let place = edge - self.nodes[node].edges().start;
        let children = self.nodes[node].children;
        if !listed(children) {
            let link = children + place as u32;
            return if self.link_child(link) == NO_NODE { NO_LINK } else { link };
        }
        let mut link = children;
        while link != NO_LINK && (self.links[link as usize].place as usize) < place {
            link = self.links[link as usize].next;
        }
        if link != NO_LINK && self.links[link as usize].place as usize == place { link } else { NO_LINK }
    }

    //the node's moves with a node in their order, each as (its place among the node's moves, its name)
    pub fn linked(&self, node: usize) -> TreeLinked<'_> {
        let children = self.nodes[node].children;
        let solid = if listed(children) {
            &[][..]
        } else {
            let first = (children & !SOLID) as usize;
            &self.slots[first..first + self.nodes[node].edges().len()]
        };
        TreeLinked { links: &self.links, children, at: if listed(children) { children } else { 0 }, solid }
    }

    //a move's visits by its name: its node's, or with transpositions the move's own (its node counts every path's)
    fn move_visits(&self, link: u32) -> u32 {
        match (link, &self.graph) {
            (NO_LINK, _) => 0,
            (_, Some(_)) => self.own_visits(link),
            (_, None) => self.nodes[self.link_child(link) as usize].visit_count,
        }
    }

    pub fn child_index(&self, node: usize, edge: usize) -> Option<usize> {
        let link = self.link_of(node, edge);
        (link != NO_LINK).then(|| self.link_child(link) as usize)
    }

    pub fn child(&self, node: usize, edge: usize) -> Option<&Node> {
        self.child_index(node, edge).map(|child| &self.nodes[child])
    }

    pub fn visits(&self, node: usize, edge: usize) -> u32 {
        self.move_visits(self.link_of(node, edge))
    }

    //the link a waiting leaf's path holds for a move of the node, renamed if the node was made solid since
    fn current_link(&self, node: usize, link: u32) -> u32 {
        let children = self.nodes[node].children;
        if !listed(children) && listed(link) { children + self.links[link as usize].place } else { link }
    }

    //the node's list copied into slots, one a move; under a memory limit only into the room make_room made
    fn make_solid(&mut self, node: usize) {
        let moves = self.nodes[node].edges().len();
        let head = self.nodes[node].children;
        if !listed(head) || moves == 0 || (self.memory_limit != usize::MAX && self.slots.len() + moves > self.slots.capacity()) {
            return;
        }
        let first = self.slots.len();
        self.slots.resize(first + moves, UNVISITED_SLOT);
        let mut link = head;
        while link != NO_LINK {
            let Link { child, visits, next, place } = self.links[link as usize];
            self.slots[first + place as usize] = Slot { child, visits };
            link = next;
        }
        self.nodes[node].children = first as u32 | SOLID;
    }

    fn own_visits(&self, link: u32) -> u32 {
        match link {
            NO_LINK => 0,
            _ if listed(link) => self.links[link as usize].visits,
            _ => self.slots[(link & !SOLID) as usize].visits,
        }
    }

    pub fn pending(&self) -> usize {
        self.pending.iter().map(|(_, leaves)| leaves.len()).sum()
    }

    pub fn round_moves(&self, round: u64) -> impl Iterator<Item = (&[Move], Color)> {
        self.pending
            .iter()
            .filter(move |(id, _)| *id == round)
            .flat_map(|(_, leaves)| leaves.iter().map(|leaf| (&leaf.moves[..], leaf.turn)))
    }

    //the round's waiting leaves, in the room of a round whose leaves are all back if it has none
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

    fn proven_lost(&self, node: usize, edge: usize) -> bool {
        self.child(node, edge).and_then(|child| child.proof).is_some_and(|proof| proof.outcome == Outcome::Loss)
    }

    //the visits of the root's two most visited moves, leaving out proven losses (most_visited_edge puts them last)
    pub fn two_most_visits(&self) -> (u32, u32) {
        let (mut most, mut second) = (0, 0);
        let root = self.root;
        for visits in self.root().edges().filter(|&edge| !self.proven_lost(root, edge)).map(|edge| self.visits(root, edge)) {
            if visits > most {
                (most, second) = (visits, most);
            } else if visits > second {
                second = visits;
            }
        }
        (most, second)
    }

    //mcts.select_child, with the simulations waiting under a child counted in its visits (lc0's n started); an edge without a node is a
    //child without visits. A proven loss is passed over while any other move is left. With transpositions a move that repeats a position
    //of the path (`repeats`) is worth a draw. Returns the edge and its move's name (NO_LINK for a list's move without a node, a solid
    //slot's whether it has a node or not)
    fn select_edge(&self, node: usize, settings: &SearchSettings, repeats: Option<&Repeats>) -> (usize, u32) {
        let parent = &self.nodes[node];
        let c_puct =
            settings.c_puct + settings.c_puct_factor * ((f64::from(parent.visit_count) + settings.c_puct_base) / settings.c_puct_base).ln();
        let children_visits = parent.visit_count.saturating_sub(1).max(1);
        let edges = parent.edges();
        let first_play_urgency = (1.0 - parent.mean_score()) - settings.fpu_reduction * parent.visited_policy_sqrt;
        let sqrt_children_visits = f64::from(children_visits).sqrt();

        let scan = Scan {
            c_puct,
            sqrt_children_visits,
            first_play_urgency,
            noised: node == self.root && !self.noised_priors.is_empty(),
            transpositions: self.graph.is_some(),
            draw: settings.draw,
        };
        let value =
            |index: usize, prior: f32, child: u32, own_visits: u32| self.move_value(&scan, repeats, index, prior, child, own_visits);
        let (mut best, mut best_u, mut best_link) = (0, f64::NEG_INFINITY, NO_LINK);
        let mut consider = |index: usize, link: u32, found: Option<(f64, u64, u32)>| {
            if let Some((u, child_edges, child_moves)) = found
                && u > best_u
            {
                (best, best_u, best_link) = (index, u, link);
                if child_edges & 255 > 0 {
                    self.prefetch_edges((child_edges >> 8) as usize, child_moves);
                }
            }
        };
        let priors = &self.priors[edges.clone()];
        let children = parent.children;
        if listed(children) {
            let place_of = |link: u32| if link == NO_LINK { usize::MAX } else { self.links[link as usize].place as usize };
            let (mut next_link, mut next_place) = (children, place_of(children));
            for (index, &prior) in priors.iter().enumerate() {
                if index == next_place {
                    let Link { child, visits, next, .. } = self.links[next_link as usize];
                    consider(index, next_link, value(index, prior, child, visits));
                    (next_link, next_place) = (next, place_of(next));
                } else {
                    consider(index, NO_LINK, value(index, prior, NO_NODE, 0));
                }
            }
        } else {
            let first = (children & !SOLID) as usize;
            for (index, (&prior, slot)) in priors.iter().zip(&self.slots[first..first + edges.len()]).enumerate() {
                consider(index, children + index as u32, value(index, prior, slot.child, slot.visits));
            }
        }
        (edges.start + best, best_link)
    }

    //select_edge's value of a move from its prior, node (NO_NODE for none) and own visits, with its node's edges_at and children (0 and
    //NO_LINK for none) for the prefetch; None for a proven loss, never chosen
    #[inline(always)]
    fn move_value(
        &self,
        scan: &Scan,
        repeats: Option<&Repeats>,
        index: usize,
        prior: f32,
        child: u32,
        own_visits: u32,
    ) -> Option<(f64, u64, u32)> {
        let prior = if scan.noised { self.noised_priors[index] } else { f64::from(prior) };
        if child == NO_NODE {
            //sqrt_children_visits / 1.0 is sqrt_children_visits: no division
            return Some((scan.first_play_urgency + (scan.c_puct * prior * scan.sqrt_children_visits), 0, NO_LINK));
        }
        let id = child as usize;
        let child = &self.nodes[id];
        if child.proof.is_some_and(|proof| proof.outcome == Outcome::Loss) {
            return None;
        }
        let repeat =
            repeats.is_some_and(|repeats| repeats.repeats(self.node_keys[id], self.graph.as_ref().expect("the graph").halfmoves[id]));
        let q = if repeat {
            1.0 - scan.draw
        } else if child.visit_count > 0 {
            child.mean_score()
        } else {
            scan.first_play_urgency
        };
        //with transpositions the move's own visits, as its node's count every path's
        let started = if scan.transpositions { own_visits } else { child.visit_count } + child.in_flight;
        Some((q + (scan.c_puct * prior * (scan.sqrt_children_visits / f64::from(1 + started))), child.edges_at, child.children))
    }

    //prefetches the priors from edge `first` on and the first link or slots of `moves` (a node's children)
    #[inline(always)]
    fn prefetch_edges(&self, first: usize, moves: u32) {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            let line: *const i8 = if listed(moves) {
                self.links.as_ptr().wrapping_add(moves as usize).cast()
            } else {
                self.slots.as_ptr().wrapping_add((moves & !SOLID) as usize).cast()
            };
            //SAFETY: sse is part of x86_64, and a prefetch reads nothing the program sees, a pointer past the vectors too
            unsafe {
                _mm_prefetch::<_MM_HINT_T0>(self.priors.as_ptr().wrapping_add(first).cast());
                if moves != NO_LINK {
                    _mm_prefetch::<_MM_HINT_T0>(line);
                }
            }
        }
    }

    //the node of the parent's move `edge` (select_edge's name for it), made if the move has none yet, and the move's name; board is the
    //node's position. With transpositions an existing node of the same position and fifty-move count is the move's
    fn child_node(&mut self, parent: usize, edge: usize, link: u32, board: &Chess) -> (usize, u32) {
        //select_edge names no move when it chose none (every move a proven loss, the first taken): its node if it has one
        let link = if link == NO_LINK { self.link_of(parent, edge) } else { link };
        if link != NO_LINK && self.link_child(link) != NO_NODE {
            return (self.link_child(link) as usize, link);
        }
        let key = position_key(board);
        let shared = self.graph.as_ref().and_then(|graph| graph.table.get(&graph_key(key, board.halfmoves())).copied());
        let child = shared.unwrap_or_else(|| {
            let child = self.nodes.len() as u32;
            if let Some(graph) = &mut self.graph {
                graph.halfmoves.push(halfmove_count(board.halfmoves()));
                graph.table.insert(graph_key(key, board.halfmoves()), child);
            }
            self.nodes.push(Node::new());
            self.node_keys.push(key);
            child
        });
        let place = (edge - self.nodes[parent].edges().start) as u32;
        let children = self.nodes[parent].children;
        if !listed(children) {
            self.slots[(children & !SOLID) as usize + place as usize].child = child;
            return (child as usize, children + place);
        }
        let made = self.links.len() as u32;
        let (mut previous, mut next) = (NO_LINK, self.nodes[parent].children);
        while next != NO_LINK && self.links[next as usize].place < place {
            (previous, next) = (next, self.links[next as usize].next);
        }
        self.links.push(Link { child, visits: 0, next, place });
        if previous == NO_LINK {
            self.nodes[parent].children = made;
        } else {
            self.links[previous as usize].next = made;
        }
        (child as usize, made)
    }

    //mcts.descend and finish until the round has its leaves or simulations, the leaves' boards added to boards; a leaf that needs no
    //network (a game's result, a cached answer) is finished at once. A collision counts as waiting along its path so the next descents
    //turn elsewhere, and is taken off at the end. Returns the leaves gathered
    pub fn gather(
        &mut self,
        round: &Round,
        settings: &SearchSettings,
        tables: Option<&Tables>,
        cache: Option<&Cache>,
        boards: &mut Vec<u8>,
    ) -> u32 {
        let (mut simulations, mut gathered) = (0, 0);
        let mut collisions: Vec<Vec<usize>> = Vec::new();
        let mut path = Vec::new();
        let mut edges = Vec::new();
        //the tables finish the nodes they cover below the root; below a root they cover (a size with no dtz tables) only those right
        //after a capture or pawn move, as lc0 probes: the tables' result starts the fifty move count there
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
            edges.clear();
            let mut node = self.root;
            let mut stop = None;
            while self.nodes[node].has_edges() {
                let halfmoves = board.halfmoves();
                let repeats = self.graph.is_some().then(|| Repeats::new(&self.keys, root_keys - 1, halfmoves));
                let (edge, selected) = self.select_edge(node, settings, repeats.as_ref());
                board.play_unchecked(self.moves[edge].to_move(&board));
                let link;
                (node, link) = self.child_node(node, edge, selected, &board);
                path.push(node);
                if let Some(graph) = &self.graph {
                    edges.push(link as usize);
                    //a repeat of this path's position, whatever the node is to other paths, or a node that knows more than its move
                    if Repeats::new(&self.keys, root_keys - 1, halfmoves).repeats(self.node_keys[node], graph.halfmoves[node]) {
                        stop = Some(Stop::Repeat);
                    } else if self.nodes[node].has_edges()
                        && self.nodes[node].proof.is_none()
                        && self.move_visits(link) < self.nodes[node].visit_count
                    {
                        stop = Some(Stop::CatchUp);
                    }
                }
                self.keys.push(self.node_keys[node]);
                if self.nodes[node].proof.is_some() || stop.is_some() {
                    break;
                }
            }

            //a repeat is a draw for this path, a transposition visit the shared node's mean; only the move counts the visit
            if let Some(stop) = stop {
                self.keys.truncate(root_keys);
                simulations += 1;
                let value = match stop {
                    Stop::Repeat => 1.0 - settings.draw,
                    //clamped as SharedTree does: a mean read there beside other backups may be off
                    Stop::CatchUp => self.nodes[node].mean_score().clamp(0.0, 1.0),
                };
                path.pop();
                self.back_up(&path, &edges, value, 0);
                continue;
            }

            if let Some(proof) = self.nodes[node].proof {
                self.keys.truncate(root_keys);
                simulations += 1;
                self.back_up(&path, &edges, 1.0 - proof.value, 0);
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
                game_result(&board, &self.keys, &legal_moves, settings.draw, node_tables)
            };
            //a repeat inside the search is a draw, but not a proven one: the game need not repeat (as in lc0)
            let (&leaf_key, path_keys) = self.keys[root_keys - 1..].split_last().expect("the root's key");
            //with transpositions a descent stops above a repeat, so a leaf is none; a repetition the rules count holds for every path
            let value = game_over.or_else(|| path_keys.contains(&leaf_key).then_some(settings.draw));
            self.keys.truncate(root_keys);

            simulations += 1;
            if let Some(value) = value {
                //the root is never proven: the rules and the tables never score it
                if let Some(result) = game_over
                    && settings.certainty
                    && node != self.root
                {
                    //the result is for the side to move, the proof for the side that moved here; a win or loss with moves left
                    //is the tables', after any mate
                    let (outcome, plies) = if result == 0.0 {
                        (Outcome::Win, if legal_moves.is_empty() { 0 } else { TABLEBASE_PLIES })
                    } else if result == 1.0 {
                        (Outcome::Loss, TABLEBASE_PLIES)
                    } else {
                        (Outcome::Draw, 0)
                    };
                    self.prove(&path, Proof { outcome, value: 1.0 - result, plies });
                }
                self.back_up(&path, &edges, value, 0);
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
                    self.back_up(&path, &edges, f64::from(answer.value), 0);
                }
                None => {
                    encode_board(&board, boards);
                    self.add_in_flight(&path);
                    let leaf = PendingLeaf { path: path.clone(), edges: edges.clone(), key: leaf_key, moves, turn: board.turn() };
                    self.round_leaves(round.id).push_back(leaf);
                    gathered += 1;
                }
            }
        }
        for path in collisions {
            for node in path {
                self.nodes[node].in_flight -= 1;
            }
        }
        gathered
    }

    //mcts.finish for the round's oldest waiting leaf; answer is the move logits, then the expected score. The cache keeps it, sharpened
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
        self.back_up(&leaf.path, &leaf.edges, f64::from(value), 1);
        if let Some(cache) = cache {
            cache.insert(leaf.key, &priors, value);
        }
        self.logits = priors;
    }

    fn expand(&mut self, node: usize, moves: &MoveList, priors: &[f32]) {
        let first_edge = self.moves.len();
        for (&chess_move, &prior) in moves.iter().zip(priors) {
            self.moves.push(PackedMove::new(chess_move));
            self.priors.push(prior);
        }
        self.nodes[node].edges_at = (first_edge as u64) << 8 | (self.moves.len() - first_edge) as u64;
    }

    fn add_in_flight(&mut self, path: &[usize]) {
        for &node in path {
            self.nodes[node].in_flight += 1;
        }
    }

    //mcts.backup along the path, for a simulation that had in_flight visits waiting there. With transpositions `edges` names the moves
    //between the path's nodes (their links), and one more when the path ends above the node the descent stopped at: each counts the visit
    //and a node's visited policy is refreshed when one of its moves has its first
    fn back_up(&mut self, path: &[usize], edges: &[usize], value: f64, in_flight: u32) {
        let transpositions = self.graph.is_some();
        let mut score = 1.0 - value;
        for (depth, &node) in path.iter().enumerate().rev() {
            //a node proven while the simulation waited for the network gives its proven score (as in lc0)
            if let Some(proof) = self.nodes[node].proof {
                score = proof.value;
            }
            self.nodes[node].visit_count += 1;
            self.nodes[node].in_flight -= in_flight;
            self.nodes[node].value_sum += score;
            if self.nodes[node].visit_count == SOLID_TREE_THRESHOLD {
                self.make_solid(node);
            }
            if !transpositions {
                if self.nodes[node].visit_count == 1 && depth > 0 {
                    self.refresh_visited_policy(path[depth - 1]);
                }
            } else if let Some(&link) = edges.get(depth) {
                let visits = self.link_visits(self.current_link(node, link as u32));
                *visits += 1;
                if *visits == 1 {
                    self.refresh_visited_policy(node);
                }
            }
            score = 1.0 - score;
        }
    }

    //proves the path's last node: its visits so far count the proven score instead of theirs, in its ancestors too (lc0's
    //AdjustForTerminal), alternating as a visit's score s goes to the node and 1 - s to the parent. With transpositions only the
    //node's own: its ancestors had only some of its visits, at scores of their own, and take the proven score with their next visits
    fn prove(&mut self, path: &[usize], proof: Proof) {
        let (&node, ancestors) = path.split_last().expect("a node to prove");
        let change = f64::from(self.nodes[node].visit_count) * proof.value - self.nodes[node].value_sum;
        self.nodes[node].value_sum += change;
        self.nodes[node].proof = Some(proof);
        if self.graph.is_some() {
            return;
        }
        for (distance, &ancestor) in ancestors.iter().rev().enumerate() {
            self.nodes[ancestor].value_sum += if distance % 2 == 0 { -change } else { change };
        }
    }

    fn propagate_proofs(&mut self, path: &[usize]) {
        for depth in (1..path.len().saturating_sub(1)).rev() {
            match self.proof_from_moves(path[depth]) {
                Some(proof) => self.prove(&path[..=depth], proof),
                None => break,
            }
        }
    }

    //an expanded node's result from its moves' proofs: lost for the side that moved into it if a move wins (the quickest), and otherwise
    //known once every move's result is, the best of them
    fn proof_from_moves(&self, node: usize) -> Option<Proof> {
        let moves = self.nodes[node].edges().len();
        if moves == 0 {
            return None;
        }
        let (mut best, mut proven): (Option<Proof>, usize) = (None, 0);
        for (_, link) in self.linked(node) {
            if let Some(proof) = self.nodes[self.link_child(link) as usize].proof {
                proven += 1;
                if best.is_none_or(|best| proof_rank(&proof) > proof_rank(&best)) {
                    best = Some(proof);
                }
            }
        }
        let all_proven = proven == moves;
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

    //a kept tree under a new root (Searches' tree reuse; self-play keeps mcts.py's tree as it is): a leaf scored by a rule that held only
    //from the old root comes out of the tree, its visits taken off its ancestors, as lc0 drops a two-fold draw whose first position is no
    //longer in the search, or a tables' result the new root would not probe. Leaves below a proven node stay: no descent reaches them.
    //With transpositions nothing is taken out: that would miscount a node other paths share
    pub fn drop_stale_leaves(&mut self, settings: &SearchSettings, tables: Option<&Tables>) {
        if self.graph.is_some() {
            return;
        }
        let root_covered = tables.is_some_and(|tables| tables.covers(&self.board));
        let mut stale = Vec::new();
        self.find_stale_leaves(&mut vec![self.root], &mut Vec::new(), settings, tables, root_covered, &mut stale);
        for path in stale {
            self.take_out(&path);
        }
    }

    //the paths to the stale leaves under the path's last node, which the line's moves (edges) reach from the root; self.keys has its keys
    fn find_stale_leaves(
        &mut self,
        path: &mut Vec<usize>,
        line: &mut Vec<usize>,
        settings: &SearchSettings,
        tables: Option<&Tables>,
        root_covered: bool,
        stale: &mut Vec<Vec<usize>>,
    ) {
        let node = *path.last().expect("the root at least");
        let edges = self.nodes[node].edges();
        if edges.is_empty() {
            if node != self.root && self.nodes[node].visit_count > 0 && self.is_stale(path, line, settings, tables, root_covered) {
                stale.push(path.clone());
            }
            return;
        }
        let children: Vec<(usize, u32)> = self.linked(node).map(|(place, link)| (place, self.link_child(link))).collect();
        for (place, child) in children {
            if self.nodes[child as usize].proof.is_some() {
                continue;
            }
            path.push(child as usize);
            line.push(edges.start + place);
            self.keys.push(self.node_keys[child as usize]);
            self.find_stale_leaves(path, line, settings, tables, root_covered, stale);
            self.keys.pop();
            line.pop();
            path.pop();
        }
    }

    //whether the leaf at the end of path, which the line's moves (edges) reach from the root, is no longer what gather scored it as;
    //self.keys ends with the path's keys
    fn is_stale(&self, path: &[usize], line: &[usize], settings: &SearchSettings, tables: Option<&Tables>, root_covered: bool) -> bool {
        let mut board = self.board.clone();
        for &edge in line {
            board.play_unchecked(self.moves[edge].to_move(&board));
        }
        let legal_moves = board.legal_moves();
        let node_tables = tables.filter(|_| !root_covered || board.halfmoves() == 0);
        let over = game_result(&board, &self.keys, &legal_moves, settings.draw, node_tables);
        let (&leaf_key, path_keys) = self.keys[self.keys.len() - path.len()..].split_last().expect("the leaf's key");
        over.is_none() && !path_keys.contains(&leaf_key)
    }

    //the leaves drop_stale_leaves may find stale once the root has moved further down, found before the moves are known (Searches'
    //advance): every leaf below the root with visits, no moves and no proof, each as the places of its line's moves among their nodes'
    //moves, which hold across compact. The walk runs on `threads` threads, each taking the next of split_for_walk's parts, their
    //candidates joined in the parts' order. None once abort is set; with transpositions none, as nothing is taken out
    pub fn stale_candidates(&self, threads: usize, abort: &AtomicBool) -> Option<Vec<Vec<u16>>> {
        if self.graph.is_some() {
            return (!abort.load(Ordering::Relaxed)).then(Vec::new);
        }
        let share = (self.root().visit_count / (threads * PARTS_A_THREAD).max(1) as u32).max(1);
        let mut parts = Vec::new();
        self.split_for_walk(self.root, &mut Vec::new(), share, &mut parts);
        let next = AtomicUsize::new(0);
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

    //the parts of the tree below node (its line's moves at `places`) in the walk's order: a subtree of at most `share` visits is one
    //part, a bigger one is split into its moves'; a candidate leaf met on the way is a part of its own
    fn split_for_walk(&self, node: usize, places: &mut Vec<u16>, share: u32, parts: &mut Vec<Part>) {
        if node != self.root && self.nodes[node].visit_count <= share {
            parts.push(Part::Subtree(node, places.clone()));
            return;
        }
        for (place, link) in self.linked(node) {
            places.push(place as u16);
            let child = self.link_child(link) as usize;
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
        if aborted(*looked, Some(abort)) {
            return false;
        }
        for (place, link) in self.linked(node) {
            places.push(place as u16);
            let child = self.link_child(link) as usize;
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

    //drop_stale_leaves from the candidates found at the root the moves at `line`'s places moved it down from: of those below the root,
    //the leaves its walk would find stale, taken out in its order
    pub fn drop_stale_candidates(
        &mut self,
        line: &[u16],
        candidates: &[Vec<u16>],
        settings: &SearchSettings,
        tables: Option<&Tables>,
    ) {
        if self.graph.is_some() {
            return;
        }
        let root_covered = tables.is_some_and(|tables| tables.covers(&self.board));
        let root_keys = self.keys.len();
        let mut stale = Vec::new();
        'candidates: for places in candidates.iter().filter(|places| places.len() > line.len() && places.starts_with(line)) {
            let (mut path, mut edges) = (vec![self.root], Vec::new());
            for &place in &places[line.len()..] {
                let parent = *path.last().expect("the root at least");
                let edge = self.nodes[parent].edges().start + usize::from(place);
                let child = self.child_index(parent, edge).expect("a candidate's moves have nodes");
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
                if found {
                    stale.push(path);
                }
            }
        }
        for path in stale {
            self.take_out(&path);
        }
    }

    //takes a leaf's visits out of the tree: a visit's score s went to the leaf, 1 - s to its parent and so on up
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
        //keeps python's summation order and compensation, including after noise on a reused root
        let first = self.nodes[node].edges().start;
        let visited = self.linked(node).filter(|&(_, link)| self.move_visits(link) > 0);
        self.nodes[node].visited_policy_sqrt = python_sum(visited.map(|(place, _)| self.prior(first + place))).sqrt();
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

    //mcts.most_visited_move: ties to the better mean score, then the prior, then the first. A proven win comes before every other move
    //(the quickest first) and a proven loss after (the slowest first), as in lc0
    pub fn most_visited_edge(&self, node: usize) -> usize {
        let preference = |edge: usize| {
            let link = self.link_of(node, edge);
            let visits = self.move_visits(link);
            let child = (link != NO_LINK).then(|| &self.nodes[self.link_child(link) as usize]);
            let mean_score = child.filter(|_| visits > 0).map_or(0.0, Node::mean_score);
            let proven = match child.and_then(|child| child.proof) {
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

    pub fn principal_variation(&self) -> Vec<usize> {
        let (mut line, mut node) = (Vec::new(), self.root);
        while self.nodes[node].has_edges() {
            let edge = self.most_visited_edge(node);
            if !line.is_empty() && self.visits(node, edge) == 0 {
                break;
            }
            line.push(edge);
            let Some(child) = self.child_index(node, edge) else { break };
            node = child;
        }
        line
    }

    //the root's edges, the most visited first and ties in their order
    pub fn edges_by_visits(&self) -> Vec<usize> {
        let mut edges: Vec<usize> = self.root().edges().collect();
        edges.sort_by(|&a, &b| self.visits(self.root, b).cmp(&self.visits(self.root, a)));
        edges
    }

    pub fn play(&mut self, edge: usize, keep_subtree: bool) {
        assert_eq!(self.pending(), 0, "a move is played once the search's leaves are back");
        if keep_subtree && self.child_index(self.root, edge).is_some() {
            self.move_root(&[edge]);
            self.compact(1, None);
        } else {
            self.board.play_unchecked(self.moves[edge].to_move(&self.board));
            self.keys.push(position_key(&self.board));
            (self.nodes, self.moves, self.priors) = (vec![Node::new()], Vec::new(), Vec::new());
            (self.links, self.slots) = (vec![UNLINKED], Vec::new());
            self.node_keys = vec![*self.keys.last().expect("the root's key")];
            self.noised_priors.clear();
            self.full = false;
            self.root = 0;
            if self.graph.is_some() {
                self.graph = Some(Graph::new(self.node_keys[0], self.board.halfmoves()));
            }
        }
    }

    pub fn play_line(&mut self, line: &[usize]) {
        self.move_root(line);
        let old = self.compact(1, None);
        //the old tree is freed on a thread of its own, off the move's clock
        let _ = std::thread::Builder::new().spawn(move || drop(old));
    }

    //plays a line of moves from the root (edges, each of the node the last one leads to) and makes the node it ends on the root where it
    //is; the rest stays in the vectors until compact. The new root's proof (a root is never proven) and the old root's noise are dropped
    pub fn move_root(&mut self, line: &[usize]) {
        assert_eq!(self.pending(), 0, "a move is played once the search's leaves are back");
        for &edge in line {
            let child = self.child_index(self.root, edge).expect("a line through the tree's nodes");
            self.board.play_unchecked(self.moves[edge].to_move(&self.board));
            self.keys.push(position_key(&self.board));
            self.root = child;
        }
        if !line.is_empty() {
            self.nodes[self.root].proof = None;
            self.noised_priors.clear();
            self.full = false;
        }
    }

    //the root's part of the tree copied out of the rest (left there by move_root) as a tree of its own, in breadth-first order. Its proofs
    //still hold: they come from the game's end by its rules or the tables, which a new root leaves as it was. On more than one thread the
    //order is found a level at a time, then each stretch of it is copied into its own part of the new vectors: the same tree. Returns the
    //old vectors to free, or None once abort is set (the tree then left as it was)
    pub fn compact(&mut self, threads: usize, abort: Option<&AtomicBool>) -> Option<Vectors> {
        if self.graph.is_some() {
            return self.compact_graph(abort);
        }
        if threads <= 1 {
            return self.compact_on_one_thread(abort);
        }
        let tree: &Tree = self;
        //the kept nodes in breadth-first order (their numbers in the new tree), and of each its moves with a node and its moves
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

        //the order in stretches, one a thread, each with its edges, links and slots and where they and its children's numbers start. A node
        //with SOLID_TREE_THRESHOLD visits gets a slot a move, any other a link a move with a node
        let size = chunk_size(order.len(), threads);
        let positions: Vec<Range<usize>> = (0..order.len()).step_by(size).map(|start| start..(start + size).min(order.len())).collect();
        let solid = |position: usize| tree.nodes[order[position] as usize].visit_count >= SOLID_TREE_THRESHOLD;
        let (mut first_edge, mut first_child, mut first_link, mut first_slot) = (0, 1, 1, 0);
        let starts: Vec<[usize; 7]> = positions
            .iter()
            .map(|positions| {
                let (mut edges, mut with_node, mut links, mut slots) = (0, 0, 0, 0);
                for position in positions.clone() {
                    let (node_with_node, node_edges) = (usize::from(counts[position].0), usize::from(counts[position].1));
                    edges += node_edges;
                    with_node += node_with_node;
                    if solid(position) {
                        slots += node_edges;
                    } else {
                        links += node_with_node;
                    }
                }
                let start = [edges, links, slots, first_edge, first_child, first_link, first_slot];
                (first_edge, first_child, first_link, first_slot) =
                    (first_edge + edges, first_child + with_node, first_link + links, first_slot + slots);
                start
            })
            .collect();

        //room for the whole tree, so the copy never regrows; unwritten room takes no memory
        let (node_room, edge_room) = (tree.nodes.len(), tree.moves.len());
        let (mut nodes, mut node_keys) = (Vec::with_capacity(node_room), Vec::with_capacity(node_room));
        let (mut moves, mut priors) = (Vec::with_capacity(edge_room), Vec::with_capacity(edge_room));
        let (mut links, mut slots) = (Vec::with_capacity(first_link), Vec::with_capacity(first_slot));
        links.push(UNLINKED);
        {
            let mut nodes_left = &mut nodes.spare_capacity_mut()[..order.len()];
            let mut keys_left = &mut node_keys.spare_capacity_mut()[..order.len()];
            let mut moves_left = &mut moves.spare_capacity_mut()[..first_edge];
            let mut priors_left = &mut priors.spare_capacity_mut()[..first_edge];
            let mut links_left = &mut links.spare_capacity_mut()[..first_link - 1];
            let mut slots_left = &mut slots.spare_capacity_mut()[..first_slot];
            let mut stretches = Vec::with_capacity(positions.len());
            for (positions, &[edges, link_count, slot_count, first_edge, first_child, first_link, first_slot]) in
                positions.iter().zip(&starts)
            {
                let (nodes, node_keys, moves, priors, links, slots);
                (nodes, nodes_left) = std::mem::take(&mut nodes_left).split_at_mut(positions.len());
                (node_keys, keys_left) = std::mem::take(&mut keys_left).split_at_mut(positions.len());
                (moves, moves_left) = std::mem::take(&mut moves_left).split_at_mut(edges);
                (priors, priors_left) = std::mem::take(&mut priors_left).split_at_mut(edges);
                (links, links_left) = std::mem::take(&mut links_left).split_at_mut(link_count);
                (slots, slots_left) = std::mem::take(&mut slots_left).split_at_mut(slot_count);
                stretches.push(Stretch {
                    positions: positions.clone(),
                    first_edge,
                    first_child,
                    first_link,
                    first_slot,
                    nodes,
                    node_keys,
                    moves,
                    priors,
                    links,
                    slots,
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
        //SAFETY: the stretches split the first order.len() nodes, first_edge edges, the links after link 0 and the slots between them, and
        //each wrote all of its own
        unsafe {
            nodes.set_len(order.len());
            node_keys.set_len(order.len());
            moves.set_len(first_edge);
            priors.set_len(first_edge);
            links.set_len(first_link);
            slots.set_len(first_slot);
        }
        self.root = 0;
        Some(self.replace_vectors(nodes, node_keys, moves, priors, links, slots))
    }

    fn replace_vectors(
        &mut self,
        nodes: Vec<Node>,
        node_keys: Vec<u64>,
        moves: Vec<PackedMove>,
        priors: Vec<f32>,
        links: Vec<Link>,
        slots: Vec<Slot>,
    ) -> Vectors {
        (
            std::mem::replace(&mut self.nodes, nodes),
            std::mem::replace(&mut self.node_keys, node_keys),
            std::mem::replace(&mut self.moves, moves),
            std::mem::replace(&mut self.priors, priors),
            std::mem::replace(&mut self.links, links),
            std::mem::replace(&mut self.slots, slots),
        )
    }

    fn next_level(&self, chunk: &[u32], abort: Option<&AtomicBool>) -> Option<Level> {
        let (mut next, mut counts) = (Vec::new(), Vec::with_capacity(chunk.len()));
        for (index, &node) in chunk.iter().enumerate() {
            if aborted(index, abort) {
                return None;
            }
            let before = next.len();
            next.extend(self.linked(node as usize).map(|(_, link)| self.link_child(link)));
            counts.push(((next.len() - before) as u8, self.nodes[node as usize].edges().len() as u8));
        }
        Some((next, counts))
    }

    fn compact_on_one_thread(&mut self, abort: Option<&AtomicBool>) -> Option<Vectors> {
        let (node_room, edge_room) = (self.nodes.len(), self.moves.len());
        let (mut nodes, mut node_keys, mut old_of) =
            (Vec::with_capacity(node_room), Vec::with_capacity(node_room), Vec::with_capacity(node_room));
        nodes.push(Node { proof: None, ..self.nodes[self.root] });
        node_keys.push(self.node_keys[self.root]);
        old_of.push(self.root);
        let (mut moves, mut priors) = (Vec::with_capacity(edge_room), Vec::with_capacity(edge_room));
        let (mut links, mut slots) = (Vec::with_capacity(self.links.len()), Vec::with_capacity(self.slots.len()));
        links.push(UNLINKED);
        let mut children = Vec::new();
        let mut index = 0;
        while index < nodes.len() {
            if aborted(index, abort) {
                return None;
            }
            let old = old_of[index];
            let edges = self.nodes[old].edges();
            nodes[index].set_edges(moves.len(), edges.len());
            moves.extend_from_slice(&self.moves[edges.clone()]);
            priors.extend_from_slice(&self.priors[edges]);
            children.clear();
            for (place, link) in self.linked(old) {
                let child = self.link_child(link) as usize;
                children.push((place, nodes.len() as u32, self.own_visits(link)));
                nodes.push(self.nodes[child]);
                node_keys.push(self.node_keys[child]);
                old_of.push(child);
            }
            let solid = self.nodes[old].visit_count >= SOLID_TREE_THRESHOLD;
            link_up(&mut nodes[index], &mut links, &mut slots, solid, children.iter().copied());
            index += 1;
        }
        self.root = 0;
        Some(self.replace_vectors(nodes, node_keys, moves, priors, links, slots))
    }

    //a node that several moves lead to is copied once, at the first, and each move is given its new number and its own visits
    fn compact_graph(&mut self, abort: Option<&AtomicBool>) -> Option<Vectors> {
        let graph = self.graph.as_ref().expect("the graph");
        let (node_room, edge_room) = (self.nodes.len(), self.moves.len());
        let mut copied_as = vec![NO_NODE; node_room];
        let mut order = vec![self.root];
        copied_as[self.root] = 0;
        let mut nodes = Vec::with_capacity(node_room);
        let mut node_keys = Vec::with_capacity(node_room);
        let mut halfmoves = Vec::with_capacity(node_room);
        let (mut moves, mut priors) = (Vec::with_capacity(edge_room), Vec::with_capacity(edge_room));
        let (mut links, mut slots) = (Vec::with_capacity(self.links.len()), Vec::with_capacity(self.slots.len()));
        links.push(UNLINKED);
        let mut children = Vec::new();
        let mut index = 0;
        while index < order.len() {
            if aborted(index, abort) {
                return None;
            }
            let old = order[index];
            let mut node = self.nodes[old];
            if index == 0 {
                node.proof = None;
            }
            let edges = node.edges();
            node.set_edges(moves.len(), edges.len());
            node_keys.push(self.node_keys[old]);
            halfmoves.push(graph.halfmoves[old]);
            moves.extend_from_slice(&self.moves[edges.clone()]);
            priors.extend_from_slice(&self.priors[edges]);
            children.clear();
            for (place, link) in self.linked(old) {
                let child = self.link_child(link) as usize;
                if copied_as[child] == NO_NODE {
                    copied_as[child] = order.len() as u32;
                    order.push(child);
                }
                children.push((place, copied_as[child], self.own_visits(link)));
            }
            let solid = node.visit_count >= SOLID_TREE_THRESHOLD;
            link_up(&mut node, &mut links, &mut slots, solid, children.iter().copied());
            nodes.push(node);
            index += 1;
        }
        let mut table = KeyMap::default();
        table.reserve(nodes.len());
        for (node, (&key, &count)) in node_keys.iter().zip(&halfmoves).enumerate() {
            table.insert(graph_key(key, u32::from(count)), node as u32);
        }
        self.graph = Some(Graph { table, halfmoves });
        self.root = 0;
        Some(self.replace_vectors(nodes, node_keys, moves, priors, links, slots))
    }

    //copies a stretch, numbering the moves' nodes as the order has them; false once abort is set
    fn copy_stretch(&self, stretch: Stretch, order: &[u32], abort: Option<&AtomicBool>) -> bool {
        let (mut edge, mut child) = (stretch.first_edge, stretch.first_child);
        let (mut link, mut slot) = (stretch.first_link, stretch.first_slot);
        for (copied, position) in stretch.positions.clone().enumerate() {
            if aborted(copied, abort) {
                return false;
            }
            let old = order[position] as usize;
            let mut node = self.nodes[old];
            if position == 0 {
                node.proof = None;
            }
            let edges = node.edges();
            node.set_edges(edge, edges.len());
            for old_edge in edges.clone() {
                let at = edge - stretch.first_edge;
                stretch.moves[at].write(self.moves[old_edge]);
                stretch.priors[at].write(self.priors[old_edge]);
                edge += 1;
            }
            if node.visit_count >= SOLID_TREE_THRESHOLD {
                let first = slot;
                for at in first..first + edges.len() {
                    stretch.slots[at - stretch.first_slot].write(UNVISITED_SLOT);
                }
                for (place, old_link) in self.linked(old) {
                    stretch.slots[first + place - stretch.first_slot]
                        .write(Slot { child: child as u32, visits: self.own_visits(old_link) });
                    child += 1;
                }
                node.children = first as u32 | SOLID;
                slot += edges.len();
            } else {
                node.children = NO_LINK;
                let mut previous = None;
                for (place, old_link) in self.linked(old) {
                    if previous.is_none() {
                        node.children = link as u32;
                    }
                    let next = Link { child: child as u32, visits: self.own_visits(old_link), next: NO_LINK, place: place as u32 };
                    if let Some((at, mut written)) = previous.replace((link, next)) {
                        written.next = link as u32;
                        stretch.links[at - stretch.first_link].write(written);
                    }
                    child += 1;
                    link += 1;
                }
                if let Some((at, last)) = previous {
                    stretch.links[at - stretch.first_link].write(last);
                }
            }
            stretch.nodes[position - stretch.positions.start].write(node);
            stretch.node_keys[position - stretch.positions.start].write(self.node_keys[old]);
        }
        true
    }
}

//adds a node's moves with a node, as (place, node, own visits), to the links or slots of a tree being made
fn link_up(node: &mut Node, links: &mut Vec<Link>, slots: &mut Vec<Slot>, solid: bool, children: impl Iterator<Item = (usize, u32, u32)>) {
    if solid {
        let first = slots.len();
        slots.resize(first + node.edges().len(), UNVISITED_SLOT);
        for (place, child, visits) in children {
            slots[first + place] = Slot { child, visits };
        }
        node.children = first as u32 | SOLID;
        return;
    }
    node.children = NO_LINK;
    for (place, child, visits) in children {
        let link = links.len() as u32;
        if node.children == NO_LINK {
            node.children = link;
        } else {
            links[link as usize - 1].next = link;
        }
        links.push(Link { child, visits, next: NO_LINK, place: place as u32 });
    }
}

pub struct TreeLinked<'a> {
    links: &'a [Link],
    children: u32,
    at: u32, //a list's next link, a solid node's next place
    solid: &'a [Slot],
}

impl Iterator for TreeLinked<'_> {
    type Item = (usize, u32);

    fn next(&mut self) -> Option<(usize, u32)> {
        if listed(self.children) {
            if self.at == NO_LINK {
                return None;
            }
            let link = self.at;
            self.at = self.links[link as usize].next;
            return Some((self.links[link as usize].place as usize, link));
        }
        while let Some(slot) = self.solid.get(self.at as usize) {
            let place = self.at;
            self.at += 1;
            if slot.child != NO_NODE {
                return Some((place as usize, self.children + place));
            }
        }
        None
    }
}

struct Scan {
    c_puct: f64,
    sqrt_children_visits: f64,
    first_play_urgency: f64,
    noised: bool,
    transpositions: bool,
    draw: f64,
}

//a chunk's next level: the nodes its moves lead to, and of each node in the chunk (its moves with a node, its moves)
type Level = (Vec<u32>, Vec<(u8, u8)>);

//one thread's stretch of compact's order: where its edges, links, slots and children's numbers start, and its parts of the new vectors
struct Stretch<'a> {
    positions: Range<usize>,
    first_edge: usize,
    first_child: usize,
    first_link: usize,
    first_slot: usize,
    nodes: &'a mut [MaybeUninit<Node>],
    node_keys: &'a mut [MaybeUninit<u64>],
    moves: &'a mut [MaybeUninit<PackedMove>],
    priors: &'a mut [MaybeUninit<f32>],
    links: &'a mut [MaybeUninit<Link>],
    slots: &'a mut [MaybeUninit<Slot>],
}

fn chunk_size(items: usize, threads: usize) -> usize {
    if items < PARALLEL_ITEMS { items.max(1) } else { items.div_ceil(threads.max(1)) }
}

//f of each item, in order, on a scoped thread each but the first; items is not empty
fn each_in_parallel<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    std::thread::scope(|scope| {
        let workers: Vec<_> = items[1..].iter().map(|item| scope.spawn(|| f(item))).collect();
        let mut done = vec![f(&items[0])];
        done.extend(workers.into_iter().map(|worker| worker.join().expect("a thread of the copy")));
        done
    })
}

mod shared;
pub use shared::{SharedCache, SharedTree};

#[cfg(test)]
mod tests;

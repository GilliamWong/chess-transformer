//mcts.py's search with the same order of operations and arithmetic, so that it visits the same nodes. A round may also
//gather many leaves for one network call (gather), as lc0 does; one leaf a round is mcts.py's search exactly
use crate::rules::{MoveIds, Tables, encode_board, game_result, position_key, sort_moves_in_python_order};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Gamma};
use shakmaty::{Chess, Color, Move, MoveList, Position};
use shakmaty_syzygy::SyzygyError;
use std::collections::{HashMap, VecDeque};
use std::ops::Range;

//mcts.py's constants
pub struct SearchSettings {
    pub c_puct: f64,
    pub c_puct_factor: f64,
    pub c_puct_base: f64,
    pub fpu_reduction: f64,
    pub draw: f64,
    pub repetitions_for_a_draw: usize,
    pub value_sharpness: f64, //the network's value through sharpen; 1 is mcts.py's search
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
    first_edge: u32,
    edge_count: u32, //0 until the node is expanded
}

//an edge without a node
const NO_NODE: u32 = u32::MAX;

impl Node {
    fn new() -> Node {
        Node { visit_count: 0, in_flight: 0, value_sum: 0.0, visited_policy_sqrt: 0.0, first_edge: 0, edge_count: 0 }
    }

    //the node's moves, as indices into the tree's edges
    pub fn edges(&self) -> Range<usize> {
        self.first_edge as usize..(self.first_edge + self.edge_count) as usize
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
struct Answer {
    priors: Box<[f32]>,
    value: f32,
}

//the network's answers by position (lc0's nn cache): a position reached again by another order of moves (a third of a
//d512 search's leaves at 1 s a move) needs no network call. The network sees no history, so a position's answer is the
//same however it was reached. In two generations: when the newer is full the older is dropped, keeping the latest
pub struct Cache {
    newer: HashMap<u64, Answer>,
    older: HashMap<u64, Answer>,
    generation: usize, //positions the newer generation holds before it becomes the older
}

impl Cache {
    pub fn new(positions: usize) -> Cache {
        Cache { newer: HashMap::new(), older: HashMap::new(), generation: positions.div_ceil(2) }
    }

    fn get(&self, key: u64) -> Option<&Answer> {
        self.newer.get(&key).or_else(|| self.older.get(&key))
    }

    fn insert(&mut self, key: u64, answer: Answer) {
        if self.newer.len() >= self.generation {
            self.older = std::mem::take(&mut self.newer);
        }
        self.newer.insert(key, answer);
    }
}

//how many leaves a round may gather for the network (lc0's minibatch), how many simulations it may do in all (those
//finished at once too: lc0's out of order evaluations), and how many of its descents may end on a leaf already waiting
//(collisions). The simulations waiting for the network count as visits in the exploration term, so a round spreads out
pub struct Round {
    pub leaves: u32,
    pub simulations: u32,
    pub collisions: u32,
    pub visit_limit: u32, //the search's simulations, those already done and waiting included
}

pub struct Tree {
    pub board: Chess,   //the root's position
    pub keys: Vec<u64>, //the position_key of every position of the game, the root's last
    pub nodes: Vec<Node>,
    node_keys: Vec<u64>, //each node's position_key, made with the node
    //the edges: every expanded node's moves in python-chess's order with their priors, and each move's node once it
    //has one (NO_NODE before)
    moves: Vec<Move>,
    priors: Vec<f64>,
    children: Vec<u32>,
    pending: VecDeque<PendingLeaf>, //oldest first, the order the network's answers come back in
    logits: Vec<f32>,
}

impl Tree {
    pub fn new(board: Chess, keys: Vec<u64>) -> Tree {
        let root_key = *keys.last().expect("the root's key");
        Tree {
            board,
            keys,
            nodes: vec![Node::new()],
            node_keys: vec![root_key],
            moves: Vec::new(),
            priors: Vec::new(),
            children: Vec::new(),
            pending: VecDeque::new(),
            logits: Vec::new(),
        }
    }

    pub fn root(&self) -> &Node {
        &self.nodes[0]
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
        self.pending.len()
    }

    //the visits of the root's two most visited moves, 0 for a move it does not have
    pub fn two_most_visits(&self) -> (u32, u32) {
        let (mut most, mut second) = (0, 0);
        for visits in self.root().edges().map(|edge| self.visits(edge)) {
            if visits > most {
                (most, second) = (visits, most);
            } else if visits > second {
                second = visits;
            }
        }
        (most, second)
    }

    //mcts.select_child, with the simulations under a child that wait for the network counted in its visits for the
    //exploration term (lc0's n started); an edge without a node is a child without visits. Returns the edge
    fn select_edge(&self, node: usize, settings: &SearchSettings) -> usize {
        let parent = &self.nodes[node];
        let c_puct =
            settings.c_puct + settings.c_puct_factor * ((f64::from(parent.visit_count) + settings.c_puct_base) / settings.c_puct_base).ln();
        let children_visits = parent.visit_count.saturating_sub(1).max(1);
        let edges = parent.edges();
        let first_play_urgency = (1.0 - parent.mean_score()) - settings.fpu_reduction * parent.visited_policy_sqrt;
        let sqrt_children_visits = f64::from(children_visits).sqrt();

        let (mut best, mut best_u) = (0, f64::NEG_INFINITY);
        for (index, (&prior, &child)) in self.priors[edges.clone()].iter().zip(&self.children[edges.clone()]).enumerate() {
            let (q, started) = match child {
                NO_NODE => (first_play_urgency, 0),
                child => {
                    let child = &self.nodes[child as usize];
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
            && self.root().visit_count as usize + self.pending.len() < round.visit_limit as usize
        {
            let mut board = self.board.clone();
            let root_keys = self.keys.len();
            path.clear();
            path.push(0);
            let mut node = 0;
            while self.nodes[node].edge_count > 0 {
                let edge = self.select_edge(node, settings);
                board.play_unchecked(self.moves[edge]);
                //a descent never ends without a visit or a simulation waiting, so the node is needed
                node = self.child_node(edge, &board);
                path.push(node);
                self.keys.push(self.node_keys[node]);
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
            let node_tables = tables.filter(|_| node != 0 && (!root_covered || board.halfmoves() == 0));
            let mut value = game_result(&board, &self.keys, &legal_moves, settings.draw, settings.repetitions_for_a_draw, node_tables)?;
            //a repeat inside the search is a draw
            let (&leaf_key, path_keys) = self.keys[root_keys - 1..].split_last().expect("the root's key");
            if value.is_none() && path_keys.contains(&leaf_key) {
                value = Some(settings.draw);
            }
            self.keys.truncate(root_keys);

            simulations += 1;
            if let Some(value) = value {
                self.backup(&path, value, 0);
                continue;
            }
            sort_moves_in_python_order(&board, &mut legal_moves);
            let moves = legal_moves;
            //a hash collision would give another number of moves
            match cache.and_then(|cache| cache.get(leaf_key)).filter(|answer| answer.priors.len() == moves.len()) {
                Some(answer) => {
                    self.expand(node, &moves, &answer.priors);
                    self.backup(&path, f64::from(answer.value), 0);
                }
                None => {
                    encode_board(&board, boards);
                    self.add_in_flight(&path);
                    self.pending.push_back(PendingLeaf { path: path.clone(), key: leaf_key, moves, turn: board.turn() });
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

    //mcts.finish for the oldest waiting leaf; answer is the move logits, then the expected score. The cache keeps it, sharpened
    pub fn finish_leaf(
        &mut self,
        answer: impl Fn(usize) -> f32,
        answer_width: usize,
        move_ids: &MoveIds,
        cache: Option<&mut Cache>,
        settings: &SearchSettings,
    ) {
        let leaf = self.pending.pop_front().expect("a waiting simulation");
        self.logits.clear();
        self.logits.extend(leaf.moves.iter().map(|&chess_move| answer(move_ids.id(chess_move, leaf.turn))));
        softmax(&mut self.logits);
        let value = sharpen(answer(answer_width - 1), settings.value_sharpness);

        let node = *leaf.path.last().expect("the root at least");
        let priors = std::mem::take(&mut self.logits);
        self.expand(node, &leaf.moves, &priors);
        self.backup(&leaf.path, f64::from(value), 1);
        match cache {
            Some(cache) => cache.insert(leaf.key, Answer { priors: priors.into_boxed_slice(), value }),
            None => self.logits = priors,
        }
    }

    //mcts.expand: the moves as edges, without nodes
    fn expand(&mut self, node: usize, moves: &MoveList, priors: &[f32]) {
        let first_edge = self.moves.len();
        for (&chess_move, &prior) in moves.iter().zip(priors) {
            self.moves.push(chess_move);
            self.priors.push(f64::from(prior));
        }
        self.children.resize(self.moves.len(), NO_NODE);
        self.nodes[node].first_edge = first_edge as u32;
        self.nodes[node].edge_count = (self.moves.len() - first_edge) as u32;
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
            self.nodes[node].visit_count += 1;
            self.nodes[node].in_flight -= in_flight;
            self.nodes[node].value_sum += score;
            if self.nodes[node].visit_count == 1 && depth > 0 {
                self.refresh_visited_policy(path[depth - 1]);
            }
            score = 1.0 - score;
        }
    }

    fn refresh_visited_policy(&mut self, node: usize) {
        //Keep Python's summation order and compensation, including after noise on a reused root.
        let edges = self.nodes[node].edges();
        self.nodes[node].visited_policy_sqrt = python_sum(edges.filter(|&edge| self.visits(edge) > 0).map(|edge| self.priors[edge])).sqrt();
    }

    //mcts.add_root_noise: dirichlet noise as gamma draws scaled to add up to 1
    pub fn add_root_noise(&mut self, random: &mut ChaCha8Rng, alpha: f64, share: f64) {
        let gamma = Gamma::new(alpha, 1.0).expect("a positive noise alpha");
        let edges = self.nodes[0].edges();
        let draws: Vec<f64> = edges.clone().map(|_| gamma.sample(random)).collect();
        let total: f64 = draws.iter().sum();
        for (edge, draw) in edges.zip(draws) {
            let prior = self.priors[edge];
            self.priors[edge] = (1.0 - share) * prior + share * (draw / total);
        }
        self.refresh_visited_policy(0);
    }

    //mcts.most_visited_move: ties to the better mean score, then the prior, then the first. Returns the edge
    pub fn most_visited_edge(&self, node: usize) -> usize {
        let preference = |edge: usize| {
            let visits = self.visits(edge);
            let mean_score = if visits > 0 { self.nodes[self.children[edge] as usize].mean_score() } else { 0.0 };
            (visits, mean_score, self.priors[edge])
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
        let (mut line, mut node) = (Vec::new(), 0);
        while self.nodes[node].edge_count > 0 {
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
        assert!(self.pending.is_empty(), "a move is played once the search's leaves are back");
        let chess_move = self.moves[edge];
        let child = self.children[edge];
        self.board.play_unchecked(chess_move);
        self.keys.push(position_key(&self.board));
        if keep_subtree && child != NO_NODE {
            self.keep_subtree(child as usize);
        } else {
            (self.nodes, self.moves, self.priors, self.children) = (vec![Node::new()], Vec::new(), Vec::new(), Vec::new());
            self.node_keys = vec![*self.keys.last().expect("the root's key")];
        }
    }

    //the tree under node as a tree of its own, node its root
    fn keep_subtree(&mut self, node: usize) {
        let mut nodes = vec![self.nodes[node]];
        let mut node_keys = vec![self.node_keys[node]];
        let (mut moves, mut priors, mut children) = (Vec::new(), Vec::new(), Vec::new());
        let mut index = 0;
        while index < nodes.len() {
            let edges = nodes[index].edges();
            nodes[index].first_edge = moves.len() as u32;
            moves.extend_from_slice(&self.moves[edges.clone()]);
            priors.extend_from_slice(&self.priors[edges.clone()]);
            for edge in edges {
                children.push(match self.children[edge] {
                    NO_NODE => NO_NODE,
                    child => {
                        nodes.push(self.nodes[child as usize]);
                        node_keys.push(self.node_keys[child as usize]);
                        (nodes.len() - 1) as u32
                    }
                });
            }
            index += 1;
        }
        (self.nodes, self.moves, self.priors, self.children, self.node_keys) = (nodes, moves, priors, children, node_keys);
    }
}

//mcts.py's search with the same order of operations and arithmetic, so that it visits the same nodes. A round may also
//gather many leaves for one network call (gather), as lc0 does; one leaf a round is mcts.py's search exactly
use crate::rules::{MoveIds, Tables, encode_board, game_result, legal_moves_in_python_order, position_key};
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
}

//mcts.Node, in one vector per tree: the root first, every node's children contiguous
#[derive(Clone, Copy)]
pub struct Node {
    pub chess_move: Option<Move>, //the move into this node, None for a root
    pub prior: f64,
    pub visit_count: u32,
    //simulations through this node waiting for the network, and while a round is gathered its collisions
    in_flight: u32,
    pub value_sum: f64,
    first_child: u32,
    child_count: u32, //0 until the node is expanded
}

impl Node {
    fn new(chess_move: Option<Move>, prior: f64) -> Node {
        Node { chess_move, prior, visit_count: 0, in_flight: 0, value_sum: 0.0, first_child: 0, child_count: 0 }
    }

    pub fn children(&self) -> Range<usize> {
        self.first_child as usize..(self.first_child + self.child_count) as usize
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

//one simulation a round, mcts.py's search
pub const ONE_LEAF: Round = Round { leaves: 1, simulations: 1, collisions: 0, visit_limit: u32::MAX };

pub struct Tree {
    pub board: Chess,   //the root's position
    pub keys: Vec<u64>, //the position_key of every position of the game, the root's last
    pub nodes: Vec<Node>,
    pending: VecDeque<PendingLeaf>, //oldest first, the order the network's answers come back in
    logits: Vec<f32>,
}

impl Tree {
    pub fn new(board: Chess, keys: Vec<u64>) -> Tree {
        Tree { board, keys, nodes: vec![Node::new(None, 1.0)], pending: VecDeque::new(), logits: Vec::new() }
    }

    pub fn root(&self) -> &Node {
        &self.nodes[0]
    }

    //simulations waiting for the network
    pub fn pending(&self) -> usize {
        self.pending.len()
    }

    //mcts.select_child, with the simulations under a child that wait for the network counted in its visits for the
    //exploration term (lc0's n started)
    fn select_child(&self, node: usize, settings: &SearchSettings) -> usize {
        let parent = &self.nodes[node];
        let c_puct =
            settings.c_puct + settings.c_puct_factor * ((f64::from(parent.visit_count) + settings.c_puct_base) / settings.c_puct_base).ln();
        let children_visits = parent.visit_count.saturating_sub(1).max(1);
        let children = &self.nodes[parent.children()];
        let visited_policy = python_sum(children.iter().filter(|child| child.visit_count > 0).map(|child| child.prior));
        let first_play_urgency = (1.0 - parent.mean_score()) - settings.fpu_reduction * visited_policy.sqrt();
        let sqrt_children_visits = f64::from(children_visits).sqrt();

        let (mut best, mut best_u) = (0, f64::NEG_INFINITY);
        for (index, child) in children.iter().enumerate() {
            let q = if child.visit_count > 0 { child.mean_score() } else { first_play_urgency };
            let started = child.visit_count + child.in_flight;
            let u = q + (c_puct * child.prior * (sqrt_children_visits / f64::from(1 + started)));
            if u > best_u {
                (best, best_u) = (index, u);
            }
        }
        parent.first_child as usize + best
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
        while gathered < round.leaves
            && simulations < round.simulations
            && self.root().visit_count as usize + self.pending.len() < round.visit_limit as usize
        {
            let mut board = self.board.clone();
            let root_keys = self.keys.len();
            path.clear();
            path.push(0);
            let mut node = 0;
            while self.nodes[node].child_count > 0 {
                node = self.select_child(node, settings);
                board.play_unchecked(self.nodes[node].chess_move.expect("a child has its move"));
                path.push(node);
                self.keys.push(position_key(&board));
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

            let legal_moves = board.legal_moves();
            let mut value = game_result(&board, &self.keys, &legal_moves, settings.draw, settings.repetitions_for_a_draw, tables)?;
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
            let moves = legal_moves_in_python_order(&board);
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

    //mcts.descend, and mcts.finish at once when the leaf needs no network. True when its board was added to boards
    pub fn descend(&mut self, settings: &SearchSettings, tables: Option<&Tables>, boards: &mut Vec<u8>) -> Result<bool, SyzygyError> {
        Ok(self.gather(&ONE_LEAF, settings, tables, None, boards)? == 1)
    }

    //mcts.finish for the oldest waiting leaf; answer is the move logits, then the expected score. The cache keeps it
    pub fn finish_leaf(&mut self, answer: impl Fn(usize) -> f32, answer_width: usize, move_ids: &MoveIds, cache: Option<&mut Cache>) {
        let leaf = self.pending.pop_front().expect("a waiting simulation");
        self.logits.clear();
        self.logits.extend(leaf.moves.iter().map(|&chess_move| answer(move_ids.id(chess_move, leaf.turn))));
        softmax(&mut self.logits);
        let value = answer(answer_width - 1);

        let node = *leaf.path.last().expect("the root at least");
        let priors = std::mem::take(&mut self.logits);
        self.expand(node, &leaf.moves, &priors);
        self.backup(&leaf.path, f64::from(value), 1);
        match cache {
            Some(cache) => cache.insert(leaf.key, Answer { priors: priors.into_boxed_slice(), value }),
            None => self.logits = priors,
        }
    }

    //mcts.expand
    fn expand(&mut self, node: usize, moves: &MoveList, priors: &[f32]) {
        self.nodes[node].first_child = self.nodes.len() as u32;
        self.nodes[node].child_count = moves.len() as u32;
        for (&chess_move, &prior) in moves.iter().zip(priors) {
            self.nodes.push(Node::new(Some(chess_move), f64::from(prior)));
        }
    }

    fn add_in_flight(&mut self, path: &[usize]) {
        for &node in path {
            self.nodes[node].in_flight += 1;
        }
    }

    //mcts.backup along the path, for a simulation that had in_flight visits waiting there
    fn backup(&mut self, path: &[usize], value: f64, in_flight: u32) {
        let mut score = 1.0 - value;
        for &node in path.iter().rev() {
            self.nodes[node].visit_count += 1;
            self.nodes[node].in_flight -= in_flight;
            self.nodes[node].value_sum += score;
            score = 1.0 - score;
        }
    }

    //mcts.add_root_noise: dirichlet noise as gamma draws scaled to add up to 1
    pub fn add_root_noise(&mut self, random: &mut ChaCha8Rng, alpha: f64, share: f64) {
        let gamma = Gamma::new(alpha, 1.0).expect("a positive noise alpha");
        let children = self.nodes[0].children();
        let draws: Vec<f64> = children.clone().map(|_| gamma.sample(random)).collect();
        let total: f64 = draws.iter().sum();
        for (child, draw) in children.zip(draws) {
            let prior = self.nodes[child].prior;
            self.nodes[child].prior = (1.0 - share) * prior + share * (draw / total);
        }
    }

    //mcts.most_visited_move: ties to the better mean score, then the prior, then the first
    pub fn most_visited_child(&self, node: usize) -> usize {
        let preference = |node: &Node| {
            let mean_score = if node.visit_count > 0 { node.mean_score() } else { 0.0 };
            (node.visit_count, mean_score, node.prior)
        };
        let mut children = self.nodes[node].children();
        let mut best = children.next().expect("a searched node has children");
        for child in children {
            if preference(&self.nodes[child]) > preference(&self.nodes[best]) {
                best = child;
            }
        }
        best
    }

    //mcts.principal_variation
    pub fn principal_variation(&self) -> Vec<usize> {
        let (mut line, mut node) = (Vec::new(), 0);
        while self.nodes[node].child_count > 0 {
            let child = self.most_visited_child(node);
            if !line.is_empty() && self.nodes[child].visit_count == 0 {
                break;
            }
            line.push(child);
            node = child;
        }
        line
    }

    //the root's children, the most visited first and ties in their order (a stable sort)
    pub fn children_by_visits(&self) -> Vec<usize> {
        let mut children: Vec<usize> = self.root().children().collect();
        children.sort_by(|&a, &b| self.nodes[b].visit_count.cmp(&self.nodes[a].visit_count));
        children
    }

    //plays the root's child and makes it the root, with its subtree (tree reuse) or without
    pub fn play(&mut self, child: usize, keep_subtree: bool) {
        assert!(self.pending.is_empty(), "a move is played once the search's leaves are back");
        let chess_move = self.nodes[child].chess_move.expect("a child has its move");
        self.nodes = if keep_subtree { self.subtree(child) } else { vec![Node::new(None, 1.0)] };
        self.board.play_unchecked(chess_move);
        self.keys.push(position_key(&self.board));
    }

    //the tree under node as a tree of its own, node its root
    fn subtree(&self, node: usize) -> Vec<Node> {
        let mut kept = vec![self.nodes[node]];
        let mut index = 0;
        while index < kept.len() {
            let children = kept[index].children();
            if !children.is_empty() {
                kept[index].first_child = kept.len() as u32;
                kept.extend_from_slice(&self.nodes[children]);
            }
            index += 1;
        }
        kept
    }
}

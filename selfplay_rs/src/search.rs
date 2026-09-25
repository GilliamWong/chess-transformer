//mcts.py's search with the same order of operations and arithmetic, so that it visits the same nodes
use crate::rules::{MoveIds, Tables, encode_board, game_result, legal_moves_in_python_order, position_key};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Gamma};
use shakmaty::{Chess, Color, Move, MoveList, Position};
use shakmaty_syzygy::SyzygyError;
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
    pub value_sum: f64,
    first_child: u32,
    child_count: u32, //0 until the node is expanded
}

impl Node {
    fn new(chess_move: Option<Move>, prior: f64) -> Node {
        Node { chess_move, prior, visit_count: 0, value_sum: 0.0, first_child: 0, child_count: 0 }
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

//the eval head's logit as a score, as evaluate_many's torch.sigmoid
pub fn sigmoid(logit: f32) -> f32 {
    1.0 / (1.0 + (-logit).exp())
}

pub struct Tree {
    pub board: Chess,   //the root's position
    pub keys: Vec<u64>, //the position_key of every position of the game, the root's last
    pub nodes: Vec<Node>,
    //the simulation waiting for the network: its nodes from the root down, and its leaf's legal moves
    path: Vec<usize>,
    leaf_moves: MoveList,
    leaf_turn: Color,
    logits: Vec<f32>,
}

impl Tree {
    pub fn new(board: Chess, keys: Vec<u64>) -> Tree {
        Tree {
            board,
            keys,
            nodes: vec![Node::new(None, 1.0)],
            path: Vec::new(),
            leaf_moves: MoveList::new(),
            leaf_turn: Color::White,
            logits: Vec::new(),
        }
    }

    pub fn root(&self) -> &Node {
        &self.nodes[0]
    }

    //mcts.select_child
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
            let u = q + (c_puct * child.prior * (sqrt_children_visits / f64::from(1 + child.visit_count)));
            if u > best_u {
                (best, best_u) = (index, u);
            }
        }
        parent.first_child as usize + best
    }

    //mcts.descend, and mcts.finish at once when the leaf needs no network. True when its board was added to boards
    pub fn descend(&mut self, settings: &SearchSettings, tables: Option<&Tables>, boards: &mut Vec<u8>) -> Result<bool, SyzygyError> {
        let mut board = self.board.clone();
        let root_keys = self.keys.len();
        self.path.clear();
        self.path.push(0);
        let mut node = 0;
        while self.nodes[node].child_count > 0 {
            node = self.select_child(node, settings);
            board.play_unchecked(self.nodes[node].chess_move.expect("a child has its move"));
            self.path.push(node);
            self.keys.push(position_key(&board));
        }

        let legal_moves = board.legal_moves();
        let mut value = game_result(&board, &self.keys, &legal_moves, settings.draw, settings.repetitions_for_a_draw, tables)?;
        //a repeat inside the search is a draw
        let (leaf_key, path_keys) = self.keys[root_keys - 1..].split_last().expect("the root's key");
        if value.is_none() && path_keys.contains(leaf_key) {
            value = Some(settings.draw);
        }
        self.keys.truncate(root_keys);

        match value {
            Some(value) => {
                self.backup(value);
                Ok(false)
            }
            None => {
                self.leaf_moves = legal_moves_in_python_order(&board);
                self.leaf_turn = board.turn();
                encode_board(&board, boards);
                Ok(true)
            }
        }
    }

    //mcts.finish for the waiting leaf; answer is the move logits, then the eval logit
    pub fn finish_leaf(&mut self, answer: impl Fn(usize) -> f32, answer_width: usize, move_ids: &MoveIds) {
        self.logits.clear();
        self.logits.extend(self.leaf_moves.iter().map(|&chess_move| answer(move_ids.id(chess_move, self.leaf_turn))));
        softmax(&mut self.logits);

        let leaf = *self.path.last().expect("a waiting simulation");
        self.nodes[leaf].first_child = self.nodes.len() as u32;
        self.nodes[leaf].child_count = self.leaf_moves.len() as u32;
        for (&chess_move, &prior) in self.leaf_moves.iter().zip(&self.logits) {
            self.nodes.push(Node::new(Some(chess_move), f64::from(prior)));
        }
        self.backup(f64::from(sigmoid(answer(answer_width - 1))));
    }

    //mcts.backup along the path
    fn backup(&mut self, value: f64) {
        let mut score = 1.0 - value;
        for &node in self.path.iter().rev() {
            self.nodes[node].visit_count += 1;
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

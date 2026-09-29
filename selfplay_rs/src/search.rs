//mcts.py's search with the same order of operations and arithmetic, so that it visits the same nodes. A round may also
//gather many leaves for one network call (gather), as lc0 does; one leaf a round is mcts.py's search exactly
use crate::rules::{MoveIds, Tables, encode_board, game_result, position_key, sort_moves_in_python_order};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Gamma};
use shakmaty::{Chess, Color, Move, MoveList, Position};
use shakmaty_syzygy::SyzygyError;
use std::collections::{HashMap, VecDeque};
use std::ops::Range;

//mcts.py's constants, and the engine's own
pub struct SearchSettings {
    pub c_puct: f64,
    pub c_puct_factor: f64,
    pub c_puct_base: f64,
    pub fpu_reduction: f64,
    pub draw: f64,
    pub repetitions_for_a_draw: usize,
    //the priors are the softmax of the logits over this (lc0's policy softmax temperature): above 1 flattens them.
    //mcts.py has none, which is 1
    pub policy_temperature: f64,
    //certainty propagation (lc0's sticky endgames): a game's end in the tree (mate, a draw by the rules, the
    //tablebase) is a proven result, and so is a node with a winning move or with every move's result proven; a
    //two-fold repetition stays an unproven draw. A proven node's visits all count its result, and the search no longer
    //goes below it. mcts.py has none
    pub certainty: bool,
    //the network does not see the fifty move rule's count: its score is drawn towards a draw by halfmoves / this
    //(stockfish scales its eval so), so that the side ahead makes progress. 0 for none, as mcts.py
    pub fifty_move_horizon: f64,
    //exploration scaled by how much the scores backed up through a node vary (katago's cpuctUtilityStdevScale): a
    //node's c_puct is multiplied by 1 + scale * (stdev / prior - 1), its stdev taken as if `prior` had been seen
    //prior_weight times more. A scale of 0 for none, as mcts.py
    pub cpuct_stdev_scale: f64,
    pub cpuct_stdev_prior: f64,
    pub cpuct_stdev_prior_weight: f64,
    //the move played is the root move with the best lower confidence bound on its score, its mean less lcb_stdevs
    //standard errors, among those with at least lcb_least_share of the most visited's visits (leela zero's and
    //katago's lcb). 0 for none: the most visited, as mcts.py
    pub lcb_stdevs: f64,
    pub lcb_least_share: f64,
}

impl SearchSettings {
    //mcts.py's search: its constants, the engine's own settings off
    pub fn mcts(c_puct: f64, c_puct_factor: f64, c_puct_base: f64, fpu_reduction: f64, draw: f64, repetitions_for_a_draw: usize) -> Self {
        SearchSettings {
            c_puct,
            c_puct_factor,
            c_puct_base,
            fpu_reduction,
            draw,
            repetitions_for_a_draw,
            policy_temperature: 1.0,
            certainty: false,
            fifty_move_horizon: 0.0,
            cpuct_stdev_scale: 0.0,
            cpuct_stdev_prior: 0.2,
            cpuct_stdev_prior_weight: 2.0,
            lcb_stdevs: 0.0,
            lcb_least_share: 0.1,
        }
    }
}

//the network's score for the side to move, drawn towards a draw by SearchSettings' fifty_move_horizon
fn fifty_move_scaled(value: f64, halfmoves: u32, settings: &SearchSettings) -> f64 {
    if settings.fifty_move_horizon == 0.0 {
        return value;
    }
    let kept = (1.0 - f64::from(halfmoves) / settings.fifty_move_horizon).max(0.0);
    settings.draw + (value - settings.draw) * kept
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
    value_square_sum: f64, //of the scores' squares, for their spread
    visited_policy_sqrt: f64, //sqrt of the sum of visited children's priors, refreshed when that set or the priors change
    first_edge: u32,
    edge_count: u32, //0 until the node is expanded
    pub proof: Option<Proof>, //only with SearchSettings' certainty, and never at the root
}

//an edge without a node
const NO_NODE: u32 = u32::MAX;

impl Node {
    fn new() -> Node {
        Node {
            visit_count: 0,
            in_flight: 0,
            value_sum: 0.0,
            value_square_sum: 0.0,
            visited_policy_sqrt: 0.0,
            first_edge: 0,
            edge_count: 0,
            proof: None,
        }
    }

    //the node's moves, as indices into the tree's edges
    pub fn edges(&self) -> Range<usize> {
        self.first_edge as usize..(self.first_edge + self.edge_count) as usize
    }

    pub fn mean_score(&self) -> f64 {
        self.value_sum / f64::from(self.visit_count)
    }

    //the standard deviation of the scores backed up through the node, as if `prior` had been seen `weight` times more
    //(katago's); `prior` itself until there are two
    fn score_stdev(&self, prior: f64, weight: f64) -> f64 {
        let visits = f64::from(self.visit_count);
        if visits <= 1.0 {
            return prior;
        }
        let mean = self.mean_score();
        let variance = ((mean * mean + prior * prior) * weight + self.value_square_sum) / (weight + visits - 1.0) - mean * mean;
        variance.max(0.0).sqrt()
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
    halfmoves: u32, //since the last capture or pawn move
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
    //the edges: every expanded node's moves in python-chess's order with their priors, and each move's node once it
    //has one (NO_NODE before)
    moves: Vec<Move>,
    priors: Vec<f64>,
    children: Vec<u32>,
    pending: VecDeque<PendingLeaf>, //oldest first, the order the network's answers come back in
    logits: Vec<f32>,
    //taken off a draw's score for the root's side to move and added for the other side (contempt): above 0 the search
    //plays on rather than draw
    pub contempt: f64,
}

impl Tree {
    pub fn new(board: Chess, keys: Vec<u64>) -> Tree {
        Tree {
            board,
            keys,
            nodes: vec![Node::new()],
            moves: Vec::new(),
            priors: Vec::new(),
            children: Vec::new(),
            pending: VecDeque::new(),
            logits: Vec::new(),
            contempt: 0.0,
        }
    }

    //a draw's score for the side to move in a position of the tree
    fn draw_score(&self, turn: Color, settings: &SearchSettings) -> f64 {
        if turn == self.board.turn() { settings.draw - self.contempt } else { settings.draw + self.contempt }
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
        let mut c_puct =
            settings.c_puct + settings.c_puct_factor * ((f64::from(parent.visit_count) + settings.c_puct_base) / settings.c_puct_base).ln();
        if settings.cpuct_stdev_scale != 0.0 {
            let stdev = parent.score_stdev(settings.cpuct_stdev_prior, settings.cpuct_stdev_prior_weight);
            c_puct *= 1.0 + settings.cpuct_stdev_scale * (stdev / settings.cpuct_stdev_prior - 1.0);
        }
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

    //the edge's node, made if the edge has none yet
    fn child_node(&mut self, edge: usize) -> usize {
        if self.children[edge] == NO_NODE {
            self.children[edge] = self.nodes.len() as u32;
            self.nodes.push(Node::new());
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
                node = self.child_node(edge);
                path.push(node);
                self.keys.push(position_key(&board));
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
            let draw = self.draw_score(board.turn(), settings);
            let game_over = game_result(&board, &self.keys, &legal_moves, draw, settings.repetitions_for_a_draw, tables)?;
            //a repeat inside the search is a draw, but not a proven one: the game need not repeat (lc0 gives two-fold
            //draws no bounds either)
            let (&leaf_key, path_keys) = self.keys[root_keys - 1..].split_last().expect("the root's key");
            let value = game_over.or_else(|| path_keys.contains(&leaf_key).then_some(draw));
            self.keys.truncate(root_keys);

            simulations += 1;
            if let Some(value) = value {
                //the root is never proven: start refuses a root whose game is over
                if let Some(result) = game_over
                    && settings.certainty
                    && node != 0
                {
                    //the result is for the side to move, the proof for the side that moved here. A win or loss with
                    //moves left is the tablebase's, after any mate
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
                    self.expand(node, &moves, &answer.priors);
                    self.backup(&path, fifty_move_scaled(f64::from(answer.value), board.halfmoves(), settings), 0);
                }
                None => {
                    encode_board(&board, boards);
                    self.add_in_flight(&path);
                    let (turn, halfmoves) = (board.turn(), board.halfmoves());
                    self.pending.push_back(PendingLeaf { path: path.clone(), key: leaf_key, moves, turn, halfmoves });
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

    //mcts.finish for the oldest waiting leaf; answer is the move logits, then the expected score. The cache keeps it
    pub fn finish_leaf(
        &mut self,
        answer: impl Fn(usize) -> f32,
        answer_width: usize,
        move_ids: &MoveIds,
        settings: &SearchSettings,
        cache: Option<&mut Cache>,
    ) {
        let leaf = self.pending.pop_front().expect("a waiting simulation");
        self.logits.clear();
        self.logits.extend(leaf.moves.iter().map(|&chess_move| answer(move_ids.id(chess_move, leaf.turn))));
        //at 1 the logits are left alone, so the priors stay mcts.py's to the last bit
        if settings.policy_temperature != 1.0 {
            let temperature = settings.policy_temperature as f32;
            self.logits.iter_mut().for_each(|logit| *logit /= temperature);
        }
        softmax(&mut self.logits);
        let value = answer(answer_width - 1);

        let node = *leaf.path.last().expect("the root at least");
        let priors = std::mem::take(&mut self.logits);
        self.expand(node, &leaf.moves, &priors);
        self.backup(&leaf.path, fifty_move_scaled(f64::from(value), leaf.halfmoves, settings), 1);
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
            //from a node proven while the simulation waited for the network, the proven score (lc0's backup does the same)
            if let Some(proof) = self.nodes[node].proof {
                score = proof.value;
            }
            self.nodes[node].visit_count += 1;
            self.nodes[node].in_flight -= in_flight;
            self.nodes[node].value_sum += score;
            self.nodes[node].value_square_sum += score * score;
            if self.nodes[node].visit_count == 1 && depth > 0 {
                self.refresh_visited_policy(path[depth - 1]);
            }
            score = 1.0 - score;
        }
    }

    //proves the path's last node: its visits so far count the proven score instead of theirs, in its ancestors too
    //(lc0's AdjustForTerminal). A visit's score s goes to the node and 1 - s to the parent, so the sums' change
    //alternates, and the squares' change in the parent's is the node's less twice the sums'
    fn prove(&mut self, path: &[usize], proof: Proof) {
        let (&node, ancestors) = path.split_last().expect("a node to prove");
        let visits = f64::from(self.nodes[node].visit_count);
        let (proven_sum, proven_squares) = (visits * proof.value, visits * proof.value * proof.value);
        let change = proven_sum - self.nodes[node].value_sum;
        let square_change = proven_squares - self.nodes[node].value_square_sum;
        (self.nodes[node].value_sum, self.nodes[node].value_square_sum) = (proven_sum, proven_squares);
        self.nodes[node].proof = Some(proof);
        for (distance, &ancestor) in ancestors.iter().rev().enumerate() {
            let flipped = distance % 2 == 0;
            self.nodes[ancestor].value_sum += if flipped { -change } else { change };
            self.nodes[ancestor].value_square_sum += if flipped { square_change - 2.0 * change } else { square_change };
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
        self.proof_from_moves(0).is_some()
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
            (proven, visits, mean_score, self.priors[edge])
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

    //the root's move to play: with SearchSettings' lcb_stdevs the move with the best lower confidence bound on its
    //score among those visited at least lcb_least_share as often as the most visited (at least twice), unless the
    //most visited is a proven win; else the most visited
    pub fn chosen_edge(&self, settings: &SearchSettings) -> usize {
        let most_visited = self.most_visited_edge(0);
        let proven_win = self.child(most_visited).and_then(|child| child.proof).is_some_and(|proof| proof.outcome == Outcome::Win);
        if settings.lcb_stdevs == 0.0 || proven_win || self.visits(most_visited) < 2 {
            return most_visited;
        }
        let least_visits = (f64::from(self.visits(most_visited)) * settings.lcb_least_share).max(2.0);
        let lower_bound = |edge: usize| {
            let child = self.child(edge).expect("a visited move");
            let stdev = child.score_stdev(settings.cpuct_stdev_prior, settings.cpuct_stdev_prior_weight);
            child.mean_score() - settings.lcb_stdevs * stdev / f64::from(child.visit_count).sqrt()
        };
        let mut best = (most_visited, lower_bound(most_visited));
        for edge in self.root().edges().filter(|&edge| f64::from(self.visits(edge)) >= least_visits) {
            let bound = lower_bound(edge);
            if bound > best.1 {
                best = (edge, bound);
            }
        }
        best.0
    }

    //mcts.principal_variation, as edges, from the root's chosen_edge
    pub fn principal_variation(&self, settings: &SearchSettings) -> Vec<usize> {
        let (mut line, mut node) = (Vec::new(), 0);
        while self.nodes[node].edge_count > 0 {
            let edge = if node == 0 { self.chosen_edge(settings) } else { self.most_visited_edge(node) };
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
        if keep_subtree && child != NO_NODE {
            self.keep_subtree(child as usize);
        } else {
            (self.nodes, self.moves, self.priors, self.children) = (vec![Node::new()], Vec::new(), Vec::new(), Vec::new());
        }
        self.board.play_unchecked(chess_move);
        self.keys.push(position_key(&self.board));
    }

    //the tree under node as a tree of its own, node its root. Its proofs still hold: a proof comes from the game's end
    //by its rules or the tablebase, which a new root leaves as it was (a two-fold repetition proves nothing)
    fn keep_subtree(&mut self, node: usize) {
        let mut nodes = vec![self.nodes[node]];
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
                        (nodes.len() - 1) as u32
                    }
                });
            }
            index += 1;
        }
        (self.nodes, self.moves, self.priors, self.children) = (nodes, moves, priors, children);
    }
}

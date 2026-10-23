//gather_batch: a round's visits shared out from the root in bulk (lc0's PickNodesToExtend), on worker threads
//a different search from gather's, its leaves the same for any number of workers; passes of one visit are gather's
use super::*;
use crate::rules::TOKENS_PER_BOARD;
use crate::workers::Workers;
use std::cmp;
use std::collections::BinaryHeap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

//with workers, a node with more than its share of their items' visits is shared out before they take it
const ITEMS_PER_WORKER: u32 = 2;

//a line from the root (edges, in a buffer of them), the node it ends on (NO_NODE: none yet) and its visits
#[derive(Clone)]
pub(super) struct Share {
    pub(super) line: Range<usize>,
    pub(super) node: u32,
    pub(super) visits: u32,
}

//a pick's leaf, prepared, its moves in the worker's buffer; Scored: the rules', the tables' or a repeat's value
enum Leaf<'c> {
    Scored { value: f64, game_over: Option<f64>, no_moves: bool },
    Cached { moves: Range<usize>, answer: Answer<'c> },
    Network { moves: Range<usize>, turn: Color, tokens: [u8; TOKENS_PER_BOARD] },
}

pub(super) struct Pick<'c> {
    pub(super) share: Share,
    key: u64,               //the leaf's position key, for its node if it has none yet
    leaf: Option<Leaf<'c>>, //None for a leaf waiting for the network or a proven node
}

//a worker's picks of a pass, by item
#[derive(Default)]
pub(super) struct Output<'c> {
    pub(super) edges: Vec<u32>,
    pub(super) picks: Vec<Pick<'c>>,
    moves: Vec<Move>,
    items: Vec<(usize, Range<usize>)>,
    error: Option<SyzygyError>,
    share_out: Duration,
    picking: Duration,
    preparing: Duration,
}

//a move in share_visits' heap: the highest u first, of equal ones the first move, as select_edge picks
#[derive(Clone, Copy, PartialEq)]
struct Candidate {
    u: f64,
    index: u32,
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Candidate) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Candidate) -> cmp::Ordering {
        self.u.total_cmp(&other.u).then_with(|| other.index.cmp(&self.index))
    }
}

//select_edge's numbers for a node's moves while its visits are shared out
#[derive(Default)]
pub(super) struct Split {
    exploration: Vec<f64>, //c_puct x prior
    q: Vec<f64>,
    started: Vec<u32>, //visits, those in flight and those given in the pass
    given: Vec<u32>,
    candidates: BinaryHeap<Candidate>, //the moves select_edge may pick: not a proven loss
}

#[derive(Default)]
pub(super) struct Scratch {
    pub(super) line: Vec<u32>,
    shares: Vec<(u32, u32)>,
    split: Split,
    keys: Vec<u64>,
    tokens: Vec<u8>,
}

//what gather_batch did, added up over its calls
#[derive(Default, Clone, Debug)]
pub struct BatchCounts {
    pub rounds: u64,
    pub passes: u64,
    pub items: u64,
    pub picks: u64,
    pub leaves: u64,         //for the network
    pub cached: u64,         //cache hits
    pub scored: u64,         //the visits of results found or proven
    pub collisions: u64,     //visits that ended on a leaf waiting for the network
    pub unspent: u64,        //a cache hit's visits after its first, left to the round's next passes
    pub share_out: Duration, //the caller's in the pass's top, shared out as the workers start on it
    pub workers: Duration,   //the pass's picks and preparations, its top's included
    pub apply: Duration,
    pub picking: Duration,   //the workers' time in picks, added up over them
    pub preparing: Duration, //and in preparations
}

//sets the flag when dropped
struct SetOnDrop<'a>(&'a AtomicBool);

impl Drop for SetOnDrop<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

//a round's totals so far, and its collisions' paths (nodes, in one vector) and visits
#[derive(Default)]
struct RoundState {
    simulations: u32,
    gathered: u32,
    collided: u32,
    capped: bool, //a collision found no room under the round's limit: the round ends
    collided_nodes: Vec<usize>,
    collisions: Vec<(Range<usize>, u32)>,
}

impl Tree {
    //whether a pass's visits go on through the node, as gather's descents do: expanded and not proven
    fn shares_out(&self, node: u32) -> bool {
        node != NO_NODE && self.nodes[node as usize].has_edges() && self.nodes[node as usize].proof.is_none()
    }

    //the visits as that many descents' select_edge would share them, nothing backed up between: (edge, visits)
    pub(super) fn share_visits(
        &self,
        node: usize,
        visits: u32,
        settings: &SearchSettings,
        split: &mut Split,
        shares: &mut Vec<(u32, u32)>,
    ) {
        if visits == 1 {
            shares.push((self.select_edge(node, settings) as u32, 1));
            return;
        }
        let parent = &self.nodes[node];
        let c_puct =
            settings.c_puct + settings.c_puct_factor * ((f64::from(parent.visit_count) + settings.c_puct_base) / settings.c_puct_base).ln();
        let children_visits = parent.visit_count.saturating_sub(1).max(1);
        let edges = parent.edges();
        let first_play_urgency = (1.0 - parent.mean_score()) - settings.fpu_reduction * parent.visited_policy_sqrt;
        let sqrt_children_visits = f64::from(children_visits).sqrt();
        //select_edge's u, its arithmetic in its order
        let u = |exploration: f64, q: f64, started: u32| q + (exploration * (sqrt_children_visits / f64::from(1 + started)));

        let noised = node == self.root_index() && !self.noised_priors.is_empty();
        //select_edge's numbers for a move: c_puct x prior, q, started visits, and false for a proven loss it passes over
        let numbers = |index: usize, prior: f32, child: u32| {
            let exploration = c_puct * if noised { self.noised_priors[index] } else { f64::from(prior) };
            match child {
                NO_NODE => (exploration, first_play_urgency, 0, true),
                child => {
                    let child = &self.nodes[child as usize];
                    let lost = child.proof.is_some_and(|proof| proof.outcome == Outcome::Loss);
                    let q = if child.visit_count > 0 { child.mean_score() } else { first_play_urgency };
                    (exploration, q, child.visit_count + child.in_flight, !lost)
                }
            }
        };
        let moves = || self.priors[edges.clone()].iter().zip(&self.children[edges.clone()]).enumerate();

        //mostly one move keeps them all (a busy node's trunk): one scan finds select_edge's move and the best u before it
        //(which wins a tie) and after it
        let (mut best, mut best_numbers, mut best_u) = (None, (0.0, 0.0, 0), f64::NEG_INFINITY);
        let (mut before, mut after) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for (index, (&prior, &child)) in moves() {
            let (exploration, child_q, child_started, open) = numbers(index, prior, child);
            if !open {
                continue;
            }
            let child_u = u(exploration, child_q, child_started);
            //the old best's u is the highest before the new one
            if child_u > best_u {
                (before, after) = (best_u, f64::NEG_INFINITY);
                (best, best_numbers, best_u) = (Some(index), (exploration, child_q, child_started), child_u);
            } else {
                after = after.max(child_u);
            }
        }
        let keeps_all = |(exploration, q, started): (f64, f64, u32)| {
            let last_u = u(exploration, q, started + visits - 1);
            last_u > before && last_u >= after
        };
        match best {
            //every move a proven loss: select_edge's first move
            None => {
                shares.push((edges.start as u32, visits));
                return;
            }
            Some(index) if keeps_all(best_numbers) => {
                shares.push(((edges.start + index) as u32, visits));
                return;
            }
            Some(_) => {}
        }

        let Split { exploration, q, started, given, candidates } = split;
        for column in [&mut *exploration, &mut *q] {
            column.clear();
        }
        started.clear();
        given.clear();
        let mut open_moves = std::mem::take(candidates).into_vec();
        open_moves.clear();
        for (index, (&prior, &child)) in moves() {
            let (child_exploration, child_q, child_started, open) = numbers(index, prior, child);
            let child_u = u(child_exploration, child_q, child_started);
            //a NaN never beats select_edge's best
            if open && !child_u.is_nan() {
                open_moves.push(Candidate { u: child_u, index: index as u32 });
            }
            exploration.push(child_exploration);
            q.push(child_q);
            started.push(child_started);
            given.push(0);
        }
        *candidates = BinaryHeap::from(open_moves);

        let mut left = visits;
        while left > 0 {
            //every move a proven loss: select_edge's first move
            let Some(&best) = candidates.peek() else {
                given[0] += left;
                break;
            };
            let chosen_index = best.index as usize;
            //the next best is the better of the heap's top's two children
            let heap = candidates.as_slice();
            let next = match (heap.get(1), heap.get(2)) {
                (Some(&first), Some(&second)) => Some(first.max(second)),
                (first, _) => first.copied(),
            };
            //it keeps the visits while its falling u still beats the next best's, which wins a tie from before it
            let taken = match next {
                None => left,
                Some(next) => {
                    let chosen = |more: u32| {
                        let best_u = u(exploration[chosen_index], q[chosen_index], started[chosen_index] + more);
                        if next.index < best.index { best_u > next.u } else { best_u >= next.u }
                    };
                    //mostly one visit or a few: a galloping search, then a binary one
                    let (mut chosen_more, mut step) = (0, 1);
                    while chosen_more + step < left && chosen(chosen_more + step) {
                        chosen_more += step;
                        step *= 2;
                    }
                    let mut passed_over = (chosen_more + step).min(left);
                    while passed_over - chosen_more > 1 {
                        let middle = chosen_more + (passed_over - chosen_more) / 2;
                        if chosen(middle) {
                            chosen_more = middle;
                        } else {
                            passed_over = middle;
                        }
                    }
                    passed_over
                }
            };
            given[chosen_index] += taken;
            started[chosen_index] += taken;
            left -= taken;
            if left > 0 {
                candidates.peek_mut().expect("the best").u = u(exploration[chosen_index], q[chosen_index], started[chosen_index]);
            }
        }
        shares.extend(
            given.iter().enumerate().filter(|&(_, &visits)| visits > 0).map(|(index, &visits)| ((edges.start + index) as u32, visits)),
        );
    }

    //the visits passed down the node's subtree: a pick for each line they end on, in the tree's order
    pub(super) fn pick_down(&self, node: u32, visits: u32, settings: &SearchSettings, scratch: &mut Scratch, output: &mut Output<'_>) {
        if !self.shares_out(node) {
            let start = output.edges.len();
            output.edges.extend_from_slice(&scratch.line);
            output.picks.push(Pick { share: Share { line: start..output.edges.len(), node, visits }, key: 0, leaf: None });
            return;
        }
        let start = scratch.shares.len();
        self.share_visits(node as usize, visits, settings, &mut scratch.split, &mut scratch.shares);
        for index in start..scratch.shares.len() {
            let (edge, visits) = scratch.shares[index];
            scratch.line.push(edge);
            self.pick_down(self.children[edge as usize], visits, settings, scratch, output);
            scratch.line.pop();
        }
        scratch.shares.truncate(start);
    }

    //the pass's top shared out to nodes of at most `most` visits, each published as the workers' next item
    fn share_out(
        &self,
        node: u32,
        visits: u32,
        most: u32,
        settings: &SearchSettings,
        scratch: &mut Scratch,
        publish: &mut dyn FnMut(&[u32], u32, u32),
    ) {
        if visits <= most || !self.shares_out(node) {
            publish(&scratch.line, node, visits);
            return;
        }
        let start = scratch.shares.len();
        self.share_visits(node as usize, visits, settings, &mut scratch.split, &mut scratch.shares);
        for index in start..scratch.shares.len() {
            let (edge, visits) = scratch.shares[index];
            scratch.line.push(edge);
            self.share_out(self.children[edge as usize], visits, most, settings, scratch, publish);
            scratch.line.pop();
        }
        scratch.shares.truncate(start);
    }

    //a pick's leaf as gather finds it: rules, tables, repeats, cache, encoding (scratch.keys: the game's keys)
    #[allow(clippy::too_many_arguments)]
    fn prepare<'c>(
        &self,
        pick: &mut Pick<'c>,
        line: &[u32],
        settings: &SearchSettings,
        tables: Option<&Tables>,
        root_covered: bool,
        cache: Option<&'c Cache>,
        moves: &mut Vec<Move>,
        scratch: &mut Scratch,
    ) -> Result<(), SyzygyError> {
        let root_keys = self.keys.len();
        scratch.keys.truncate(root_keys);
        let mut board = self.board.clone();
        for &edge in line {
            board.play_unchecked(self.moves[edge as usize]);
            let child = self.children[edge as usize];
            scratch.keys.push(if child == NO_NODE { position_key(&board) } else { self.node_keys[child as usize] });
        }
        let mut legal_moves = board.legal_moves();
        let root = line.is_empty();
        let node_tables = tables.filter(|_| !root && (!root_covered || board.halfmoves() == 0));
        let game_over = if root {
            None
        } else {
            game_result(&board, &scratch.keys, &legal_moves, settings.draw, settings.repetitions_for_a_draw, node_tables)?
        };
        let (&leaf_key, path_keys) = scratch.keys[root_keys - 1..].split_last().expect("the root's key");
        pick.key = leaf_key;
        let value = game_over.or_else(|| path_keys.contains(&leaf_key).then_some(settings.draw));
        pick.leaf = Some(match value {
            Some(value) => Leaf::Scored { value, game_over, no_moves: legal_moves.is_empty() },
            None => {
                sort_moves_in_python_order(&board, &mut legal_moves);
                let start = moves.len();
                moves.extend_from_slice(&legal_moves);
                let leaf_moves = start..moves.len();
                match cache.and_then(|cache| cache.get(leaf_key)).filter(|answer| answer.priors.len() == legal_moves.len()) {
                    Some(answer) => Leaf::Cached { moves: leaf_moves, answer },
                    None => {
                        scratch.tokens.clear();
                        encode_board(&board, &mut scratch.tokens);
                        let tokens = scratch.tokens.as_slice().try_into().expect("a board's tokens");
                        Leaf::Network { moves: leaf_moves, turn: board.turn(), tokens }
                    }
                }
            }
        });
        Ok(())
    }

    //a pick to prepare: on an edge without a node, or on a node nothing waits on that is not proven
    fn to_prepare(&self, node: u32) -> bool {
        node == NO_NODE || (self.nodes[node as usize].in_flight == 0 && self.nodes[node as usize].proof.is_none())
    }

    //a pass of `visits`: the workers' picks and leaves, and each item's worker and picks in the tree's order
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn pick_pass<'c>(
        &self,
        visits: u32,
        settings: &SearchSettings,
        tables: Option<&Tables>,
        root_covered: bool,
        cache: Option<&'c Cache>,
        workers: &Workers,
        counts: &mut BatchCounts,
    ) -> Result<(Vec<Output<'c>>, Vec<(usize, Range<usize>)>), SyzygyError> {
        let started = Instant::now();
        let most = if workers.count() == 1 { u32::MAX } else { (visits / (ITEMS_PER_WORKER * workers.count() as u32)).max(1) };
        //the items (lines in a buffer of edges), published by the caller while the others take them
        let items: Mutex<(Vec<Share>, Vec<u32>)> = Mutex::new((Vec::new(), Vec::new()));
        let (published, shared_out) = (AtomicUsize::new(0), AtomicBool::new(false));
        let outputs: Vec<Mutex<Output<'c>>> = (0..workers.count()).map(|_| Mutex::new(Output::default())).collect();
        let next = AtomicUsize::new(0);
        workers.run(&|worker| {
            let mut output = outputs[worker].lock().expect("a worker's output");
            let mut scratch = Scratch { keys: self.keys.clone(), ..Scratch::default() };
            if worker == 0 {
                //set however share_out ends, so that no worker waits on for items
                let _shared_out = SetOnDrop(&shared_out);
                let sharing = Instant::now();
                self.share_out(self.root_index() as u32, visits, most, settings, &mut Scratch::default(), &mut |line, node, visits| {
                    let (shares, edges) = &mut *items.lock().expect("the pass's items");
                    let start = edges.len();
                    edges.extend_from_slice(line);
                    shares.push(Share { line: start..edges.len(), node, visits });
                    published.store(shares.len(), Ordering::Release);
                });
                output.share_out = sharing.elapsed();
            }
            loop {
                let item = next.fetch_add(1, Ordering::Relaxed);
                //the item, once it is published, unless the pass has fewer
                while published.load(Ordering::Acquire) <= item && !shared_out.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                if published.load(Ordering::Acquire) <= item {
                    break;
                }
                let share = {
                    let (shares, edges) = &*items.lock().expect("the pass's items");
                    scratch.line.clear();
                    scratch.line.extend_from_slice(&edges[shares[item].line.clone()]);
                    shares[item].clone()
                };
                let start = output.picks.len();
                let picking = Instant::now();
                self.pick_down(share.node, share.visits, settings, &mut scratch, &mut output);
                let preparing = Instant::now();
                output.picking += preparing - picking;
                let Output { edges, picks, moves, error, .. } = &mut *output;
                for pick in &mut picks[start..] {
                    if error.is_none() && self.to_prepare(pick.share.node) {
                        let line = &edges[pick.share.line.clone()];
                        if let Err(failed) = self.prepare(pick, line, settings, tables, root_covered, cache, moves, &mut scratch) {
                            *error = Some(failed);
                        }
                    }
                }
                output.preparing += preparing.elapsed();
                let end = output.picks.len();
                output.items.push((item, start..end));
            }
        });
        counts.workers += started.elapsed();
        let items = published.into_inner();
        counts.items += items as u64;

        let mut outputs: Vec<Output<'c>> = outputs.into_iter().map(|output| output.into_inner().expect("a worker's output")).collect();
        counts.share_out += outputs[0].share_out;
        counts.picking += outputs.iter().map(|output| output.picking).sum();
        counts.preparing += outputs.iter().map(|output| output.preparing).sum();
        if let Some(error) = outputs.iter_mut().find_map(|output| output.error.take()) {
            return Err(error);
        }
        let mut order = vec![(0, 0..0); items];
        for (worker, output) in outputs.iter().enumerate() {
            for (item, picks) in &output.items {
                order[*item] = (worker, picks.clone());
            }
        }
        Ok((outputs, order))
    }

    //collisions on the path, as many as the round's limit has room for
    fn collide(&mut self, path: &[usize], visits: u32, limit: u32, state: &mut RoundState, counts: &mut BatchCounts) {
        let taken = visits.min(limit - state.collided);
        state.capped |= taken < visits;
        if taken > 0 {
            for &node in path {
                self.nodes[node].in_flight += taken;
            }
            let start = state.collided_nodes.len();
            state.collided_nodes.extend_from_slice(path);
            state.collisions.push((start..state.collided_nodes.len(), taken));
            state.collided += taken;
            counts.collisions += u64::from(taken);
        }
    }

    //a pick applied as gather ends a descent: its node made, its result backed up, or its leaf left waiting
    #[allow(clippy::too_many_arguments)]
    fn apply_pick(
        &mut self,
        pick: Pick<'_>,
        line: &[u32],
        moves: &[Move],
        round: &Round,
        settings: &SearchSettings,
        boards: &mut Vec<u8>,
        path: &mut Vec<usize>,
        state: &mut RoundState,
        counts: &mut BatchCounts,
    ) {
        let root = self.root_index();
        path.clear();
        path.push(root);
        for &edge in line {
            let edge = edge as usize;
            if self.children[edge] == NO_NODE {
                self.children[edge] = self.nodes.len() as u32;
                self.nodes.push(Node::new());
                self.node_keys.push(pick.key);
            }
            path.push(self.children[edge] as usize);
        }
        let node = *path.last().expect("the root at least");
        let visits = pick.share.visits;
        let move_list = |range: Range<usize>| -> MoveList { moves[range].iter().copied().collect() };
        match pick.leaf {
            None => match self.nodes[node].proof {
                Some(proof) => {
                    for _ in 0..visits {
                        self.backup(path, 1.0 - proof.value, 0);
                    }
                    state.simulations += visits;
                    counts.scored += u64::from(visits);
                }
                None => self.collide(path, visits, round.collisions, state, counts),
            },
            Some(Leaf::Scored { value, game_over, no_moves }) => {
                if let Some(result) = game_over
                    && settings.certainty
                    && node != root
                {
                    let (outcome, plies) = if result == 0.0 {
                        (Outcome::Win, if no_moves { 0 } else { TABLEBASE_PLIES })
                    } else if result == 1.0 {
                        (Outcome::Loss, TABLEBASE_PLIES)
                    } else {
                        (Outcome::Draw, 0)
                    };
                    self.prove(path, Proof { outcome, value: 1.0 - result, plies });
                }
                for _ in 0..visits {
                    self.backup(path, value, 0);
                }
                if settings.certainty {
                    self.propagate_proofs(path);
                }
                state.simulations += visits;
                counts.scored += u64::from(visits);
            }
            Some(Leaf::Cached { moves, answer }) => {
                self.expand(node, &move_list(moves), answer.priors);
                self.backup(path, f64::from(answer.value), 0);
                state.simulations += 1;
                counts.cached += 1;
                counts.unspent += u64::from(visits - 1);
            }
            Some(Leaf::Network { moves, turn, tokens }) => {
                boards.extend_from_slice(&tokens);
                self.add_in_flight(path);
                self.round_leaves(round.id).push_back(PendingLeaf { path: path.clone(), key: pick.key, moves: move_list(moves), turn });
                state.gathered += 1;
                state.simulations += 1;
                counts.leaves += 1;
                self.collide(path, visits - 1, round.collisions, state, counts);
            }
        }
    }

    //make_room for `count` more simulations at once: their nodes, and the moves of the leaves waiting and theirs
    fn make_room_for(&mut self, count: usize, memory_limit: usize) -> bool {
        if !node_fits(self.nodes.len() + count - 1) {
            return false;
        }
        if memory_limit == usize::MAX {
            return true;
        }
        let spare = memory_limit.saturating_sub(self.allocated_bytes());
        let node_bytes = size_of::<Node>() + size_of::<u64>();
        let Some(nodes) = grown_capacity(self.nodes.capacity().min(self.node_keys.capacity()), self.nodes.len() + count, node_bytes, spare)
        else {
            return false;
        };
        reserve_to(&mut self.nodes, nodes);
        reserve_to(&mut self.node_keys, nodes);
        let spare = memory_limit.saturating_sub(self.allocated_bytes());
        let edge_bytes = size_of::<Move>() + size_of::<f32>() + size_of::<u32>();
        let capacity = self.moves.capacity().min(self.priors.capacity()).min(self.children.capacity());
        let Some(edges) = grown_capacity(capacity, self.moves.len() + (self.pending() + count) * MOST_MOVES, edge_bytes, spare) else {
            return false;
        };
        reserve_to(&mut self.moves, edges);
        reserve_to(&mut self.priors, edges);
        reserve_to(&mut self.children, edges);
        true
    }

    //gather's round in passes of at most pass_visits visits (u32::MAX: all it has left); returns its leaves
    #[allow(clippy::too_many_arguments)]
    pub fn gather_batch(
        &mut self,
        round: &Round,
        settings: &SearchSettings,
        tables: Option<&Tables>,
        cache: Option<&Cache>,
        boards: &mut Vec<u8>,
        workers: &Workers,
        pass_visits: u32,
        counts: &mut BatchCounts,
    ) -> Result<u32, SyzygyError> {
        let root_covered = tables.is_some_and(|tables| tables.covers(&self.board));
        let mut state = RoundState::default();
        let mut path = Vec::new();
        counts.rounds += 1;
        let gathered = loop {
            let waiting = self.root().visit_count as usize + self.pending();
            let room = (round.visit_limit as usize).saturating_sub(waiting).min(u32::MAX as usize) as u32;
            let visits = (round.leaves.saturating_sub(state.gathered))
                .min(round.simulations.saturating_sub(state.simulations))
                .min(room)
                .min(pass_visits);
            if visits == 0 {
                break Ok(state.gathered);
            }
            //the root's own evaluation always fits, so that a full search still has moves to report
            if self.root().has_edges() && !self.make_room_for(visits as usize, round.memory_limit) {
                self.full = true;
                break Ok(state.gathered);
            }
            let (outputs, order) = match self.pick_pass(visits, settings, tables, root_covered, cache, workers, counts) {
                Ok(pass) => pass,
                Err(error) => break Err(error),
            };
            let applying = Instant::now();
            counts.passes += 1;
            //a worker took its items in their order, so its picks come in the tree's order
            let mut buffers = Vec::with_capacity(outputs.len());
            let mut picks = Vec::with_capacity(outputs.len());
            for output in outputs {
                buffers.push((output.edges, output.moves));
                picks.push(output.picks.into_iter());
            }
            for (worker, item_picks) in order {
                let (edges, moves) = &buffers[worker];
                for _ in item_picks {
                    let pick = picks[worker].next().expect("the item's pick");
                    let line = &edges[pick.share.line.clone()];
                    self.apply_pick(pick, line, moves, round, settings, boards, &mut path, &mut state, counts);
                    counts.picks += 1;
                }
            }
            counts.apply += applying.elapsed();
            if state.capped {
                break Ok(state.gathered);
            }
        };
        for (nodes, visits) in state.collisions {
            for &node in &state.collided_nodes[nodes] {
                self.nodes[node].in_flight -= visits;
            }
        }
        gathered
    }

    //finish_leaf for the round's `count` oldest leaves: softmaxes on the workers, then expansions and backups in order
    #[allow(clippy::too_many_arguments)]
    pub fn finish_batch(
        &mut self,
        round: u64,
        count: usize,
        answer: &(dyn Fn(usize, usize) -> f32 + Sync),
        answer_width: usize,
        move_ids: &MoveIds,
        mut cache: Option<&mut Cache>,
        settings: &SearchSettings,
        workers: &Workers,
    ) {
        //each worker's leaves (index, priors in its buffer, value), and the next chunk of leaves to take
        let answers: Vec<Mutex<(Vec<(usize, Range<usize>, f32)>, Vec<f32>)>> = (0..workers.count()).map(|_| Mutex::default()).collect();
        let next = AtomicUsize::new(0);
        let chunk = count.div_ceil(4 * workers.count()).max(1);
        if count == 0 {
            return;
        }
        let slot =
            self.pending.iter().position(|(id, leaves)| *id == round && !leaves.is_empty()).expect("the round's waiting simulations");
        let pending = &self.pending[slot].1;
        workers.run(&|worker| {
            let (leaves, priors) = &mut *answers[worker].lock().expect("a worker's answers");
            loop {
                let start = next.fetch_add(chunk, Ordering::Relaxed);
                if start >= count {
                    break;
                }
                for (index, leaf) in pending.range(start..(start + chunk).min(count)).enumerate() {
                    let first = priors.len();
                    priors.extend(leaf.moves.iter().map(|&chess_move| answer(start + index, move_ids.id(chess_move, leaf.turn))));
                    softmax(&mut priors[first..]);
                    let value = sharpen(answer(start + index, answer_width - 1), settings.value_sharpness);
                    leaves.push((start + index, first..priors.len(), value));
                }
            }
        });
        let answers: Vec<_> = answers.into_iter().map(|answers| answers.into_inner().expect("a worker's answers")).collect();
        let mut by_leaf = vec![(0, 0..0, 0.0); count];
        for (worker, (leaves, _)) in answers.iter().enumerate() {
            for (index, priors, value) in leaves {
                by_leaf[*index] = (worker, priors.clone(), *value);
            }
        }
        for (worker, priors, value) in by_leaf {
            let leaf = self.pending[slot].1.pop_front().expect("a waiting simulation");
            let priors = &answers[worker].1[priors];
            let node = *leaf.path.last().expect("the root at least");
            self.expand(node, &leaf.moves, priors);
            self.backup(&leaf.path, f64::from(value), 1);
            if let Some(cache) = cache.as_deref_mut() {
                cache.insert(leaf.key, priors, value);
            }
        }
    }
}

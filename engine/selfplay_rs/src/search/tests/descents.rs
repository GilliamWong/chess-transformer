//the reference for Tree::gather: the round's descents made one after another, each from the root, as gather was before it sent
//them down in runs. gather must leave the tree this leaves, number for number, and the same boards in the same order
use super::*;

//why a descent stopped above a node: a repeat on its path, or a move's first visit of a node other paths made, which takes the
//node's mean without going on (Tree::stops_at)
enum Stop {
    Repeat,
    FirstVisit,
}

impl Tree {
    pub fn gather_by_descents(
        &mut self,
        round: &Round,
        settings: &SearchSettings,
        tables: Option<&Tables>,
        cache: Option<&Cache>,
        boards: &mut Vec<u8>,
    ) -> u32 {
        let (mut simulations, mut gathered) = (0, 0);
        let mut collisions: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
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
                if self.graph.is_some() && self.known_pass_due(node) {
                    self.link_known_moves(node, &board);
                }
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
                    } else if self.nodes[node].has_edges() && self.nodes[node].proof.is_none() {
                        if self.stops_at(link) {
                            stop = Some(Stop::FirstVisit);
                        } else {
                            self.take_mean(path[path.len() - 2], link, node);
                        }
                    }
                }
                self.keys.push(self.node_keys[node]);
                if self.nodes[node].proof.is_some() || stop.is_some() {
                    break;
                }
            }

            if let Some(stop) = stop {
                self.keys.truncate(root_keys);
                simulations += 1;
                //a repeat is a visit of the node at a draw's score for the side that moved into it (selection's for the move),
                //as in lc0: the move's mean is its node's, whatever the path
                if matches!(stop, Stop::Repeat) {
                    self.count_own_visit(node, 1.0 - settings.draw);
                }
                path.pop();
                self.back_up_means(&path, &edges, None, 0);
                continue;
            }

            if let Some(proof) = self.nodes[node].proof {
                self.keys.truncate(root_keys);
                simulations += 1;
                self.back_up(&path, &edges, 1.0 - proof.value, 0);
                //with transpositions this parent may not have met the proof, made on another path
                if self.graph.is_some() && settings.certainty {
                    self.propagate_proofs(&path);
                }
                self.learn_proofs(&path, &edges);
                continue;
            }

            if self.nodes[node].in_flight > 0 {
                self.keys.truncate(root_keys);
                if collisions.len() == round.collisions as usize {
                    break;
                }
                self.add_in_flight(&path, &edges, 1);
                collisions.push((path.clone(), edges.clone()));
                continue;
            }

            let mut legal_moves = board.legal_moves();
            let node_tables = tables.filter(|_| node != self.root && (!root_covered || board.halfmoves() == 0));
            //the root is searched whatever the rules say of it: Searches.start refuses only a mate or stalemate
            let game_over =
                if node == self.root { None } else { game_result(&board, &self.keys, &legal_moves, settings.draw, node_tables) };
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
                    self.learn_proofs(&path, &edges);
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
                    self.add_in_flight(&path, &edges, 1);
                    let leaf = PendingLeaf { path: path.clone(), edges: edges.clone(), key: leaf_key, moves, turn: board.turn() };
                    self.round_leaves(round.id).push_back(leaf);
                    gathered += 1;
                }
            }
        }
        for (path, edges) in collisions {
            for &node in self.in_flight_nodes(&path) {
                self.nodes[node].in_flight -= 1;
            }
            for (&node, &link) in path.iter().zip(&edges) {
                let link = self.current_link(node, link as u32);
                self.slot_mut(link).pending -= 1;
            }
        }
        gathered
    }
}

//a round's limits for the comparisons: a leaf for every 16 visits of the root up to `leaves`, as the engine grows its rounds, so
//many simulations a leaf, and collisions (None: as many as its leaves, the engine's)
#[derive(Clone, Copy)]
struct Limits {
    leaves: u32,
    simulations_a_leaf: u32,
    collisions: Option<u32>,
    memory_limit: usize,
}

//the engine's round beside another round that is out, and without one
const BUSY: Limits = Limits { leaves: 64, simulations_a_leaf: 64, collisions: None, memory_limit: usize::MAX };
const IDLE: Limits = Limits { simulations_a_leaf: 3, ..BUSY };

impl Limits {
    fn round(&self, tree: &Tree, id: u64, visit_limit: u32) -> Round {
        let leaves = (tree.root().visit_count / 16).clamp(1, self.leaves);
        Round {
            id,
            leaves,
            simulations: self.simulations_a_leaf * leaves,
            collisions: self.collisions.unwrap_or(leaves),
            visit_limit,
            memory_limit: self.memory_limit,
        }
    }
}

//what a tree's nodes, moves and waiting leaves hold, for a comparison after every gather (assert_same_trees is for the end)
type Numbers = (Vec<(u32, u32, u64, u64, u64, bool, (u8, u8))>, Vec<Vec<(usize, u32, u32, (u32, u16, bool, u8))>>, Vec<(u64, u64, Vec<usize>, Vec<usize>)>, bool);

fn numbers(tree: &Tree) -> Numbers {
    let nodes = tree.nodes.iter().map(|node| {
        let sums = (node.value_sum.to_bits(), node.visited_policy_sqrt.to_bits());
        (node.visit_count, node.in_flight, sums.0, sums.1, node.edges_at, node.proof.is_some(), (node.known_passes, node.known_to))
    });
    let waiting = tree.pending.iter().flat_map(|(round, leaves)| leaves.iter().map(|leaf| (*round, leaf.key, leaf.path.clone(), leaf.edges.clone())));
    (nodes.collect(), moves_with_nodes(tree), waiting.collect(), tree.full)
}

//the counts of waiting descents recounted from the leaves that wait: once a gathering call has taken its collisions off, a
//node's in_flight and a move's pending are exactly the waiting leaves that count them, and none when no leaf waits. It would
//catch a collision's count left behind, or a leaf's taken off twice
fn assert_waiting_counts(tree: &Tree) {
    let mut at_nodes = vec![0u32; tree.nodes.len()];
    let mut at_moves: HashMap<u32, u16> = HashMap::new();
    for leaf in tree.pending.iter().flat_map(|(_, leaves)| leaves) {
        for &node in tree.in_flight_nodes(&leaf.path) {
            at_nodes[node] += 1;
        }
        for (&node, &link) in leaf.path.iter().zip(&leaf.edges) {
            *at_moves.entry(tree.current_link(node, link as u32)).or_default() += 1;
        }
    }
    for (node, state) in tree.nodes.iter().enumerate() {
        assert_eq!(state.in_flight, at_nodes[node], "node {node}: the descents that wait at it");
        for (place, link) in tree.linked(node) {
            let waiting = at_moves.get(&link).copied().unwrap_or(0);
            assert_eq!(tree.slot(link).pending, waiting, "node {node}, move {place}: the descents that wait through it");
        }
    }
}

//two copies of the tree searched to `visits` in the same rounds, with `in_flight` rounds out, one by gather and one by
//gather_by_descents: after every gather the two hold the same boards, numbers and waiting leaves, and gather's counts of
//waiting descents are its waiting leaves. `noise` mixes root noise into both once the root has its moves. Returns gather's
//tree and the rounds gathered
#[allow(clippy::too_many_arguments)]
fn searched_alike(
    start: &Tree,
    settings: &SearchSettings,
    tables: Option<&Tables>,
    limits: Limits,
    visits: u32,
    in_flight: usize,
    noise: bool,
    what: &str,
) -> (Tree, usize) {
    use rand::SeedableRng;
    let (mut by_descents, mut by_runs) = (copied(start), copied(start));
    let (mut descents_network, mut runs_network) = (Network::new(), Network::new());
    let (mut out, mut gathered, mut noised) = (VecDeque::new(), 0, false);
    loop {
        if noise && !noised && by_runs.root().has_edges() && by_runs.pending() == 0 {
            for tree in [&mut by_descents, &mut by_runs] {
                tree.add_root_noise(&mut ChaCha8Rng::seed_from_u64(7), 0.3, 0.25);
            }
            noised = true;
        }
        let room = by_runs.root().visit_count as usize + by_runs.pending() < visits as usize && !by_runs.full;
        if room && out.len() < in_flight {
            let round = limits.round(&by_runs, gathered as u64, visits);
            let (mut descents_boards, mut runs_boards) = (Vec::new(), Vec::new());
            let leaves = by_descents.gather_by_descents(&round, settings, tables, Some(&descents_network.cache), &mut descents_boards);
            let found = by_runs.gather(&round, settings, tables, Some(&runs_network.cache), &mut runs_boards);
            assert_eq!(found, leaves, "{what}, round {gathered}: the leaves gathered");
            assert!(runs_boards == descents_boards, "{what}, round {gathered}: the leaves' boards");
            assert!(numbers(&by_runs) == numbers(&by_descents), "{what}, round {gathered}: the trees after the gather");
            assert_waiting_counts(&by_runs);
            gathered += 1;
            out.push_back((round.id, runs_boards));
            continue;
        }
        let Some((id, boards)) = out.pop_front() else { break };
        finish_leaves(&mut by_descents, id, &boards, settings, &mut descents_network);
        finish_leaves(&mut by_runs, id, &boards, settings, &mut runs_network);
    }
    assert_same_trees(&by_descents, &by_runs);
    assert_waiting_counts(&by_runs);
    assert!(by_runs.nodes.iter().all(|node| node.in_flight == 0), "{what}: nothing waits once every leaf is back");
    (by_runs, gathered)
}

//the rules a search runs under in the comparisons: mcts.py's and the engine's transpositions
fn rules(certainty: bool) -> Vec<(&'static str, SearchSettings)> {
    vec![("no transpositions", settings(certainty)), ("transpositions", SearchSettings { transpositions: true, ..settings(certainty) })]
}

fn start_of(fen: &str, moves: &[&str], settings: &SearchSettings) -> Tree {
    tree_after(fen, moves).with_transpositions(settings.transpositions)
}

//gather sends descents down in runs; this is the claim it rests on, that the tree it leaves is the tree of one descent from the
//root after another, to the last bit and in every round: across the tests' positions (repeats, proofs, the tables' results),
//with and without transpositions, with one, two and sixteen rounds out, in the engine's rounds and in rounds
//whose collisions run out early or never. It would catch a run that is one descent too long or short, a result that the nodes
//above learnt of late, a mean taken in passing at another moment, a round that ends at another descent
#[test]
fn gather_leaves_the_tree_of_one_descent_after_another() {
    let tables = tables();
    let few = Limits { collisions: Some(3), ..BUSY };
    let many = Limits { collisions: Some(1000), ..BUSY };
    let (mut searches, mut rounds) = (0, 0);
    for (fen, moves, _, certainty, tables) in cases(&tables) {
        for (rule, settings) in rules(certainty) {
            let start = start_of(fen, moves, &settings);
            for (in_flight, limits, visits) in [(1, IDLE, 2_500), (2, BUSY, 2_500), (16, BUSY, 5_000), (2, few, 2_000), (16, many, 2_500)] {
                let what = format!("{fen}, {rule}, {in_flight} rounds out, collisions {:?}", limits.collisions);
                rounds += searched_alike(&start, &settings, tables, limits, visits, in_flight, false, &what).1;
                searches += 1;
            }
        }
    }
    println!("{searches} searches, {rounds} rounds, each the same by gather and by gather_by_descents");
}

//the same with root noise, whose priors are out of order (every move of the root is weighed), and under a memory limit that
//the search fills: the two end full at the same descent
#[test]
fn gather_is_the_descents_with_root_noise_and_at_a_memory_limit() {
    for (rule, settings) in rules(true) {
        let start = start_of(START, OPENING, &settings);
        for in_flight in [2, 16] {
            let what = format!("{rule}, {in_flight} rounds out, root noise");
            searched_alike(&start, &settings, None, BUSY, 2_500, in_flight, true, &what);
            let limited = Limits { memory_limit: 4_000_000, ..BUSY };
            let what = format!("{rule}, {in_flight} rounds out, a memory limit");
            let (filled, _) = searched_alike(&start, &settings, None, limited, 1_000_000, in_flight, false, &what);
            assert!(filled.full && filled.allocated_bytes() <= limited.memory_limit, "{what}: the search fills its limit and stays within");
        }
    }
}

//a graph's sums hold after searches in the engine's rounds: a node's sum is its own scores and its moves' visits at the means
//they hold (assert_move_means), with two and sixteen rounds out. The searches use no cache, so that every
//expanded node's own score is its position's answer
#[test]
fn a_graphs_sums_hold_after_rounds_gathered_in_runs() {
    let network = Network::new();
    for (fen, moves) in [(START, OPENING), (LOCKED_PAWNS, &[][..]), (QUEEN_AND_PAWN, &[]), (ROOKS, &[])] {
        for in_flight in [2, 16] {
            let (mut tree, settings) = graph_tree(fen, moves);
            let (mut out, mut gathered) = (VecDeque::new(), 0u64);
            loop {
                if (tree.root().visit_count as usize + tree.pending() < 4_000) && out.len() < in_flight {
                    let mut boards = Vec::new();
                    tree.gather(&BUSY.round(&tree, gathered, 4_000), &settings, None, None, &mut boards);
                    out.push_back((gathered, boards));
                    gathered += 1;
                    continue;
                }
                let Some((id, boards)) = out.pop_front() else { break };
                for board in boards.as_chunks::<TOKENS_PER_BOARD>().0 {
                    let seed = board.iter().fold(0, |hash, &token| mix(hash ^ u64::from(token)));
                    tree.finish_leaf(id, |column| answer(board, seed, column), ANSWER_WIDTH, &network.move_ids, None, &settings);
                }
            }
            assert_move_means(&tree, &settings);
            assert_waiting_counts(&tree);
        }
    }
}

//more_in_a_row for the move select_edge takes at the node, against what it stands for: with one descent gone down the move,
//the further descents that select_edge sends down it, each finding one more waiting there. They are counted by letting them
//wait there one at a time and asking select_edge. None for a move without a node, which no run goes down
fn run_counted(tree: &Tree, settings: &SearchSettings, node: usize, most: u32, what: &str) -> Option<u32> {
    let (edge, link, scan, rivals) = tree.select_with_rivals(node, settings, None);
    assert_eq!((edge, link), tree.select_edge(node, settings, None), "{what}, node {node}: the move chosen");
    if link == NO_LINK || tree.link_child(link) == NO_NODE {
        return None;
    }
    let place = edge - tree.nodes[node].edges().start;
    assert_eq!(tree.rivals(node, &scan, None, place), Some(rivals), "{what}, node {node}: the move's rivals");
    let more = tree.more_in_a_row(&scan, None, (place, tree.priors[edge], link), rivals, most);
    let mut waiting = copied(tree);
    let mut counted = 0;
    loop {
        if settings.transpositions {
            waiting.slot_mut(link).pending += 1;
        } else {
            let child = waiting.link_child(link) as usize;
            waiting.nodes[child].in_flight += 1;
        }
        if counted == most || waiting.select_edge(node, settings, None).0 != edge {
            break;
        }
        counted += 1;
    }
    assert_eq!(more, counted, "{what}, node {node}: the descents that follow the first down its move");
    Some(more)
}

//a run is as long as select_edge would make it: at every node of searched trees, and at nodes made up so that the move
//taken comes level with another after some descents, where the earlier move of the two is select_edge's. It would catch a
//run cut at the wrong value, a tie given to the wrong move, a move left out of the rivals; and it says that runs happen
#[test]
fn more_in_a_row_counts_the_descents_select_edge_would_send_the_same_way() {
    const MOST: u32 = 40;
    let (mut nodes_asked, mut longest, mut runs, mut ties) = (0, 0, 0, [0; 2]);
    for (rule, settings) in rules(true) {
        for (fen, moves) in [(START, OPENING), (ROOKS, &[][..])] {
            let mut tree = start_of(fen, moves, &settings);
            search(&mut tree, 3_000, usize::MAX, &settings, None, &mut Network::new());
            for node in (0..tree.nodes.len()).filter(|&node| tree.nodes[node].has_edges() && tree.nodes[node].proof.is_none()) {
                if let Some(more) = run_counted(&tree, &settings, node, MOST, rule) {
                    nodes_asked += 1;
                    longest = longest.max(more);
                    runs += usize::from(more > 0);
                }
                //two moves of one prior that hold no mean, every other move with a node out of the way: the one with fewer
                //descents waiting is taken, and is level with the other once it has as many
                let first_edge = tree.nodes[node].edges().start;
                let with_node: Vec<(usize, u32)> = tree.linked(node).collect();
                let pair = with_node.iter().enumerate().find_map(|(at, &(place, link))| {
                    let same = |&&(other, _): &&(usize, u32)| tree.priors[first_edge + other].to_bits() == tree.priors[first_edge + place].to_bits();
                    with_node[at + 1..].iter().find(same).map(|&(_, later)| (link, later))
                });
                let Some((earlier, later)) = pair.filter(|_| settings.transpositions) else { continue };
                for (case, (waiting_earlier, waiting_later)) in [(3, 0), (0, 3)].into_iter().enumerate() {
                    let mut made_up = copied(&tree);
                    for &(_, link) in &with_node {
                        made_up.slot_mut(link).lost = true;
                    }
                    for (link, waiting) in [(earlier, waiting_earlier), (later, waiting_later)] {
                        let held = Slot { visits: 0, mean: NO_MEAN, pending: waiting, lost: false, ..*made_up.slot(link) };
                        *made_up.slot_mut(link) = held;
                    }
                    let taken = made_up.select_edge(node, &settings, None).1;
                    //a move without a node may outweigh both: then there is no tie to ask about
                    if taken == [later, earlier][case] {
                        let more = run_counted(&made_up, &settings, node, MOST, "a made-up tie").expect("a move with a node");
                        //the later move stops one short of the earlier's count, the earlier goes on to the later's; fewer
                        //if a move without a node comes between
                        ties[case] += usize::from(more == [2, 3][case]);
                    }
                }
            }
        }
    }
    assert!(runs > nodes_asked / 20 && longest == MOST, "{runs} of {nodes_asked} nodes send more than one descent the same way, {longest} at most");
    assert!(ties[0] > 0 && ties[1] > 0, "ties asked about: {ties:?}");
}

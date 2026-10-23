//tree reuse (Searches): play_line's one copy and drop_stale_leaves' walk against the moves played one at a time (play)
//and a walk that plays out every node's position, as trees were reused before. They must give the same tree, number for
//number, on trees the engine's rounds grow with a stand-in for the network
use super::*;
use crate::rules::{MOVE_ID_TABLE_LENGTH, TOKENS_PER_BOARD};
use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, EnPassantMode};
use std::fmt::Debug;

//encoding.NUM_MOVE_IDS, and an answer's numbers: the moves' logits, then the expected score
const MOVE_IDS: usize = 1858;
const ANSWER_WIDTH: usize = MOVE_IDS + 1;
//by shakmaty's role numbers, which the encoding's piece tokens are
const PIECE_VALUES: [f32; 7] = [0.0, 1.0, 3.0, 3.0, 5.0, 9.0, 0.0];

//splitmix64's finaliser
fn mix(value: u64) -> u64 {
    let mut mixed = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^ (mixed >> 31)
}

//a number in [0, 1) from the value's hash
fn uniform(value: u64) -> f32 {
    (mix(value) >> 40) as f32 / (1u64 << 24) as f32
}

//the stand-in network's answer for a board (seed its hash): a few moves much likelier than the rest, and an expected
//score that follows the material of the side to move
fn answer(board: &[u8], seed: u64, column: usize) -> f32 {
    if column == ANSWER_WIDTH - 1 {
        let material: f32 = board[..64]
            .iter()
            .map(|&token| match token as usize {
                0 => 0.0,
                own @ 1..=6 => PIECE_VALUES[own],
                other => -PIECE_VALUES[other - 6],
            })
            .sum();
        return 1.0 / (1.0 + (-(material + uniform(seed) - 0.5) / 3.0).exp());
    }
    8.0 * uniform(seed ^ column as u64).powi(2)
}

//the stand-in network's move ids (any id a column of its answer) and the search's cache of its answers
struct Network {
    move_ids: MoveIds,
    cache: Cache,
}

impl Network {
    fn new() -> Network {
        Network {
            move_ids: MoveIds::new((0..MOVE_ID_TABLE_LENGTH).map(|index| (index % MOVE_IDS) as i16).collect()),
            cache: Cache::new(1 << 20),
        }
    }
}

//engine_search.py's rounds until the root has its simulations or the tree is full: a leaf per 16 simulations up to 64, as
//many collisions, three simulations a leaf
fn search(
    tree: &mut Tree,
    simulations: u32,
    memory_limit: usize,
    settings: &SearchSettings,
    tables: Option<&Tables>,
    network: &mut Network,
) {
    let mut boards = Vec::new();
    while tree.root().visit_count < simulations && !(tree.full && tree.pending.is_empty()) {
        let leaves = (tree.root().visit_count / 16).clamp(1, 64);
        let round = Round { leaves, simulations: 3 * leaves, collisions: leaves, visit_limit: simulations, memory_limit };
        boards.clear();
        tree.gather(&round, settings, tables, Some(&network.cache), &mut boards).expect("the tables");
        for board in boards.as_chunks::<TOKENS_PER_BOARD>().0 {
            let seed = board.iter().fold(0, |hash, &token| mix(hash ^ u64::from(token)));
            tree.finish_leaf(|column| answer(board, seed, column), ANSWER_WIDTH, &network.move_ids, Some(&mut network.cache), settings);
        }
    }
}

fn tree_after(fen: &str, moves: &[&str]) -> Tree {
    let mut board: Chess = fen.parse::<Fen>().expect("a fen").into_position(CastlingMode::Standard).expect("a legal position");
    let mut keys = vec![position_key(&board)];
    for uci in moves {
        board.play_unchecked(uci.parse::<UciMove>().expect("a uci move").to_move(&board).expect("a legal move"));
        keys.push(position_key(&board));
    }
    Tree::new(board, keys)
}

//a copy of a tree whose search is done (nothing waits for the network)
fn copied(tree: &Tree) -> Tree {
    assert!(tree.pending.is_empty());
    Tree {
        board: tree.board.clone(),
        keys: tree.keys.clone(),
        nodes: tree.nodes.clone(),
        node_keys: tree.node_keys.clone(),
        moves: tree.moves.clone(),
        priors: tree.priors.clone(),
        noised_priors: tree.noised_priors.clone(),
        children: tree.children.clone(),
        pending: VecDeque::new(),
        logits: Vec::new(),
        full: tree.full,
    }
}

//drop_stale_leaves' walk as it was, every node's position played out on the way down: the paths to the stale leaves
//under the path's last node, whose position is board
fn stale_leaves_by_positions(
    tree: &mut Tree,
    board: &Chess,
    path: &mut Vec<usize>,
    settings: &SearchSettings,
    tables: Option<&Tables>,
    root_covered: bool,
    stale: &mut Vec<Vec<usize>>,
) {
    let node = *path.last().expect("the root at least");
    let edges = tree.nodes[node].edges();
    if edges.is_empty() {
        if node != 0 && tree.nodes[node].visit_count > 0 {
            let legal_moves = board.legal_moves();
            let node_tables = tables.filter(|_| !root_covered || board.halfmoves() == 0);
            let over = game_result(board, &tree.keys, &legal_moves, settings.draw, settings.repetitions_for_a_draw, node_tables)
                .expect("the tables");
            let (&leaf_key, path_keys) = tree.keys[tree.keys.len() - path.len()..].split_last().expect("the leaf's key");
            if over.is_none() && !path_keys.contains(&leaf_key) {
                stale.push(path.clone());
            }
        }
        return;
    }
    for edge in edges {
        let child = tree.children[edge];
        if child == NO_NODE || tree.nodes[child as usize].proof.is_some() {
            continue;
        }
        let mut child_board = board.clone();
        child_board.play_unchecked(tree.moves[edge]);
        path.push(child as usize);
        tree.keys.push(tree.node_keys[child as usize]);
        stale_leaves_by_positions(tree, &child_board, path, settings, tables, root_covered, stale);
        tree.keys.pop();
        path.pop();
    }
}

//Searches' tree reuse as it was: the moves played one at a time, each copying the subtree, then the stale leaves
//taken out. Returns how many were
fn reused_move_by_move(tree: &mut Tree, moves: &[Move], settings: &SearchSettings, tables: Option<&Tables>) -> usize {
    for &played in moves {
        let edge = tree.root().edges().find(|&edge| tree.chess_move(edge) == played).expect("the line's move");
        tree.play(edge, true);
    }
    let root_covered = tables.is_some_and(|tables| tables.covers(&tree.board));
    let mut stale = Vec::new();
    let board = tree.board.clone();
    stale_leaves_by_positions(tree, &board, &mut vec![0], settings, tables, root_covered, &mut stale);
    for path in &stale {
        tree.take_out(path);
    }
    stale.len()
}

fn assert_same<T: PartialEq + Debug>(what: &str, old: &[T], new: &[T]) {
    assert_eq!(old.len(), new.len(), "the number of {what}");
    if let Some(index) = (0..old.len()).find(|&index| old[index] != new[index]) {
        panic!("{what} {index}: {:?} move by move, {:?} at once", old[index], new[index]);
    }
}

//every number of the two trees, the floats by their bits
fn assert_same_trees(old: &Tree, new: &Tree) {
    let fen = |tree: &Tree| Fen::from_position(&tree.board, EnPassantMode::Legal).to_string();
    assert_eq!(fen(old), fen(new));
    assert_same("keys", &old.keys, &new.keys);
    let numbers = |tree: &Tree| -> Vec<_> {
        tree.nodes
            .iter()
            .map(|node| {
                let proof = node.proof.map(|proof| (proof.outcome, proof.value.to_bits(), proof.plies));
                let (value_sum, visited_policy_sqrt) = (node.value_sum.to_bits(), node.visited_policy_sqrt.to_bits());
                (node.visit_count, node.in_flight, value_sum, visited_policy_sqrt, node.first_edge, node.edge_count, proof)
            })
            .collect()
    };
    assert_same("nodes", &numbers(old), &numbers(new));
    assert_same("node keys", &old.node_keys, &new.node_keys);
    assert_same("moves", &old.moves, &new.moves);
    let bits = |priors: &[f32]| -> Vec<u32> { priors.iter().map(|prior| prior.to_bits()).collect() };
    assert_same("priors", &bits(&old.priors), &bits(&new.priors));
    assert_same("children", &old.children, &new.children);
}

//lines a game may have played since a search, for which Searches reuses the tree (every move has its node, and the
//last node is expanded): the expected move, the expected two and three, and the expected move answered by the reply
//searched second most
fn reusable_lines(tree: &Tree) -> Vec<Vec<usize>> {
    let expected = tree.principal_variation();
    let mut lines: Vec<Vec<usize>> = (1..=expected.len().min(3)).map(|plies| expected[..plies].to_vec()).collect();
    if let Some(reply_node) = tree.child(expected[0]) {
        let mut replies: Vec<usize> = reply_node.edges().filter(|&edge| tree.visits(edge) > 0).collect();
        replies.sort_by_key(|&edge| std::cmp::Reverse(tree.visits(edge)));
        lines.extend(replies.get(1).map(|&reply| vec![expected[0], reply]));
    }
    lines.retain(|line| {
        line.iter().all(|&edge| tree.children[edge] != NO_NODE)
            && tree.nodes[tree.children[*line.last().expect("a move")] as usize].edge_count > 0
    });
    lines
}

//mcts.py's constants with the engine's value sharpening
fn settings(certainty: bool) -> SearchSettings {
    SearchSettings {
        c_puct: 1.745 / 2.0,
        c_puct_factor: 3.894 / 2.0,
        c_puct_base: 38739.0,
        fpu_reduction: 0.33 / 2.0,
        draw: 0.5,
        repetitions_for_a_draw: 3,
        value_sharpness: 3.0,
        certainty,
    }
}

//the syzygy tables of up to 5 pieces, None without them
fn tables() -> Option<Tables> {
    let folder = concat!(env!("CARGO_MANIFEST_DIR"), "/../../artifacts/engines/syzygy");
    std::path::Path::new(folder).is_dir().then(|| Tables::open(folder, 5).expect("the tables"))
}

//a middlegame after its opening, a rook ending and kings behind locked pawns whose shuffles repeat positions (the draws
//that go stale), a mate in two that certainty proves, and a rook and pawn ending whose captures the tables score
#[test]
fn a_line_played_at_once_leaves_the_tree_its_moves_played_one_at_a_time_leave() {
    const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    const OPENING: &[&str] = &["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7", "e2e3", "e8g8", "g1f3", "b8d7"];
    const ROOKS: &str = "8/8/4k3/8/8/4K3/R7/7r w - - 0 1";
    const LOCKED_PAWNS: &str = "4k3/8/8/1p1p1p1p/1P1P1P1P/8/8/4K3 w - - 0 1";
    const PAWNS_AND_ROOKS: &str = "8/8/4k3/3p4/3P4/4K3/R7/7r w - - 0 1";
    let tables = tables();
    let mut cases = vec![
        (START, OPENING, 20_000, true, None),
        (ROOKS, &[][..], 10_000, true, None),
        (ROOKS, &[], 10_000, false, None),
        (LOCKED_PAWNS, &[], 10_000, true, None),
        ("k7/8/2K5/8/8/8/8/7R w - - 0 1", &[], 10_000, true, None),
    ];
    match &tables {
        Some(tables) => {
            cases.extend([(PAWNS_AND_ROOKS, &[][..], 10_000, false, Some(tables)), (PAWNS_AND_ROOKS, &[], 10_000, true, Some(tables))])
        }
        None => eprintln!("no syzygy tables: the ending they score is left out"),
    }

    let (mut lines, mut stale, mut proven) = (0, 0, 0);
    for (fen, moves, simulations, certainty, tables) in cases {
        let settings = settings(certainty);
        let mut tree = tree_after(fen, moves);
        search(&mut tree, simulations, usize::MAX, &settings, tables, &mut Network::new());
        for line in reusable_lines(&tree) {
            let mut old = copied(&tree);
            let line_moves: Vec<Move> = line.iter().map(|&edge| tree.chess_move(edge)).collect();
            stale += reused_move_by_move(&mut old, &line_moves, &settings, tables);
            let mut new = copied(&tree);
            new.play_line(&line);
            new.drop_stale_leaves(&settings, tables).expect("the tables");
            assert_same_trees(&old, &new);
            lines += 1;
            proven += new.nodes.iter().filter(|node| node.proof.is_some()).count();
        }
    }
    assert!(lines >= 15 && stale > 0 && proven > 0, "{lines} lines, {stale} stale leaves, {proven} proven nodes");
}

//the answer cache keeps its two latest generations and a position's latest answer, and turns a generation over without
//taking time out of the search (dropping a generation's boxed answers there stalled it)
#[test]
fn the_cache_keeps_two_generations_and_turns_one_over_at_once() {
    let generation = 1_000_000u64;
    let mut cache = Cache::new(2 * generation as usize);
    let priors = |key: u64| -> Vec<f32> { (0..37).map(|index| (key + index) as f32).collect() };
    let mut slowest_turnover = std::time::Duration::ZERO;
    for key in 0..=3 * generation {
        let started = std::time::Instant::now();
        cache.insert(key, &priors(key), key as f32);
        if key >= 2 * generation && key % generation == 0 {
            slowest_turnover = slowest_turnover.max(started.elapsed());
        }
    }
    cache.insert(3 * generation, &priors(7), 7.0);

    let found = |key: u64| cache.get(key).map(|answer| (answer.priors.to_vec(), answer.value));
    assert_eq!(found(2 * generation), Some((priors(2 * generation), (2 * generation) as f32)));
    assert_eq!(found(3 * generation - 1), Some((priors(3 * generation - 1), (3 * generation - 1) as f32)));
    assert_eq!(found(3 * generation), Some((priors(7), 7.0)));
    assert_eq!(found(2 * generation - 1), None);
    assert!(slowest_turnover.as_millis() < 20, "a turnover took {slowest_turnover:?}");
}

//the edges' u32 indices: gathering stops while the moves of the leaves waiting and of one more might not fit
#[test]
fn the_edges_stop_short_of_their_u32_indices() {
    let most = u32::MAX as usize;
    assert!(edges_fit(most - 2 * MOST_MOVES, 1));
    assert!(!edges_fit(most - 2 * MOST_MOVES + 1, 1));
    assert!(!edges_fit(most, 0));
    assert_eq!(grown_capacity(8, 9, 13, usize::MAX), Some(16));
    assert_eq!(grown_capacity(8, 40, 13, usize::MAX), Some(40));
    assert_eq!(grown_capacity(8, 9, 13, 13 * 12), Some(12));
    assert_eq!(grown_capacity(8, 9, 13, 13 * 8), None);
}

//a search under a memory limit ends once its tree has no room left, its vectors within the limit (a growth's copy
//included, which leaves the vectors half to two thirds of it), and until then it is the search without a limit, node
//for node
#[test]
fn a_search_ends_at_its_memory_limit_as_the_same_search() {
    let memory_limit = 4_000_000;
    let settings = settings(true);
    let mut limited = tree_after("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", &[]);
    search(&mut limited, 1_000_000, memory_limit, &settings, None, &mut Network::new());
    let reached = limited.root().visit_count;
    assert!(limited.full && reached < 1_000_000 && limited.allocated_bytes() <= memory_limit, "{reached} {}", limited.allocated_bytes());

    let mut unlimited = tree_after("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", &[]);
    search(&mut unlimited, reached, usize::MAX, &settings, None, &mut Network::new());
    assert_same_trees(&unlimited, &limited);
    assert!(!unlimited.full && limited.allocated_bytes() > memory_limit / 3);
}

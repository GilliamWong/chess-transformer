use super::*;
use crate::rules::{MOVE_ID_TABLE_LENGTH, TOKENS_PER_BOARD};
use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, EnPassantMode};
use std::fmt::Debug;

mod bench;
mod descents;
mod shared;

//model.encoding.NUM_MOVE_IDS
const MOVE_IDS: usize = 1858;
const ANSWER_WIDTH: usize = MOVE_IDS + 1;
const ANSWER_WIDTH_LAST: usize = ANSWER_WIDTH - 1;
//indexed by shakmaty's role numbers, which the piece tokens are
const PIECE_VALUES: [f32; 7] = [0.0, 1.0, 3.0, 3.0, 5.0, 9.0, 0.0];

//splitmix64's finaliser
fn mix(value: u64) -> u64 {
    let mut mixed = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^ (mixed >> 31)
}

fn uniform(value: u64) -> f32 {
    (mix(value) >> 40) as f32 / (1u64 << 24) as f32
}

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

//engine_search.py's round: a leaf per 16 visits up to 64, as many collisions, three simulations a leaf
fn new_round(tree: &Tree, id: u64, visit_limit: u32, memory_limit: usize) -> Round {
    let leaves = (tree.root().visit_count / 16).clamp(1, 64);
    Round { id, leaves, simulations: 3 * leaves, collisions: leaves, visit_limit, memory_limit }
}

fn finish_leaves(tree: &mut Tree, id: u64, boards: &[u8], settings: &SearchSettings, network: &mut Network) {
    for board in boards.as_chunks::<TOKENS_PER_BOARD>().0 {
        let seed = board.iter().fold(0, |hash, &token| mix(hash ^ u64::from(token)));
        tree.finish_leaf(id, |column| answer(board, seed, column), ANSWER_WIDTH, &network.move_ids, Some(&mut network.cache), settings);
    }
}

//the leaves finished from their move logits in the order of their moves, as the engine's lean rows give them
fn finish_leaves_in_order(tree: &mut Tree, id: u64, boards: &[u8], settings: &SearchSettings, network: &mut Network) {
    for board in boards.as_chunks::<TOKENS_PER_BOARD>().0 {
        let seed = board.iter().fold(0, |hash, &token| mix(hash ^ u64::from(token)));
        let (moves, turn) = tree.round_moves(id).next().expect("a waiting leaf");
        let logits: Vec<f32> = moves.iter().map(|&chess_move| answer(board, seed, network.move_ids.id(chess_move, turn))).collect();
        let value = answer(board, seed, ANSWER_WIDTH_LAST);
        tree.finish_leaf_in_order(id, logits.into_iter(), value, Some(&mut network.cache), settings);
    }
}

fn search(
    tree: &mut Tree,
    simulations: u32,
    memory_limit: usize,
    settings: &SearchSettings,
    tables: Option<&Tables>,
    network: &mut Network,
) {
    search_with(finish_leaves, tree, simulations, memory_limit, settings, tables, network);
}

fn search_with(
    finish: fn(&mut Tree, u64, &[u8], &SearchSettings, &mut Network),
    tree: &mut Tree,
    simulations: u32,
    memory_limit: usize,
    settings: &SearchSettings,
    tables: Option<&Tables>,
    network: &mut Network,
) {
    let mut boards = Vec::new();
    while tree.root().visit_count < simulations && !(tree.full && tree.pending() == 0) {
        let round = new_round(tree, 0, simulations, memory_limit);
        boards.clear();
        tree.gather(&round, settings, tables, Some(&network.cache), &mut boards);
        finish(tree, round.id, &boards, settings, network);
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

fn copied(tree: &Tree) -> Tree {
    assert_eq!(tree.pending(), 0);
    Tree {
        board: tree.board.clone(),
        keys: tree.keys.clone(),
        nodes: tree.nodes.clone(),
        root: tree.root,
        node_keys: tree.node_keys.clone(),
        moves: tree.moves.clone(),
        priors: tree.priors.clone(),
        noised_priors: tree.noised_priors.clone(),
        links: tree.links.clone(),
        slots: tree.slots.clone(),
        memory_limit: tree.memory_limit,
        pending: Vec::new(),
        logits: Vec::new(),
        full: tree.full,
        graph: tree.graph.clone(),
    }
}

//find_stale_leaves' reference: plays out every node's position
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
            let over = game_result(board, &tree.keys, &legal_moves, settings.draw, node_tables);
            let (&leaf_key, path_keys) = tree.keys[tree.keys.len() - path.len()..].split_last().expect("the leaf's key");
            if over.is_none() && !path_keys.contains(&leaf_key) {
                stale.push(path.clone());
            }
        }
        return;
    }
    for edge in edges {
        let Some(child) = tree.child_index(node, edge).filter(|&child| tree.nodes[child].proof.is_none()) else { continue };
        let mut child_board = board.clone();
        child_board.play_unchecked(tree.moves[edge].to_move(board));
        path.push(child);
        tree.keys.push(tree.node_keys[child]);
        stale_leaves_by_positions(tree, &child_board, path, settings, tables, root_covered, stale);
        tree.keys.pop();
        path.pop();
    }
}

//the reference for Searches' tree reuse: the moves played one at a time, then the stale leaves taken out
fn reused_move_by_move(tree: &mut Tree, moves: &[PackedMove], settings: &SearchSettings, tables: Option<&Tables>) -> usize {
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

//every node's moves with a node: place, node, visits, and what else the move holds: the bits of its mean, the simulations
//waiting through it, whether it has learnt that its node is a proven loss, and its node's repeat filter
fn moves_with_nodes(tree: &Tree) -> Vec<Vec<(usize, u32, u32, (u32, u16, bool, u8))>> {
    let move_of = |(place, link): (usize, u32)| {
        let slot = tree.slot(link);
        (place, slot.child, tree.move_visits(link), (slot.mean.to_bits(), slot.pending, slot.lost, slot.repeats))
    };
    (0..tree.nodes.len()).map(|node| tree.linked(node).map(move_of).collect()).collect()
}

fn line_end(tree: &Tree, line: &[usize]) -> usize {
    line.iter().fold(tree.root_index(), |node, &edge| tree.child_index(node, edge).expect("a line through the tree's nodes"))
}

//a node's head and tail are each in the order of their priors and its moves with a node are the first ones of each (a root with noise
//has them as the noise put them), so that select_edge may skip past each one's first move without a node: it must pick what a look at
//every move picks
fn assert_by_prior(tree: &Tree) {
    let settings = settings(true);
    for (id, node) in tree.nodes.iter().enumerate() {
        let edges = node.edges();
        let priors = &tree.priors[edges.clone()];
        let (head, tail) = priors.split_at(priors.len().min(HEAD_MOVES));
        assert!(
            head.is_sorted_by(|before, after| before >= after) && tail.is_sorted_by(|before, after| before >= after),
            "node {id}'s priors"
        );
        //the moves that link_known_moves gave nodes are any of the node's
        if (id != tree.root || tree.noised_priors.is_empty()) && node.known_passes == 0 {
            let places: Vec<usize> = tree.linked(id).map(|(place, _)| place).collect();
            assert!(first_of_head_and_tail(&places), "node {id}'s moves with a node are {places:?}");
        }
        if edges.is_empty() {
            continue;
        }
        let scan = tree.scan(id, &settings);
        let mut best = (edges.start, f64::NEG_INFINITY);
        for (place, edge) in edges.enumerate() {
            let link = tree.link_of(id, edge);
            let slot = if link == NO_LINK { &UNVISITED_SLOT } else { tree.slot(link) };
            if let Some(score) = tree.move_value(&scan, None, place, tree.priors[edge], slot)
                && score > best.1
            {
                best = (edge, score);
            }
        }
        assert_eq!(tree.select_edge(id, &settings, None).0, best.0, "node {id}'s pick");
    }
}

fn assert_same_trees(old: &Tree, new: &Tree) {
    assert_by_prior(old);
    assert_by_prior(new);
    let fen = |tree: &Tree| Fen::from_position(&tree.board, EnPassantMode::Legal).to_string();
    assert_eq!(fen(old), fen(new));
    assert_same("keys", &old.keys, &new.keys);
    let numbers = |tree: &Tree| -> Vec<_> {
        tree.nodes
            .iter()
            .map(|node| {
                let proof = node.proof.map(|proof| (proof.outcome, proof.value.to_bits(), proof.plies));
                let (value_sum, visited_policy_sqrt) = (node.value_sum.to_bits(), node.visited_policy_sqrt.to_bits());
                let known = (node.known_passes, node.known_to);
                (node.visit_count, node.in_flight, value_sum, visited_policy_sqrt, node.edges_at, proof, known)
            })
            .collect()
    };
    assert_same("nodes", &numbers(old), &numbers(new));
    assert_same("node keys", &old.node_keys, &new.node_keys);
    assert_same("moves", &old.moves, &new.moves);
    let bits = |priors: &[f32]| -> Vec<u32> { priors.iter().map(|prior| prior.to_bits()).collect() };
    assert_same("priors", &bits(&old.priors), &bits(&new.priors));
    assert_same("children", &moves_with_nodes(old), &moves_with_nodes(new));
    //with transpositions, the fifty-move counts and the table too (the moves' visits are the children's)
    assert_eq!(old.graph.is_some(), new.graph.is_some(), "one tree with transpositions, the other without");
    if let (Some(old), Some(new)) = (&old.graph, &new.graph) {
        assert_same("fifty-move counts", &old.halfmoves, &new.halfmoves);
        let table = |graph: &Graph| -> Vec<(u64, u32)> {
            let mut entries: Vec<(u64, u32)> = graph.table.iter().map(|(&entry, &node)| (entry, node)).collect();
            entries.sort_unstable();
            entries
        };
        assert_same("table", &table(old), &table(new));
    }
}

fn reusable_lines(tree: &Tree) -> Vec<Vec<usize>> {
    let expected = tree.principal_variation();
    let mut lines: Vec<Vec<usize>> = (1..=expected.len().min(3)).map(|plies| expected[..plies].to_vec()).collect();
    if let Some(reply_node) = tree.child_index(tree.root_index(), expected[0]) {
        let mut replies: Vec<usize> = tree.nodes[reply_node].edges().filter(|&edge| tree.visits(reply_node, edge) > 0).collect();
        replies.sort_by_key(|&edge| std::cmp::Reverse(tree.visits(reply_node, edge)));
        lines.extend(replies.get(1).map(|&reply| vec![expected[0], reply]));
    }
    lines.retain(|line| {
        let mut node = Some(tree.root_index());
        for &edge in line {
            node = node.and_then(|node| tree.child_index(node, edge));
        }
        node.is_some_and(|node| tree.nodes[node].has_edges())
    });
    lines
}

//Searches' advance and next start, with the "walk" or the "copy" stopped as asked
fn advanced(
    tree: &Tree,
    line: &[usize],
    threads: usize,
    stopped: Option<&str>,
    settings: &SearchSettings,
    tables: Option<&Tables>,
) -> (Tree, bool) {
    let mut node = tree.root;
    let places: Vec<u16> = line
        .iter()
        .map(|&edge| {
            let place = (edge - tree.nodes[node].edges().start) as u16;
            node = tree.child_index(node, edge).expect("the line's node");
            place
        })
        .collect();
    let mut advanced = copied(tree);
    advanced.move_root(&line[..1]);
    let (never, always) = (AtomicBool::new(false), AtomicBool::new(true));
    let candidates = advanced.stale_candidates(threads, if stopped == Some("walk") { &always } else { &never });
    if candidates.is_some() {
        advanced.compact(threads, Some(if stopped == Some("copy") { &always } else { &never }));
    }
    let (mut rest, mut node) = (Vec::new(), advanced.root);
    for &place in &places[1..] {
        let edge = advanced.nodes[node].edges().start + usize::from(place);
        rest.push(edge);
        node = advanced.child_index(node, edge).expect("the line's node");
    }
    advanced.move_root(&rest);
    assert!(advanced.nodes[advanced.root].proof.is_none(), "a root is never proven");
    let found = candidates.is_some();
    match candidates {
        Some(candidates) => advanced.drop_stale_candidates(&places[1..], &candidates, settings, tables),
        None => advanced.drop_stale_leaves(settings, tables),
    }
    advanced.compact(1, None);
    (advanced, found)
}

//what a move keeps of its node never hides a repeat: whenever the node repeats a position of the path, the move's filter says it
//may, so selection reads the node and finds the repeat; and the filter spares the reading of most nodes that repeat none
#[test]
fn a_moves_filter_of_its_node_lets_every_repeat_through() {
    let keys: Vec<u64> = (0..40).map(mix).collect();
    let others = (1_000..1_400).map(mix);
    let (mut repeats_found, mut spared, mut looked) = (0, 0, 0);
    for position in 5..=keys.len() {
        for halfmoves in [0, 3, 10, 60] {
            let repeats = Repeats::new(&keys[..position], 0, halfmoves);
            for key in keys.iter().copied().chain(others.clone()) {
                for count in [0, 1, 7] {
                    let (repeat, may) = (repeats.repeats(key, count), repeats.may_repeat(repeat_filter(key, count)));
                    assert!(may || !repeat, "a repeat that the filter hides: {key:x} at count {count}");
                    (repeats_found, spared, looked) = (repeats_found + usize::from(repeat), spared + usize::from(!may), looked + 1);
                }
            }
        }
    }
    assert!(repeats_found > 1_000 && spared > looked * 9 / 10, "{repeats_found} repeats, {spared} of {looked} nodes not read");
}

//mcts.py's constants with the engine's value sharpening
//16 bit priors are within half a step of 2^-11 of the prior, 8 bit ones of 2^-4, and both rise with it, so a node's head and tail
//keep by_prior's order
#[test]
fn stored_priors_are_close_and_keep_the_order() {
    let (mut head_before, mut tail_before) = (0.0, 0.0);
    for step in 1..=200_000u32 {
        let prior = (f64::from(step) / 200_000.0).powi(6) as f32;
        let (head, tail) = (head_prior(head_code(prior)), tail_prior(tail_code(prior)));
        assert!(head >= head_before && tail >= tail_before, "{prior}: the stored priors fall");
        (head_before, tail_before) = (head, tail);
        if prior >= 1e-9 {
            assert!((head - prior).abs() <= prior / 4096.0, "{prior}: 16 bits {head}");
        }
        if (4e-6..=1.0 / 9.0).contains(&prior) {
            assert!((tail - prior).abs() <= prior / 32.0, "{prior}: 8 bits {tail}");
        }
    }
    let mut state = 12_345u64;
    for _ in 0..2_000 {
        let count = 2 + (state >> 33) as usize % 60;
        let mut logits: Vec<f32> = (0..count)
            .map(|_| {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                //ties now and then, as equal logits give
                ((state >> 40) % 64) as f32 / 8.0
            })
            .collect();
        softmax(&mut logits);
        let stored: Vec<f32> = by_prior(&logits, &mut [0; MOST_MOVES])
            .iter()
            .enumerate()
            .map(|(rank, key)| stored_prior(rank, logits[(key & 255) as usize]))
            .collect();
        let (head, tail) = stored.split_at(count.min(HEAD_MOVES));
        assert!(head.is_sorted_by(|before, after| before >= after) && tail.is_sorted_by(|before, after| before >= after), "{stored:?}");
    }
}

//the prior stored for the move of `rank` of `count` equally likely ones
fn equal_prior(rank: usize, count: usize) -> f32 {
    stored_prior(rank, 1.0 / count as f32)
}

fn settings(certainty: bool) -> SearchSettings {
    SearchSettings {
        c_puct: 1.745 / 2.0,
        c_puct_factor: 3.894 / 2.0,
        c_puct_base: 38739.0,
        fpu_reduction: 0.33 / 2.0,
        draw: 0.5,
        value_sharpness: 3.0,
        certainty,
        transpositions: false,
    }
}

fn tables() -> Option<Tables> {
    let folder = concat!(env!("CARGO_MANIFEST_DIR"), "/../../artifacts/engines/syzygy");
    std::path::Path::new(folder).is_dir().then(|| Tables::open(folder, 5).expect("the tables"))
}

//tables that fail every probe: KQvK is a file of the right size with nothing in it, and any other table is missing
fn failing_tables() -> Tables {
    static FOLDER: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
    let folder = FOLDER.get_or_init(|| {
        let folder = std::env::temp_dir().join("selfplay_rs_failing_tables");
        std::fs::create_dir_all(&folder).expect("a folder");
        std::fs::write(folder.join("KQvK.rtbw"), [0u8; 16]).expect("a table file");
        folder
    });
    Tables::open(folder.to_str().expect("a path"), 3).expect("the folder's tables")
}

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const OPENING: &[&str] = &["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7", "e2e3", "e8g8", "g1f3", "b8d7"];
//rooks and locked pawns repeat positions (draws that go stale), the mate in two is proven, the tables score the captures
const ROOKS: &str = "8/8/4k3/8/8/4K3/R7/7r w - - 0 1";
const LOCKED_PAWNS: &str = "4k3/8/8/1p1p1p1p/1P1P1P1P/8/8/4K3 w - - 0 1";
const MATE_IN_TWO: &str = "k7/8/2K5/8/8/8/8/7R w - - 0 1";
const PAWNS_AND_ROOKS: &str = "8/8/4k3/3p4/3P4/4K3/R7/7r w - - 0 1";
//the queen takes the pawn into a king and queen ending, which three man tables cover
const QUEEN_AND_PAWN: &str = "8/8/8/4k3/8/2p5/8/3QK3 w - - 0 1";

fn cases(tables: &Option<Tables>) -> Vec<(&'static str, &'static [&'static str], u32, bool, Option<&Tables>)> {
    let mut cases = vec![
        (START, OPENING, 6_000, true, None),
        (ROOKS, &[][..], 4_000, true, None),
        (ROOKS, &[], 4_000, false, None),
        (LOCKED_PAWNS, &[], 4_000, true, None),
        (MATE_IN_TWO, &[], 4_000, true, None),
    ];
    match tables {
        Some(tables) => {
            cases.extend([(PAWNS_AND_ROOKS, &[][..], 4_000, false, Some(tables)), (PAWNS_AND_ROOKS, &[], 4_000, true, Some(tables))])
        }
        None => eprintln!("no syzygy tables: the ending they score is left out"),
    }
    cases
}

fn search_in_flight(
    tree: &mut Tree,
    simulations: u32,
    in_flight: usize,
    settings: &SearchSettings,
    network: &mut Network,
    mut gather: impl FnMut(&mut Tree, &Round, &Cache, &mut Vec<u8>) -> u32,
) -> Vec<Vec<u8>> {
    let (mut rounds, mut gathered) = (VecDeque::new(), Vec::new());
    loop {
        let room = tree.root().visit_count as usize + tree.pending() < simulations as usize && !tree.full;
        if room && rounds.len() < in_flight {
            let round = new_round(tree, gathered.len() as u64, simulations, usize::MAX);
            let mut boards = Vec::new();
            let count = gather(tree, &round, &network.cache, &mut boards);
            assert_eq!(boards.len(), count as usize * TOKENS_PER_BOARD);
            gathered.push(boards.clone());
            rounds.push_back((round.id, boards));
            continue;
        }
        let Some((id, boards)) = rounds.pop_front() else { break };
        finish_leaves(tree, id, &boards, settings, network);
    }
    gathered
}

#[test]
fn a_line_played_at_once_leaves_the_tree_its_moves_played_one_at_a_time_leave() {
    let tables = tables();
    let mut cases = vec![
        (START, OPENING, 20_000, true, None),
        (ROOKS, &[][..], 10_000, true, None),
        (ROOKS, &[], 10_000, false, None),
        (LOCKED_PAWNS, &[], 10_000, true, None),
        (MATE_IN_TWO, &[], 10_000, true, None),
        ("8/8/8/4k3/8/8/8/4K2Q w - - 0 1", &[], 20_000, true, None),
        ("8/8/8/4k3/8/8/8/4K2R w - - 0 1", &[], 20_000, true, None),
    ];
    match &tables {
        Some(tables) => {
            cases.extend([(PAWNS_AND_ROOKS, &[][..], 10_000, false, Some(tables)), (PAWNS_AND_ROOKS, &[], 10_000, true, Some(tables))])
        }
        None => eprintln!("no syzygy tables: the ending they score is left out"),
    }

    let (mut lines, mut stale, mut proven, mut walked) = (0, 0, 0, 0);
    for (fen, moves, simulations, certainty, tables) in cases {
        let settings = settings(certainty);
        let mut tree = tree_after(fen, moves);
        search(&mut tree, simulations, usize::MAX, &settings, tables, &mut Network::new());
        for line in reusable_lines(&tree) {
            let mut old = copied(&tree);
            let line_moves: Vec<PackedMove> = line.iter().map(|&edge| tree.chess_move(edge)).collect();
            stale += reused_move_by_move(&mut old, &line_moves, &settings, tables);
            let mut new = copied(&tree);
            new.play_line(&line);
            new.drop_stale_leaves(&settings, tables);
            assert_same_trees(&old, &new);
            for (threads, stopped) in [(1, None), (1, Some("copy")), (1, Some("walk")), (4, None), (4, Some("copy")), (4, Some("walk"))] {
                let (advanced, found) = advanced(&tree, &line, threads, stopped, &settings, tables);
                assert_same_trees(&new, &advanced);
                walked += usize::from(!found);
            }
            let (mut moved, never) = (copied(&tree), AtomicBool::new(false));
            moved.move_root(&line[..1]);
            assert_eq!(moved.stale_candidates(4, &never), moved.stale_candidates(1, &never));
            let (mut one, mut four) = (copied(&new), copied(&new));
            one.compact(1, None);
            four.compact(4, None);
            assert_same_trees(&one, &four);
            lines += 1;
            proven += new.nodes.iter().filter(|node| node.proof.is_some()).count();
        }
    }
    assert!(
        lines >= 15 && stale > 0 && proven > 0 && walked > 0,
        "{lines} lines, {stale} stale leaves, {proven} proven nodes, {walked} walks"
    );
}

//a proven node hides the leaves below it from the stale walk unless it is the new root; searches rarely leave a stale leaf
//below one, so the proofs are set by hand
#[test]
fn proven_nodes_hide_stale_leaves_from_advance_as_from_the_walk() {
    let settings = settings(true);
    let mut tree = tree_after(LOCKED_PAWNS, &[]);
    search(&mut tree, 10_000, usize::MAX, &settings, None, &mut Network::new());
    //a two-ply line with a stale leaf at least two nodes below its end
    let (line, path) = reusable_lines(&tree)
        .into_iter()
        .filter(|line| line.len() == 2)
        .find_map(|line| {
            let mut moved = copied(&tree);
            moved.move_root(&line);
            let mut stale = Vec::new();
            moved.find_stale_leaves(&mut vec![moved.root], &mut Vec::new(), &settings, None, false, &mut stale);
            stale.into_iter().find(|path| path.len() >= 3).map(|path| (line, path))
        })
        .expect("a stale leaf two nodes below a reusable line");
    let draw = Proof { outcome: Outcome::Draw, value: 0.5, plies: 0 };
    for proven in [path[0], path[1]] {
        let mut proved = copied(&tree);
        proved.nodes[proven].proof = Some(draw);
        let mut walked = copied(&proved);
        walked.play_line(&line);
        walked.drop_stale_leaves(&settings, None);
        let (advanced, _) = advanced(&proved, &line, 4, None, &settings, None);
        assert_same_trees(&walked, &advanced);
    }
}

#[test]
fn the_cache_keeps_two_generations_and_turns_one_over_at_once() {
    let generation = 1_000_000u64;
    let mut cache = Cache::new(2 * generation as usize);
    let priors = |key: u64| -> Vec<f32> { (0..37).map(|index| (key + index) as f32).collect() };
    let (mut slowest_turnover, mut room) = (std::time::Duration::ZERO, None);
    for key in 0..=3 * generation {
        let started = std::time::Instant::now();
        cache.insert(key, &priors(key), key as f32);
        let made = Some((cache.newer.priors.as_ptr(), cache.newer.priors.capacity(), cache.newer.places.capacity()));
        if key % generation == 0 {
            slowest_turnover = slowest_turnover.max(started.elapsed());
            room = made;
        } else {
            assert_eq!(made, room, "insert {key} regrew the newer generation");
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

//a probe that fails is no result: the network scores the position, as if there were no tables, and the stale check agrees with it
#[test]
fn a_search_with_tables_that_fail_is_the_search_without_tables() {
    let (tables, settings) = (failing_tables(), settings(true));
    let (mut with, mut without) = (tree_after(QUEEN_AND_PAWN, &[]), tree_after(QUEEN_AND_PAWN, &[]));
    search(&mut with, 3_000, usize::MAX, &settings, Some(&tables), &mut Network::new());
    search(&mut without, 3_000, usize::MAX, &settings, None, &mut Network::new());
    assert!(tables.failures() > 0, "no probe failed");
    assert_same_trees(&with, &without);
    assert_eq!(with.pending(), 0);

    let line = [with.principal_variation()[0]];
    for (tree, tables) in [(&mut with, Some(&tables)), (&mut without, None)] {
        tree.play_line(&line);
        tree.drop_stale_leaves(&settings, tables);
    }
    assert_same_trees(&with, &without);
}

//(the first move's logit, the value, whether the priors come out equal, the value backed up) for answers that are no numbers: a logit or
//the value is a NaN or an infinity. A logit that is none leaves the priors equal, a value that is none makes the leaf a draw
const NOT_NUMBERS: [(f32, f32, bool, f32); 4] =
    [(f32::NAN, 0.25, true, 0.25), (f32::INFINITY, 0.25, true, 0.25), (0.5, f32::NAN, false, 0.5), (0.5, f32::INFINITY, false, 0.5)];

#[test]
fn an_answer_that_is_no_number_leaves_equal_priors_or_a_draw_and_is_not_cached() {
    let settings = SearchSettings { value_sharpness: 1.0, ..settings(false) };
    for (logit, value, equal, backed_up) in NOT_NUMBERS {
        let (mut tree, mut network) = (tree_after(START, &[]), Network::new());
        let mut boards = Vec::new();
        tree.gather(&new_round(&tree, 0, 100, usize::MAX), &settings, None, Some(&network.cache), &mut boards);
        let leaf = &tree.round_leaves(0)[0];
        let (key, first_move) = (leaf.key, network.move_ids.id(leaf.moves[0], leaf.turn));
        let answer = |column: usize| match column {
            ANSWER_WIDTH_LAST => value,
            _ if column == first_move => logit,
            _ => 0.01 * column as f32,
        };
        tree.finish_leaf(0, answer, ANSWER_WIDTH, &network.move_ids, Some(&mut network.cache), &settings);
        let moves = tree.nodes[0].edges();
        let equal_priors = moves.clone().enumerate().all(|(rank, edge)| tree.priors[edge] == equal_prior(rank, moves.len()));
        assert_eq!((tree.nodes[0].visit_count, equal_priors), (1, equal), "{logit} {value}");
        assert!(moves.clone().all(|edge| tree.priors[edge].is_finite()), "{logit} {value}: priors {:?}", &tree.priors[moves.clone()]);
        assert_eq!(tree.nodes[0].value_sum, f64::from(1.0 - backed_up), "{logit} {value}");
        assert!(network.cache.get(key).is_none(), "{logit} {value}: the answer is cached");
    }
}

#[test]
fn the_nodes_stop_short_of_their_u32_indices() {
    assert!(node_fits(NO_NODE as usize - 1));
    assert!(!node_fits(NO_NODE as usize));
    assert_eq!(grown_capacity(8, 9, 13, usize::MAX), Some(16));
    assert_eq!(grown_capacity(8, 40, 13, usize::MAX), Some(40));
    assert_eq!(grown_capacity(8, 9, 13, 13 * 12), Some(12));
    assert_eq!(grown_capacity(8, 9, 13, 13 * 8), None);
}

#[test]
fn leaves_finished_from_their_logits_in_move_order_are_those_finished_from_rows() {
    let settings = settings(true);
    let (mut by_rows, mut in_order) = (tree_after(START, OPENING), tree_after(START, OPENING));
    search_with(finish_leaves, &mut by_rows, 3_000, usize::MAX, &settings, None, &mut Network::new());
    search_with(finish_leaves_in_order, &mut in_order, 3_000, usize::MAX, &settings, None, &mut Network::new());
    assert!(by_rows.nodes.len() > 1_000);
    assert_same_trees(&by_rows, &in_order);
}

//a growth's copy counts toward the limit, so the vectors end half to two thirds of it
#[test]
fn a_search_ends_at_its_memory_limit_as_the_same_search() {
    let memory_limit = 4_000_000;
    let settings = settings(true);
    let mut limited = tree_after(START, &[]);
    search(&mut limited, 1_000_000, memory_limit, &settings, None, &mut Network::new());
    let reached = limited.root().visit_count;
    assert!(limited.full && reached < 1_000_000 && limited.allocated_bytes() <= memory_limit, "{reached} {}", limited.allocated_bytes());

    let mut unlimited = tree_after(START, &[]);
    search(&mut unlimited, reached, usize::MAX, &settings, None, &mut Network::new());
    assert_same_trees(&unlimited, &limited);
    assert!(!unlimited.full && limited.allocated_bytes() > memory_limit / 3);
}

#[test]
fn a_node_whose_every_move_is_lost_descends_to_its_first_moves_node() {
    let lost = Proof { outcome: Outcome::Loss, value: 0.0, plies: 1 };
    let after = |tree: &Tree, edge: usize| {
        let mut board = tree.board.clone();
        board.play_unchecked(tree.moves[edge].to_move(&tree.board));
        board
    };
    for solid in [false, true] {
        let mut tree = tree_after(START, &[]);
        let moves = tree.board.legal_moves();
        let priors = vec![1.0 / moves.len() as f32; moves.len()];
        tree.expand(0, &moves, &priors);
        tree.nodes[0].visit_count = SOLID_TREE_THRESHOLD;
        tree.nodes[0].value_sum = 0.5 * f64::from(SOLID_TREE_THRESHOLD);
        let mut nodes_of = Vec::new();
        for edge in tree.nodes[0].edges() {
            let board = after(&tree, edge);
            let (node, _) = tree.child_node(0, edge, NO_LINK, &board);
            tree.nodes[node].proof = Some(lost);
            nodes_of.push(node);
        }
        if solid {
            tree.make_solid(0);
            assert!(!listed(tree.nodes[0].children), "the root made solid");
        }
        let (links, slots, nodes) = (tree.links.len(), tree.slots.len(), tree.nodes.len());
        let (edge, link) = tree.select_edge(0, &settings(true), None);
        assert_eq!(edge, tree.nodes[0].edges().start, "the first move when every move is lost");
        let board = after(&tree, edge);
        let (node, _) = tree.child_node(0, edge, link, &board);
        assert_eq!(node, nodes_of[0], "the first move's own node");
        assert_eq!((tree.links.len(), tree.slots.len(), tree.nodes.len()), (links, slots, nodes), "nothing made for a move with a node");
    }
}

//a tree with transpositions
fn graph_tree(fen: &str, moves: &[&str]) -> (Tree, SearchSettings) {
    let settings = SearchSettings { transpositions: true, ..settings(false) };
    (tree_after(fen, moves).with_transpositions(true), settings)
}

//search without the cache, so that every expanded node's own score is its position's answer
fn search_without_cache(tree: &mut Tree, simulations: u32, settings: &SearchSettings, network: &Network) {
    let mut boards = Vec::new();
    while tree.root().visit_count < simulations {
        let round = new_round(tree, 0, simulations, usize::MAX);
        boards.clear();
        tree.gather(&round, settings, None, None, &mut boards);
        for board in boards.as_chunks::<TOKENS_PER_BOARD>().0 {
            let seed = board.iter().fold(0, |hash, &token| mix(hash ^ u64::from(token)));
            tree.finish_leaf(0, |column| answer(board, seed, column), ANSWER_WIDTH, &network.move_ids, None, settings);
        }
    }
}

//every node the root reaches with its position, each once
fn nodes_with_boards(tree: &Tree) -> Vec<(usize, Chess)> {
    let mut seen = vec![false; tree.nodes.len()];
    let mut found = vec![(tree.root, tree.board.clone())];
    seen[tree.root] = true;
    let mut next = 0;
    while next < found.len() {
        let (node, board) = found[next].clone();
        next += 1;
        for edge in tree.nodes[node].edges() {
            if let Some(child) = tree.child_index(node, edge).filter(|&child| !seen[child]) {
                seen[child] = true;
                let mut after = board.clone();
                after.play_unchecked(tree.moves[edge].to_move(&board));
                found.push((child, after));
            }
        }
    }
    found
}

//a graph's sums: a node's visits are its own and its moves', its sum its own scores and each move's visits at the mean the move
//holds. An expanded node's own scores are its position's answer and a draw for each repeat that ended at it. Returns the nodes
//that several moves lead to
fn assert_move_means(tree: &Tree, settings: &SearchSettings) -> usize {
    assert_move_means_of(tree, settings, true)
}

//assert_move_means, with the root's sum left out unless `root_too`: a kept tree's root that was proven before it was the root has
//its proof dropped, and the sum the proof wrote is not its moves'
fn assert_move_means_of(tree: &Tree, settings: &SearchSettings, root_too: bool) -> usize {
    let mut parents = vec![0u32; tree.nodes.len()];
    let found = nodes_with_boards(tree);
    for (node, board) in &found {
        let state = &tree.nodes[*node];
        let (mut moves_visits, mut moves_sum) = (0u32, 0.0);
        for (_, link) in tree.linked(*node) {
            let (child, visits, mean) = (tree.link_child(link) as usize, tree.own_visits(link), f64::from(tree.slot(link).mean));
            parents[child] += 1;
            assert!(visits <= tree.nodes[child].visit_count, "a move has no more visits than its node");
            assert!(visits == 0 || (0.0..=1.0).contains(&mean), "a visited move's mean is a score, not {mean}");
            moves_visits += visits;
            moves_sum += f64::from(visits) * (1.0 - mean);
        }
        assert!(state.visit_count >= moves_visits, "a node's visits are its own and its moves'");
        let own_visits = state.visit_count - moves_visits;
        if state.has_edges() && state.proof.is_none() && (root_too || *node != tree.root) {
            let mut tokens = Vec::new();
            encode_board(board, &mut tokens);
            let seed = tokens.iter().fold(0, |hash, &token| mix(hash ^ u64::from(token)));
            let value = sharpen(answer(&tokens, seed, ANSWER_WIDTH_LAST), settings.value_sharpness);
            assert!(own_visits >= 1, "an expanded node's own visits: its evaluation's and repeats'");
            let own_sum = (1.0 - f64::from(value)) + f64::from(own_visits - 1) * (1.0 - settings.draw);
            let difference = (state.value_sum - own_sum - moves_sum).abs();
            assert!(
                difference <= 1e-6 * f64::from(state.visit_count),
                "node {node}: sum {} against its own {own_sum} and its moves' {moves_sum} ({} visits, {own_visits} its own)",
                state.value_sum,
                state.visit_count
            );
        }
    }
    parents.iter().filter(|&&count| count > 1).count()
}

#[test]
fn a_graphs_nodes_sum_is_its_own_scores_and_its_moves_visits_at_their_means() {
    let network = Network::new();
    for (fen, moves, visits) in [(START, &[][..], 6000), (START, OPENING, 6000), (LOCKED_PAWNS, &[][..], 3000), (QUEEN_AND_PAWN, &[][..], 3000)] {
        let (mut tree, settings) = graph_tree(fen, moves);
        search_without_cache(&mut tree, visits, &settings, &network);
        let shared = assert_move_means(&tree, &settings);
        assert!(shared > 0, "{fen}: a search of {visits} visits shares some nodes");
        //the tree kept for the next move, then searched on
        let played = tree.most_visited_edge(tree.root);
        tree.play(played, true);
        assert_move_means(&tree, &settings);
        search_without_cache(&mut tree, visits, &settings, &network);
        assert_move_means(&tree, &settings);
    }
}

//with proven results too: a proven node's sum is its visits at its proven score, and the nodes above it hold their sums by the moves'
//means as any other (assert_move_means), a move to a proven node its proven score once a descent has met it since
#[test]
fn in_a_graph_proven_nodes_keep_their_scores_and_the_sums_above_them() {
    let network = Network::new();
    //the mate in two is proven at the root by 3,100 visits
    for (fen, moves, visits) in
        [(MATE_IN_TWO, &[][..], 4000), (ROOKS, &[][..], 3000), (QUEEN_AND_PAWN, &[][..], 3000), (LOCKED_PAWNS, &[][..], 3000), (START, OPENING, 4000)]
    {
        let settings = SearchSettings { transpositions: true, ..settings(true) };
        let mut tree = tree_after(fen, moves).with_transpositions(true);
        search_without_cache(&mut tree, visits, &settings, &network);
        assert_move_means(&tree, &settings);
        let mut proven = 0;
        for (node, _) in nodes_with_boards(&tree) {
            let state = &tree.nodes[node];
            let Some(proof) = state.proof else { continue };
            proven += 1;
            let difference = (state.value_sum - f64::from(state.visit_count) * proof.value).abs();
            assert!(difference <= 1e-9 * f64::from(state.visit_count), "{fen}: a proven node's sum {} for {} visits", state.value_sum, state.visit_count);
        }
        if fen == MATE_IN_TWO {
            assert!(proven > 0, "a mate in two is proven");
            //the mating move's node is a loss for the side to move there, a win for the root's side: the move's mean says so
            let best = tree.most_visited_edge(tree.root);
            let link = tree.link_of(tree.root, best);
            let child = &tree.nodes[tree.link_child(link) as usize];
            assert!(child.proof.is_some_and(|proof| proof.outcome == Outcome::Win), "the root's best move is a proven win");
            assert!(tree.slot(link).mean > 0.99, "and its mean a win's, not {}", tree.slot(link).mean);
        }
    }
}

//a graph's node's moves take the nodes their positions already have, from KNOWN_FIRST_VISITS visits on, whether its moves with
//a node are a list still or slots: some moves hold a visited node before their own first visit, selection weighs such a move by
//its node's mean, a list keeps its links in the moves' order, and the sums hold as before
#[test]
fn in_a_graph_a_nodes_moves_take_the_nodes_their_positions_have() {
    let network = Network::new();
    //kings that can only walk: most positions are reached by many paths
    let (mut tree, settings) = graph_tree(LOCKED_PAWNS, &[]);
    search_without_cache(&mut tree, 6000, &settings, &network);
    assert_move_means(&tree, &settings);
    let (mut known_before_visited, mut in_lists) = (0, 0);
    for (node, board) in nodes_with_boards(&tree) {
        let state = &tree.nodes[node];
        let places: Vec<usize> = tree.linked(node).map(|(place, _)| place).collect();
        assert!(places.is_sorted_by(|before, after| before < after), "node {node}'s moves with a node at {places:?}");
        if state.known_passes == 0 {
            continue;
        }
        assert!(state.visit_count >= KNOWN_FIRST_VISITS, "a pass at {} visits", state.visit_count);
        for (place, link) in tree.linked(node) {
            let child = tree.link_child(link) as usize;
            if tree.own_visits(link) == 0 && tree.nodes[child].visit_count > 0 {
                known_before_visited += 1;
                in_lists += usize::from(listed(state.children));
                let mean = tree.slot(link).mean;
                assert!((0.0..=1.0).contains(&mean), "a move given a node holds the node's mean of then, not {mean}");
                let edge = state.edges().start + place;
                let mut after = board.clone();
                after.play_unchecked(tree.moves[edge].to_move(&board));
                assert_eq!(tree.node_keys[child], position_key(&after), "the node is the move's position's");
            }
        }
    }
    assert!(known_before_visited > in_lists && in_lists > 0, "{known_before_visited} moves given their position's node, {in_lists} in lists");
    assert!(size_of::<Node>() == 64, "the passes' count fits the node's line");
}

//a later pass gives moves the first could not: their positions' nodes were made after it. The search is of one simulation a
//round, so that a move a descent picked always has its visit: a move with a node and no visit was given the node
#[test]
fn in_a_graph_a_later_pass_gives_moves_the_nodes_made_since_the_first() {
    let network = Network::new();
    let (mut tree, settings) = graph_tree(LOCKED_PAWNS, &[]);
    let search_to = |tree: &mut Tree, visits: u32| {
        let round = Round { id: 0, leaves: 1, simulations: 1, collisions: 0, visit_limit: visits, memory_limit: usize::MAX };
        let mut boards = Vec::new();
        while tree.root().visit_count < visits {
            boards.clear();
            tree.gather(&round, &settings, None, None, &mut boards);
            for board in boards.as_chunks::<TOKENS_PER_BOARD>().0 {
                let seed = board.iter().fold(0, |hash, &token| mix(hash ^ u64::from(token)));
                tree.finish_leaf(0, |column| answer(board, seed, column), ANSWER_WIDTH, &network.move_ids, None, &settings);
            }
        }
    };
    search_to(&mut tree, 2_000);
    //the nodes that have had their first pass and no other, each with its moves still without a node
    let without_a_node =
        |tree: &Tree, node: usize| -> Vec<usize> { tree.nodes[node].edges().filter(|&edge| tree.link_of(node, edge) == NO_LINK).collect() };
    let passed_once = (0..tree.nodes.len()).filter(|&node| tree.nodes[node].known_passes == 1);
    let once: Vec<(usize, Vec<usize>)> = passed_once.map(|node| (node, without_a_node(&tree, node))).collect();
    let nodes_then = tree.nodes.len();
    let lists_then = once.iter().filter(|&&(node, _)| listed(tree.nodes[node].children)).count();
    assert!(lists_then > 0 && lists_then < once.len(), "{lists_then} of the {} nodes that have had their first pass are lists", once.len());
    search_to(&mut tree, 12_000);
    let (mut passed_again, mut given_since) = (0, 0);
    for (node, without) in once {
        let state = &tree.nodes[node];
        assert_eq!(u32::from(state.known_passes), known_passes_due(state.visit_count).max(1), "node {node}");
        if state.known_passes < 2 {
            continue;
        }
        passed_again += 1;
        for edge in without {
            let link = tree.link_of(node, edge);
            let made_since = link != NO_LINK && tree.link_child(link) as usize >= nodes_then;
            given_since += usize::from(made_since && tree.own_visits(link) == 0);
        }
    }
    assert!(passed_again > 0 && given_since > 0, "{passed_again} nodes had a later pass, which gave {given_since} moves nodes made since");
}

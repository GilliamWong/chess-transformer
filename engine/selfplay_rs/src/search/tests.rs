//tree reuse (Searches): play_line's one copy and drop_stale_leaves' walk against the moves played one at a time (play)
//and a walk that plays out every node's position, as trees were reused before, and against advance's way (the root
//moved in place, the stale leaves from candidates). They must give the same tree, number for number, on trees the
//engine's rounds grow with a stand-in for the network
use super::*;
use crate::rules::{MOVE_ID_TABLE_LENGTH, TOKENS_PER_BOARD};
use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, EnPassantMode};
use std::fmt::Debug;
use std::sync::atomic::AtomicBool;

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
    while tree.root().visit_count < simulations && !(tree.full && tree.pending() == 0) {
        let leaves = (tree.root().visit_count / 16).clamp(1, 64);
        let round = Round { id: 0, leaves, simulations: 3 * leaves, collisions: leaves, visit_limit: simulations, memory_limit };
        boards.clear();
        tree.gather(&round, settings, tables, Some(&network.cache), &mut boards).expect("the tables");
        for board in boards.as_chunks::<TOKENS_PER_BOARD>().0 {
            let seed = board.iter().fold(0, |hash, &token| mix(hash ^ u64::from(token)));
            tree.finish_leaf(0, |column| answer(board, seed, column), ANSWER_WIDTH, &network.move_ids, Some(&mut network.cache), settings);
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

//Searches' tree reuse as it was: the moves played one at a time, each copying the subtree, then the stale leaves
//taken out. Returns how many were
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

//each node's moves with a node, in their order, as (place, node, visits): what the links and slots hold, whichever holds it
fn moves_with_nodes(tree: &Tree) -> Vec<Vec<(usize, u32, u32)>> {
    (0..tree.nodes.len())
        .map(|node| tree.linked(node).map(|(place, link)| (place, tree.link_child(link), tree.move_visits(link))).collect())
        .collect()
}

//the node a line of moves from the root ends on
fn line_end(tree: &Tree, line: &[usize]) -> usize {
    line.iter().fold(tree.root_index(), |node, &edge| tree.child_index(node, edge).expect("a line through the tree's nodes"))
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
                (node.visit_count, node.in_flight, value_sum, visited_policy_sqrt, node.edges_at, proof)
            })
            .collect()
    };
    assert_same("nodes", &numbers(old), &numbers(new));
    assert_same("node keys", &old.node_keys, &new.node_keys);
    assert_same("moves", &old.moves, &new.moves);
    let bits = |priors: &[f32]| -> Vec<u32> { priors.iter().map(|prior| prior.to_bits()).collect() };
    assert_same("priors", &bits(&old.priors), &bits(&new.priors));
    assert_same("children", &moves_with_nodes(old), &moves_with_nodes(new));
    //with transpositions the same graph: the fifty-move counts and the table (the moves' visits are the children's)
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

//lines a game may have played since a search, for which Searches reuses the tree (every move has its node, and the
//last node is expanded): the expected move, the expected two and three, and the expected move answered by the reply
//searched second most
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

//Searches' advance and the next start: the line's first move in place, the stale candidates and the copy on `threads`
//threads unless stopped before them, then the rest of the line in place and the candidates under it (a walk without
//them), compacted to be compared node for node. Returns the tree and whether the candidates were found
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
        Some(candidates) => advanced.drop_stale_candidates(&places[1..], &candidates, settings, tables).expect("the tables"),
        None => advanced.drop_stale_leaves(settings, tables).expect("the tables"),
    }
    advanced.compact(1, None);
    (advanced, found)
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
        transpositions: false,
    }
}

//the syzygy tables of up to 5 pieces, None without them
fn tables() -> Option<Tables> {
    let folder = concat!(env!("CARGO_MANIFEST_DIR"), "/../../artifacts/engines/syzygy");
    std::path::Path::new(folder).is_dir().then(|| Tables::open(folder, 5).expect("the tables"))
}

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const OPENING: &[&str] = &["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7", "e2e3", "e8g8", "g1f3", "b8d7"];
const ROOKS: &str = "8/8/4k3/8/8/4K3/R7/7r w - - 0 1";
const LOCKED_PAWNS: &str = "4k3/8/8/1p1p1p1p/1P1P1P1P/8/8/4K3 w - - 0 1";
const MATE_IN_TWO: &str = "k7/8/2K5/8/8/8/8/7R w - - 0 1";
const PAWNS_AND_ROOKS: &str = "8/8/4k3/3p4/3P4/4K3/R7/7r w - - 0 1";

//the positions of the reuse test: a middlegame, endings with repeats, a mate certainty proves, and the tables' ending
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

//search's rounds, up to in_flight waiting, oldest finished first; every round's boards
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
            let leaves = (tree.root().visit_count / 16).clamp(1, 64);
            let round = Round {
                id: gathered.len() as u64,
                leaves,
                simulations: 3 * leaves,
                collisions: leaves,
                visit_limit: simulations,
                memory_limit: usize::MAX,
            };
            let mut boards = Vec::new();
            let count = gather(tree, &round, &network.cache, &mut boards);
            assert_eq!(boards.len(), count as usize * TOKENS_PER_BOARD);
            gathered.push(boards.clone());
            rounds.push_back((round.id, boards));
            continue;
        }
        let Some((id, boards)) = rounds.pop_front() else { break };
        for board in boards.as_chunks::<TOKENS_PER_BOARD>().0 {
            let seed = board.iter().fold(0, |hash, &token| mix(hash ^ u64::from(token)));
            tree.finish_leaf(id, |column| answer(board, seed, column), ANSWER_WIDTH, &network.move_ids, Some(&mut network.cache), settings);
        }
    }
    gathered
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
            new.drop_stale_leaves(&settings, tables).expect("the tables");
            assert_same_trees(&old, &new);
            for (threads, stopped) in [(1, None), (1, Some("copy")), (1, Some("walk")), (4, None), (4, Some("copy")), (4, Some("walk"))] {
                let (advanced, found) = advanced(&tree, &line, threads, stopped, &settings, tables);
                assert_same_trees(&new, &advanced);
                walked += usize::from(!found);
            }
            //the parallel walk finds the one-thread walk's candidates in its order, and the parallel copy leaves the
            //tree the one-thread copy leaves, number for number
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

//a proven node hides the leaves below it from the stale walk unless it is the new root, whose proof is dropped:
//advance's candidates must leave and take out the same ones. Searches rarely leave a stale leaf below a proven node, so
//the proofs are set by hand on a tree whose shuffles leave stale draws
#[test]
fn proven_nodes_hide_stale_leaves_from_advance_as_from_the_walk() {
    let settings = settings(true);
    let mut tree = tree_after("4k3/8/8/1p1p1p1p/1P1P1P1P/8/8/4K3 w - - 0 1", &[]);
    search(&mut tree, 10_000, usize::MAX, &settings, None, &mut Network::new());
    //a two-ply line under which a stale leaf lies at least two nodes down, as the walk finds it in place
    let (line, path) = reusable_lines(&tree)
        .into_iter()
        .filter(|line| line.len() == 2)
        .find_map(|line| {
            let mut moved = copied(&tree);
            moved.move_root(&line);
            let mut stale = Vec::new();
            moved.find_stale_leaves(&mut vec![moved.root], &mut Vec::new(), &settings, None, false, &mut stale).expect("no tables");
            stale.into_iter().find(|path| path.len() >= 3).map(|path| (line, path))
        })
        .expect("a stale leaf two nodes below a reusable line");
    let draw = Proof { outcome: Outcome::Draw, value: 0.5, plies: 0 };
    for proven in [path[0], path[1]] {
        let mut proved = copied(&tree);
        proved.nodes[proven].proof = Some(draw);
        let mut walked = copied(&proved);
        walked.play_line(&line);
        walked.drop_stale_leaves(&settings, None).expect("no tables");
        let (advanced, _) = advanced(&proved, &line, 4, None, &settings, None);
        assert_same_trees(&walked, &advanced);
    }
}

//the answer cache keeps its two latest generations and a position's latest answer, and no insert takes time out of the
//search: a turnover is quick (dropping a generation's boxed answers there stalled it), and a generation's vectors are
//made at its first answer, never regrown while it fills (a regrowth copies all they hold)
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

//the nodes' u32 indices: gathering stops before a node would be numbered NO_NODE; a vector under a memory limit grows by
//doubling as far as the limit lets it
#[test]
fn the_nodes_stop_short_of_their_u32_indices() {
    assert!(node_fits(NO_NODE as usize - 1));
    assert!(!node_fits(NO_NODE as usize));
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

mod bench;

mod shared;

//a node whose every move is a proven loss chooses none, and the descent takes its first move (select_edge names no move
//then): to that move's own node, as a list or as solid slots, never a second link or another move's slot
#[test]
fn a_node_whose_every_move_is_lost_descends_to_its_first_moves_node() {
    let lost = Proof { outcome: Outcome::Loss, value: 0.0, plies: 1 };
    for solid in [false, true] {
        let mut tree = tree_after(START, &[]);
        let moves = tree.board.legal_moves();
        let priors = vec![1.0 / moves.len() as f32; moves.len()];
        tree.expand(0, &moves, &priors);
        tree.nodes[0].visit_count = SOLID_TREE_THRESHOLD;
        tree.nodes[0].value_sum = 0.5 * f64::from(SOLID_TREE_THRESHOLD);
        let after = |tree: &Tree, edge: usize| {
            let mut board = tree.board.clone();
            board.play_unchecked(tree.moves[edge].to_move(&tree.board));
            board
        };
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

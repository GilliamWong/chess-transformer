use super::*;
use crate::search::shared::{SharedCache, SharedTree};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::time::{Duration, Instant};

const KING_AND_QUEEN: &str = "8/8/8/4k3/8/8/8/4K2Q w - - 0 1";
const KING_AND_ROOK: &str = "8/8/8/4k3/8/8/8/4K2R w - - 0 1";
const CORNERED_KING: &str = "7k/8/5K2/8/8/8/8/6Q1 w - - 0 1";
const POSITIONS: [(&str, &[&str]); 5] = [(START, OPENING), (ROOKS, &[]), (LOCKED_PAWNS, &[]), (MATE_IN_TWO, &[]), (KING_AND_ROOK, &[])];

struct SharedNetwork {
    move_ids: MoveIds,
    cache: SharedCache,
}

impl SharedNetwork {
    fn new() -> SharedNetwork {
        SharedNetwork {
            move_ids: MoveIds::new((0..MOVE_ID_TABLE_LENGTH).map(|index| (index % MOVE_IDS) as i16).collect()),
            cache: SharedCache::new(1 << 20),
        }
    }
}

fn shared_tree_after(fen: &str, moves: &[&str]) -> SharedTree {
    let tree = tree_after(fen, moves);
    SharedTree::new(tree.board, tree.keys).with_solid_links(8, 2)
}

fn seed_of(board: &[u8]) -> u64 {
    board.iter().fold(0, |hash, &token| mix(hash ^ u64::from(token)))
}

fn finish_round(tree: &SharedTree, round: u64, boards: &[u8], settings: &SearchSettings, network: &SharedNetwork, memory_limit: usize) {
    let _hold = tree.hold();
    let leaves = tree.take_round(round);
    let boards = boards.as_chunks::<TOKENS_PER_BOARD>().0;
    assert_eq!(leaves.len(), boards.len());
    let mut logits = Vec::new();
    for (index, (leaf, board)) in leaves.into_iter().zip(boards).enumerate() {
        let seed = seed_of(board);
        let answer = |column| answer(board, seed, column);
        //every other leaf is finished from its logits in move order, as the engine's lean rows give them
        if index % 2 == 0 {
            tree.finish_leaf(leaf, answer, ANSWER_WIDTH, &network.move_ids, Some(&network.cache), settings, memory_limit, &mut logits);
        } else {
            logits.clear();
            logits.extend(leaf.moves.iter().map(|&chess_move| answer(network.move_ids.id(chess_move, leaf.turn))));
            tree.finish_leaf_in_order(leaf, &mut logits, answer(ANSWER_WIDTH_LAST), Some(&network.cache), settings, memory_limit);
        }
    }
}

//search_in_flight's rounds on this thread; returns every round's boards
fn run_rounds(
    tree: &SharedTree,
    next_round: &AtomicU64,
    simulations: u32,
    in_flight: usize,
    most_leaves: u32,
    settings: &SearchSettings,
    tables: Option<&Tables>,
    network: &SharedNetwork,
    memory_limit: usize,
) -> Vec<Vec<u8>> {
    let (mut rounds, mut gathered) = (VecDeque::new(), Vec::new());
    loop {
        let room = tree.root().visits() as usize + tree.pending() < simulations as usize && !tree.is_full();
        if room && rounds.len() < in_flight {
            let leaves = (tree.root().visits() / 16).clamp(1, most_leaves);
            let id = next_round.fetch_add(1, Ordering::Relaxed);
            let round = Round { id, leaves, simulations: 3 * leaves, collisions: leaves, visit_limit: simulations, memory_limit };
            let mut boards = Vec::new();
            let queued = tree.gather(&round, settings, tables, Some(&network.cache), &mut boards);
            assert_eq!(boards.len(), queued.len() * TOKENS_PER_BOARD);
            tree.queue(id, queued);
            gathered.push(boards.clone());
            rounds.push_back((id, boards));
            continue;
        }
        let Some((id, boards)) = rounds.pop_front() else { break };
        finish_round(tree, id, &boards, settings, network, memory_limit);
    }
    gathered
}

fn shared_in_flight(
    tree: &SharedTree,
    simulations: u32,
    in_flight: usize,
    settings: &SearchSettings,
    tables: Option<&Tables>,
    network: &SharedNetwork,
    memory_limit: usize,
) -> Vec<Vec<u8>> {
    run_rounds(tree, &AtomicU64::new(0), simulations, in_flight, 64, settings, tables, network, memory_limit)
}

fn search_on_threads(
    tree: &SharedTree,
    threads: usize,
    simulations: u32,
    in_flight: usize,
    most_leaves: u32,
    settings: &SearchSettings,
    network: &SharedNetwork,
    memory_limit: usize,
) {
    crate::search::shared::JITTER.store(true, Ordering::Relaxed);
    let next_round = AtomicU64::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| run_rounds(tree, &next_round, simulations, in_flight, most_leaves, settings, None, network, memory_limit));
        }
    });
}

fn gather_with<'a>(
    settings: &'a SearchSettings,
    tables: Option<&'a Tables>,
) -> impl FnMut(&mut Tree, &Round, &Cache, &mut Vec<u8>) -> u32 + 'a {
    move |tree, round, cache, boards| tree.gather(round, settings, tables, Some(cache), boards)
}

fn proven_nodes(tree: &Tree) -> usize {
    tree.nodes.iter().filter(|node| node.proof.is_some()).count()
}

//the same, in a shared tree: no leaf stays waiting for the probe that failed, and the numbers are the search without tables'
#[test]
fn a_shared_search_with_tables_that_fail_is_the_search_without_tables() {
    let (tables, settings) = (failing_tables(), settings(true));
    for in_flight in [1, 3] {
        let (with, without) = (shared_tree_after(QUEEN_AND_PAWN, &[]), shared_tree_after(QUEEN_AND_PAWN, &[]));
        let rounds_with = shared_in_flight(&with, 3_000, in_flight, &settings, Some(&tables), &SharedNetwork::new(), usize::MAX);
        let rounds_without = shared_in_flight(&without, 3_000, in_flight, &settings, None, &SharedNetwork::new(), usize::MAX);
        assert_eq!(rounds_with, rounds_without);
        assert_eq!((with.pending(), with.root().visits()), (0, without.root().visits()));
        assert_eq!(with.principal_variation(), without.principal_variation());
    }
    assert!(tables.failures() > 0, "no probe failed");
    let (mut with, mut without) = (shared_tree_after(QUEEN_AND_PAWN, &[]), shared_tree_after(QUEEN_AND_PAWN, &[]));
    shared_in_flight(&with, 1_000, 1, &settings, Some(&tables), &SharedNetwork::new(), usize::MAX);
    shared_in_flight(&without, 1_000, 1, &settings, None, &SharedNetwork::new(), usize::MAX);
    let line = [best_edge(&with).expect("a move")];
    for (tree, tables) in [(&mut with, Some(&tables)), (&mut without, None)] {
        tree.move_root(&line);
        tree.drop_stale_leaves(&settings, tables);
        tree.drain();
    }
    assert_eq!(with.root().visits(), without.root().visits());
    assert_eq!(with.allocated_bytes(), without.allocated_bytes());
}

//a reader of the moves' nodes needs no gate: a node made solid beside it must not turn a list it has found into a slot's number
#[test]
fn a_list_made_solid_between_a_readers_loads_is_read_as_it_was() {
    let (settings, network) = (settings(false), SharedNetwork::new());
    let tree = SharedTree::new(tree_after(START, OPENING).board, tree_after(START, OPENING).keys).with_solid_links(1_000, 2);
    shared_in_flight(&tree, 600, 2, &settings, None, &network, usize::MAX);
    let root = tree.root();
    let found: Vec<_> = root.edges().map(|edge| (tree.child_index(root, edge), tree.visits(root, edge))).collect();
    assert!(found.iter().filter(|(child, _)| child.is_some()).count() > 5);
    assert_eq!(tree.lists_and_solids().1, 0);

    crate::search::shared::AFTER_CHILDREN_LOADED.set(Some(|tree, node| tree.solidify(node)));
    let read: Vec<_> = root.edges().map(|edge| (tree.child_index(root, edge), tree.visits(root, edge))).collect();
    crate::search::shared::AFTER_CHILDREN_LOADED.set(None);
    assert_eq!(read, found);
    assert!(tree.lists_and_solids().1 > 0, "no list was made solid");
}

//a descent over the memory limit gives up when no garbage is left and no thread is freeing any, and the room is looked at then: the
//thread that held the last batch may have freed it since the descent last looked
#[test]
fn a_descent_gets_the_room_that_the_last_batch_frees() {
    let (settings, network) = (settings(false), SharedNetwork::new());
    let searched = shared_tree_after(START, OPENING);
    shared_in_flight(&searched, 2_000, 2, &settings, None, &network, usize::MAX);
    let edge = best_edge(&searched).expect("a move");
    let moved = || {
        let mut tree = SharedTree::from_tree(&searched.to_tree());
        tree.move_root(&[edge]);
        tree
    };
    let drained = moved();
    assert!(drained.has_garbage());
    drained.drain();
    let limit = drained.allocated_bytes() + crate::search::shared::DESCENT_BYTES;
    let mut refused = 0;
    for _ in 0..300 {
        let tree = Arc::new(moved());
        tree.start_sweepers(2);
        refused += usize::from(!tree.reserve(limit, true));
        tree.quiet(|| tree.close());
        tree.join_sweepers();
    }
    assert_eq!(refused, 0, "{refused} descents of 300 were refused room that the garbage frees");
}

//the same, in a shared tree: and the leaf stops waiting and gives back its reservation, as any leaf does
#[test]
fn a_shared_answer_that_is_no_number_leaves_equal_priors_or_a_draw_and_is_not_cached() {
    let settings = SearchSettings { value_sharpness: 1.0, ..settings(false) };
    for (logit, value, equal, backed_up) in NOT_NUMBERS {
        let (tree, network) = (shared_tree_after(START, &[]), SharedNetwork::new());
        let round = Round { id: 0, leaves: 1, simulations: 1, collisions: 0, visit_limit: 100, memory_limit: 1 << 30 };
        let queued = tree.gather(&round, &settings, None, Some(&network.cache), &mut Vec::new());
        let (key, first_move) = (queued[0].key, network.move_ids.id(queued[0].moves[0], queued[0].turn));
        tree.queue(0, queued);
        let answer = |column: usize| match column {
            ANSWER_WIDTH_LAST => value,
            _ if column == first_move => logit,
            _ => 0.01 * column as f32,
        };
        {
            let _hold = tree.hold();
            for leaf in tree.take_round(0) {
                tree.finish_leaf(leaf, answer, ANSWER_WIDTH, &network.move_ids, Some(&network.cache), &settings, 1 << 30, &mut Vec::new());
            }
        }
        let root = tree.root();
        let equal_priors = root.edges().all(|edge| tree.prior(edge) == tree.prior(root.edges().start));
        assert_eq!((root.visits(), tree.pending(), tree.reserved_bytes(), equal_priors), (1, 0, 0, equal), "{logit} {value}");
        assert!(root.edges().all(|edge| tree.prior(edge).is_finite()), "{logit} {value}");
        assert_eq!(root.value_sum(), f64::from(1.0 - backed_up), "{logit} {value}");
        assert!(network.cache.get(key, &mut Vec::new()).is_none(), "{logit} {value}: the answer is cached");
    }
}

#[test]
fn one_thread_searches_as_tree_does() {
    let tables = tables();
    let mut proven = 0;
    for (fen, moves, simulations, certainty, tables) in cases(&tables) {
        let settings = settings(certainty);
        for in_flight in [1, 2, 3] {
            let mut old = tree_after(fen, moves);
            let old_rounds =
                search_in_flight(&mut old, simulations, in_flight, &settings, &mut Network::new(), gather_with(&settings, tables));
            let new = shared_tree_after(fen, moves);
            let new_rounds = shared_in_flight(&new, simulations, in_flight, &settings, tables, &SharedNetwork::new(), usize::MAX);
            assert!(old_rounds == new_rounds, "{fen}, {in_flight} in flight: the rounds' boards");
            assert_eq!(new.counts.orphans.load(Ordering::Relaxed), 0);
            let new = new.to_tree();
            assert_same_trees(&old, &new);
            proven += proven_nodes(&new);
        }
    }
    assert!(proven > 0, "{proven} proven nodes");
}

fn check_tree(tree: &SharedTree, settings: &SearchSettings) {
    walk(tree, tree.root_index(), &tree.board, &mut tree.keys.clone(), settings, false);
}

//a tree whose root moved: a leaf taken out as stale has no visit until it is searched again
fn check_reused_tree(tree: &SharedTree, settings: &SearchSettings) {
    walk(tree, tree.root_index(), &tree.board, &mut tree.keys.clone(), settings, true);
}

fn walk(tree: &SharedTree, id: usize, board: &Chess, keys: &mut Vec<u64>, settings: &SearchSettings, reused: bool) {
    let node = tree.node(id);
    let (visits, in_flight) = node.counts();
    assert_eq!(in_flight, 0, "a node with simulations in flight after the search");
    assert!(visits > 0 || reused, "a node of the tree without a visit");
    if visits == 0 {
        assert!(node.edges().is_empty(), "a node without a visit has moves");
        return;
    }
    let edges = node.edges();
    let has_visits = |child: &usize| !reused || tree.node(*child).visits() > 0;
    let children: Vec<(usize, usize)> =
        edges.clone().filter_map(|edge| tree.child_index(node, edge).map(|child| (edge, child))).filter(|(_, child)| has_visits(child)).collect();
    if edges.is_empty() {
        //a game's end or a repeat, reached again and again: every visit scored its result
        let legal_moves = board.legal_moves();
        let over = if id == tree.root_index() { None } else { game_result(board, keys, &legal_moves, settings.draw, None) };
        let (&leaf_key, path_keys) = keys[tree.keys.len() - 1..].split_last().expect("the root's key");
        let value = over.or_else(|| path_keys.contains(&leaf_key).then_some(settings.draw));
        match (node.proof(), value) {
            (Some(proof), Some(_)) => assert_close(node.value_sum(), f64::from(visits) * proof.value, visits, "a proven leaf's sum"),
            (None, Some(value)) => assert_close(node.value_sum(), f64::from(visits) * (1.0 - value), visits, "a leaf's sum"),
            (_, None) => panic!("a leaf the network never expanded and the rules do not end"),
        }
        return;
    }
    let mut tokens = Vec::new();
    encode_board(board, &mut tokens);
    let own = 1.0 - f64::from(sharpen(answer(&tokens, seed_of(&tokens), ANSWER_WIDTH - 1), settings.value_sharpness));
    let moves: u32 = children.iter().map(|&(_, child)| tree.node(child).visits()).sum();
    let below: f64 = children.iter().map(|&(_, child)| f64::from(tree.node(child).visits()) - tree.node(child).value_sum()).sum();
    match node.proof() {
        None => {
            assert_eq!(visits, 1 + moves, "visits of an unproven node");
            assert_close(node.value_sum(), own + below, visits, "an unproven node's sum");
        }
        Some(proof) => assert_close(node.value_sum(), f64::from(visits) * proof.value, visits, "a proven node's sum"),
    }
    let visited = python_sum(children.iter().map(|&(edge, _)| tree.prior(edge))).sqrt();
    assert_eq!(node.visited_policy_sqrt().to_bits(), visited.to_bits(), "visited policy");
    let priors: f64 = edges.clone().map(|edge| tree.prior(edge)).sum();
    assert!((priors - 1.0).abs() < 1e-3, "priors add up to {priors}");
    if let Some(proof) = node.proof().filter(|_| id != tree.root_index()) {
        let mut best = None::<Proof>;
        let mut all_proven = true;
        for edge in edges.clone() {
            match tree.child(node, edge).and_then(|child| child.proof()) {
                Some(found) if best.is_none_or(|best| proof_rank(&found) > proof_rank(&best)) => best = Some(found),
                Some(_) => {}
                None => all_proven = false,
            }
        }
        let best = best.expect("a proven node with a proven move");
        assert!(best.outcome == Outcome::Win || all_proven, "a proof without its moves' proofs");
        let outcome = match best.outcome {
            Outcome::Win => Outcome::Loss,
            Outcome::Draw => Outcome::Draw,
            Outcome::Loss => Outcome::Win,
        };
        assert_eq!(proof.outcome, outcome);
    }
    for (edge, child) in children {
        let mut next = board.clone();
        next.play_unchecked(tree.chess_move(edge).to_move(&next));
        keys.push(tree.node(child).key());
        walk(tree, child, &next, keys, settings, reused);
        keys.pop();
    }
}

fn assert_close(found: f64, expected: f64, visits: u32, what: &str) {
    assert!((found - expected).abs() <= 1e-9 * f64::from(visits.max(1)), "{what}: {found} against {expected} over {visits} visits");
}

fn search_each_position_on_threads(transpositions: bool, mut check: impl FnMut(&SharedTree, &SearchSettings)) {
    for (fen, moves) in POSITIONS {
        for certainty in [false, true] {
            let settings = SearchSettings { transpositions, ..settings(certainty) };
            for threads in [2, 4, 8, 24] {
                for in_flight in [1, 3] {
                    let tree = shared_tree_after(fen, moves).with_transpositions(transpositions);
                    search_on_threads(&tree, threads, 3_000, in_flight, 16, &settings, &SharedNetwork::new(), usize::MAX);
                    assert_eq!(tree.pending(), 0);
                    let visits = tree.root().visits();
                    assert!((3_000..3_000 + 64 * threads as u32).contains(&visits), "{visits}");
                    check(&tree, &settings);
                }
            }
        }
    }
}

#[test]
fn searches_on_many_threads_leave_a_consistent_tree() {
    search_each_position_on_threads(false, |tree, settings| {
        check_tree(tree, settings);
        let (lists, solids) = tree.lists_and_solids();
        assert!(lists > 0 && solids > 0, "{lists} lists, {solids} solid");
    });
}

#[test]
fn proofs_on_many_threads_leave_consistent_sums() {
    let settings = settings(true);
    let mut exclusive = 0;
    for fen in [KING_AND_QUEEN, KING_AND_ROOK, MATE_IN_TWO, CORNERED_KING] {
        for repeat in 0..12 {
            let tree = shared_tree_after(fen, &[]);
            search_on_threads(
                &tree,
                [2, 3, 5, 8, 16, 32][repeat % 6],
                4_000,
                1 + repeat % 3,
                16,
                &settings,
                &SharedNetwork::new(),
                usize::MAX,
            );
            check_tree(&tree, &settings);
            exclusive += tree.counts.exclusive.load(Ordering::Relaxed);
        }
    }
    assert!(exclusive > 100, "{exclusive} exclusive sections");
}

#[test]
fn a_search_ends_at_its_memory_limit_as_the_same_search() {
    let memory_limit = 4_000_000;
    let settings = settings(true);
    let limited = shared_tree_after(START, &[]);
    shared_in_flight(&limited, 1_000_000, 2, &settings, None, &SharedNetwork::new(), memory_limit);
    let reached = limited.root().visits();
    assert!(
        limited.is_full() && reached < 1_000_000 && limited.allocated_bytes() <= memory_limit,
        "{reached} {}",
        limited.allocated_bytes()
    );

    let unlimited = shared_tree_after(START, &[]);
    shared_in_flight(&unlimited, reached, 2, &settings, None, &SharedNetwork::new(), usize::MAX);
    assert!(!unlimited.is_full() && limited.allocated_bytes() > memory_limit / 2);
    assert_same_trees(&unlimited.to_tree(), &limited.to_tree());

    for threads in [2, 8] {
        let crowded = shared_tree_after(START, &[]);
        search_on_threads(&crowded, threads, 1_000_000, 2, 64, &settings, &SharedNetwork::new(), memory_limit);
        assert!(crowded.is_full() && crowded.pending() == 0 && crowded.allocated_bytes() <= memory_limit);
        check_tree(&crowded, &settings);
    }
}

#[test]
fn a_search_under_a_memory_limit_gives_back_every_reservation() {
    let settings = settings(true);
    let tree = shared_tree_after(START, &[]);
    shared_in_flight(&tree, 3_000, 2, &settings, None, &SharedNetwork::new(), 4_000_000);
    assert!(!tree.is_full());
    assert_eq!(tree.reserved_bytes(), 0, "one thread");
    for threads in [2, 8] {
        let crowded = shared_tree_after(START, &[]);
        search_on_threads(&crowded, threads, 3_000, 2, 16, &settings, &SharedNetwork::new(), 4_000_000);
        assert_eq!(crowded.reserved_bytes(), 0, "{threads} threads");
    }
}

#[test]
fn the_shared_cache_gives_back_what_threads_put_in() {
    let cache = SharedCache::new(1 << 16);
    let priors = |key: u64| -> Vec<f32> { (0..37).map(|index| (key % 1000 + index) as f32).collect() };
    std::thread::scope(|scope| {
        for worker in 0..4u64 {
            let (cache, priors) = (&cache, &priors);
            scope.spawn(move || {
                let mut found = Vec::new();
                for turn in 0..5_000u64 {
                    let key = mix(turn * 4 + worker);
                    cache.insert(key, &priors(key), key as f32);
                    let value = cache.get(key, &mut found).expect("the answer just inserted");
                    assert_eq!((found.clone(), value), (priors(key), key as f32));
                }
            });
        }
    });
    let mut found = Vec::new();
    for key in 0..4 * 5_000u64 {
        let key = mix(key);
        assert_eq!(cache.get(key, &mut found), Some(key as f32));
        assert_eq!(found, priors(key));
    }
    cache.clear();
    assert!(cache.get(mix(0), &mut found).is_none());
}

fn edges_of(tree: &SharedTree, line: &[PackedMove]) -> Vec<usize> {
    let mut node = tree.root_index();
    line.iter()
        .map(|&played| {
            let edge = tree.node(node).edges().find(|&edge| tree.chess_move(edge) == played).expect("the line's move");
            node = tree.child_index(tree.node(node), edge).expect("the line's node");
            edge
        })
        .collect()
}

//tests.rs advanced() on a SharedTree, with the walk stopped or not: the root moves one move, then the rest, and the garbage is freed
fn shared_advanced(
    tree: &SharedTree,
    line: &[PackedMove],
    threads: usize,
    walk_stopped: bool,
    settings: &SearchSettings,
    tables: Option<&Tables>,
) -> SharedTree {
    let mut advanced = SharedTree::from_tree(&tree.to_tree());
    let edges = edges_of(&advanced, line);
    let mut node = advanced.root_index();
    let places: Vec<u16> = edges
        .iter()
        .map(|&edge| {
            let place = (edge - advanced.node(node).edges().start) as u16;
            node = advanced.child_index(advanced.node(node), edge).expect("the line's node");
            place
        })
        .collect();
    advanced.move_root(&edges[..1]);
    let (never, always) = (AtomicBool::new(false), AtomicBool::new(true));
    let candidates = advanced.stale_candidates(threads, if walk_stopped { &always } else { &never });
    let rest = edges_of(&advanced, &line[1..]);
    advanced.move_root(&rest);
    assert!(advanced.root().proof().is_none(), "a root is never proven");
    match candidates {
        Some(candidates) => advanced.drop_stale_candidates(&places[1..], &candidates, settings, tables),
        None => advanced.drop_stale_leaves(settings, tables),
    }
    advanced.drain();
    advanced
}

//a tree whose garbage was all freed holds what the root reaches and nothing else
fn check_drained(tree: &SharedTree) {
    tree.check_refs();
    assert_eq!(tree.in_use(), tree.reachable(), "room in use that the root does not reach");
    if let Some(entries) = tree.table_entries() {
        assert_eq!(entries, tree.reachable().nodes, "table entries of nodes the root does not reach");
    }
}

//the root moved and the stale leaves dropped as a Tree's reuse does: before, halfway through and after the garbage is freed, the
//tree the root reaches is the Tree's compacted one
fn assert_moved_as_compacted(old: &Tree, moved: SharedTree) {
    assert_same_trees(old, &moved.to_reachable_tree());
    moved.check_refs();
    moved.reclaim(7);
    assert_same_trees(old, &moved.to_reachable_tree());
    moved.check_refs();
    moved.drain();
    assert_same_trees(old, &moved.to_reachable_tree());
    check_drained(&moved);
}

#[test]
fn moving_the_root_then_draining_leaves_the_tree_a_compacted_one_is() {
    let tables = tables();
    let (mut lines, mut stale, mut proven) = (0, 0, 0);
    //a draw of 0.5 takes as much from a parent as it leaves it: 0.35 tells which of the two a sign took
    let cases: Vec<_> = cases(&tables).into_iter().flat_map(|case| [(case, 0.5), (case, 0.35)]).collect();
    for ((fen, moves, simulations, certainty, tables), draw) in cases {
        let settings = SearchSettings { draw, ..settings(certainty) };
        let mut tree = tree_after(fen, moves);
        search(&mut tree, simulations, usize::MAX, &settings, tables, &mut Network::new());
        let shared = shared_tree_after(fen, moves);
        shared_in_flight(&shared, simulations, 1, &settings, tables, &SharedNetwork::new(), usize::MAX);
        assert_eq!(tree.root().visit_count, shared.root().visits());
        for line in reusable_lines(&tree) {
            let line_moves: Vec<PackedMove> = line.iter().map(|&edge| tree.chess_move(edge)).collect();
            let mut old = copied(&tree);
            stale += reused_move_by_move(&mut old, &line_moves, &settings, tables);

            let mut at_once = SharedTree::from_tree(&shared.to_tree());
            at_once.move_root(&edges_of(&at_once, &line_moves));
            at_once.drop_stale_leaves(&settings, tables);
            assert_moved_as_compacted(&old, at_once);
            for (threads, walk_stopped) in [(1, false), (1, true), (4, false), (4, true)] {
                let advanced = shared_advanced(&shared, &line_moves, threads, walk_stopped, &settings, tables);
                assert_same_trees(&old, &advanced.to_reachable_tree());
                check_drained(&advanced);
            }
            let mut moved = SharedTree::from_tree(&shared.to_tree());
            moved.move_root(&edges_of(&moved, &line_moves[..1]));
            let never = AtomicBool::new(false);
            assert_eq!(moved.stale_candidates(4, &never), moved.stale_candidates(1, &never));
            lines += 1;
            proven += proven_nodes(&old);
        }
    }
    assert!(lines >= 15 && stale > 0 && proven > 0, "{lines} lines, {stale} stale leaves, {proven} proven nodes");
}

#[test]
fn moving_the_root_of_a_tree_with_transpositions_then_draining_leaves_the_tree_a_compacted_one_is() {
    let tables = tables();
    let mut lines = 0;
    for (fen, moves, simulations, certainty, tables) in cases(&tables) {
        let settings = graph_settings(certainty);
        let mut tree = graph_tree_after(fen, moves);
        search(&mut tree, simulations, usize::MAX, &settings, tables, &mut Network::new());
        for line in reusable_lines(&tree) {
            let line_moves: Vec<PackedMove> = line.iter().map(|&edge| tree.chess_move(edge)).collect();
            let mut old = copied(&tree);
            old.play_line(&line);
            old.drop_stale_leaves(&settings, tables);

            //a graph is not copied by from_tree: the search is made again, as the tree's was
            let mut shared = graph_shared_tree_after(fen, moves);
            shared_in_flight(&shared, simulations, 1, &settings, tables, &SharedNetwork::new(), usize::MAX);
            assert_same_trees(&tree, &shared.to_tree());
            shared.move_root(&edges_of(&shared, &line_moves));
            shared.drop_stale_leaves(&settings, tables);
            assert_moved_as_compacted(&old, shared);
            lines += 1;
        }
    }
    assert!(lines >= 15, "{lines} lines");
}

//a proof hides the leaves below it from the stale walk unless it is the new root
#[test]
fn proven_nodes_hide_stale_leaves_from_a_shared_trees_advance_as_from_the_walk() {
    let settings = settings(true);
    let mut tree = tree_after(LOCKED_PAWNS, &[]);
    search(&mut tree, 10_000, usize::MAX, &settings, None, &mut Network::new());
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
    let line_moves: Vec<PackedMove> = line.iter().map(|&edge| tree.chess_move(edge)).collect();
    let draw = Proof { outcome: Outcome::Draw, value: 0.5, plies: 0 };
    for proven in [path[0], path[1]] {
        let mut proved = copied(&tree);
        proved.nodes[proven].proof = Some(draw);
        let mut walked = copied(&proved);
        walked.play_line(&line);
        walked.drop_stale_leaves(&settings, None);
        let advanced = shared_advanced(&SharedTree::from_tree(&proved), &line_moves, 4, false, &settings, None);
        assert_same_trees(&walked, &advanced.to_reachable_tree());
    }
}

#[test]
fn a_closed_tree_gathers_nothing() {
    let (settings, network) = (settings(true), SharedNetwork::new());
    let mut tree = shared_tree_after(START, &[]);
    shared_in_flight(&tree, 200, 1, &settings, None, &network, usize::MAX);
    let round = Round { id: 0, leaves: 8, simulations: 24, collisions: 8, visit_limit: u32::MAX, memory_limit: usize::MAX };
    let gathered = |tree: &SharedTree| tree.gather(&round, &settings, None, Some(&network.cache), &mut Vec::new()).len();
    assert!(gathered(&tree) > 0);
    tree.close();
    assert_eq!(gathered(&tree), 0);
    tree.reopen();
    assert!(gathered(&tree) > 0);
}

#[test]
fn rounds_without_leaves_queue_nothing() {
    let tree = shared_tree_after(START, &[]);
    for round in 0..10_000 {
        tree.queue(round, Vec::new());
    }
    assert_eq!(tree.queued_rounds(), 0);
}

#[test]
fn proofs_made_after_the_leaves_were_visited_rewrite_the_sums_as_tree_does() {
    //a draw worth less the second time, so that a stalemate's proof rewrites what its visits scored
    let (without, with) = (settings(false), SearchSettings { draw: 0.3, ..settings(true) });
    for fen in [MATE_IN_TWO, KING_AND_ROOK, CORNERED_KING] {
        let mut old = tree_after(fen, &[]);
        search_in_flight(&mut old, 3_000, 1, &without, &mut Network::new(), gather_with(&without, None));
        search_in_flight(&mut old, 8_000, 1, &with, &mut Network::new(), gather_with(&with, None));
        let new = shared_tree_after(fen, &[]);
        shared_in_flight(&new, 3_000, 1, &without, None, &SharedNetwork::new(), usize::MAX);
        shared_in_flight(&new, 8_000, 1, &with, None, &SharedNetwork::new(), usize::MAX);
        assert_same_trees(&old, &new.to_tree());
        assert!(proven_nodes(&old) > 0, "{fen}");
    }
}

#[test]
fn tables_met_after_the_leaves_were_visited_prove_them_as_tree_does() {
    let Some(tables) = tables() else {
        eprintln!("no syzygy tables: the ending they score is left out");
        return;
    };
    let settings = settings(true);
    let mut old = tree_after(PAWNS_AND_ROOKS, &[]);
    search_in_flight(&mut old, 4_000, 1, &settings, &mut Network::new(), gather_with(&settings, None));
    let before = proven_nodes(&old);
    search_in_flight(&mut old, 12_000, 1, &settings, &mut Network::new(), gather_with(&settings, Some(&tables)));
    let new = shared_tree_after(PAWNS_AND_ROOKS, &[]);
    shared_in_flight(&new, 4_000, 1, &settings, None, &SharedNetwork::new(), usize::MAX);
    shared_in_flight(&new, 12_000, 1, &settings, Some(&tables), &SharedNetwork::new(), usize::MAX);
    assert_same_trees(&old, &new.to_tree());
    assert!(proven_nodes(&old) > before + 5);
}

//a leaf taken after the close would return its round to a search already gone from Searches
#[test]
fn a_call_takes_no_leaf_once_its_search_has_ended() {
    let (settings, network) = (settings(true), SharedNetwork::new());
    let round = Round { id: 0, leaves: 30_000, simulations: 200_000, collisions: 150_000, visit_limit: u32::MAX, memory_limit: usize::MAX };
    //the close lands between two descents depending on the threads' timing: a few tries
    for _ in 0..8 {
        let tree = shared_tree_after(START, &[]);
        shared_in_flight(&tree, 20_000, 2, &settings, None, &network, usize::MAX);
        let (gathered, at_close) = std::thread::scope(|scope| {
            let gathering = scope.spawn(|| tree.gather(&round, &settings, None, Some(&network.cache), &mut Vec::new()).len());
            std::thread::sleep(Duration::from_millis(5));
            let at_close = tree.quiet(|| {
                tree.close();
                tree.pending()
            });
            (gathering.join().expect("a thread"), at_close)
        });
        assert!(gathered < round.leaves as usize, "the call ended before the search did: {gathered} leaves");
        assert_eq!(tree.pending(), at_close, "leaves taken after the close");
    }
}

//timing: a few tries, as the machine may be busy
#[test]
fn a_thread_that_wants_the_tree_to_itself_waits_for_a_descent_not_a_call() {
    let (settings, network) = (settings(true), SharedNetwork::new());
    let tree = shared_tree_after(START, &[]);
    shared_in_flight(&tree, 20_000, 2, &settings, None, &network, usize::MAX);
    let round = Round { id: 0, leaves: 30_000, simulations: 200_000, collisions: 150_000, visit_limit: u32::MAX, memory_limit: usize::MAX };
    let a_fraction = |wait: Duration, call: Duration| wait * 4 < call && call > Duration::from_millis(30);
    let mut tries = Vec::new();
    for _ in 0..5 {
        let (call, wait) = std::thread::scope(|scope| {
            let gathering = scope.spawn(|| {
                let started = Instant::now();
                tree.gather(&round, &settings, None, Some(&network.cache), &mut Vec::new());
                started.elapsed()
            });
            std::thread::sleep(Duration::from_millis(5));
            let started = Instant::now();
            tree.quiet(|| ());
            let waited = started.elapsed();
            (gathering.join().expect("a thread"), waited)
        });
        tries.push((wait, call));
        if a_fraction(wait, call) {
            break;
        }
    }
    let &(wait, call) = tries.last().expect("a try");
    let gave_way = tree.counts.gave_way.load(Ordering::Relaxed);
    assert!(gave_way > 0 && a_fraction(wait, call), "waits and calls {tries:?}, gave way {gave_way} times");
}

fn graph_settings(certainty: bool) -> SearchSettings {
    SearchSettings { transpositions: true, ..settings(certainty) }
}

fn graph_tree_after(fen: &str, moves: &[&str]) -> Tree {
    tree_after(fen, moves).with_transpositions(true)
}

fn graph_shared_tree_after(fen: &str, moves: &[&str]) -> SharedTree {
    shared_tree_after(fen, moves).with_transpositions(true)
}

fn check_graph(tree: &Tree) -> usize {
    let graph = tree.graph.as_ref().expect("a tree with transpositions");
    assert_eq!(graph.halfmoves.len(), tree.nodes.len(), "a fifty-move count for every node");
    let mut incoming = vec![0u32; tree.nodes.len()];
    let (mut seen, mut order) = (vec![false; tree.nodes.len()], vec![tree.root_index()]);
    seen[tree.root_index()] = true;
    while let Some(id) = order.pop() {
        let node = &tree.nodes[id];
        assert_eq!(node.in_flight, 0, "a node with simulations in flight after the search");
        let mut moves = 0;
        if !listed(node.children) {
            let first = (node.children & !SOLID) as usize;
            let slots = &tree.slots[first..first + node.edges().len()];
            assert!(slots.iter().all(|slot| slot.child != NO_NODE || slot.visits == 0), "a move without a node, visited");
        }
        for (_, link) in tree.linked(id) {
            let child = tree.link_child(link);
            moves += tree.move_visits(link);
            incoming[child as usize] += 1;
            if !std::mem::replace(&mut seen[child as usize], true) {
                order.push(child as usize);
            }
        }
        //a root's proof is dropped: the visits it stopped with belong to no move
        if node.has_edges() && node.proof.is_none() && id != tree.root_index() {
            assert_eq!(node.visit_count, 1 + moves, "visits of an unproven node and its moves'");
        }
        if node.visit_count > 0 {
            let mean = node.mean_score();
            assert!((-1e-9..=1.0 + 1e-9).contains(&mean), "a mean of {mean}");
        }
    }
    for (&entry, &id) in &graph.table {
        assert_eq!(graph_key(tree.node_keys[id as usize], u32::from(graph.halfmoves[id as usize])), entry, "a node under another key");
    }
    assert_eq!(graph.table.len(), tree.nodes.len(), "nodes the table does not have, or one position made twice");
    incoming.iter().filter(|&&moves| moves > 1).count()
}

#[test]
fn one_thread_with_transpositions_searches_as_tree_does() {
    let tables = tables();
    let mut shared = 0;
    for (fen, moves, simulations, certainty, tables) in cases(&tables) {
        let settings = graph_settings(certainty);
        for in_flight in [1, 2, 3] {
            let mut old = graph_tree_after(fen, moves);
            let old_rounds =
                search_in_flight(&mut old, simulations, in_flight, &settings, &mut Network::new(), gather_with(&settings, tables));
            let new = graph_shared_tree_after(fen, moves);
            let new_rounds = shared_in_flight(&new, simulations, in_flight, &settings, tables, &SharedNetwork::new(), usize::MAX);
            assert!(old_rounds == new_rounds, "{fen}, {in_flight} in flight: the rounds' boards");
            assert_eq!(new.counts.orphans.load(Ordering::Relaxed), 0);
            //the move choice too, whose ties a node's own mean breaks
            let new_line: Vec<PackedMove> = new.principal_variation().into_iter().map(|edge| new.chess_move(edge)).collect();
            let old_line: Vec<PackedMove> = old.principal_variation().into_iter().map(|edge| old.chess_move(edge)).collect();
            assert_eq!(new_line, old_line);
            assert_same_trees(&old, &new.to_tree());
            shared += check_graph(&old);
        }
    }
    assert!(shared > 0, "no node shared by two moves");
}

#[test]
fn searches_with_transpositions_on_many_threads_leave_a_consistent_graph() {
    let mut shared = 0;
    search_each_position_on_threads(true, |tree, _| shared += check_graph(&tree.to_tree()));
    assert!(shared > 0, "no node shared by two moves");
}

fn reachable(tree: &Tree, from: usize) -> usize {
    let mut seen = vec![false; tree.nodes.len()];
    let mut order = vec![from];
    seen[from] = true;
    while let Some(id) = order.pop() {
        for (_, link) in tree.linked(id) {
            let child = tree.link_child(link) as usize;
            if !std::mem::replace(&mut seen[child], true) {
                order.push(child);
            }
        }
    }
    seen.iter().filter(|&&seen| seen).count()
}

fn line_after(tree: &Tree, line: &[PackedMove]) -> Vec<usize> {
    let mut node = tree.root_index();
    line.iter()
        .map(|&played| {
            let edge = tree.nodes[node].edges().find(|&edge| tree.chess_move(edge) == played).expect("the line's move");
            node = tree.child_index(node, edge).expect("the line's node");
            edge
        })
        .collect()
}

#[test]
fn reusing_a_tree_with_transpositions_keeps_it_whole() {
    let tables = tables();
    let mut lines = 0;
    for (fen, moves, simulations, certainty, tables) in cases(&tables) {
        let settings = graph_settings(certainty);
        let mut tree = graph_tree_after(fen, moves);
        search(&mut tree, simulations, usize::MAX, &settings, tables, &mut Network::new());
        for line in reusable_lines(&tree) {
            let line_moves: Vec<PackedMove> = line.iter().map(|&edge| tree.chess_move(edge)).collect();
            let new_root = line_end(&tree, &line);
            let (visits, value_sum) = (tree.nodes[new_root].visit_count, tree.nodes[new_root].value_sum);
            let mut at_once = copied(&tree);
            at_once.play_line(&line);
            at_once.drop_stale_leaves(&settings, tables);
            check_graph(&at_once);
            assert_eq!((at_once.root().visit_count, at_once.root().value_sum.to_bits()), (visits, value_sum.to_bits()));
            assert_eq!(at_once.nodes.len(), reachable(&tree, new_root), "a node copied twice or one left out");

            let mut advanced = copied(&tree);
            advanced.move_root(&line[..1]);
            let never = AtomicBool::new(false);
            let candidates = advanced.stale_candidates(4, &never).expect("not stopped");
            assert!(candidates.is_empty(), "candidates with transpositions");
            advanced.compact(4, Some(&never));
            let rest = line_after(&advanced, &line_moves[1..]);
            advanced.move_root(&rest);
            advanced.drop_stale_candidates(&[], &candidates, &settings, tables);
            advanced.compact(1, None);
            assert_same_trees(&at_once, &advanced);
            lines += 1;
        }
    }
    assert!(lines >= 15, "{lines} lines");
}

//the root's most visited move, if the node it leads to has moves to search on
fn best_edge(tree: &SharedTree) -> Option<usize> {
    let edge = *tree.principal_variation().first()?;
    tree.child(tree.root(), edge).filter(|child| child.has_edges()).map(|_| edge)
}

//a search over: the sweepers leave, and the tree is its owner's again, open for the next
fn owned_again(tree: Arc<SharedTree>) -> SharedTree {
    tree.quiet(|| tree.close());
    tree.join_sweepers();
    let mut tree = Arc::into_inner(tree).expect("no one else holds the tree");
    tree.reopen();
    tree
}

//the visits of the nodes that moves of the reachable tree do not account for: a node's visits come with its moves', but those of
//a node a dead node's move led to are in its numbers and not in any move that is left
fn visits_without_a_move(tree: &Tree) -> usize {
    let mut by_moves = vec![0u32; tree.nodes.len()];
    for id in 0..tree.nodes.len() {
        for (_, link) in tree.linked(id) {
            by_moves[tree.link_child(link) as usize] += tree.move_visits(link);
        }
    }
    (1..tree.nodes.len()).filter(|&id| tree.nodes[id].visit_count > by_moves[id]).count()
}

#[test]
fn continuing_a_reused_tree_searches_as_a_compacted_one_does() {
    let tables = tables();
    let mut continued = 0;
    for transpositions in [false, true] {
        for (fen, moves, simulations, certainty, tables) in cases(&tables).into_iter().take(4) {
            let settings = SearchSettings { transpositions, ..settings(certainty) };
            let mut tree = tree_after(fen, moves).with_transpositions(transpositions);
            search(&mut tree, simulations, usize::MAX, &settings, tables, &mut Network::new());
            for line in reusable_lines(&tree).into_iter().take(2) {
                let line_moves: Vec<PackedMove> = line.iter().map(|&edge| tree.chess_move(edge)).collect();
                let mut old = copied(&tree);
                old.play_line(&line);
                old.drop_stale_leaves(&settings, tables);
                search(&mut old, simulations + 3_000, usize::MAX, &settings, tables, &mut Network::new());
                old.compact(1, None);

                let mut shared = shared_tree_after(fen, moves).with_transpositions(transpositions);
                shared_in_flight(&shared, simulations, 1, &settings, tables, &SharedNetwork::new(), usize::MAX);
                shared.move_root(&edges_of(&shared, &line_moves));
                shared.drop_stale_leaves(&settings, tables);
                shared.drain();
                shared_in_flight(&shared, simulations + 3_000, 1, &settings, tables, &SharedNetwork::new(), usize::MAX);
                assert_same_trees(&old, &shared.to_reachable_tree());
                shared.move_root(&[]);
                shared.drain();
                check_drained(&shared);
                continued += 1;
            }
        }
    }
    assert!(continued >= 12, "{continued} continued searches");
}

#[test]
fn a_dead_node_is_revived_by_a_transposition_before_the_sweep() {
    let tables = tables();
    let (mut revived_trees, mut lazy_visits, mut strict_visits) = (0, 0, 0);
    for (fen, moves, simulations, certainty, tables) in cases(&tables).into_iter().take(4) {
        let settings = graph_settings(certainty);
        let built = || {
            let tree = graph_shared_tree_after(fen, moves);
            shared_in_flight(&tree, simulations, 1, &settings, tables, &SharedNetwork::new(), usize::MAX);
            tree
        };
        let lines: Vec<Vec<PackedMove>> = {
            let tree = built().to_tree();
            reusable_lines(&tree).into_iter().take(2).map(|line| line.iter().map(|&edge| tree.chess_move(edge)).collect()).collect()
        };
        for line in lines {
            //the same search twice: the dead rest of the tree freed at once, and not at all before the search goes on
            let mut strict = built();
            strict.move_root(&edges_of(&strict, &line));
            strict.drain();
            shared_in_flight(&strict, simulations + 2_000, 1, &settings, tables, &SharedNetwork::new(), usize::MAX);
            let strict_tree = strict.to_reachable_tree();
            check_graph(&strict_tree);
            strict.move_root(&[]);
            strict.drain();
            check_drained(&strict);

            //halfway, the first dead nodes freed: their children have no references left, and their table entries are made over
            for freed in [0, 1, 40] {
                let mut lazy = built();
                lazy.move_root(&edges_of(&lazy, &line));
                lazy.reclaim(freed);
                shared_in_flight(&lazy, simulations + 2_000, 1, &settings, tables, &SharedNetwork::new(), usize::MAX);
                lazy.check_refs();
                let lazy_tree = lazy.to_reachable_tree();
                check_graph(&lazy_tree);
                lazy.move_root(&[]);
                lazy.drain();
                assert_same_trees(&lazy_tree, &lazy.to_reachable_tree());
                check_drained(&lazy);
                if freed == 0 {
                    lazy_visits += visits_without_a_move(&lazy_tree);
                    strict_visits += visits_without_a_move(&strict_tree);
                    revived_trees += usize::from(!trees_alike(&lazy_tree, &strict_tree));
                }
            }
        }
    }
    assert!(revived_trees > 0, "no dead node was revived");
    assert!(lazy_visits > strict_visits, "{lazy_visits} nodes carry old visits against {strict_visits} in a tree made afresh");
}

//whether two trees have the same nodes' numbers, whatever the order
fn trees_alike(one: &Tree, other: &Tree) -> bool {
    let numbers = |tree: &Tree| -> Vec<(u32, u64)> {
        let mut numbers: Vec<_> = tree.nodes.iter().map(|node| (node.visit_count, node.value_sum.to_bits())).collect();
        numbers.sort_unstable();
        numbers
    };
    numbers(one) == numbers(other)
}

//the checks of a tree no thread works in, whatever garbage it holds; returns the nodes that several moves lead to
fn check_after_search(tree: &SharedTree, transpositions: bool, certainty: bool, settings: &SearchSettings) -> usize {
    tree.check_refs();
    if transpositions {
        return check_graph(&tree.to_reachable_tree());
    }
    if !certainty {
        check_reused_tree(tree, settings);
    }
    0
}

#[test]
fn gc_workers_beside_searching_threads_lose_no_live_node() {
    let network = SharedNetwork::new();
    let (mut reclaimed, mut shared_nodes) = (0, 0);
    for transpositions in [false, true] {
        for certainty in [false, true] {
            let settings = SearchSettings { transpositions, ..settings(certainty) };
            for threads in [2, 8, 24] {
                let mut tree = shared_tree_after(START, OPENING).with_transpositions(transpositions);
                for _ in 0..8 {
                    let searching = Arc::new(tree);
                    searching.start_sweepers(2);
                    let target = searching.root().visits() + 1_500;
                    search_on_threads(&searching, threads, target, 2, 16, &settings, &network, usize::MAX);
                    tree = owned_again(searching);
                    shared_nodes += check_after_search(&tree, transpositions, certainty, &settings);
                    reclaimed += tree.counts.reclaimed.load(Ordering::Relaxed);
                    let Some(edge) = best_edge(&tree) else { break };
                    tree.move_root(&[edge]);
                    //a repeat or a game's end of the old root's game may be neither from the new root's
                    tree.drop_stale_leaves(&settings, None);
                }
                //whatever the sweepers left is freed at the end
                tree.drain();
                check_drained(&tree);
            }
        }
    }
    assert!(reclaimed > 0 && shared_nodes > 0, "{reclaimed} nodes freed, {shared_nodes} shared");
}

#[test]
fn lost_races_free_their_references() {
    let network = SharedNetwork::new();
    for transpositions in [false, true] {
        let settings = SearchSettings { transpositions, ..settings(true) };
        let mut orphans = 0;
        for _ in 0..6 {
            let mut tree = shared_tree_after(START, &[]).with_transpositions(transpositions);
            search_on_threads(&tree, 24, 3_000, 3, 16, &settings, &network, usize::MAX);
            orphans += tree.counts.orphans.load(Ordering::Relaxed);
            //the node and the link a thread that lost made are in the pools again, and every reference is one of the moves'
            tree.check_refs();
            tree.move_root(&[]);
            tree.drain();
            check_drained(&tree);
        }
        assert!(orphans > 0, "no thread lost a race, transpositions {transpositions}");
    }
}

//a game of `moves` moves of the engine's kind: a search, the most visited move played, again. The searches end at `visits` at the
//root, or only when the memory is full; returns the footprint (arenas' bytes) the tree had at each move and its last owner
fn play_a_game(moves: usize, visits: Option<u32>, limit: usize, sweepers: usize, check_footprint: bool) -> SharedTree {
    let (settings, network) = (settings(true), SharedNetwork::new());
    let fresh_visits = {
        let tree = shared_tree_after(START, OPENING);
        shared_in_flight(&tree, visits.unwrap_or(u32::MAX >> 1), 2, &settings, None, &network, limit);
        tree.root().visits()
    };
    let mut tree = shared_tree_after(START, OPENING);
    let (mut restarts, mut played) = (0, 0);
    while played < moves {
        let searching = Arc::new(tree);
        searching.start_sweepers(sweepers);
        //the room is looked at between steps of the search
        while !searching.is_full() && visits.is_none_or(|visits| searching.root().visits() < visits) {
            let target = searching.root().visits() + 50;
            shared_in_flight(&searching, visits.map_or(target, |visits| target.min(visits)), 2, &settings, None, &network, limit);
            assert!(searching.allocated_bytes() <= limit, "{} bytes of {limit}", searching.allocated_bytes());
        }
        assert!(!searching.has_garbage() || !searching.is_full(), "a search ended full with garbage left");
        let reached = f64::from(searching.root().visits());
        assert!(reached >= 0.8 * f64::from(fresh_visits), "move {played}: {reached} visits against a fresh tree's {fresh_visits}");
        tree = owned_again(searching);
        assert!(
            !check_footprint || played < 20 || !tree.sprawls(limit),
            "move {played}: {} bytes of arenas for a limit of {limit}; (arena, in use) of nodes, edges, links and slots: {:?}",
            tree.footprint_bytes(),
            tree.arena_bytes()
        );
        match best_edge(&tree) {
            Some(edge) => tree.move_root(&[edge]),
            None => {
                tree = shared_tree_after(START, OPENING);
                restarts += 1;
            }
        }
        played += 1;
    }
    assert!(restarts <= 3, "{restarts} games ended");
    tree
}

fn one_trees_bytes(visits: u32) -> usize {
    let tree = shared_tree_after(START, OPENING);
    shared_in_flight(&tree, visits, 2, &settings(true), None, &SharedNetwork::new(), usize::MAX);
    tree.allocated_bytes()
}

//the memory is more than a move's tree and the last move's dead rest: descents seldom free any, and the arenas stay near the limit
#[test]
fn a_game_stays_within_the_memory_limit() {
    let limit = one_trees_bytes(800) * 5 / 2;
    for sweepers in [0, 2] {
        play_a_game(200, Some(800), limit, sweepers, true);
    }
}

//the memory is full at every move and the last move's dead rest is more than the room left: the descents or the sweepers free
//it as the search needs it, and the freed room is used again
#[test]
fn a_game_that_fills_the_memory_every_move_keeps_to_the_limit() {
    let limit = one_trees_bytes(800) * 8 / 5;
    let tree = play_a_game(120, None, limit, 0, false);
    assert!(tree.counts.assists.load(Ordering::Relaxed) > 0, "no descent freed any garbage");
    //the arenas have the room freed runs of every length left, but not much more than the limit: the runs are joined
    assert!(tree.footprint_bytes() <= limit * 3 / 2, "{} bytes of arenas for a limit of {limit}: {:?}", tree.footprint_bytes(), tree.arena_bytes());
    play_a_game(120, None, limit, 2, false);
}

#[test]
fn search_move_drain_100_times_returns_to_the_same_bytes() {
    let (settings, network) = (settings(true), SharedNetwork::new());
    let mut tree = shared_tree_after(START, OPENING);
    let (mut footprints, mut restarts) = (Vec::new(), 0);
    while footprints.len() < 100 {
        let target = tree.root().visits() + 500;
        shared_in_flight(&tree, target, 2, &settings, None, &network, usize::MAX);
        match best_edge(&tree) {
            Some(edge) => tree.move_root(&[edge]),
            None => {
                tree = shared_tree_after(START, OPENING);
                restarts += 1;
                continue;
            }
        }
        tree.drain();
        //the room in use is what the root reaches: no node, link, slot or edge is left behind by a move, a list made solid or a loss
        check_drained(&tree);
        footprints.push(tree.footprint_bytes());
    }
    assert!(restarts <= 3, "{restarts} games ended");
    let (settled, last) = (footprints[20..].iter().copied().min().expect("moves"), footprints[99]);
    assert!(last <= settled * 3, "arenas of {last} bytes against {settled} a while ago");
}

#[test]
fn a_kept_tree_is_owned_again_after_every_search() {
    let (settings, network) = (graph_settings(true), SharedNetwork::new());
    let mut tree = graph_shared_tree_after(START, OPENING);
    for _ in 0..50 {
        let searching = Arc::new(tree);
        searching.start_sweepers(2);
        let target = searching.root().visits() + 300;
        search_on_threads(&searching, 4, target, 2, 16, &settings, &network, usize::MAX);
        searching.quiet(|| searching.close());
        searching.join_sweepers();
        assert_eq!(Arc::weak_count(&searching), 0, "a sweeper still holds the tree");
        tree = Arc::into_inner(searching).expect("the last owner");
        tree.reopen();
        match best_edge(&tree) {
            Some(edge) => tree.move_root(&[edge]),
            None => tree = graph_shared_tree_after(START, OPENING),
        }
    }

    //a tree dropped with its sweepers at work: they find it gone and leave
    let searching = Arc::new(graph_shared_tree_after(START, OPENING));
    search_on_threads(&searching, 8, 20_000, 2, 16, &settings, &network, usize::MAX);
    let mut tree = owned_again(searching);
    tree.move_root(&[best_edge(&tree).expect("a move")]);
    let searching = Arc::new(tree);
    assert!(searching.has_garbage());
    searching.start_sweepers(2);
    let sweepers = searching.take_sweepers();
    assert_eq!(sweepers.len(), 2);
    drop(searching);
    let (done, left) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        sweepers.into_iter().for_each(|sweeper| sweeper.join().expect("a sweeper"));
        done.send(()).expect("the test");
    });
    left.recv_timeout(Duration::from_secs(20)).expect("the sweepers leave once the tree is gone");
}

#[test]
fn starting_on_a_reused_300k_node_tree_takes_under_20_ms() {
    let (settings, network) = (graph_settings(true), SharedNetwork::new());
    let mut tree = graph_shared_tree_after(START, OPENING);
    search_on_threads(&tree, 8, 340_000, 3, 64, &settings, &network, usize::MAX);
    let nodes = tree.in_use().nodes;
    assert!(nodes >= 300_000, "{nodes} nodes");
    let edge = best_edge(&tree).expect("a move");
    let started = Instant::now();
    let searching = Arc::new(tree);
    searching.join_sweepers();
    tree = Arc::into_inner(searching).expect("the last owner");
    tree.reopen();
    assert!(!tree.sprawls(usize::MAX));
    tree.move_root(&[edge]);
    tree.drop_stale_leaves(&settings, None);
    let searching = Arc::new(tree);
    searching.start_sweepers(2);
    let took = started.elapsed();
    assert!(took < Duration::from_millis(20), "starting took {took:?}");

    //the search goes on beside the sweepers, which leave their garbage when it ends
    let target = searching.root().visits() + 2_000;
    search_on_threads(&searching, 4, target, 2, 16, &settings, &network, usize::MAX);
    let mut tree = owned_again(searching);
    tree.check_refs();
    tree.move_root(&[]);
    tree.drain();
    check_drained(&tree);
    assert!(tree.counts.reclaimed.load(Ordering::Relaxed) > 0);
}

//gather_batch against gather and itself, in searches with rounds in flight as the engine's lanes have them
use super::*;
use crate::search::batch::{Output, Scratch};
use crate::workers::Workers;
use rand::SeedableRng;
use std::collections::BTreeMap;
use std::time::Duration;

pub(super) const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
pub(super) const OPENING: &[&str] = &["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7", "e2e3", "e8g8", "g1f3", "b8d7"];
pub(super) const ROOKS: &str = "8/8/4k3/8/8/4K3/R7/7r w - - 0 1";
pub(super) const LOCKED_PAWNS: &str = "4k3/8/8/1p1p1p1p/1P1P1P1P/8/8/4K3 w - - 0 1";
pub(super) const MATE_IN_TWO: &str = "k7/8/2K5/8/8/8/8/7R w - - 0 1";
pub(super) const PAWNS_AND_ROOKS: &str = "8/8/4k3/3p4/3P4/4K3/R7/7r w - - 0 1";

//the positions of the reuse test: a middlegame, endings with repeats, a mate certainty proves, and the tables' ending
pub(super) fn cases(tables: &Option<Tables>) -> Vec<(&'static str, &'static [&'static str], u32, bool, Option<&Tables>)> {
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

//search's rounds, up to in_flight waiting, oldest finished first (finish_batch on finishers); every round's boards
pub(super) fn search_in_flight(
    tree: &mut Tree,
    simulations: u32,
    in_flight: usize,
    settings: &SearchSettings,
    network: &mut Network,
    finishers: Option<&Workers>,
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
        let boards = boards.as_chunks::<TOKENS_PER_BOARD>().0;
        let seeds: Vec<u64> = boards.iter().map(|board| board.iter().fold(0, |hash, &token| mix(hash ^ u64::from(token)))).collect();
        match finishers {
            Some(workers) => {
                let answers = |leaf: usize, column: usize| answer(&boards[leaf], seeds[leaf], column);
                tree.finish_batch(id, boards.len(), &answers, ANSWER_WIDTH, &network.move_ids, Some(&mut network.cache), settings, workers);
            }
            None => {
                for (board, &seed) in boards.iter().zip(&seeds) {
                    let answer = |column| answer(board, seed, column);
                    tree.finish_leaf(id, answer, ANSWER_WIDTH, &network.move_ids, Some(&mut network.cache), settings);
                }
            }
        }
    }
    gathered
}

//the first two moves of the tree's expected line, if the root can move down them (both have nodes, the second moves)
fn line_down(tree: &Tree) -> Option<Vec<usize>> {
    let line: Vec<usize> = tree.principal_variation().into_iter().take(2).collect();
    let nodes: Vec<u32> = line.iter().map(|&edge| tree.children[edge]).collect();
    (nodes.len() == 2 && !nodes.contains(&NO_NODE) && tree.nodes[nodes[1] as usize].has_edges()).then_some(line)
}

//search_in_flight, then the root moved down two moves in place (move_root: no longer node 0) and the search carried on
//to as many visits there
fn search_moving_on(
    tree: &mut Tree,
    simulations: u32,
    in_flight: usize,
    settings: &SearchSettings,
    network: &mut Network,
    mut gather: impl FnMut(&mut Tree, &Round, &Cache, &mut Vec<u8>) -> u32,
) -> Vec<Vec<u8>> {
    let mut rounds = search_in_flight(tree, simulations, in_flight, settings, network, None, &mut gather);
    if let Some(line) = line_down(tree) {
        tree.move_root(&line);
        assert_ne!(tree.root_index(), 0);
        rounds.extend(search_in_flight(tree, simulations, in_flight, settings, network, None, &mut gather));
    }
    rounds
}

//passes of one visit are gather's descents, so the search is gather's, number for number, on a moved root too
#[test]
fn passes_of_one_visit_are_gathers_search() {
    let tables = tables();
    let workers = Workers::new(1, Duration::ZERO, &[]);
    let mut moved = 0;
    for (fen, moves, simulations, certainty, tables) in cases(&tables) {
        let settings = settings(certainty);
        for in_flight in [1, 3] {
            let mut old = tree_after(fen, moves);
            let old_rounds =
                search_moving_on(&mut old, simulations, in_flight, &settings, &mut Network::new(), |tree, round, cache, boards| {
                    tree.gather(round, &settings, tables, Some(cache), boards).expect("the tables")
                });
            let mut new = tree_after(fen, moves);
            let mut counts = BatchCounts::default();
            let new_rounds =
                search_moving_on(&mut new, simulations, in_flight, &settings, &mut Network::new(), |tree, round, cache, boards| {
                    tree.gather_batch(round, &settings, tables, Some(cache), boards, &workers, 1, &mut counts).expect("the tables")
                });
            moved += usize::from(old.root_index() != 0);
            assert!(old_rounds == new_rounds, "{fen}: the rounds' leaves");
            assert_same_trees(&old, &new);
            assert!(counts.collisions > 0 || in_flight == 1, "{fen}: {counts:?}");
        }
    }
    assert!(moved >= 8, "{moved} searches went on below a moved root");
}

//a copy of a tree's nodes and edges for select_edge, without its waiting leaves
fn nodes_copied(tree: &Tree) -> Tree {
    Tree {
        board: tree.board.clone(),
        keys: tree.keys.clone(),
        nodes: tree.nodes.clone(),
        root: tree.root,
        node_keys: tree.node_keys.clone(),
        moves: tree.moves.clone(),
        priors: tree.priors.clone(),
        noised_priors: tree.noised_priors.clone(),
        children: tree.children.clone(),
        pending: Vec::new(),
        logits: Vec::new(),
        full: tree.full,
    }
}

//gather's descents with nothing backed up, each counted in flight for the next: the lines they end on, how many
fn descents(tree: &Tree, visits: u32, settings: &SearchSettings) -> BTreeMap<Vec<u32>, u32> {
    let mut tree = nodes_copied(tree);
    let mut lines = BTreeMap::new();
    for _ in 0..visits {
        let (mut node, mut path, mut line) = (tree.root_index(), vec![tree.root_index()], Vec::new());
        while tree.nodes[node].has_edges() {
            let edge = tree.select_edge(node, settings);
            line.push(edge as u32);
            if tree.children[edge] == NO_NODE {
                tree.children[edge] = tree.nodes.len() as u32;
                tree.nodes.push(Node::new());
            }
            node = tree.children[edge] as usize;
            path.push(node);
            if tree.nodes[node].proof.is_some() {
                break;
            }
        }
        for node in path {
            tree.nodes[node].in_flight += 1;
        }
        *lines.entry(line).or_insert(0) += 1;
    }
    lines
}

//a pass's picks: the lines its visits end on, in its order
fn picks(tree: &Tree, visits: u32, settings: &SearchSettings) -> Vec<(Vec<u32>, u32)> {
    let mut output = Output::default();
    tree.pick_down(tree.root_index() as u32, visits, settings, &mut Scratch::default(), &mut output);
    output.picks.iter().map(|pick| (output.edges[pick.share.line.clone()].to_vec(), pick.share.visits)).collect()
}

//a tree of a network that answers every position the same (equal priors, a draw), so that ties are everywhere
fn even_tree(simulations: u32, settings: &SearchSettings) -> Tree {
    let (move_ids, mut tree) = (Network::new().move_ids, tree_after(START, &[]));
    while tree.root().visit_count < simulations {
        let round = Round { id: 0, leaves: 16, simulations: 16, collisions: 16, visit_limit: simulations, memory_limit: usize::MAX };
        tree.gather(&round, settings, None, None, &mut Vec::new()).expect("no tables");
        while tree.pending() > 0 {
            tree.finish_leaf(0, |column| if column == ANSWER_WIDTH - 1 { 0.5 } else { 0.0 }, ANSWER_WIDTH, &move_ids, None, settings);
        }
    }
    tree
}

//a pass shares its visits as the descents would, with leaves waiting, root noise, proofs and ties, in line order
#[test]
fn a_pass_shares_its_visits_as_that_many_descents_would() {
    let tables = tables();
    let mut trees = Vec::new();
    for (fen, moves, simulations, certainty, tables) in cases(&tables) {
        let search_settings = settings(certainty);
        let mut network = Network::new();
        let mut tree = tree_after(fen, moves);
        search(&mut tree, simulations, usize::MAX, &search_settings, tables, &mut network);
        let round = Round { id: 0, leaves: 64, simulations: 192, collisions: 64, visit_limit: u32::MAX, memory_limit: usize::MAX };
        let mut waiting = nodes_copied(&tree);
        waiting.gather(&round, &search_settings, tables, Some(&network.cache), &mut Vec::new()).expect("the tables");
        let mut noised = nodes_copied(&tree);
        noised.add_root_noise(&mut ChaCha8Rng::seed_from_u64(7), 0.3, 0.25);
        if let Some(line) = line_down(&tree) {
            let mut moved = nodes_copied(&tree);
            moved.move_root(&line);
            trees.push((fen, settings(certainty), moved));
        }
        trees.extend([(fen, settings(certainty), tree), (fen, settings(certainty), waiting), (fen, settings(certainty), noised)]);
    }
    trees.push(("even", settings(false), even_tree(300, &settings(false))));
    for (fen, settings, tree) in &trees {
        for visits in [1, 2, 7, 64, 500] {
            let expected: Vec<(Vec<u32>, u32)> = descents(tree, visits, settings).into_iter().collect();
            assert_eq!(picks(tree, visits, settings), expected, "{fen} {visits} visits");
        }
    }
    assert!(trees.len() >= 20);
}

//a new leaf's visits after its first are collisions: with no room for any the round ends after one pass
#[test]
fn a_leafs_visits_after_its_first_are_collisions() {
    let settings = settings(true);
    let mut network = Network::new();
    let mut tree = tree_after(START, OPENING);
    search(&mut tree, 1, usize::MAX, &settings, None, &mut network);
    let lines = descents(&tree, 64, &settings);
    assert!(lines.values().any(|&descents| descents > 1), "{lines:?}");

    let round = Round { id: 0, leaves: 64, simulations: 64, collisions: 0, visit_limit: u32::MAX, memory_limit: usize::MAX };
    let mut counts = BatchCounts::default();
    let gathered =
        tree.gather_batch(&round, &settings, None, None, &mut Vec::new(), &Workers::new(1, Duration::ZERO, &[]), u32::MAX, &mut counts);
    let leaf_lines: Vec<Vec<u32>> = tree
        .pending
        .iter()
        .flat_map(|(_, leaves)| leaves)
        .map(|leaf| {
            leaf.path
                .windows(2)
                .map(|nodes| tree.children.iter().position(|&child| child as usize == nodes[1]).expect("an edge") as u32)
                .collect()
        })
        .collect();
    assert_eq!(gathered.expect("no tables") as usize, lines.len());
    assert_eq!(leaf_lines, lines.keys().cloned().collect::<Vec<_>>());
    //each leaf waits in flight at the root and at itself, the collisions taken off again
    let in_flight: usize = tree.nodes.iter().map(|node| node.in_flight as usize).sum();
    assert!(counts.passes == 1 && counts.collisions == 0 && in_flight == 2 * lines.len(), "{counts:?}");
}

//a round's leaves and the search are the same for any number of workers (collisions, cache hits, proofs, passes),
//pinned to cpus or not
#[test]
fn a_rounds_leaves_are_the_same_for_any_number_of_workers() {
    let tables = tables();
    let mut single = BatchCounts::default();
    for (fen, moves, simulations, certainty, tables) in cases(&tables) {
        let settings = settings(certainty);
        for in_flight in [1, 4] {
            let mut reference: Option<(Vec<Vec<u8>>, Tree)> = None;
            for (count, spin, pinned) in [(1, 0, false), (2, 0, false), (3, 50, false), (3, 50, true), (8, 50, false)] {
                let cpus: Vec<usize> =
                    if pinned { crate::affinity::allowed().iter().rev().take(count - 1).copied().collect() } else { Vec::new() };
                let workers = Workers::new(count, Duration::from_micros(spin), &cpus);
                let mut tree = tree_after(fen, moves);
                let mut counts = BatchCounts::default();
                let rounds =
                    search_moving_on(&mut tree, simulations, in_flight, &settings, &mut Network::new(), |tree, round, cache, boards| {
                        tree.gather_batch(round, &settings, tables, Some(cache), boards, &workers, u32::MAX, &mut counts)
                            .expect("the tables")
                    });
                assert!(count == 1 || counts.items > counts.passes, "{fen}: {counts:?}");
                if count == 1 {
                    single.passes += counts.passes - counts.rounds;
                    (single.cached, single.scored, single.collisions) =
                        (single.cached + counts.cached, single.scored + counts.scored, single.collisions + counts.collisions);
                }
                match &reference {
                    None => reference = Some((rounds, tree)),
                    Some((reference_rounds, reference_tree)) => {
                        assert!(rounds == *reference_rounds, "{fen}, {count} workers: the rounds' leaves");
                        assert_same_trees(reference_tree, &tree);
                    }
                }
            }
        }
    }
    assert!(single.passes > 0 && single.cached > 0 && single.scored > 0 && single.collisions > 0, "{single:?}");
}

//finish_batch leaves the tree, and the cache the next rounds read, as finish_leaf does leaf by leaf
#[test]
fn finish_batch_finishes_as_finish_leaf_does() {
    let tables = tables();
    for (fen, moves, simulations, certainty, tables) in cases(&tables) {
        let settings = settings(certainty);
        let gather = |tree: &mut Tree, round: &Round, cache: &Cache, boards: &mut Vec<u8>| {
            tree.gather(round, &settings, tables, Some(cache), boards).expect("the tables")
        };
        let mut old = tree_after(fen, moves);
        let old_rounds = search_in_flight(&mut old, simulations, 3, &settings, &mut Network::new(), None, gather);
        for count in [1, 3] {
            let mut new = tree_after(fen, moves);
            let workers = Workers::new(count, Duration::ZERO, &[]);
            let new_rounds = search_in_flight(&mut new, simulations, 3, &settings, &mut Network::new(), Some(&workers), gather);
            assert!(old_rounds == new_rounds, "{fen}, {count} workers: the rounds' leaves");
            assert_same_trees(&old, &new);
        }
    }
}

//every worker runs every job once, with the caller's data, whether it was spinning or asleep
#[test]
fn the_workers_run_each_job_once_on_each() {
    for spin in [0, 200] {
        let workers = Workers::new(4, Duration::from_micros(spin), &[]);
        let calls: Vec<std::sync::atomic::AtomicU32> = (0..4).map(|_| std::sync::atomic::AtomicU32::new(0)).collect();
        for job in 0..2_000 {
            if job % 500 == 0 {
                std::thread::sleep(Duration::from_millis(2));
            }
            workers.run(&|worker| {
                calls[worker].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            });
        }
        assert!(calls.iter().all(|calls| calls.load(std::sync::atomic::Ordering::Relaxed) == 2_000));
    }
}

mod bench;

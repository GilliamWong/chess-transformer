//the tree many threads search at once against Tree (one thread: the same search, number for number) and against its own
//invariants after searches on many threads
use super::*;
use crate::search::shared::{SharedCache, SharedTree};
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::time::Duration;

//the stand-in network of tests.rs, with the concurrent cache
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

//the solid links of nodes with 8 visits, two queued at a time: the searches here make many and copy lists as others wait
fn shared_tree_after(fen: &str, moves: &[&str]) -> SharedTree {
    let tree = tree_after(fen, moves);
    SharedTree::new(tree.board, tree.keys).with_solid_links(8, 2)
}

fn seed_of(board: &[u8]) -> u64 {
    board.iter().fold(0, |hash, &token| mix(hash ^ u64::from(token)))
}

//the round's leaves finished with the stand-in network's answers for their boards
fn finish_round(
    tree: &SharedTree,
    round: u64,
    boards: &[u8],
    settings: &SearchSettings,
    network: &SharedNetwork,
    memory_limit: usize,
    logits: &mut Vec<f32>,
) {
    let _hold = tree.hold();
    let leaves = tree.take_round(round);
    let boards = boards.as_chunks::<TOKENS_PER_BOARD>().0;
    assert_eq!(leaves.len(), boards.len());
    for (leaf, board) in leaves.into_iter().zip(boards) {
        let seed = seed_of(board);
        let answer = |column| answer(board, seed, column);
        tree.finish_leaf(leaf, answer, ANSWER_WIDTH, &network.move_ids, Some(&network.cache), settings, memory_limit, logits);
    }
}

//engine_search's rounds on one thread until the root has its simulations or the tree is full, up to in_flight waiting,
//oldest finished first, as tests.rs search_in_flight; every round's boards
fn shared_in_flight(
    tree: &SharedTree,
    simulations: u32,
    in_flight: usize,
    settings: &SearchSettings,
    tables: Option<&Tables>,
    network: &SharedNetwork,
    memory_limit: usize,
) -> Vec<Vec<u8>> {
    let (mut rounds, mut gathered, mut logits) = (VecDeque::new(), Vec::new(), Vec::new());
    loop {
        let room = tree.root().visits() as usize + tree.pending() < simulations as usize && !tree.is_full();
        if room && rounds.len() < in_flight {
            let leaves = (tree.root().visits() / 16).clamp(1, 64);
            let round = Round {
                id: gathered.len() as u64,
                leaves,
                simulations: 3 * leaves,
                collisions: leaves,
                visit_limit: simulations,
                memory_limit,
            };
            let mut boards = Vec::new();
            let queued = tree.gather(&round, settings, tables, Some(&network.cache), &mut boards).expect("the tables");
            assert_eq!(boards.len(), queued.len() * TOKENS_PER_BOARD);
            tree.queue(round.id, queued);
            gathered.push(boards.clone());
            rounds.push_back((round.id, boards));
            continue;
        }
        let Some((id, boards)) = rounds.pop_front() else { break };
        finish_round(tree, id, &boards, settings, network, memory_limit, &mut logits);
    }
    gathered
}

//one thread's search is Tree's: the same boards in the same rounds and the same tree, number for number, with rounds in
//flight (collisions, cache hits), certainty (proofs), the tables and repeats
#[test]
fn one_thread_searches_as_tree_does() {
    let tables = tables();
    let (mut collided, mut proven) = (0, 0);
    for (fen, moves, simulations, certainty, tables) in cases(&tables) {
        let settings = settings(certainty);
        for in_flight in [1, 2, 3] {
            let mut old = tree_after(fen, moves);
            let old_rounds =
                search_in_flight(&mut old, simulations, in_flight, &settings, &mut Network::new(), |tree, round, cache, boards| {
                    tree.gather(round, &settings, tables, Some(cache), boards).expect("the tables")
                });
            let new = shared_tree_after(fen, moves);
            let new_rounds = shared_in_flight(&new, simulations, in_flight, &settings, tables, &SharedNetwork::new(), usize::MAX);
            assert!(old_rounds == new_rounds, "{fen}, {in_flight} in flight: the rounds' boards");
            assert_eq!(new.counts.orphans.load(Ordering::Relaxed), 0);
            let new = new.to_tree();
            assert_same_trees(&old, &new);
            proven += new.nodes.iter().filter(|node| node.proof.is_some()).count();
            collided += usize::from(in_flight > 1);
        }
    }
    assert!(collided > 0 && proven > 0, "{collided} searches with rounds in flight, {proven} proven nodes");
}

//an independent walk of a finished search's tree: nothing in flight, every simulation counted once in a node's visits
//and its moves' (a node's visits are one for itself and its moves' together), a node's sum its own score and its moves'
//(what a simulation scored for the side that moved into it), the visited policy and the proofs as their definitions
fn check_tree(tree: &SharedTree, settings: &SearchSettings) {
    let mut nodes = 0;
    let board = tree.board.clone();
    let mut keys = tree.keys.clone();
    let root_keys = keys.len();
    walk(tree, tree.root_index(), &board, &mut keys, root_keys, settings, &mut nodes);
    assert!(nodes >= 1);
}

fn walk(tree: &SharedTree, id: usize, board: &Chess, keys: &mut Vec<u64>, root_keys: usize, settings: &SearchSettings, nodes: &mut usize) {
    let node = tree.node(id);
    *nodes += 1;
    let (visits, in_flight) = node.counts();
    assert_eq!(in_flight, 0, "a node with simulations in flight after the search");
    assert!(visits > 0, "a node of the tree without a visit");
    let edges = node.edges();
    let children: Vec<(usize, usize)> = edges.clone().filter_map(|edge| tree.child_index(node, edge).map(|child| (edge, child))).collect();
    if edges.is_empty() {
        //a game's end or a repeat, reached again and again: every visit scored its result
        let legal_moves = board.legal_moves();
        let over = if id == tree.root_index() {
            None
        } else {
            game_result(board, keys, &legal_moves, settings.draw, settings.repetitions_for_a_draw, None).expect("no tables")
        };
        let (&leaf_key, path_keys) = keys[root_keys - 1..].split_last().expect("the root's key");
        let value = over.or_else(|| path_keys.contains(&leaf_key).then_some(settings.draw));
        match (node.proof(), value) {
            (Some(proof), Some(_)) => assert_close(node.value_sum(), f64::from(visits) * proof.value, visits, "a proven leaf's sum"),
            (None, Some(value)) => assert_close(node.value_sum(), f64::from(visits) * (1.0 - value), visits, "a leaf's sum"),
            (_, None) => panic!("a leaf the network never expanded and the rules do not end"),
        }
        return;
    }
    let tokens = {
        let mut tokens = Vec::new();
        encode_board(board, &mut tokens);
        tokens
    };
    let own = 1.0 - f64::from(sharpen(answer(&tokens, seed_of(&tokens), ANSWER_WIDTH - 1), settings.value_sharpness));
    let moves: u32 = children.iter().map(|&(_, child)| tree.node(child).visits()).sum();
    let below: f64 = children.iter().map(|&(_, child)| f64::from(tree.node(child).visits()) - tree.node(child).value_sum()).sum();
    match node.proof() {
        None => {
            assert_eq!(visits, 1 + moves, "visits of an unproven node");
            assert_close(node.value_sum(), own + below, visits, "an unproven node's sum");
        }
        //every visit counts the proven score
        Some(proof) => assert_close(node.value_sum(), f64::from(visits) * proof.value, visits, "a proven node's sum"),
    }
    //the policy of the moves visited, in edge order
    let visited = python_sum(children.iter().map(|&(edge, _)| tree.prior(edge))).sqrt();
    assert_eq!(node.visited_policy_sqrt().to_bits(), visited.to_bits(), "visited policy");
    let priors: f64 = edges.clone().map(|edge| tree.prior(edge)).sum();
    assert!((priors - 1.0).abs() < 1e-3, "priors add up to {priors}");
    if let Some(proof) = node.proof().filter(|_| id != tree.root_index()) {
        //a proof from the moves': their best, or all of them
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
        walk(tree, child, &next, keys, root_keys, settings, nodes);
        keys.pop();
    }
}

fn assert_close(found: f64, expected: f64, visits: u32, what: &str) {
    assert!((found - expected).abs() <= 1e-9 * f64::from(visits.max(1)), "{what}: {found} against {expected} over {visits} visits");
}

//rounds on `threads` threads, each with its own rounds in flight, until the root has its simulations
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
            scope.spawn(|| {
                let (mut rounds, mut logits) = (VecDeque::new(), Vec::new());
                loop {
                    let room = tree.root().visits() as usize + tree.pending() < simulations as usize && !tree.is_full();
                    if room && rounds.len() < in_flight {
                        let leaves = (tree.root().visits() / 16).clamp(1, most_leaves);
                        let id = next_round.fetch_add(1, Ordering::Relaxed);
                        let round =
                            Round { id, leaves, simulations: 3 * leaves, collisions: leaves, visit_limit: simulations, memory_limit };
                        let mut boards = Vec::new();
                        let queued = tree.gather(&round, settings, None, Some(&network.cache), &mut boards).expect("no tables");
                        tree.queue(id, queued);
                        rounds.push_back((id, boards));
                        continue;
                    }
                    let Some((id, boards)) = rounds.pop_front() else { break };
                    finish_round(tree, id, &boards, settings, network, memory_limit, &mut logits);
                }
            });
        }
    });
}

//after searches on many threads, in positions with many collisions, repeats and proofs, the tree holds every invariant:
//no simulation lost, none counted twice, none left in flight
#[test]
fn searches_on_many_threads_leave_a_consistent_tree() {
    let positions = [(START, OPENING), (ROOKS, &[][..]), (LOCKED_PAWNS, &[]), (MATE_IN_TWO, &[]), ("8/8/8/4k3/8/8/8/4K2R w - - 0 1", &[])];
    let mut collided = 0;
    for (fen, moves) in positions {
        for certainty in [false, true] {
            let settings = settings(certainty);
            for threads in [2, 4, 8, 24] {
                for in_flight in [1, 3] {
                    let tree = shared_tree_after(fen, moves);
                    search_on_threads(&tree, threads, 3_000, in_flight, 16, &settings, &SharedNetwork::new(), usize::MAX);
                    assert_eq!(tree.pending(), 0);
                    assert!(
                        tree.root().visits() >= 3_000 && tree.root().visits() < 3_000 + 64 * threads as u32,
                        "{}",
                        tree.root().visits()
                    );
                    check_tree(&tree, &settings);
                    collided += tree.counts.orphans.load(Ordering::Relaxed);
                    //lc0's layout both ways: the busy nodes' links made solid during the search, the rest lists
                    let (lists, solids) = tree.lists_and_solids();
                    assert!(lists > 0 && solids > 0, "{lists} lists, {solids} solid");
                }
            }
        }
    }
    eprintln!("{collided} nodes made twice");
}

//proofs made while other threads back up through the same nodes: endings full of mates, stalemates and drawn material,
//where a proof rewrites sums that backups are adding to
#[test]
fn proofs_on_many_threads_leave_consistent_sums() {
    let settings = settings(true);
    let mut exclusive = 0;
    for fen in ["8/8/8/4k3/8/8/8/4K2Q w - - 0 1", "8/8/8/4k3/8/8/8/4K2R w - - 0 1", MATE_IN_TWO, "7k/8/5K2/8/8/8/8/6Q1 w - - 0 1"] {
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

//a search ends at its memory limit as Tree's does, its bytes within the limit, and until then it is the search without
//a limit, node for node; on many threads the limit holds too
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

//the answers are found by position from any thread, and a thread's copy is the answer inserted
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

//the edges in a SharedTree that a line of moves goes down from the root
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

//Searches' advance and the next start on a SharedTree, as tests.rs advanced() does them on a Tree: the line's first move in
//place, the stale candidates and the copy unless stopped before them, then the rest of the line in place and the candidates
//under it (a walk without them), compacted to be compared node for node
fn shared_advanced(
    tree: &SharedTree,
    line: &[PackedMove],
    threads: usize,
    stopped: Option<&str>,
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
    let candidates = advanced.stale_candidates(threads, if stopped == Some("walk") { &always } else { &never });
    if candidates.is_some() {
        advanced.compact(Some(if stopped == Some("copy") { &always } else { &never }));
    }
    let rest = edges_of(&advanced, &line[1..]);
    advanced.move_root(&rest);
    assert!(advanced.root().proof().is_none(), "a root is never proven");
    match candidates {
        Some(candidates) => advanced.drop_stale_candidates(&places[1..], &candidates, settings, tables).expect("the tables"),
        None => advanced.drop_stale_leaves(settings, tables).expect("the tables"),
    }
    advanced.compact(None);
    advanced
}

//tests.rs' reuse test on SharedTrees: a line played at once (play_line, then the stale leaves) and by advance's way leave the
//tree Tree's moves played one at a time leave, number for number, in searches with repeats, proofs and the tables
#[test]
fn reusing_a_shared_tree_leaves_the_tree_a_tree_is_left() {
    let tables = tables();
    let (mut lines, mut stale, mut proven) = (0, 0, 0);
    //draws worth 0.5 take as much from a parent as they leave it: another value tells which of the two a sign took
    let cases: Vec<_> = cases(&tables).into_iter().flat_map(|case| [(case, 0.5), (case, 0.35)]).collect();
    for ((fen, moves, simulations, certainty, tables), draw) in cases {
        let settings = SearchSettings { draw, ..settings(certainty) };
        let mut tree = tree_after(fen, moves);
        search(&mut tree, simulations, usize::MAX, &settings, tables, &mut Network::new());
        let shared = shared_tree_after(fen, moves);
        shared_in_flight(&shared, simulations, 1, &settings, tables, &SharedNetwork::new(), usize::MAX);
        //one thread's search is Tree's, which the other test shows with rounds in flight: here one round at a time
        assert_eq!(tree.root().visit_count, shared.root().visits());
        for line in reusable_lines(&tree) {
            let line_moves: Vec<PackedMove> = line.iter().map(|&edge| tree.chess_move(edge)).collect();
            let mut old = copied(&tree);
            stale += reused_move_by_move(&mut old, &line_moves, &settings, tables);

            let mut at_once = SharedTree::from_tree(&shared.to_tree());
            at_once.play_line(&edges_of(&at_once, &line_moves));
            at_once.drop_stale_leaves(&settings, tables).expect("the tables");
            assert_same_trees(&old, &at_once.to_tree());
            for (threads, stopped) in [(1, None), (1, Some("copy")), (1, Some("walk")), (4, None), (4, Some("copy")), (4, Some("walk"))] {
                let advanced = shared_advanced(&shared, &line_moves, threads, stopped, &settings, tables);
                assert_same_trees(&old, &advanced.to_tree());
            }
            //the parallel walk finds the one-thread walk's candidates in its order
            let mut moved = SharedTree::from_tree(&shared.to_tree());
            moved.move_root(&edges_of(&moved, &line_moves[..1]));
            let never = AtomicBool::new(false);
            assert_eq!(moved.stale_candidates(4, &never), moved.stale_candidates(1, &never));
            lines += 1;
            proven += old.nodes.iter().filter(|node| node.proof.is_some()).count();
        }
    }
    assert!(lines >= 15 && stale > 0 && proven > 0, "{lines} lines, {stale} stale leaves, {proven} proven nodes");
}

//a proven node hides the leaves below it from the stale walk unless it is the new root, whose proof is dropped: advance's
//candidates must leave and take out the same ones on a SharedTree as the walk does on a Tree, with the proofs set by hand
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
            moved.find_stale_leaves(&mut vec![moved.root], &mut Vec::new(), &settings, None, false, &mut stale).expect("no tables");
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
        walked.drop_stale_leaves(&settings, None).expect("no tables");
        let advanced = shared_advanced(&SharedTree::from_tree(&proved), &line_moves, 4, None, &settings, None);
        assert_same_trees(&walked, &advanced.to_tree());
    }
}

//a closed tree (its search is over and reported) takes no leaf for a call that still has it
#[test]
fn a_closed_tree_gathers_nothing() {
    let (settings, network) = (settings(true), SharedNetwork::new());
    let tree = shared_tree_after(START, &[]);
    shared_in_flight(&tree, 200, 1, &settings, None, &network, usize::MAX);
    let round = |id| Round { id, leaves: 8, simulations: 24, collisions: 8, visit_limit: u32::MAX, memory_limit: usize::MAX };
    let open = tree.gather(&round(0), &settings, None, Some(&network.cache), &mut Vec::new()).expect("no tables");
    assert!(!open.is_empty());
    tree.close();
    assert!(tree.gather(&round(1), &settings, None, Some(&network.cache), &mut Vec::new()).expect("no tables").is_empty());
    //the game's next search starts from the tree it left
    let mut tree = tree;
    tree.reopen();
    assert!(!tree.gather(&round(2), &settings, None, Some(&network.cache), &mut Vec::new()).expect("no tables").is_empty());
}

//a round that found no leaves keeps nothing in the tree: the search loop may run a great many of them while it waits
#[test]
fn rounds_without_leaves_queue_nothing() {
    let tree = shared_tree_after(START, &[]);
    for round in 0..10_000 {
        tree.queue(round, Vec::new());
    }
    assert_eq!(tree.queued_rounds(), 0);
}

//certainty turned on after the game's ends were visited (a visited leaf is proven by its next visit, its visits rewritten with the
//proven score in every sum above): the same tree as Tree's, number for number
#[test]
fn proofs_made_after_the_leaves_were_visited_rewrite_the_sums_as_tree_does() {
    //draws worth less the second time, so that the proofs of the stalemates found rewrite what their visits scored
    let (without, with) = (settings(false), SearchSettings { draw: 0.3, ..settings(true) });
    let gather_with = |settings: &'static SearchSettings| {
        move |tree: &mut Tree, round: &Round, cache: &Cache, boards: &mut Vec<u8>| {
            tree.gather(round, settings, None, Some(cache), boards).expect("no tables")
        }
    };
    let (without, with): (&'static SearchSettings, &'static SearchSettings) = (Box::leak(Box::new(without)), Box::leak(Box::new(with)));
    for fen in [MATE_IN_TWO, "8/8/8/4k3/8/8/8/4K2R w - - 0 1", "7k/8/5K2/8/8/8/8/6Q1 w - - 0 1"] {
        let mut old = tree_after(fen, &[]);
        search_in_flight(&mut old, 3_000, 1, without, &mut Network::new(), gather_with(without));
        search_in_flight(&mut old, 8_000, 1, with, &mut Network::new(), gather_with(with));
        let new = shared_tree_after(fen, &[]);
        shared_in_flight(&new, 3_000, 1, without, None, &SharedNetwork::new(), usize::MAX);
        shared_in_flight(&new, 8_000, 1, with, None, &SharedNetwork::new(), usize::MAX);
        assert_same_trees(&old, &new.to_tree());
        assert!(old.nodes.iter().any(|node| node.proof.is_some()), "{fen}");
    }
}

//tables met after the leaves were visited (a search carried on with the tables, as a reused tree meets them when its root moves
//down) prove leaves that have visits, which rewrites the sums above them: the same tree as Tree's, number for number
#[test]
fn tables_met_after_the_leaves_were_visited_prove_them_as_tree_does() {
    let Some(tables) = tables() else {
        eprintln!("no syzygy tables: the ending they score is left out");
        return;
    };
    let settings = settings(true);
    let gather_with = |tables: Option<&'static Tables>| {
        let settings = &settings;
        move |tree: &mut Tree, round: &Round, cache: &Cache, boards: &mut Vec<u8>| {
            tree.gather(round, settings, tables, Some(cache), boards).expect("the tables")
        }
    };
    let tables: &'static Tables = Box::leak(Box::new(tables));
    let mut old = tree_after(PAWNS_AND_ROOKS, &[]);
    search_in_flight(&mut old, 4_000, 1, &settings, &mut Network::new(), gather_with(None));
    let before = old.nodes.iter().filter(|node| node.proof.is_some()).count();
    search_in_flight(&mut old, 12_000, 1, &settings, &mut Network::new(), gather_with(Some(tables)));
    let new = shared_tree_after(PAWNS_AND_ROOKS, &[]);
    shared_in_flight(&new, 4_000, 1, &settings, None, &SharedNetwork::new(), usize::MAX);
    shared_in_flight(&new, 12_000, 1, &settings, Some(tables), &SharedNetwork::new(), usize::MAX);
    assert_same_trees(&old, &new.to_tree());
    assert!(old.nodes.iter().filter(|node| node.proof.is_some()).count() > before + 5);
}

//a thread that wants the tree to itself waits for the descent the others are in and not for their whole calls: with a call of
//thousands of descents running, the wait is a fraction of the call (of a few tries, on a machine that may be busy)
#[test]
fn a_thread_that_wants_the_tree_to_itself_waits_for_a_descent_not_a_call() {
    let (settings, network) = (settings(true), SharedNetwork::new());
    let tree = shared_tree_after(START, &[]);
    shared_in_flight(&tree, 20_000, 2, &settings, None, &network, usize::MAX);
    let mut tries = Vec::new();
    for id in 0..5 {
        let round =
            Round { id, leaves: 30_000, simulations: 200_000, collisions: 150_000, visit_limit: u32::MAX, memory_limit: usize::MAX };
        let (call, wait) = std::thread::scope(|scope| {
            let gathering = scope.spawn(|| {
                let started = std::time::Instant::now();
                tree.gather(&round, &settings, None, Some(&network.cache), &mut Vec::new()).expect("no tables");
                started.elapsed()
            });
            std::thread::sleep(Duration::from_millis(5));
            let started = std::time::Instant::now();
            tree.quiet(|| ());
            let waited = started.elapsed();
            (gathering.join().expect("a thread"), waited)
        });
        tries.push((wait, call));
        if wait * 4 < call && call > Duration::from_millis(30) {
            break;
        }
    }
    let &(wait, call) = tries.last().expect("a try");
    let gave_way = tree.counts.gave_way.load(Ordering::Relaxed);
    assert!(gave_way > 0 && wait * 4 < call && call > Duration::from_millis(30), "waits and calls {tries:?}, gave way {gave_way} times");
}

//transpositions (search.rs Graph): the settings with them, and a tree and a shared tree that have them
fn graph_settings(certainty: bool) -> SearchSettings {
    SearchSettings { transpositions: true, ..settings(certainty) }
}

fn graph_tree_after(fen: &str, moves: &[&str]) -> Tree {
    tree_after(fen, moves).with_transpositions(true)
}

fn graph_shared_tree_after(fen: &str, moves: &[&str]) -> SharedTree {
    shared_tree_after(fen, moves).with_transpositions(true)
}

//a finished search's graph against its own definitions: nothing in flight; an unproven node's visits one for itself and
//its moves' own together (a move's visits, not its node's, which every path's share), a move without a node never
//visited; the table every node's and nothing else, under its position and fifty-move count. Returns the nodes more
//than one move leads to
fn check_graph(tree: &Tree) -> usize {
    let graph = tree.graph.as_ref().expect("a tree with transpositions");
    assert_eq!(graph.edge_visits.len(), tree.moves.len(), "a visit count for every move");
    assert_eq!(graph.halfmoves.len(), tree.nodes.len(), "a fifty-move count for every node");
    let mut incoming = vec![0u32; tree.nodes.len()];
    let (mut seen, mut order) = (vec![false; tree.nodes.len()], vec![tree.root_index()]);
    seen[tree.root_index()] = true;
    while let Some(id) = order.pop() {
        let node = &tree.nodes[id];
        assert_eq!(node.in_flight, 0, "a node with simulations in flight after the search");
        let mut moves = 0;
        for edge in node.edges() {
            let child = tree.children[edge];
            if child == NO_NODE {
                assert_eq!(graph.edge_visits[edge], 0, "a move without a node, visited");
                continue;
            }
            moves += graph.edge_visits[edge];
            incoming[child as usize] += 1;
            if !std::mem::replace(&mut seen[child as usize], true) {
                order.push(child as usize);
            }
        }
        //a root's proof is dropped, and the visits it stopped there with belong to no move
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

//one thread's search with transpositions is Tree's: the same boards in the same rounds and the same graph, number for
//number, with rounds in flight, proofs, the tables, repeats and nodes shared by several moves
#[test]
fn one_thread_with_transpositions_searches_as_tree_does() {
    let tables = tables();
    let mut shared = 0;
    for (fen, moves, simulations, certainty, tables) in cases(&tables) {
        let settings = graph_settings(certainty);
        for in_flight in [1, 2, 3] {
            let mut old = graph_tree_after(fen, moves);
            let old_rounds =
                search_in_flight(&mut old, simulations, in_flight, &settings, &mut Network::new(), |tree, round, cache, boards| {
                    tree.gather(round, &settings, tables, Some(cache), boards).expect("the tables")
                });
            let new = graph_shared_tree_after(fen, moves);
            let new_rounds = shared_in_flight(&new, simulations, in_flight, &settings, tables, &SharedNetwork::new(), usize::MAX);
            assert!(old_rounds == new_rounds, "{fen}, {in_flight} in flight: the rounds' boards");
            assert_eq!(new.counts.orphans.load(Ordering::Relaxed), 0);
            //the move choice too, whose ties a node's own mean breaks (not its move's visits)
            let line = |moves: Vec<usize>, chess_move: &dyn Fn(usize) -> PackedMove| moves.into_iter().map(chess_move).collect::<Vec<_>>();
            assert_eq!(
                line(new.principal_variation(), &|edge| new.chess_move(edge)),
                line(old.principal_variation(), &|edge| old.chess_move(edge))
            );
            let new = new.to_tree();
            assert_same_trees(&old, &new);
            shared += check_graph(&old);
        }
    }
    assert!(shared > 0, "no node shared by two moves");
}

//searches with transpositions on many threads, in positions with many collisions, repeats, proofs and transpositions,
//leave a consistent graph: no simulation lost, none counted twice, none left in flight
#[test]
fn searches_with_transpositions_on_many_threads_leave_a_consistent_graph() {
    let positions = [(START, OPENING), (ROOKS, &[][..]), (LOCKED_PAWNS, &[]), (MATE_IN_TWO, &[]), ("8/8/8/4k3/8/8/8/4K2R w - - 0 1", &[])];
    let mut shared = 0;
    for (fen, moves) in positions {
        for certainty in [false, true] {
            let settings = graph_settings(certainty);
            for threads in [2, 4, 8, 24] {
                for in_flight in [1, 3] {
                    let tree = graph_shared_tree_after(fen, moves);
                    search_on_threads(&tree, threads, 3_000, in_flight, 16, &settings, &SharedNetwork::new(), usize::MAX);
                    assert_eq!(tree.pending(), 0);
                    assert!(tree.root().visits() >= 3_000 && tree.root().visits() < 3_000 + 64 * threads as u32);
                    shared += check_graph(&tree.to_tree());
                }
            }
        }
    }
    assert!(shared > 0, "no node shared by two moves");
}

//the nodes the root of a tree reaches
fn reachable(tree: &Tree, from: usize) -> usize {
    let mut seen = vec![false; tree.nodes.len()];
    let mut order = vec![from];
    seen[from] = true;
    while let Some(id) = order.pop() {
        for edge in tree.nodes[id].edges() {
            let child = tree.children[edge];
            if child != NO_NODE && !std::mem::replace(&mut seen[child as usize], true) {
                order.push(child as usize);
            }
        }
    }
    seen.iter().filter(|&&seen| seen).count()
}

//the edges from a tree's root along the moves
fn line_after(tree: &Tree, line: &[PackedMove]) -> Vec<usize> {
    let mut node = tree.root_index();
    line.iter()
        .map(|&played| {
            let edge = tree.nodes[node].edges().find(|&edge| tree.chess_move(edge) == played).expect("the line's move");
            node = tree.children[edge] as usize;
            edge
        })
        .collect()
}

//tree reuse with transpositions: a line played at once leaves a graph with its invariants, each node it reaches copied
//once, the new root's numbers as they were; advance's way (the root moved in place, candidates that are none, the copy
//on threads) leaves the same graph, and so does a SharedTree's line played at once
#[test]
fn reusing_a_graph_keeps_it_whole() {
    let tables = tables();
    let mut lines = 0;
    for (fen, moves, simulations, certainty, tables) in cases(&tables) {
        let settings = graph_settings(certainty);
        let mut tree = graph_tree_after(fen, moves);
        search(&mut tree, simulations, usize::MAX, &settings, tables, &mut Network::new());
        for line in reusable_lines(&tree) {
            let line_moves: Vec<PackedMove> = line.iter().map(|&edge| tree.chess_move(edge)).collect();
            let new_root = tree.children[*line.last().expect("a line")] as usize;
            let (visits, value_sum) = (tree.nodes[new_root].visit_count, tree.nodes[new_root].value_sum);
            let mut at_once = copied(&tree);
            at_once.play_line(&line);
            at_once.drop_stale_leaves(&settings, tables).expect("the tables");
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
            advanced.drop_stale_candidates(&[], &candidates, &settings, tables).expect("the tables");
            advanced.compact(1, None);
            assert_same_trees(&at_once, &advanced);

            //one round at a time, a SharedTree's search is the Tree's (the test above)
            let mut shared = graph_shared_tree_after(fen, moves);
            shared_in_flight(&shared, simulations, 1, &settings, tables, &SharedNetwork::new(), usize::MAX);
            assert_same_trees(&tree, &shared.to_tree());
            shared.play_line(&edges_of(&shared, &line_moves));
            shared.drop_stale_leaves(&settings, tables).expect("the tables");
            assert_same_trees(&at_once, &shared.to_tree());
            lines += 1;
        }
    }
    assert!(lines >= 15, "{lines} lines");
}

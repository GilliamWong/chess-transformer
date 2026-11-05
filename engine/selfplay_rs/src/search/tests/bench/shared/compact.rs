//cargo test --release bench_compact -- --ignored --nocapture --test-threads 1
use super::*;

//a tree grown to BENCH_VISITS, with transpositions unless BENCH_TRANSPOSITIONS=0, and moved down its two most visited plies, its dead
//rest left in its arenas; then compacted on each count of BENCH_THREADS threads, in another order every repeat
#[test]
#[ignore]
fn bench_compact_copy() {
    let (visits, repeats) = (setting("BENCH_VISITS", 2_000_000u32), setting("BENCH_REPEATS", 3usize));
    let transpositions = setting("BENCH_TRANSPOSITIONS", 1u32) == 1;
    let thread_counts = settings_list("BENCH_THREADS", "1,2,4,8,16,32");
    let drivers = Drivers::from_environment(transpositions);
    let (fen, played, _) = POSITIONS[setting("BENCH_FIRST", 0usize)];
    let start = tree_after(fen, played);
    let mut tree = SharedTree::new(start.board.clone(), start.keys.clone()).with_transpositions(transpositions);
    let cache = SharedCache::new(5_000_000);
    drivers.drive(&tree, &cache, u64::MAX, visits, f64::MAX);
    drop(cache);
    let line: Vec<usize> = tree.principal_variation().into_iter().take(2).collect();
    tree.move_root(&line);
    println!(
        "{fen}: {} visits, transpositions {transpositions}, old arenas {:.1} MB, kept {} visits",
        visits,
        tree.footprint_bytes() as f64 / 1e6,
        tree.root().visits()
    );
    for repeat in 0..repeats {
        for turn in 0..thread_counts.len() {
            let threads = thread_counts[(turn + repeat) % thread_counts.len()];
            let began = Instant::now();
            let copy = tree.compact(threads, None, usize::MAX).expect("a compaction without a time");
            let seconds = began.elapsed().as_secs_f64();
            println!(
                "COMPACT_COPY\trepeat {repeat}\tthreads {threads}\tseconds {seconds:.4}\tkept_mb {:.1}\tkept_nodes {}\told_footprint_mb {:.1}",
                copy.footprint_bytes() as f64 / 1e6,
                copy.in_use().nodes,
                tree.footprint_bytes() as f64 / 1e6
            );
        }
    }
}

//BENCH_MOVES moves of a game with transpositions under BENCH_LIMIT_MB, played twice side by side: each move searched to BENCH_MOVE_VISITS
//at the root (the kept tree's visits count) in both trees, one after the other and in turn first, and then the move and reply that both
//have searched played in both, as Searches would. The compacting tree, a moved tree whose arenas give out freed room, is compacted on
//BENCH_COMPACT_THREADS threads (0: all) and its old arenas dropped on a thread of their own; the other is searched as it is, its dead rest
//freed by BENCH_GC sweepers beside the search. A tree without the move's nodes starts afresh. Per move and tree: the seconds from the end
//of the search to the next one's start, and the simulations a second of the search alone and with them. Each tree has a cache of its own
#[test]
#[ignore]
fn bench_compact_game() {
    let (moves, move_visits, gc) =
        (setting("BENCH_MOVES", 20usize), setting("BENCH_MOVE_VISITS", 2_000_000u32), setting("BENCH_GC", 2usize));
    let threads = match setting("BENCH_COMPACT_THREADS", 0usize) {
        0 => std::thread::available_parallelism().map_or(1, |threads| threads.get()),
        threads => threads,
    };
    let drivers = Drivers::from_environment(true);
    let (fen, played, _) = POSITIONS[setting("BENCH_FIRST", 0usize)];
    let start = tree_after(fen, played);
    let new_tree = || Arc::new(SharedTree::new(start.board.clone(), start.keys.clone()).with_transpositions(true));
    let arms = ["swept", "compacted"];
    let caches = [SharedCache::new(5_000_000), SharedCache::new(5_000_000)];
    println!(
        "{fen}: {} drivers, compactions on {threads} threads, {gc} sweepers, {moves} moves to {move_visits} visits, limit {}",
        drivers.drivers, drivers.memory_limit
    );
    let mut trees = [new_tree(), new_tree()];
    let mut past_limit = [false, false];
    for turn in 0..moves {
        let chess_moves = line_of_both(&trees[0], &trees[1]);
        for arm in [turn % 2, 1 - turn % 2] {
            let mut searching = std::mem::replace(&mut trees[arm], new_tree());
            let began = Instant::now();
            let (mut moved, mut compacted, mut gives_out) = (false, false, false);
            if turn > 0 {
                end_search(&searching);
                let edges = edges_along(&searching, &chess_moves);
                if let Some(edges) = edges.filter(|edges| edges.len() == 2) {
                    let mut tree = Arc::into_inner(searching).expect("the last owner");
                    tree.reopen();
                    tree.move_root(&edges);
                    gives_out = tree.gives_out_freed_room(drivers.memory_limit);
                    if arm == 1 && gives_out {
                        let copy = tree.compact(threads, None, usize::MAX).expect("a compaction without a time");
                        crate::drop_later(tree);
                        (tree, compacted) = (copy, true);
                    } else if gc == 0 {
                        tree.drain();
                    }
                    searching = Arc::new(tree);
                    if !compacted {
                        start_sweepers(&searching, gc);
                    }
                    moved = true;
                } else {
                    searching = new_tree();
                }
            }
            let move_seconds = began.elapsed().as_secs_f64();
            past_limit[arm] |= gives_out;
            let (visits_at_start, footprint_at_start) = (searching.root().visits(), footprint(&searching));
            let (seconds, _) = drivers.drive(&searching, &caches[arm], u64::MAX, move_visits, f64::MAX);
            let simulations = f64::from(searching.root().visits() - visits_at_start);
            println!(
                "COMPACT_GAME	move {turn}	arm {}	moved {moved}	past_limit {}	gives_out {gives_out}	compacted {compacted}	move_s {move_seconds:.4}	search_s {seconds:.3}	sims {simulations}	sims_per_s {:.0}	sims_per_s_with_move {:.0}	visits_at_start {visits_at_start}	footprint_mb_at_start {:.1}	footprint_mb {:.1}	in_use_mb {:.1}	full {}",
                arms[arm],
                past_limit[arm],
                simulations / seconds,
                simulations / (seconds + move_seconds),
                footprint_at_start as f64 / 1e6,
                footprint(&searching) as f64 / 1e6,
                searching.allocated_bytes() as f64 / 1e6,
                searching.is_full(),
            );
            trees[arm] = searching;
        }
    }
}

//the move and reply with the most visits in the two trees together, of those both have searched (a node with moves to search after the reply)
fn line_of_both(one: &SharedTree, other: &SharedTree) -> Vec<PackedMove> {
    let mut line = Vec::new();
    for _ in 0..2 {
        let searched = |tree: &SharedTree, chess_move: PackedMove| -> Option<u32> {
            let (mut node, mut played) = (tree.root(), line.clone());
            played.push(chess_move);
            let mut visits = 0;
            for &chess_move in &played {
                let edge = node.edges().find(|&edge| tree.chess_move(edge) == chess_move)?;
                visits = tree.visits(node, edge);
                node = tree.child(node, edge)?;
            }
            (played.len() < 2 || node.has_edges()).then_some(visits)
        };
        let Some(edges) = edges_along(one, &line).map(|edges| {
            let node = edges.iter().fold(one.root(), |node, &edge| one.child(node, edge).expect("a node"));
            node.edges().map(|edge| one.chess_move(edge)).collect::<Vec<_>>()
        }) else {
            break;
        };
        let best = edges
            .into_iter()
            .filter_map(|chess_move| Some((searched(one, chess_move)? + searched(other, chess_move)?, chess_move)))
            .max_by_key(|&(visits, _)| visits);
        let Some((_, chess_move)) = best else { break };
        line.push(chess_move);
    }
    line
}

//the edges of the moves down from the root as far as the tree has their nodes, the last node with moves to search
fn edges_along(tree: &SharedTree, chess_moves: &[PackedMove]) -> Option<Vec<usize>> {
    let (mut edges, mut node) = (Vec::new(), tree.root());
    for &chess_move in chess_moves {
        let edge = node.edges().find(|&edge| tree.chess_move(edge) == chess_move)?;
        node = tree.child(node, edge)?;
        edges.push(edge);
    }
    node.has_edges().then_some(edges)
}

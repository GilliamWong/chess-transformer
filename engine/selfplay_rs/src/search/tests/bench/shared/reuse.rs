//cargo test --release bench_reuse_gap -- --ignored --nocapture --test-threads 1
use super::*;

//BENCH_MOVES moves of a game, each searched to BENCH_MOVE_VISITS at the root and then played with its reply (the kept tree's visits
//count), as bench_shared_reuse does. Then the tree of the last move is searched for BENCH_ROUNDS rounds in three ways, one after the other
//and in another order every repeat: as it is (moved, its dead rest freed by the sweepers beside the search, or before it with
//BENCH_SWEPT=1), copied in breadth first order (compacted), and a tree grown afresh to the same visits. Each starts with a copy of the game's cache
#[test]
#[ignore]
fn bench_reuse_gap() {
    let (moves, move_visits, gc) = (setting("BENCH_MOVES", 8usize), setting("BENCH_MOVE_VISITS", 500_000u32), setting("BENCH_GC", 2usize));
    let (timed, repeats, swept) = (setting("BENCH_ROUNDS", 60_000u64), setting("BENCH_REPEATS", 3usize), setting("BENCH_SWEPT", 0u32) == 1);
    let arms = ["moved", "compacted", "fresh"];
    let drivers = Drivers::from_environment(true);
    let (fen, played, _) = POSITIONS[setting("BENCH_FIRST", 0usize)];
    let start = tree_after(fen, played);
    let new_tree = |board: &Chess, keys: &[u64]| Arc::new(SharedTree::new(board.clone(), keys.to_vec()).with_transpositions(true));
    println!(
        "{fen}: {} drivers, {gc} sweepers, swept {swept}, {moves} moves to {move_visits} visits, {timed} timed rounds",
        drivers.drivers
    );
    for repeat in 0..repeats {
        let cache = SharedCache::new(5_000_000);
        let (mut searching, mut played_moves, mut restarts) = (new_tree(&start.board, &start.keys), 0, 0);
        drivers.drive(&searching, &cache, u64::MAX, move_visits, f64::MAX);
        while played_moves < moves {
            let line: Vec<usize> = searching.principal_variation().into_iter().take(2).collect();
            let follows = line.len() == 2
                && searching
                    .child(searching.child(searching.root(), line[0]).expect("a node"), line[1])
                    .is_some_and(|node| node.has_edges());
            end_search(&searching);
            if follows {
                searching = Arc::new(move_owned(searching, &line, gc, &drivers));
                played_moves += 1;
                //the next move's search is all but the last: the tree it is left with is the one to measure
                if played_moves == moves {
                    break;
                }
                start_sweepers(&searching, gc);
            } else {
                restarts += 1;
                searching = new_tree(&start.board, &start.keys);
            }
            drivers.drive(&searching, &cache, u64::MAX, move_visits, f64::MAX);
        }
        let visits = searching.root().visits();
        let compacted = Arc::new(searching.compacted());
        let fresh = new_tree(&searching.board, &searching.keys);
        let fresh_cache = cache.duplicate(5_000_000);
        drivers.drive(&fresh, &fresh_cache, u64::MAX, visits, f64::MAX);
        let trees = [searching, compacted, fresh];
        for turn in 0..arms.len() {
            let arm = (turn + repeat) % arms.len();
            let (tree, cache) = (&trees[arm], cache.duplicate(5_000_000));
            let (in_use, arenas) = (tree.allocated_bytes(), footprint(tree));
            if arm == 0 {
                if swept { tree.drain() } else { start_sweepers(tree, gc) }
            }
            let start_visits = tree.root().visits();
            drivers.gather_nanos.store(0, Ordering::Relaxed);
            drivers.finish_nanos.store(0, Ordering::Relaxed);
            let (seconds, leaves) = drivers.drive(tree, &cache, timed, u32::MAX, f64::MAX);
            let simulations = tree.root().visits() - start_visits;
            println!(
                "REUSE_GAP\trepeat {repeat}\tarm {}\tevals_per_s {:.0}\tsims_per_s {:.0}\tus_per_eval {:.3}\tgather_us {:.3}\tfinish_us {:.3}\tvisits_at_start {start_visits}\tin_use_mb_at_start {:.1}\tfootprint_mb_at_start {:.1}\trestarts {restarts}",
                arms[arm],
                leaves as f64 / seconds,
                f64::from(simulations) / seconds,
                seconds * 1e6 / leaves as f64,
                drivers.gather_nanos.load(Ordering::Relaxed) as f64 / 1e3 / leaves as f64,
                drivers.finish_nanos.load(Ordering::Relaxed) as f64 / 1e3 / leaves as f64,
                in_use as f64 / 1e6,
                arenas as f64 / 1e6,
            );
            end_search(tree);
        }
    }
}

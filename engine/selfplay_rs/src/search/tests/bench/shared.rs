//cargo test --release bench_shared -- --ignored --nocapture --test-threads 1: bench_drivers' drivers on a SharedTree, no lock
//around the tree (the same environment: BENCH_VISITS ROUNDS IN_FLIGHT DRIVERS REPEATS FIRST DRIVER_CPUS COLLISIONS and the
//stand-in's), the line it prints is bench_drivers'. BENCH_TRANSPOSITIONS=1 searches with transpositions: the drivers grow
//the graph from the position to BENCH_VISITS themselves (a Tree's copy has no graph), then the timed rounds.
//BENCH_SOLID_THRESHOLD and BENCH_SOLID_BATCH set the tree's solid slots (with_solid_links)
use super::*;
use crate::search::shared::{SharedCache, SharedTree};

//BENCH_DRIVERS threads on one SharedTree, each with BENCH_IN_FLIGHT rounds out and the stand-in's answers at once
#[test]
#[ignore]
fn bench_shared_drivers() {
    let (visits, timed, repeats) =
        (setting("BENCH_VISITS", 750_000u32), setting("BENCH_ROUNDS", 12_000u64), setting("BENCH_REPEATS", 3usize));
    let (drivers, in_flight) = (setting("BENCH_DRIVERS", 8usize), setting("BENCH_IN_FLIGHT", 2usize));
    let collisions = setting("BENCH_COLLISIONS", 0u32);
    let transpositions = setting("BENCH_TRANSPOSITIONS", 0u32) == 1;
    let driver_cpus = if std::env::var("BENCH_DRIVER_CPUS").is_ok() { settings_list("BENCH_DRIVER_CPUS", "") } else { Vec::new() };
    let (stand_in, mut settings, move_ids) = (StandIn::from_environment(), settings(true), Network::new().move_ids);
    settings.transpositions = transpositions;
    let (fen, moves, _) = POSITIONS[setting("BENCH_FIRST", 0usize)];
    let (mut base, mut base_cache) = (tree_after(fen, moves), Cache::new(5_000_000));
    if !transpositions {
        rounds(&mut base, &mut base_cache, (usize::MAX, visits), 2, 0, stand_in, &settings, None, &move_ids);
    }
    println!(
        "{fen}: {} nodes; {drivers} shared drivers {driver_cpus:?}, {in_flight} rounds out each, collisions {collisions}, {timed} rounds, transpositions {transpositions}",
        base.nodes.len()
    );
    //the drivers' rounds until `rounds` have started or the root has `stop_at` visits: their seconds and leaves
    let drive = |tree: &SharedTree, cache: &SharedCache, rounds: u64, stop_at: u32| -> (f64, u64) {
        let (run, leaves) = ((Mutex::new(DriversRun::default()), Condvar::new()), AtomicU64::new(0));
        let started = Instant::now();
        std::thread::scope(|scope| {
            for driver in 0..drivers {
                let (tree, cache, leaves, settings, move_ids, run) = (tree, cache, &leaves, &settings, &move_ids, &run);
                let cpu = (!driver_cpus.is_empty()).then(|| driver_cpus[driver % driver_cpus.len()]);
                scope.spawn(move || {
                    if let Some(cpu) = cpu {
                        assert!(crate::affinity::pin_current(cpu), "driver pinned to cpu {cpu}");
                    }
                    let mut waiting: VecDeque<(u64, Vec<u8>)> = VecDeque::new();
                    let mut logits = Vec::new();
                    loop {
                        let next_round = {
                            let mut state = run.0.lock().expect("the run");
                            loop {
                                let more = state.started < rounds && tree.root().visits() < stop_at;
                                let blocked = state.starved == Some(state.answered) && state.out > waiting.len();
                                if more && waiting.len() < in_flight && !blocked {
                                    state.started += 1;
                                    state.out += 1;
                                    break Some((state.started - 1, state.answered));
                                }
                                if waiting.is_empty() && more {
                                    state = run.1.wait(state).expect("the run");
                                    continue;
                                }
                                break None;
                            }
                        };
                        if let Some((id, answered)) = next_round {
                            let mut boards = Vec::new();
                            let count = (tree.root().visits() / 16).clamp(1, 64);
                            let round = Round {
                                id,
                                leaves: count,
                                simulations: 3 * count,
                                collisions: if collisions == 0 { count } else { collisions },
                                visit_limit: u32::MAX,
                                memory_limit: usize::MAX,
                            };
                            let queued = tree.gather(&round, settings, None, Some(cache), &mut boards).expect("no tables");
                            tree.queue(id, queued);
                            leaves.fetch_add((boards.len() / TOKENS_PER_BOARD) as u64, Ordering::Relaxed);
                            if boards.is_empty() {
                                run.0.lock().expect("the run").starved = Some(answered);
                            }
                            waiting.push_back((id, boards));
                            continue;
                        }
                        let Some((id, boards)) = waiting.pop_front() else { break };
                        {
                            let _hold = tree.hold();
                            let queued = tree.take_round(id);
                            for (leaf, board) in queued.into_iter().zip(boards.as_chunks::<TOKENS_PER_BOARD>().0) {
                                let seed = StandIn::seed(board);
                                tree.finish_leaf(
                                    leaf,
                                    |column| stand_in.column(board, seed, column),
                                    ANSWER_WIDTH,
                                    move_ids,
                                    Some(cache),
                                    settings,
                                    usize::MAX,
                                    &mut logits,
                                );
                            }
                        }
                        let mut state = run.0.lock().expect("the run");
                        state.out -= 1;
                        state.answered += u64::from(!boards.is_empty());
                        run.1.notify_all();
                    }
                });
            }
        });
        (started.elapsed().as_secs_f64(), leaves.load(Ordering::Relaxed))
    };
    let (threshold, batch) = (setting("BENCH_SOLID_THRESHOLD", 100u32), setting("BENCH_SOLID_BATCH", 32usize));
    for repeat in 0..repeats {
        let (tree, cache) = if transpositions {
            let tree = SharedTree::new(base.board.clone(), base.keys.clone()).with_transpositions(true).with_solid_links(threshold, batch);
            let cache = SharedCache::new(5_000_000);
            drive(&tree, &cache, u64::MAX, visits);
            (tree, cache)
        } else {
            (SharedTree::from_tree(&base).with_solid_links(threshold, batch), SharedCache::from_cache(&base_cache, 5_000_000))
        };
        let start = tree.root().visits();
        let (seconds, leaves) = drive(&tree, &cache, timed, u32::MAX);
        let simulations = tree.root().visits() - start;
        println!(
            "SHARED\trepeat {repeat}\tevals_per_s {:.0}\tsims_per_s {:.0}\tus_per_eval {:.3}\tleaves_per_round {:.1}\torphans {}\texclusive {}\tvisits_at_start {start}\tbytes_per_visit {:.0}",
            leaves as f64 / seconds,
            f64::from(simulations) / seconds,
            seconds * 1e6 / leaves as f64,
            leaves as f64 / timed as f64,
            tree.counts.orphans.load(Ordering::Relaxed),
            tree.counts.exclusive.load(Ordering::Relaxed),
            tree.allocated_bytes() as f64 / f64::from(tree.root().visits())
        );
    }
}

mod replay;

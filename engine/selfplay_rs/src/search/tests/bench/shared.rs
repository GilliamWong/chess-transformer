//cargo test --release bench_shared -- --ignored --nocapture --test-threads 1
use super::*;
use crate::search::shared::{SharedCache, SharedTree};
use std::sync::Arc;

//BENCH_DRIVERS threads on one tree, as rounds.rs runs a thread a gpu, each with BENCH_IN_FLIGHT rounds out. The answers are lean rows
//as the engine's networks give them, unless BENCH_LEAN is 0 (whole rows read by move id)
struct Drivers {
    drivers: usize,
    lean: bool,
    in_flight: usize,
    collisions: u32,
    driver_cpus: Vec<usize>,
    settings: SearchSettings,
    move_ids: MoveIds,
    stand_in: StandIn,
    memory_limit: usize,
    //the nanoseconds the threads spent gathering and finishing, all together
    gather_nanos: AtomicU64,
    finish_nanos: AtomicU64,
}

impl Drivers {
    fn from_environment(transpositions: bool) -> Drivers {
        let driver_cpus = if std::env::var("BENCH_DRIVER_CPUS").is_ok() { settings_list("BENCH_DRIVER_CPUS", "") } else { Vec::new() };
        let mut settings = settings(true);
        settings.transpositions = transpositions;
        let memory_limit = setting("BENCH_LIMIT_MB", 0usize);
        Drivers {
            drivers: setting("BENCH_DRIVERS", 8usize),
            lean: setting("BENCH_LEAN", 1u32) == 1,
            in_flight: setting("BENCH_IN_FLIGHT", 2usize),
            collisions: setting("BENCH_COLLISIONS", 0u32),
            driver_cpus,
            settings,
            move_ids: Network::new().move_ids,
            stand_in: StandIn::from_environment(),
            memory_limit: if memory_limit == 0 { usize::MAX } else { memory_limit * 1_000_000 },
            gather_nanos: AtomicU64::new(0),
            finish_nanos: AtomicU64::new(0),
        }
    }

    fn round(&self, id: u64, root_visits: u32) -> Round {
        Round { memory_limit: self.memory_limit, ..round_of(id, root_visits, self.collisions, u32::MAX) }
    }

    //concurrent.rs's shared_finish of one leaf: `row` is the lean row that the engine's gpu makes from the columns of the descent
    fn finish(&self, tree: &SharedTree, cache: &SharedCache, leaf: PendingLeaf, board: &[u8], row: &mut Vec<f32>, logits: &mut Vec<f32>) {
        let seed = StandIn::seed(board);
        if !self.lean {
            let answer = |column| self.stand_in.column(board, seed, column);
            return tree.finish_leaf(leaf, answer, ANSWER_WIDTH, &self.move_ids, Some(cache), &self.settings, logits);
        }
        row.clear();
        row.extend(leaf.moves.iter().map(|&chess_move| self.stand_in.column(board, seed, self.move_ids.id(chess_move, leaf.turn))));
        row.resize(crate::LEAN_COLUMNS, self.stand_in.column(board, seed, ANSWER_WIDTH - 1));
        logits.clear();
        logits.extend_from_slice(&row[..leaf.moves.len()]);
        let value = row[crate::LEAN_COLUMNS - 1];
        tree.finish_leaf_in_order(leaf, logits, value, Some(cache), &self.settings);
    }

    //rounds until `rounds` are started, the root has `stop_at` visits or `seconds` have passed; returns the seconds it took and the leaves
    fn drive(&self, tree: &SharedTree, cache: &SharedCache, rounds: u64, stop_at: u32, seconds: f64) -> (f64, u64) {
        let (run, leaves) = ((Mutex::new(DriversRun::default()), Condvar::new()), AtomicU64::new(0));
        let started = Instant::now();
        std::thread::scope(|scope| {
            for driver in 0..self.drivers {
                let (run, leaves) = (&run, &leaves);
                let cpu = (!self.driver_cpus.is_empty()).then(|| self.driver_cpus[driver % self.driver_cpus.len()]);
                scope.spawn(move || {
                    if let Some(cpu) = cpu {
                        assert!(crate::affinity::pin_current(cpu), "driver pinned to cpu {cpu}");
                    }
                    let mut waiting: VecDeque<(u64, Vec<u8>)> = VecDeque::new();
                    let (mut logits, mut row) = (Vec::new(), Vec::new());
                    loop {
                        let next_round = {
                            let mut state = run.0.lock().expect("the run");
                            loop {
                                let more = state.started < rounds && tree.root().visits() < stop_at && !tree.is_full() && started.elapsed().as_secs_f64() < seconds;
                                let blocked = state.starved == Some(state.answered) && state.out > waiting.len();
                                if more && waiting.len() < self.in_flight && !blocked {
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
                            let round = self.round(id, tree.root().visits());
                            let gathering = Instant::now();
                            tree.queue(id, tree.gather(&round, &self.settings, None, Some(cache), &mut boards));
                            self.gather_nanos.fetch_add(gathering.elapsed().as_nanos() as u64, Ordering::Relaxed);
                            leaves.fetch_add((boards.len() / TOKENS_PER_BOARD) as u64, Ordering::Relaxed);
                            if boards.is_empty() {
                                run.0.lock().expect("the run").starved = Some(answered);
                            }
                            waiting.push_back((id, boards));
                            continue;
                        }
                        let Some((id, boards)) = waiting.pop_front() else { break };
                        let finishing = Instant::now();
                        {
                            let _hold = tree.hold();
                            for (leaf, board) in tree.take_round(id).into_iter().zip(boards.as_chunks::<TOKENS_PER_BOARD>().0) {
                                self.finish(tree, cache, leaf, board, &mut row, &mut logits);
                            }
                        }
                        self.finish_nanos.fetch_add(finishing.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        let mut state = run.0.lock().expect("the run");
                        state.out -= 1;
                        state.answered += u64::from(!boards.is_empty());
                        run.1.notify_all();
                    }
                });
            }
        });
        (started.elapsed().as_secs_f64(), leaves.load(Ordering::Relaxed))
    }
}

#[test]
#[ignore]
fn bench_shared_drivers() {
    let (visits, timed, repeats) =
        (setting("BENCH_VISITS", 750_000u32), setting("BENCH_ROUNDS", 12_000u64), setting("BENCH_REPEATS", 3usize));
    let transpositions = setting("BENCH_TRANSPOSITIONS", 0u32) == 1;
    let drivers = Drivers::from_environment(transpositions);
    let (fen, moves, _) = POSITIONS[setting("BENCH_FIRST", 0usize)];
    let (mut base, mut base_cache) = (tree_after(fen, moves), Cache::new(5_000_000));
    if !transpositions {
        rounds(&mut base, &mut base_cache, (usize::MAX, visits), 2, 0, drivers.stand_in, &drivers.settings, None, &drivers.move_ids);
    }
    println!(
        "{fen}: {} nodes; {} shared drivers {:?}, {} rounds out each, collisions {}, {timed} rounds, transpositions {transpositions}",
        base.nodes.len(),
        drivers.drivers,
        drivers.driver_cpus,
        drivers.in_flight,
        drivers.collisions
    );
    for repeat in 0..repeats {
        //from_tree copies no transposition graph, so with transpositions the drivers grow the tree from the position
        let (tree, cache) = if transpositions {
            let tree = SharedTree::new(base.board.clone(), base.keys.clone()).with_transpositions(true);
            let cache = SharedCache::new(5_000_000);
            drivers.drive(&tree, &cache, u64::MAX, visits, f64::MAX);
            (tree, cache)
        } else {
            (SharedTree::from_tree(&base), SharedCache::from_cache(&base_cache, 5_000_000))
        };
        let start = tree.root().visits();
        drivers.gather_nanos.store(0, Ordering::Relaxed);
        drivers.finish_nanos.store(0, Ordering::Relaxed);
        let (seconds, leaves) = drivers.drive(&tree, &cache, timed, u32::MAX, f64::MAX);
        let simulations = tree.root().visits() - start;
        println!(
            "SHARED\trepeat {repeat}\tevals_per_s {:.0}\tsims_per_s {:.0}\tus_per_eval {:.3}\tgather_us {:.3}\tfinish_us {:.3}\tleaves_per_round {:.1}\torphans {}\texclusive {}\tvisits_at_start {start}\tbytes_per_visit {:.0}",
            leaves as f64 / seconds,
            f64::from(simulations) / seconds,
            seconds * 1e6 / leaves as f64,
            drivers.gather_nanos.load(Ordering::Relaxed) as f64 / 1e3 / leaves as f64,
            drivers.finish_nanos.load(Ordering::Relaxed) as f64 / 1e3 / leaves as f64,
            leaves as f64 / timed as f64,
            tree.counts.orphans.load(Ordering::Relaxed),
            tree.counts.exclusive.load(Ordering::Relaxed),
            tree.allocated_bytes() as f64 / f64::from(tree.root().visits())
        );
    }
}

//BENCH_MOVES moves of a game with transpositions, each searched to BENCH_MOVE_VISITS at the root (the visits the kept tree has count):
//the engine's move and the most visited reply are played, as Searches would between two searches. Per move: the time from the call
//that moves the root to the first leaf gathered, the simulations a second in the first BENCH_FIRST_SECONDS and in the rest, and the memory.
//BENCH_GC is the sweepers (0 frees it all at the move, as a copy did); BENCH_LIMIT_MB the memory limit
#[test]
#[ignore]
fn bench_shared_reuse() {
    let (moves, move_visits, gc) = (setting("BENCH_MOVES", 20usize), setting("BENCH_MOVE_VISITS", 500_000u32), setting("BENCH_GC", 2usize));
    let first_seconds = setting("BENCH_FIRST_SECONDS", 2.0f64);
    let drivers = Drivers::from_environment(true);
    let (fen, played, _) = POSITIONS[setting("BENCH_FIRST", 0usize)];
    let start = tree_after(fen, played);
    let new_tree = || Arc::new(SharedTree::new(start.board.clone(), start.keys.clone()).with_transpositions(true));
    let cache = SharedCache::new(5_000_000);
    println!("{fen}: {} drivers, {gc} sweepers, {moves} moves to {move_visits} visits, limit {:?}", drivers.drivers, drivers.memory_limit);
    let (mut searching, mut footprint_top, mut games) = (new_tree(), 0, 0);
    for turn in 0..moves {
        let line: Vec<usize> = searching.principal_variation().into_iter().take(2).collect();
        let follows = line.len() == 2 && searching.child(searching.child(searching.root(), line[0]).expect("a node"), line[1]).is_some_and(|node| node.has_edges());
        let moved = turn > 0 && follows;
        //the opponent's time: the search is over and its sweepers gone
        if turn > 0 {
            end_search(&searching);
        }
        let began = Instant::now();
        let mut moved_at = 0.0;
        if moved {
            let tree = move_owned(searching, &line, gc, &drivers);
            moved_at = began.elapsed().as_secs_f64();
            searching = Arc::new(tree);
            start_sweepers(&searching, gc);
        } else if turn > 0 {
            games += 1;
            searching = new_tree();
        }
        //the first leaf: the time since the root moved, as the next search's first round gives it
        let first_round = {
            let mut boards = Vec::new();
            let round = Round { leaves: 1, simulations: 3, collisions: 1, ..drivers.round(u64::MAX, searching.root().visits()) };
            let leaves = searching.gather(&round, &drivers.settings, None, Some(&cache), &mut boards);
            //a tree that fills the memory limit by itself has no room for a leaf
            let at = if leaves.is_empty() { f64::NAN } else { began.elapsed().as_secs_f64() };
            searching.queue(u64::MAX, leaves);
            let _hold = searching.hold();
            let (mut logits, mut row) = (Vec::new(), Vec::new());
            for (leaf, board) in searching.take_round(u64::MAX).into_iter().zip(boards.as_chunks::<TOKENS_PER_BOARD>().0) {
                drivers.finish(&searching, &cache, leaf, board, &mut row, &mut logits);
            }
            at
        };
        let visits_at_start = searching.root().visits();
        let (early_seconds, _) = drivers.drive(&searching, &cache, u64::MAX, move_visits, first_seconds);
        let visits_after_early = searching.root().visits();
        let (rest_seconds, _) = drivers.drive(&searching, &cache, u64::MAX, move_visits, f64::MAX);
        let (early_sims, rest_sims) = (visits_after_early - visits_at_start, searching.root().visits() - visits_after_early);
        footprint_top = footprint_top.max(footprint(&searching));
        let (assists, reclaimed) = gc_counts(&searching);
        println!(
            "REUSE	move {turn}	moved {moved}	moved_us {:.1}	first_leaf_us {:.1}	visits_at_start {visits_at_start}	early_sims_per_s {:.0}	rest_sims_per_s {:.0}	in_use_mb {:.1}	footprint_mb {:.1}	full {}	assists {assists}	reclaimed {reclaimed}",
            moved_at * 1e6,
            first_round * 1e6,
            f64::from(early_sims) / early_seconds,
            f64::from(rest_sims) / rest_seconds.max(1e-9),
            searching.allocated_bytes() as f64 / 1e6,
            footprint(&searching) as f64 / 1e6,
            searching.is_full(),
        );
    }
    println!("REUSE	footprint_top_mb {:.1}	games_ended {games}", footprint_top as f64 / 1e6);
}

//what differs between a tree that moves its root and frees the rest beside the search, and one that copies the kept part out

//the search ends: no call gathers any more and the sweepers leave
fn end_search(tree: &Arc<SharedTree>) {
    tree.quiet(|| tree.close());
    tree.join_sweepers();
}

//the tree moved down `line` and ready to search again
fn move_owned(tree: Arc<SharedTree>, line: &[usize], gc: usize, drivers: &Drivers) -> SharedTree {
    let mut tree = Arc::into_inner(tree).expect("the last owner");
    tree.reopen();
    tree.move_root(line);
    tree.drop_stale_leaves(&drivers.settings, None);
    if gc == 0 {
        tree.drain();
    }
    tree
}

fn start_sweepers(tree: &Arc<SharedTree>, gc: usize) {
    tree.start_sweepers(gc);
}

fn footprint(tree: &SharedTree) -> usize {
    tree.footprint_bytes()
}

//(descents that freed garbage, nodes freed)
fn gc_counts(tree: &SharedTree) -> (u64, u64) {
    (tree.counts.assists.load(Ordering::Relaxed), tree.counts.reclaimed.load(Ordering::Relaxed))
}

mod replay;

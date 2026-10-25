//cargo test --release bench_ -- --ignored --nocapture --test-threads 1: the round loop's threads on one tree (stand-in
//network). Environment: BENCH_VISITS ROUNDS IN_FLIGHT REPEATS FIRST COLLISIONS (0: the round's leaves), the stand-in's
//VALUE_NOISE (logit sd) and PRIOR_SCALE, and bench_drivers' DRIVERS and DRIVER_CPUS
use super::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::Instant;

//four middlegames of the kit's positions, the opening's, and a rook and pawn ending the tables score
const POSITIONS: [(&str, &[&str], bool); 6] = [
    ("rnb1kbnr/ppp1pppp/8/4q3/8/2N5/PPPP1PPP/R1BQKBNR w KQkq - 0 1", &[], false),
    ("r1bqk2r/pppp1ppp/8/2bPp3/2P5/5N2/PPP2PPP/R1BQK2R b KQkq - 0 1", &[], false),
    ("2kr1bnr/pppq2pp/2npp3/8/2PNP3/8/PP3PPP/RNBQ1RK1 w - - 0 1", &[], false),
    ("rnbqkb1r/5ppp/2p1pn2/p7/Np1P4/4PN2/PP2BPPP/R1BQK2R w KQkq - 0 1", &[], false),
    (START, OPENING, false),
    (PAWNS_AND_ROOKS, &[], true),
];

fn setting<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|value| value.parse().ok()).unwrap_or(default)
}

fn settings_list(name: &str, default: &str) -> Vec<usize> {
    std::env::var(name).unwrap_or_else(|_| default.to_string()).split(',').map(|value| value.parse().expect("a number")).collect()
}

//a copy with room for `rounds` more rounds, so that no timed round regrows the vectors (a clone's first push would)
fn tree_with_room(tree: &Tree, rounds: usize) -> Tree {
    fn with_room<T: Clone>(vector: &[T], room: usize) -> Vec<T> {
        let mut copied = Vec::with_capacity(vector.len() + room);
        copied.extend_from_slice(vector);
        copied
    }
    let (nodes, edges) = (rounds * 4 * 64, rounds * 64 * MOST_MOVES / 4);
    Tree {
        nodes: with_room(&tree.nodes, nodes),
        node_keys: with_room(&tree.node_keys, nodes),
        moves: with_room(&tree.moves, edges),
        priors: with_room(&tree.priors, edges),
        links: with_room(&tree.links, nodes),
        slots: with_room(&tree.slots, nodes),
        ..copied(tree)
    }
}

//the cache's copy, with room for `rounds` more rounds' answers
fn cache_with_room(cache: &Cache, rounds: usize) -> Cache {
    let copied = |generation: &Generation| {
        let mut places = generation.places.clone();
        places.reserve(rounds * 64);
        let mut priors = Vec::with_capacity(generation.priors.len() + rounds * 64 * 64);
        priors.extend_from_slice(&generation.priors);
        Generation { places, priors }
    };
    Cache { newer: copied(&cache.newer), older: copied(&cache.older), generation: cache.generation }
}

//the tests' stand-in, its logits scaled and uniform noise of that standard deviation on its value's logit
#[derive(Clone, Copy)]
struct StandIn {
    prior_scale: f32,
    value_noise: f32,
}

impl StandIn {
    fn from_environment() -> StandIn {
        StandIn { prior_scale: setting("BENCH_PRIOR_SCALE", 1.0), value_noise: setting("BENCH_VALUE_NOISE", 0.0) }
    }

    fn seed(board: &[u8]) -> u64 {
        board.iter().fold(0, |hash, &token| mix(hash ^ u64::from(token)))
    }

    //a column of a board's answer: a move's logit, or the value in the last
    fn column(&self, board: &[u8], seed: u64, column: usize) -> f32 {
        if column < ANSWER_WIDTH - 1 {
            return self.prior_scale * answer(board, seed, column);
        }
        let value = answer(board, seed, column);
        let noise = self.value_noise * 3f32.sqrt() * (2.0 * uniform(seed ^ 0x5eed) - 1.0);
        1.0 / (1.0 + (-((value / (1.0 - value)).ln() + noise)).exp())
    }

    //the whole answer for a board, as the network's row of logits and value
    fn row(&self, board: &[u8]) -> Vec<f32> {
        let seed = StandIn::seed(board);
        (0..ANSWER_WIDTH).map(|column| self.column(board, seed, column)).collect()
    }

    //a row of `width` for a network with more moves ids than the stand-in's: logits for every column but the last, the value in it
    fn row_of(&self, board: &[u8], width: usize) -> Vec<f32> {
        let seed = StandIn::seed(board);
        let mut row: Vec<f32> = (0..width - 1).map(|column| self.prior_scale * 8.0 * uniform(seed ^ column as u64).powi(2)).collect();
        row.push(self.column(board, seed, ANSWER_WIDTH - 1));
        row
    }
}

//gather's rounds until `rounds` are gathered or the root has `visits`, up to in_flight waiting, then those waiting finished
#[allow(clippy::too_many_arguments)]
fn rounds(
    tree: &mut Tree,
    cache: &mut Cache,
    (rounds, visits): (usize, u32),
    in_flight: usize,
    collisions: u32,
    stand_in: StandIn,
    settings: &SearchSettings,
    tables: Option<&Tables>,
    move_ids: &MoveIds,
) {
    let mut gathered = 0;
    let mut waiting: VecDeque<(u64, Vec<Vec<f32>>)> = VecDeque::new();
    loop {
        let more = gathered < rounds && tree.root().visit_count as usize + tree.pending() < visits as usize;
        if more && waiting.len() < in_flight {
            let leaves = (tree.root().visit_count / 16).clamp(1, 64);
            let collisions = if collisions == 0 { leaves } else { collisions };
            let round =
                Round { id: gathered as u64, leaves, simulations: 3 * leaves, collisions, visit_limit: visits, memory_limit: usize::MAX };
            let mut boards = Vec::new();
            tree.gather(&round, settings, tables, Some(cache), &mut boards).expect("the tables");
            gathered += 1;
            waiting.push_back((round.id, boards.as_chunks::<TOKENS_PER_BOARD>().0.iter().map(|board| stand_in.row(board)).collect()));
            continue;
        }
        let Some((id, rows)) = waiting.pop_front() else { break };
        for row in &rows {
            tree.finish_leaf(id, |column| row[column], ANSWER_WIDTH, move_ids, Some(cache), settings);
        }
    }
}

//bench_drivers' rounds started and out, those with leaves finished, and the finished count when a round last found none
#[derive(Default)]
struct DriversRun {
    started: u64,
    out: usize,
    answered: u64,
    starved: Option<u64>,
}

//BENCH_DRIVERS threads on one tree behind a lock, as Searches' round loop runs a thread a gpu (rounds.rs), each with
//BENCH_IN_FLIGHT rounds out and the stand-in's answers at once (read in the finish, only the columns it asks for): a
//round gathered, then finished, BENCH_ROUNDS in all. DRIVER_CPUS pins the threads (in turn)
#[test]
#[ignore]
fn bench_drivers() {
    let (visits, timed, repeats) =
        (setting("BENCH_VISITS", 750_000u32), setting("BENCH_ROUNDS", 12_000u64), setting("BENCH_REPEATS", 3usize));
    let (drivers, in_flight) = (setting("BENCH_DRIVERS", 8usize), setting("BENCH_IN_FLIGHT", 2usize));
    let collisions = setting("BENCH_COLLISIONS", 0u32);
    let driver_cpus = if std::env::var("BENCH_DRIVER_CPUS").is_ok() { settings_list("BENCH_DRIVER_CPUS", "") } else { Vec::new() };
    let (stand_in, mut settings, move_ids) = (StandIn::from_environment(), settings(true), Network::new().move_ids);
    //BENCH_TRANSPOSITIONS=1 searches with transpositions
    settings.transpositions = setting("BENCH_TRANSPOSITIONS", 0u32) == 1;
    let (fen, moves, _) = POSITIONS[setting("BENCH_FIRST", 0usize)];
    let (mut base, mut base_cache) = (tree_after(fen, moves).with_transpositions(settings.transpositions), Cache::new(5_000_000));
    rounds(&mut base, &mut base_cache, (usize::MAX, visits), 2, 0, stand_in, &settings, None, &move_ids);
    println!(
        "{fen}: {} nodes; {drivers} drivers {driver_cpus:?}, {in_flight} rounds out each, collisions {collisions}, {timed} rounds",
        base.nodes.len()
    );
    for repeat in 0..repeats {
        let shared = Mutex::new((tree_with_room(&base, timed as usize), cache_with_room(&base_cache, timed as usize)));
        let (run, leaves) = ((Mutex::new(DriversRun::default()), Condvar::new()), AtomicU64::new(0));
        let start = base.root().visit_count;
        let started = Instant::now();
        std::thread::scope(|scope| {
            for driver in 0..drivers {
                let (shared, leaves, settings, move_ids) = (&shared, &leaves, &settings, &move_ids);
                let cpu = (!driver_cpus.is_empty()).then(|| driver_cpus[driver % driver_cpus.len()]);
                let run = &run;
                scope.spawn(move || {
                    if let Some(cpu) = cpu {
                        assert!(crate::affinity::pin_current(cpu), "driver pinned to cpu {cpu}");
                    }
                    let mut waiting: VecDeque<(u64, Vec<u8>)> = VecDeque::new();
                    loop {
                        //rounds.rs' choice: a driver whose last round found no leaves gathers again only once another
                        //round is back, rather than holding the tree with empty rounds
                        let next_round = {
                            let mut state = run.0.lock().expect("the run");
                            loop {
                                let blocked = state.starved == Some(state.answered) && state.out > waiting.len();
                                if state.started < timed && waiting.len() < in_flight && !blocked {
                                    state.started += 1;
                                    state.out += 1;
                                    break Some((state.started - 1, state.answered));
                                }
                                if waiting.is_empty() && state.started < timed {
                                    state = run.1.wait(state).expect("the run");
                                    continue;
                                }
                                break None;
                            }
                        };
                        if let Some((id, answered)) = next_round {
                            {
                                let mut boards = Vec::new();
                                let mut guard = shared.lock().expect("the tree");
                                let (tree, cache) = &mut *guard;
                                let count = (tree.root().visit_count / 16).clamp(1, 64);
                                let round = Round {
                                    id,
                                    leaves: count,
                                    simulations: 3 * count,
                                    collisions: if collisions == 0 { count } else { collisions },
                                    visit_limit: u32::MAX,
                                    memory_limit: usize::MAX,
                                };
                                tree.gather(&round, settings, None, Some(cache), &mut boards).expect("no tables");
                                drop(guard);
                                leaves.fetch_add((boards.len() / TOKENS_PER_BOARD) as u64, Ordering::Relaxed);
                                if boards.is_empty() {
                                    run.0.lock().expect("the run").starved = Some(answered);
                                }
                                waiting.push_back((id, boards));
                            }
                            continue;
                        }
                        let Some((id, boards)) = waiting.pop_front() else { break };
                        {
                            let mut guard = shared.lock().expect("the tree");
                            let (tree, cache) = &mut *guard;
                            for board in boards.as_chunks::<TOKENS_PER_BOARD>().0 {
                                let seed = StandIn::seed(board);
                                tree.finish_leaf(
                                    id,
                                    |column| stand_in.column(board, seed, column),
                                    ANSWER_WIDTH,
                                    move_ids,
                                    Some(cache),
                                    settings,
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
        let seconds = started.elapsed().as_secs_f64();
        let simulations = shared.lock().expect("the tree").0.root().visit_count - start;
        let leaves = leaves.load(Ordering::Relaxed);
        println!(
            "DRIVERS\trepeat {repeat}\tevals_per_s {:.0}\tsims_per_s {:.0}\tus_per_eval {:.3}\tleaves_per_round {:.1}",
            leaves as f64 / seconds,
            f64::from(simulations) / seconds,
            seconds * 1e6 / leaves as f64,
            leaves as f64 / timed as f64
        );
    }
}

mod shared;

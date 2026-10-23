//cargo test --release bench_ -- --ignored --nocapture --test-threads 1: gather_batch against gather (stand-in network)
//environment: BENCH_VISITS ROUNDS IN_FLIGHT WORKERS REPEATS FIRST POSITIONS SPIN_US COLLISIONS (0: the round's leaves)
//FENS (a json list), FINISH_BATCH (0: finish_leaf), and the stand-in's VALUE_NOISE (logit sd) and PRIOR_SCALE
use super::*;
use crate::search::batch::BatchCounts;
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
        children: with_room(&tree.children, edges),
        ..nodes_copied(tree)
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

    //the whole answer for a board, as the network's row of logits and value
    fn row(&self, board: &[u8]) -> Vec<f32> {
        let seed = board.iter().fold(0, |hash, &token| mix(hash ^ u64::from(token)));
        let mut row: Vec<f32> = (0..ANSWER_WIDTH - 1).map(|column| self.prior_scale * answer(board, seed, column)).collect();
        let value = answer(board, seed, ANSWER_WIDTH - 1);
        let noise = self.value_noise * 3f32.sqrt() * (2.0 * uniform(seed ^ 0x5eed) - 1.0);
        row.push(1.0 / (1.0 + (-((value / (1.0 - value)).ln() + noise)).exp()));
        row
    }
}

#[derive(Default, Clone)]
struct Run {
    gather: Duration,
    finish: Duration,
    rounds: u64,
    leaves: u64,
    simulations: u64,
    counts: BatchCounts,
}

//gather, gather_batch's passes of one visit (gather's search, its collisions counted), or gather_batch
#[derive(Clone, Copy)]
enum Method<'w> {
    Gather,
    Counted(&'w Workers),
    Batch(&'w Workers),
}

//rounds until `rounds` are gathered or the root has `visits`, up to in_flight waiting, then those waiting finished
#[allow(clippy::too_many_arguments)]
fn rounds(
    tree: &mut Tree,
    cache: &mut Cache,
    method: Method<'_>,
    (rounds, visits): (usize, u32),
    in_flight: usize,
    collisions: u32,
    stand_in: StandIn,
    settings: &SearchSettings,
    tables: Option<&Tables>,
    move_ids: &MoveIds,
) -> Run {
    let start = tree.root().visit_count;
    let mut run = Run::default();
    let mut waiting: VecDeque<(u64, Vec<Vec<f32>>)> = VecDeque::new();
    loop {
        let more = run.rounds < rounds as u64 && tree.root().visit_count as usize + tree.pending() < visits as usize;
        if more && waiting.len() < in_flight {
            let leaves = (tree.root().visit_count / 16).clamp(1, 64);
            let collisions = if collisions == 0 { leaves } else { collisions };
            let round =
                Round { id: run.rounds, leaves, simulations: 3 * leaves, collisions, visit_limit: visits, memory_limit: usize::MAX };
            let mut boards = Vec::new();
            let started = Instant::now();
            match method {
                Method::Gather => tree.gather(&round, settings, tables, Some(cache), &mut boards),
                Method::Counted(workers) => {
                    tree.gather_batch(&round, settings, tables, Some(cache), &mut boards, workers, 1, &mut run.counts)
                }
                Method::Batch(workers) => {
                    tree.gather_batch(&round, settings, tables, Some(cache), &mut boards, workers, u32::MAX, &mut run.counts)
                }
            }
            .expect("the tables");
            run.gather += started.elapsed();
            run.rounds += 1;
            run.leaves += (boards.len() / TOKENS_PER_BOARD) as u64;
            waiting.push_back((round.id, boards.as_chunks::<TOKENS_PER_BOARD>().0.iter().map(|board| stand_in.row(board)).collect()));
            continue;
        }
        let Some((id, rows)) = waiting.pop_front() else { break };
        let started = Instant::now();
        match method {
            Method::Batch(workers) if setting("BENCH_FINISH_BATCH", 1) == 1 => {
                let answers = |leaf: usize, column: usize| rows[leaf][column];
                tree.finish_batch(id, rows.len(), &answers, ANSWER_WIDTH, move_ids, Some(cache), settings, workers);
            }
            _ => {
                for row in &rows {
                    tree.finish_leaf(id, |column| row[column], ANSWER_WIDTH, move_ids, Some(cache), settings);
                }
            }
        }
        run.finish += started.elapsed();
    }
    run.simulations = u64::from(tree.root().visit_count - start);
    run
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

//the stand-in's priors over the tree's expanded nodes: the mean top prior and the mean entropy (nats)
fn prior_shape(tree: &Tree) -> (f64, f64) {
    let expanded: Vec<&Node> = tree.nodes.iter().filter(|node| node.edges().len() > 1).take(20_000).collect();
    let (mut top, mut entropy) = (0.0, 0.0);
    for node in &expanded {
        let priors = &tree.priors[node.edges()];
        top += f64::from(priors.iter().copied().fold(0.0, f32::max));
        entropy -= priors.iter().filter(|&&prior| prior > 0.0).map(|&prior| f64::from(prior) * f64::from(prior).ln()).sum::<f64>();
    }
    (top / expanded.len() as f64, entropy / expanded.len() as f64)
}

#[test]
#[ignore]
fn bench_gather_batch() {
    let (visits, timed, repeats) =
        (setting("BENCH_VISITS", 35_000u32), setting("BENCH_ROUNDS", 200usize), setting("BENCH_REPEATS", 3usize));
    let (in_flights, worker_counts) = (settings_list("BENCH_IN_FLIGHT", "2,16"), settings_list("BENCH_WORKERS", "1,2,4,8"));
    let first = setting("BENCH_FIRST", 0usize);
    let positions = setting("BENCH_POSITIONS", POSITIONS.len() - first);
    let spin = Duration::from_micros(setting("BENCH_SPIN_US", 50u64));
    let collisions = setting("BENCH_COLLISIONS", 0u32);
    let stand_in = StandIn::from_environment();
    let all_tables = tables();
    let settings = settings(true);
    let single = Workers::new(1, spin);
    let workers: Vec<Workers> = worker_counts.iter().map(|&count| Workers::new(count, spin)).collect();
    let mut methods = vec![("gather".to_string(), Method::Gather), ("counted".to_string(), Method::Counted(&single))];
    methods.extend(worker_counts.iter().zip(&workers).map(|(count, workers)| (format!("batch {count}"), Method::Batch(workers))));
    println!(
        "visits {visits}, {timed} rounds, {repeats} repeats, spin {spin:?}, collisions {collisions}, prior scale {}, value noise {}",
        stand_in.prior_scale, stand_in.value_noise
    );
    for &(fen, moves, with_tables) in &POSITIONS[first..first + positions] {
        let tables = all_tables.as_ref().filter(|_| with_tables);
        if with_tables && tables.is_none() {
            continue;
        }
        let move_ids = Network::new().move_ids;
        let (mut base, mut base_cache) = (tree_after(fen, moves), Cache::new(5_000_000));
        let built = Instant::now();
        rounds(&mut base, &mut base_cache, Method::Gather, (usize::MAX, visits), 2, 0, stand_in, &settings, tables, &move_ids);
        let (top, entropy) = prior_shape(&base);
        println!(
            "{fen}: {} nodes, {:.0} MB, built in {:.1} s; priors: top {top:.2}, entropy {entropy:.2} nats",
            base.nodes.len(),
            base.allocated_bytes() as f64 / 1e6,
            built.elapsed().as_secs_f64()
        );
        for &in_flight in &in_flights {
            let mut runs: Vec<Vec<Run>> = vec![Vec::new(); methods.len()];
            for repeat in 0..repeats {
                let order: Vec<usize> = if repeat % 2 == 0 { (0..methods.len()).collect() } else { (0..methods.len()).rev().collect() };
                for method in order {
                    let (mut tree, mut cache) = (tree_with_room(&base, timed), cache_with_room(&base_cache, timed));
                    let limits = (timed, u32::MAX);
                    let run = rounds(
                        &mut tree,
                        &mut cache,
                        methods[method].1,
                        limits,
                        in_flight,
                        collisions,
                        stand_in,
                        &settings,
                        tables,
                        &move_ids,
                    );
                    runs[method].push(run);
                }
            }
            let gather_us = median(runs[0].iter().map(|run| run.gather.as_secs_f64() * 1e6 / run.rounds as f64).collect());
            for ((name, _), method_runs) in methods.iter().zip(&runs) {
                let per_round =
                    |value: &dyn Fn(&Run) -> f64| median(method_runs.iter().map(|run| value(run) / run.rounds as f64).collect());
                let per_leaf = |value: &dyn Fn(&Run) -> f64| median(method_runs.iter().map(|run| value(run) / run.leaves as f64).collect());
                let us = |duration: Duration| duration.as_secs_f64() * 1e6;
                let round_us = per_round(&|run| us(run.gather));
                println!(
                    "ROW\t{fen}\t{in_flight}\t{name}\tround_us {round_us:.1}\tx {:.2}\tsim_us {:.3}\tleaf_us {:.3}\tfinish_leaf_us {:.3}\tleaves {:.1}\tsims {:.1}\t\
                     passes {:.2}\titems {:.1}\tcoll_leaf {:.2}\tcached_leaf {:.2}\tscored_leaf {:.2}\tunspent_leaf {:.2}\t\
                     share_out_us {:.1}\tworkers_us {:.1}\tpicking_us {:.1}\tpreparing_us {:.1}\tapply_us {:.1}",
                    gather_us / round_us,
                    median(method_runs.iter().map(|run| us(run.gather) / run.simulations as f64).collect()),
                    per_leaf(&|run| us(run.gather)),
                    per_leaf(&|run| us(run.finish)),
                    per_round(&|run| run.leaves as f64),
                    per_round(&|run| run.simulations as f64),
                    per_round(&|run| run.counts.passes as f64),
                    median(method_runs.iter().map(|run| run.counts.items as f64 / run.counts.passes.max(1) as f64).collect()),
                    per_leaf(&|run| run.counts.collisions as f64),
                    per_leaf(&|run| run.counts.cached as f64),
                    per_leaf(&|run| run.counts.scored as f64),
                    per_leaf(&|run| run.counts.unspent as f64),
                    per_round(&|run| us(run.counts.share_out)),
                    per_round(&|run| us(run.counts.workers)),
                    per_round(&|run| us(run.counts.picking)),
                    per_round(&|run| us(run.counts.preparing)),
                    per_round(&|run| us(run.counts.apply)),
                );
            }
        }
    }
}

//the fens of BENCH_FENS (a json list of strings), else POSITIONS' without moves or tables
fn fens() -> Vec<String> {
    match std::env::var("BENCH_FENS") {
        Ok(path) => {
            let text = std::fs::read_to_string(path).expect("the fens");
            text.split('"').skip(1).step_by(2).map(str::to_string).collect()
        }
        Err(_) => POSITIONS.iter().filter(|(_, moves, tables)| moves.is_empty() && !tables).map(|(fen, _, _)| fen.to_string()).collect(),
    }
}

//fixed-visit searches against gather's with 2 rounds in flight: same move, root visits' distance, collisions, rounds
#[test]
#[ignore]
fn bench_search_differences() {
    let visits = setting("BENCH_VISITS", 20_000u32);
    let stand_in = StandIn::from_environment();
    let settings = settings(true);
    let (single, workers) = (Workers::new(1, Duration::ZERO), Workers::new(4, Duration::from_micros(50)));
    let methods = [
        ("gather 2", Method::Counted(&single), 2, 0),
        ("gather 3", Method::Counted(&single), 3, 0),
        ("gather 16", Method::Counted(&single), 16, 0),
        ("batch 2", Method::Batch(&workers), 2, 0),
        ("batch 16", Method::Batch(&workers), 16, 0),
        ("batch 16, 1024 collisions", Method::Batch(&workers), 16, 1024),
    ];
    let fens = fens();
    let move_ids = Network::new().move_ids;
    //per method: agreements, distances, collisions, leaves, rounds
    let mut totals = vec![(0usize, 0.0, 0u64, 0u64, 0u64); methods.len()];
    for fen in &fens {
        let mut reference: Option<(usize, Vec<f64>)> = None;
        for (index, &(_, method, in_flight, collisions)) in methods.iter().enumerate() {
            let (mut tree, mut cache) = (tree_after(fen, &[]), Cache::new(5_000_000));
            let run =
                rounds(&mut tree, &mut cache, method, (usize::MAX, visits), in_flight, collisions, stand_in, &settings, None, &move_ids);
            let best = tree.most_visited_edge(tree.root_index()) - tree.root().edges().start;
            let total = f64::from(tree.root().visit_count - 1);
            let shares: Vec<f64> = tree.root().edges().map(|edge| f64::from(tree.visits(edge)) / total).collect();
            let (best_reference, shares_reference) = reference.get_or_insert_with(|| (best, shares.clone()));
            let distance = 0.5 * shares.iter().zip(shares_reference.iter()).map(|(share, other)| (share - other).abs()).sum::<f64>();
            let totals = &mut totals[index];
            totals.0 += usize::from(best == *best_reference);
            totals.1 += distance;
            totals.2 += run.counts.collisions;
            totals.3 += run.leaves;
            totals.4 += run.rounds;
        }
    }
    println!("{} positions, {visits} visits, prior scale {}, value noise {}", fens.len(), stand_in.prior_scale, stand_in.value_noise);
    for ((name, ..), (agreed, distance, collisions, leaves, rounds)) in methods.iter().zip(totals) {
        println!(
            "DIFF\t{name}\tsame move {:.1}%\tvisits' distance {:.3}\tcollisions/eval {:.2}\tleaves/round {:.1}\trounds/search {:.0}",
            100.0 * agreed as f64 / fens.len() as f64,
            distance / fens.len() as f64,
            collisions as f64 / leaves as f64,
            leaves as f64 / rounds as f64,
            rounds as f64 / fens.len() as f64
        );
    }
}

//Workers::run of an empty job: the time from the call to its return, after the caller has been busy for `gap`
#[test]
#[ignore]
fn bench_worker_latency() {
    let busy = |gap: Duration| {
        let started = Instant::now();
        while started.elapsed() < gap {
            std::hint::spin_loop();
        }
    };
    for count in [2, 4, 8] {
        for spin_us in [0, 50, 500] {
            let workers = Workers::new(count, Duration::from_micros(spin_us));
            for gap_us in [0, 30, 300] {
                let mut times = Vec::new();
                for _ in 0..2_000 {
                    busy(Duration::from_micros(gap_us));
                    let started = Instant::now();
                    workers.run(&|_| {});
                    times.push(started.elapsed().as_secs_f64() * 1e6);
                }
                times.sort_by(f64::total_cmp);
                println!(
                    "LATENCY\tworkers {count}\tspin_us {spin_us}\tgap_us {gap_us}\tmedian_us {:.2}\tp90_us {:.2}\tp99_us {:.2}",
                    times[times.len() / 2],
                    times[times.len() * 9 / 10],
                    times[times.len() * 99 / 100]
                );
            }
        }
    }
}

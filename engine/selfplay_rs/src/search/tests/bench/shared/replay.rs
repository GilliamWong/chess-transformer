//cargo test --release bench_replay -- --ignored --nocapture --test-threads 1: bench_drivers' drivers on the real network's
//recorded answers, a Tree behind a lock (bench_replay_locked) or a SharedTree (bench_replay_shared). Environment: BENCH_STORE
//(rust-selection's answers_all.pkl, exported by concurrent-descents/export_store.py), BENCH_MOVE_IDS (the move id table it is
//by, concurrent-descents/move_ids.bin), BENCH_POSITIONS (a line a position: game, move, fen, moves), BENCH_GAME and BENCH_MOVE
//(the position), BENCH_VISITS (what one thread searches of it first); the rest is bench_drivers'. A board the recording lacks
//gets the stand-in's answer, and "recorded" tells the recorded share of the timed rounds' boards
use super::*;

//an answer: the 1,968 move ids' logits, then the expected score
const WIDTH: usize = 1969;
const CACHE_POSITIONS: usize = 5_000_000;

fn engine_move_ids() -> MoveIds {
    let bytes = std::fs::read(std::env::var("BENCH_MOVE_IDS").expect("BENCH_MOVE_IDS")).expect("the move ids");
    MoveIds::new(bytes.as_chunks::<2>().0.iter().copied().map(i16::from_le_bytes).collect())
}

struct Replay {
    index: HashMap<u64, usize>,
    data: Vec<u8>,
    stand_in: StandIn,
    hits: AtomicU64,
    misses: AtomicU64,
}

fn read<const N: usize>(data: &[u8], at: usize) -> [u8; N] {
    data[at..at + N].try_into().expect("bytes of the file")
}

impl Replay {
    //"ANS1", u64 count, then a record a board: tokens, u16 moves, f32 value, i16 move ids, f32 logits
    fn load(path: &str) -> Replay {
        let data = std::fs::read(path).expect("BENCH_STORE, the exported answers");
        assert_eq!(&data[..4], b"ANS1");
        let count = u64::from_le_bytes(read(&data, 4)) as usize;
        let (mut index, mut at) = (HashMap::with_capacity(count), 12);
        for _ in 0..count {
            let head = at + TOKENS_PER_BOARD;
            index.insert(StandIn::seed(&data[at..head]), at);
            at = head + 6 + 6 * usize::from(u16::from_le_bytes(read(&data, head)));
        }
        Replay { index, data, stand_in: StandIn::from_environment(), hits: AtomicU64::new(0), misses: AtomicU64::new(0) }
    }

    fn row(&self, board: &[u8]) -> Vec<f32> {
        let Some(&at) = self.index.get(&StandIn::seed(board)) else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return self.stand_in.row_of(board, WIDTH);
        };
        self.hits.fetch_add(1, Ordering::Relaxed);
        let data = &self.data;
        let head = at + TOKENS_PER_BOARD;
        let (moves, value) = (usize::from(u16::from_le_bytes(read(data, head))), f32::from_le_bytes(read(data, head + 2)));
        let (ids, logits) = (head + 6, head + 6 + 2 * moves);
        let mut row = vec![-30.0; WIDTH];
        for place in 0..moves {
            row[i16::from_le_bytes(read(data, ids + 2 * place)) as usize] = f32::from_le_bytes(read(data, logits + 4 * place));
        }
        row[WIDTH - 1] = value;
        row
    }

    fn rows(&self, boards: &[u8]) -> Vec<Vec<f32>> {
        boards.as_chunks::<TOKENS_PER_BOARD>().0.iter().map(|board| self.row(board)).collect()
    }

    fn recorded(&self) -> (u64, u64) {
        (self.hits.load(Ordering::Relaxed), self.misses.load(Ordering::Relaxed))
    }
}

trait Searcher: Sync {
    fn visits(&self) -> u32;
    fn gather(&self, round: &Round) -> Vec<u8>;
    fn finish(&self, round: u64, rows: &[Vec<f32>]);
}

struct Locked {
    tree: Mutex<(Tree, Cache)>,
    settings: SearchSettings,
    move_ids: MoveIds,
}

impl Searcher for Locked {
    fn visits(&self) -> u32 {
        self.tree.lock().expect("the tree").0.root().visit_count
    }

    fn gather(&self, round: &Round) -> Vec<u8> {
        let mut boards = Vec::new();
        let mut guard = self.tree.lock().expect("the tree");
        let (tree, cache) = &mut *guard;
        tree.gather(round, &self.settings, None, Some(cache), &mut boards).expect("no tables");
        boards
    }

    fn finish(&self, round: u64, rows: &[Vec<f32>]) {
        let mut guard = self.tree.lock().expect("the tree");
        let (tree, cache) = &mut *guard;
        for row in rows {
            tree.finish_leaf(round, |column| row[column], WIDTH, &self.move_ids, Some(cache), &self.settings);
        }
    }
}

struct Concurrent {
    tree: SharedTree,
    cache: SharedCache,
    settings: SearchSettings,
    move_ids: MoveIds,
}

impl Searcher for Concurrent {
    fn visits(&self) -> u32 {
        self.tree.root().visits()
    }

    fn gather(&self, round: &Round) -> Vec<u8> {
        let mut boards = Vec::new();
        let queued = self.tree.gather(round, &self.settings, None, Some(&self.cache), &mut boards).expect("no tables");
        self.tree.queue(round.id, queued);
        boards
    }

    fn finish(&self, round: u64, rows: &[Vec<f32>]) {
        let _hold = self.tree.hold();
        let mut logits = Vec::new();
        for (leaf, row) in self.tree.take_round(round).into_iter().zip(rows) {
            let (cache, settings) = (Some(&self.cache), &self.settings);
            self.tree.finish_leaf(leaf, |column| row[column], WIDTH, &self.move_ids, cache, settings, usize::MAX, &mut logits);
        }
    }
}

//rounds on the recorded answers until the root has `visits`, two out at a time as the recorded searches had
fn base_search(fen: &str, moves: &[String], visits: u32, replay: &Replay, move_ids: &MoveIds, settings: &SearchSettings) -> (Tree, Cache) {
    let moves: Vec<&str> = moves.iter().map(String::as_str).collect();
    let (mut tree, mut cache) = (tree_after(fen, &moves), Cache::new(CACHE_POSITIONS));
    let mut waiting: VecDeque<(u64, Vec<Vec<f32>>)> = VecDeque::new();
    let mut id = 0;
    loop {
        if tree.root().visit_count as usize + tree.pending() < visits as usize && waiting.len() < 2 {
            let leaves = (tree.root().visit_count / 16).clamp(1, 64);
            let round = Round { id, leaves, simulations: 3 * leaves, collisions: leaves, visit_limit: visits, memory_limit: usize::MAX };
            let mut boards = Vec::new();
            tree.gather(&round, settings, None, Some(&cache), &mut boards).expect("no tables");
            waiting.push_back((id, replay.rows(&boards)));
            id += 1;
            continue;
        }
        let Some((round, rows)) = waiting.pop_front() else { break };
        for row in &rows {
            tree.finish_leaf(round, |column| row[column], WIDTH, move_ids, Some(&mut cache), settings);
        }
    }
    (tree, cache)
}

fn position(game: usize, place: usize) -> (String, Vec<String>) {
    let text = std::fs::read_to_string(std::env::var("BENCH_POSITIONS").expect("BENCH_POSITIONS")).expect("the positions");
    let fields = text
        .lines()
        .map(|line| line.split('\t').collect::<Vec<_>>())
        .find(|fields| fields[0] == game.to_string() && fields[1] == place.to_string())
        .expect("the position");
    (fields[2].to_string(), fields[3].split_whitespace().map(String::from).collect())
}

fn replay_drivers(shared: bool) {
    let (visits, timed) = (setting("BENCH_VISITS", 100_000u32), setting("BENCH_ROUNDS", 6_000u64));
    let (drivers, in_flight) = (setting("BENCH_DRIVERS", 8usize), setting("BENCH_IN_FLIGHT", 2usize));
    let collisions = setting("BENCH_COLLISIONS", 0u32);
    let driver_cpus = if std::env::var("BENCH_DRIVER_CPUS").is_ok() { settings_list("BENCH_DRIVER_CPUS", "") } else { Vec::new() };
    let search_settings = settings(true);
    let replay = Replay::load(&std::env::var("BENCH_STORE").expect("BENCH_STORE"));
    let (fen, moves) = position(setting("BENCH_GAME", 0usize), setting("BENCH_MOVE", 0usize));
    let move_ids = engine_move_ids();
    let built = Instant::now();
    let (base, base_cache) = base_search(&fen, &moves, visits, &replay, &move_ids, &search_settings);
    let (hits, misses) = replay.recorded();
    println!(
        "{fen} + {} moves: {} nodes, {:.1}% of the boards recorded ({hits} of {}), built in {:.1} s; {drivers} {} drivers, {in_flight} rounds out each, {timed} rounds",
        moves.len(),
        base.nodes.len(),
        100.0 * hits as f64 / (hits + misses) as f64,
        hits + misses,
        built.elapsed().as_secs_f64(),
        if shared { "shared" } else { "locked" }
    );
    let searcher: Box<dyn Searcher> = if shared {
        Box::new(Concurrent {
            tree: SharedTree::from_tree(&base),
            cache: SharedCache::from_cache(&base_cache, CACHE_POSITIONS),
            settings: search_settings,
            move_ids,
        })
    } else {
        Box::new(Locked {
            tree: Mutex::new((tree_with_room(&base, timed as usize), cache_with_room(&base_cache, timed as usize))),
            settings: search_settings,
            move_ids,
        })
    };
    drop((base, base_cache));
    let (run, leaves) = ((Mutex::new(DriversRun::default()), Condvar::new()), AtomicU64::new(0));
    //the nanoseconds the drivers spent in gather and in finish
    let spent = (AtomicU64::new(0), AtomicU64::new(0));
    let start = searcher.visits();
    let started = Instant::now();
    std::thread::scope(|scope| {
        for driver in 0..drivers {
            let (searcher, leaves, run, replay, spent) = (&searcher, &leaves, &run, &replay, &spent);
            let cpu = (!driver_cpus.is_empty()).then(|| driver_cpus[driver % driver_cpus.len()]);
            scope.spawn(move || {
                if let Some(cpu) = cpu {
                    assert!(crate::affinity::pin_current(cpu), "driver pinned to cpu {cpu}");
                }
                let mut waiting: VecDeque<(u64, Vec<Vec<f32>>)> = VecDeque::new();
                loop {
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
                        let count = (searcher.visits() / 16).clamp(1, 64);
                        let round = Round {
                            id,
                            leaves: count,
                            simulations: 3 * count,
                            collisions: if collisions == 0 { count } else { collisions },
                            visit_limit: u32::MAX,
                            memory_limit: usize::MAX,
                        };
                        let gathering = Instant::now();
                        let boards = searcher.gather(&round);
                        spent.0.fetch_add(gathering.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        //answered out of any lock
                        let rows = replay.rows(&boards);
                        leaves.fetch_add(rows.len() as u64, Ordering::Relaxed);
                        if rows.is_empty() {
                            run.0.lock().expect("the run").starved = Some(answered);
                        }
                        waiting.push_back((id, rows));
                        continue;
                    }
                    let Some((id, rows)) = waiting.pop_front() else { break };
                    let finishing = Instant::now();
                    searcher.finish(id, &rows);
                    spent.1.fetch_add(finishing.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    let mut state = run.0.lock().expect("the run");
                    state.out -= 1;
                    state.answered += u64::from(!rows.is_empty());
                    run.1.notify_all();
                }
            });
        }
    });
    let seconds = started.elapsed().as_secs_f64();
    let (simulations, leaves) = (searcher.visits() - start, leaves.load(Ordering::Relaxed));
    let (recorded_hits, recorded_misses) = replay.recorded();
    let (hits, misses) = (recorded_hits - hits, recorded_misses - misses);
    println!(
        "{}\trepeat 0\tevals_per_s {:.0}\tsims_per_s {:.0}\tus_per_eval {:.3}\tleaves_per_round {:.1}\trecorded {:.3}\tgather_us {:.3}\tfinish_us {:.3}",
        if shared { "SHARED" } else { "DRIVERS" },
        leaves as f64 / seconds,
        f64::from(simulations) / seconds,
        seconds * 1e6 / leaves as f64,
        leaves as f64 / timed as f64,
        hits as f64 / (hits + misses).max(1) as f64,
        spent.0.load(Ordering::Relaxed) as f64 / 1e3 / leaves as f64,
        spent.1.load(Ordering::Relaxed) as f64 / 1e3 / leaves as f64
    );
}

#[test]
#[ignore]
fn bench_replay_locked() {
    replay_drivers(false);
}

#[test]
#[ignore]
fn bench_replay_shared() {
    replay_drivers(true);
}

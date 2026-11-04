use crate::affinity;
use crate::rules::TOKENS_PER_BOARD;
use crate::{LEAN_COLUMNS, LEAN_ROW, RootNumbers, Search, SearchReport, Searches, State, root_numbers, root_score_of};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::any::Any;
use std::collections::{HashMap, VecDeque};
use std::ffi::{CStr, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const HOST_TO_DEVICE: i32 = 1;
const DEVICE_TO_HOST: i32 = 2;

//the cuda runtime's functions by address; pointers travel as usize, as on 64-bit targets
type SetDevice = unsafe extern "C" fn(i32) -> i32;
type Memcpy = unsafe extern "C" fn(usize, usize, usize, i32, usize) -> i32;
type OnStream = unsafe extern "C" fn(usize, usize) -> i32;
type Wait = unsafe extern "C" fn(usize) -> i32;
type ErrorString = unsafe extern "C" fn(i32) -> *const c_char;

#[derive(Clone, Copy)]
struct CudaCalls {
    set_device: SetDevice,
    memcpy: Memcpy,
    graph_launch: OnStream,
    event_record: OnStream,
    event_synchronize: Wait,
    stream_synchronize: Wait,
    error_string: ErrorString,
}

impl CudaCalls {
    fn new(addresses: &HashMap<String, usize>) -> PyResult<CudaCalls> {
        let address = |name: &str| match addresses.get(name) {
            Some(&address) if address != 0 => Ok(address),
            _ => Err(PyValueError::new_err(format!("no address for {name}"))),
        };
        //SAFETY: the addresses are the runtime's functions of these signatures, which the library torch loaded keeps
        unsafe {
            Ok(CudaCalls {
                set_device: std::mem::transmute::<usize, SetDevice>(address("cudaSetDevice")?),
                memcpy: std::mem::transmute::<usize, Memcpy>(address("cudaMemcpyAsync")?),
                graph_launch: std::mem::transmute::<usize, OnStream>(address("cudaGraphLaunch")?),
                event_record: std::mem::transmute::<usize, OnStream>(address("cudaEventRecord")?),
                event_synchronize: std::mem::transmute::<usize, Wait>(address("cudaEventSynchronize")?),
                stream_synchronize: std::mem::transmute::<usize, Wait>(address("cudaStreamSynchronize")?),
                error_string: std::mem::transmute::<usize, ErrorString>(address("cudaGetErrorString")?),
            })
        }
    }

    fn check(&self, error: i32, what: &str) -> PyResult<()> {
        if error == 0 {
            return Ok(());
        }
        //SAFETY: the runtime's strings are static
        let text = unsafe { CStr::from_ptr((self.error_string)(error)) }.to_string_lossy().into_owned();
        Err(PyRuntimeError::new_err(format!("{what}: cuda error {error}, {text}")))
    }
}

fn panicked<E>(_: E) -> PyErr {
    PyRuntimeError::new_err("a round loop thread panicked")
}

fn panic_error(payload: Box<dyn Any + Send>) -> PyErr {
    let message = payload.downcast_ref::<&str>().map(|message| message.to_string()).or_else(|| payload.downcast_ref::<String>().cloned());
    PyRuntimeError::new_err(format!("a round loop thread panicked: {}", message.unwrap_or_default()))
}

struct Lane {
    stream: usize,
    done: usize,
    rows_host: usize,
    rows: usize,
    answers_host: usize,
    //by size, smallest first: (size, the graph's exec, its answers on the device)
    graphs: Vec<(usize, usize, usize)>,
    pending: bool,
}

struct Network {
    device: i32,
    blocking_wait: bool,
    lanes: Vec<Lane>,
    cpu: Option<usize>,
}

fn item<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(key)?.ok_or_else(|| PyValueError::new_err(format!("no {key}")))
}

impl Network {
    fn get(network: &Bound<'_, PyDict>) -> PyResult<Network> {
        let mut lanes = Vec::new();
        let mut device = 0;
        for handles in item(network, "lanes")?.extract::<Vec<Bound<'_, PyDict>>>()? {
            device = item(&handles, "device")?.extract()?;
            let mut graphs: Vec<(usize, usize, usize)> = item(&handles, "graphs")?
                .extract::<HashMap<usize, (usize, usize)>>()?
                .into_iter()
                .map(|(size, (exec, answers))| (size, exec, answers))
                .collect();
            graphs.sort_unstable();
            lanes.push(Lane {
                stream: item(&handles, "stream")?.extract()?,
                done: item(&handles, "done")?.extract()?,
                rows_host: item(&handles, "rows_host")?.extract()?,
                rows: item(&handles, "rows")?.extract()?,
                answers_host: item(&handles, "answers_host")?.extract()?,
                graphs,
                pending: false,
            });
        }
        if lanes.is_empty() {
            return Err(PyValueError::new_err("a network without lanes"));
        }
        let cpu: Option<usize> = network.get_item("cpu")?.map(|cpu| cpu.extract()).transpose()?;
        if let Some(cpu) = cpu.filter(|cpu| !affinity::allowed().contains(cpu)) {
            return Err(PyValueError::new_err(format!("cpu {cpu} is not one this process may run on: {:?}", affinity::allowed())));
        }
        Ok(Network { device, blocking_wait: item(network, "blocking_wait")?.extract()?, lanes, cpu })
    }

    fn free_lane(&self) -> PyResult<usize> {
        self.lanes.iter().position(|lane| !lane.pending).ok_or_else(|| PyRuntimeError::new_err("more batches in flight than lanes"))
    }

    fn launch(&mut self, calls: &CudaCalls, index: usize, count: usize) -> PyResult<usize> {
        let lane = &mut self.lanes[index];
        let graph = lane
            .graphs
            .iter()
            .position(|&(size, _, _)| size >= count)
            .ok_or_else(|| PyValueError::new_err(format!("no graph holds {count} boards")))?;
        //SAFETY: the handles are the live lane's (RoundLoop keeps the network), the rows written fit the buffers
        unsafe {
            calls.check((calls.set_device)(self.device), "the lane's device")?;
            calls.check((calls.memcpy)(lane.rows, lane.rows_host, count * LEAN_ROW, HOST_TO_DEVICE, lane.stream), "the boards' copy")?;
            lane.pending = true;
            calls.check((calls.graph_launch)(lane.graphs[graph].1, lane.stream), "the graph's launch")?;
            calls.check((calls.event_record)(lane.done, lane.stream), "the batch's event")?;
        }
        Ok(graph)
    }

    fn collect(&mut self, calls: &CudaCalls, (index, graph): (usize, usize), count: usize) -> PyResult<&[f32]> {
        let lane = &mut self.lanes[index];
        //SAFETY: as launch's; the pinned answers stay put until this lane's next collect
        unsafe {
            calls.check((calls.set_device)(self.device), "the lane's device")?;
            if self.blocking_wait {
                calls.check((calls.event_synchronize)(lane.done), "the batch")?;
            }
            calls.check(
                (calls.memcpy)(lane.answers_host, lane.graphs[graph].2, count * LEAN_COLUMNS * 4, DEVICE_TO_HOST, lane.stream),
                "the answers' copy",
            )?;
            calls.check((calls.stream_synchronize)(lane.stream), "the batch")?;
            lane.pending = false;
            Ok(std::slice::from_raw_parts(lane.answers_host as *const f32, count * LEAN_COLUMNS))
        }
    }

    fn settle(&mut self, calls: &CudaCalls) {
        for lane in self.lanes.iter_mut().filter(|lane| lane.pending) {
            //SAFETY: as launch's
            unsafe {
                (calls.set_device)(self.device);
                (calls.stream_synchronize)(lane.stream);
            }
            lane.pending = false;
        }
    }

    fn room(&self) -> usize {
        self.lanes[0].graphs.last().map_or(0, |&(size, _, _)| size)
    }
}

#[derive(FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct LoopSettings {
    game: u64,
    leaves: u32,
    rounds_in_flight: usize,
    grow_leaves: u32,
    collisions: u32,
    simulations_per_leaf: u32,
    smart_pruning: bool,
    smart_pruning_factor: f64,
    certainty: bool,
    rounds_at_once: bool,
    speed_rounds: usize,
    round_times_kept: usize,
    fall_start_cp: f64,
    fall_full_cp: f64,
    max_reported_score: f64,
    expected_score_per_centipawn: f64,
    value_sharpness: f64,
}

//times are seconds since the loop's start
struct Run {
    simulations: Option<u32>,
    deadline: Option<f64>,
    extension: Option<(i64, f64)>,
    only_move: bool,
    report: Option<SearchReport>,
    stopping: bool,
    gathered: bool,
    rounds_out: usize,
    rounds_answered: u64,
    starved: Option<u64>,
    next_round: u64,
    progress: VecDeque<(f64, u32)>,
    root: RootNumbers,
    round_times: VecDeque<f64>,
    round_seconds: f64,
    error: Option<PyErr>,
    running: usize,
    batches: u64,
    boards: u64,
}

impl Run {
    fn seen(&mut self, root: Option<RootNumbers>) {
        if let Some(root) = root
            && root.0 >= self.root.0
        {
            self.root = root;
        }
    }

    fn stop(&mut self, report: Option<SearchReport>) {
        self.stopping = true;
        self.report = report;
    }

    fn simulations_left(&self) -> f64 {
        let (&(then, visits_then), &(now, visits_now)) =
            (self.progress.front().expect("progress"), self.progress.back().expect("a round's progress"));
        let mut left = self.simulations.map_or(f64::INFINITY, |simulations| f64::from(simulations) - f64::from(visits_now));
        if let Some(deadline) = self.deadline {
            let speed = if now > then { (f64::from(visits_now) - f64::from(visits_then)) / (now - then) } else { f64::INFINITY };
            left = left.min((deadline - now) * speed);
        }
        left
    }
}

struct Shared {
    run: Mutex<Run>,
    changed: Condvar,
    told: AtomicBool, //uci's stop
    started: Instant,
}

impl Shared {
    fn now(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    fn lock(&self) -> PyResult<MutexGuard<'_, Run>> {
        self.run.lock().map_err(panicked)
    }

    //for ending a loop: also after a thread panicked with the run held, when the rounds fail to lock it but the loop is still to end
    fn lock_always(&self) -> MutexGuard<'_, Run> {
        self.run.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

fn sharpen(score: f64, sharpness: f64) -> f64 {
    if sharpness == 1.0 || score == 0.0 || score == 1.0 { score } else { 1.0 / (1.0 + ((1.0 - score) / score).powf(sharpness)) }
}

fn median(times: &VecDeque<f64>) -> f64 {
    let mut sorted: Vec<f64> = times.iter().copied().collect();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 1 { sorted[middle] } else { (sorted[middle - 1] + sorted[middle]) / 2.0 }
}

fn running_search(state: &State, game: u64) -> PyResult<&Search> {
    state.searches.iter().find(|search| search.id == game).ok_or_else(|| PyValueError::new_err("no search"))
}

//a launched round: its name, its lane and graph (None for a round without leaves), its boards, when it was launched
type Launched = (u64, Option<(usize, usize)>, usize, f64);

//one network's rounds, as engine_search.py's rounds_on
struct Rounds<'a> {
    searches: &'a Searches,
    settings: &'a LoopSettings,
    shared: &'a Shared,
    calls: CudaCalls,
    networks: usize,
}

impl Rounds<'_> {
    fn round_leaves(&self, simulations: Option<u32>, visits: u32) -> (u32, usize) {
        let settings = self.settings;
        let (mut count, mut rounds) = (settings.leaves, settings.rounds_in_flight * self.networks);
        if settings.grow_leaves > 0 {
            let visits = i64::from(visits);
            let done_or_left = simulations.map_or(visits, |simulations| visits.min(i64::from(simulations) - visits));
            count = done_or_left.div_euclid(i64::from(settings.grow_leaves)).clamp(1, i64::from(settings.leaves)) as u32;
            rounds = if count == settings.leaves { rounds } else { 1 };
        }
        (count, rounds)
    }

    fn extra_for_fall(&self, (last, seconds): (i64, f64)) -> PyResult<f64> {
        let settings = self.settings;
        let score = if self.searches.concurrent_descents {
            //a search another thread's finish just ended needs no extension
            let Some(score) = self.searches.shared_root_score_if_running(settings.game)? else { return Ok(0.0) };
            score
        } else {
            let state = self.searches.lock()?;
            root_score_of(running_search(&state, settings.game)?)
        };
        let score = sharpen(score, 1.0 / settings.value_sharpness).clamp(1.0 - settings.max_reported_score, settings.max_reported_score);
        let centipawns = ((score / (1.0 - score)).ln() / settings.expected_score_per_centipawn).round_ties_even();
        let fall = last as f64 - centipawns;
        let share = ((fall - settings.fall_start_cp) / (settings.fall_full_cp - settings.fall_start_cp)).clamp(0.0, 1.0);
        Ok(share * seconds)
    }

    fn rounds_on(&self, network: &mut Network) -> PyResult<()> {
        let (settings, shared, calls) = (self.settings, self.shared, &self.calls);
        let mut rounds: VecDeque<Launched> = VecDeque::new();
        let mut last_answers: Option<f64> = None;
        let mut seen: Option<RootNumbers> = None;
        let room = network.room();
        loop {
            let mut gathering = None;
            {
                let mut run = shared.lock()?;
                run.seen(seen.take());
                while run.report.is_none() && !run.stopping {
                    let round_times = if settings.rounds_at_once { rounds.len().max(1) } else { rounds.len() + 1 } as f64;
                    let told_to_stop = shared.told.load(Ordering::Relaxed);
                    let late = run.deadline.is_some_and(|deadline| shared.now() + round_times * run.round_seconds >= deadline);
                    //as lc0, no stop before the root has an answer: a search with no visit and no leaf out cannot be stopped
                    if run.gathered && run.root.0 > 0 && (told_to_stop || late) {
                        if let Some(extension) = run.extension
                            && !told_to_stop
                        {
                            let extra = self.extra_for_fall(extension)?;
                            run.deadline = run.deadline.map(|deadline| deadline + extra);
                            run.extension = None;
                            continue;
                        }
                        let report = self.searches.stop_running(settings.game)?;
                        run.stop(report);
                        shared.changed.notify_all();
                        break;
                    }
                    let (count, in_flight) = self.round_leaves(run.simulations, run.root.0);
                    let blocked = run.starved == Some(run.rounds_answered) && run.rounds_out > rounds.len();
                    if run.rounds_out < in_flight && rounds.len() < settings.rounds_in_flight && !blocked {
                        gathering = Some((run.next_round, count, run.rounds_answered));
                        run.next_round += 1;
                        run.rounds_out += 1;
                        run.gathered = true;
                    } else if rounds.is_empty() {
                        //the rounds out are other networks': one must come back first
                        run = shared.changed.wait(run).map_err(panicked)?;
                        continue;
                    }
                    break;
                }
                if gathering.is_none() && rounds.is_empty() {
                    return Ok(());
                }
            }
            if let Some((name, count, answered)) = gathering {
                if count as usize > room {
                    return Err(PyValueError::new_err(format!("rounds of {count} leaves, past the lanes' {room}")));
                }
                let lane = network.free_lane()?;
                let (written, root) = self.searches.descend_rows(
                    network.lanes[lane].rows_host,
                    room,
                    count,
                    settings.collisions.min(count),
                    settings.simulations_per_leaf * count,
                    name,
                    settings.game,
                )?;
                seen = root;
                if written == 0 {
                    shared.lock()?.starved = Some(answered);
                }
                let launched_at = shared.now();
                let launched = if written > 0 { Some((lane, network.launch(calls, lane, written)?)) } else { None };
                rounds.push_back((name, launched, written, launched_at));
                continue;
            }

            let (name, launched, boards, launched_at) = rounds.pop_front().expect("a round out");
            let answers: &[f32] = match launched {
                Some(launched) => network.collect(calls, launched, boards)?,
                None => &[],
            };
            let answered = shared.now();
            //concurrent descents finish outside the run's lock; a thread may then find the search gone before its report is
            //in, which every call it makes then takes as the search being done
            let finished = if self.searches.concurrent_descents {
                let reports = self.searches.shared_finish(answers, Some(name))?;
                Some((reports, self.searches.shared_root_numbers(settings.game)?))
            } else {
                None
            };
            let mut run = shared.lock()?;
            //under the lock, so that no thread finds the search gone from Searches (done) before its report is in
            let (reports, root) = match finished {
                Some(finished) => finished,
                None => {
                    let mut state = self.searches.lock()?;
                    let reports = self.searches.finish_round(&mut state, answers, Some(name))?;
                    (reports, root_numbers(&state, settings.game))
                }
            };
            run.seen(root);
            run.rounds_out -= 1;
            if launched.is_some() {
                run.rounds_answered += 1;
                run.batches += 1;
                run.boards += boards as u64;
                //rounds come back one after another while the gpu is kept busy; the search's first from its launch
                run.round_times.push_back(answered - last_answers.unwrap_or(launched_at));
                while run.round_times.len() > settings.round_times_kept {
                    run.round_times.pop_front();
                }
                run.round_seconds = median(&run.round_times);
                last_answers = Some(answered);
            }
            if let Some(report) = reports.into_iter().next() {
                run.report = Some(report);
            } else if run.report.is_none() && settings.smart_pruning && !run.stopping {
                //a round without leaves counts too: all its simulations may have ended the game (a mate found)
                let (visits, most, second) = run.root;
                let now = shared.now();
                run.progress.push_back((now, visits));
                while run.progress.len() > settings.speed_rounds {
                    run.progress.pop_front();
                }
                let decided = visits > 0 && f64::from(most - second) > run.simulations_left() / settings.smart_pruning_factor;
                let proven = settings.certainty
                    && if self.searches.concurrent_descents {
                        self.searches.shared_tree(settings.game)?.is_some_and(|tree| tree.root_decided())
                    } else {
                        let state = self.searches.lock()?;
                        let tree = &running_search(&state, settings.game)?.tree;
                        tree.root_decided()
                    };
                //once: a stop smart pruning would make is extended instead when the score has fallen
                let mut extra = 0.0;
                if decided
                    && !(run.only_move || proven)
                    && let Some(extension) = run.extension
                {
                    extra = self.extra_for_fall(extension)?;
                    run.extension = None;
                }
                if extra > 0.0 {
                    run.deadline = run.deadline.map(|deadline| deadline + extra);
                } else if visits > 0 && (run.only_move || decided || proven) {
                    let report = self.searches.stop_running(settings.game)?;
                    run.stop(report);
                }
            }
            shared.changed.notify_all();
        }
    }
}

//SAFETY: rows points at room for every board's LEAN_ROW bytes, which nothing else touches meanwhile
pub(crate) unsafe fn write_lean_rows(rows: *mut u8, tokens: &[u8], columns: &[u16]) {
    for (index, (board, board_columns)) in
        tokens.as_chunks::<TOKENS_PER_BOARD>().0.iter().zip(columns.as_chunks::<LEAN_COLUMNS>().0).enumerate()
    {
        unsafe {
            let row = rows.add(index * LEAN_ROW);
            std::ptr::copy_nonoverlapping(board.as_ptr(), row, TOKENS_PER_BOARD);
            for (slot, column) in board_columns.iter().enumerate() {
                std::ptr::copy_nonoverlapping(column.to_le_bytes().as_ptr(), row.add(TOKENS_PER_BOARD + 2 * slot), 2);
            }
        }
    }
}

#[pyclass]
pub(crate) struct RoundLoop {
    shared: Arc<Shared>,
    game: u64,
    threads: Vec<JoinHandle<()>>,
    //held while the threads use them; the last Arc of the searches is dropped with the gil
    _searches: Arc<Py<Searches>>,
    _networks: Vec<Py<PyAny>>,
}

#[pymethods]
impl RoundLoop {
    //None if the rounds take longer than `seconds`; a thread's failure (a cuda error, a panic) is raised here, once all the threads have
    //ended, and the search is then gone from the searches, with the rounds it had out
    fn wait(&mut self, py: Python<'_>, seconds: f64) -> PyResult<Option<SearchReport>> {
        let done = py.detach(|| {
            let run = self.shared.lock_always();
            let (run, _) = self
                .shared
                .changed
                .wait_timeout_while(run, Duration::from_secs_f64(seconds.max(0.0)), |run| run.running > 0)
                .unwrap_or_else(PoisonError::into_inner);
            run.running == 0
        });
        if !done {
            return Ok(None);
        }
        py.detach(|| {
            for thread in self.threads.drain(..) {
                thread.join().map_err(panicked)?;
            }
            Ok::<_, PyErr>(())
        })?;
        let mut run = self.shared.lock_always();
        if let Some(error) = run.error.take() {
            drop(run);
            self._searches.get().abort(self.game);
            return Err(error);
        }
        run.report.take().map(Some).ok_or_else(|| PyRuntimeError::new_err("the round loop ended without a report"))
    }

    fn tell_stop(&self) {
        self.shared.told.store(true, Ordering::Relaxed);
    }

    fn counts(&self) -> PyResult<(Vec<f64>, u64, u64)> {
        let run = self.shared.lock()?;
        Ok((run.round_times.iter().copied().collect(), run.batches, run.boards))
    }
}

//a loop dropped before its report (an exception in the waiting thread) stops and joins its threads before the networks can go
impl Drop for RoundLoop {
    fn drop(&mut self) {
        self.shared.told.store(true, Ordering::Relaxed);
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn start_loop(
    searches: Arc<Py<Searches>>,
    networks: Vec<(Py<PyAny>, Bound<'_, PyDict>)>,
    addresses: HashMap<String, usize>,
    settings: LoopSettings,
    simulations: Option<u32>,
    seconds: Option<f64>,
    extension: Option<(i64, f64)>,
    only_move: bool,
    root: RootNumbers,
    round_times: Vec<f64>,
) -> PyResult<RoundLoop> {
    let calls = CudaCalls::new(&addresses)?;
    let built = networks.iter().map(|(_, network)| Network::get(network)).collect::<PyResult<Vec<_>>>()?;
    let objects = networks.into_iter().map(|(object, _)| object).collect();
    let game = settings.game;
    let round_times: VecDeque<f64> = round_times.into();
    let round_seconds = median(&round_times);
    let started = Instant::now();
    let shared = Arc::new(Shared {
        run: Mutex::new(Run {
            simulations,
            deadline: seconds,
            extension,
            only_move,
            report: None,
            stopping: false,
            gathered: false,
            rounds_out: 0,
            rounds_answered: 0,
            starved: None,
            next_round: 0,
            progress: VecDeque::new(),
            root,
            round_times,
            round_seconds,
            error: None,
            running: built.len(),
            batches: 0,
            boards: 0,
        }),
        changed: Condvar::new(),
        told: AtomicBool::new(false),
        started,
    });
    let settings = Arc::new(settings);
    let count = built.len();
    let threads = built
        .into_iter()
        .map(|mut network| {
            let (searches, shared, settings) = (Arc::clone(&searches), Arc::clone(&shared), Arc::clone(&settings));
            std::thread::spawn(move || {
                if let Some(cpu) = network.cpu {
                    affinity::pin_current(cpu);
                }
                let rounds = Rounds { searches: searches.get(), settings: &settings, shared: &shared, calls, networks: count };
                //a panic ends the thread as an error does: the loop is to end, not to wait for a thread that is gone
                let failed =
                    catch_unwind(AssertUnwindSafe(|| rounds.rounds_on(&mut network))).unwrap_or_else(|payload| Err(panic_error(payload))).err();
                if failed.is_some() {
                    network.settle(&calls);
                }
                {
                    let mut run = shared.lock_always();
                    if let Some(error) = failed {
                        run.stopping = true;
                        run.error.get_or_insert(error);
                    }
                    run.running -= 1;
                }
                shared.changed.notify_all();
            })
        })
        .collect();
    Ok(RoundLoop { shared, game, threads, _searches: searches, _networks: objects })
}

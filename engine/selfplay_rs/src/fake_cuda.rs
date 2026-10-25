//gpus that are not there, for timing Searches' own round loop (rounds.rs) on more gpus than the machine has: the cuda
//runtime's calls the loop makes (fake_cuda_addresses, in cuda_runtime.addresses' place), and graphs that answer a lean
//round (fake_cuda_graph). A graph's launch writes every board's answer at once (its legal moves' logits from a hash of
//its tokens, its expected score near a draw) and makes its stream busy for a batch's time, fixed plus per board, after
//the batches its device (the thread's set_device) has already: one gpu runs one batch at a time, whatever its lanes.
//The waits spin until then, as a blocking wait holds the thread. Streams, events and devices are numbers below STREAMS
use crate::rules::TOKENS_PER_BOARD;
use crate::{LEAN_COLUMNS, LEAN_ROW};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::ffi::c_char;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{OnceLock, RwLock};
use std::time::{Duration, Instant};

const STREAMS: usize = 256;

//a graph: its boards, where its rows and answers are, and a batch's time
struct Graph {
    size: usize,
    rows: usize,
    answers: usize,
    fixed: Duration,
    per_board: Duration,
}

static GRAPHS: RwLock<Vec<Graph>> = RwLock::new(Vec::new());
//when each stream's batch is done, in nanoseconds since START, and each event's stream
static READY: [AtomicU64; STREAMS] = [const { AtomicU64::new(0) }; STREAMS];
static EVENT_STREAMS: [AtomicU64; STREAMS] = [const { AtomicU64::new(0) }; STREAMS];
//when each device's batches are all done, in nanoseconds since START
static DEVICE_READY: [AtomicU64; STREAMS] = [const { AtomicU64::new(0) }; STREAMS];

thread_local!(static DEVICE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) });
static START: OnceLock<Instant> = OnceLock::new();

fn nanos() -> u64 {
    START.get_or_init(Instant::now).elapsed().as_nanos() as u64
}

fn wait_for(stream: usize) {
    let ready = READY[stream % STREAMS].load(Ordering::Acquire);
    while nanos() < ready {
        std::hint::spin_loop();
        std::thread::yield_now();
    }
}

extern "C" fn set_device(device: i32) -> i32 {
    DEVICE.with(|current| current.set(device as usize % STREAMS));
    0
}

extern "C" fn memcpy(target: usize, source: usize, bytes: usize, _kind: i32, _stream: usize) -> i32 {
    //SAFETY: the loop copies between buffers fake_cuda_graph's caller keeps alive, of the sizes it gave
    unsafe { std::ptr::copy_nonoverlapping(source as *const u8, target as *mut u8, bytes) };
    0
}

//a number from a board's tokens, its answer's seed
fn seed(tokens: &[u8]) -> u64 {
    tokens.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, &token| (hash ^ u64::from(token)).wrapping_mul(0x0000_0100_0000_01b3))
}

extern "C" fn graph_launch(exec: usize, stream: usize) -> i32 {
    let graphs = GRAPHS.read().expect("the fake graphs");
    let Some(graph) = exec.checked_sub(1).and_then(|index| graphs.get(index)) else { return 1 };
    //SAFETY: the graph's rows and answers are its caller's buffers of its size (fake_cuda_graph)
    let (rows, answers) = unsafe {
        (
            std::slice::from_raw_parts(graph.rows as *const u8, graph.size * LEAN_ROW),
            std::slice::from_raw_parts_mut(graph.answers as *mut f32, graph.size * LEAN_COLUMNS),
        )
    };
    //a row's columns end in the expected score's, which pads them after the legal moves': only those get logits
    for (row, answer) in rows.chunks_exact(LEAN_ROW).zip(answers.chunks_exact_mut(LEAN_COLUMNS)) {
        let board = seed(&row[..TOKENS_PER_BOARD]);
        let column = |slot: usize| u16::from_le_bytes([row[TOKENS_PER_BOARD + 2 * slot], row[TOKENS_PER_BOARD + 2 * slot + 1]]);
        let score_column = column(LEAN_COLUMNS - 1);
        for slot in (0..LEAN_COLUMNS - 1).take_while(|&slot| column(slot) != score_column) {
            let mixed = (board ^ u64::from(column(slot)).wrapping_mul(0x9e37_79b9_7f4a_7c15)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            answer[slot] = ((mixed >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0;
        }
        answer[LEAN_COLUMNS - 1] = 0.35 + 0.3 * ((board >> 40) as f32 / (1u64 << 24) as f32);
    }
    let busy = (graph.fixed + graph.per_board * graph.size as u32).as_nanos() as u64;
    //after the device's batches already out (its one thread launches them, so no other thread moves this meanwhile)
    let device = DEVICE.with(std::cell::Cell::get);
    let done = DEVICE_READY[device].load(Ordering::Acquire).max(nanos()) + busy;
    DEVICE_READY[device].store(done, Ordering::Release);
    READY[stream % STREAMS].store(done, Ordering::Release);
    0
}

extern "C" fn event_record(event: usize, stream: usize) -> i32 {
    EVENT_STREAMS[event % STREAMS].store(stream as u64, Ordering::Release);
    0
}

extern "C" fn event_synchronize(event: usize) -> i32 {
    wait_for(EVENT_STREAMS[event % STREAMS].load(Ordering::Acquire) as usize);
    0
}

extern "C" fn stream_synchronize(stream: usize) -> i32 {
    wait_for(stream);
    0
}

extern "C" fn error_string(_error: i32) -> *const c_char {
    c"a fake gpu's error".as_ptr()
}

//the fake runtime's calls under cuda_runtime.addresses' names
#[pyfunction]
pub fn fake_cuda_addresses() -> HashMap<String, usize> {
    HashMap::from([
        ("cudaSetDevice".to_string(), set_device as *const () as usize),
        ("cudaMemcpyAsync".to_string(), memcpy as *const () as usize),
        ("cudaGraphLaunch".to_string(), graph_launch as *const () as usize),
        ("cudaEventRecord".to_string(), event_record as *const () as usize),
        ("cudaEventSynchronize".to_string(), event_synchronize as *const () as usize),
        ("cudaStreamSynchronize".to_string(), stream_synchronize as *const () as usize),
        ("cudaGetErrorString".to_string(), error_string as *const () as usize),
    ])
}

//a fake graph of `size` boards reading lean rows at `rows` and writing their answers at `answers` (the caller's buffers,
//alive while the loop runs), a batch taking fixed + per_board seconds a board: its exec's handle
#[pyfunction]
pub fn fake_cuda_graph(size: usize, rows: usize, answers: usize, fixed_seconds: f64, per_board_seconds: f64) -> PyResult<usize> {
    if size == 0 || rows == 0 || answers == 0 {
        return Err(PyValueError::new_err("a fake graph needs boards and buffers"));
    }
    let mut graphs = GRAPHS.write().expect("the fake graphs");
    graphs.push(Graph {
        size,
        rows,
        answers,
        fixed: Duration::from_secs_f64(fixed_seconds),
        per_board: Duration::from_secs_f64(per_board_seconds),
    });
    Ok(graphs.len())
}

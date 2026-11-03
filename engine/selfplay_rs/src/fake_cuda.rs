//fake gpus for timing Searches' round loop (rounds.rs) on more gpus than the machine has: the cuda runtime calls the loop
//makes (cuda_runtime.addresses') and graphs that answer a lean round from a hash of each board. A launch keeps its stream
//busy for fixed + per_board * size after its device's (the thread's set_device) other batches: a gpu runs one at a time.
//Streams, events and devices are numbers taken modulo STREAMS; the waits spin, as a blocking wait holds the thread.
use crate::rules::TOKENS_PER_BOARD;
use crate::{LEAN_COLUMNS, LEAN_ROW};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::c_char;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{OnceLock, RwLock};
use std::time::{Duration, Instant};

const STREAMS: usize = 256;

//rows and answers are the caller's buffers' addresses; busy is a batch's nanoseconds
struct Graph {
    size: usize,
    rows: usize,
    answers: usize,
    busy: u64,
}

static GRAPHS: RwLock<Vec<Graph>> = RwLock::new(Vec::new());
//nanoseconds since START when each stream's and each device's batches are done, and each event's stream
static READY: [AtomicU64; STREAMS] = [const { AtomicU64::new(0) }; STREAMS];
static DEVICE_READY: [AtomicU64; STREAMS] = [const { AtomicU64::new(0) }; STREAMS];
static EVENT_STREAMS: [AtomicU64; STREAMS] = [const { AtomicU64::new(0) }; STREAMS];

thread_local!(static DEVICE: Cell<usize> = const { Cell::new(0) });
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

fn seed(tokens: &[u8]) -> u64 {
    tokens.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, &token| (hash ^ u64::from(token)).wrapping_mul(0x0000_0100_0000_01b3))
}

//a hash's top 24 bits as a fraction of 1
fn unit(hash: u64) -> f32 {
    (hash >> 40) as f32 / (1u64 << 24) as f32
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
    //the last column is the expected score's, which also pads the legal moves' columns: only those before it get logits
    for (row, answer) in rows.chunks_exact(LEAN_ROW).zip(answers.chunks_exact_mut(LEAN_COLUMNS)) {
        let board = seed(&row[..TOKENS_PER_BOARD]);
        let column = |slot: usize| u16::from_le_bytes([row[TOKENS_PER_BOARD + 2 * slot], row[TOKENS_PER_BOARD + 2 * slot + 1]]);
        let score_column = column(LEAN_COLUMNS - 1);
        for slot in (0..LEAN_COLUMNS - 1).take_while(|&slot| column(slot) != score_column) {
            let mixed = (board ^ u64::from(column(slot)).wrapping_mul(0x9e37_79b9_7f4a_7c15)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            answer[slot] = unit(mixed) * 4.0 - 2.0;
        }
        answer[LEAN_COLUMNS - 1] = 0.35 + 0.3 * unit(board);
    }
    //a device's one thread launches its batches, so nothing else moves its ready time meanwhile
    let device = DEVICE.with(Cell::get);
    let done = DEVICE_READY[device].load(Ordering::Acquire).max(nanos()) + graph.busy;
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

#[pyfunction]
pub fn fake_cuda_addresses() -> HashMap<String, usize> {
    [
        ("cudaSetDevice", set_device as *const () as usize),
        ("cudaMemcpyAsync", memcpy as *const () as usize),
        ("cudaGraphLaunch", graph_launch as *const () as usize),
        ("cudaEventRecord", event_record as *const () as usize),
        ("cudaEventSynchronize", event_synchronize as *const () as usize),
        ("cudaStreamSynchronize", stream_synchronize as *const () as usize),
        ("cudaGetErrorString", error_string as *const () as usize),
    ]
    .into_iter()
    .map(|(name, address)| (name.to_string(), address))
    .collect()
}

//a fake graph of `size` boards reading lean rows at `rows` and writing their answers at `answers`, both kept alive by the
//caller while the loop runs: its exec's handle
#[pyfunction]
pub fn fake_cuda_graph(size: usize, rows: usize, answers: usize, fixed_seconds: f64, per_board_seconds: f64) -> PyResult<usize> {
    if size == 0 || rows == 0 || answers == 0 {
        return Err(PyValueError::new_err("a fake graph needs boards and buffers"));
    }
    let busy = Duration::from_secs_f64(fixed_seconds) + Duration::from_secs_f64(per_board_seconds) * size as u32;
    let mut graphs = GRAPHS.write().expect("the fake graphs");
    graphs.push(Graph { size, rows, answers, busy: busy.as_nanos() as u64 });
    Ok(graphs.len())
}

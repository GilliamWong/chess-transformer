import ctypes
import gc
import io
import itertools
import json
import os
import queue
import runpy
import subprocess
import sys
import threading
import time
from collections import Counter
from contextlib import suppress
from pathlib import Path

import chess
import chess.engine
import numpy as np
import pytest

import selfplay_rs
import torch
from torch import nn
from model.encoding import TOKENS_PER_BOARD, encode_move_to_id
from fake_models import BoardModel, CountingNetwork, EvenModel, UniformModel
from engine.inference import GraphedBatches, ImmediateBatches, cpu_batches, small_graph_sizes
from model.model import ChessTransformer
from engine.selfplay import ANSWER_WIDTH, rust_search_settings
from model.checkpoint import save_checkpoint
from engine.tablebase import SYZYGY_DIR, TABLEBASE_PIECES
from engine import engine_search
from engine.engine_search import EngineSearch, SearchReport, engine_search_settings, parse_changes
from engine import uci
from engine.uci import (CACHE_POSITION_BYTES, FITTED_CACHE_POSITIONS, MOST_EXTENSION, MOVE_OVERHEAD_SECONDS, RESERVE_SECONDS, Clock, Options,
                        fitted_memory, memory_within_limits)

REPOSITORY = Path(__file__).parent.parent
FREE_QUEEN = "4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1"
MATE_IN_ONE = "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1"
MATE_IN_TWO = "k7/8/2K5/8/8/8/8/7R w - - 0 1"
#both mate in two
MATES_IN_TWO = ("c6b6", "c6c7")
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")

@pytest.fixture(autouse=True)
def without_engine_settings(monkeypatch):
    monkeypatch.setattr(engine_search, "ENGINE_SETTINGS", {})

@pytest.fixture
def model_path(tmp_path) -> Path:
    settings = {"d_model": 32, "num_layers": 1, "n_heads": 2, "mlp_hidden": 64, "value_bins": 16}
    save_checkpoint(ChessTransformer(**settings), settings, tmp_path / "model.pt")
    return tmp_path / "model.pt"

@pytest.fixture
def middlegame(carlsen_games) -> chess.Board:
    return carlsen_position(carlsen_games, 0, 16)

def carlsen_position(games, game: int, plies: int) -> chess.Board:
    board = games[game].board()
    for move in list(games[game].mainline_moves())[:plies]:
        board.push(move)
    return board

def leaf_boards(leaves: bytes) -> np.ndarray:
    return np.frombuffer(bytearray(leaves), dtype=np.uint8).reshape(-1, TOKENS_PER_BOARD)

#random logits and a score of a quarter, a half or three quarters: sums of quarters are exact in any order
def answers_by_board(boards: np.ndarray) -> np.ndarray:
    answers = np.zeros((len(boards), ANSWER_WIDTH), dtype=np.float32)
    for row, board in enumerate(boards):
        random = np.random.default_rng(list(board))
        answers[row, :-1] = random.normal(scale=2.0, size=ANSWER_WIDTH - 1)
        answers[row, -1] = random.integers(1, 4) / 4
    return answers

def answers_of_0_6(boards: np.ndarray) -> np.ndarray:
    answers = np.zeros((len(boards), ANSWER_WIDTH), dtype=np.float32)
    answers[:, -1] = 0.6
    return answers


class WrappedNetwork:
    def __init__(self, network):
        self.network = network

    def launch(self, boards: np.ndarray):
        return self.network.launch(boards)

    def collect(self, launched):
        return self.network.collect(launched)


class Laned(WrappedNetwork):
    def __init__(self, network, lanes: list):
        super().__init__(network)
        self.lanes = lanes


class RootVisits:
    def __init__(self, visits: int):
        self.visits = visits

    def root_visits(self, game):
        return self.visits, 0, 0

def test_a_round_gathers_different_leaves():
    evaluate = cpu_batches(UniformModel()).evaluate_boards
    searches = selfplay_rs.Searches(**rust_search_settings(False))
    searches.start(0, chess.STARTING_FEN, [], 1000)

    root = leaf_boards(searches.descend(16, 16))
    searches.finish(evaluate(root))
    first, second = leaf_boards(searches.descend(8, 16)), leaf_boards(searches.descend(8, 16))
    searches.finish(evaluate(first))
    searches.finish(evaluate(second))

    assert len(root) == 1
    assert len(first) == len(second) == 8
    assert len({board.tobytes() for board in np.concatenate([first, second])}) == 16

def test_rounds_are_finished_by_name_in_any_order():
    found = []
    for order in ([0, 1, 2, 3], [2, 0, 3, 1]):
        searches = selfplay_rs.Searches(**rust_search_settings(False))
        searches.start(0, chess.STARTING_FEN, [], 300)
        while searches.root_visits(0)[0] < 40:
            searches.finish(answers_by_board(leaf_boards(searches.descend())))
        out = {name: leaf_boards(searches.descend(8, 64, 24, round=name)) for name in range(4)}
        with pytest.raises(ValueError):
            searches.descend(8, 64, 24, round=1)
        for name in order:
            searches.finish(answers_by_board(out[name]), round=name)
        with pytest.raises(RuntimeError):
            searches.finish(answers_by_board(out[0]), round=0)
        following = leaf_boards(searches.descend(64, 64, 192))
        reports = searches.finish(answers_by_board(following))
        while not reports:
            reports = searches.finish(answers_by_board(leaf_boards(searches.descend())))
        found.append(([len(leaves) for leaves in out.values()], following.tobytes(), reports[0]))

    assert found[0] == found[1] and found[0][0] == [8, 8, 8, 8] and found[0][2][4] == 300

@pytest.mark.parametrize("dtype", [np.uint8, np.int64])
def test_descend_into_writes_descends_boards(dtype):
    evaluate = cpu_batches(BoardModel()).evaluate_boards
    plain, into = (selfplay_rs.Searches(**rust_search_settings(False), cache_positions=1000) for _ in range(2))
    for searches in (plain, into):
        searches.start(0, chess.STARTING_FEN, [], 600)
    boards, reports, rounds = np.full((16, TOKENS_PER_BOARD), 99, dtype=dtype), [], 0
    while not reports:
        expected = leaf_boards(plain.descend(16, 16, 48))
        written, root = into.descend_into(boards, 16, 16, 48, root_of=0)
        assert written == len(expected) and (boards[:written] == expected).all() and root == into.root_visits(0) == plain.root_visits(0)
        answers = evaluate(expected) if written else np.zeros((0, ANSWER_WIDTH), dtype=np.float32)
        reports, root = into.finish_with_root(answers, root_of=0)
        assert reports == plain.finish(answers)
        assert root is None if reports else root == into.root_visits(0)
        rounds += 1
    assert rounds > 10 and reports[0][4] == 600

def test_lean_rounds_search_as_whole_rows():
    evaluate = cpu_batches(BoardModel()).evaluate_boards
    whole, lean = (selfplay_rs.Searches(**rust_search_settings(False), cache_positions=1000) for _ in range(2))
    for searches in (whole, lean):
        searches.start(0, "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", [], 800)
    rows, reports, rounds = np.full((16, selfplay_rs.LEAN_ROW), 7, dtype=np.uint8), [], 0
    while not reports:
        boards = leaf_boards(whole.descend(16, 16, 48))
        written, _ = lean.descend_into(rows, 16, 16, 48, legal_columns=True)
        assert written == len(boards) and (rows[:written, :TOKENS_PER_BOARD] == boards).all()
        columns = rows[:written, TOKENS_PER_BOARD:].copy().view("<i2").astype(np.int64)
        assert (columns[:, -1] == ANSWER_WIDTH - 1).all() and (columns < ANSWER_WIDTH).all()
        answers = evaluate(boards) if written else np.zeros((0, ANSWER_WIDTH), dtype=np.float32)
        reports = whole.finish(answers)
        assert lean.finish(np.take_along_axis(answers, columns, axis=1)) == reports
        assert reports or lean.root_visits(0) == whole.root_visits(0)
        rounds += 1
    assert rounds > 10 and reports[0][4] == 800

#a table file of the right size with nothing in it: the tables cover the position after the capture and cannot score it
def test_tables_that_fail_to_probe_are_searched_without_them(tmp_path):
    (tmp_path / "KQvK.rtbw").write_bytes(bytes(16))
    board, evaluate = chess.Board("8/8/8/4k3/8/2p5/8/3QK3 w - - 0 1"), cpu_batches(BoardModel()).evaluate_boards
    for concurrent_descents in (False, True):
        found = []
        for settings in (rust_search_settings(True, 3, str(tmp_path)), rust_search_settings(False)):
            searches = selfplay_rs.Searches(**settings, cache_positions=1000, concurrent_descents=concurrent_descents)
            searches.start(0, board.fen(), [], 800)
            reports = []
            while not reports:
                boards = leaf_boards(searches.descend(8, 8, 24))
                reports = searches.finish(evaluate(boards) if len(boards) else np.zeros((0, ANSWER_WIDTH), dtype=np.float32))
            found.append(reports)
        assert found[0] == found[1] and found[0][0][4] == 800 and len(searches) == 0

#a NaN logit or value would stay in every node it is backed up through, the root's among them
@pytest.mark.parametrize("concurrent_descents", [False, True])
def test_answers_that_are_no_numbers_do_not_spoil_a_search(concurrent_descents):
    searches = selfplay_rs.Searches(**rust_search_settings(False), cache_positions=1000, concurrent_descents=concurrent_descents)
    searches.start(0, chess.STARTING_FEN, [], 400)
    reports = []
    while not reports:
        answers = answers_by_board(leaf_boards(searches.descend(8, 8, 24)))
        if len(answers) > 1:
            answers[::2, -1] = np.nan
            answers[1::3, :-1] = np.inf
        reports = searches.finish(answers)
    _, line, score, root_moves, visits = reports[0]
    assert 0 <= score <= 1 and visits == 400 and sum(move_visits for _, move_visits in root_moves) == 399

def test_a_failed_search_settles_its_network():
    class Failing(ImmediateBatches):
        settled = 0

        def collect(self, launched):
            raise RuntimeError("the gpu went away")

        def settle(self):
            self.settled += 1
    network = Failing(cpu_batches(BoardModel()).evaluate_boards)
    with pytest.raises(RuntimeError, match="went away"):
        EngineSearch(network, 16).search(chess.Board(), simulations=200)
    assert network.settled == 1

@gpu
@pytest.mark.parametrize("blocking_wait, leaves, settings", [(False, 16, {}), (True, 16, {}), (False, 1, {}),
                                                             (False, 16, {"smart_pruning": False}),
                                                             (True, 16, {"pin_threads": True})])
def test_rust_rounds_search_as_the_python_loop(blocking_wait, leaves, settings):
    board, reports = chess.Board(), {}
    for rust_rounds in (False, True):
        search = EngineSearch(lean_network(blocking_wait), leaves, rust_rounds=rust_rounds, **settings)
        found = []
        for move in ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]:
            found.append(search.search(board, simulations=600 if leaves > 1 else 60))
            board.push_uci(move)
        board = chess.Board()
        reports[rust_rounds] = found
    assert reports[True] == reports[False]
    assert search.rust_counts[0] > 20 and search.rust_counts[1] >= search.rust_counts[0]

def test_rust_rounds_run_on_lean_lanes_only():
    lean, whole = Laned(cpu_batches(BoardModel()), [0, 0]), cpu_batches(BoardModel())
    lean.lean = True
    assert EngineSearch([lean, lean], 16).rust_rounds and EngineSearch(lean, 16, rust_rounds=True).rust_rounds
    assert not EngineSearch(lean, 16).rust_rounds and not EngineSearch(whole, 16).rust_rounds
    assert not EngineSearch([lean, whole], 16).rust_rounds and not EngineSearch([lean, lean], 16, rust_rounds=False).rust_rounds
    for networks in ([lean, whole], whole):
        with pytest.raises(ValueError, match="lean lanes"):
            EngineSearch(networks, 16, rust_rounds=True)

class Lane:
    device = 0

@pytest.mark.parametrize("capability, rust", [((8, 0), True), ((12, 0), False), ((8, 6), False)])
def test_one_network_rounds_in_rust_on_the_gpus_listed(monkeypatch, capability, rust):
    monkeypatch.setattr(engine_search.torch.cuda, "get_device_capability", lambda device=None: capability)
    lean, whole = Laned(cpu_batches(BoardModel()), [Lane(), Lane()]), Laned(cpu_batches(BoardModel()), [Lane(), Lane()])
    lean.lean = True
    assert EngineSearch(lean, 16).rust_rounds == rust
    assert not EngineSearch(lean, 16, rust_rounds=False).rust_rounds and not EngineSearch(whole, 16).rust_rounds

@gpu
def test_rust_rounds_drive_every_network():
    networks = [lean_network(True), lean_network(True)]
    search = EngineSearch(networks, 16, smart_pruning=False, rust_rounds=True, concurrent_descents=False)
    assert search.search(chess.Board(), simulations=3000).simulations == 3000
    assert search.rust_counts[0] > 40 and all(not lane.pending for network in networks for lane in network.lanes)

@gpu
@pytest.mark.parametrize("concurrent_descents", [False, None])
def test_the_rust_rounds_threads_are_pinned(concurrent_descents):
    networks = [lean_network(True), lean_network(True)]
    search = EngineSearch(networks, 16, smart_pruning=False, concurrent_descents=concurrent_descents)
    before, seen = pinned_threads(), []
    search.search(chess.Board(), seconds=1.5, info=lambda report: seen.append(pinned_threads() - before))
    assert search.rust_rounds and search.concurrent_descents == (concurrent_descents is None)
    assert seen and seen[0] == Counter(search.network_cpus) and search.network_cpus == engine_search.thread_cpus(2) != [], seen
    handles = networks[0].rust_network() | {"cpu": 100_000}
    networks[0].rust_network = lambda: handles
    with pytest.raises(ValueError, match="may run on"):
        EngineSearch(networks[0], 16, rust_rounds=True).search(chess.Board(), simulations=600)

@gpu
def test_a_cuda_error_in_the_rust_rounds_is_raised():
    network = lean_network()
    handles = network.rust_network()
    handles["lanes"] = [{**lane, "device": 99} for lane in handles["lanes"]]
    network.rust_network = lambda: handles
    search = EngineSearch(network, 16, rust_rounds=True)
    with pytest.raises(RuntimeError, match="cuda error"):
        search.search(chess.Board(), simulations=600)

#a thread that fails ends the search with its error, and the search and its rounds are gone from Searches: the next can start
@gpu
@pytest.mark.parametrize("concurrent_descents", [False, True])
def test_a_thread_that_panics_or_fails_ends_the_search_with_an_error(concurrent_descents):
    network = lean_network()
    search = EngineSearch(network, 16, rust_rounds=True, concurrent_descents=concurrent_descents)
    settings = engine_search_settings(False)
    #a legal move without an id panics in the thread (the Rust way of a bug), a device that is none fails it (a cuda error)
    broken = selfplay_rs.Searches(**{**settings, "move_id_table": [-1] * len(settings["move_id_table"])}, tree_reuse=True,
                                  concurrent_descents=concurrent_descents)
    handles = network.rust_network()
    for searches, rust_network, message in ((broken, handles, "panicked"),
                                            (search.searches, {**handles, "lanes": [{**lane, "device": 99} for lane in handles["lanes"]]}, "cuda error")):
        search.searches = searches
        network.rust_network = lambda handles=rust_network: handles
        errors = []

        def run():
            try:
                search.search(chess.Board(), simulations=600)
            except BaseException as error:
                errors.append(error)
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(30)
        assert not worker.is_alive(), "the search never ended"
        assert len(errors) == 1 and isinstance(errors[0], RuntimeError) and message in str(errors[0]), errors
        assert len(searches) == 0
        searches.start(0, chess.STARTING_FEN, [], 600)
        assert len(searches) == 1

#a thread that finds the deadline due must not stop a fresh tree before the first leaf of another thread is out
@gpu
@pytest.mark.parametrize("rust_rounds", [False, True])
def test_a_deadline_already_due_on_a_fresh_tree_still_gives_a_move(rust_rounds):
    search = EngineSearch([lean_network(True) for _ in range(4)], 16, rust_rounds=rust_rounds, concurrent_descents=True)
    board = chess.Board()
    for _ in range(1500):
        search.new_game()
        report = search.search(board, seconds=0.0)
        assert report.simulations >= 1 and report.line[0] in board.legal_moves

#the extension of a score that fell is judged on the root's answer: before it, the score is no number
@gpu
@pytest.mark.parametrize("concurrent_descents", [False, True])
def test_an_extension_on_a_fresh_tree_does_not_keep_a_due_search_running(concurrent_descents):
    search = EngineSearch(lean_network(True), 16, rust_rounds=True, concurrent_descents=concurrent_descents)
    stop = threading.Event()
    timer = threading.Timer(4, stop.set)
    timer.start()
    try:
        report = search.search(chess.Board(), seconds=0.0, extension=(0, 0.1), stop=stop)
    finally:
        timer.cancel()
    assert not stop.is_set() and report.simulations >= 1

def current_cpu() -> int:
    if sys.platform == "win32":
        return ctypes.windll.kernel32.GetCurrentProcessorNumber()
    return ctypes.CDLL(None).sched_getcpu()


class ThreadEntry(ctypes.Structure):
    _fields_ = [("size", ctypes.c_uint32), ("usage", ctypes.c_uint32), ("thread", ctypes.c_uint32), ("process", ctypes.c_uint32),
                ("base_priority", ctypes.c_int32), ("delta_priority", ctypes.c_int32), ("flags", ctypes.c_uint32)]


class GroupAffinity(ctypes.Structure):
    _fields_ = [("mask", ctypes.c_size_t), ("group", ctypes.c_uint16), ("reserved", ctypes.c_uint16 * 3)]

def pinned_threads() -> Counter:
    pinned = Counter()
    if sys.platform != "win32":
        for status in Path("/proc/self/task").glob("*/status"):
            allowed = next(line.split(":")[1].strip() for line in status.read_text().splitlines() if line.startswith("Cpus_allowed_list"))
            if allowed.isdigit():
                pinned[int(allowed)] += 1
        return pinned
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateToolhelp32Snapshot.restype = kernel32.OpenThread.restype = ctypes.c_void_p
    kernel32.Thread32First.argtypes = kernel32.Thread32Next.argtypes = [ctypes.c_void_p, ctypes.POINTER(ThreadEntry)]
    kernel32.GetThreadGroupAffinity.argtypes = [ctypes.c_void_p, ctypes.POINTER(GroupAffinity)]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    snapshot, entry = kernel32.CreateToolhelp32Snapshot(0x4, 0), ThreadEntry() #TH32CS_SNAPTHREAD
    entry.size = ctypes.sizeof(ThreadEntry)
    found = kernel32.Thread32First(snapshot, ctypes.byref(entry))
    while found:
        if entry.process == os.getpid() and (handle := kernel32.OpenThread(0x0800, False, entry.thread)): #THREAD_QUERY_LIMITED_INFORMATION
            affinity = GroupAffinity()
            if kernel32.GetThreadGroupAffinity(handle, ctypes.byref(affinity)) and affinity.mask & (affinity.mask - 1) == 0:
                pinned[affinity.mask.bit_length() - 1] += 1
            kernel32.CloseHandle(handle)
        found = kernel32.Thread32Next(snapshot, ctypes.byref(entry))
    kernel32.CloseHandle(snapshot)
    return pinned

#topologies are (L3 sizes with their cpus, smt sibling groups)
def test_the_threads_cpus(monkeypatch):
    windows = ([(96 << 20, list(range(16))), (32 << 20, list(range(16, 32)))], [[cpu, cpu + 1] for cpu in range(0, 32, 2)])
    assert engine_search.thread_cpus(1, windows) == [0] and engine_search.thread_cpus(8, windows) == list(range(0, 16, 2))
    assert engine_search.thread_cpus(11, windows) == [*range(0, 16, 2), 16, 18, 20]
    assert engine_search.thread_cpus(18, windows) == [*range(0, 32, 2), 1, 3]
    flipped = ([(32 << 20, list(range(16))), (96 << 20, list(range(16, 32)))], windows[1])
    assert engine_search.thread_cpus(3, flipped) == [16, 18, 20] and engine_search.thread_cpus(9, flipped) == [*range(16, 32, 2), 0]
    linux = ([(96 << 20, [*range(8), *range(16, 24)]), (32 << 20, [*range(8, 16), *range(24, 32)])], [[core, core + 16] for core in range(16)])
    assert engine_search.thread_cpus(8, linux) == list(range(8)) and engine_search.thread_cpus(18, linux) == [*range(16), 16, 17]
    epyc = ([(32 << 20, [core, core + 32]) for core in range(32)], [[core, core + 32] for core in range(32)])
    assert engine_search.thread_cpus(8, epyc) == list(range(8)) and engine_search.thread_cpus(34, epyc) == [*range(32), 32, 33]
    socket = ([(32 << 20, [core, core + 32]) for core in range(16)], [[core, core + 32] for core in range(16)])
    assert engine_search.thread_cpus(16, socket) == list(range(16)) and engine_search.thread_cpus(17, socket) == [*range(16), 32]
    uneven = ([(64 << 20, [0, 1, 2, 3]), (32 << 20, list(range(4, 12)))], [[cpu, cpu + 1] for cpu in range(0, 12, 2)])
    assert engine_search.thread_cpus(2, uneven) == [0, 2] and engine_search.thread_cpus(3, uneven) == [4, 6, 8]
    assert engine_search.thread_cpus(5, uneven) == [0, 2, 4, 6, 8]
    assert engine_search.thread_cpus(0, windows) == [] and engine_search.thread_cpus(33, windows) == []
    monkeypatch.setattr(engine_search.selfplay_rs, "cpu_topology", lambda: None)
    assert engine_search.thread_cpus(4) == []


class CpuRecording(WrappedNetwork):
    def __init__(self, network):
        super().__init__(network)
        self.cpus = []

    def launch(self, boards: np.ndarray):
        self.cpus.append(current_cpu())
        return super().launch(boards)

def test_pinning_changes_no_search():
    board, reports = chess.Board(), {}
    for pinned in (False, True):
        network = CpuRecording(cpu_batches(BoardModel()))
        search = EngineSearch(network, 16, pin_threads=pinned)
        found = []
        for move in ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]:
            found.append(search.search(board, simulations=600))
            board.push_uci(move)
            if len(found) == 1:
                #torch's openmp unpins the thread at its first parallel region (the cpu model's); the next search's pin undoes that
                network.cpus.clear()
        board = chess.Board()
        reports[pinned] = found
    cpus = engine_search.thread_cpus(1)
    assert reports[True] == reports[False]
    assert search.network_cpus == cpus != [] and set(network.cpus) == set(cpus), (search.network_cpus, set(network.cpus))

def test_the_python_loops_threads_are_pinned():
    for concurrent_descents in (False, None):
        #numpy's answers: torch's openmp would unpin the threads
        networks = [CpuRecording(ImmediateBatches(answers_by_board)) for _ in range(3)]
        search = EngineSearch(networks, 16, smart_pruning=False, concurrent_descents=concurrent_descents)
        search.search(chess.Board(), simulations=1500)
        assert search.concurrent_descents == (concurrent_descents is None) and search.network_cpus == engine_search.thread_cpus(3) != []
        assert [set(network.cpus) for network in networks] == [{cpu} for cpu in search.network_cpus]

def test_threads_are_pinned_by_default_on_several_networks():
    one, three = cpu_batches(BoardModel()), [cpu_batches(BoardModel()) for _ in range(3)]
    assert EngineSearch(three, 16).network_cpus == engine_search.thread_cpus(3) and len(engine_search.thread_cpus(3)) == 3
    assert EngineSearch(one, 16).network_cpus == EngineSearch(three, 16, pin_threads=False).network_cpus == []
    assert EngineSearch(one, 16, pin_threads=True).network_cpus == engine_search.thread_cpus(1)

@pytest.mark.parametrize("capability, lean", [((12, 0), True), ((8, 0), True), ((8, 6), False), ((9, 0), False)])
def test_lean_lanes_by_capability(monkeypatch, capability, lean):
    monkeypatch.setattr(engine_search.torch.cuda, "get_device_capability", lambda device=None: capability)
    assert engine_search.lean_lanes() == lean

def test_descend_into_wants_room_to_write():
    searches = selfplay_rs.Searches(**rust_search_settings(False))
    searches.start(0, chess.STARTING_FEN, [], 600)
    with pytest.raises(ValueError):
        searches.descend_into(np.zeros((8, TOKENS_PER_BOARD), dtype=np.uint8), 16, 16, 48)
    unwritable = np.zeros((16, TOKENS_PER_BOARD), dtype=np.uint8)
    unwritable.flags.writeable = False
    with pytest.raises((ValueError, BufferError)):
        searches.descend_into(unwritable, 16, 16, 48)
    assert searches.descend_into(np.zeros((16, TOKENS_PER_BOARD), dtype=np.uint8), 16, 16, 48, root_of=0) == (1, (0, 0, 0))

def test_searches_let_other_threads_run():
    searches = selfplay_rs.Searches(**rust_search_settings(False))
    for search_id in range(400):
        searches.start(search_id, chess.STARTING_FEN, [], 1000)
    searches.finish(answers_of_0_6(leaf_boards(searches.descend())))
    ticks, counting = [0], threading.Event()

    def count():
        counting.set()
        while counting.is_set():
            ticks[0] += 1
    counter = threading.Thread(target=count)
    counter.start()
    counting.wait()
    before, started = ticks[0], time.perf_counter()
    searches.descend(64, 64, 192)
    seconds, during = time.perf_counter() - started, ticks[0] - before
    counting.clear()
    counter.join()

    assert during > 1000, (seconds, during)

def test_the_cache_answers_transpositions_as_the_network_would():
    evaluate = cpu_batches(BoardModel()).evaluate_boards
    found = {}
    for cache_positions in (0, 100_000):
        searches = selfplay_rs.Searches(**rust_search_settings(False), cache_positions=cache_positions)
        searches.start(0, chess.STARTING_FEN, [], 1000)
        evaluated, reports = 0, []
        while not reports:
            boards = leaf_boards(searches.descend())
            evaluated += len(boards)
            reports = searches.finish(evaluate(boards))
        found[cache_positions] = (reports[0], evaluated)

    (without, evaluated_without), (with_cache, evaluated_with_cache) = found[0], found[100_000]
    assert with_cache == without
    assert evaluated_with_cache < 0.9 * evaluated_without

def test_one_leaf_a_round_is_the_exact_search_at_every_move(monkeypatch, carlsen_games):
    monkeypatch.setattr(engine_search, "ENGINE_SETTINGS", {"c_puct": 0.5, "fpu_reduction": 0.25})
    evaluate = cpu_batches(BoardModel()).evaluate_boards
    search = EngineSearch(cpu_batches(BoardModel()), 1)
    board = carlsen_position(carlsen_games, 0, 20)
    for _ in range(8):
        report = search.search(board, simulations=60)

        exact = selfplay_rs.Searches(**rust_search_settings(False))
        exact.start(0, board.root().fen(), [played.uci() for played in board.move_stack], 60)
        reports = []
        while not reports:
            reports = exact.finish(evaluate(leaf_boards(exact.descend())))
        _, line, score, root_moves, visits = reports[0]
        assert ([played.uci() for played in report.line], report.score, report.simulations) == (line, score, visits)
        assert [(played.uci(), move_visits) for played, move_visits in report.root_moves] == root_moves[:len(report.root_moves)]
        board.push(report.line[0])

def test_the_engines_settings(monkeypatch):
    monkeypatch.setattr(engine_search, "ENGINE_SETTINGS", {"c_puct": 0.5, "fpu_reduction": 0.25})
    assert engine_search_settings(False) == {**rust_search_settings(False), "fpu_reduction": 0.25, "c_puct": 0.5}
    assert engine_search_settings(False, {"c_puct": 0.7})["c_puct"] == 0.7
    assert engine_search_settings(False, own=False) == rust_search_settings(False)

#the value goes into the search and the cache sharpened (its logit times value_sharpness), so a cached position is not sharpened again
@pytest.mark.parametrize("sharpness, expected", [(1.0, 0.6), (3.0, 1 / (1 + (0.4 / 0.6) ** 3))])
def test_the_value_is_sharpened_once(sharpness, expected):
    searches = selfplay_rs.Searches(**rust_search_settings(False), cache_positions=1000, value_sharpness=sharpness)
    scores = []
    for _ in range(2):
        searches.start(0, chess.STARTING_FEN, [], 1)
        scores.append(searches.finish(answers_of_0_6(leaf_boards(searches.descend())))[0][2])
    assert scores == pytest.approx([expected, expected])

def test_the_engine_reports_the_score_unsharpened():
    search = EngineSearch(ImmediateBatches(answers_of_0_6), 16, settings={"value_sharpness": 3.0})
    assert search.search(chess.Board(), simulations=1).score == pytest.approx(0.6)

@pytest.mark.parametrize("simulations", [1, 37, 300])
def test_a_search_of_many_leaves_a_round_stops_at_its_simulations(carlsen_games, simulations):
    board = carlsen_position(carlsen_games, 1, 10)
    search = EngineSearch(cpu_batches(BoardModel()), 16, smart_pruning=False)

    report = search.search(board, simulations=simulations)

    assert report.simulations == simulations and report.line[0] in board.legal_moves


class SizesNetwork(WrappedNetwork):
    def __init__(self, network):
        super().__init__(network)
        self.sizes = []

    def launch(self, boards: np.ndarray):
        self.sizes.append(len(boards))
        return super().launch(boards)

def test_a_rounds_leaves_grow_with_the_search(carlsen_games):
    board = carlsen_position(carlsen_games, 1, 10)
    growing, fixed = SizesNetwork(cpu_batches(BoardModel())), SizesNetwork(cpu_batches(BoardModel()))
    growing_search = EngineSearch(growing, 64, smart_pruning=False, grow_leaves=16)
    fixed_search = EngineSearch(fixed, 64, smart_pruning=False, grow_leaves=0)

    report = growing_search.search(board, simulations=3000)
    fixed_search.search(board, simulations=3000)

    assert report.simulations == 3000
    assert growing.sizes[:4] == [1, 1, 1, 1] and max(growing.sizes) == 64
    assert fixed.sizes[1] > 16 and len(growing.sizes) > len(fixed.sizes)

@pytest.mark.parametrize("visits, simulations, expected", [(0, 1000, (1, 1)), (160, 1000, (10, 1)), (1024, None, (64, 2)), (995, 1000, (1, 1))])
def test_the_leaves_of_a_round(visits, simulations, expected):
    search = EngineSearch(cpu_batches(BoardModel()), 64, grow_leaves=16)
    search.searches = RootVisits(visits)

    assert search.round_leaves(simulations) == expected

#a free queen, which the material count finds, and mates in one, which a flat eval finds only by searching
@pytest.mark.parametrize("model, fen, best", [
    (BoardModel(), FREE_QUEEN, "e4d5"),
    (UniformModel(), MATE_IN_ONE, "e1e8"),
    (UniformModel(), "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "h5f7"),
    (UniformModel(), "3r2k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1", "d8d1"),
])
@pytest.mark.parametrize("leaves", [1, 16, 64])
def test_many_leaves_a_round_find_what_one_finds(model, fen, best, leaves):
    report = EngineSearch(cpu_batches(model), leaves).search(chess.Board(fen), simulations=400)

    assert report.line[0].uci() == best

def test_a_timed_search_ends_by_its_deadline():
    search = EngineSearch(cpu_batches(BoardModel()), 16)

    started = time.perf_counter()
    report = search.search(chess.Board(), seconds=0.3)

    assert report.simulations > 16 and time.perf_counter() - started < 0.3 + 0.1

def test_rounds_run_at_once_only_on_lanes_of_their_own():
    assert EngineSearch(Laned(cpu_batches(BoardModel()), [0, 0]), 4).rounds_at_once
    assert not EngineSearch(Laned(cpu_batches(BoardModel()), [0, 0]), 1).rounds_at_once
    assert not EngineSearch(Laned(cpu_batches(BoardModel()), [0]), 4).rounds_at_once
    assert not EngineSearch(cpu_batches(BoardModel()), 4).rounds_at_once


#networks that answer after `seconds` without the gil, as a gpu's collect does, counting the rounds out at once; each raises
#at the collect of its failing_collect'th round (0: never)
class SlowNetworks:
    def __init__(self, count: int, seconds: float, model: nn.Module, failing_collect: int = 0):
        self.lock = threading.Lock()
        self.out = self.most_out = 0
        self.networks = [SlowNetwork(self, seconds, model, failing_collect) for _ in range(count)]

class SlowNetwork:
    def __init__(self, group: SlowNetworks, seconds: float, model: nn.Module, failing_collect: int):
        self.group, self.seconds, self.failing_collect = group, seconds, failing_collect
        self.network = cpu_batches(model)
        self.rounds = 0

    def launch(self, boards: np.ndarray):
        with self.group.lock:
            self.group.out += 1
            self.group.most_out = max(self.group.most_out, self.group.out)
        self.rounds += 1
        return self.network.launch(boards), time.perf_counter()

    def collect(self, launched):
        answers, launched_at = launched
        time.sleep(max(0.0, launched_at + self.seconds - time.perf_counter()))
        with self.group.lock:
            self.group.out -= 1
        if self.rounds == self.failing_collect:
            raise RuntimeError("the gpu fell over")
        return self.network.collect(answers)

#the simulations past the search's that a search on several networks may end at: one a thread after the first in concurrent
#descents (a check and a leaf are not one step, SharedTree::gather)
def overrun(concurrent_descents: bool | None, networks: int = 3) -> int:
    return 0 if concurrent_descents is False else networks - 1

@pytest.mark.parametrize("concurrent_descents", [False, None])
def test_a_search_on_several_networks_keeps_them_all_busy(concurrent_descents):
    group = SlowNetworks(3, 0.003, BoardModel())
    search = EngineSearch(group.networks, 16, smart_pruning=False, concurrent_descents=concurrent_descents)
    report = search.search(chess.Board(FREE_QUEEN), simulations=3000)

    assert 3000 <= report.simulations <= 3000 + overrun(concurrent_descents) and report.line[0].uci() == "e4d5"
    assert group.most_out == 6 and all(network.rounds > 5 for network in group.networks)


class RoundsRecorded:
    def __init__(self, searches):
        self.searches, self.rounds, self.simulations = searches, [], []

    def __getattr__(self, name):
        return getattr(self.searches, name)

    def descend_into(self, boards, leaves, collisions, *arguments, **keywords):
        written, root = self.searches.descend_into(boards, leaves, collisions, *arguments, **keywords)
        self.rounds.append((leaves, collisions, written))
        self.simulations.append(arguments[0])
        return written, root

@pytest.mark.parametrize("concurrent_descents", [False, None])
def test_a_network_without_leaves_waits_for_the_others(concurrent_descents):
    search = EngineSearch(SlowNetworks(3, 0.005, BoardModel()).networks, 64, smart_pruning=False, grow_leaves=0, concurrent_descents=concurrent_descents)
    search.searches = recorded = RoundsRecorded(search.searches)

    assert 4000 <= search.search(chess.Board(), simulations=4000).simulations <= 4000 + overrun(concurrent_descents)
    written = [leaves_written for _, _, leaves_written in recorded.rounds]
    empty = written.count(0)
    assert empty < len(written) - empty, (empty, len(written) - empty)

def test_a_round_beside_a_round_out_takes_more_simulations_for_its_leaves():
    busy, alone = engine_search.BUSY_SIMULATIONS_PER_LEAF, engine_search.SIMULATIONS_PER_LEAF
    search = EngineSearch(cpu_batches(BoardModel()), 16, smart_pruning=False, grow_leaves=0)
    search.searches = recorded = RoundsRecorded(search.searches)
    search.search(chess.Board(), simulations=2000)
    #the first round has none out, every later one is gathered beside the one before it
    assert recorded.simulations[0] == alone * 16 and set(recorded.simulations[1:]) == {busy * 16}

    #rounds that grow with the search go out one at a time
    growing = EngineSearch(cpu_batches(BoardModel()), 16, smart_pruning=False)
    growing.searches = recorded = RoundsRecorded(growing.searches)
    growing.search(chess.Board(), simulations=200)
    assert all(simulations == alone * leaves for (leaves, _, _), simulations in zip(recorded.rounds, recorded.simulations) if leaves < 16)

def test_a_round_near_the_deadline_is_launched_as_soon_as_one_with_none_out():
    busy, alone = engine_search.BUSY_SIMULATIONS_PER_LEAF, engine_search.SIMULATIONS_PER_LEAF
    search = EngineSearch(cpu_batches(BoardModel()), 16)
    search.round_seconds = 0.01
    now = time.perf_counter()
    assert search.simulations_per_leaf(1, 1, None) == busy and search.simulations_per_leaf(0, 1, None) == alone
    #a round gathered now is back a round's time after its gather, taken as twice a round's time while none has been timed
    assert search.simulations_per_leaf(1, 1, now + 1.0) == busy and search.simulations_per_leaf(1, 1, now + 0.025) == alone
    #and then as twice the longest of the last ones
    search.busy_gathers.extend([0.001, 0.1, 0.002])
    assert search.simulations_per_leaf(1, 1, now + 1.0) == busy and search.simulations_per_leaf(1, 1, now + 0.15) == alone

def test_rounds_fill_where_most_simulations_need_no_network():
    #the mate's node takes most visits and none of them needs the network
    search = EngineSearch(cpu_batches(UniformModel()), 16, smart_pruning=False, grow_leaves=0)
    search.searches = recorded = RoundsRecorded(search.searches)
    search.search(chess.Board(MATE_IN_ONE), simulations=20_000)
    written = [leaves_written for _, _, leaves_written in recorded.rounds[1:-1]]
    assert sum(written) / len(written) > 12, sum(written) / len(written)

def test_one_network_in_a_list_is_the_search_on_it(carlsen_games):
    board = carlsen_position(carlsen_games, 1, 10)
    alone, listed = (EngineSearch(network, 16).search(board, simulations=2000) for network in (cpu_batches(BoardModel()), [cpu_batches(BoardModel())]))

    assert (alone.line, alone.score, alone.simulations, alone.root_moves) == (listed.line, listed.score, listed.simulations, listed.root_moves)
    with pytest.raises(ValueError):
        EngineSearch([cpu_batches(BoardModel())] * 2, 1)

def test_a_search_on_several_networks_keeps_to_its_deadline_and_pruning():
    search = EngineSearch(SlowNetworks(3, 0.003, UniformModel()).networks, 16)
    started = time.perf_counter()
    timed = search.search(chess.Board(), seconds=0.3)
    seconds = time.perf_counter() - started
    mate = EngineSearch(SlowNetworks(3, 0.003, UniformModel()).networks, 16).search(chess.Board(MATE_IN_ONE), simulations=40_000)
    (best, most), (_, second) = mate.root_moves[:2]

    assert timed.simulations > 300 and seconds < 0.3 + 0.1
    #the mate's visits need no network, so a round beside another takes all its simulations: the networks' rounds under way when the lead is
    #past half the search bring up to that many each
    overrun = 4 * 16 * engine_search.BUSY_SIMULATIONS_PER_LEAF
    assert best.uci() == "e1e8" and mate.simulations < 20_000 + overrun and most - second > 40_000 - mate.simulations

def test_a_failing_network_ends_the_search_with_its_error():
    search = EngineSearch(SlowNetworks(3, 0.003, BoardModel(), failing_collect=5).networks, 16, smart_pruning=False)
    started = time.perf_counter()
    with pytest.raises(RuntimeError, match="fell over"):
        search.search(chess.Board(), simulations=100_000)
    assert time.perf_counter() - started < 2

def test_stop_ends_a_search_on_several_networks():
    search = EngineSearch(SlowNetworks(3, 0.003, BoardModel()).networks, 16, smart_pruning=False)
    stop, stopped_at = threading.Event(), []

    #a timer may fire a little early: the search is timed from the stop itself
    def stop_now():
        stopped_at.append(time.perf_counter())
        stop.set()
    threading.Timer(0.2, stop_now).start()
    report = search.search(chess.Board(), simulations=300_000, stop=stop)

    #0.2 s of rounds that grow from one leaf are 70 to 180 simulations: more than a round's 16 shows the search ran
    assert 0 <= time.perf_counter() - stopped_at[0] < 0.1 and report.simulations > 16

def test_a_search_on_several_networks_tells_how_it_stands(monkeypatch):
    monkeypatch.setattr(engine_search, "INFO_SECONDS", 0.05)
    told = []
    report = EngineSearch(SlowNetworks(3, 0.003, BoardModel()).networks, 16, smart_pruning=False).search(chess.Board(), seconds=0.5, info=told.append)
    nodes = [running.simulations for running in told]

    assert len(told) >= 3 and nodes == sorted(nodes) and nodes[-1] <= report.simulations

def test_the_next_search_carries_on_the_last_tree():
    network = CountingNetwork(BoardModel())
    search = EngineSearch(network, 16, smart_pruning=False)
    board = chess.Board(FREE_QUEEN)
    first = search.search(board, simulations=400)

    network.boards = 0
    assert search.search(board, simulations=600).simulations == 600
    assert 0 < network.boards <= 200

    board.push(first.line[0])
    board.push(first.line[1])
    network.boards = 0
    assert search.search(board, simulations=400).simulations == 400
    assert network.boards < 400

    other = chess.Board(MATE_IN_ONE)
    network.boards = 0
    report = search.search(other, simulations=100)
    assert report.line[0] in other.legal_moves and network.boards > 50

def test_go_limits():
    clock = Clock()
    white, black = chess.Board(), chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    assert clock.limits("movetime 1000".split(), white, 800) == (None, pytest.approx(1 - MOVE_OVERHEAD_SECONDS))
    assert clock.limits("movetime 500 nodes 100".split(), white, 800) == (100, pytest.approx(0.5 - MOVE_OVERHEAD_SECONDS))
    assert clock.limits("nodes 5000".split(), black, 800) == (5000, None)
    assert clock.limits(["infinite"], black, 800) == (800, None)

    nodes, seconds = clock.limits("wtime 60000 btime 30000 winc 1000 binc 500".split(), black, 800)
    assert nodes is None and seconds == pytest.approx(Clock().seconds_for_move(30 - MOVE_OVERHEAD_SECONDS, 0.5, None, 32))
    assert 0.5 < seconds < 30 - RESERVE_SECONDS


class RecordingPlayer:
    def __init__(self):
        self.searches = []

    def search(self, board, nodes, seconds, extension, stop=None, info=None):
        self.searches.append((board.fen(), nodes, seconds, extension))
        return SearchReport(0.5, [next(iter(board.legal_moves))], 1, [])

#a search that raises (a rust panic is a BaseException)
class Panic(BaseException):
    pass

#a RecordingPlayer whose searches numbered in `failing` (from 0) tell `reported` through info, if given, and then raise `error`; the others play the last legal move
class FailingPlayer(RecordingPlayer):
    def __init__(self, failing: set[int], error: BaseException, reported: str | None = None):
        super().__init__()
        self.failing, self.error, self.reported = failing, error, reported

    def search(self, board, nodes, seconds, extension, stop=None, info=None):
        number = len(self.searches)
        self.searches.append((board.fen(), nodes, seconds, extension))
        if number in self.failing:
            if self.reported is not None:
                info(SearchReport(0.5, [chess.Move.from_uci(self.reported)], 1, []))
            raise self.error
        return SearchReport(0.5, [list(board.legal_moves)[-1]], 1, [])

#uci.main on the script with a RecordingPlayer (or `player`): its searches (fen, nodes, seconds, extension), the lines sent and load_player's arguments
def run_uci(monkeypatch, script: list[str], player: RecordingPlayer | None = None, **main_arguments) -> tuple[list, list[str], list]:
    player, sent, built = RecordingPlayer() if player is None else player, [], []
    monkeypatch.setattr(uci, "load_player", lambda *arguments: built.append(arguments) or player)
    monkeypatch.setattr(uci, "send", sent.append)
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(script)))
    uci.main(Path("unused.pt"), **main_arguments)
    return player.searches, sent, built

def searched_gos(monkeypatch, gos: list[str]) -> tuple[list, list[str]]:
    searches, sent, _ = run_uci(monkeypatch, ["position startpos", *gos, "quit"], simulations=800)
    return [(nodes, seconds) for _, nodes, seconds, _ in searches], sent

def test_go_nodes_0_plays_at_once(monkeypatch):
    given, sent = searched_gos(monkeypatch, ["go nodes 0", "go nodes -5", "go nodes 7"])
    assert given == [(1, None), (1, None), (7, None)] and len([line for line in sent if line.startswith("bestmove")]) == 3

#Searches.start takes a u32: a bigger count would raise in the binding and end the engine
def test_go_nodes_are_capped_at_what_a_search_takes(monkeypatch):
    given, sent = searched_gos(monkeypatch, ["go nodes 5000000000", f"go nodes {engine_search.UNLIMITED}", "go nodes 99999999999999999999"])
    assert given == [(engine_search.UNLIMITED, None)] * 3 and len([line for line in sent if line.startswith("bestmove")]) == 3

def test_go_leaves_out_numbers_that_are_not_whole(monkeypatch):
    given, sent = searched_gos(monkeypatch, ["go nodes x", "go movetime 1.5 nodes 30", "go wtime 1000.5 btime 1000"])
    assert given == [(800, None), (30, None), (800, None)] and len([line for line in sent if line.startswith("bestmove")]) == 3
    told = [line for line in sent if line.startswith("info string")]
    assert told == [f"info string go's {number} is not a whole number: left out" for number in ("nodes x", "movetime 1.5", "wtime 1000.5")]

#nodes are the root's visits, as lc0's (a kept tree's included); nps is the speed of this search's own
def test_info_nps_counts_only_the_visits_this_search_made():
    def reported(report: SearchReport) -> dict[str, str]:
        words = uci.info_line(report, 2.0).split()
        return {key: words[words.index(key) + 1] for key in ("nodes", "nps")}

    line = [chess.Move.from_uci("e2e4")]
    assert reported(SearchReport(0.5, line, 1500, [], reused=1000)) == {"nodes": "1500", "nps": "250"}
    assert reported(SearchReport(0.5, line, 1500, [])) == {"nodes": "1500", "nps": "750"}

#a search that fails still answers: with the newest info line's move, else the first legal move. The gui is told why, stderr has the traceback,
#and the player, whose search may be left half done, is built again for the next go
@pytest.mark.parametrize("error", [RuntimeError("syzygy tables:\nbad file"), Panic("boom")])
@pytest.mark.parametrize("reported", [None, "d2d4"])
def test_a_failed_search_still_answers_with_a_legal_move(monkeypatch, capsys, error, reported):
    board = chess.Board()
    normal, first = list(board.legal_moves)[-1].uci(), next(iter(board.legal_moves)).uci()
    player = FailingPlayer({1}, error, reported)

    searches, sent, built = run_uci(monkeypatch, ["isready", "position startpos", "go nodes 10", "go nodes 10", "go nodes 10", "quit"], player)

    assert [line for line in sent if line.startswith("bestmove")] == [f"bestmove {move}" for move in (normal, reported or first, normal)]
    told = [line for line in sent if line.startswith("info string")]
    assert len(told) == 1 and "\n" not in told[0] and type(error).__name__ in told[0] and (reported or first) in told[0]
    assert len(searches) == 3 and len(built) == 2
    assert type(error).__name__ in capsys.readouterr().err

#a search that fails before its first info line plays what the last search planned, if the reply was the one that search expected
@pytest.mark.parametrize("reply, planned", [("e7e5", True), ("c7c5", False)])
def test_a_search_that_fails_at_once_plays_the_last_searchs_plan(monkeypatch, capsys, reply, planned):
    line = [chess.Move.from_uci(move) for move in ("e2e4", "e7e5", "g1f3")]

    class PlanningPlayer(FailingPlayer):
        def search(self, board, nodes, seconds, extension, stop=None, info=None):
            if self.searches:
                return super().search(board, nodes, seconds, extension, stop, info)
            self.searches.append((board.fen(), nodes, seconds, extension))
            return SearchReport(0.5, line, 1, [])

    after = chess.Board()
    for move in ("e2e4", reply):
        after.push_uci(move)
    script = ["position startpos", "go nodes 10", f"position startpos moves e2e4 {reply}", "go nodes 10", "quit"]
    _, sent, _ = run_uci(monkeypatch, script, PlanningPlayer({1}, RuntimeError("the gpu fell over")))

    answer = "g1f3" if planned else next(iter(after.legal_moves)).uci()
    assert [line for line in sent if line.startswith("bestmove")] == ["bestmove e2e4", f"bestmove {answer}"] and answer in sent[-2]
    capsys.readouterr()

def test_a_search_is_not_answered_for_an_exit(monkeypatch):
    with pytest.raises(KeyboardInterrupt):
        run_uci(monkeypatch, ["position startpos", "go nodes 10", "quit"], FailingPlayer({0}, KeyboardInterrupt()))

def board_with_men(men: int) -> chess.Board:
    board = chess.Board()
    for square in [square for square in chess.SQUARES if board.piece_type_at(square) not in (None, chess.KING)][:32 - men]:
        board.remove_piece_at(square)
    return board

#what a search is given and does not use is banked for the next move; a movetime search is not on the clock, and no move gets the reserve
def test_unused_time_goes_to_the_next_move():
    clock, board = Clock(), chess.Board()
    _, first = clock.limits("wtime 60000 btime 60000".split(), board, 800)
    clock.searched(first / 2, 0)
    _, second = clock.limits("wtime 59000 btime 60000".split(), board, 800)
    _, without_saving = Clock().limits("wtime 59000 btime 60000".split(), board, 800)
    assert second == pytest.approx(clock.share + first / 2) and second > 1.4 * without_saving

    clock.searched(second, 0)
    _, third = clock.limits("wtime 58000 btime 60000".split(), board, 800)
    assert third == pytest.approx(Clock().limits("wtime 58000 btime 60000".split(), board, 800)[1])

    clock.searched(third / 4, 0)
    clock.limits("movetime 1000".split(), board, 800)
    clock.searched(0.1, 0)
    assert clock.saved == pytest.approx(3 / 4 * third)
    assert clock.limits("wtime 1500 btime 60000".split(), board, 800)[1] == pytest.approx(1.5 - MOVE_OVERHEAD_SECONDS - RESERVE_SECONDS)

#a 60+0.6 game losing men by the points below, every search taking all it is given
def test_the_clock_follows_the_material_and_never_runs_out():
    points = [(1, 32), (15, 32), (40, 16), (60, 10), (85, 6), (120, 6)]
    men_at = lambda move: next(round(men + (later_men - men) * (move - start) / (end - start))
                               for (start, men), (end, later_men) in zip(points, points[1:]) if start <= move <= end)
    clock, left, spent, low, late_middlegame_left = Clock(), 60.0, {}, 60.0, None
    for move in range(1, 121):
        men = men_at(move)
        if men <= 15 and late_middlegame_left is None:
            late_middlegame_left = left
        _, seconds = clock.limits(f"wtime {round(left * 1000)} btime 60000 winc 600 binc 600".split(), board_with_men(men), 800)
        clock.searched(seconds, 0)
        left += 0.6 - seconds - MOVE_OVERHEAD_SECONDS
        spent.setdefault("opening" if men >= 26 else "middle game" if men >= 16 else "late", []).append(seconds)
        low = min(low, left)
    opening, middle_game, endgame = (sum(moves) / len(moves) for moves in (spent["opening"], spent["middle game"], spent["late"][-30:]))
    assert middle_game > 1.1 * opening and endgame < 0.7
    assert late_middlegame_left > 12 and low > RESERVE_SECONDS / 2

def test_a_clocks_plan_and_cap_can_be_set():
    board = chess.Board()
    arguments = "wtime 60000 btime 60000 winc 600 binc 600".split()
    assert Clock(moves_per_man=1.0).limits(arguments, board, 800)[1] > 1.25 * Clock().limits(arguments, board, 800)[1]
    #late in the game the least moves left and the reserve decide the share
    board = board_with_men(4)
    arguments = "wtime 10000 btime 10000 winc 600 binc 600".split()
    assert Clock(least_moves_left=8, reserve_seconds=0.0).limits(arguments, board, 800)[1] == pytest.approx((10 - MOVE_OVERHEAD_SECONDS) / 8 + 0.6)
    assert Clock().limits(arguments, board, 800)[1] == pytest.approx((10 - MOVE_OVERHEAD_SECONDS - RESERVE_SECONDS) / 10 + 0.6)

def test_a_new_game_starts_a_fresh_clock(monkeypatch):
    go = ["position startpos", "go wtime 60000 btime 60000 winc 600 binc 600"]
    searches, _, _ = run_uci(monkeypatch, ["uci", "isready", *go * 8, "ucinewgame", *go, "quit"])
    seconds, extensions = [seconds for _, _, seconds, _ in searches], [extension for *_, extension in searches]
    assert seconds[7] > 2 * seconds[0] and seconds[8] == seconds[0]
    assert extensions[0] is None and extensions[8] is None
    last_score, extra = extensions[1]
    assert last_score == 0 and extra > 0

#the start position with the turn and move number of a game `ply` plies in
def board_at_ply(ply: int) -> chess.Board:
    board = chess.Board()
    board.turn, board.fullmove_number = ply % 2 == 0, 1 + ply // 2
    return board

LC0_FIRST_MOVE = {"first_move_bonus": 1.8, "book_ply_bonus": 0.25}
CLOCK_60 = "wtime 60000 btime 60000 winc 600 binc 600".split()

#lc0's first move bonus: the time without it times 1 + 1.8 + 0.25 x min(ply, 12)
@pytest.mark.parametrize("arguments", [CLOCK_60, "wtime 180000 btime 180000 winc 2000 binc 2000".split(), "wtime 60000 btime 60000 movestogo 40".split()])
@pytest.mark.parametrize("ply, bonus", [(0, 2.8), (1, 3.05), (8, 4.8), (16, 5.8), (40, 5.8)])
def test_the_games_first_search_gets_lc0s_bonus(arguments, ply, bonus):
    board = board_at_ply(ply)
    assert Clock(**LC0_FIRST_MOVE).limits(arguments, board, 800)[1] == pytest.approx(bonus * Clock().limits(arguments, board, 800)[1])

#the bonus goes to the first search on the clock only; what it leaves unused is banked as any search's
def test_the_first_search_bonus_is_given_once_and_banked():
    clock, board = Clock(**LC0_FIRST_MOVE), board_at_ply(16)
    clock.limits("movetime 1000".split(), board, 800)
    clock.searched(0.98, 0)
    _, first = clock.limits(CLOCK_60, board, 800)
    plain = Clock()
    _, plain_first = plain.limits(CLOCK_60, board, 800)
    assert first == pytest.approx(5.8 * plain_first) and clock.saved == 0
    assert clock.extension == (0, pytest.approx(MOST_EXTENSION * plain.share))

    bank = plain_first / 2
    clock.searched(first - bank, 0)
    plain.searched(plain_first - bank, 0)
    assert clock.saved == pytest.approx(bank) == plain.saved
    arguments = f"wtime {round((60 - first + bank) * 1000)} btime 60000 winc 600 binc 600".split()
    _, second = clock.limits(arguments, board_at_ply(18), 800)
    assert second == pytest.approx(plain.limits(arguments, board_at_ply(18), 800)[1]) == plain.share + bank
    assert clock.saved == pytest.approx(plain.saved) and clock.extension == pytest.approx(plain.extension)

def test_the_first_search_bonus_keeps_to_the_reserve_and_the_time_left():
    board, arguments = board_at_ply(16), "wtime 3000 btime 60000 winc 1000 binc 1000".split()
    assert Clock().limits(arguments, board, 800)[1] < 1.1
    assert Clock(**LC0_FIRST_MOVE).limits(arguments, board, 800)[1] == pytest.approx(3 - MOVE_OVERHEAD_SECONDS - RESERVE_SECONDS)
    assert Clock(reserve_seconds=0.0, **LC0_FIRST_MOVE).limits(arguments, board, 800)[1] == pytest.approx(3 - MOVE_OVERHEAD_SECONDS)

def test_a_falling_score_runs_the_search_on():
    search = EngineSearch(cpu_batches(UniformModel()), 16)
    timed = {}
    for last_score in (0, engine_search.FALL_FULL_CP):
        started = time.perf_counter()
        search.search(chess.Board(), seconds=0.1, extension=(last_score, 0.3))
        timed[last_score] = time.perf_counter() - started
    assert timed[0] < 0.1 + 0.1
    assert 0.1 + 0.25 < timed[engine_search.FALL_FULL_CP] < 0.1 + 0.3 + 0.15

def test_a_decided_search_stops_early():
    mate = EngineSearch(cpu_batches(UniformModel()), 16).search(chess.Board(MATE_IN_ONE), simulations=4000)
    (best, most), (_, second) = mate.root_moves[:2]
    #the mate's visits need no network: the round under way when the lead is past half the search takes all its simulations
    overrun = 16 * engine_search.BUSY_SIMULATIONS_PER_LEAF
    assert best.uci() == "e1e8" and mate.simulations < 2000 + overrun and most - second > 4000 - mate.simulations

    start = EngineSearch(cpu_batches(UniformModel()), 16).search(chess.Board(), simulations=400)
    assert start.simulations == 400

def test_a_search_without_smart_pruning_uses_its_whole_budget():
    mate = EngineSearch(cpu_batches(UniformModel()), 16, smart_pruning=False).search(chess.Board(MATE_IN_ONE), simulations=4000)
    assert mate.line[0].uci() == "e1e8" and mate.simulations == 4000

def test_pruning_never_changes_the_move(carlsen_games):
    positions = []
    for game in carlsen_games:
        board = game.board()
        for ply, move in enumerate(game.mainline_moves()):
            board.push(move)
            if ply % 9 == 6 and not board.is_game_over():
                positions.append(chess.Board(board.fen()))

    stopped_early = 0
    for board in positions:
        pruned = EngineSearch(cpu_batches(BoardModel()), 16).search(board, simulations=300)
        whole = EngineSearch(cpu_batches(BoardModel()), 16, smart_pruning=False).search(board, simulations=300)
        assert pruned.line[0] == whole.line[0] and whole.simulations == 300, board.fen()
        stopped_early += pruned.simulations < 300
    assert len(positions) > 20 and stopped_early > 5


#Searches scripted for smart pruning: the root has `root` after every round, the search is done after 30 rounds unless stopped,
#which notes the time
class ScriptedSearches:
    def __init__(self, visits: int, most: int, second: int, proven: bool = False, lone: bool = False):
        self.root, self.proven, self.lone = (visits, most, second), proven, lone
        self.finishes, self.stopped_at = 0, None

    def start(self, game, fen, moves, simulations, seconds=None):
        pass

    def descend_into(self, boards, count, collisions, simulations, round=None, root_of=None, legal_columns=False, alone=None):
        boards[:count] = 0
        return count, self.root

    def finish_with_root(self, answers, round=None, root_of=None):
        self.finishes += 1
        return [self.report()] if self.finishes == 30 and self.stopped_at is None else [], self.root

    def stop(self, game):
        self.stopped_at = time.perf_counter()
        return self.report()

    def root_visits(self, game):
        return self.root

    def decided(self, game):
        return self.proven

    def one_move_left(self, game):
        return self.lone

    def report(self):
        return engine_search.GAME, ["a1a2"], 0.5, [], self.root[0]

#1000 simulations with 600 done: the lead needed is more than 400 / factor
@pytest.mark.parametrize("factor, lead, stops", [(1.0, 400, False), (1.0, 401, True), (1.33, 300, False), (1.33, 301, True)])
def test_smart_prunings_factor(factor, lead, stops):
    search = EngineSearch(ImmediateBatches(answers_of_0_6), 16, grow_leaves=0, smart_pruning_factor=factor)
    search.searches = ScriptedSearches(600, 500, 500 - lead)
    search.search(chess.Board(), simulations=1000)
    assert (search.searches.stopped_at is not None) == stops


class SixteenthsNetwork:
    def __init__(self, now: list[float]):
        self.now = now

    def launch(self, boards: np.ndarray) -> np.ndarray:
        return boards

    def collect(self, boards: np.ndarray) -> np.ndarray:
        self.now[0] += 1 / 16
        return answers_of_0_6(boards)

#a decided root (more than the 400 left behind), a proven one and a single legal move stop at the first answers; an undecided root goes on
@pytest.mark.parametrize("fen, root, proven, stopped_at", [
    (chess.STARTING_FEN, (600, 600, 0), False, 1 / 16), (chess.STARTING_FEN, (600, 300, 290), True, 1 / 16),
    ("k7/8/8/8/8/8/r7/K1q5 w - - 0 1", (600, 300, 290), False, 1 / 16), (chess.STARTING_FEN, (600, 300, 290), False, None)])
def test_a_decided_proven_or_only_move_search_stops_at_the_first_answers(monkeypatch, fen, root, proven, stopped_at):
    now = [0.0]
    monkeypatch.setattr(engine_search.time, "perf_counter", lambda: now[0])
    search = EngineSearch(SixteenthsNetwork(now), 16, grow_leaves=0, settings={"certainty": True})
    search.searches = ScriptedSearches(*root, proven)
    search.search(chess.Board(fen), simulations=1000)
    assert search.searches.stopped_at == stopped_at

#every move but one proven lost stops the search at the first answers, as a proven result does, with smart pruning and proofs only
@pytest.mark.parametrize("smart_pruning, certainty, stopped_at", [(True, True, 1 / 16), (False, True, None), (True, False, None)])
def test_a_root_with_one_move_not_lost_stops_at_the_first_answers(monkeypatch, smart_pruning, certainty, stopped_at):
    now = [0.0]
    monkeypatch.setattr(engine_search.time, "perf_counter", lambda: now[0])
    search = EngineSearch(SixteenthsNetwork(now), 16, grow_leaves=0, smart_pruning=smart_pruning, settings={"certainty": certainty})
    search.searches = ScriptedSearches(600, 300, 290, lone=True)
    search.search(chess.Board(), simulations=1000)
    assert search.searches.stopped_at == stopped_at

#ScriptedSearches whose k-th round's finish moves the clock on 1/64 s and leaves the root at root(k): the rounds of several networks
#finish one at a time under the run's lock, so their checks see the same times and visits as one network's
class PacedSearches(ScriptedSearches):
    def __init__(self, now: list[float], root):
        super().__init__(*root(0))
        self.now, self.script = now, root

    def finish_with_root(self, answers, round=None, root_of=None):
        self.now[0] += 1 / 64
        self.root = self.script(self.finishes + 1)
        return super().finish_with_root(answers, round, root_of)

def paced_search(monkeypatch, networks: int, root, **limits) -> float | None:
    now = [0.0]
    monkeypatch.setattr(engine_search.time, "perf_counter", lambda: now[0])
    network = [ImmediateBatches(answers_of_0_6) for _ in range(networks)]
    search = EngineSearch(network if networks > 1 else network[0], 16, grow_leaves=0)
    search.searches = PacedSearches(now, root)
    search.search(chess.Board(), **limits)
    return search.searches.stopped_at

#a lead of 50 on a root that gains 100 visits a round, then none after 20 rounds, with 10 s to go: the last 8 rounds' speed falls to 0
#and stops one network's search at the 27th round; several networks take the whole search's speed, and the lead is far short of it
@pytest.mark.parametrize("networks, stopped_at", [(1, 27 / 64), (3, None)])
def test_rounds_without_new_visits_stop_a_search_on_one_network_only(monkeypatch, networks, stopped_at):
    def stalled(round: int) -> tuple[int, int, int]:
        visits = 1000 + 100 * min(round, 20)
        return visits, visits // 2 + 25, visits // 2 - 25
    assert paced_search(monkeypatch, networks, stalled, seconds=10.0) == stopped_at

#a lead of 6300 on a root that gains 6,400 visits a second with 1 s given is decided from the second round on; several networks wait
#SPEED_WAIT_SECONDS after their first check (the 14th round, 13/64 s on), unless the simulations left are fewer than the lead
@pytest.mark.parametrize("networks, simulations, stopped_at", [(1, None, 2 / 64), (3, None, 14 / 64), (1, 12_000, 1 / 64), (3, 12_000, 1 / 64)])
def test_a_decided_search_on_several_networks_stops_once_its_first_moments_are_past(monkeypatch, networks, simulations, stopped_at):
    def steady(round: int) -> tuple[int, int, int]:
        visits = 10_000 + 100 * round
        return visits, visits - 1000, visits - 7300
    assert paced_search(monkeypatch, networks, steady, seconds=1.0, simulations=simulations) == stopped_at

#the king has three moves, two of which lose to a mate in one
THREE_MOVES_TWO_LOST = "8/1K2r3/3r4/5P2/8/4k3/8/5Q2 w - - 0 1"

def certain_python_search() -> EngineSearch:
    return EngineSearch(cpu_batches(UniformModel()), 16, settings={"certainty": True})

def certain_rust_search() -> EngineSearch:
    return EngineSearch(lean_network(), 16, rust_rounds=True, settings={"certainty": True})

#once the two lost moves are proven the third is the move, not when its visits pass the simulations left (half of them)
@pytest.mark.parametrize("make_search", [certain_python_search, pytest.param(certain_rust_search, marks=gpu)])
def test_a_search_stops_once_one_move_is_not_proven_lost(make_search):
    report = make_search().search(chess.Board(THREE_MOVES_TWO_LOST), simulations=4000)
    assert report.line[0].uci() == "b7c8" and report.simulations < 1000

def engine_command(model_path: Path | str, *options: str) -> list[str]:
    return [sys.executable, "-m", "engine.uci", str(model_path), *options]

def run_engine(model_path: Path | str, *options: str) -> subprocess.CompletedProcess:
    return subprocess.run(engine_command(model_path, *options), cwd=REPOSITORY, capture_output=True, text=True, input="")

def test_the_engine_takes_search_settings(model_path):
    found = []
    for changes in ("c_puct=0.1", "c_puct=10"):
        engine = chess.engine.SimpleEngine.popen_uci(engine_command(model_path, "--search-settings", changes), cwd=REPOSITORY)
        try:
            played = engine.play(chess.Board(), chess.engine.Limit(nodes=400), info=chess.engine.INFO_ALL)
        finally:
            engine.quit()
        found.append((played.info["pv"], played.info["score"], played.info["nodes"]))
    assert found[0][2] == found[1][2] == 400 and found[0][:2] != found[1][:2]

#a tournament's build asks the engine its name where it has no gpu (the CCC's runs the image without them and waits for "id name")
def test_the_engine_answers_uci_where_all_the_gpus_are_none():
    def answer(commands: str) -> subprocess.CompletedProcess:
        return subprocess.run(engine_command("unused.pt", "--gpu", "--gpus", "all", "--fit-memory"), cwd=REPOSITORY, capture_output=True, text=True,
                              input=commands, env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"), timeout=120)
    named = answer("uci\nquit\n")
    assert named.returncode == 0, named.stderr
    lines = named.stdout.splitlines()
    assert lines[0].startswith("id name Fry ") and lines[-1] == "uciok"
    #it cannot search there: isready ends it with the reason
    loading = answer("isready\n")
    assert loading.returncode != 0 and "--gpus all found no cuda device" in loading.stderr and "readyok" not in loading.stdout

#records (precision, fp16 blocks, blocking wait) of each network built
def fake_engine_network(built: list | None = None):
    def engine_network(checkpoint, leaves, precision, fp16_blocks, device=None, blocking_wait=False):
        if built is not None:
            built.append((precision, fp16_blocks, blocking_wait))
        return cpu_batches(BoardModel())
    return engine_network

def run_uci_main(monkeypatch, *options: str, stdin: str = "isready\nquit\n") -> None:
    monkeypatch.setattr(sys, "argv", ["uci.py", "unused.pt", "--gpu", *options])
    #run as __main__, uci imports anew: its checkpoint is not there to read
    monkeypatch.setattr("model.checkpoint.served_value_sharpness", lambda checkpoint_path: None)
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    runpy.run_module("engine.uci", run_name="__main__")

uci_as_main = pytest.mark.filterwarnings("ignore:'engine.uci' found in sys.modules")

def test_the_engine_takes_a_precision(monkeypatch, model_path):
    built = []
    monkeypatch.setattr(uci, "engine_network", fake_engine_network(built))
    search = uci.load_player(model_path, True, 4, False, {"smart_pruning": False}, "fp16", 1)
    uci.load_player(model_path, True, 4, False, {"smart_pruning": False}, "int8", 0)
    refused = [run_engine(model_path, *options) for options in (["--precision", "fp16"], ["--gpu", "--precision", "fp16", "--fp16-blocks", "0"])]

    assert [(precision, fp16_blocks) for precision, fp16_blocks, _ in built] == [("fp16", 1), ("int8", 0)] and search.search(chess.Board(), simulations=20).simulations == 20
    assert refused[0].returncode == 2 and "--precision needs --gpu" in refused[0].stderr
    assert refused[1].returncode == 2 and "--fp16-blocks needs --gpu and the int8 network" in refused[1].stderr

def test_the_engine_takes_gpus(monkeypatch, model_path):
    built = []
    monkeypatch.setattr(uci, "engine_networks", lambda checkpoint, leaves, precision, fp16_blocks, devices, blocking_wait:
                        [built.append((fp16_blocks, device, blocking_wait)) or cpu_batches(BoardModel()) for device in devices])
    uci.load_player(model_path, True, 4, False, {"smart_pruning": False}, "int8", 1, [1])
    search = uci.load_player(model_path, True, 4, False, {"smart_pruning": False}, "int8", 1, [0, 2])
    refused = [run_engine("unused.pt", *options, "--gpus", "0,1") for options in (["--leaves", "4"], ["--gpu", "--leaves", "1"])]

    assert built == [(1, 1, False), (1, 0, True), (1, 2, True)] and len(search.networks) == 2 and search.concurrent_descents
    assert search.network_cpus == engine_search.thread_cpus(2) != [] and search.search(chess.Board(), simulations=200).simulations >= 200
    assert all(run.returncode == 2 and "--gpus needs --gpu" in run.stderr for run in refused)

#a gpu whose network cannot be built is left out and the gui told, also in the line that says what was loaded; with none left the failure is raised
def test_a_gpu_that_fails_is_left_out(monkeypatch, model_path):
    def engine_networks(checkpoint, leaves, precision, fp16_blocks, devices, blocking_wait):
        if set(devices) & {1, 3}:
            raise RuntimeError("the gpu fell over")
        return [cpu_batches(BoardModel()) for _ in devices]
    sent = []
    monkeypatch.setattr(uci, "engine_networks", engine_networks)
    monkeypatch.setattr(uci, "send", sent.append)

    search = uci.load_player(model_path, True, 4, False, {"smart_pruning": False}, "int8", 1, [0, 1, 2])

    assert len(search.networks) == 2 and search.search(chess.Board(), simulations=200).simulations >= 200
    assert sent == ["info string gpu 1 failed (RuntimeError('the gpu fell over')): searching without it",
                    f"info string Fry on 2 of 3 gpus (0,2); no tables; no tree limit; cache {engine_search.CACHE_POSITIONS} positions"]
    for devices in ([1], [1, 3]):
        with pytest.raises(RuntimeError, match="fell over"):
            uci.load_player(model_path, True, 4, False, {}, "int8", 1, devices)

def test_the_gui_is_told_what_was_loaded(monkeypatch, tablebase, model_path):
    sent = []
    monkeypatch.setattr(uci, "engine_network", fake_engine_network())
    monkeypatch.setattr(uci, "send", sent.append)

    uci.load_player(model_path, True, 4, True, {"memory_limit_mb": 64, "cache_positions": 1000}, "int8", 1)

    assert sent == ["info string Fry on one gpu; 145 wdl and 145 dtz tables, used up to 5 pieces; tree limit 64 MB; cache 1000 positions"]

#a checkpoint recording the value sharpness it is served at sets the engine's default; one the engine is told wins over it
def test_the_engine_serves_a_checkpoint_at_its_own_value_sharpness(monkeypatch, model_path):
    monkeypatch.setattr(uci, "engine_network", fake_engine_network([]))
    monkeypatch.setattr(engine_search, "ENGINE_SETTINGS", {"value_sharpness": 3.0})
    default = uci.load_player(model_path, True, 4, False, {}, "int8", 1)
    torch.save({**torch.load(model_path), "value_sharpness": 1.25}, model_path)
    served = uci.load_player(model_path, True, 4, False, {}, "int8", 1)
    told = uci.load_player(model_path, True, 4, False, {"settings": {"value_sharpness": 2.0}}, "int8", 1)

    assert (default.value_sharpness, served.value_sharpness, told.value_sharpness) == (3.0, 1.25, 2.0)

@uci_as_main
def test_the_engine_plays_in_int8_unless_told_otherwise(monkeypatch):
    built = []
    monkeypatch.setattr(engine_search, "engine_network", fake_engine_network(built))
    for options in ([], ["--fp16-blocks", "0"]):
        run_uci_main(monkeypatch, *options)

    assert built == [("int8", 1, False), ("int8", 0, False)]

@uci_as_main
def test_the_engine_takes_lc0s_time_options(monkeypatch):
    given = []
    def search(self, board, nodes, seconds, extension, stop, info=None):
        given.append((seconds, self.smart_pruning_factor))
        return SearchReport(0.5, [next(iter(board.legal_moves))], 1, [])
    monkeypatch.setattr(engine_search, "engine_network", fake_engine_network())
    monkeypatch.setattr(EngineSearch, "search", search)
    go = "go " + " ".join(CLOCK_60)
    script = ["isready", "position startpos", go, "position startpos moves e2e4 e7e5", go, "ucinewgame", "position startpos", go, "quit"]
    options = ["--first-move-bonus", "1.8", "--book-ply-bonus", "0.25", "--smart-pruning-factor", "1.33"]
    run_uci_main(monkeypatch, *options, stdin="\n".join(script))

    first = pytest.approx(2.8 * Clock().limits(CLOCK_60, chess.Board(), 800)[1])
    assert given[0] == (first, 1.33) and given[1][0] != first and given[2] == given[0]
    with pytest.raises(SystemExit) as refused:
        run_uci_main(monkeypatch, "--smart-pruning-factor", "0")
    assert refused.value.code == 2

def test_the_engine_keeps_to_gos_limits(model_path):
    engine = chess.engine.SimpleEngine.popen_uci(engine_command(model_path), cwd=REPOSITORY)
    try:
        board = chess.Board()
        engine.play(board, chess.engine.Limit(nodes=10))

        started = time.perf_counter()
        timed = engine.play(board, chess.engine.Limit(time=0.5), info=chess.engine.INFO_BASIC)
        seconds = time.perf_counter() - started
        other = chess.Board(MATE_IN_ONE)
        counted = engine.play(other, chess.engine.Limit(nodes=50), info=chess.engine.INFO_BASIC)
        #a game of its own ("clock", so ucinewgame), a 10+0.1 clock: ~0.27 s for the first move, ~1 s for a 9 man endgame
        clock = chess.engine.Limit(white_clock=10, black_clock=10, white_inc=0.1, black_inc=0.1)
        started = time.perf_counter()
        engine.play(board, clock, game="clock")
        first_move = time.perf_counter() - started
        started = time.perf_counter()
        engine.play(chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 40"), clock, game="clock")
        endgame_move = time.perf_counter() - started
    finally:
        engine.quit()

    assert timed.move in board.legal_moves and timed.info["nodes"] > 10 and 0.3 < seconds < 0.5 + 0.1
    assert counted.info["nodes"] == 50
    assert first_move < 0.5 and 0.8 < endgame_move < 1.3

#the report (id, line, score, root moves, visits) of Searches' search of board, one leaf a round
def searched(searches: selfplay_rs.Searches, board: chess.Board, simulations: int, evaluate, search_id: int = 0):
    searches.start(search_id, board.root().fen(), [move.uci() for move in board.move_stack], simulations)
    reports = []
    while not reports:
        reports = searches.finish(evaluate(leaf_boards(searches.descend())))
    return reports[0]

def test_every_game_keeps_its_tree():
    network = CountingNetwork(BoardModel())
    searches = selfplay_rs.Searches(**rust_search_settings(False), tree_reuse=True)
    boards = [chess.Board(FREE_QUEEN), chess.Board()]
    for search_id, board in enumerate(boards):
        searched(searches, board, 400, network.launch, search_id)

    for search_id, board in enumerate(boards):
        network.boards = 0
        assert searched(searches, board, 600, network.launch, search_id)[4] == 600
        assert 0 < network.boards <= 200

#mate in two: the defender's one reply is mated; the tree kept for the next move keeps the proof of the mate in one
def test_certainty_proves_a_mate_and_the_search_stops():
    board = chess.Board(MATE_IN_TWO)
    network = CountingNetwork(UniformModel())
    search = EngineSearch(network, 16, settings={"certainty": True})
    proven = search.search(board, simulations=20_000)
    assert proven.line[0].uci() in MATES_IN_TWO and proven.score == 1.0 and proven.simulations < 5_000

    unproven = EngineSearch(cpu_batches(UniformModel()), 16).search(board, simulations=proven.simulations)
    assert unproven.score < 1.0

    board.push_uci("c6b6")
    board.push_uci("a8b8")
    network.boards = 0
    mate = search.search(board, simulations=20_000)
    assert mate.line[0].uci() == "h1h8" and mate.score == 1.0 and network.boards <= 2 * 16

def test_certainty_stops_descents_at_proven_nodes():
    network = CountingNetwork(UniformModel())
    search = EngineSearch(network, 16, smart_pruning=False, settings={"certainty": True})
    report = search.search(chess.Board(MATE_IN_TWO), simulations=6000)

    assert report.simulations == 6000 and report.score == 1.0
    assert network.boards < 3000


#the uniform model, but for the moves it likes (uci, side), e^logit times more likely than any other
class LikesMoves(nn.Module):
    def __init__(self, moves: list[tuple[str, chess.Color]], logit: float):
        super().__init__()
        self.move_ids = [encode_move_to_id(chess.Move.from_uci(move), side) for move, side in moves]
        self.logit = logit

    def forward(self, boards):
        logits, values = UniformModel()(boards)
        logits[:, self.move_ids] = self.logit
        return logits, values

def test_certainty_plays_a_proven_win_over_the_most_visited_move():
    board = chess.Board(MATE_IN_ONE)
    model = LikesMoves([("g1f1", chess.WHITE)], 5.0)
    report = EngineSearch(cpu_batches(model), 16, settings={"certainty": True}).search(board, simulations=4000)

    assert report.line[0].uci() == "e1e8" and report.score == 1.0
    assert report.root_moves[0][0].uci() == "g1f1"

#the network all but insists on Ra2, after which Rd1 mates
def test_certainty_avoids_a_proven_loss_the_network_likes():
    board = chess.Board("3r2k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    model = LikesMoves([("a1a2", chess.WHITE)], 8.0)
    assert EngineSearch(cpu_batches(model), 16).search(board, simulations=4000).line[0].uci() == "a1a2"
    report = EngineSearch(cpu_batches(model), 16, settings={"certainty": True}).search(board, simulations=4000)

    assert report.line[0].uci() != "a1a2" and dict(report.root_moves)[report.line[0]] > 100
    board.push(report.line[0])
    for reply in list(board.legal_moves):
        board.push(reply)
        assert not board.is_checkmate()
        board.pop()

def test_search_settings_take_true_and_false():
    settings = {"certainty": True, "c_puct": 0.8}
    assert parse_changes("certainty=false,c_puct=1.5", settings) == {"certainty": False, "c_puct": 1.5}
    with pytest.raises(ValueError):
        parse_changes("certainty=0", settings)

#only numbers and true/false are settings to change; a gui's setoption must not end the engine
def test_search_settings_that_are_not_numbers_are_refused():
    settings = engine_search_settings(False)
    assert settings["tablebase_folder"] is None and isinstance(settings["move_id_table"], list)
    for change in ("move_id_table=1", "tablebase_folder=x"):
        with pytest.raises(ValueError):
            parse_changes(change, settings)
    options = Options(False, TABLEBASE_PIECES)
    for change in ("move_id_table=1", "tablebase_folder=x", "repetitions_for_a_draw=inf", "c_puct=x", "nope=1"):
        assert options.set("SearchSettings", change, built=False).endswith("the search's settings stay as they were")
    assert options.settings is None


#a competition's file as its gui reads it: TCEC's a list of engines with their options as names and values, the CCC's one engine with its
#options a table
def competition_options(competition: Path) -> dict[str, str]:
    file = json.loads(competition.read_text())
    if isinstance(file, list):
        return {option["name"]: str(option["value"]) for option in file[0]["options"]}
    return file["options"]

#a competition's gui sends its file's options as setoptions: a misspelt name or a value out of range is refused there. SyzygyPath's is a placeholder
@pytest.mark.parametrize("competition", sorted((Path(__file__).parent.parent / "engine" / "linux").glob("*.json")), ids=lambda path: path.name)
def test_a_competitions_options_are_ones_the_engine_takes(competition):
    options = Options(False, TABLEBASE_PIECES, gpu=True)
    set_options = competition_options(competition)
    assert "SyzygyPath" in set_options
    for name, value in set_options.items():
        assert name in options.names()
        if name != "SyzygyPath":
            assert options.set(name, value, built=False) is None


GIB_MB = 2**30 / 1_000_000

#TCEC's machine has 768 GiB and lets an engine use 320: the fitted cache, the tree at its peak and the rest of the process take the
#engine's two fifths and no more
def test_fitted_memory_fills_the_engines_share_of_the_machine():
    cache_positions, tree_mb = fitted_memory(int(768 * GIB_MB), 8)
    assert cache_positions == FITTED_CACHE_POSITIONS
    used_mb = tree_mb * 1.5 + cache_positions * CACHE_POSITION_BYTES / 1_000_000 + 4000 + 8 * 1500
    assert 0.39 * 768 * GIB_MB < used_mb <= 0.4 * 768 * GIB_MB < 320 * GIB_MB
    #one gpu's tree is copied whole when it is kept, so the same memory allows it a lower limit; a small machine's cache is its share
    one_gpu_cache, one_gpu_tree_mb = fitted_memory(int(64 * GIB_MB), 1)
    assert one_gpu_cache * CACHE_POSITION_BYTES / 1_000_000 <= 0.2 * 0.4 * 64 * GIB_MB
    assert 0.38 * 64 * GIB_MB < one_gpu_tree_mb * 2 + one_gpu_cache * CACHE_POSITION_BYTES / 1_000_000 + 5500 <= 0.4 * 64 * GIB_MB
    assert fitted_memory(int(64 * GIB_MB), 2)[1] > one_gpu_tree_mb
    #a cache that was set leaves the tree the rest
    assert fitted_memory(int(64 * GIB_MB), 1, cache_positions=1_000_000) == (1_000_000, 10_000)
    assert fitted_memory(16_000, 1) is None

def test_a_containers_limit_counts_as_the_machines_memory(tmp_path):
    (tmp_path / "limit").write_text("2000000000\n")
    (tmp_path / "none").write_text("max\n")
    files = [str(tmp_path / name) for name in ("none", "missing", "limit")]
    assert memory_within_limits(64_000_000_000, files) == 2_000_000_000
    assert memory_within_limits(1_000_000_000, files) == 1_000_000_000
    assert memory_within_limits(64_000_000_000, files[:2]) == 64_000_000_000

#--fit-memory's sizes are the options' defaults: the gui is shown them, and an option it sets wins
def test_options_fit_the_machines_memory_until_they_are_set():
    plain = Options(False, TABLEBASE_PIECES, gpu=True).search()[1]
    assert (plain["cache_positions"], plain["memory_limit_mb"]) == (engine_search.CACHE_POSITIONS, 0)
    memory_mb = int(768 * GIB_MB)
    options = Options(False, TABLEBASE_PIECES, gpu=True, gpus=list(range(8)), memory_mb=memory_mb)
    cache_positions, tree_mb = fitted_memory(memory_mb, 8)
    assert (options.search()[1]["cache_positions"], options.search()[1]["memory_limit_mb"]) == (cache_positions, tree_mb)
    assert f"option name CachePositions type spin default {cache_positions} " in "\n".join(options.lines())
    assert f"option name RamLimitMb type spin default {tree_mb} " in "\n".join(options.lines())
    assert options.set("CachePositions", "50000000", built=False) is None
    assert options.search()[1]["cache_positions"] == 50_000_000 and options.search()[1]["memory_limit_mb"] > tree_mb
    assert options.set("RamLimitMb", "0", built=False) is None
    assert options.search()[1]["memory_limit_mb"] == 0
    #a machine too small to fit keeps the defaults
    small = Options(False, TABLEBASE_PIECES, gpu=True, memory_mb=16_000).search()[1]
    assert (small["cache_positions"], small["memory_limit_mb"]) == (engine_search.CACHE_POSITIONS, 0)


#answers come back late once, as on a cuda graph's first capture or a gpu shared with another engine: on the on_collect'th collect
class StallsOnce(WrappedNetwork):
    def __init__(self, network, stall_seconds: float, on_collect: int):
        super().__init__(network)
        self.stall_seconds, self.on_collect = stall_seconds, on_collect
        self.collects = self.boards = 0

    def launch(self, boards: np.ndarray):
        self.boards += len(boards)
        return super().launch(boards)

    def collect(self, launched):
        self.collects += 1
        if self.collects == self.on_collect:
            time.sleep(self.stall_seconds)
        return super().collect(launched)

@pytest.mark.parametrize("grow_leaves", [0, 16])
def test_a_stalled_round_does_not_cut_the_next_searches_short(grow_leaves):
    network = StallsOnce(cpu_batches(BoardModel()), 1.5, on_collect=2)
    search = EngineSearch(network, 16, smart_pruning=False, grow_leaves=grow_leaves)
    search.search(chess.Board(), seconds=0.05)
    for fen in [FREE_QUEEN, MATE_IN_ONE, "3r2k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1"]:
        network.boards = 0
        search.search(chess.Board(fen), seconds=0.3)
        assert network.boards > 5 * 16

def test_a_new_game_starts_a_fresh_tree():
    board = chess.Board(FREE_QUEEN)
    network = CountingNetwork(BoardModel())
    search = EngineSearch(network, 16, smart_pruning=False)
    search.search(board, simulations=300)
    network.boards = 0
    assert search.search(board, simulations=300).simulations == 300 and network.boards == 0

    search.new_game()
    again = search.search(board, simulations=300)
    fresh = EngineSearch(cpu_batches(BoardModel()), 16, smart_pruning=False).search(board, simulations=300)
    assert network.boards > 100
    assert (again.line, again.root_moves, again.score) == (fresh.line, fresh.root_moves, fresh.score)

#the knights out and back, which the network all but insists on, so that the search reaches the start position again
KNIGHTS_OUT_AND_BACK = [("g1f3", chess.WHITE), ("g8f6", chess.BLACK), ("f3g1", chess.WHITE), ("f6g8", chess.BLACK)]
SHUFFLES = ["g8f6", "f6g8", "a7a6", "g8f6", "f6g8", "a6a5", "b8c6", "c6b8", "h7h6", "h6h5"]

def shuffling_model() -> LikesMoves:
    return LikesMoves(KNIGHTS_OUT_AND_BACK + [("b1c3", chess.WHITE), ("c3b1", chess.WHITE)], 4.0)

def test_stale_draws_come_out_of_a_kept_tree():
    search = EngineSearch(cpu_batches(LikesMoves(KNIGHTS_OUT_AND_BACK, 8.0)), 16, smart_pruning=False, grow_leaves=0)
    board = chess.Board()
    first = search.search(board, simulations=400)
    assert first.root_moves[0] == (chess.Move.from_uci("g1f3"), 399)
    board.push_uci("g1f3")
    #a search of one simulation reports the kept tree's visits
    assert search.search(board, simulations=1).simulations < 50

#locked pawns with White one up, so that Black wants the draw. The kings step to and fro, and 4...Ke8 makes the position the game's
#third time. The search at move 2 came to it by other king moves, when the game had had it once, and its node is kept with moves
def test_a_kept_tree_with_transpositions_knows_the_games_third_time():
    def player() -> EngineSearch:
        return EngineSearch(cpu_batches(EvenModel()), 16, smart_pruning=False, settings={"transpositions": True, "certainty": True})

    board, kept = chess.Board("4k3/8/8/1p1p1p1p/1P1P1P1P/1P6/8/4K3 w - - 0 1"), player()
    for king, other_king in [("e1d1", "e8d8"), ("d1e1", "d8e8"), ("e1d1", "e8d8")]:
        board.push_uci(king)
        board.push_uci(other_king)
        kept.search(board, simulations=20_000)
    board.push_uci("d1e1")
    for search in (kept, player()):
        report = search.search(board, simulations=4_000)
        assert (report.line[0], report.score) == (chess.Move.from_uci("d8e8"), 0.5)

def test_a_root_the_last_search_never_expanded_is_searched():
    search = EngineSearch(cpu_batches(LikesMoves(KNIGHTS_OUT_AND_BACK, 8.0)), 16, smart_pruning=False, grow_leaves=0)
    board = chess.Board()
    search.search(board, simulations=100)
    for move, _ in KNIGHTS_OUT_AND_BACK:
        board.push_uci(move)
    report = search.search(board, simulations=50)
    assert report.line and report.line[0] in board.legal_moves and report.simulations >= 50

@pytest.mark.parametrize("fen, moves", [(chess.STARTING_FEN, [move for move, _ in KNIGHTS_OUT_AND_BACK] * 2),
                                         ("8/8/4k3/8/2R5/4K3/8/8 w - - 100 120", [])])
@pytest.mark.parametrize("leaves", [1, 16])
def test_the_engine_plays_on_in_a_draw_left_to_be_claimed(fen, moves, leaves):
    board = chess.Board(fen)
    for move in moves:
        board.push_uci(move)
    assert board.is_repetition(3) or board.is_fifty_moves()

    report = EngineSearch(cpu_batches(BoardModel()), leaves).search(board, simulations=200)
    assert report.line[0] in board.legal_moves and report.simulations > 1

#plays `moves` searches, each answered by the reply its search expects (or `replies`). way: "plain", "advance" (uci.py's after each
#move) or "drop" (an advance to another game). Returns the reports, the replies, the network's boards, the advances that had a tree
#to carry on, and the search
def advancing_game(model, board: chess.Board, moves: int, way: str, replies: list[str] | None = None, **options):
    network = CountingNetwork(model)
    search = EngineSearch(network, 16, smart_pruning=False, **options)
    reports, played, advanced = [], [], 0
    for ply in range(moves):
        report = search.search(board, simulations=600)
        reports.append((report.line, report.score, report.simulations, report.root_moves))
        board.push(report.line[0])
        if way == "advance":
            advanced += search.advance(board)
        elif way == "drop":
            search.advance(chess.Board("8/8/8/8/8/8/8/K6k w - - 0 1"))
        reply = chess.Move.from_uci(replies[ply]) if replies else report.line[1] if len(report.line) > 1 else next(iter(board.legal_moves))
        #a game that went another way than the replies' ends
        if reply not in board.legal_moves:
            break
        board.push(reply)
        played.append(reply.uci())
    return reports, played, network.boards, advanced, search

def test_advancing_the_tree_changes_no_search(middlegame):
    reports, replies, boards, _, _ = advancing_game(BoardModel(), middlegame.copy(), 8, "plain")
    advanced_reports, _, advanced_boards, advanced, _ = advancing_game(BoardModel(), middlegame.copy(), 8, "advance", replies)
    dropped_reports = advancing_game(BoardModel(), middlegame.copy(), 8, "drop", replies)[0]
    assert advanced_reports == reports != dropped_reports and advanced == 8 and advanced_boards == boards

    model = shuffling_model()
    reports = advancing_game(model, chess.Board(), len(SHUFFLES), "plain", SHUFFLES)[0]
    advanced_reports, _, _, advanced, search = advancing_game(model, chess.Board(), len(SHUFFLES), "advance", SHUFFLES)
    assert advanced_reports == reports and advanced == len(SHUFFLES)

    board = chess.Board()
    report = search.search(board, simulations=50)
    played, other = board.copy(), board.copy()
    played.push(report.line[0])
    search.advance(played)
    other.push(next(move for move in board.legal_moves if move != report.line[0]))
    other.push(next(iter(other.legal_moves)))
    assert search.search(other, simulations=50).line[0] in other.legal_moves

#the visits the root already had are those of the tree kept from the last search
def test_a_report_tells_the_visits_it_found_in_a_kept_tree():
    search = EngineSearch(cpu_batches(BoardModel()), 16, smart_pruning=False)
    board = chess.Board()
    first = search.search(board, simulations=300)
    board.push(first.line[0])
    board.push(first.line[1])

    second = search.search(board, simulations=600)

    assert first.reused == 0 and 0 < second.reused < 300 < 600 <= second.simulations

def test_a_search_ends_at_its_memory_limit():
    evaluate = cpu_batches(BoardModel()).evaluate_boards
    reports = {limit: searched(selfplay_rs.Searches(**rust_search_settings(False), memory_limit_mb=limit), chess.Board(), 3000, evaluate)
               for limit in (0, 1, 1000)}
    assert reports[1000] == reports[0] and reports[0][4] == 3000
    assert 100 < reports[1][4] < 3000


#Searches must be used on the thread that made it, so load_player makes its player with make_player (unless None) on the engine's thread
class Gui:
    def __init__(self, monkeypatch, make_player, checkpoint: Path = Path("unused.pt"), **main_arguments):
        self.lines: queue.Queue = queue.Queue()
        self.sent: list[tuple[float, str]] = []
        self.errors: list[BaseException] = []
        monkeypatch.setattr(sys, "stdin", iter(self.lines.get, None))
        monkeypatch.setattr(uci, "send", lambda line: self.sent.append((time.perf_counter(), line)))
        if make_player is not None:
            monkeypatch.setattr(uci, "load_player", lambda *arguments: make_player())
        self.engine = threading.Thread(target=self.run, args=(checkpoint,), kwargs=main_arguments, daemon=True)
        self.engine.start()

    def run(self, checkpoint: Path, **main_arguments) -> None:
        try:
            uci.main(checkpoint, **main_arguments)
        except BaseException as error:
            self.errors.append(error)

    def tell(self, line: str) -> float:
        told = time.perf_counter()
        self.lines.put(line + "\n")
        return told

    #the time of the first line sent since `since` that starts with `start`, waiting up to `seconds` for one
    def sent_at(self, start: str, since: float = 0.0, seconds: float = 0.0) -> float | None:
        give_up = time.perf_counter() + seconds
        while True:
            found = [at for at, line in list(self.sent) if at >= since and line.startswith(start)]
            if found or time.perf_counter() > give_up:
                return found[0] if found else None
            time.sleep(0.005)

    def lines_sent(self, start: str) -> list[str]:
        return [line for _, line in list(self.sent) if line.startswith(start)]

    def last_info(self, name: str) -> int:
        words = self.lines_sent("info depth")[-1].split()
        return int(words[words.index(name) + 1])

    def close(self, last: str | None = "quit", seconds: float = 5.0) -> None:
        if last is not None:
            self.tell(last)
        self.lines.put(None)
        self.engine.join(seconds)
        assert not self.engine.is_alive() and not self.errors, self.errors

def board_search() -> EngineSearch:
    return EngineSearch(cpu_batches(BoardModel()), 16)

#a small network on lean lanes of 16 boards, collecting the earlier tests' graphs first: one freed while these are recorded breaks the recording
def lean_network(blocking_wait: bool = False) -> GraphedBatches:
    gc.collect()
    torch.manual_seed(0)
    model = ChessTransformer(d_model=32, num_layers=1, n_heads=2).cuda().eval()
    return GraphedBatches(model, small_graph_sizes(16), 2, blocking_wait=blocking_wait, lean=True)

def rust_search() -> EngineSearch:
    return EngineSearch(lean_network(), 16, rust_rounds=True)

both_loops = pytest.mark.parametrize("make_search", [board_search, pytest.param(rust_search, marks=gpu)])

@both_loops
def test_stop_ends_a_running_search(monkeypatch, make_search):
    gui = Gui(monkeypatch, make_search)
    gui.tell("position startpos")
    gui.tell("go movetime 10000")
    time.sleep(0.3)
    stopped = gui.tell("stop")
    answered = gui.sent_at("bestmove", seconds=3)
    assert answered is not None and stopped < answered < stopped + 1 and gui.last_info("nodes") > 16
    gui.close()

@both_loops
def test_isready_is_answered_while_the_engine_searches(monkeypatch, make_search):
    gui = Gui(monkeypatch, make_search)
    gui.tell("go movetime 10000")
    time.sleep(0.3)
    asked = gui.tell("isready")
    ready = gui.sent_at("readyok", seconds=1)
    assert ready is not None and ready < asked + 0.5 and gui.sent_at("bestmove") is None

    #readyok waits for a command the engine has not taken yet
    gui.tell("position startpos moves e2e4")
    asked = gui.tell("isready")
    time.sleep(0.3)
    assert gui.sent_at("readyok", since=asked) is None
    gui.tell("stop")
    answered, ready = gui.sent_at("bestmove", seconds=3), gui.sent_at("readyok", since=asked, seconds=3)
    assert answered is not None and ready is not None and answered < ready
    gui.close()

@both_loops
def test_quit_ends_a_running_search(monkeypatch, make_search):
    gui = Gui(monkeypatch, make_search)
    gui.tell("go movetime 10000")
    time.sleep(0.3)
    gui.tell("quit")
    gui.engine.join(2)
    assert not gui.engine.is_alive() and not gui.errors
    gui.close(last=None)

#go infinite answers only at stop, also once its search has ended by itself (an only move); the end of the input ends one too
@pytest.mark.parametrize("fen", [chess.STARTING_FEN, "7k/8/8/8/8/8/8/K5R1 b - - 0 1"])
def test_go_infinite_searches_until_stop(monkeypatch, fen):
    gui = Gui(monkeypatch, board_search, simulations=50, infinite_until_stop=True)
    gui.tell(f"position fen {fen}")
    gui.tell("go infinite")
    time.sleep(0.5)
    assert gui.sent_at("bestmove") is None
    stopped = gui.tell("stop")
    assert gui.sent_at("bestmove", since=stopped, seconds=2) is not None
    assert gui.last_info("nodes") > 50 or chess.Board(fen).legal_moves.count() == 1

    gui.tell("go infinite")
    time.sleep(0.2)
    gui.close(last=None)
    assert len(gui.lines_sent("bestmove")) == 2

def test_go_infinite_runs_the_simulations_by_default(monkeypatch):
    gui = Gui(monkeypatch, lambda: EngineSearch(cpu_batches(BoardModel()), 16, smart_pruning=False), simulations=300)
    gui.tell("go infinite")
    assert gui.sent_at("bestmove", seconds=10) is not None and gui.last_info("nodes") == 300
    gui.close()

#the engine declares no Ponder option; a go ponder from a gui that sends one anyway is searched as a go on the clock, and ponderhit is ignored
def test_go_ponder_is_an_ordinary_go(monkeypatch):
    searches, sent, _ = run_uci(monkeypatch, ["position startpos", "go ponder wtime 10000 btime 10000", "ponderhit", "go ponder movetime 500", "quit"])

    assert [(nodes, seconds) for _, nodes, seconds, _ in searches] == [(None, pytest.approx(Clock().limits("wtime 10000".split(), chess.Board(), 800)[1])),
                                                                         (None, pytest.approx(0.5 - MOVE_OVERHEAD_SECONDS))]
    assert len([line for line in sent if line.startswith("bestmove")]) == 2

#MoveOverhead reaches the next go's time; an option set once the search is built, or to a value it could not take, stays as it was and the gui is told why
def test_the_options_reach_the_search(monkeypatch):
    script = ["uci", "setoption name cachepositions value 1000", "setoption name RamLimitMb value 64", "setoption name SyzygyProbeLimit value 6",
              "setoption name AdvanceThreads value 12", "setoption name AdvanceThreads value 257", "setoption name GcThreads value 6", "setoption name GcThreads value 65",
              "setoption name MoveOverhead value 500", "setoption name MoveOverhead value -1", "setoption name SyzygyProbeLimit value 8",
              "setoption name Hash value 16", "isready", "setoption name CachePositions value 5", "go movetime 1000",
              "setoption name MoveOverhead value 100", "go movetime 1000", "ucinewgame", "go movetime 1000", "quit"]
    searches, sent, built = run_uci(monkeypatch, script, search_options={"smart_pruning": False})

    assert [line.split()[2] for line in sent if line.startswith("option name")] == ["SyzygyPath", "SyzygyProbeLimit", "MoveOverhead", "CachePositions",
                                                                                     "RamLimitMb", "AdvanceThreads", "GcThreads", "FirstMoveBonus",
                                                                                     "BookPlyBonus", "SearchSettings"]
    assert len(built) == 1 and built[0][3] is False
    assert built[0][4] == {"smart_pruning": False, "tablebase_pieces": 6, "syzygy_path": None, "cache_positions": 1000, "memory_limit_mb": 64,
                           "advance_threads": 12, "gc_threads": 6}
    assert [seconds for _, _, seconds, _ in searches] == [pytest.approx(0.5), pytest.approx(0.9), pytest.approx(0.9)]
    told = [line for line in sent if line.startswith("info string")]
    assert len(told) == 6 and "AdvanceThreads" in told[0] and "GcThreads" in told[1] and "-1" in told[2] and "8" in told[3] and "Hash" in told[4]
    assert "CachePositions" in told[5]
    assert "option name GcThreads type spin default 2 min 0 max 64" in sent

#values the engine could not take (a device that is not there, one leaf on several gpus, an unknown setting) leave the options as they were
def test_the_competition_options_reach_the_search(monkeypatch):
    script = ["uci", "setoption name GPUs value 0,1,2", "setoption name GPUs value 0,7", "setoption name Leaves value 32",
              "setoption name SmartPruningFactor value 0", "setoption name SearchSettings value fpu_reduction=0.3",
              "setoption name SearchSettings value nope=1", "setoption name FirstMoveBonus value 1.8", "setoption name BookPlyBonus value x",
              "isready", "setoption name Leaves value 16", "position startpos", "go " + " ".join(CLOCK_60), "ucinewgame", "position startpos",
              "go " + " ".join(CLOCK_60), "quit"]
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    searches, sent, built = run_uci(monkeypatch, script, gpu=True, gpus=[0], search_options={"settings": {"c_puct": 0.7}})

    names = [line.split()[2] for line in sent if line.startswith("option name")]
    assert names[7:] == ["Leaves", "GPUs", "SmartPruningFactor", "FirstMoveBonus", "BookPlyBonus", "SearchSettings"]
    assert "option name GPUs type string default 0" in sent
    (_, _, leaves, _, options, _, _, gpus), = built
    assert (leaves, gpus) == (32, [0, 1, 2])
    assert options["smart_pruning"] is False and options["settings"] == {"c_puct": 0.7, "fpu_reduction": 0.3}
    first = pytest.approx(2.8 * Clock().limits(CLOCK_60, chess.Board(), 800)[1])
    assert [seconds for _, _, seconds, _ in searches] == [first, first]
    told = [line for line in sent if line.startswith("info string")]
    assert len(told) == 4 and "device 7" in told[0] and "nope" in told[1] and "BookPlyBonus" in told[2] and "Leaves" in told[3]

def test_the_search_takes_its_cache_size_memory_limit_advance_and_gc_threads(monkeypatch):
    made = []
    searches = selfplay_rs.Searches
    monkeypatch.setattr(selfplay_rs, "Searches", lambda **settings: made.append(settings) or searches(**settings))
    EngineSearch(cpu_batches(BoardModel()), 16)
    limited = EngineSearch(cpu_batches(BoardModel()), 16, smart_pruning=False, cache_positions=1000, memory_limit_mb=1, advance_threads=12, gc_threads=3)
    assert ([(settings["cache_positions"], settings["memory_limit_mb"], settings["advance_threads"], settings["gc_threads"]) for settings in made]
            == [(engine_search.CACHE_POSITIONS, 0, engine_search.ADVANCE_THREADS, engine_search.GC_THREADS), (1000, 1, 12, 3)])
    report = limited.search(chess.Board(), simulations=100_000)
    assert 100 < report.simulations < 3000 and report.line[0] in chess.Board().legal_moves

#TCEC's two engines share one box: with no advance threads advance does nothing, and the next search's start carries the kept tree down both moves
@pytest.mark.parametrize("concurrent_descents", [False, True])
def test_no_advance_threads_carry_the_tree_on_at_the_next_search(concurrent_descents):
    search = EngineSearch(cpu_batches(BoardModel()), 16, smart_pruning=False, advance_threads=0, concurrent_descents=concurrent_descents)
    board = chess.Board()
    report = search.search(board, simulations=3000)
    board.push(report.line[0])

    assert search.advance(board) is False
    board.push(report.line[1])
    assert search.search(board, simulations=1).simulations > 1

def test_a_syzygy_path_reaches_the_search(monkeypatch, tablebase, tmp_path, model_path):
    (tmp_path / "no_tables").mkdir()
    gui = Gui(monkeypatch, None, model_path)
    gui.tell(f"setoption name SyzygyPath value {tmp_path / 'no_tables'}")
    assert gui.sent_at("info string", seconds=5) is not None and gui.lines_sent("info string")[0].endswith("the tables stay as they were")
    gui.tell(f"setoption name SyzygyPath value {SYZYGY_DIR}")
    gui.tell("position fen k7/8/1K6/8/8/8/8/7R w - - 0 1")
    gui.tell("go nodes 50")
    assert gui.sent_at("bestmove", seconds=30) is not None
    assert gui.lines_sent("bestmove") == ["bestmove h1h8"] and gui.last_info("nodes") == 0
    #the refused path, then what the player was built with
    assert [line.split("; ")[1] for line in gui.lines_sent("info string")[1:]] == ["145 wdl and 145 dtz tables, used up to 5 pieces"]
    gui.close()

def test_a_bad_position_leaves_the_last_one(monkeypatch):
    bad = ["startpos moves e2e4 e7e4", "startpos moves e2e4 e7", "fen not a fen", "fen 8/8/8/8/8/8/8/8 w - - 0 1", "", "kiwipete",
           "startpos moves e2e4 0000", "fen 4k3/8/8/8/8/8/PPPPPPPP/QQ2K3 w - - 0 1"]
    script = ["position startpos moves e2e4", *(f"position {position}" for position in bad), "go nodes 10",
              "position fen 4R1k1/5ppp/8/8/8/8/5PPP/6K1 b - - 1 1", "go nodes 10", "quit"]
    searches, sent, _ = run_uci(monkeypatch, script)

    after_e4 = chess.Board()
    after_e4.push_uci("e2e4")
    assert [fen for fen, *_ in searches] == [after_e4.fen()]
    assert len([line for line in sent if line.startswith("info string") and line.endswith("the last position stays")]) == len(bad)
    assert [line for line in sent if line.startswith("bestmove")][1] == "bestmove 0000"

#parse_position refuses the fens the rust search refuses (python-chess's status, and shakmaty's promoted material, which that leaves out);
#a fen whose castling rights or en passant square python-chess cleans both take
@pytest.mark.parametrize("fen", [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w KQkq - 0 1", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e3 0 1",
    "4k3/8/8/8/8/8/8/8 w - - 0 1", "4k3/8/8/8/8/8/8/K3K3 w - - 0 1", "4k3/8/8/8/8/8/8/4R1K1 w - - 0 1", "P3k3/8/8/8/8/8/8/4K3 w - - 0 1",
    "4k3/8/8/8/8/P7/PPPPPPPP/4K3 w - - 0 1", "4k3/8/5N2/1B6/8/8/8/4R1K1 b - - 0 1", "4k3/8/8/8/1NNNNNNN/8/PPPPPPPP/RNB1K3 w - - 0 1",
    "4k3/8/8/8/8/8/PPPPPPPP/QQ2K3 w - - 0 1", "4k3/8/8/8/8/8/1PPPPPPP/QQ2K3 w - - 0 1", "4k3/8/8/8/8/8/PPPPPPPP/B1B1K3 w - - 0 1",
    "4k3/8/8/8/8/8/PPPPPPPP/BB2K3 w - - 0 1", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
])
def test_the_positions_refused_are_the_searchs(fen):
    refused, searched = True, False
    with suppress(ValueError):
        uci.parse_position(["fen", *fen.split()])
        refused = False
    with suppress(ValueError):
        selfplay_rs.Searches(**rust_search_settings(False)).start(0, chess.Board(fen).root().fen(), [], 10)
        searched = True
    assert refused != searched

def test_a_long_search_tells_the_gui_how_it_stands(monkeypatch):
    monkeypatch.setattr(engine_search, "INFO_SECONDS", 0.05)
    told = []
    report = EngineSearch(cpu_batches(BoardModel()), 16, smart_pruning=False).search(chess.Board(), seconds=0.5, info=told.append)
    nodes = [running.simulations for running in told]
    assert len(told) >= 3 and nodes == sorted(nodes) and nodes[-1] <= report.simulations
    assert all(running.line and running.line[0] in chess.Board().legal_moves for running in told)

    gui = Gui(monkeypatch, lambda: EngineSearch(cpu_batches(BoardModel()), 16, smart_pruning=False))
    gui.tell("position startpos")
    went = gui.tell("go movetime 500")
    assert gui.sent_at("bestmove", since=went, seconds=5) is not None and len(gui.lines_sent("info depth")) >= 4
    gui.close()

def test_a_rounds_collisions_are_at_most_its_leaves():
    search = EngineSearch(cpu_batches(BoardModel()), 128, smart_pruning=False)
    search.searches = recorded = RoundsRecorded(search.searches)

    assert search.search(chess.Board(), simulations=6000).simulations == 6000
    rounds = [(leaves, collisions) for leaves, collisions, _ in recorded.rounds]
    assert all(collisions == min(64, leaves) for leaves, collisions in rounds)
    assert min(rounds) == (1, 1) and max(rounds) == (128, 64), rounds

def test_concurrent_descents_on_one_network_are_the_plain_search(middlegame):
    for way in ("plain", "advance", "drop"):
        reports, replies, boards, advanced, _ = advancing_game(BoardModel(), middlegame.copy(), 8, way)
        concurrent = advancing_game(BoardModel(), middlegame.copy(), 8, way, replies, concurrent_descents=True, gc_threads=0)
        assert concurrent[0] == reports and concurrent[2] == boards and concurrent[3] == advanced

    model = shuffling_model()
    for options in ({}, {"settings": {"certainty": True}}):
        plain = advancing_game(model, chess.Board(), len(SHUFFLES), "advance", SHUFFLES, **options)
        concurrent = advancing_game(model, chess.Board(), len(SHUFFLES), "advance", SHUFFLES, concurrent_descents=True, gc_threads=0, **options)
        assert concurrent[0] == plain[0] and concurrent[3] == plain[3] == len(SHUFFLES)

def test_concurrent_descents_start_afresh_where_the_tree_does_not_follow():
    search = EngineSearch(cpu_batches(BoardModel()), 16, smart_pruning=False, concurrent_descents=True)
    board = chess.Board()
    report = search.search(board, simulations=300)
    played, other = board.copy(), board.copy()
    played.push(report.line[0])
    assert search.advance(played)
    other.push(next(move for move in board.legal_moves if move != report.line[0]))
    other.push(next(iter(other.legal_moves)))
    assert search.search(other, simulations=50).line[0] in other.legal_moves
    search.new_game()
    assert search.search(board, simulations=300) == report

@pytest.mark.parametrize("networks", [2, 3, 4])
def test_concurrent_descents_on_several_networks_search_one_tree(networks):
    group = SlowNetworks(networks, 0.002, BoardModel())
    search = EngineSearch(group.networks, 16, smart_pruning=False, concurrent_descents=True)
    board = chess.Board(FREE_QUEEN)
    report = search.search(board, simulations=3000)

    assert 3000 <= report.simulations < 3000 + 3 * 16 * networks and report.line[0] in board.legal_moves
    assert group.most_out == 2 * networks and all(network.rounds > 5 for network in group.networks)
    #the timer stops a search that does not come back, which would run empty rounds for ever
    stop = threading.Timer(20, lambda: search.searches.stop(0))
    stop.start()
    again = search.search(board, simulations=report.simulations + 500)
    stop.cancel()
    assert again.simulations >= report.simulations + 500

#the dead rest of the moved tree is freed beside the next search (2 gc threads) or at its start (0), and a move down the tree costs no more than a line's length
@pytest.mark.parametrize("gc_threads", [0, 2])
def test_a_game_on_several_networks_moves_its_tree_beside_the_gc_threads(gc_threads):
    group = SlowNetworks(3, 0.002, BoardModel())
    search = EngineSearch(group.networks, 16, smart_pruning=False, concurrent_descents=True, gc_threads=gc_threads)
    board = chess.Board()
    for _ in range(6):
        report = search.search(board, simulations=2000)
        assert report.simulations >= 2000 and report.line[0] in board.legal_moves and len(report.line) > 1
        board.push(report.line[0])
        assert search.advance(board)
        board.push(report.line[1])

@pytest.mark.parametrize("settings", [{"certainty": True}, {"certainty": True, "transpositions": True}], ids=["certainty", "certainty_and_transpositions"])
def test_concurrent_descents_prove_a_mate_on_several_networks(settings):
    search = EngineSearch(SlowNetworks(3, 0.003, UniformModel()).networks, 16, concurrent_descents=True, settings=settings)
    proven = search.search(chess.Board(MATE_IN_TWO), simulations=20_000)

    assert proven.line[0].uci() in MATES_IN_TWO and proven.score == 1.0 and proven.simulations < 5_000

def test_concurrent_descents_want_leaves_above_1():
    with pytest.raises(ValueError, match="leaves above 1"):
        EngineSearch(cpu_batches(BoardModel()), 1, concurrent_descents=True)

def test_concurrent_descents_by_default_on_several_networks(monkeypatch):
    made, searches = [], selfplay_rs.Searches
    monkeypatch.setattr(engine_search.selfplay_rs, "Searches", lambda **settings: made.append(settings.get("concurrent_descents", False)) or searches(**settings))
    one, three = cpu_batches(BoardModel()), [cpu_batches(BoardModel()) for _ in range(3)]
    found = [EngineSearch(networks, 16, **options).concurrent_descents for networks, options in
             ((one, {}), (three, {}), (three, {"concurrent_descents": False}), (one, {"concurrent_descents": True}))]

    assert found == made == [False, True, False, True]

@gpu
@pytest.mark.parametrize("blocking_wait, settings", [(False, {}), (True, {"smart_pruning": False})])
def test_rust_rounds_with_concurrent_descents_search_as_the_python_loop(blocking_wait, settings):
    board, reports = chess.Board(), {}
    for rust_rounds in (False, True):
        search = EngineSearch(lean_network(blocking_wait), 16, rust_rounds=rust_rounds, concurrent_descents=True, gc_threads=0, **settings)
        found = []
        for move in ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]:
            found.append(search.search(board, simulations=600))
            board.push_uci(move)
        board = chess.Board()
        reports[rust_rounds] = found
    assert reports[True] == reports[False]
    assert search.rust_counts[0] > 20

@gpu
@pytest.mark.parametrize("count", [2, 4])
def test_rust_rounds_with_concurrent_descents_drive_every_network(count):
    networks = [lean_network(True) for _ in range(count)]
    search = EngineSearch(networks, 16, smart_pruning=False)
    assert search.rust_rounds and search.concurrent_descents
    board = chess.Board(FREE_QUEEN)
    report = search.search(board, simulations=3000)
    assert 3000 <= report.simulations < 3000 + 16 * count * 3 and report.line[0] in board.legal_moves
    assert search.rust_counts[0] > 40 * count // 2 and all(not lane.pending for network in networks for lane in network.lanes)
    assert search.search(board, simulations=report.simulations + 500).simulations >= report.simulations + 500

def test_concurrent_searches_end_under_stops_at_any_moment():
    evaluate = cpu_batches(BoardModel()).evaluate_boards
    searches = selfplay_rs.Searches(**engine_search_settings(False), tree_reuse=True, cache_positions=100_000, concurrent_descents=True)
    names, board, ended = itertools.count(), chess.Board(), []
    for turn in range(40):
        reports, errors, fen, moves = [], [], board.root().fen(), [move.uci() for move in board.move_stack]
        searches.start(0, fen, moves, 2**31)

        def drive():
            try:
                while not reports:
                    name = next(names)
                    boards = leaf_boards(searches.descend(16, 16, 48, round=name))
                    reports.extend(searches.finish(evaluate(boards) if len(boards) else np.zeros((0, ANSWER_WIDTH), dtype=np.float32), round=name))
            except BaseException as error:
                errors.append(error)

        def stop():
            time.sleep(0.003 + 0.002 * (turn % 7))
            while not reports:
                try:
                    stopped = searches.stop(0)
                except RuntimeError:
                    time.sleep(0.001)
                    continue
                except ValueError:
                    break
                reports.extend([stopped] if stopped else [])
                break
        threads = [threading.Thread(target=drive) for _ in range(4)] + [threading.Thread(target=stop)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors and len(reports) == 1 and reports[0][4] > 0, (turn, errors, reports)
        ended.append(reports[0][4])
        move = chess.Move.from_uci(reports[0][1][0])
        board.push(move)
        if not board.is_game_over():
            board.push(next(iter(board.legal_moves)))
        if board.is_game_over():
            board = chess.Board()
        else:
            searches.advance(0, fen, [move.uci() for move in board.move_stack])
    assert len(ended) == 40 and sum(ended) > 40 * 20

def test_transpositions_on_one_network_are_the_same_on_both_trees(middlegame):
    graph = {"settings": {"transpositions": True}}
    for way in ("plain", "advance", "drop"):
        reports, replies, boards, advanced, _ = advancing_game(BoardModel(), middlegame.copy(), 8, way, **graph)
        concurrent = advancing_game(BoardModel(), middlegame.copy(), 8, way, replies, concurrent_descents=True, gc_threads=0, **graph)
        #a plain tree that was advanced keeps the nodes the reply left behind until its next advance, and a transposition comes to them
        #again; the concurrent tree has freed them (no gc threads). The two search alike only until one does
        assert concurrent[3] == advanced and (way == "advance" or (concurrent[0], concurrent[2]) == (reports, boards))
    assert advancing_game(BoardModel(), middlegame.copy(), 1, "plain")[0] != reports[:1]

    model = shuffling_model()
    for certainty in (False, True):
        options = {"settings": {"transpositions": True, "certainty": certainty}}
        plain = advancing_game(model, chess.Board(), len(SHUFFLES), "advance", SHUFFLES, **options)
        concurrent = advancing_game(model, chess.Board(), len(SHUFFLES), "advance", SHUFFLES, concurrent_descents=True, gc_threads=0, **options)
        assert concurrent[0] == plain[0] and concurrent[3] == plain[3] == len(SHUFFLES)

def test_transpositions_on_several_networks():
    group = SlowNetworks(3, 0.002, BoardModel())
    search = EngineSearch(group.networks, 16, smart_pruning=False, concurrent_descents=True, settings={"transpositions": True})
    board = chess.Board(FREE_QUEEN)
    report = search.search(board, simulations=3000)
    assert 3000 <= report.simulations < 3000 + 3 * 16 * 3 and report.line[0] in board.legal_moves

#topologies as test_the_threads_cpus'
def test_the_helpers_cpus_follow_the_networks():
    windows = ([(96 << 20, list(range(16))), (32 << 20, list(range(16, 32)))], [[cpu, cpu + 1] for cpu in range(0, 32, 2)])
    assert engine_search.helped_cpus(2, 2, windows) == ([0, 2], [4, 6]) and engine_search.helped_cpus(8, 8, windows) == ([*range(0, 16, 2)], [*range(16, 32, 2)])
    assert engine_search.helped_cpus(3, 0, windows) == ([0, 2, 4], [])
    #too few for all: the networks keep theirs and there are none for helpers; no topology: none for either
    assert engine_search.helped_cpus(8, 30, windows) == ([*range(0, 16, 2)], []) and engine_search.helped_cpus(2, 2, ([], [])) == ([], None)

def test_helpers_need_concurrent_descents():
    with pytest.raises(ValueError, match="concurrent descents"):
        EngineSearch(cpu_batches(BoardModel()), 16, descent_helpers=2)
    with pytest.raises(ValueError, match="concurrent descents"):
        selfplay_rs.Searches(**rust_search_settings(False), gather_helpers=1)
    assert EngineSearch(cpu_batches(BoardModel()), 16).descent_helpers == 0

@pytest.mark.parametrize("networks, pin_threads", [(1, False), (3, False), (2, True)])
def test_helpers_join_the_rounds_short_of_leaves(networks, pin_threads):
    #the mate's node takes most visits and none of them needs the network: every round beside another is short of leaves
    group = SlowNetworks(networks, 0.002, UniformModel())
    search = EngineSearch(group.networks, 16, smart_pruning=False, grow_leaves=0, concurrent_descents=True, pin_threads=pin_threads, descent_helpers=2 * networks)
    report = search.search(chess.Board(MATE_IN_ONE), simulations=30_000)
    joined, found = search.searches.helper_counts()

    #a helper's descent under way when the search has its simulations adds one, as a network's
    assert 30_000 <= report.simulations <= 30_000 + 3 * networks and report.line[0].uci() == "e1e8"
    assert search.descent_helpers == 2 * networks and joined > 0 and found > 0
    assert (len(search.network_cpus), search.network_cpus[:1]) == ((networks, engine_search.thread_cpus(3 * networks)[:1]) if pin_threads else (0, []))

@gpu
def test_helpers_join_the_rust_rounds():
    search = EngineSearch(lean_network(True), 16, rust_rounds=True, concurrent_descents=True, smart_pruning=False, grow_leaves=0, descent_helpers=2)
    report = search.search(chess.Board(MATE_IN_ONE), simulations=30_000)
    assert 30_000 <= report.simulations <= 30_000 + 3 and search.searches.helper_counts()[0] > 0

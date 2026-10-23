#the uci engine's search (engine/engine_search.py): rounds of many leaves (selfplay_rs.Searches.descend), its
#deadline and node limit, smart pruning, the tree kept for the next move, the engine's own settings, and uci.py's go
#limits and clock, the commands it takes while it searches, its options and the positions it refuses
import ctypes
import gc
import io
import os
import queue
import runpy
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import chess
import chess.engine
import numpy as np
import pytest

import selfplay_rs
import torch
from torch import nn
from model.encoding import TOKENS_PER_BOARD, encode_move_to_id
from fake_models import BoardModel, CountingNetwork, UniformModel
from engine.inference import GraphedBatches, ImmediateBatches, cpu_batches, small_graph_sizes
from model.model import ChessTransformer
from engine.selfplay import ANSWER_WIDTH, rust_search_settings
from tools.checkpoint import save_checkpoint
from tools.tablebase import SYZYGY_DIR
from engine import engine_search
from engine.engine_search import EngineSearch, engine_search_settings, parse_changes
from engine.player import SearchReport
from engine import uci
from engine.uci import MOST_EXTENSION, MOST_OF_THE_TIME_LEFT, MOVE_OVERHEAD_SECONDS, RESERVE_SECONDS, Clock

REPOSITORY = Path(__file__).parent.parent


#the searches here are mcts.py's, whose constants the expectations were set by; the engine's own changes over them are
#test_the_engines_settings'
@pytest.fixture(autouse=True)
def mcts_constants(monkeypatch):
    monkeypatch.setattr(engine_search, "ENGINE_SETTINGS", {})


def leaf_boards(leaves: bytes) -> np.ndarray:
    return np.frombuffer(bytearray(leaves), dtype=np.uint8).reshape(-1, TOKENS_PER_BOARD)

#the first round finds only the root to evaluate; then a round spreads over the root's moves, and a round gathered
#while another is out takes none of its leaves again
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


#every move's logit drawn at random for each board, and its score a quarter, a half or three quarters: sums of quarters
#are exact in any order
def answers_by_board(boards: np.ndarray) -> np.ndarray:
    answers = np.zeros((len(boards), ANSWER_WIDTH), dtype=np.float32)
    for row, board in enumerate(boards):
        random = np.random.default_rng(list(board))
        answers[row, :-1] = random.normal(scale=2.0, size=ANSWER_WIDTH - 1)
        answers[row, -1] = random.integers(1, 4) / 4
    return answers

#rounds named at descend come back in any order, as on several gpus, and each round's answers reach their own leaves:
#four rounds out at once finished shuffled leave the tree that finishing them oldest first leaves, where the next round
#goes and the search's report. A name already out, or one that is not, is refused
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


#descend_into writes descend's boards into an array of bytes or of int64s (a lane's) and gives root_visits' numbers after
#the round; finish_with_root gives finish's reports and the numbers after it, None once the search is done
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


#a lean round (descend_into's legal_columns) writes rows of each board's tokens and its answer columns, its legal moves'
#padded with the expected score's; finished with those columns of the answer rows, it searches as rounds of whole rows do
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


#a search that fails settles its network (GraphedBatches.settle: its batches still on the gpu waited for, its lanes
#freed) before the error goes on
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


#Searches' own round loop (rust_rounds) makes the python loop's search, report for report and tree for tree (the next
#searches start from the kept trees), with smart pruning, gather_batch or a cap on the leaves out, waiting by event or
#by the copy
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
@pytest.mark.parametrize("blocking_wait, settings", [(False, {}), (True, {}), (False, {"gather_workers": 2, "collision_cap": 32}),
                                                     (False, {"visits_per_leaf_out": 8, "smart_pruning": False}),
                                                     (True, {"gather_workers": 2, "collision_cap": 32, "thread_affinity": "cores"})])
def test_rust_rounds_search_as_the_python_loop(blocking_wait, settings):
    board, reports = chess.Board(), {}
    for rust_rounds in (False, True):
        search = EngineSearch(lean_network(blocking_wait), 16, rust_rounds=rust_rounds, **settings)
        found = []
        for move in ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]:
            found.append(search.search(board, simulations=600))
            board.push_uci(move)
        board = chess.Board()
        reports[rust_rounds] = found
    assert reports[True] == reports[False]
    assert search.rust_counts[0] > 20 and search.rust_counts[1] > search.rust_counts[0]


#on two networks Searches' own loop runs both, rounds out on each at once
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
def test_rust_rounds_drive_every_network():
    networks = [lean_network(True), lean_network(True)]
    search = EngineSearch(networks, 16, smart_pruning=False, rust_rounds=True)
    assert search.search(chess.Board(), simulations=3000).simulations == 3000
    assert search.rust_counts[0] > 40 and all(not lane.pending for network in networks for lane in network.lanes)


#pinned, Searches' own loop runs each network's thread on its cpu while the search runs; a cpu the process may not run
#on is refused
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
def test_the_rust_rounds_threads_are_pinned():
    networks = [lean_network(True), lean_network(True)]
    search = EngineSearch(networks, 16, smart_pruning=False, rust_rounds=True, thread_affinity="ccd")
    before, seen = pinned_threads(), []
    search.search(chess.Board(), seconds=1.5, info=lambda report: seen.append(pinned_threads() - before))
    assert seen and seen[0] == Counter(search.network_cpus) and len(search.network_cpus) == 2, seen
    handles = networks[0].rust_network() | {"cpu": 100_000}
    networks[0].rust_network = lambda: handles
    with pytest.raises(ValueError, match="may run on"):
        EngineSearch(networks[0], 16, rust_rounds=True).search(chess.Board(), simulations=600)


#a cuda error in Searches' own loop comes back as python's exception, its other rounds waited for
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
def test_a_cuda_error_in_the_rust_rounds_is_raised():
    network = lean_network()
    handles = network.rust_network()
    handles["lanes"] = [{**lane, "device": 99} for lane in handles["lanes"]]
    network.rust_network = lambda: handles
    search = EngineSearch(network, 16, rust_rounds=True)
    with pytest.raises(RuntimeError, match="cuda error"):
        search.search(chess.Board(), simulations=600)


#Searches' own round loop takes networks of lean lanes only
def test_rust_rounds_want_lean_lanes():
    with pytest.raises(ValueError, match="lean lanes"):
        EngineSearch(cpu_batches(BoardModel()), 16, rust_rounds=True)


#the cpu the calling thread runs on now
def current_cpu() -> int:
    if sys.platform == "win32":
        return ctypes.windll.kernel32.GetCurrentProcessorNumber()
    return ctypes.CDLL(None).sched_getcpu()


class ThreadEntry(ctypes.Structure):
    _fields_ = [("size", ctypes.c_uint32), ("usage", ctypes.c_uint32), ("thread", ctypes.c_uint32), ("process", ctypes.c_uint32),
                ("base_priority", ctypes.c_int32), ("delta_priority", ctypes.c_int32), ("flags", ctypes.c_uint32)]


class GroupAffinity(ctypes.Structure):
    _fields_ = [("mask", ctypes.c_size_t), ("group", ctypes.c_uint16), ("reserved", ctypes.c_uint16 * 3)]


#the cpus of this process's threads that may run on one cpu only, one entry a thread
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


#thread_cpus, the largest L3 first: "ccd" an L3's physical cores, then their smt siblings, then the next L3's; "cores" a
#physical core a thread, then the next L3's cores, then the smt siblings. On the 9175F's one core an L3 (linux numbers a
#core's siblings k and k + 32) in the os's order. None for "none", no threads or no topology, and a policy it does not
#know is refused
def test_the_threads_cpus(monkeypatch):
    zen = ([(96 << 20, list(range(16))), (32 << 20, list(range(16, 32)))], [[cpu, cpu + 1] for cpu in range(0, 32, 2)])
    assert engine_search.thread_cpus("ccd", 11, zen) == [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5]
    assert engine_search.thread_cpus("ccd", 18, zen)[16:] == [16, 18]
    assert engine_search.thread_cpus("cores", 11, zen) == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    assert engine_search.thread_cpus("cores", 18, zen)[16:] == [1, 3]
    flipped = ([(32 << 20, list(range(16))), (96 << 20, list(range(16, 32)))], zen[1])
    assert engine_search.thread_cpus("ccd", 3, flipped) == engine_search.thread_cpus("cores", 3, flipped) == [16, 18, 20]
    epyc = ([(32 << 20, [core, core + 32]) for core in range(32)], [[core, core + 32] for core in range(32)])
    assert engine_search.thread_cpus("ccd", 4, epyc) == [0, 32, 1, 33]
    assert engine_search.thread_cpus("cores", 34, epyc) == [*range(32), 32, 33]
    assert engine_search.thread_cpus("none", 4, zen) == [] and engine_search.thread_cpus("cores", 0, zen) == []
    with pytest.raises(ValueError, match="thread_affinity"):
        engine_search.thread_cpus("spread", 4, zen)
    monkeypatch.setattr(engine_search.selfplay_rs, "cpu_topology", lambda: None)
    assert engine_search.thread_cpus("cores", 4) == []


#a network recording the cpu each launch came from
class CpuRecording:
    def __init__(self, network):
        self.network, self.cpus = network, []

    def launch(self, boards: np.ndarray):
        self.cpus.append(current_cpu())
        return self.network.launch(boards)

    def collect(self, launched):
        return self.network.collect(launched)


#pinned threads search as the os's do, report for report over a game's moves (the tree kept), the network's thread on
#its cpu for every round, the gather workers on the cpus before it. torch's openmp unpins a thread at its first parallel
#region (the cpu model's here), which the next search's pin undoes: the first search's rounds are not counted
@pytest.mark.parametrize("policy", ["ccd", "cores"])
def test_pinning_changes_no_search(policy):
    board, reports = chess.Board(), {}
    for affinity in ("none", policy):
        network = CpuRecording(cpu_batches(BoardModel()))
        before = pinned_threads()
        search = EngineSearch(network, 16, gather_workers=3, collision_cap=32, thread_affinity=affinity)
        workers = pinned_threads() - before
        found = []
        for move in ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]:
            found.append(search.search(board, simulations=600))
            board.push_uci(move)
            if len(found) == 1:
                network.cpus.clear()
        board = chess.Board()
        reports[affinity] = found
    cpus = engine_search.thread_cpus(policy, 3)
    assert reports[policy] == reports["none"]
    assert search.network_cpus == cpus[2:] and set(network.cpus) == set(cpus[2:]), (search.network_cpus, set(network.cpus))
    assert workers == Counter(cpus[:2]), workers


#the gather workers' cpus: one for each but the caller, each one the process may run on
def test_gather_workers_cpus_are_checked():
    settings = engine_search_settings(False)
    with pytest.raises(ValueError, match="a cpu for each but the caller"):
        selfplay_rs.Searches(**settings, gather_workers=3, gather_cpus=[0])
    with pytest.raises(ValueError, match="may run on"):
        selfplay_rs.Searches(**settings, gather_workers=2, gather_cpus=[100_000])


#lean lanes by compute capability: the gpus of LEAN_CAPABILITIES, today's lanes on the others
@pytest.mark.parametrize("capability, lean", [((12, 0), True), ((8, 0), False), ((8, 6), False), ((9, 0), False)])
def test_lean_lanes_by_capability(monkeypatch, capability, lean):
    monkeypatch.setattr(engine_search.torch.cuda, "get_device_capability", lambda device=None: capability)
    assert engine_search.lean_lanes() == lean


#descend_into refuses an array without room for a round's boards, or one it cannot write, before it descends
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


#Searches' calls release the gil while they work, so that a thread driving another gpu runs meanwhile: a python thread
#counts on through a long descend (a round of 400 searches), where it stood still while the call held the gil
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


#a position reached again by another order of moves takes its answer from the cache: the same search with fewer
#network calls
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


#one leaf a round is mcts.py's search (Searches' defaults, test_selfplay_rs.py) at every move, as evaluate_elo's logged
#results were played, whatever the engine's own settings: a second round out on the network would change it. From a
#middlegame, where captures move the material count, so that an answer changes the next choice
def test_one_leaf_a_round_is_the_exact_search_at_every_move(monkeypatch, carlsen_games):
    monkeypatch.setattr(engine_search, "ENGINE_SETTINGS", {"c_puct": 0.5, "fpu_reduction": 0.25})
    evaluate = cpu_batches(BoardModel()).evaluate_boards
    search = EngineSearch(cpu_batches(BoardModel()), 1)
    board = carlsen_games[0].board()
    for move in list(carlsen_games[0].mainline_moves())[:20]:
        board.push(move)
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


#the engine's settings are mcts.py's with its own changes over them, and a caller's over those
def test_the_engines_settings(monkeypatch):
    monkeypatch.setattr(engine_search, "ENGINE_SETTINGS", {"c_puct": 0.5, "fpu_reduction": 0.25})
    assert engine_search_settings(False) == {**rust_search_settings(False), "fpu_reduction": 0.25, "c_puct": 0.5}
    assert engine_search_settings(False, {"c_puct": 0.7})["c_puct"] == 0.7
    assert engine_search_settings(False, own=False) == rust_search_settings(False)


#every position's answer: no move preferred, expected score 0.6
def answers_of_0_6(boards: np.ndarray) -> np.ndarray:
    answers = np.zeros((len(boards), ANSWER_WIDTH), dtype=np.float32)
    answers[:, -1] = 0.6
    return answers


#the network's value goes into the search and the cache sharpened (its logit times value_sharpness), so a position the
#cache answers is not sharpened again; 1 leaves the value as it was
@pytest.mark.parametrize("sharpness, expected", [(1.0, 0.6), (3.0, 1 / (1 + (0.4 / 0.6) ** 3))])
def test_the_value_is_sharpened_once(sharpness, expected):
    searches = selfplay_rs.Searches(**rust_search_settings(False), cache_positions=1000, value_sharpness=sharpness)
    scores = []
    for _ in range(2):
        searches.start(0, chess.STARTING_FEN, [], 1)
        scores.append(searches.finish(answers_of_0_6(leaf_boards(searches.descend())))[0][2])
    assert scores == pytest.approx([expected, expected])


#the engine reports its score on the network's own scale, as before the value was sharpened
def test_the_engine_reports_the_score_unsharpened():
    search = EngineSearch(ImmediateBatches(answers_of_0_6), 16, settings={"value_sharpness": 3.0})
    assert search.search(chess.Board(), simulations=1).score == pytest.approx(0.6)


#with leaves out on the network, rounds stop at the search's simulations (without smart pruning, which stops sooner)
@pytest.mark.parametrize("simulations", [1, 37, 300])
def test_a_search_of_many_leaves_a_round_stops_at_its_simulations(carlsen_games, simulations):
    board = carlsen_games[1].board()
    for move in list(carlsen_games[1].mainline_moves())[:10]:
        board.push(move)
    search = EngineSearch(cpu_batches(BoardModel()), 16)
    search.smart_pruning = False

    report = search.search(board, simulations=simulations)

    assert report.simulations == simulations and report.line[0] in board.legal_moves


class SizesNetwork:
    def __init__(self, model):
        self.network = cpu_batches(model)
        self.sizes = []

    def launch(self, boards: np.ndarray) -> np.ndarray:
        self.sizes.append(len(boards))
        return self.network.launch(boards)

    def collect(self, launched: np.ndarray) -> np.ndarray:
        return self.network.collect(launched)


#a fresh tree's rounds start at one leaf and grow to 64, still stopping at the simulations; fixed rounds are large at once
def test_a_rounds_leaves_grow_with_the_search(carlsen_games):
    board = carlsen_games[1].board()
    for move in list(carlsen_games[1].mainline_moves())[:10]:
        board.push(move)
    growing, fixed = SizesNetwork(BoardModel()), SizesNetwork(BoardModel())
    growing_search, fixed_search = EngineSearch(growing, 64, grow_leaves=16), EngineSearch(fixed, 64, grow_leaves=0)
    growing_search.smart_pruning = fixed_search.smart_pruning = False

    report = growing_search.search(board, simulations=3000)
    fixed_search.search(board, simulations=3000)

    assert report.simulations == 3000
    assert growing.sizes[:4] == [1, 1, 1, 1] and max(growing.sizes) == 64
    assert fixed.sizes[1] > 16 and len(growing.sizes) > len(fixed.sizes)


#a leaf per 16 simulations done or left (the fewer), one round out until a round takes all 64
@pytest.mark.parametrize("visits, simulations, expected", [(0, 1000, (1, 1)), (160, 1000, (10, 1)), (1024, None, (64, 2)), (995, 1000, (1, 1))])
def test_the_leaves_of_a_round(visits, simulations, expected):
    class Searches:
        def root_visits(self, game):
            return visits, 0, 0
    search = EngineSearch(cpu_batches(BoardModel()), 64, grow_leaves=16)
    search.searches = Searches()

    assert search.round_leaves(simulations) == expected


#with a cap the rounds out on all the networks hold a leaf per visits_per_leaf_out root visits at most: grow_leaves' one
#round until a round takes all its leaves, then more as the tree grows, one at least; without (the default) the networks'
#two each once a round takes all its leaves
@pytest.mark.parametrize("visits, cap, networks, expected", [(1024, 0, 4, (64, 8)), (1024, 16, 4, (64, 1)), (5000, 16, 4, (64, 4)),
                                                             (100_000, 16, 4, (64, 8)), (160, 16, 1, (10, 1)), (50, 64, 1, (3, 1))])
def test_a_cap_keeps_the_leaves_out_to_the_trees_size(visits, cap, networks, expected):
    class Searches:
        def root_visits(self, game):
            return visits, 0, 0
    search = EngineSearch([cpu_batches(BoardModel())] * networks, 64, grow_leaves=16, visits_per_leaf_out=cap)
    search.searches = Searches()

    assert search.round_leaves(None) == expected


#a free queen, which the material count finds, and mates in one, which a flat eval finds by searching every move
@pytest.mark.parametrize("model, fen, best", [
    (BoardModel(), "4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1", "e4d5"),
    (UniformModel(), "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", "e1e8"),
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


#a network with lanes, as a wrapper passes on GraphedBatches'
class Laned:
    def __init__(self, network, lanes: list):
        self.network, self.lanes = network, lanes

    def launch(self, boards: np.ndarray):
        return self.network.launch(boards)

    def collect(self, launched):
        return self.network.collect(launched)


#the deadline counts rounds as coming back together only when they run at once: many leaves a round on two lanes or more
def test_rounds_run_at_once_only_on_lanes_of_their_own():
    assert EngineSearch(Laned(cpu_batches(BoardModel()), [0, 0]), 4).rounds_at_once
    assert not EngineSearch(Laned(cpu_batches(BoardModel()), [0, 0]), 1).rounds_at_once
    assert not EngineSearch(Laned(cpu_batches(BoardModel()), [0]), 4).rounds_at_once
    assert not EngineSearch(cpu_batches(BoardModel()), 4).rounds_at_once


#networks of the model that take a few ms to answer, waiting without the gil as a gpu's collect does: they count the
#rounds out on all of them at once, and each one's rounds. failing_collect: the collect of each network that raises
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


#on several networks (gpus) a search keeps two rounds out on each, every one of them busy, and ends at its simulations
#with the move one network finds
def test_a_search_on_several_networks_keeps_them_all_busy():
    group = SlowNetworks(3, 0.003, BoardModel())
    report = EngineSearch(group.networks, 16, smart_pruning=False).search(chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1"), simulations=3000)

    assert report.simulations == 3000 and report.line[0].uci() == "e4d5"
    assert group.most_out == 6 and all(network.rounds > 5 for network in group.networks)


#Searches counting its rounds that found leaves and those that found none
class CountingRounds:
    def __init__(self, searches):
        self.searches = searches
        self.empty = self.with_leaves = 0

    def __getattr__(self, name):
        return getattr(self.searches, name)

    def descend_into(self, *arguments, **keywords):
        written, root = self.searches.descend_into(*arguments, **keywords)
        if written:
            self.with_leaves += 1
        else:
            self.empty += 1
        return written, root


#a network whose round finds no leaves, every descent ending on a leaf already out (all of a fresh tree's), waits for
#another network's round to come back rather than holding Searches with empty rounds meanwhile (1,398 of them against
#54 with leaves, on 5 ms networks)
def test_a_network_without_leaves_waits_for_the_others():
    search = EngineSearch(SlowNetworks(3, 0.005, BoardModel()).networks, 64, smart_pruning=False, grow_leaves=0)
    search.searches = rounds = CountingRounds(search.searches)

    assert search.search(chess.Board(), simulations=4000).simulations == 4000
    assert rounds.empty < rounds.with_leaves, (rounds.empty, rounds.with_leaves)


#the cap holds in a search on several networks: at 3,000 simulations and a leaf out per 64 root visits, two rounds of 16
#are out at most, where the three networks take six without it
def test_a_cap_holds_on_several_networks():
    group = SlowNetworks(3, 0.003, BoardModel())
    search = EngineSearch(group.networks, 16, smart_pruning=False, visits_per_leaf_out=64)
    report = search.search(chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1"), simulations=3000)

    assert report.simulations == 3000 and group.most_out == 2


#one network in a list is the search on that network; one leaf a round, mcts.py's search, is for one network only
def test_one_network_in_a_list_is_the_search_on_it(carlsen_games):
    board = carlsen_games[1].board()
    for move in list(carlsen_games[1].mainline_moves())[:10]:
        board.push(move)
    alone, listed = (EngineSearch(network, 16).search(board, simulations=2000) for network in (cpu_batches(BoardModel()), [cpu_batches(BoardModel())]))

    assert (alone.line, alone.score, alone.simulations, alone.root_moves) == (listed.line, listed.score, listed.simulations, listed.root_moves)
    with pytest.raises(ValueError):
        EngineSearch([cpu_batches(BoardModel())] * 2, 1)


#on several networks a timed search ends by its deadline, and smart pruning stops a decided mate early, as on one
def test_a_search_on_several_networks_keeps_to_its_deadline_and_pruning():
    search = EngineSearch(SlowNetworks(3, 0.003, UniformModel()).networks, 16)
    started = time.perf_counter()
    timed = search.search(chess.Board(), seconds=0.3)
    seconds = time.perf_counter() - started
    mate = EngineSearch(SlowNetworks(3, 0.003, UniformModel()).networks, 16).search(chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1"), simulations=4000)
    (best, most), (_, second) = mate.root_moves[:2]

    assert timed.simulations > 300 and seconds < 0.3 + 0.1
    assert best.uci() == "e1e8" and mate.simulations < 2200 and most - second > 4000 - mate.simulations


#a network that fails ends the search with its error once the other networks' rounds are back, rather than leaving them
#running or waiting for it
def test_a_failing_network_ends_the_search_with_its_error():
    search = EngineSearch(SlowNetworks(3, 0.003, BoardModel(), failing_collect=5).networks, 16, smart_pruning=False)
    started = time.perf_counter()
    with pytest.raises(RuntimeError, match="fell over"):
        search.search(chess.Board(), simulations=100_000)
    assert time.perf_counter() - started < 2


#uci's stop reaches a search on several networks: told to stop 0.2 s in, a search of far more simulations reports soon
#after
def test_stop_ends_a_search_on_several_networks():
    search = EngineSearch(SlowNetworks(3, 0.003, BoardModel()).networks, 16, smart_pruning=False)
    stop = threading.Event()
    threading.Timer(0.2, stop.set).start()
    started = time.perf_counter()
    report = search.search(chess.Board(), simulations=300_000, stop=stop)

    assert 0.2 <= time.perf_counter() - started < 0.2 + 0.1 and report.simulations > 100


#a search on several networks tells how it stands while it runs as one network's does, its nodes growing
def test_a_search_on_several_networks_tells_how_it_stands(monkeypatch):
    monkeypatch.setattr(engine_search, "INFO_SECONDS", 0.05)
    told = []
    report = EngineSearch(SlowNetworks(3, 0.003, BoardModel()).networks, 16, smart_pruning=False).search(chess.Board(), seconds=0.5, info=told.append)
    nodes = [running.simulations for running in told]

    assert len(told) >= 3 and nodes == sorted(nodes) and nodes[-1] <= report.simulations


#the same position again needs only the simulations it lacks; the reply to the move played is searched from the kept
#tree; another game's position gets a tree of its own
def test_the_next_search_carries_on_the_last_tree():
    network = CountingNetwork(BoardModel())
    search = EngineSearch(network, 16)
    search.smart_pruning = False
    board = chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")
    first = search.search(board, simulations=400)

    network.boards = 0
    assert search.search(board, simulations=600).simulations == 600
    assert 0 < network.boards <= 200

    board.push(first.line[0])
    board.push(first.line[1])
    network.boards = 0
    assert search.search(board, simulations=400).simulations == 400
    assert network.boards < 400

    other = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
    network.boards = 0
    report = search.search(other, simulations=100)
    assert report.line[0] in other.legal_moves and network.boards > 50


#the side to move's clock, milliseconds as seconds, and --simulations when go sets no limit (go infinite)
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


#the (nodes, seconds) each of the gos from the start position gave a search, and the lines the engine sent
def searched_gos(monkeypatch, gos: list[str]) -> tuple[list, list[str]]:
    given, sent = [], []
    class Player:
        def search(self, board, nodes, seconds, extension, stop=None, info=None):
            given.append((nodes, seconds))
            return SearchReport(0.5, [next(iter(board.legal_moves))], 1, [])
    monkeypatch.setattr(uci, "load_player", lambda *arguments: Player())
    monkeypatch.setattr(uci, "send", sent.append)
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(["position startpos", *gos, "quit"])))
    uci.main(Path("unused.pt"), selection="mcts", simulations=800)
    return given, sent

#go nodes 0 (or less) plays at once, a search of one simulation, as lc0 answers at once (Searches refused 0, which
#ended the engine)
def test_go_nodes_0_plays_at_once(monkeypatch):
    given, sent = searched_gos(monkeypatch, ["go nodes 0", "go nodes -5", "go nodes 7"])
    assert given == [(1, None), (1, None), (7, None)] and len([line for line in sent if line.startswith("bestmove")]) == 3


#a go number that is not a whole number is left out, the gui told so (int() raised, which ended the engine): that go is
#searched as if it had not come with it
def test_go_leaves_out_numbers_that_are_not_whole(monkeypatch):
    given, sent = searched_gos(monkeypatch, ["go nodes x", "go movetime 1.5 nodes 30", "go wtime 1000.5 btime 1000"])
    assert given == [(800, None), (30, None), (800, None)] and len([line for line in sent if line.startswith("bestmove")]) == 3
    told = [line for line in sent if line.startswith("info string")]
    assert told == [f"info string go's {number} is not a whole number: left out" for number in ("nodes x", "movetime 1.5", "wtime 1000.5")]


#the start position with pieces taken off, the kings kept, until `men` are left
def board_with_men(men: int) -> chess.Board:
    board = chess.Board()
    for square in [square for square in chess.SQUARES if board.piece_type_at(square) not in (None, chess.KING)][:32 - men]:
        board.remove_piece_at(square)
    return board


#what a search is given and does not use (smart pruning stopped it early) is banked, and the next move gets all of it;
#a movetime search is not on the clock, and no move gets the reserve
def test_unused_time_goes_to_the_next_move():
    clock, board = Clock(), chess.Board()
    _, first = clock.limits("wtime 60000 btime 60000".split(), board, 800)
    clock.searched(first / 2, 0)
    _, second = clock.limits("wtime 59000 btime 60000".split(), board, 800)
    _, without_saving = Clock().limits("wtime 59000 btime 60000".split(), board, 800)
    assert second == pytest.approx(clock.share + first / 2) and second > 1.4 * without_saving

    #a search that takes all it is given leaves the bank empty
    clock.searched(second, 0)
    _, third = clock.limits("wtime 58000 btime 60000".split(), board, 800)
    assert third == pytest.approx(Clock().limits("wtime 58000 btime 60000".split(), board, 800)[1])

    clock.searched(third / 4, 0)
    clock.limits("movetime 1000".split(), board, 800)
    clock.searched(0.1, 0)
    assert clock.saved == pytest.approx(3 / 4 * third)
    assert clock.limits("wtime 1500 btime 60000".split(), board, 800)[1] == pytest.approx(1.5 - MOVE_OVERHEAD_SECONDS - RESERVE_SECONDS)


#a game at 60+0.6 losing material as the games against BT4 do (16 men by move 40, 10 by 60, 6 by 85), every search
#taking all it is given: the middlegame moves get more than the opening's, the late middlegame starts with time to
#spend, the endgame lives on the increment, and the clock never runs below the reserve
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
    #the last 30 moves are at 6 men
    opening, middle_game, endgame = (sum(moves) / len(moves) for moves in (spent["opening"], spent["middle game"], spent["late"][-30:]))
    assert middle_game > 1.1 * opening and endgame < 0.7
    assert late_middlegame_left > 12 and low > RESERVE_SECONDS / 2


#a clock planned over fewer moves a man spends more now, and a move's cap is the part of the time left given
def test_a_clocks_plan_and_cap_can_be_set():
    board = chess.Board()
    arguments = "wtime 60000 btime 60000 winc 600 binc 600".split()
    assert Clock(moves_per_man=1.0).limits(arguments, board, 800)[1] > 1.25 * Clock().limits(arguments, board, 800)[1]
    assert Clock(most_of_the_time_left=0.01).limits(arguments, board, 800)[1] == pytest.approx((60 - MOVE_OVERHEAD_SECONDS) * 0.01)
    #late in the game the least moves left and the reserve decide the share
    board = board_with_men(4)
    arguments = "wtime 10000 btime 10000 winc 600 binc 600".split()
    assert Clock(least_moves_left=8, reserve_seconds=0.0).limits(arguments, board, 800)[1] == pytest.approx((10 - MOVE_OVERHEAD_SECONDS) / 8 + 0.6)
    assert Clock().limits(arguments, board, 800)[1] == pytest.approx((10 - MOVE_OVERHEAD_SECONDS - RESERVE_SECONDS) / 10 + 0.6)


#ucinewgame starts a fresh clock: searches that used little of their time bank it for later moves (the 8th move here
#gets over twice the first's), but the next game's first move gets what the first game's did. A search after the first
#may run on if its score falls below the last one's
def test_a_new_game_starts_a_fresh_clock(monkeypatch):
    given = []
    class Player:
        def search(self, board, nodes, seconds, extension, stop, info=None):
            given.append((seconds, extension))
            return SearchReport(0.5, [next(iter(board.legal_moves))], 1, [])
    go = ["position startpos", "go wtime 60000 btime 60000 winc 600 binc 600"]
    monkeypatch.setattr(uci, "load_player", lambda *arguments: Player())
    monkeypatch.setattr(uci, "send", lambda line: None)
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(["uci", "isready", *go * 8, "ucinewgame", *go, "quit"])))
    uci.main(Path("unused.pt"), selection="mcts")
    seconds = [move_seconds for move_seconds, _ in given]
    assert seconds[7] > 2 * seconds[0] and seconds[8] == seconds[0]
    assert given[0][1] is None and given[8][1] is None
    last_score, extra = given[1][1]
    assert last_score == 0 and extra > 0


#the start position's men with `ply` plies before it, as a book line's fen counts them
def board_at_ply(ply: int) -> chess.Board:
    board = chess.Board()
    board.turn, board.fullmove_number = ply % 2 == 0, 1 + ply // 2
    return board

LC0_FIRST_MOVE = {"first_move_bonus": 1.8, "book_ply_bonus": 0.25}
CLOCK_60 = "wtime 60000 btime 60000 winc 600 binc 600".split()

#lc0's first move bonus: the game's first search on the clock gets 1 + 1.8 + 0.25 x its ply, at most 12, times what it
#gets without (x5.8 out of a UHO book: ~9.9 s at 60+0.6 for a share of 1.7 s)
@pytest.mark.parametrize("arguments", [CLOCK_60, "wtime 180000 btime 180000 winc 2000 binc 2000".split(), "wtime 60000 btime 60000 movestogo 40".split()])
@pytest.mark.parametrize("ply, bonus", [(0, 2.8), (1, 3.05), (8, 4.8), (16, 5.8), (40, 5.8)])
def test_the_games_first_search_gets_lc0s_bonus(arguments, ply, bonus):
    board = board_at_ply(ply)
    assert Clock(**LC0_FIRST_MOVE).limits(arguments, board, 800)[1] == pytest.approx(bonus * Clock().limits(arguments, board, 800)[1])


#the bonus is the first search's own, out of the time left: what it leaves unused is banked as any search's is, and the
#next move gets that and no bonus, as from a clock without one that had banked as much. A search off the clock
#(movetime) leaves the bonus to the first on it, as lc0's does, and a falling score's extension counts in shares without it
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


#however large the bonus, the first search never gets the reserve, nor more than the time left, nor past its cap
def test_the_first_search_bonus_keeps_to_the_reserve_and_the_time_left():
    board, arguments = board_at_ply(16), "wtime 3000 btime 60000 winc 1000 binc 1000".split()
    assert Clock().limits(arguments, board, 800)[1] < 1.1
    assert Clock(**LC0_FIRST_MOVE).limits(arguments, board, 800)[1] == pytest.approx(3 - MOVE_OVERHEAD_SECONDS - RESERVE_SECONDS)
    assert Clock(reserve_seconds=0.0, **LC0_FIRST_MOVE).limits(arguments, board, 800)[1] == pytest.approx(3 - MOVE_OVERHEAD_SECONDS)
    assert Clock(most_of_the_time_left=0.25, **LC0_FIRST_MOVE).limits(arguments, board, 800)[1] == pytest.approx((3 - MOVE_OVERHEAD_SECONDS) / 4)


#a search on the clock whose score has fallen FALL_FULL_CP below the last search's runs on past its deadline by all of
#its extension's seconds, once; one whose score held stops at its deadline
def test_a_falling_score_runs_the_search_on():
    search = EngineSearch(cpu_batches(UniformModel()), 16)
    timed = {}
    for last_score in (0, engine_search.FALL_FULL_CP):
        started = time.perf_counter()
        search.search(chess.Board(), seconds=0.1, extension=(last_score, 0.3))
        timed[last_score] = time.perf_counter() - started
    assert timed[0] < 0.1 + 0.1
    assert 0.1 + 0.25 < timed[engine_search.FALL_FULL_CP] < 0.1 + 0.3 + 0.15


#smart pruning stops once the simulations left could not change the most visited move: a mate in one is decided when
#it has about half the budget, and at the stop its lead is more than the simulations left; the start with a flat
#eval is never decided
def test_a_decided_search_stops_early():
    mate = EngineSearch(cpu_batches(UniformModel()), 16).search(chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1"), simulations=4000)
    (best, most), (_, second) = mate.root_moves[:2]
    assert best.uci() == "e1e8" and mate.simulations < 2200 and most - second > 4000 - mate.simulations

    start = EngineSearch(cpu_batches(UniformModel()), 16).search(chess.Board(), simulations=400)
    assert start.simulations == 400


#uci.py --no-smart-pruning, for matches where both engines must spend all of their time: the decided mate in one
#still gets the whole budget
def test_a_search_without_smart_pruning_uses_its_whole_budget():
    mate = EngineSearch(cpu_batches(UniformModel()), 16, smart_pruning=False).search(chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1"), simulations=4000)
    assert mate.line[0].uci() == "e1e8" and mate.simulations == 4000


#with a number of simulations, the pruned search plays the move of the whole search (both from a new tree and cache)
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
        whole_search = EngineSearch(cpu_batches(BoardModel()), 16)
        whole_search.smart_pruning = False
        whole = whole_search.search(board, simulations=300)
        assert pruned.line[0] == whole.line[0] and whole.simulations == 300, board.fen()
        stopped_early += pruned.simulations < 300
    assert len(positions) > 20 and stopped_early > 5


#Searches as smart pruning sees them, scripted: after every round the root has `visits`, its two most visited moves
#`most` and `second`, and is proven or not; the search is done after 30 rounds unless stopped first, when it notes the
#time
class ScriptedSearches:
    def __init__(self, visits: int, most: int, second: int, proven: bool = False):
        self.root, self.proven = (visits, most, second), proven
        self.finishes, self.stopped_at = 0, None

    def start(self, game, fen, moves, simulations):
        pass

    def descend_into(self, boards, count, collisions, simulations, round=None, root_of=None, legal_columns=False):
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

    def report(self):
        return engine_search.GAME, ["a1a2"], 0.5, [], self.root[0]


#lc0's factor 1.33: a search of 1,000 simulations with 600 done stops once the most visited move leads by more than
#400 / 1.33 = 300.75, where 1 waits for a lead of more than 400
@pytest.mark.parametrize("factor, lead, stops", [(1.0, 400, False), (1.0, 401, True), (1.33, 300, False), (1.33, 301, True)])
def test_smart_prunings_factor(factor, lead, stops):
    search = EngineSearch(ImmediateBatches(answers_of_0_6), 16, grow_leaves=0, smart_pruning_factor=factor)
    search.searches = ScriptedSearches(600, 500, 500 - lead)
    search.search(chess.Board(), simulations=1000)
    assert (search.searches.stopped_at is not None) == stops


#a network whose every round takes 1/16 s of the test's clock
class SixteenthsNetwork:
    def __init__(self, now: list[float]):
        self.now = now

    def launch(self, boards: np.ndarray) -> np.ndarray:
        return boards

    def collect(self, boards: np.ndarray) -> np.ndarray:
        self.now[0] += 1 / 16
        return answers_of_0_6(boards)


#smart pruning stops a decided search no sooner than smart_pruning_wait after its first answers, at the first round
#since (lc0's 0.2 s; 0.25 is a round exactly); one legal move and a proven root stop at the first answers all the same
@pytest.mark.parametrize("fen, proven, wait, stopped_at", [
    (chess.STARTING_FEN, False, 0.0, 1 / 16), (chess.STARTING_FEN, False, 0.2, 5 / 16), (chess.STARTING_FEN, False, 0.25, 5 / 16),
    (chess.STARTING_FEN, True, 0.2, 1 / 16), ("k7/8/8/8/8/8/r7/K1q5 w - - 0 1", False, 0.2, 1 / 16)])
def test_smart_pruning_waits_after_the_first_answers(monkeypatch, fen, proven, wait, stopped_at):
    now = [0.0]
    monkeypatch.setattr(engine_search.time, "perf_counter", lambda: now[0])
    search = EngineSearch(SixteenthsNetwork(now), 16, grow_leaves=0, smart_pruning_wait=wait, settings={"certainty": True})
    search.searches = ScriptedSearches(600, 600, 0, proven)
    search.search(chess.Board(fen), simulations=1000)
    assert search.searches.stopped_at == stopped_at


#with certainty, a search whose proofs leave one root move not lost stops at once (lc0's), where smart pruning waits for
#the move's lead: every white move here but Rxf7 lets black mate at once
def test_a_search_with_one_move_left_stops():
    board = chess.Board("8/4Rq2/7K/5P2/5Prk/8/8/8 w - - 0 1")
    searched = {stop: EngineSearch(cpu_batches(UniformModel()), 16, settings={"certainty": True}, stop_at_one_move_left=stop).search(board, simulations=20_000)
                for stop in (False, True)}
    assert searched[True].line[0].uci() == searched[False].line[0].uci() == "e7f7"
    assert searched[True].simulations < 1000 and searched[False].simulations > 5000


#uci.py's --search-settings reach the search: the same search of the same network finds another line and score with
#c_puct 0.1 than with 10
def test_the_engine_takes_search_settings(tmp_path):
    settings = {"d_model": 32, "num_layers": 1, "n_heads": 2, "mlp_hidden": 64, "value_bins": 16}
    save_checkpoint(ChessTransformer(**settings), settings, tmp_path / "model.pt")
    found = []
    for changes in ("c_puct=0.1", "c_puct=10"):
        engine = chess.engine.SimpleEngine.popen_uci([sys.executable, "-m", "engine.uci", str(tmp_path / "model.pt"), "--select", "mcts",
                                                      "--search-settings", changes], cwd=REPOSITORY)
        try:
            played = engine.play(chess.Board(), chess.engine.Limit(nodes=400), info=chess.engine.INFO_ALL)
        finally:
            engine.quit()
        found.append((played.info["pv"], played.info["score"], played.info["nodes"]))
    assert found[0][2] == found[1][2] == 400 and found[0][:2] != found[1][:2]


#uci.py's --precision and --fp16-blocks reach the network the search runs, and are refused where no such network runs
def test_the_engine_takes_a_precision(monkeypatch, tmp_path):
    settings = {"d_model": 32, "num_layers": 1, "n_heads": 2, "mlp_hidden": 64, "value_bins": 16}
    save_checkpoint(ChessTransformer(**settings), settings, tmp_path / "model.pt")
    built = []
    monkeypatch.setattr(uci, "engine_network", lambda checkpoint, leaves, precision, fp16_blocks, device=None, blocking_wait=False: built.append((precision, fp16_blocks))
                        or cpu_batches(BoardModel()))
    search = uci.load_player(tmp_path / "model.pt", "mcts", True, 4, False, {"smart_pruning": False}, "fp16", 1)
    uci.load_player(tmp_path / "model.pt", "mcts", True, 4, False, {"smart_pruning": False}, "int8", 0)
    refused = [subprocess.run([sys.executable, "-m", "engine.uci", str(tmp_path / "model.pt"), "--select", "mcts", *options],
                              cwd=REPOSITORY, capture_output=True, text=True, input="")
               for options in (["--precision", "fp16"], ["--gpu", "--precision", "fp16", "--fp16-blocks", "0"])]

    assert built == [("fp16", 1), ("int8", 0)] and search.search(chess.Board(), simulations=20).simulations == 20
    assert refused[0].returncode == 2 and "--precision needs --select mcts --gpu" in refused[0].stderr
    assert refused[1].returncode == 2 and "--fp16-blocks needs --select mcts --gpu and the int8 network" in refused[1].stderr


#uci.py's --gpus builds the network on each of the devices, each waited for asleep, and searches on all of them, and is
#refused without the gpu search, or with one leaf a round on several
def test_the_engine_takes_gpus(monkeypatch, tmp_path):
    built = []
    monkeypatch.setattr(uci, "engine_network", lambda checkpoint, leaves, precision, fp16_blocks, device=None, blocking_wait=False: built.append((fp16_blocks, device, blocking_wait)) or cpu_batches(BoardModel()))
    search = uci.load_player(tmp_path / "model.pt", "mcts", True, 4, False, {"smart_pruning": False}, "int8", 1, [0, 2])
    refused = [subprocess.run([sys.executable, "-m", "engine.uci", "unused.pt", "--select", "mcts", *gpu, "--gpus", "0,1"], cwd=REPOSITORY,
                              capture_output=True, text=True, input="") for gpu in (["--leaves", "4"], ["--gpu", "--leaves", "1"])]

    assert built == [(1, 0, True), (1, 2, True)] and len(search.networks) == 2 and search.search(chess.Board(), simulations=200).simulations == 200
    assert all(run.returncode == 2 and "--gpus needs --select mcts --gpu" in run.stderr for run in refused)


#uci.py's --visits-per-leaf-out reaches the search, which has no cap without it
@pytest.mark.filterwarnings("ignore:'engine.uci' found in sys.modules")
@pytest.mark.parametrize("arguments, cap", [([], 0), (["--visits-per-leaf-out", "16"], 16)])
def test_the_engine_takes_a_cap_on_the_leaves_out(monkeypatch, arguments, cap):
    built = []

    class Recorded(EngineSearch):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            built.append(self.visits_per_leaf_out)
    monkeypatch.setattr(engine_search, "engine_network", lambda checkpoint, leaves, precision, fp16_blocks, device=None, blocking_wait=False: cpu_batches(BoardModel()))
    monkeypatch.setattr(engine_search, "EngineSearch", Recorded)
    monkeypatch.setattr(sys, "argv", ["uci.py", "unused.pt", "--select", "mcts", "--gpu", *arguments])
    monkeypatch.setattr(sys, "stdin", io.StringIO("isready\nquit\n"))
    runpy.run_module("engine.uci", run_name="__main__")

    assert built == [cap]


#uci.py's --gather-workers, --gather-spin-us, --collision-cap and --collision-scaling reach the search, which has none of
#them without
@pytest.mark.filterwarnings("ignore:'engine.uci' found in sys.modules")
@pytest.mark.parametrize("arguments, expected", [([], (0, 50, 0, False)), (["--gather-workers", "4", "--gather-spin-us", "500",
                                                                          "--collision-cap", "256", "--collision-scaling"], (4, 500, 256, True))])
def test_the_engine_takes_the_gathers_settings(monkeypatch, arguments, expected):
    built = []

    class Recorded(EngineSearch):
        def __init__(self, *args, gather_workers=0, gather_spin_us=50, **kwargs):
            super().__init__(*args, gather_workers=gather_workers, gather_spin_us=gather_spin_us, **kwargs)
            built.append((gather_workers, gather_spin_us, self.collision_cap, self.collision_scaling))
    monkeypatch.setattr(engine_search, "engine_network", lambda checkpoint, leaves, precision, fp16_blocks, device=None, blocking_wait=False: cpu_batches(BoardModel()))
    monkeypatch.setattr(engine_search, "EngineSearch", Recorded)
    monkeypatch.setattr(sys, "argv", ["uci.py", "unused.pt", "--select", "mcts", "--gpu", *arguments])
    monkeypatch.setattr(sys, "stdin", io.StringIO("isready\nquit\n"))
    runpy.run_module("engine.uci", run_name="__main__")

    assert built == [expected]


#uci.py's --thread-affinity reaches the search, whose threads stay unpinned without it; it is refused without the gpu
#search
@pytest.mark.filterwarnings("ignore:'engine.uci' found in sys.modules")
@pytest.mark.parametrize("arguments, policy", [([], "none"), (["--thread-affinity", "cores"], "cores")])
def test_the_engine_takes_a_thread_affinity(monkeypatch, arguments, policy):
    built = []

    class Recorded(EngineSearch):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            built.append(self.network_cpus)
    monkeypatch.setattr(engine_search, "engine_network", lambda checkpoint, leaves, precision, fp16_blocks, device=None, blocking_wait=False: cpu_batches(BoardModel()))
    monkeypatch.setattr(engine_search, "EngineSearch", Recorded)
    monkeypatch.setattr(sys, "argv", ["uci.py", "unused.pt", "--select", "mcts", "--gpu", *arguments])
    monkeypatch.setattr(sys, "stdin", io.StringIO("isready\nquit\n"))
    runpy.run_module("engine.uci", run_name="__main__")
    refused = subprocess.run([sys.executable, "-m", "engine.uci", "unused.pt", "--select", "mcts", "--thread-affinity", "cores"], cwd=REPOSITORY,
                             capture_output=True, text=True, input="")

    assert built == [engine_search.thread_cpus(policy, 1)] and len(built[0]) == (policy == "cores")
    assert refused.returncode == 2 and "--thread-affinity needs --select mcts --gpu" in refused.stderr


#uci.py started without --precision builds the engine's network in int8 with the block whose int8 costs the policy most
#kept in fp16 besides the last, and --fp16-blocks 0 every other block in int8 (self-play and labels stay fp16: they never
#build it), spinning while it waits for the gpu unless --blocking-wait
@pytest.mark.filterwarnings("ignore:'engine.uci' found in sys.modules")
def test_the_engine_plays_in_int8_unless_told_otherwise(monkeypatch):
    built = []
    monkeypatch.setattr(engine_search, "engine_network", lambda checkpoint, leaves, precision, fp16_blocks, device=None, blocking_wait=False:
                        built.append((precision, fp16_blocks, blocking_wait)) or cpu_batches(BoardModel()))
    for options in ([], ["--fp16-blocks", "0"], ["--blocking-wait"]):
        monkeypatch.setattr(sys, "argv", ["uci.py", "unused.pt", "--select", "mcts", "--gpu", *options])
        monkeypatch.setattr(sys, "stdin", io.StringIO("isready\nquit\n"))
        runpy.run_module("engine.uci", run_name="__main__")

    assert built == [("int8", 1, False), ("int8", 0, False), ("int8", 1, True)]


#uci.py's lc0 time options reach the clock and the search: the first move bonus on each game's first move, smart
#pruning's factor, wait and one-move-left stop; a factor of 0 or less is refused
@pytest.mark.filterwarnings("ignore:'engine.uci' found in sys.modules")
def test_the_engine_takes_lc0s_time_options(monkeypatch):
    given = []
    def search(self, board, nodes, seconds, extension, stop, info=None):
        given.append((seconds, self.smart_pruning_factor, self.smart_pruning_wait, self.stop_at_one_move_left))
        return SearchReport(0.5, [next(iter(board.legal_moves))], 1, [])
    monkeypatch.setattr(engine_search, "engine_network", lambda checkpoint, leaves, precision, fp16_blocks, device=None, blocking_wait=False: cpu_batches(BoardModel()))
    monkeypatch.setattr(EngineSearch, "search", search)
    go = "go " + " ".join(CLOCK_60)
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(["isready", "position startpos", go, "position startpos moves e2e4 e7e5", go,
                                                               "ucinewgame", "position startpos", go, "quit"])))
    options = ["--first-move-bonus", "1.8", "--book-ply-bonus", "0.25", "--smart-pruning-factor", "1.33", "--smart-pruning-wait", "0.2",
               "--stop-at-one-move-left"]
    monkeypatch.setattr(sys, "argv", ["uci.py", "unused.pt", "--select", "mcts", "--gpu", *options])
    runpy.run_module("engine.uci", run_name="__main__")

    first = pytest.approx(2.8 * Clock().limits(CLOCK_60, chess.Board(), 800)[1])
    assert given[0] == (first, 1.33, 0.2, True) and given[1][0] != first and given[2] == given[0]
    monkeypatch.setattr(sys, "argv", ["uci.py", "unused.pt", "--select", "mcts", "--gpu", "--smart-pruning-factor", "0"])
    with pytest.raises(SystemExit) as refused:
        runpy.run_module("engine.uci", run_name="__main__")
    assert refused.value.code == 2


#the engine as a gui drives it: a move within go movetime, go nodes' simulations (in a new game: a kept tree's visits
#count, as lc0 counts them), and on a clock more time for move 40 than for the first move
def test_the_engine_keeps_to_gos_limits(tmp_path):
    settings = {"d_model": 32, "num_layers": 1, "n_heads": 2, "mlp_hidden": 64, "value_bins": 16}
    save_checkpoint(ChessTransformer(**settings), settings, tmp_path / "model.pt")
    engine = chess.engine.SimpleEngine.popen_uci([sys.executable, "-m", "engine.uci", str(tmp_path / "model.pt"), "--select", "mcts"],
                                                 cwd=REPOSITORY)
    try:
        board = chess.Board()
        engine.play(board, chess.engine.Limit(nodes=10))

        started = time.perf_counter()
        timed = engine.play(board, chess.engine.Limit(time=0.5), info=chess.engine.INFO_BASIC)
        seconds = time.perf_counter() - started
        other = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
        counted = engine.play(other, chess.engine.Limit(nodes=50), info=chess.engine.INFO_BASIC)
        #on a 10+0.1 clock the first move gets ~0.27 s and a 9 man endgame ~1 s: the time is planned over the moves the
        #material has left. A game of their own (ucinewgame), so the last search's score is the first move's
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


#Searches' search of board to its simulations, one leaf a round; its report (id, line, score, root moves, visits)
def searched(searches: selfplay_rs.Searches, board: chess.Board, simulations: int, evaluate, search_id: int = 0):
    searches.start(search_id, board.root().fen(), [move.uci() for move in board.move_stack], simulations)
    reports = []
    while not reports:
        reports = searches.finish(evaluate(leaf_boards(searches.descend())))
    return reports[0]

#with tree reuse every game (search id) keeps its own tree: a game's next search carries on its tree although another
#game searched in between, and needs only the simulations it lacks
def test_every_game_keeps_its_tree():
    network = CountingNetwork(BoardModel())
    searches = selfplay_rs.Searches(**rust_search_settings(False), tree_reuse=True)
    boards = [chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1"), chess.Board()]
    for search_id, board in enumerate(boards):
        searched(searches, board, 400, network.launch, search_id)

    for search_id, board in enumerate(boards):
        network.boards = 0
        assert searched(searches, board, 600, network.launch, search_id)[4] == 600
        assert 0 < network.boards <= 200


#certainty propagation proves a mate in two (the defender's one reply is mated) and the engine stops once it is
#proven: the mate's first move, scored as a certain win, long before its budget; without it the win is only likely.
#The tree kept for the next move keeps the proof of the mate in one
def test_certainty_proves_a_mate_and_the_search_stops():
    board = chess.Board("k7/8/2K5/8/8/8/8/7R w - - 0 1")
    network = CountingNetwork(UniformModel())
    search = EngineSearch(network, 16, settings={"certainty": True})
    proven = search.search(board, simulations=20_000)
    assert proven.line[0].uci() == "c6b6" and proven.score == 1.0 and proven.simulations < 5_000

    unproven = EngineSearch(cpu_batches(UniformModel()), 16).search(board, simulations=proven.simulations)
    assert unproven.score < 1.0

    board.push_uci("c6b6")
    board.push_uci("a8b8")
    network.boards = 0
    mate = search.search(board, simulations=20_000)
    assert mate.line[0].uci() == "h1h8" and mate.score == 1.0 and network.boards <= 2 * 16


#the search goes no further than a proven node: with the whole budget searched (no smart pruning), the simulations
#after the proof take the proven score at once and the network sees few boards
def test_certainty_stops_descents_at_proven_nodes():
    network = CountingNetwork(UniformModel())
    search = EngineSearch(network, 16, smart_pruning=False, settings={"certainty": True})
    report = search.search(chess.Board("k7/8/2K5/8/8/8/8/7R w - - 0 1"), simulations=6000)

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


#a proven win is played although another move has more visits (lc0 ranks proven wins first): the network likes a king
#move, the search finds the rook's mate
def test_certainty_plays_a_proven_win_over_the_most_visited_move():
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
    model = LikesMoves([("g1f1", chess.WHITE)], 5.0)
    report = EngineSearch(cpu_batches(model), 16, settings={"certainty": True}).search(board, simulations=4000)

    assert report.line[0].uci() == "e1e8" and report.score == 1.0
    assert report.root_moves[0][0].uci() == "g1f1"


#the network all but insists on Ra2, after which Rd1 mates: the plain search spends its visits there and plays it, while
#with certainty the proven loss is passed over and its visits go to the other moves, one of which is played
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


#--search-settings takes a setting's type: true or false for certainty, a number otherwise
def test_search_settings_take_true_and_false():
    settings = {"certainty": True, "c_puct": 0.8}
    assert parse_changes("certainty=false,c_puct=1.5", settings) == {"certainty": False, "c_puct": 1.5}
    with pytest.raises(ValueError):
        parse_changes("certainty=0", settings)


#a network whose answers come back late once, as on a cuda graph's first capture or a gpu shared with another engine,
#on a round's given collect; it counts the boards it is asked for
class StallsOnce:
    def __init__(self, network, stall_seconds: float, on_collect: int):
        self.network, self.stall_seconds, self.on_collect = network, stall_seconds, on_collect
        self.collects = self.boards = 0

    def launch(self, boards: np.ndarray):
        self.boards += len(boards)
        return self.network.launch(boards)

    def collect(self, launched):
        self.collects += 1
        if self.collects == self.on_collect:
            time.sleep(self.stall_seconds)
        return self.network.collect(launched)


#one slow round at the end of a search does not cut the next searches on a clock short: a running mean of the round
#times carried it on, and the next searches stopped after one round each, whose time never updated the mean
#(2026-10-12: streaks of 3-5 moves at 6-41 nodes in 10 of 174 games). With growing rounds a cut search is one leaf
@pytest.mark.parametrize("grow_leaves", [0, 16])
def test_a_stalled_round_does_not_cut_the_next_searches_short(grow_leaves):
    network = StallsOnce(cpu_batches(BoardModel()), 1.5, on_collect=2)
    search = EngineSearch(network, 16, smart_pruning=False, grow_leaves=grow_leaves)
    search.search(chess.Board(), seconds=0.05)
    for fen in ["4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1", "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", "3r2k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1"]:
        network.boards = 0
        search.search(chess.Board(fen), seconds=0.3)
        assert network.boards > 5 * 16


#ucinewgame (new_game) drops the kept tree and the cache: the same position searched again is searched from nothing,
#as a fresh engine searches it, where without it the kept tree already has the simulations
def test_a_new_game_starts_a_fresh_tree():
    board = chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")
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


#the knights out and back, which the network all but insists on: the search scores the start position reached again
#inside it as a draw, most of its visits there. After Nf3 that repeat is of a position before the root, which the game
#needs a third time for: its draws come out of the kept tree, which keeps only the few visits elsewhere (a search of
#one simulation reports the kept tree's)
KNIGHTS_OUT_AND_BACK = [("g1f3", chess.WHITE), ("g8f6", chess.BLACK), ("f3g1", chess.WHITE), ("f6g8", chess.BLACK)]

def test_stale_draws_come_out_of_a_kept_tree():
    search = EngineSearch(cpu_batches(LikesMoves(KNIGHTS_OUT_AND_BACK, 8.0)), 16, smart_pruning=False, grow_leaves=0)
    board = chess.Board()
    first = search.search(board, simulations=400)
    assert first.root_moves[0] == (chess.Move.from_uci("g1f3"), 399)
    board.push_uci("g1f3")
    assert search.search(board, simulations=1).simulations < 50


#once all four moves are played the root is the repeated start position, which the last search never expanded: a
#search of fewer simulations than it had there still searches its moves (it reported an empty line, a rust panic)
def test_a_root_the_last_search_never_expanded_is_searched():
    search = EngineSearch(cpu_batches(LikesMoves(KNIGHTS_OUT_AND_BACK, 8.0)), 16, smart_pruning=False, grow_leaves=0)
    board = chess.Board()
    search.search(board, simulations=100)
    for move, _ in KNIGHTS_OUT_AND_BACK:
        board.push_uci(move)
    report = search.search(board, simulations=50)
    assert report.line and report.line[0] in board.legal_moves and report.simulations >= 50


#a gui that leaves a draw to be claimed (fide's rules, lichess) asks for a move in a threefold repetition or at the fifty
#move limit: the engine searches it and plays a move (it raised that the game was over, which ended the engine)
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


#an engine's game of `moves` searches from board, each answered by the reply its search expects (or the given replies),
#with uci.py's advance after each move or not, or with the kept tree dropped instead (an advance to another game; the
#cache keeps its answers): the reports, the replies, the network's boards and the advances that had a tree to carry on
def advancing_game(model, board: chess.Board, moves: int, way: str, replies: list[str] | None = None):
    network = CountingNetwork(model)
    search = EngineSearch(network, 16, smart_pruning=False)
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
        board.push(reply)
        played.append(reply.uci())
    return reports, played, network.boards, advanced, search

#uci.py's advance once its move is sent carries the kept tree on to the game after the move, while the opponent thinks,
#and the next search plays on from it exactly as from the tree the last search left: the same reports move for move, in
#a middlegame that reuses its trees (as dropping them shows) and in a game whose knights shuffle so that draws go stale.
#A position that does not follow the move is searched afresh
def test_advancing_the_tree_changes_no_search(carlsen_games):
    middlegame = carlsen_games[0].board()
    for move in list(carlsen_games[0].mainline_moves())[:16]:
        middlegame.push(move)
    reports, replies, boards, _, _ = advancing_game(BoardModel(), middlegame.copy(), 8, "plain")
    advanced_reports, _, advanced_boards, advanced, _ = advancing_game(BoardModel(), middlegame.copy(), 8, "advance", replies)
    dropped_reports = advancing_game(BoardModel(), middlegame.copy(), 8, "drop", replies)[0]
    assert advanced_reports == reports != dropped_reports and advanced == 8 and advanced_boards == boards

    shuffles = ["g8f6", "f6g8", "a7a6", "g8f6", "f6g8", "a6a5", "b8c6", "c6b8", "h7h6", "h6h5"]
    model = LikesMoves(KNIGHTS_OUT_AND_BACK + [("b1c3", chess.WHITE), ("c3b1", chess.WHITE)], 4.0)
    reports = advancing_game(model, chess.Board(), len(shuffles), "plain", shuffles)[0]
    advanced_reports, _, _, advanced, search = advancing_game(model, chess.Board(), len(shuffles), "advance", shuffles)
    assert advanced_reports == reports and advanced == len(shuffles)

    board = chess.Board()
    report = search.search(board, simulations=50)
    played, other = board.copy(), board.copy()
    played.push(report.line[0])
    search.advance(played)
    other.push(next(move for move in board.legal_moves if move != report.line[0]))
    other.push(next(iter(other.legal_moves)))
    assert search.search(other, simulations=50).line[0] in other.legal_moves


#a search whose tree reaches Searches' memory_limit_mb ends there with its report, as at its simulations (the engine's
#rounds too, which would otherwise wait on a search that gathers nothing); one that stays below its limit is the search
#without a limit
def test_a_search_ends_at_its_memory_limit():
    evaluate = cpu_batches(BoardModel()).evaluate_boards
    reports = {limit: searched(selfplay_rs.Searches(**rust_search_settings(False), memory_limit_mb=limit), chess.Board(), 3000, evaluate)
               for limit in (0, 1, 1000)}
    assert reports[1000] == reports[0] and reports[0][4] == 3000
    assert 100 < reports[1][4] < 3000

    search = EngineSearch(cpu_batches(BoardModel()), 16, smart_pruning=False)
    search.searches = selfplay_rs.Searches(**engine_search_settings(False), tree_reuse=True, memory_limit_mb=1)
    report = search.search(chess.Board(), simulations=100_000)
    assert 100 < report.simulations < 3000 and report.line[0] in chess.Board().legal_moves


#a gui driving uci.main on a thread of its own: the lines told are the engine's stdin, and the lines it sends are kept
#with when it sent them. load_player makes its player with make_player (unless None): on the engine's thread, as
#Searches must be used on the thread that made it; main_arguments are main's
class Gui:
    def __init__(self, monkeypatch, make_player, checkpoint: Path = Path("unused.pt"), **main_arguments):
        self.lines: queue.Queue = queue.Queue()
        self.sent: list[tuple[float, str]] = []
        self.errors: list[BaseException] = []
        monkeypatch.setattr(sys, "stdin", iter(self.lines.get, None))
        monkeypatch.setattr(uci, "send", lambda line: self.sent.append((time.perf_counter(), line)))
        if make_player is not None:
            monkeypatch.setattr(uci, "load_player", lambda *arguments: make_player())
        self.engine = threading.Thread(target=self.run, args=(checkpoint,), kwargs={"selection": "mcts", **main_arguments}, daemon=True)
        self.engine.start()

    def run(self, checkpoint: Path, **main_arguments) -> None:
        try:
            uci.main(checkpoint, **main_arguments)
        except BaseException as error:
            self.errors.append(error)

    #when the line was told
    def tell(self, line: str) -> float:
        told = time.perf_counter()
        self.lines.put(line + "\n")
        return told

    #when the engine sent its first line starting with `start` since `since`, waiting up to `seconds` for one; None if none
    def sent_at(self, start: str, since: float = 0.0, seconds: float = 0.0) -> float | None:
        give_up = time.perf_counter() + seconds
        while True:
            found = [at for at, line in list(self.sent) if at >= since and line.startswith(start)]
            if found or time.perf_counter() > give_up:
                return found[0] if found else None
            time.sleep(0.005)

    def lines_sent(self, start: str) -> list[str]:
        return [line for _, line in list(self.sent) if line.startswith(start)]

    #a number of the last info line, e.g. its nodes
    def last_info(self, name: str) -> int:
        words = self.lines_sent("info depth")[-1].split()
        return int(words[words.index(name) + 1])

    #the input ends after `last`; the engine must then be done within `seconds`
    def close(self, last: str | None = "quit", seconds: float = 5.0) -> None:
        if last is not None:
            self.tell(last)
        self.lines.put(None)
        self.engine.join(seconds)
        assert not self.engine.is_alive() and not self.errors, self.errors

def board_search() -> EngineSearch:
    return EngineSearch(cpu_batches(BoardModel()), 16)

#a small network on lean lanes (inference.Lane's) of 16 boards. The earlier tests' graphs are collected first: one freed
#while these are recorded breaks the recording
def lean_network(blocking_wait: bool = False) -> GraphedBatches:
    gc.collect()
    torch.manual_seed(0)
    model = ChessTransformer(d_model=32, num_layers=1, n_heads=2).cuda().eval()
    return GraphedBatches(model, small_graph_sizes(16), 2, blocking_wait=blocking_wait, lean=True)

#board_search's in Searches' own round loop (rust_rounds), on a gpu
def rust_search() -> EngineSearch:
    return EngineSearch(lean_network(), 16, rust_rounds=True)

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
#the uci tests below with the python loop, and with Searches' own
both_loops = pytest.mark.parametrize("make_search", [board_search, pytest.param(rust_search, marks=gpu)])

#stop ends a running search as its deadline would (10 s here), and the engine answers with its move at once (stop was
#read only once the search had ended)
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

#isready during a search is answered at once, the search going on (UCI); one after a command the engine has not taken
#yet waits for it, so that readyok still means all before it is done
@both_loops
def test_isready_is_answered_while_the_engine_searches(monkeypatch, make_search):
    gui = Gui(monkeypatch, make_search)
    gui.tell("go movetime 10000")
    time.sleep(0.3)
    asked = gui.tell("isready")
    ready = gui.sent_at("readyok", seconds=1)
    assert ready is not None and ready < asked + 0.5 and gui.sent_at("bestmove") is None

    gui.tell("position startpos moves e2e4")
    asked = gui.tell("isready")
    time.sleep(0.3)
    assert gui.sent_at("readyok", since=asked) is None
    gui.tell("stop")
    answered, ready = gui.sent_at("bestmove", seconds=3), gui.sent_at("readyok", since=asked, seconds=3)
    assert answered is not None and ready is not None and answered < ready
    gui.close()

#quit during a search ends it, and the engine
@both_loops
def test_quit_ends_a_running_search(monkeypatch, make_search):
    gui = Gui(monkeypatch, make_search)
    gui.tell("go movetime 10000")
    time.sleep(0.3)
    gui.tell("quit")
    gui.engine.join(2)
    assert not gui.engine.is_alive() and not gui.errors
    gui.close(last=None)

#with --infinite-until-stop go infinite searches past --simulations until stop, and answers only then, also once its
#search has ended by itself (an only move); the end of the input ends one too
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

#without it go infinite runs --simulations, as cutechess-cli's tc=inf games need
def test_go_infinite_runs_the_simulations_by_default(monkeypatch):
    gui = Gui(monkeypatch, lambda: EngineSearch(cpu_batches(BoardModel()), 16, smart_pruning=False), simulations=300)
    gui.tell("go infinite")
    assert gui.sent_at("bestmove", seconds=10) is not None and gui.last_info("nodes") == 300
    gui.close()

#go ponder searches the opponent's time until ponderhit, then searches as the go asks with its time counted from the
#ponderhit; at stop it answers at once, a move the gui throws away, and the clock never hears of it (it answered at
#its move's time, before ponderhit)
def test_go_ponder_searches_until_ponderhit(monkeypatch):
    limits = []
    class Recorded(EngineSearch):
        def search(self, board, simulations=None, seconds=None, extension=None, stop=None, info=None):
            limits.append((simulations, seconds))
            return super().search(board, simulations, seconds, extension, stop, info)
    gui = Gui(monkeypatch, lambda: Recorded(cpu_batches(BoardModel()), 16))
    board = chess.Board()
    board.push_uci("e2e4")
    gui.tell("position startpos moves e2e4")
    go = "go ponder wtime 10000 btime 10000"
    gui.tell(go)
    time.sleep(0.5)
    assert gui.sent_at("bestmove") is None
    hit = gui.tell("ponderhit")
    answered = gui.sent_at("bestmove", seconds=3)
    assert answered is not None and gui.last_info("time") < (answered - hit) * 1000 + 50
    assert limits == [(None, None), (None, pytest.approx(Clock().limits(go.split()[1:], board, 200)[1]))]

    gui.tell(go)
    time.sleep(0.3)
    stopped = gui.tell("stop")
    assert gui.sent_at("bestmove", since=stopped, seconds=2) is not None and len(limits) == 3
    gui.close()

#the options a gui sets reach the search when it is built (isready), MoveOverhead the next go's time. Set once the
#search is built, or to a value it could not take, they stay as they were and the gui is told why
def test_the_options_reach_the_search(monkeypatch):
    built, given = [], []
    class Player:
        def search(self, board, nodes, seconds, extension, stop, info=None):
            given.append(seconds)
            return SearchReport(0.5, [next(iter(board.legal_moves))], 1, [])
    script = ["uci", "setoption name cachepositions value 1000", "setoption name RamLimitMb value 64", "setoption name SyzygyProbeLimit value 6",
              "setoption name AdvanceThreads value 12", "setoption name AdvanceThreads value 0",
              "setoption name MoveOverhead value 500", "setoption name MoveOverhead value -1", "setoption name SyzygyProbeLimit value 7",
              "setoption name Hash value 16", "isready", "setoption name CachePositions value 5", "go movetime 1000",
              "setoption name MoveOverhead value 100", "go movetime 1000", "ucinewgame", "go movetime 1000", "quit"]
    sent = []
    monkeypatch.setattr(uci, "load_player", lambda *arguments: built.append(arguments) or Player())
    monkeypatch.setattr(uci, "send", sent.append)
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(script)))
    uci.main(Path("unused.pt"), selection="mcts", search_options={"smart_pruning": False})

    assert [line.split()[2] for line in sent if line.startswith("option name")] == ["SyzygyPath", "SyzygyProbeLimit", "MoveOverhead", "CachePositions",
                                                                                     "RamLimitMb", "AdvanceThreads"]
    assert len(built) == 1 and built[0][4] is False
    assert built[0][5] == {"smart_pruning": False, "tablebase_pieces": 6, "syzygy_path": None, "cache_positions": 1000, "memory_limit_mb": 64,
                           "advance_threads": 12}
    assert given == [pytest.approx(0.5), pytest.approx(0.9), pytest.approx(0.9)]
    told = [line for line in sent if line.startswith("info string")]
    assert len(told) == 5 and "AdvanceThreads" in told[0] and "-1" in told[1] and "7" in told[2] and "Hash" in told[3] and "CachePositions" in told[4]

#EngineSearch makes its Searches with the cache size, memory limit and advance threads it is given, and a search ends at
#the limit
def test_the_search_takes_its_cache_size_memory_limit_and_advance_threads(monkeypatch):
    made = []
    searches = selfplay_rs.Searches
    monkeypatch.setattr(selfplay_rs, "Searches", lambda **settings: made.append(settings) or searches(**settings))
    EngineSearch(cpu_batches(BoardModel()), 16)
    limited = EngineSearch(cpu_batches(BoardModel()), 16, smart_pruning=False, cache_positions=1000, memory_limit_mb=1, advance_threads=12)
    assert ([(settings["cache_positions"], settings["memory_limit_mb"], settings["advance_threads"]) for settings in made]
            == [(engine_search.CACHE_POSITIONS, 0, engine_search.ADVANCE_THREADS), (1000, 1, 12)])
    assert 100 < limited.search(chess.Board(), simulations=100_000).simulations < 3000

#a SyzygyPath the gui sets reaches the search: the root takes the tables' move (no simulations) from its folders. A
#folder without the tables is refused, and the engine plays on as it was
def test_a_syzygy_path_reaches_the_search(monkeypatch, tablebase, tmp_path):
    settings = {"d_model": 32, "num_layers": 1, "n_heads": 2, "mlp_hidden": 64, "value_bins": 16}
    save_checkpoint(ChessTransformer(**settings), settings, tmp_path / "model.pt")
    (tmp_path / "no_tables").mkdir()
    gui = Gui(monkeypatch, None, tmp_path / "model.pt")
    gui.tell(f"setoption name SyzygyPath value {tmp_path / 'no_tables'}")
    assert gui.sent_at("info string", seconds=5) is not None and gui.lines_sent("info string")[0].endswith("the tables stay as they were")
    gui.tell(f"setoption name SyzygyPath value {SYZYGY_DIR}")
    gui.tell("position fen k7/8/1K6/8/8/8/8/7R w - - 0 1")
    gui.tell("go nodes 50")
    assert gui.sent_at("bestmove", seconds=30) is not None
    assert gui.lines_sent("bestmove") == ["bestmove h1h8"] and gui.last_info("nodes") == 0 and len(gui.lines_sent("info string")) == 1
    gui.close()

#a position the search could not start from leaves the last one, and the gui is told why (each of these ended the
#engine, the last two at go); in a game that is over go answers with the null move
def test_a_bad_position_leaves_the_last_one(monkeypatch):
    searched = []
    class Player:
        def search(self, board, nodes, seconds, extension, stop=None, info=None):
            searched.append(board.fen())
            return SearchReport(0.5, [next(iter(board.legal_moves))], 1, [])
    bad = ["startpos moves e2e4 e7e4", "startpos moves e2e4 e7", "fen not a fen", "fen 8/8/8/8/8/8/8/8 w - - 0 1", "", "kiwipete",
           "startpos moves e2e4 0000", "fen 4k3/8/8/8/8/8/PPPPPPPP/QQ2K3 w - - 0 1"]
    script = ["position startpos moves e2e4", *(f"position {position}" for position in bad), "go nodes 10",
              "position fen 4R1k1/5ppp/8/8/8/8/5PPP/6K1 b - - 1 1", "go nodes 10", "quit"]
    sent = []
    monkeypatch.setattr(uci, "load_player", lambda *arguments: Player())
    monkeypatch.setattr(uci, "send", sent.append)
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(script)))
    uci.main(Path("unused.pt"), selection="mcts")

    after_e4 = chess.Board()
    after_e4.push_uci("e2e4")
    assert searched == [after_e4.fen()]
    assert len([line for line in sent if line.startswith("info string") and line.endswith("the last position stays")]) == len(bad)
    assert [line for line in sent if line.startswith("bestmove")][1] == "bestmove 0000"

#the fens parse_position refuses are those the rust search refuses (python-chess's status, and shakmaty's promoted
#material, which that leaves out); a fen whose castling rights or en passant square python-chess cleans both take
@pytest.mark.parametrize("fen", [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w KQkq - 0 1", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e3 0 1",
    "4k3/8/8/8/8/8/8/8 w - - 0 1", "4k3/8/8/8/8/8/8/K3K3 w - - 0 1", "4k3/8/8/8/8/8/8/4R1K1 w - - 0 1", "P3k3/8/8/8/8/8/8/4K3 w - - 0 1",
    "4k3/8/8/8/8/P7/PPPPPPPP/4K3 w - - 0 1", "4k3/8/5N2/1B6/8/8/8/4R1K1 b - - 0 1", "4k3/8/8/8/1NNNNNNN/8/PPPPPPPP/RNB1K3 w - - 0 1",
    "4k3/8/8/8/8/8/PPPPPPPP/QQ2K3 w - - 0 1", "4k3/8/8/8/8/8/1PPPPPPP/QQ2K3 w - - 0 1", "4k3/8/8/8/8/8/PPPPPPPP/B1B1K3 w - - 0 1",
    "4k3/8/8/8/8/8/PPPPPPPP/BB2K3 w - - 0 1", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
])
def test_the_positions_refused_are_the_searchs(fen):
    try:
        uci.parse_position(["fen", *fen.split()])
        refused = False
    except ValueError:
        refused = True
    try:
        selfplay_rs.Searches(**rust_search_settings(False)).start(0, chess.Board(fen).root().fen(), [], 10)
        searched = True
    except ValueError:
        searched = False
    assert refused != searched


#a search tells its report as it stands about every INFO_SECONDS while it runs, its nodes growing, and the gui gets each
#as an info line before bestmove, as lc0 sends one a second
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


#Searches' gather_workers (search/batch.rs): the engine's searches, on a kept tree too, the same for any number
def test_gather_workers_make_the_same_search_for_any_number():
    reports, counts = [], []
    for workers in (1, 3):
        search = EngineSearch(cpu_batches(BoardModel()), 64, smart_pruning=False, gather_workers=workers)
        board = chess.Board()
        first = search.search(board, simulations=3000)
        board.push(first.line[0])
        board.push(first.line[1])
        reports.append((first, search.search(board, simulations=6000)))
        counts.append(search.searches.gather_counts())
    assert reports[0] == reports[1] and reports[0][1].simulations == 6000
    #all but the items, which the workers share a pass's picks out in
    rounds, passes, _, picks, leaves, cached = counts[0][:6]
    assert counts[0][:2] + counts[0][3:] == counts[1][:2] + counts[1][3:] and passes > rounds and picks >= leaves > 0 and cached > 0


#Searches with the collisions of every round it descends
class CollisionsRecorded:
    def __init__(self, searches: selfplay_rs.Searches):
        self.searches, self.collisions = searches, []

    def descend_into(self, boards: np.ndarray, leaves: int, collisions: int, simulations: int, round: int | None = None,
                     root_of: int | None = None, legal_columns: bool = False) -> tuple:
        self.collisions.append(collisions)
        return self.searches.descend_into(boards, leaves, collisions, simulations, round=round, root_of=root_of, legal_columns=legal_columns)

    def __getattr__(self, name: str):
        return getattr(self.searches, name)


#BoardModel with five times its logits: priors peaked enough for a round's leaves to collide
class PeakedModel(BoardModel):
    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, values = super().forward(boards)
        return 5 * logits, values


#EngineSearch's collision_cap gives every round that many collisions, more than its leaves; 0 keeps COLLISIONS, at
#most the round's leaves. gather_batch then counts more of them
def test_a_collision_cap_lets_rounds_collide_more():
    recorded = {}
    for cap in (0, 1000):
        search = EngineSearch(cpu_batches(PeakedModel()), 64, smart_pruning=False, collision_cap=cap, gather_workers=1)
        search.searches = CollisionsRecorded(search.searches)
        assert search.search(chess.Board(), simulations=3000).simulations == 3000
        recorded[cap] = (search.searches.collisions, search.searches.gather_counts())
    assert max(recorded[0][0]) == 64 and set(recorded[1000][0]) == {1000}
    #collisions per leaf for the network
    (_, counts), (_, capped) = recorded[0], recorded[1000]
    assert capped[7] / capped[4] > counts[7] / counts[4]


#with collision_scaling a round's collisions follow lc0's CalculateCollisionsLeft on the visits done or left (the
#fewer), at most collision_cap's; without, collision_cap's or COLLISIONS' at most the round's leaves
@pytest.mark.parametrize("scaling, cap, visits, simulations, expected", [
    (False, 0, 1000, None, 40), (False, 300, 1000, None, 300), (True, 0, 28, None, 1), (True, 0, 1000, None, 154),
    (True, 0, 10_000, None, 2819), (True, 0, 145_000, None, 80_000), (True, 1024, 10_000, None, 1024), (True, 0, 10_000, 10_100, 6)])
def test_the_collisions_of_a_round(scaling, cap, visits, simulations, expected):
    class Searches:
        def root_visits(self, game):
            return visits, 0, 0
    search = EngineSearch(cpu_batches(BoardModel()), 64, collision_cap=cap, collision_scaling=scaling)
    search.searches = Searches()

    assert search.round_collisions(40, simulations) == expected


#a search with collision_scaling gives its rounds few collisions while the tree is small and more as it grows
def test_collision_scaling_grows_the_collisions_with_the_tree():
    search = EngineSearch(cpu_batches(PeakedModel()), 64, smart_pruning=False, collision_scaling=True, gather_workers=1)
    search.searches = CollisionsRecorded(search.searches)

    assert search.search(chess.Board(), simulations=6000).simulations == 6000
    collisions = search.searches.collisions
    assert collisions[0] == 1 and max(collisions) > 64

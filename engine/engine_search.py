from __future__ import annotations
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
from pathlib import Path
import statistics
import threading
import time
import chess
import numpy as np
import selfplay_rs
import torch
from model.encoding import EXPECTED_SCORE_PER_CENTIPAWN, TOKENS_PER_BOARD
from engine.selfplay import ANSWER_WIDTH, SIMULATIONS_PER_LEAF, rust_search_settings
from engine.inference import ExactSizeCompiled, GraphedBatches, chunk_positions, fp16_model, int8_model, small_graph_sizes
from engine.cuda_runtime import cuda_runtime
from engine.tablebase import TABLEBASE_PIECES, covers, open_tablebase, open_tablebase_path, ranked_moves, ranked_pieces, wdl_score

LEAVES = 64
#descents that may end on a leaf already waiting before a round stops gathering
COLLISIONS = 64
ROUNDS_IN_FLIGHT = 2
#a round takes a leaf per this many simulations done or left; 0 for fixed
GROW_LEAVES = 16
#the most simulations Searches.start takes, for a search with only a deadline or a stop
UNLIMITED = 2**32 - 1
GAME = 0
FIRST_ROUND_SECONDS = 0.01
#the median of the last rounds' times is the next round's, so that one stall does not carry on
ROUND_TIMES = 31
#smart pruning stops once the most visited move leads the second by more than the simulations left / this
SMART_PRUNING_FACTOR = 1.0
#a search on the clock whose score fell more than FALL_START_CP below the last one's runs on when it would stop, by all its extension at FALL_FULL_CP
FALL_START_CP = 10
FALL_FULL_CP = 30
SPEED_ROUNDS = 8
INFO_SECONDS = 1.0
RUST_WAIT_SECONDS = 0.005
CACHE_POSITIONS = 5_000_000
#the threads that look for a kept tree's stale leaves after a move, while the opponent thinks (a tree with transpositions has none to look for);
#0 for the next search's start to look
ADVANCE_THREADS = 4
#the threads that free the dead rest of a moved tree beside the next search; 0 for the search's start to free it all, on the engine's own clock
GC_THREADS = 2
#the engine's changes to mcts.py's search, which self-play keeps as it is
ENGINE_SETTINGS: dict = {"value_sharpness": 3.0, "certainty": True, "transpositions": True}
ROOT_MOVES_REPORTED = 3
MAX_REPORTED_SCORE = 0.9999 #about 25 pawns


#score is for the side to move (0 lost to 1 won), line starts with the move played; simulations are the root's visits, reused those it had at the start
@dataclass
class SearchReport:
    score: float
    line: list[chess.Move]
    simulations: int
    root_moves: list[tuple[chess.Move, int]]
    reused: int = 0

def score_to_centipawns(score: float) -> int:
    score = min(max(score, 1 - MAX_REPORTED_SCORE), MAX_REPORTED_SCORE)
    return round(math.log(score / (1 - score)) / EXPECTED_SCORE_PER_CENTIPAWN)


#search.rs sharpen: the value's logit times sharpness
def sharpen(score: float, sharpness: float) -> float:
    return score if sharpness == 1 or score in (0.0, 1.0) else 1 / (1 + ((1 - score) / score) ** sharpness)

#Searches' settings: mcts.py's, the engine's own over them (own), then changes over those
def engine_search_settings(use_tablebase: bool, changes: dict | None = None, own: bool = True, tablebase_pieces: int = TABLEBASE_PIECES,
                           syzygy_path: str | None = None) -> dict:
    return {**rust_search_settings(use_tablebase, tablebase_pieces, syzygy_path), **(ENGINE_SETTINGS if own else {}), **(changes or {})}

def parse_changes(text: str, settings: dict) -> dict:
    changes = {}
    for change in filter(None, text.split(",")):
        name, value = change.split("=")
        if name not in settings:
            raise ValueError(f"{name} is not a setting, these are: {', '.join(sorted(settings))}")
        kind = type(settings[name])
        if kind not in (bool, int, float):
            raise ValueError(f"{name} is not a number or true or false")
        if kind is bool:
            if value not in ("true", "false"):
                raise ValueError(f"{name} is true or false, not {value}")
            changes[name] = value == "true"
        else:
            changes[name] = kind(float(value))
    return changes

#int8 is not for self-play or labels: its answers depend on the rest of the batch
PRECISIONS = ("fp16", "int8")
ENGINE_PRECISION = "int8"
#the int8 network's blocks kept in fp16 besides the last; 0 is every block but the last in int8
ENGINE_FP16_BLOCKS = 1


#cpus for count threads in one tree, from selfplay_rs.cpu_topology's (l3s, cores) or the machine's: a physical core each, in one L3 if one has enough
#(the largest), else the largest L3s' cores first; smt siblings only once every core has a thread. [] without a topology or for more threads than cpus
def thread_cpus(count: int, topology: tuple | None = None) -> list[int]:
    if not count:
        return []
    topology = topology if topology is not None else selfplay_rs.cpu_topology()
    if topology is None:
        return []
    l3s, cores = topology
    by_l3 = [(size, [core for core in cores if core[0] in cpus]) for size, cpus in l3s]
    #stable: equal L3s stay in the os's order
    by_l3.sort(key=lambda l3: (len(l3[1]) < count, -l3[0]))
    cpus = [core[0] for _, l3 in by_l3 for core in l3] + [cpu for _, l3 in by_l3 for core in l3 for cpu in core[1:]]
    return cpus[:count] if count <= len(cpus) else []


LEAN_CAPABILITIES = {(12, 0)}


#the network as cuda graphs for a round's leaves, from one to max_batch, on the current gpu
def engine_network(checkpoint_path: Path, max_batch: int = LEAVES, precision: str = ENGINE_PRECISION,
                   fp16_blocks: int = ENGINE_FP16_BLOCKS, lean: bool | None = None):
    return network_graphs(network_model(checkpoint_path, precision, fp16_blocks), max_batch, lean=lean)


#engine_network on each of devices with the int8 weights read once: the models load in threads, the graphs are recorded one network at a time
#(dynamo compiles one frame at a time, and a recording forbids other threads' allocations)
def engine_networks(checkpoint_path: Path, max_batch: int, precision: str, fp16_blocks: int, devices: list[int],
                    blocking_wait: bool) -> list[GraphedBatches]:
    #imported here, as int8_model imports it: it needs Triton
    from engine.quantization import int8_weights
    with torch.cuda.device(devices[0]):
        weights = int8_weights(checkpoint_path, fp16_blocks) if precision == "int8" else None

    def load(device: int) -> torch.nn.Module:
        with torch.cuda.device(device):
            return network_model(checkpoint_path, precision, fp16_blocks, weights)
    with ThreadPoolExecutor(len(devices)) as pool:
        models = list(pool.map(load, devices))
    networks = []
    for model, device in zip(models, devices):
        with torch.cuda.device(device):
            networks.append(network_graphs(model, max_batch, blocking_wait))
    return networks


def network_model(checkpoint_path: Path, precision: str, fp16_blocks: int, weights: dict | None = None) -> torch.nn.Module:
    return int8_model(checkpoint_path, fp16_blocks, weights) if precision == "int8" else fp16_model(checkpoint_path)


def network_graphs(model: torch.nn.Module, max_batch: int, blocking_wait: bool = False, lean: bool | None = None) -> GraphedBatches:
    lanes = ROUNDS_IN_FLIGHT if max_batch > 1 else 1
    chunk = chunk_positions(model.d_model, lanes)
    sizes = small_graph_sizes(max_batch, chunk)
    #one board a round (mcts.py's search) never fills the largest graph
    compiled = ExactSizeCompiled(model, min(sizes[-1], chunk)) if max_batch > 1 else torch.compile(model, dynamic=True)
    return GraphedBatches(compiled, sizes, lanes, chunk, blocking_wait, lean_lanes() if lean is None else lean)


def lean_lanes() -> bool:
    return torch.cuda.get_device_capability() in LEAN_CAPABILITIES


#what the threads driving a search's networks share, behind changed's lock; changed is notified as a round comes back or the search stops
class SearchRun:
    def __init__(self, simulations: int | None, deadline: float | None, extension: tuple[int, float] | None, only_move: bool,
                 told: threading.Event | None, info: Callable[[SearchReport], None] | None, next_info: float,
                 root: tuple[int, int, int]):
        self.simulations, self.deadline, self.extension, self.only_move = simulations, deadline, extension, only_move
        self.told = told
        self.info, self.next_info = info, next_info
        self.changed = threading.Condition()
        self.report = None
        self.stopping = self.gathered = False
        self.rounds_out = 0 #on every network
        self.rounds_answered = 0 #rounds with leaves finished
        #rounds_answered when a round last found no leaves (every descent ended on a leaf already out): no network gathers again until another round is back
        self.starved = None
        self.next_round = 0
        self.progress: deque = deque(maxlen=SPEED_ROUNDS) #(time, root visits) after the last rounds
        #Searches.root_visits' numbers as the newest descend or finish left them, so that no thread asks for them
        self.root = root
        self.reused = root[0] #the visits of a kept tree, which this search did not make

    #numbers a descend or finish gave (None once the search is done): visits only grow, so the most is the newest
    def seen(self, root: tuple[int, int, int] | None) -> None:
        if root is not None and root[0] >= self.root[0]:
            self.root = root

    #the search stops gathering; report is None while its leaves are out (the finish of the last of them has it)
    def stop(self, report) -> None:
        self.stopping, self.report = True, report
        self.changed.notify_all()


class EngineSearch:
    #network has GraphedBatches' launch and collect (inference.ImmediateBatches on the cpu), or is a list of them, one a gpu, each driven by a thread of its own
    #one leaf a round is mcts.py's search: its settings (without ENGINE_SETTINGS), a fresh tree every move, one round on the network, no smart pruning
    #use_tablebase: the tables of up to tablebase_pieces pieces, from the folders of syzygy_path (a SyzygyPath) rather than the downloaded ones if given
    #memory_limit_mb: 0 for none (the tree's arenas may hold up to a quarter more: freed room no node fits); gc_threads: those that free a moved tree's dead rest;
    #rust_rounds: the rounds in Searches' own loop (rounds.rs), a thread a network and no python between them, the python loop's search exactly;
    #concurrent_descents: the networks' threads descend, finish and back up in the tree at once (search/shared.rs); pin_threads: the networks' threads each on a cpu
    #of thread_cpus'. None for these three: on for several networks (rust_rounds also needs lean lanes, so only on sm_120's)
    def __init__(self, network, leaves: int = LEAVES, use_tablebase: bool = False, smart_pruning: bool = True,
                 tablebase_pieces: int = TABLEBASE_PIECES, settings: dict | None = None, grow_leaves: int = GROW_LEAVES,
                 smart_pruning_factor: float = SMART_PRUNING_FACTOR, cache_positions: int = CACHE_POSITIONS, syzygy_path: str | None = None,
                 memory_limit_mb: int = 0, advance_threads: int = ADVANCE_THREADS, gc_threads: int = GC_THREADS, rust_rounds: bool | None = None,
                 concurrent_descents: bool | None = None, pin_threads: bool | None = None):
        self.networks = network if isinstance(network, list) else [network]
        self.leaves = leaves
        batched, several = leaves > 1, len(self.networks) > 1
        if several and not batched:
            raise ValueError("one leaf a round is mcts.py's search, which runs on one network")
        lean = all(getattr(network, "lean", False) for network in self.networks)
        self.rust_rounds = lean and several if rust_rounds is None else rust_rounds
        if self.rust_rounds and not lean:
            raise ValueError("Searches' own round loop runs on networks of lean lanes (engine_network's on sm_120)")
        self.concurrent_descents = several if concurrent_descents is None else concurrent_descents
        if self.concurrent_descents and not batched:
            raise ValueError("concurrent descents need leaves above 1: one leaf a round is mcts.py's search")
        #the batches and their boards Searches' own round loop launched
        self.rust_counts = [0, 0]
        self.collisions = COLLISIONS if batched else 0
        self.rounds_in_flight = ROUNDS_IN_FLIGHT if batched else 1 #on each network
        self.grow_leaves = grow_leaves if batched else 0
        self.rounds_at_once = batched and len(getattr(self.networks[0], "lanes", ())) > 1
        self.smart_pruning = batched and smart_pruning
        self.smart_pruning_factor = smart_pruning_factor
        self.advance_threads = advance_threads
        search_settings = engine_search_settings(use_tablebase, settings, own=batched, tablebase_pieces=tablebase_pieces, syzygy_path=syzygy_path)
        pinned = several if pin_threads is None else pin_threads
        self.network_cpus = thread_cpus(len(self.networks) if pinned and batched else 0)
        self.searches = selfplay_rs.Searches(**search_settings, tree_reuse=batched, cache_positions=cache_positions, memory_limit_mb=memory_limit_mb,
                                             advance_threads=advance_threads, gc_threads=gc_threads, concurrent_descents=self.concurrent_descents)
        #reports give the score on the network's own scale, before sharpening
        self.value_sharpness = search_settings.get("value_sharpness", 1.0)
        #the root takes the tables' move up to the size they can rank; a bigger one they cover is searched
        if not use_tablebase:
            self.tablebase, self.ranked_pieces = None, 0
        elif syzygy_path is None:
            self.tablebase, self.ranked_pieces = open_tablebase(tablebase_pieces), ranked_pieces(tablebase_pieces)
        else:
            self.tablebase, self.ranked_pieces = open_tablebase_path(syzygy_path, tablebase_pieces)
        self.certainty = search_settings.get("certainty", False)
        self.round_times: deque = deque([FIRST_ROUND_SECONDS], maxlen=ROUND_TIMES)
        self.round_seconds = FIRST_ROUND_SECONDS
        #the searching thread drives the first network unless the threads are pinned: on linux a pinned thread's new threads (advance's) would start on its one cpu
        self.drives_first = not self.network_cpus
        pooled = len(self.networks) - self.drives_first
        self.threads = ThreadPoolExecutor(pooled) if pooled else None

    def new_game(self) -> None:
        self.searches.new_game(GAME)

    #once the engine's move is sent, while the opponent thinks: has threads of Searches' look for the leaves the next root may make stale in a kept tree without
    #transpositions. A tree with transpositions has none to look for: its root moves at the next search's start, which costs next to nothing. False when there is no
    #tree to carry on, or no advance threads (the next search's start then does it)
    def advance(self, board: chess.Board) -> bool:
        return (self.leaves > 1 and self.advance_threads > 0
                and self.searches.advance(GAME, board.root().fen(), [move.uci() for move in board.move_stack]))

    #searches until `simulations` (counting those of a reused tree), until `seconds` have passed or until `stop` is set (uci's reader thread sets it), whichever
    #comes first; at least one of the three. extension: (the last search's score in centipawns, the seconds this one may add) for a search on the clock whose
    #score falls (FALL_START_CP). info gets the report as it stands about every INFO_SECONDS, for the gui's info lines
    def search(self, board: chess.Board, simulations: int | None = None, seconds: float | None = None,
               extension: tuple[int, float] | None = None, stop: threading.Event | None = None,
               info: Callable[[SearchReport], None] | None = None) -> SearchReport:
        if simulations is None and seconds is None and stop is None:
            raise ValueError("a search needs a number of simulations, a time or a stop")
        started = time.perf_counter()
        if self.tablebase is not None and covers(board, self.ranked_pieces):
            best = ranked_moves(self.tablebase, board)[0]
            return SearchReport(wdl_score(best.wdl), [best.move], 0, [])

        self.searches.start(GAME, board.root().fen(), [move.uci() for move in board.move_stack], UNLIMITED if simulations is None else simulations)
        deadline = started + seconds if seconds is not None else None
        run = SearchRun(simulations, deadline, extension if deadline is not None else None, board.legal_moves.count() == 1, stop, info,
                        started + INFO_SECONDS, self.searches.root_visits(GAME))
        if self.rust_rounds:
            self.rounds_in_rust(run)
            return self.search_report(run.report, run.reused)
        cpus = self.network_cpus or [None] * len(self.networks)
        driven = [self.threads.submit(self.drive, network, run, cpu) for network, cpu in list(zip(self.networks, cpus))[self.drives_first:]]
        try:
            if self.drives_first:
                self.drive(self.networks[0], run)
        finally:
            for drive in driven:
                drive.result()
        return self.search_report(run.report, run.reused)

    def search_report(self, report: tuple, reused: int) -> SearchReport:
        _, line, score, root_moves, visits = report
        return SearchReport(sharpen(score, 1 / self.value_sharpness), [chess.Move.from_uci(move) for move in line], visits,
                            [(chess.Move.from_uci(move), move_visits) for move, move_visits in root_moves[:ROOT_MOVES_REPORTED]], reused)

    #rounds_on on the thread pinned to cpu if it is given, a failure stopping the other networks' rounds too
    def drive(self, network, run: SearchRun, cpu: int | None = None) -> None:
        if cpu is not None:
            selfplay_rs.pin_thread(cpu)
        try:
            self.rounds_on(network, run)
        except BaseException:
            with run.changed:
                run.stopping = True
                run.changed.notify_all()
            #the network's rounds still out, finished on the gpu before anything else is
            if hasattr(network, "settle"):
                network.settle()
            raise

    #the search's rounds in Searches' own loop (rust_rounds), a thread a network, while this thread passes on uci's stop and gives the info lines
    def rounds_in_rust(self, run: SearchRun) -> None:
        seconds = None if run.deadline is None else run.deadline - time.perf_counter()
        cpus = self.network_cpus or [None] * len(self.networks)
        networks = [(network, network.rust_network() | ({} if cpu is None else {"cpu": cpu})) for network, cpu in zip(self.networks, cpus)]
        loop = self.searches.round_loop(networks, cuda_runtime().addresses(),
                                        self.rust_round_settings(), run.simulations, seconds, run.extension, run.only_move, run.root,
                                        list(self.round_times))
        while (report := loop.wait(RUST_WAIT_SECONDS)) is None:
            if run.told is not None and run.told.is_set():
                loop.tell_stop()
            if run.info is not None and time.perf_counter() >= run.next_info:
                run.next_info = time.perf_counter() + INFO_SECONDS
                #the search may have ended since the wait, and left Searches
                try:
                    running = self.searches.report_now(GAME)
                except ValueError:
                    running = None
                if running is not None:
                    run.info(self.search_report(running, run.reused))
        round_times, batches, boards = loop.counts()
        self.round_times = deque(round_times, maxlen=ROUND_TIMES)
        self.round_seconds = statistics.median(self.round_times)
        self.rust_counts[0] += batches
        self.rust_counts[1] += boards
        run.report = report

    def rust_round_settings(self) -> dict:
        return dict(game=GAME, leaves=self.leaves, rounds_in_flight=self.rounds_in_flight, grow_leaves=self.grow_leaves,
                    collisions=self.collisions, simulations_per_leaf=SIMULATIONS_PER_LEAF, smart_pruning=self.smart_pruning,
                    smart_pruning_factor=self.smart_pruning_factor, certainty=self.certainty, rounds_at_once=self.rounds_at_once,
                    speed_rounds=SPEED_ROUNDS, round_times_kept=ROUND_TIMES, fall_start_cp=float(FALL_START_CP), fall_full_cp=float(FALL_FULL_CP),
                    max_reported_score=MAX_REPORTED_SCORE, expected_score_per_centipawn=EXPECTED_SCORE_PER_CENTIPAWN,
                    value_sharpness=self.value_sharpness)

    #the search's rounds on one network until the report is in and the network's rounds are back: what to do next is decided under run's lock,
    #and the rounds are gathered, run and finished outside it
    def rounds_on(self, network, run: SearchRun) -> None:
        #launched rounds, oldest first, as (the round's name, the launch or None for a round without leaves, and when it was launched)
        rounds: deque = deque()
        last_answers = None
        #the root's numbers after this network's last descend, for run once its lock is held
        seen = None
        #the rounds left once the report is in have no leaves, but Searches still expects their finish
        while True:
            name = None
            with run.changed:
                if seen is not None:
                    run.seen(seen)
                    seen = None
                while run.report is None and not run.stopping:
                    #a round gathered now comes back after those out (about together, on lanes of their own): stop if that is past the deadline
                    round_times = max(len(rounds), 1) if self.rounds_at_once else len(rounds) + 1
                    told_to_stop = run.told is not None and run.told.is_set()
                    late = run.deadline is not None and time.perf_counter() + round_times * self.round_seconds >= run.deadline
                    #not before the root's first answer, so that a stop always has a move
                    if run.gathered and run.root[0] > 0 and (told_to_stop or late):
                        #once: a search whose score has fallen runs on, unless told to stop
                        if run.extension is not None and not told_to_stop:
                            run.deadline += self.extra_for_fall(run.extension)
                            run.extension = None
                            continue
                        run.stop(self.searches.stop(GAME))
                        break
                    visits = run.root[0]
                    count, in_flight = self.round_leaves(run.simulations, visits)
                    blocked = run.starved == run.rounds_answered and run.rounds_out > len(rounds)
                    if run.rounds_out < in_flight and len(rounds) < self.rounds_in_flight and not blocked:
                        name, run.next_round = run.next_round, run.next_round + 1
                        answered = run.rounds_answered
                        run.rounds_out += 1
                        run.gathered = True
                    elif not rounds:
                        #the rounds out are other networks': one must come back first
                        run.changed.wait()
                        continue
                    break
                if name is None and not rounds:
                    return
            if name is not None:
                #the boards go straight into the lane's pinned buffer (a lean lane's in rows with their answers' columns), or else an array of their own
                lane = network.free_lane() if hasattr(network, "free_lane") else None
                lean = getattr(lane, "lean", False)
                boards = np.empty((count, TOKENS_PER_BOARD), dtype=np.uint8) if lane is None else lane.rows_array if lean else lane.boards_array
                written, seen = self.searches.descend_into(boards, count, min(self.collisions, count), SIMULATIONS_PER_LEAF * count, round=name,
                                                           root_of=GAME, legal_columns=lean)
                if not written:
                    run.starved = answered
                launched_at = time.perf_counter()
                launched = None if not written else network.launch(boards[:written]) if lane is None else network.launch_filled(lane, written)
                rounds.append((name, launched, launched_at))
                continue

            name, launched, launched_at = rounds.popleft()
            answers = np.zeros((0, ANSWER_WIDTH), dtype=np.float32) if launched is None else network.collect(launched)
            answered_at = time.perf_counter()
            with run.changed:
                #under the lock, so that no thread finds the search gone from Searches (done) before its report is in
                reports, root = self.searches.finish_with_root(answers, round=name, root_of=GAME)
                run.seen(root)
                run.rounds_out -= 1
                if launched is not None:
                    run.rounds_answered += 1
                    #rounds come back one after another while the gpu is kept busy; the search's first from its launch
                    self.round_times.append(answered_at - (launched_at if last_answers is None else last_answers))
                    self.round_seconds = statistics.median(self.round_times)
                    last_answers = answered_at
                if run.info is not None and not reports and run.report is None and time.perf_counter() >= run.next_info:
                    run.next_info = time.perf_counter() + INFO_SECONDS
                    running = self.searches.report_now(GAME)
                    if running is not None:
                        run.info(self.search_report(running, run.reused))
                if reports:
                    run.report = reports[0]
                #a round without leaves counts too: all its simulations may have ended the game (a mate found)
                elif run.report is None and self.smart_pruning and not run.stopping:
                    visits, most, second = run.root
                    now = time.perf_counter()
                    run.progress.append((now, visits))
                    decided = most - second > self.simulations_left(run.simulations, run.deadline, run.progress) / self.smart_pruning_factor
                    proven = self.certainty and self.searches.decided(GAME)
                    #once: a search smart pruning would stop runs on if its score has fallen, its time left then larger
                    extra = 0.0
                    if decided and not (run.only_move or proven) and run.extension is not None:
                        extra, run.extension = self.extra_for_fall(run.extension), None
                    if extra > 0:
                        run.deadline += extra
                    elif run.only_move or decided or proven:
                        run.stop(self.searches.stop(GAME))
                run.changed.notify_all()

    #the seconds a search may run on: all of its extension's at FALL_FULL_CP below the last search's score, none at FALL_START_CP or less
    def extra_for_fall(self, extension: tuple[int, float]) -> float:
        last, seconds = extension
        fall = last - score_to_centipawns(sharpen(self.searches.root_score(GAME), 1 / self.value_sharpness))
        share = min(1.0, max(0.0, (fall - FALL_START_CP) / (FALL_FULL_CP - FALL_START_CP)))
        return share * seconds

    #the next round's leaves, and the rounds out at once over all the networks (GROW_LEAVES; at least one); visits: the root's, if known
    def round_leaves(self, simulations: int | None, visits: int | None = None) -> tuple[int, int]:
        count, rounds = self.leaves, self.rounds_in_flight * len(self.networks)
        if not self.grow_leaves:
            return count, rounds
        visits = self.searches.root_visits(GAME)[0] if visits is None else visits
        done_or_left = visits if simulations is None else min(visits, simulations - visits)
        count = max(1, min(self.leaves, done_or_left // self.grow_leaves))
        rounds = rounds if count == self.leaves else 1
        return count, rounds

    #the simulations the search has left: those waiting for the network count as left, so a move is only decided if they could not change it either.
    #With a deadline, the time left at the speed of the last rounds
    @staticmethod
    def simulations_left(simulations: int | None, deadline: float | None, progress: deque) -> float:
        (then, visits_then), (now, visits_now) = progress[0], progress[-1]
        left = math.inf if simulations is None else simulations - visits_now
        if deadline is not None:
            speed = (visits_now - visits_then) / (now - then) if now > then else math.inf
            left = min(left, (deadline - now) * speed)
        return left

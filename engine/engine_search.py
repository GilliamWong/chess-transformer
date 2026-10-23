#one search at a time as fast as the network allows, for the uci engine: the search runs in rust (selfplay_rs.Searches)
#and gathers many leaves a round for one network call, as lc0 does, while the round before is on the gpu. It stops at
#a number of simulations, at a deadline or when told to (uci's stop), or earlier once its move is decided (smart
#pruning), and keeps its tree for the game's next move and the network's answers for positions reached again. On
#several gpus a thread drives each, all on the one tree, as lc0's search threads do
from __future__ import annotations
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import statistics
import threading
import time
import chess
import numpy as np
import selfplay_rs
import torch
from model.encoding import TOKENS_PER_BOARD
from engine.selfplay import ANSWER_WIDTH, SIMULATIONS_PER_LEAF, rust_search_settings
from engine.inference import ExactSizeCompiled, GraphedBatches, chunk_positions, fp16_model, int8_model, small_graph_sizes
from engine.player import ROOT_MOVES_REPORTED, SearchReport, score_to_centipawns
from tools.tablebase import TABLEBASE_PIECES, covers, open_tablebase, open_tablebase_path, ranked_moves, ranked_pieces, wdl_score

#leaves a round (lc0's minibatch), and its descents that may end on a leaf already waiting before it stops gathering.
#With the d512 at 1 s a move 64 leaves do 17.3k simulations/s without the cache (128 16.8k, 32 14.8k). Against one leaf
#a round they lose 0.0001 +-0.0002 expected score a move at 4,000 simulations and 0.0010 at 800 (stockfish depth 14
#judging 2,000 positions)
LEAVES = 64
COLLISIONS = 64
#a round's collisions as they are, not COLLISIONS and at most its leaves, 0 for those
COLLISION_CAP = 0
#with collision_scaling a round's collisions grow with the visits done or left (the fewer) as lc0's do: 1 up to the
#start, MAX_COLLISION_VISITS from the end, along the power between; at most COLLISION_CAP's if it is set
COLLISION_SCALING = False
COLLISION_SCALING_START, COLLISION_SCALING_END, COLLISION_SCALING_POWER = 28, 145_000, 1.25
MAX_COLLISION_VISITS = 80_000
#rounds gathered by search/batch.rs gather_batch on this many threads (a different search), 0 for gather; they spin
#this long for the next round before they sleep
GATHER_WORKERS = 0
GATHER_SPIN_US = 50
#a round is gathered while the last one is on the gpu (+9% simulations/s), and the two run at once (GraphedBatches)
ROUNDS_IN_FLIGHT = 2
#a round takes a leaf per GROW_LEAVES simulations done or left, as lc0 grows its batches with the tree; 0 for fixed
GROW_LEAVES = 16
#with a cap, the rounds out on all the networks hold a leaf (a round counting as its leaves) per this many root visits
#at most, so that a small tree is not spread over as many leaves as many gpus take; 0 for none
VISITS_PER_LEAF_OUT = 0
#as many simulations as the search can do, for a search with only a deadline
UNLIMITED = 2**32 - 1
GAME = 0 #the id of the search in Searches: one game at a time
#a round's network call takes about this long before the first one is timed
FIRST_ROUND_SECONDS = 0.01
#the last rounds' times, whose median is the next round's: one stall (a graph's first capture, the gpu shared) is an
#outlier the median ignores. A running mean carried it into the next searches, which then stopped after one round on
#a clock, whose time never updated the mean (2026-10-12: streaks of 3-5 moves at 6-41 nodes in 10 of 174 games)
ROUND_TIMES = 31
#smart pruning stops once the most visited move leads the second by more than the simulations left / this. lc0 uses
#1.33, betting that the second move cannot get all of them; 1 never changes the move of a search with a number of
#simulations, and with a deadline only when the recent speed misjudges the simulations left
SMART_PRUNING_FACTOR = 1.0
#smart pruning stops no search on the visits before this many seconds after its first answers, lc0's 0.2: its other
#stops (one move, a proven root) stay at once
SMART_PRUNING_WAIT = 0.0
#with certainty, smart pruning stops at once when the proofs leave one root move not lost, as lc0 does (the move choice
#puts the losses last)
STOP_AT_ONE_MOVE_LEFT = False
#a search on the clock whose score has fallen more than FALL_START_CP below the last search's runs on when it would
#stop, by all the time its clock lets it add at a fall of FALL_FULL_CP: our mistakes cluster after our eval falls
FALL_START_CP = 10
FALL_FULL_CP = 30
#rounds whose speed estimates the simulations a deadline leaves
SPEED_ROUNDS = 8
#a search's info lines for the gui while it runs, one about this often (lc0 sends one a second)
INFO_SECONDS = 1.0
#the network's answers kept by position (search.rs Cache), ~260 bytes each (1.3 GB): replaying 6 tournament games, 4M
#rather than self-play's 500k gave 18% more nodes a search on the mean (3.5% on the median) from as many network
#calls. The engine's own: every self-play worker has a selfplay.CACHE_POSITIONS cache
CACHE_POSITIONS = 5_000_000
#the threads advance walks and copies the kept tree on after a move, while the opponent thinks (Searches' advance_threads)
ADVANCE_THREADS = 4
#the engine's own changes to mcts.py's search, which self-play keeps (tools/play/variant_match.py tunes them). One leaf
#a round (uci.py --leaves 1, evaluate_elo) stays mcts.py's search. c_puct 0.4 won +78 elo at 4k nodes a move and +46 at
#30k in self-play, but against stockfish it was -9 [-48, +30] and against lc0 it trailed (2026-09-29): self-play's bias
#towards narrow search. value_sharpness 3: the value labels' human scale is ~3x flatter than stockfish's win rate, and
#the d768 sharpened 3x scored +42 +- 29 elo against BT4 at 1000 nodes in lc0's search (2026-10-11). certainty: search.rs
#SearchSettings' (with smart pruning a search whose root is proven stops); without it a mate the search had found lost
#to a move with more visits (2026-10-13)
ENGINE_SETTINGS: dict = {"value_sharpness": 3.0, "certainty": True}


#search.rs sharpen: the value's logit times sharpness; 1 / sharpness undoes it, and 1 leaves the score exactly as it was
def sharpen(score: float, sharpness: float) -> float:
    return score if sharpness == 1 or score in (0.0, 1.0) else 1 / (1 + ((1 - score) / score) ** sharpness)

#Searches' settings: mcts.py's with the engine's own (own), then changes (parse_changes') over them
def engine_search_settings(use_tablebase: bool, changes: dict | None = None, own: bool = True, tablebase_pieces: int = TABLEBASE_PIECES,
                           syzygy_path: str | None = None) -> dict:
    return {**rust_search_settings(use_tablebase, tablebase_pieces, syzygy_path), **(ENGINE_SETTINGS if own else {}), **(changes or {})}

#"name=value,..." as changes to settings, each value of its setting's type
def parse_changes(text: str, settings: dict) -> dict:
    changes = {}
    for change in filter(None, text.split(",")):
        name, value = change.split("=")
        if name not in settings:
            raise ValueError(f"{name} is not a setting, these are: {', '.join(sorted(settings))}")
        kind = type(settings[name])
        if kind is bool:
            if value not in ("true", "false"):
                raise ValueError(f"{name} is true or false, not {value}")
            changes[name] = value == "true"
        else:
            changes[name] = kind(float(value))
    return changes

#int8 is the engine's, not for self-play or labels: its answers depend on the rest of the batch
PRECISIONS = {"fp16": fp16_model, "int8": int8_model}
ENGINE_PRECISION = "int8"
#the int8 network's blocks kept in fp16 besides the last, those whose int8 costs the policy most (quantization.py): one
#block can cost as much as all the others (bt4prior_final's 26th). 0 is every block but the last in int8
ENGINE_FP16_BLOCKS = 1


#the gpus whose lanes are lean by default (inference.Lane): sm_120's (the 5070 Ti, the 5090), where they were timed;
#others keep whole rows until timed
LEAN_CAPABILITIES = {(12, 0)}


#the network in a precision as cuda graphs for a round's leaves, from one to max_batch, a lane a round out; fp16_blocks
#for int8; on the gpu numbered device, or else the current one; blocking_wait see GraphedBatches. lean: lean lanes, by
#default on the gpus of LEAN_CAPABILITIES
def engine_network(checkpoint_path: Path, max_batch: int = LEAVES, precision: str = ENGINE_PRECISION,
                   fp16_blocks: int = ENGINE_FP16_BLOCKS, device: int | None = None, blocking_wait: bool = False,
                   lean: bool | None = None):
    with torch.cuda.device(device):
        model = int8_model(checkpoint_path, fp16_blocks) if precision == "int8" else PRECISIONS[precision](checkpoint_path)
        lanes = ROUNDS_IN_FLIGHT if max_batch > 1 else 1
        chunk = chunk_positions(model.d_model, lanes)
        sizes = small_graph_sizes(max_batch, chunk)
        #one board a round (mcts.py's search) never fills the largest graph
        compiled = ExactSizeCompiled(model, min(sizes[-1], chunk)) if max_batch > 1 else torch.compile(model, dynamic=True)
        return GraphedBatches(compiled, sizes, lanes, chunk, blocking_wait, lean_lanes() if lean is None else lean)


#whether the current gpu's lanes are lean by default
def lean_lanes() -> bool:
    return torch.cuda.get_device_capability() in LEAN_CAPABILITIES


#what the threads driving a search's networks share, behind changed's lock; changed is notified as a round comes back
#or the search stops
class SearchRun:
    def __init__(self, simulations: int | None, deadline: float | None, extension: tuple[int, float] | None, only_move: bool,
                 told: threading.Event | None, info: Callable[[SearchReport], None] | None, next_info: float,
                 root: tuple[int, int, int]):
        self.simulations, self.deadline, self.extension, self.only_move = simulations, deadline, extension, only_move
        self.told = told #set to stop the search (uci's stop)
        self.info, self.next_info = info, next_info #EngineSearch.search's, and when its next report is due
        self.changed = threading.Condition()
        self.report = None
        self.stopping = self.gathered = False
        self.rounds_out = 0 #on every network
        self.rounds_answered = 0 #rounds with leaves finished
        #rounds_answered when a round last found no leaves, every descent ending on a leaf already out: no network gathers
        #again until another round is back, rather than holding Searches with empty rounds
        self.starved = None
        self.next_round = 0 #the next round's name in Searches
        self.progress: deque = deque(maxlen=SPEED_ROUNDS) #(time, root visits) after the last rounds
        self.first_answers = None #when the search's first round came back
        #Searches.root_visits' numbers as the newest descend or finish left them, so that no thread asks for them
        self.root = root

    #numbers a descend or finish gave (None once the search is done): visits only grow, so the most is the newest
    def seen(self, root: tuple[int, int, int] | None) -> None:
        if root is not None and root[0] >= self.root[0]:
            self.root = root

    #the search stops gathering; its report, or None while its leaves are out (the finish of the last of them has it)
    def stop(self, report) -> None:
        self.stopping, self.report = True, report
        self.changed.notify_all()


class EngineSearch:
    #network has GraphedBatches' launch and collect (inference.ImmediateBatches on the cpu), or is a list of them, one a
    #gpu, each driven by a thread of its own. One leaf a round is mcts.py's search: its settings (without
    #ENGINE_SETTINGS), a fresh tree every move, no second round on the network and no smart pruning. smart_pruning False
    #searches the whole budget even once the move is decided (lc0's SmartPruningFactor=0), for matches that give both
    #engines exactly the same time; smart_pruning_factor, smart_pruning_wait and stop_at_one_move_left are
    #SMART_PRUNING_FACTOR's, SMART_PRUNING_WAIT's and STOP_AT_ONE_MOVE_LEFT's, visits_per_leaf_out VISITS_PER_LEAF_OUT's
    #cap. use_tablebase: the tables of positions of up to tablebase_pieces pieces, from the folders of syzygy_path (a
    #SyzygyPath) rather than the downloaded ones if it is given. settings: changes to engine_search_settings.
    #cache_positions: the answer cache's size; memory_limit_mb: Searches' (lc0's RamLimitMb), 0 for none; advance_threads:
    #ADVANCE_THREADS'
    def __init__(self, network, leaves: int = LEAVES, use_tablebase: bool = False, smart_pruning: bool = True,
                 tablebase_pieces: int = TABLEBASE_PIECES, settings: dict | None = None, grow_leaves: int = GROW_LEAVES,
                 smart_pruning_factor: float = SMART_PRUNING_FACTOR, smart_pruning_wait: float = SMART_PRUNING_WAIT,
                 stop_at_one_move_left: bool = STOP_AT_ONE_MOVE_LEFT, cache_positions: int = CACHE_POSITIONS, syzygy_path: str | None = None,
                 memory_limit_mb: int = 0, advance_threads: int = ADVANCE_THREADS, visits_per_leaf_out: int = VISITS_PER_LEAF_OUT,
                 collision_cap: int = COLLISION_CAP, gather_workers: int = GATHER_WORKERS, gather_spin_us: int = GATHER_SPIN_US,
                 collision_scaling: bool = COLLISION_SCALING):
        self.networks = network if isinstance(network, list) else [network]
        self.leaves = leaves
        batched = leaves > 1
        if len(self.networks) > 1 and not batched:
            raise ValueError("one leaf a round is mcts.py's search, which runs on one network")
        self.round_simulations = SIMULATIONS_PER_LEAF * leaves
        self.collisions = COLLISIONS if batched else 0
        self.collision_cap = collision_cap if batched else 0
        self.collision_scaling = batched and collision_scaling
        self.rounds_in_flight = ROUNDS_IN_FLIGHT if batched else 1 #on each network
        self.grow_leaves = grow_leaves if batched else 0
        self.visits_per_leaf_out = visits_per_leaf_out if batched else 0
        #rounds on lanes of their own (GraphedBatches') run at once
        self.rounds_at_once = batched and len(getattr(self.networks[0], "lanes", ())) > 1
        self.smart_pruning = batched and smart_pruning
        self.smart_pruning_factor, self.smart_pruning_wait = smart_pruning_factor, smart_pruning_wait
        self.stop_at_one_move_left = stop_at_one_move_left
        search_settings = engine_search_settings(use_tablebase, settings, own=batched, tablebase_pieces=tablebase_pieces, syzygy_path=syzygy_path)
        self.searches = selfplay_rs.Searches(**search_settings, tree_reuse=batched, cache_positions=cache_positions, memory_limit_mb=memory_limit_mb,
                                             advance_threads=advance_threads, gather_workers=gather_workers if batched else 0,
                                             gather_spin_us=gather_spin_us)
        #reports give the score on the network's own scale, as a gui showed and adjudicated it before sharpening
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
        #the threads that drive the networks after the first, which the searching thread drives itself
        self.threads = ThreadPoolExecutor(len(self.networks) - 1) if len(self.networks) > 1 else None

    #a new game (uci's ucinewgame): its first search starts a fresh tree with an empty cache, as lc0's does
    def new_game(self) -> None:
        self.searches.new_game(GAME)

    #once the engine's move is sent, while the opponent thinks: the kept tree carried on to the game after it (board) on
    #a thread of Searches', so that the next search's start only moves its root down the reply (Searches.advance).
    #False when there is no tree to carry on
    def advance(self, board: chess.Board) -> bool:
        return self.leaves > 1 and self.searches.advance(GAME, board.root().fen(), [move.uci() for move in board.move_stack])

    #searches until `simulations` (counting those of a reused tree, as lc0's go nodes does), until `seconds` have
    #passed or until `stop` is set (uci's reader thread sets it), whichever comes first; at least one of the three.
    #extension: (the last search's score in centipawns, the seconds this one may add) for a search on the clock whose
    #score falls (FALL_START_CP). info gets the report as it stands about every INFO_SECONDS, for the gui's info lines
    def search(self, board: chess.Board, simulations: int | None = None, seconds: float | None = None,
               extension: tuple[int, float] | None = None, stop: threading.Event | None = None,
               info: Callable[[SearchReport], None] | None = None) -> SearchReport:
        if simulations is None and seconds is None and stop is None:
            raise ValueError("a search needs a number of simulations, a time or a stop")
        started = time.perf_counter()
        #the tables know the result and the best move
        if self.tablebase is not None and covers(board, self.ranked_pieces):
            best = ranked_moves(self.tablebase, board)[0]
            return SearchReport(wdl_score(best.wdl), [best.move], 0, [])

        self.searches.start(GAME, board.root().fen(), [move.uci() for move in board.move_stack], UNLIMITED if simulations is None else simulations)
        deadline = started + seconds if seconds is not None else None
        run = SearchRun(simulations, deadline, extension if deadline is not None else None, board.legal_moves.count() == 1, stop, info,
                        started + INFO_SECONDS, self.searches.root_visits(GAME))
        driven = [self.threads.submit(self.drive, network, run) for network in self.networks[1:]] if self.threads else []
        try:
            self.drive(self.networks[0], run)
        finally:
            for drive in driven:
                drive.result()
        return self.search_report(run.report)

    #a report of Searches' as the engine gives it: the score on the network's own scale, the line and root moves as moves
    def search_report(self, report: tuple) -> SearchReport:
        _, line, score, root_moves, visits = report
        return SearchReport(sharpen(score, 1 / self.value_sharpness), [chess.Move.from_uci(move) for move in line], visits,
                            [(chess.Move.from_uci(move), move_visits) for move, move_visits in root_moves[:ROOT_MOVES_REPORTED]])

    #rounds_on, a failure stopping the other networks' rounds too
    def drive(self, network, run: SearchRun) -> None:
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

    #the search's rounds on one network until the report is in and the network's rounds are back: what to do next is
    #decided under run's lock, and the rounds are gathered, run and finished outside it
    def rounds_on(self, network, run: SearchRun) -> None:
        #launched rounds, oldest first, as (the round's name, the launch or None for a round without leaves, and when it
        #was launched)
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
                    #the round gathered now would come back after those on the gpu: stop if that is past the deadline.
                    #Rounds on lanes of their own run at once and come back about together, so it comes a round time
                    #after the newest
                    round_times = max(len(rounds), 1) if self.rounds_at_once else len(rounds) + 1
                    told_to_stop = run.told is not None and run.told.is_set()
                    late = run.deadline is not None and time.perf_counter() + round_times * self.round_seconds >= run.deadline
                    if run.gathered and (told_to_stop or late):
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
                collisions = self.round_collisions(count, run.simulations, visits)
                #the boards go straight into the lane's pinned buffer (GraphedBatches'), a lean lane's in rows with their
                #answers' columns, or else an array of their own
                lane = network.free_lane() if hasattr(network, "free_lane") else None
                lean = getattr(lane, "lean", False)
                boards = np.empty((count, TOKENS_PER_BOARD), dtype=np.uint8) if lane is None else lane.rows_array if lean else lane.boards_array
                written, seen = self.searches.descend_into(boards, count, collisions, SIMULATIONS_PER_LEAF * count, round=name, root_of=GAME,
                                                           legal_columns=lean)
                if not written:
                    run.starved = answered
                launched_at = time.perf_counter()
                launched = None if not written else network.launch(boards[:written]) if lane is None else network.launch_filled(lane, written)
                rounds.append((name, launched, launched_at))
                continue

            name, launched, launched_at = rounds.popleft()
            answers = np.zeros((0, ANSWER_WIDTH), dtype=np.float32) if launched is None else network.collect(launched)
            answered = time.perf_counter()
            with run.changed:
                #under the lock, so that no thread finds the search gone from Searches (done) before its report is in
                reports, root = self.searches.finish_with_root(answers, round=name, root_of=GAME)
                run.seen(root)
                run.rounds_out -= 1
                if launched is not None:
                    run.rounds_answered += 1
                    #rounds come back one after another while the gpu is kept busy; the search's first from its launch
                    self.round_times.append(answered - (launched_at if last_answers is None else last_answers))
                    self.round_seconds = statistics.median(self.round_times)
                    last_answers = answered
                if run.info is not None and not reports and run.report is None and time.perf_counter() >= run.next_info:
                    run.next_info = time.perf_counter() + INFO_SECONDS
                    running = self.searches.report_now(GAME)
                    if running is not None:
                        run.info(self.search_report(running))
                if reports:
                    run.report = reports[0]
                #a round without leaves counts too: all its simulations may have ended the game (a mate found)
                elif run.report is None and self.smart_pruning and not run.stopping:
                    visits, most, second = run.root
                    now = time.perf_counter()
                    run.progress.append((now, visits))
                    run.first_answers = now if run.first_answers is None else run.first_answers
                    decided = (most - second > self.simulations_left(run.simulations, run.deadline, run.progress) / self.smart_pruning_factor
                               and now - run.first_answers >= self.smart_pruning_wait)
                    proven = self.certainty and (self.searches.decided(GAME)
                                                 or self.stop_at_one_move_left and self.searches.moves_not_lost(GAME) <= 1)
                    #once: a search smart pruning would stop runs on if its score has fallen, its time left then larger
                    extra = 0.0
                    if decided and not (run.only_move or proven) and run.extension is not None:
                        extra, run.extension = self.extra_for_fall(run.extension), None
                    if extra > 0:
                        run.deadline += extra
                    elif run.only_move or decided or proven:
                        run.stop(self.searches.stop(GAME))
                run.changed.notify_all()

    #the seconds a search may run on: all of its extension's at a fall of FALL_FULL_CP below the last search's score, none
    #at FALL_START_CP or less (the score as the report gives it, on the network's own scale)
    def extra_for_fall(self, extension: tuple[int, float]) -> float:
        last, seconds = extension
        fall = last - score_to_centipawns(sharpen(self.searches.root_score(GAME), 1 / self.value_sharpness))
        share = min(1.0, max(0.0, (fall - FALL_START_CP) / (FALL_FULL_CP - FALL_START_CP)))
        return share * seconds

    #the next round's leaves, and the rounds out at once over all the networks: one until a round takes all of them, and
    #with a cap (visits_per_leaf_out) as many as the root's visits allow, one at least. visits: the root's, if known
    def round_leaves(self, simulations: int | None, visits: int | None = None) -> tuple[int, int]:
        count, rounds = self.leaves, self.rounds_in_flight * len(self.networks)
        if not self.grow_leaves and not self.visits_per_leaf_out:
            return count, rounds
        visits = self.searches.root_visits(GAME)[0] if visits is None else visits
        if self.grow_leaves:
            done_or_left = visits if simulations is None else min(visits, simulations - visits)
            count = max(1, min(self.leaves, done_or_left // self.grow_leaves))
            rounds = rounds if count == self.leaves else 1
        if self.visits_per_leaf_out:
            rounds = max(1, min(rounds, visits // (self.visits_per_leaf_out * count)))
        return count, rounds

    #the collisions a round of count leaves may have: collision_cap's, COLLISIONS' at most its leaves, or with
    #collision_scaling lc0's CalculateCollisionsLeft. visits: the root's, if known
    def round_collisions(self, count: int, simulations: int | None, visits: int | None = None) -> int:
        if not self.collision_scaling:
            return self.collision_cap or min(self.collisions, count)
        visits = self.searches.root_visits(GAME)[0] if visits is None else visits
        done_or_left = visits if simulations is None else min(visits, simulations - visits)
        if done_or_left <= COLLISION_SCALING_START:
            collisions = 1
        elif done_or_left >= COLLISION_SCALING_END:
            collisions = MAX_COLLISION_VISITS
        else:
            share = (done_or_left - COLLISION_SCALING_START) / (COLLISION_SCALING_END - COLLISION_SCALING_START)
            collisions = int(1 + (MAX_COLLISION_VISITS - 1) * share ** COLLISION_SCALING_POWER)
        return min(collisions, self.collision_cap) if self.collision_cap else collisions

    #the simulations the search has left: those waiting for the network count as left, so a move is only decided if
    #they could not change it either. With a deadline, the time left at the speed of the last rounds
    @staticmethod
    def simulations_left(simulations: int | None, deadline: float | None, progress: deque) -> float:
        (_, visits), left = progress[-1], math.inf
        if simulations is not None:
            left = simulations - visits
        if deadline is not None:
            (then, visits_then), (now, visits_now) = progress[0], progress[-1]
            speed = (visits_now - visits_then) / (now - then) if now > then else math.inf
            left = min(left, (deadline - now) * speed)
        return left

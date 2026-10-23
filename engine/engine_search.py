#one search at a time as fast as the network allows, for the uci engine: the search runs in rust (selfplay_rs.Searches)
#and gathers many leaves a round for one network call, as lc0 does, while the round before is on the gpu. It stops at
#a number of simulations, at a deadline or when told to (uci's stop), or earlier once its move is decided (smart
#pruning), and keeps its tree for the game's next move and the network's answers for positions reached again
from __future__ import annotations
from collections import deque
from collections.abc import Callable
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
from engine.inference import ExactSizeCompiled, GraphedBatches, fp16_model, int8_model, small_graph_sizes
from engine.player import ROOT_MOVES_REPORTED, SearchReport, score_to_centipawns
from tools.tablebase import TABLEBASE_PIECES, covers, open_tablebase, open_tablebase_path, ranked_moves, ranked_pieces, wdl_score

#leaves a round (lc0's minibatch), and its descents that may end on a leaf already waiting before it stops gathering.
#With the d512 at 1 s a move 64 leaves do 17.3k simulations/s without the cache (128 16.8k, 32 14.8k). Against one leaf
#a round they lose 0.0001 +-0.0002 expected score a move at 4,000 simulations and 0.0010 at 800 (stockfish depth 14
#judging 2,000 positions)
LEAVES = 64
COLLISIONS = 64
#a round is gathered while the last one is on the gpu (+9% simulations/s), and the two run at once (GraphedBatches)
ROUNDS_IN_FLIGHT = 2
#a round takes a leaf per GROW_LEAVES simulations done or left, as lc0 grows its batches with the tree; 0 for fixed
GROW_LEAVES = 16
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


#the network in a precision as cuda graphs for a round's leaves, from one to max_batch, a lane a round out
def engine_network(checkpoint_path: Path, max_batch: int = LEAVES, precision: str = ENGINE_PRECISION):
    model = PRECISIONS[precision](checkpoint_path)
    sizes = small_graph_sizes(max_batch)
    #one board a round (mcts.py's search) never fills the largest graph
    compiled = ExactSizeCompiled(model, sizes[-1]) if max_batch > 1 else torch.compile(model, dynamic=True)
    return GraphedBatches(compiled, sizes, ROUNDS_IN_FLIGHT if max_batch > 1 else 1)


class EngineSearch:
    #network has GraphedBatches' launch and collect (inference.ImmediateBatches on the cpu). One leaf a round is
    #mcts.py's search: its settings (without ENGINE_SETTINGS), a fresh tree every move, no second round on the
    #network and no smart pruning. smart_pruning False searches the whole budget even once the move is decided (lc0's
    #SmartPruningFactor=0), for matches that give both engines exactly the same time; smart_pruning_factor,
    #smart_pruning_wait and stop_at_one_move_left are SMART_PRUNING_FACTOR's, SMART_PRUNING_WAIT's and
    #STOP_AT_ONE_MOVE_LEFT's. use_tablebase: the tables of positions of up to tablebase_pieces pieces, from the folders
    #of syzygy_path (a SyzygyPath) rather than the downloaded ones if it is given. settings: changes to
    #engine_search_settings. cache_positions: the answer cache's size; memory_limit_mb: Searches' (lc0's RamLimitMb), 0
    #for none
    def __init__(self, network, leaves: int = LEAVES, use_tablebase: bool = False, smart_pruning: bool = True,
                 tablebase_pieces: int = TABLEBASE_PIECES, settings: dict | None = None, grow_leaves: int = GROW_LEAVES,
                 smart_pruning_factor: float = SMART_PRUNING_FACTOR, smart_pruning_wait: float = SMART_PRUNING_WAIT,
                 stop_at_one_move_left: bool = STOP_AT_ONE_MOVE_LEFT, cache_positions: int = CACHE_POSITIONS, syzygy_path: str | None = None,
                 memory_limit_mb: int = 0):
        self.network = network
        self.leaves = leaves
        batched = leaves > 1
        self.round_simulations = SIMULATIONS_PER_LEAF * leaves
        self.collisions = COLLISIONS if batched else 0
        self.rounds_in_flight = ROUNDS_IN_FLIGHT if batched else 1
        self.grow_leaves = grow_leaves if batched else 0
        #rounds on lanes of their own (GraphedBatches') run at once
        self.rounds_at_once = batched and len(getattr(network, "lanes", ())) > 1
        self.smart_pruning = batched and smart_pruning
        self.smart_pruning_factor, self.smart_pruning_wait = smart_pruning_factor, smart_pruning_wait
        self.stop_at_one_move_left = stop_at_one_move_left
        search_settings = engine_search_settings(use_tablebase, settings, own=batched, tablebase_pieces=tablebase_pieces, syzygy_path=syzygy_path)
        self.searches = selfplay_rs.Searches(**search_settings, tree_reuse=batched, cache_positions=cache_positions, memory_limit_mb=memory_limit_mb)
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
        if deadline is None:
            extension = None
        only_move = board.legal_moves.count() == 1
        #launched rounds, oldest first, as (the launch, None for a round without leaves, and when it was launched)
        rounds: deque = deque()
        gathered = stopping = False
        report, last_answers = None, None
        progress: deque = deque(maxlen=SPEED_ROUNDS) #(time, root visits) after the last rounds
        first_answers = None #when the search's first round came back
        next_info = started + INFO_SECONDS
        #the rounds left once the report is in have no leaves, but Searches still expects their finish
        while report is None or rounds:
            if report is None and not stopping:
                #the round gathered now would come back after those on the gpu: stop if that is past the deadline. Rounds
                #on lanes of their own run at once and come back about together, so it comes a round time after the newest
                round_times = max(len(rounds), 1) if self.rounds_at_once else len(rounds) + 1
                told_to_stop = stop is not None and stop.is_set()
                if gathered and (told_to_stop or deadline is not None and time.perf_counter() + round_times * self.round_seconds >= deadline):
                    #once: a search whose score has fallen runs on, unless told to stop
                    if extension is not None and not told_to_stop:
                        deadline += self.extra_for_fall(extension)
                        extension = None
                        continue
                    stopping = True
                    report = self.searches.stop(GAME)
                    continue
                count, in_flight = self.round_leaves(simulations)
                if len(rounds) < in_flight:
                    #a bytearray: torch warns about arrays it cannot write to
                    leaves = bytearray(self.searches.descend(count, min(self.collisions, count), SIMULATIONS_PER_LEAF * count))
                    boards = np.frombuffer(leaves, dtype=np.uint8).reshape(-1, TOKENS_PER_BOARD)
                    launched_at = time.perf_counter()
                    rounds.append((self.network.launch(boards) if len(boards) else None, launched_at))
                    gathered = True
                    continue

            launched, launched_at = rounds.popleft()
            if launched is None:
                answers = np.zeros((0, ANSWER_WIDTH), dtype=np.float32)
            else:
                answers = self.network.collect(launched)
                #rounds come back one after another while the gpu is kept busy; the search's first from its launch
                now = time.perf_counter()
                self.round_times.append(now - (launched_at if last_answers is None else last_answers))
                self.round_seconds = statistics.median(self.round_times)
                last_answers = now
            reports = self.searches.finish(answers)
            if info is not None and not reports and report is None and time.perf_counter() >= next_info:
                next_info = time.perf_counter() + INFO_SECONDS
                running = self.searches.report_now(GAME)
                if running is not None:
                    info(self.search_report(running))
            if reports:
                report = reports[0]
            #a round without leaves counts too: all its simulations may have ended the game (a mate found)
            elif report is None and self.smart_pruning and not stopping:
                visits, most, second = self.searches.root_visits(GAME)
                now = time.perf_counter()
                progress.append((now, visits))
                first_answers = now if first_answers is None else first_answers
                decided = (most - second > self.simulations_left(simulations, deadline, progress) / self.smart_pruning_factor
                           and now - first_answers >= self.smart_pruning_wait)
                proven = self.certainty and (self.searches.decided(GAME) or self.stop_at_one_move_left and self.searches.moves_not_lost(GAME) <= 1)
                #once: a search smart pruning would stop runs on if its score has fallen, its time left then larger
                extra = 0.0
                if decided and not (only_move or proven) and extension is not None:
                    extra, extension = self.extra_for_fall(extension), None
                if extra > 0:
                    deadline += extra
                elif only_move or decided or proven:
                    stopping = True
                    report = self.searches.stop(GAME)

        return self.search_report(report)

    #a report of Searches' as the engine gives it: the score on the network's own scale, the line and root moves as moves
    def search_report(self, report: tuple) -> SearchReport:
        _, line, score, root_moves, visits = report
        return SearchReport(sharpen(score, 1 / self.value_sharpness), [chess.Move.from_uci(move) for move in line], visits,
                            [(chess.Move.from_uci(move), move_visits) for move, move_visits in root_moves[:ROOT_MOVES_REPORTED]])

    #the seconds a search may run on: all of its extension's at a fall of FALL_FULL_CP below the last search's score, none
    #at FALL_START_CP or less (the score as the report gives it, on the network's own scale)
    def extra_for_fall(self, extension: tuple[int, float]) -> float:
        last, seconds = extension
        fall = last - score_to_centipawns(sharpen(self.searches.root_score(GAME), 1 / self.value_sharpness))
        share = min(1.0, max(0.0, (fall - FALL_START_CP) / (FALL_FULL_CP - FALL_START_CP)))
        return share * seconds

    #the next round's leaves, and the rounds out at once: one until a round takes all of them
    def round_leaves(self, simulations: int | None) -> tuple[int, int]:
        if not self.grow_leaves:
            return self.leaves, self.rounds_in_flight
        visits = self.searches.root_visits(GAME)[0]
        done_or_left = visits if simulations is None else min(visits, simulations - visits)
        count = max(1, min(self.leaves, done_or_left // self.grow_leaves))
        return count, self.rounds_in_flight if count == self.leaves else 1

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

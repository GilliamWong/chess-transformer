#one search at a time as fast as the network allows, for the uci engine: the search runs in rust (selfplay_rs.Searches)
#and gathers many leaves a round for one network call, as lc0 does, while the round before is on the gpu. It stops at
#a number of simulations or at a deadline, or earlier once its move is decided (smart pruning), and keeps its tree for
#the game's next move and the network's answers for positions reached again
from __future__ import annotations
from collections import deque
import math
from pathlib import Path
import time
import chess
import numpy as np
import selfplay_rs
import torch
from model.encoding import TOKENS_PER_BOARD
from engine.selfplay import ANSWER_WIDTH, SIMULATIONS_PER_LEAF, rust_search_settings
from engine.inference import ExactSizeCompiled, GraphedBatches, fp16_model, int8_model, small_graph_sizes
from engine.player import ROOT_MOVES_REPORTED, SearchReport
from tools.tablebase import TABLEBASE_PIECES, covers, open_tablebase, ranked_moves, ranked_pieces, wdl_score

#leaves a round (lc0's minibatch), and its descents that may end on a leaf already waiting before it stops gathering.
#With the d512 at 1 s a move 64 leaves do 17.3k simulations/s without the cache (128 16.8k, 32 14.8k). Against one leaf
#a round they lose 0.0001 +-0.0002 expected score a move at 4,000 simulations and 0.0010 at 800 (stockfish depth 14
#judging 2,000 positions)
LEAVES = 64
COLLISIONS = 64
#a round is gathered while the last one is on the gpu (+9% simulations/s), and the two run at once (GraphedBatches)
ROUNDS_IN_FLIGHT = 2
#as many simulations as the search can do, for a search with only a deadline
UNLIMITED = 2**32 - 1
GAME = 0 #the id of the search in Searches: one game at a time
#a round's network call takes about this long before the first one is timed
FIRST_ROUND_SECONDS = 0.01
#smart pruning stops once the most visited move leads the second by more than the simulations left / this. lc0 uses
#1.33, betting that the second move cannot get all of them; 1 never changes the move of a search with a number of
#simulations, and with a deadline only when the recent speed misjudges the simulations left
SMART_PRUNING_FACTOR = 1.0
#rounds whose speed estimates the simulations a deadline leaves
SPEED_ROUNDS = 8
#the network's answers kept by position (search.rs Cache), ~260 bytes each (1.3 GB): replaying 6 tournament games, 4M
#rather than self-play's 500k gave 18% more nodes a search on the mean (3.5% on the median) from as many network
#calls. The engine's own: every self-play worker has a selfplay.CACHE_POSITIONS cache
CACHE_POSITIONS = 5_000_000
#the engine's own changes to mcts.py's search, which self-play keeps (tools/play/variant_match.py tunes them). One leaf
#a round (uci.py --leaves 1, evaluate_elo) stays mcts.py's search. c_puct 0.4 won +78 elo at 4k nodes a move and +46 at
#30k in self-play, but against stockfish it was -9 [-48, +30] and against lc0 it trailed (2026-09-29): self-play's bias
#towards narrow search. value_sharpness 3: the value labels' human scale is ~3x flatter than stockfish's win rate, and
#the d768 sharpened 3x scored +42 +- 29 elo against BT4 at 1000 nodes in lc0's search (2026-10-11)
ENGINE_SETTINGS: dict = {"value_sharpness": 3.0}


#search.rs sharpen: the value's logit times sharpness; 1 / sharpness undoes it, and 1 leaves the score exactly as it was
def sharpen(score: float, sharpness: float) -> float:
    return score if sharpness == 1 or score in (0.0, 1.0) else 1 / (1 + ((1 - score) / score) ** sharpness)

#Searches' settings: mcts.py's with the engine's own (own), then changes (parse_changes') over them
def engine_search_settings(use_tablebase: bool, changes: dict | None = None, own: bool = True, tablebase_pieces: int = TABLEBASE_PIECES) -> dict:
    return {**rust_search_settings(use_tablebase, tablebase_pieces), **(ENGINE_SETTINGS if own else {}), **(changes or {})}

#"name=value,..." as changes to settings, each value of its setting's type
def parse_changes(text: str, settings: dict) -> dict:
    changes = {}
    for change in filter(None, text.split(",")):
        name, value = change.split("=")
        if name not in settings:
            raise ValueError(f"{name} is not a setting, these are: {', '.join(sorted(settings))}")
        kind = type(settings[name])
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
    #SmartPruningFactor=0), for matches that give both engines exactly the same time. use_tablebase: the tables of
    #positions of up to tablebase_pieces pieces. settings: changes to engine_search_settings
    def __init__(self, network, leaves: int = LEAVES, use_tablebase: bool = False, smart_pruning: bool = True,
                 tablebase_pieces: int = TABLEBASE_PIECES, settings: dict | None = None):
        self.network = network
        self.leaves = leaves
        batched = leaves > 1
        self.round_simulations = SIMULATIONS_PER_LEAF * leaves
        self.collisions = COLLISIONS if batched else 0
        self.rounds_in_flight = ROUNDS_IN_FLIGHT if batched else 1
        #rounds on lanes of their own (GraphedBatches') run at once
        self.rounds_at_once = batched and len(getattr(network, "lanes", ())) > 1
        self.smart_pruning = batched and smart_pruning
        search_settings = engine_search_settings(use_tablebase, settings, own=batched, tablebase_pieces=tablebase_pieces)
        self.searches = selfplay_rs.Searches(**search_settings, tree_reuse=batched, cache_positions=CACHE_POSITIONS)
        #reports give the score on the network's own scale, as a gui showed and adjudicated it before sharpening
        self.value_sharpness = search_settings.get("value_sharpness", 1.0)
        self.tablebase = open_tablebase(tablebase_pieces) if use_tablebase else None
        #the root takes the tables' move up to the size they can rank; a bigger one they cover is searched
        self.ranked_pieces = ranked_pieces(tablebase_pieces) if use_tablebase else 0
        self.round_seconds = FIRST_ROUND_SECONDS

    #searches until `simulations` (counting those of a reused tree, as lc0's go nodes does) or until `seconds` have
    #passed, whichever comes first; at least one of the two
    def search(self, board: chess.Board, simulations: int | None = None, seconds: float | None = None) -> SearchReport:
        if simulations is None and seconds is None:
            raise ValueError("a search needs a number of simulations or a time")
        started = time.perf_counter()
        #the tables know the result and the best move
        if self.tablebase is not None and covers(board, self.ranked_pieces):
            best = ranked_moves(self.tablebase, board)[0]
            return SearchReport(wdl_score(best.wdl), [best.move], 0, [])

        self.searches.start(GAME, board.root().fen(), [move.uci() for move in board.move_stack], UNLIMITED if simulations is None else simulations)
        deadline = started + seconds if seconds is not None else None
        only_move = board.legal_moves.count() == 1
        rounds: deque = deque() #launched rounds, oldest first, None for a round without leaves
        gathered = stopping = False
        report, last_answers = None, None
        progress: deque = deque(maxlen=SPEED_ROUNDS) #(time, root visits) after the last rounds
        #the rounds left once the report is in have no leaves, but Searches still expects their finish
        while report is None or rounds:
            if report is None and not stopping:
                #the round gathered now would come back after those on the gpu: stop if that is past the deadline. Rounds
                #on lanes of their own run at once and come back about together, so it comes a round time after the newest
                round_times = max(len(rounds), 1) if self.rounds_at_once else len(rounds) + 1
                if gathered and deadline is not None and time.perf_counter() + round_times * self.round_seconds >= deadline:
                    stopping = True
                    report = self.searches.stop(GAME)
                    continue
                if len(rounds) < self.rounds_in_flight:
                    #a bytearray: torch warns about arrays it cannot write to
                    leaves = bytearray(self.searches.descend(self.leaves, self.collisions, self.round_simulations))
                    boards = np.frombuffer(leaves, dtype=np.uint8).reshape(-1, TOKENS_PER_BOARD)
                    rounds.append(self.network.launch(boards) if len(boards) else None)
                    gathered = True
                    continue

            launched = rounds.popleft()
            if launched is None:
                answers = np.zeros((0, ANSWER_WIDTH), dtype=np.float32)
            else:
                answers = self.network.collect(launched)
                #rounds come back one after another while the gpu is kept busy
                now = time.perf_counter()
                if last_answers is not None:
                    self.round_seconds = 0.8 * self.round_seconds + 0.2 * (now - last_answers)
                last_answers = now
            reports = self.searches.finish(answers)
            if reports:
                report = reports[0]
            #a round without leaves counts too: all its simulations may have ended the game (a mate found)
            elif report is None and self.smart_pruning and not stopping:
                visits, most, second = self.searches.root_visits(GAME)
                progress.append((time.perf_counter(), visits))
                if only_move or most - second > self.simulations_left(simulations, deadline, progress) / SMART_PRUNING_FACTOR:
                    stopping = True
                    report = self.searches.stop(GAME)

        _, line, score, root_moves, visits = report
        return SearchReport(sharpen(score, 1 / self.value_sharpness), [chess.Move.from_uci(move) for move in line], visits,
                            [(chess.Move.from_uci(move), move_visits) for move, move_visits in root_moves[:ROOT_MOVES_REPORTED]])

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

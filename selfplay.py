#self play for expert iteration: the model plays itself with mcts.py's search, and every position is saved with what the
#search found there, as a training set main.py can fine-tune on (SelfPlayDataset in data.py):
#   policy target  each move's share of the search's visits, for the TOP_MOVES most visited
#   value target   the game's result and the search's mean score at the root, which data.py blends (RESULT_WEIGHT)
#
#   python selfplay.py artifacts/runs/rl_start_d128_4min/model.pt --name selfplay_r1 --positions 1_000_000 --tablebase
#   python main.py --init-checkpoint artifacts/runs/rl_start_d128_4min/model.pt --training-set selfplay_r1 --mix-set full_sf8_30M_mix
#
#it can be stopped (ctrl+c) and started again with the same flags: every finished game is kept, and a larger --positions
#carries on where it stopped.
#
#for the games to differ, every search mixes noise into its root's priors (mcts.py), and the first SAMPLED_PLIES plies
#are drawn from the visit counts instead of taking the most visited move. A game ends by the rules (a draw that could be
#claimed counts), with --tablebase once the tables know the result (5 pieces, and the result is then exact), or as a draw
#after MAX_PLIES
#
#the games are played in rust (selfplay_rs/, RustSelfPlayBatch), SelfPlayBatch's search and rules move for move at a
#fraction of the cpu (tests/test_selfplay_rs.py holds the two to each other): a change to mcts.py's search or to the
#rules here has to be made in selfplay_rs/src too. The network runs in one inference process on the gpu (inference.py)
from __future__ import annotations
import argparse
from collections.abc import Callable, Iterable, Iterator
import ctypes
from dataclasses import dataclass
import itertools
import json
import multiprocessing
from multiprocessing import shared_memory
from pathlib import Path
import queue
import signal
import subprocess
import sys
import time
import traceback
from typing import TYPE_CHECKING
try:
    import msvcrt #reads single key presses, for + and - while self play runs
except ImportError:
    msvcrt = None
import chess
import numpy as np
from tqdm import tqdm
import mcts
import selfplay_rs
from encoding import MOVE_IDS_BY_SQUARES, NUM_MOVE_IDS, TOKENS_PER_BOARD, encode_board, encode_move_to_id
from tools.label_with_stockfish import NO_MOVE
from tools.tablebase import SYZYGY_DIR, TABLEBASE_PIECES, covers, open_tablebase, wdl_score
#every worker imports this file (a spawned process runs it again), so nothing at the top imports torch: data.py,
#tools.checkpoint and inference.py do, and the main process imports them where it uses them
if TYPE_CHECKING:
    import torch
    from torch import nn

SIMULATIONS = 64 #+390 elo over the policy alone in the ladder of artifacts/mcts_ladder_2026-09-23.md
TOP_MOVES = 8
SAMPLED_PLIES = 30
#the move played keeps its subtree, and its visits count towards the next search's SIMULATIONS. Measured on 394 self play
#positions against an 800 simulation search: the targets are as good (visit shares 0.226 from the reference's either way,
#the most visited move the reference's 77.7% against 75.6%, root score off by 0.0194 against 0.0195), and a move needs
#37.9 new simulations instead of 64
TREE_REUSE = True
MAX_PLIES = 600 #the fifty move rule ends nearly every game long before
VAL_EVERY = 50 #every 50th game goes to val
#the workers play on the cpu in rust, many games each, and one inference process runs the network on the gpu for all of
#them: every round it takes the boards of all the workers that are waiting in one batch. Measured with the d128 (compiled,
#bf16), in network evaluations a second: 1 worker x 256 games 96,600, 2 x 256 138,000-144,000 (the gpu's limit, with 10%
#of the cpu busy), 3 x 256 134,000, 4 x 256 132,000, 2 x 512 134,000. The python search made 83,900 with 28 workers x 64
#games and every cpu thread busy. + and - change the number while it runs; every slot adds batch sizes the inference
#process records a graph for, and a larger model needs fewer workers (the d640's gpu limit is about 4,700 a second)
PROCESSES = 2
MAX_PROCESSES = 8
GAMES_PER_PROCESS = 256
LOG_EVERY_GAMES = 50 #between two points of the tensorboard charts
SEED = 42
RECORD_ATTEMPTS = 20 #about 10 seconds of trying to write the record


#one game: every position before a move with what the search found there, and the result from white's side
@dataclass
class Game:
    number: int
    boards: np.ndarray #(positions, TOKENS_PER_BOARD)
    visit_moves: np.ndarray #(positions, TOP_MOVES), NO_MOVE where fewer moves were visited
    visit_shares: np.ndarray #(positions, TOP_MOVES)
    root_scores: np.ndarray #for the side to move
    white_to_move: np.ndarray
    white_score: float

    #the result for the side to move at every position
    def outcomes(self) -> np.ndarray:
        return np.where(self.white_to_move, self.white_score, 1 - self.white_score).astype(np.float32)


#white's result if the game is over, None while it goes on
def final_result(board: chess.Board, max_plies: int = MAX_PLIES) -> float | None:
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        return 0.5 if outcome.winner is None else 1.0 if outcome.winner == chess.WHITE else 0.0
    if mcts.tablebase is not None and covers(board):
        score = wdl_score(mcts.tablebase.probe_wdl(board))
        return score if board.turn == chess.WHITE else 1 - score
    if len(board.move_stack) >= max_plies:
        return 0.5
    return None

#one game being played: its board, its random numbers (from the seed and its number alone, so a game can be played
#again), the search of its current move, and what it has recorded so far
class GameInProgress:
    def __init__(self, number: int, seed: int, board: chess.Board, tree_reuse: bool):
        self.number = number
        self.random = np.random.default_rng([seed, number])
        self.board = board.copy()
        self.tree_reuse = tree_reuse
        self.boards, self.visit_moves, self.visit_shares, self.root_scores, self.white_to_move = [], [], [], [], []
        self.new_search(mcts.Node(prior=1.0))

    #the search of the next move starts at root: a new node, or with tree reuse the node of the move just played, whose
    #visits from the last search count towards this one's
    def new_search(self, root: mcts.Node) -> None:
        self.root = root
        #the search walks down on the game's own board and is taken back to here after every simulation
        self.root_plies = len(self.board.move_stack)
        self.noised = False
        self.add_noise_once_expanded()

    #the root noise goes in as soon as the root has children, before any of them is chosen: at once for a reused root that
    #has them, after the first simulation otherwise (as in search())
    def add_noise_once_expanded(self) -> None:
        if not self.noised and self.root.children:
            mcts.add_root_noise(self.root, self.random)
            self.noised = True

    #the search of this move is done: record the position with what the search found, and play a move
    def play_move(self, sampled_plies: int) -> None:
        moves = list(self.root.children)
        visits = np.array([self.root.children[move].visit_count for move in moves], dtype=np.float64)

        most_visited = [index for index in np.argsort(-visits, kind="stable")[:TOP_MOVES] if visits[index] > 0]
        position_moves = np.full(TOP_MOVES, NO_MOVE, dtype=np.int16)
        position_shares = np.zeros(TOP_MOVES, dtype=np.float32)
        position_moves[:len(most_visited)] = [encode_move_to_id(moves[index], self.board.turn) for index in most_visited]
        position_shares[:len(most_visited)] = visits[most_visited] / visits[most_visited].sum()

        self.boards.append(encode_board(self.board))
        self.visit_moves.append(position_moves)
        self.visit_shares.append(position_shares)
        #the root keeps its score for the side that moved into it, the opponent of the side to move here (see Node)
        self.root_scores.append(1 - self.root.value_sum / self.root.visit_count)
        self.white_to_move.append(self.board.turn == chess.WHITE)

        if len(self.board.move_stack) < sampled_plies:
            move = moves[self.random.choice(len(moves), p=visits / visits.sum())]
        else:
            move = mcts.most_visited_move(self.root)
        next_root = self.root.children[move] if self.tree_reuse else mcts.Node(prior=1.0)
        self.board.push(move)
        self.new_search(next_root)

    def to_game(self, white_score: float) -> Game:
        return Game(self.number, np.array(self.boards, dtype=np.uint8).reshape(-1, TOKENS_PER_BOARD),
                    np.array(self.visit_moves, dtype=np.int16).reshape(-1, TOP_MOVES), np.array(self.visit_shares, dtype=np.float32).reshape(-1, TOP_MOVES),
                    np.array(self.root_scores, dtype=np.float32), np.array(self.white_to_move, dtype=bool), white_score)


#plays up to `size` games at once, one simulation of every game's search per round: each walks down its tree, the leaves
#that need the network go through it together in one call, and each backs its score up. The search is mcts.py's, with the
#root noise as in search(); a move is played once its root has `simulations` visits, and a game that ends makes room for
#the next number. The workers play RustSelfPlayBatch's games instead: these are what the rust is tested against
class SelfPlayBatch:
    def __init__(self, model: nn.Module, device: torch.device, simulations: int, numbers: Iterator[int], size: int, seed: int,
                 board: chess.Board, sampled_plies: int, max_plies: int, tree_reuse: bool):
        self.model, self.device, self.simulations = model, device, simulations
        self.numbers, self.seed, self.start_board = numbers, seed, board
        self.sampled_plies, self.max_plies, self.tree_reuse = sampled_plies, max_plies, tree_reuse
        self.games: list[GameInProgress] = []
        self.finished: list[Game] = []
        for _ in range(size):
            self.start_next_game()

    def start_next_game(self) -> None:
        for number in self.numbers:
            game = GameInProgress(number, self.seed, self.start_board, self.tree_reuse)
            white_score = final_result(game.board, self.max_plies)
            if white_score is None:
                self.games.append(game)
                return
            self.finished.append(game.to_game(white_score)) #over before it began: a game without positions

    def playing(self) -> bool:
        return bool(self.games or self.finished)

    #one round, returning the games it finished
    def step(self) -> list[Game]:
        waiting = []
        for game in self.games:
            leaf = mcts.descend(game.root, game.board)
            if leaf.value is None:
                waiting.append((game, leaf))
            else:
                mcts.finish(leaf, None, leaf.value)
        if waiting:
            #every waiting game's board is still at its leaf
            evaluations = mcts.evaluate_many(self.model, [game.board for game, _ in waiting], self.device)
            for (game, leaf), (priors, value) in zip(waiting, evaluations):
                mcts.finish(leaf, priors, value)

        playing = []
        for game in self.games:
            while len(game.board.move_stack) > game.root_plies:
                game.board.pop()
            game.add_noise_once_expanded()
            if game.root.visit_count >= self.simulations:
                game.play_move(self.sampled_plies)
                white_score = final_result(game.board, self.max_plies)
                if white_score is not None:
                    self.finished.append(game.to_game(white_score))
                    continue
            playing.append(game)

        replacements = len(self.games) - len(playing)
        self.games = playing
        for _ in range(replacements):
            self.start_next_game()
        finished, self.finished = self.finished, []
        return finished

#the games with these numbers, `size` at a time, each yielded as it ends
def play_games(model: nn.Module, device: torch.device, simulations: int, numbers: Iterable[int], size: int, seed: int = SEED,
               board: chess.Board | None = None, sampled_plies: int = SAMPLED_PLIES, max_plies: int = MAX_PLIES,
               tree_reuse: bool = TREE_REUSE) -> Iterator[Game]:
    batch = SelfPlayBatch(model, device, simulations, iter(numbers), size, seed, board if board is not None else chess.Board(), sampled_plies,
                          max_plies, tree_reuse)
    while batch.playing():
        yield from batch.step()

#game number `number` of a run on its own
def play_game(model: nn.Module, device: torch.device, simulations: int, number: int, seed: int = SEED, board: chess.Board | None = None,
              sampled_plies: int = SAMPLED_PLIES, max_plies: int = MAX_PLIES, tree_reuse: bool = TREE_REUSE) -> Game:
    return next(play_games(model, device, simulations, [number], 1, seed, board, sampled_plies, max_plies, tree_reuse))


#encoding.py's move ids for the rust games, by from square, to square and promotion (0 for none, python-chess's piece
#type otherwise), -1 where there is no such move
def move_id_table() -> np.ndarray:
    table = np.full((64, 64, 7), -1, dtype=np.int16)
    for (from_square, to_square, promotion), move_id in MOVE_IDS_BY_SQUARES.items():
        table[from_square, to_square, promotion or 0] = move_id
    return table

#what every search in rust is built with (selfplay_rs.SelfPlayGames and Searches): the move ids, mcts.py's constants read
#now (a test can change them), and the tablebase
def rust_search_settings(use_tablebase: bool) -> dict:
    return dict(move_id_table=move_id_table().ravel().tolist(), answer_width=ANSWER_WIDTH, c_puct=mcts.C_PUCT, c_puct_factor=mcts.C_PUCT_FACTOR,
                c_puct_base=mcts.C_PUCT_BASE, fpu_reduction=mcts.FPU_REDUCTION, draw=mcts.DRAW, repetitions_for_a_draw=mcts.REPETITIONS_FOR_A_DRAW,
                tablebase_folder=str(SYZYGY_DIR) if use_tablebase else None, tablebase_pieces=TABLEBASE_PIECES)

#SelfPlayBatch's games played in rust (selfplay_rs): the same games with the random numbers of rust's generator instead of
#numpy's. network takes the boards of the leaves and gives back ANSWER_WIDTH numbers for each, as the inference process does
class RustSelfPlayBatch:
    def __init__(self, network: Callable[[np.ndarray], np.ndarray], simulations: int, numbers: Iterator[int], size: int, seed: int,
                 board: chess.Board, sampled_plies: int, max_plies: int, tree_reuse: bool, use_tablebase: bool):
        self.network, self.numbers = network, numbers
        self.games = selfplay_rs.SelfPlayGames(
            **rust_search_settings(use_tablebase), start_fen=board.root().fen(), start_moves=[move.uci() for move in board.move_stack], seed=seed,
            simulations=simulations, sampled_plies=sampled_plies, max_plies=max_plies, tree_reuse=tree_reuse, top_moves=TOP_MOVES,
            no_move=NO_MOVE, root_noise_alpha=mcts.ROOT_NOISE_ALPHA, root_noise_share=mcts.ROOT_NOISE_SHARE)
        for _ in range(size):
            self.start_next_game()

    def start_next_game(self) -> None:
        for number in self.numbers:
            if self.games.start_game(number):
                return

    def playing(self) -> bool:
        return self.games.playing()

    #one round, returning the games it finished
    def step(self) -> list[Game]:
        leaves = np.frombuffer(self.games.descend(), dtype=np.uint8).reshape(-1, TOKENS_PER_BOARD)
        ended = self.games.finish(self.network(leaves) if len(leaves) else np.zeros((0, ANSWER_WIDTH), dtype=np.float32))
        for _ in range(ended):
            self.start_next_game()
        return [Game(number, np.frombuffer(boards, dtype=np.uint8).reshape(-1, TOKENS_PER_BOARD),
                     np.frombuffer(visit_moves, dtype=np.int16).reshape(-1, TOP_MOVES), np.frombuffer(visit_shares, dtype=np.float32).reshape(-1, TOP_MOVES),
                     np.frombuffer(root_scores, dtype=np.float32), np.frombuffer(white_to_move, dtype=bool), white_score)
                for number, boards, visit_moves, visit_shares, root_scores, white_to_move, white_score in self.games.take_finished()]

#play_games in rust
def play_games_in_rust(network: Callable[[np.ndarray], np.ndarray], simulations: int, numbers: Iterable[int], size: int, seed: int = SEED,
                       board: chess.Board | None = None, sampled_plies: int = SAMPLED_PLIES, max_plies: int = MAX_PLIES,
                       tree_reuse: bool = TREE_REUSE, use_tablebase: bool = False) -> Iterator[Game]:
    batch = RustSelfPlayBatch(network, simulations, iter(numbers), size, seed, board if board is not None else chess.Board(), sampled_plies,
                              max_plies, tree_reuse, use_tablebase)
    while batch.playing():
        yield from batch.step()


#the files of a self play set, on disk: every finished game is added to its split's files, and then the record of what
#has been written, first as <name>_selfplay.json.part and then moved over <name>_selfplay.json. Opening the set again
#takes the fuller of the two records that the files hold all of (the move can fail after the files and the .part are
#written), and cuts off whatever was written after it, the part of a game that an interruption left behind
class SelfPlaySet:
    def __init__(self, folder: Path, name: str, settings: dict):
        self.folder = folder
        self.name = name
        self.record_path = folder / f"{name}_selfplay.json"
        self.partial_record_path = folder / f"{name}_selfplay.json.part"

        records = [record for record in map(self.read_record, (self.record_path, self.partial_record_path)) if record is not None]
        for record in records:
            #sets played before tree reuse existed were played without it
            played, asked = {"tree_reuse": False, **record["settings"]}, {"tree_reuse": False, **settings}
            if played != asked:
                reuse = " (it was played without tree reuse: --no-tree-reuse carries on with it)" if played["tree_reuse"] != asked["tree_reuse"] else ""
                raise ValueError(f"{name} was played with {record['settings']}, not {settings}{reuse}. Give another --name or delete its files")
        complete = [record for record in records if self.files_hold(record)]
        if complete:
            self.record = max(complete, key=lambda record: record["games"])
        elif records:
            raise ValueError(f"the files of {name} are shorter than its records say")
        else:
            if list(folder.glob(f"{name}_*")):
                raise FileExistsError(f"{name} already has files in {folder} but no {self.record_path.name}, give another --name")
            self.record = {"settings": settings, "games": 0, "next_game": 0, "positions": {"train": 0, "val": 0},
                           "white_scores": {"1.0": 0, "0.5": 0, "0.0": 0}, "seconds": 0.0}

        for split in ("train", "val"):
            positions = self.record["positions"][split]
            for path, bytes_per_position in self.files(split).values():
                if path.exists() and path.stat().st_size > positions * bytes_per_position:
                    with open(path, "r+b") as file:
                        file.truncate(positions * bytes_per_position)

    #None when there is none, or only half of one (the .part of an interrupted write)
    @staticmethod
    def read_record(path: Path) -> dict | None:
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    #whether every file holds at least the positions the record counts
    def files_hold(self, record: dict) -> bool:
        return all((path.stat().st_size if path.exists() else 0) >= record["positions"][split] * bytes_per_position
                   for split in ("train", "val") for path, bytes_per_position in self.files(split).values())

    #every file of a split, with the bytes one position takes in it
    def files(self, split: str) -> dict[str, tuple[Path, int]]:
        from data import SELF_PLAY_FILES
        widths = {"boards": TOKENS_PER_BOARD, "visit_moves": TOP_MOVES, "visit_shares": TOP_MOVES, "outcomes": 1, "root_scores": 1}
        dtypes = {"boards": np.uint8, **SELF_PLAY_FILES}
        return {kind: (self.folder / f"{self.name}_{split}_{kind}.bin", widths[kind] * np.dtype(dtypes[kind]).itemsize) for kind in widths}

    def positions(self) -> int:
        return self.record["positions"]["train"] + self.record["positions"]["val"]

    #False when the record could not be moved into place this time (see write_record)
    def add(self, game: Game, seconds: float) -> bool:
        split = "val" if game.number % VAL_EVERY == 0 else "train"
        arrays = {"boards": game.boards, "visit_moves": game.visit_moves, "visit_shares": game.visit_shares, "outcomes": game.outcomes(), "root_scores": game.root_scores}
        for kind, (path, _) in self.files(split).items():
            with open(path, "ab") as file:
                file.write(arrays[kind].tobytes())

        self.record["games"] += 1
        self.record["next_game"] = max(self.record["next_game"], game.number + 1)
        self.record["positions"][split] += len(game.boards)
        self.record["white_scores"][str(game.white_score)] += 1
        self.record["seconds"] += seconds
        return self.write_record()

    #written whole under another name first, so an interruption never leaves half a record. Windows refuses the move while
    #another program has the file open for a moment (a virus scanner, an indexer, an editor): it is tried again, and if it
    #keeps failing the game is still safe, since the .part holds its record and the next game tries again
    def write_record(self) -> bool:
        for attempt in range(RECORD_ATTEMPTS):
            try:
                self.partial_record_path.write_text(json.dumps(self.record, indent=4))
                self.partial_record_path.replace(self.record_path)
                return True
            except PermissionError:
                time.sleep(0.05 * (attempt + 1))
        return False


#the network's answer for one position as the inference process sends it back: the move logits, then the eval logit
ANSWER_WIDTH = NUM_MOVE_IDS + 1

#one worker's place at the inference process: shared memory for the boards it sends (their tokens, one byte each) and for
#the answers it gets back, and a pipe for "this many boards are ready" and "the answers are in". Made by the main process,
#so a worker that is added later finds its place ready
class Slot:
    def __init__(self, games_per_process: int):
        self.boards = shared_memory.SharedMemory(create=True, size=games_per_process * TOKENS_PER_BOARD)
        self.answers = shared_memory.SharedMemory(create=True, size=games_per_process * ANSWER_WIDTH * np.dtype(np.float32).itemsize)
        self.worker_end, self.server_end = multiprocessing.Pipe()

    def close(self) -> None:
        for memory in (self.boards, self.answers):
            memory.close()
            memory.unlink()

#a slot's shared memory as arrays, in whichever process attaches to it
def slot_arrays(boards_name: str, answers_name: str, games_per_process: int) -> tuple[list, np.ndarray, np.ndarray]:
    memories = [shared_memory.SharedMemory(name=boards_name), shared_memory.SharedMemory(name=answers_name)]
    boards = np.ndarray((games_per_process, TOKENS_PER_BOARD), dtype=np.uint8, buffer=memories[0].buf)
    answers = np.ndarray((games_per_process, ANSWER_WIDTH), dtype=np.float32, buffer=memories[1].buf)
    return memories, boards, answers


#the network as a worker sees it: the boards go to the inference process through its slot, and the answers come back the
#same way. They stay good until the next call
class RemoteNetwork:
    def __init__(self, boards_name: str, answers_name: str, connection, games_per_process: int):
        self.memories, self.boards, self.answers = slot_arrays(boards_name, answers_name, games_per_process)
        self.connection = connection

    def __call__(self, boards: np.ndarray) -> np.ndarray:
        count = len(boards)
        self.boards[:count] = boards
        self.connection.send(count)
        self.connection.recv()
        return self.answers[:count]


#one self play process, on the cpu only: games_per_process games at a time in rust, their numbers from the shared counter
#next_number, their leaves evaluated by the inference process through its slot, each game sent back through results as
#(worker_id, game) as it ends. Once its stop is set it starts no new game, finishes the ones it has and sends
#(worker_id, None). A failure is sent back as its traceback
def worker(worker_id: int, slot_names: tuple[str, str], connection_, simulations: int, games_per_process: int, seed: int, use_tablebase: bool,
           tree_reuse: bool, next_number, stop, results) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        network = RemoteNetwork(*slot_names, connection_, games_per_process)

        def numbers() -> Iterator[int]:
            while not stop.is_set():
                with next_number.get_lock():
                    number = next_number.value
                    next_number.value += 1
                yield number

        for game in play_games_in_rust(network, simulations, numbers(), games_per_process, seed, tree_reuse=tree_reuse, use_tablebase=use_tablebase):
            results.put((worker_id, game))
        results.put((worker_id, None))
    except Exception:
        results.put((worker_id, traceback.format_exc()))


#the processes of a run: the inference process, and the self play workers, which can grow and shrink while it runs. Each
#worker takes the next game number from a shared counter, so no two play the same game, and one asked to stop finishes the
#games it has before it goes. Every worker has a slot at the inference process; the slot of one that finished cleanly is
#given to the next, the slot of one that failed is not (its last request may still be answered into it)
class Workers:
    def __init__(self, context, first_number: int, checkpoint_path: str, compile_model: bool, precision: str, games_per_process: int, worker_arguments: tuple):
        self.context = context
        self.results = context.Queue()
        self.next_number = context.Value("q", first_number)
        self.games_per_process = games_per_process
        self.worker_arguments = worker_arguments
        self.slots = [Slot(games_per_process) for _ in range(MAX_PROCESSES)]
        self.free_slots = list(range(MAX_PROCESSES))
        self.processes, self.stops, self.slot_of = {}, {}, {}
        self.done: set[int] = set()
        self.ids = itertools.count()

        self.served_calls, self.served_positions = context.Value("q", 0, lock=False), context.Value("q", 0, lock=False)
        self.inference_ready, self.inference_stop = context.Event(), context.Event()
        from inference import inference_process
        slot_names = [(slot.boards.name, slot.answers.name) for slot in self.slots]
        self.inference = context.Process(target=inference_process, daemon=True,
                                         args=(checkpoint_path, compile_model, precision, games_per_process, slot_names, [slot.server_end for slot in self.slots],
                                               self.inference_ready, self.inference_stop, self.served_calls, self.served_positions))
        self.inference.start()

    #waits until the inference process has its graphs recorded
    def wait_for_inference(self) -> None:
        while not self.inference_ready.wait(timeout=1):
            if not self.inference.is_alive():
                raise RuntimeError("the inference process stopped while starting")

    def start(self) -> None:
        worker_id, slot = next(self.ids), self.free_slots.pop(0)
        self.stops[worker_id], self.slot_of[worker_id] = self.context.Event(), slot
        names = (self.slots[slot].boards.name, self.slots[slot].answers.name)
        self.processes[worker_id] = self.context.Process(target=worker, daemon=True,
                                                         args=(worker_id, names, self.slots[slot].worker_end, *self.worker_arguments,
                                                               self.next_number, self.stops[worker_id], self.results))
        self.processes[worker_id].start()

    #a worker that finished its games: its slot is free again
    def finished(self, worker_id: int) -> None:
        self.done.add(worker_id)
        self.free_slots.append(self.slot_of[worker_id])

    #a worker that failed: its slot is not used again
    def failed(self, worker_id: int) -> None:
        self.done.add(worker_id)

    #the ones still starting new games
    def playing(self) -> list[int]:
        return [worker_id for worker_id in self.processes if worker_id not in self.done and not self.stops[worker_id].is_set()]

    #how many more can be started: one per free slot
    def can_start(self) -> bool:
        return bool(self.free_slots)

    #the newest one playing finishes its games and goes
    def stop_one(self) -> None:
        self.stops[self.playing()[-1]].set()

    def stop_all(self) -> None:
        for stop in self.stops.values():
            stop.set()

    def all_done(self) -> bool:
        return len(self.done) == len(self.processes)

    #the ones that ended without saying so (killed from outside), now counted as failed
    def dead(self) -> list[int]:
        dead = [worker_id for worker_id, process in self.processes.items() if worker_id not in self.done and not process.is_alive()]
        for worker_id in dead:
            self.failed(worker_id)
        return dead

    #only after a second ctrl+c or a failure are games still being played: they are lost
    def terminate(self) -> None:
        for process in self.processes.values():
            process.terminate()
        self.inference_stop.set()
        self.inference.join(timeout=10)
        if self.inference.is_alive():
            self.inference.terminate()
        for process in [*self.processes.values(), self.inference]:
            process.join()
        for slot in self.slots:
            slot.close()


#keys pressed in the console since the last look, none where there is no console to read (msvcrt is windows only)
def pressed_keys() -> list[str]:
    keys = []
    try:
        while msvcrt is not None and msvcrt.kbhit():
            keys.append(msvcrt.getwch())
    except OSError:
        pass
    return keys


#sets this process's priority class (windows only): main lowers itself to below normal, which the workers inherit, so the rest
#of the computer stays responsive for the hours this takes
def set_priority(priority_class: int) -> None:
    if sys.platform != "win32":
        return
    kernel32 = ctypes.windll.kernel32
    #a handle is a pointer: without these types ctypes passes a 32 bit int, and the call fails without a word
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    if not kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), priority_class):
        raise ctypes.WinError()


def main(checkpoint_path: Path, name: str, positions: int, simulations: int, use_tablebase: bool, processes: int, games_per_process: int, seed: int,
         tree_reuse: bool, compile_model: bool, precision: str) -> None:
    from data import PROCESSED_DATA_DIR
    from tools.checkpoint import RUNS_DIR
    if simulations < 2:
        raise ValueError("a search of fewer than 2 simulations visits no move, so there would be no visit shares to learn")
    settings = {"checkpoint": str(checkpoint_path), "simulations": simulations, "top_moves": TOP_MOVES, "sampled_plies": SAMPLED_PLIES,
                "max_plies": MAX_PLIES, "root_noise": [mcts.ROOT_NOISE_ALPHA, mcts.ROOT_NOISE_SHARE], "tablebase": use_tablebase, "seed": seed,
                "tree_reuse": tree_reuse}
    games = SelfPlaySet(PROCESSED_DATA_DIR, name, settings)
    if games.positions() >= positions:
        print(f"{name} already has {games.positions():,} positions")
        return
    if use_tablebase:
        open_tablebase().close() #fails here, not in every worker, when the tables are missing
    set_priority(subprocess.BELOW_NORMAL_PRIORITY_CLASS if sys.platform == "win32" else 0)

    workers = Workers(multiprocessing.get_context("spawn"), games.record["next_game"], str(checkpoint_path), compile_model, precision, games_per_process,
                      (simulations, games_per_process, seed, use_tablebase, tree_reuse))
    print("starting the inference process" + (" and compiling the model, 10-30 s" if compile_model else ""))
    try:
        workers.wait_for_inference()
    except BaseException:
        workers.terminate()
        raise
    for _ in range(min(processes, MAX_PROCESSES)):
        workers.start()
    print(f"{len(workers.playing())} processes playing {games_per_process} games each: the first games end together after a few minutes.\n"
          f"+ or - adds or removes a process (up to {MAX_PROCESSES}), ctrl+c finishes the games being played and saves them, a second ctrl+c stops at once")

    #charts of the run in tensorboard, next to the training runs: imported here, it is slow to import
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=str(RUNS_DIR / "selfplay" / name))
    progress = tqdm(total=positions, initial=games.positions(), unit="pos", smoothing=0.05, dynamic_ncols=True, desc=name)

    #once finishing, every process finishes the games it has and starts no more, so every game begun is saved
    finishing = False
    def finish() -> None:
        nonlocal finishing
        finishing = True
        workers.stop_all()
        progress.set_description(f"{name}, finishing the games being played")

    #the first ctrl+c only asks for the finish. An exception here could land in the middle of writing a game
    def interrupt(signal_number, frame) -> None:
        if finishing:
            raise KeyboardInterrupt
        finish()
        tqdm.write("finishing the games being played, ctrl+c again to stop at once and lose them")
    signal.signal(signal.SIGINT, interrupt)

    last = time.perf_counter()
    logged_time, logged_positions = last, games.positions()
    logged_calls, logged_evaluations = 0, 0
    try:
        while not (finishing and workers.all_done()):
            for key in pressed_keys():
                if key == "+" and not finishing and workers.can_start():
                    workers.start()
                    tqdm.write(f"{len(workers.playing())} processes")
                elif key == "-" and not finishing and len(workers.playing()) > 1:
                    workers.stop_one()
                    tqdm.write(f"{len(workers.playing())} processes, the one removed finishes its games first")

            if not workers.inference.is_alive():
                raise RuntimeError("the inference process stopped")
            try:
                worker_id, message = workers.results.get(timeout=0.5)
            except queue.Empty:
                for worker_id in workers.dead():
                    tqdm.write(f"process {worker_id} ended without finishing its games")
                if workers.all_done() and not finishing:
                    raise RuntimeError("every self play process has stopped")
                continue
            except InterruptedError:
                #on windows a ctrl+c breaks off the wait after interrupt has run, and python does not wait again itself
                continue
            if message is None:
                workers.finished(worker_id)
                continue
            if isinstance(message, str):
                workers.failed(worker_id)
                tqdm.write(f"a self play process failed and is dropped, the others play on:\n{message}")
                continue

            now = time.perf_counter()
            if not games.add(message, now - last):
                tqdm.write(f"could not move the record into place, another program has {games.record_path.name} open. "
                           f"The games are safe (the .part record holds them) and the next game tries again")
            last = now
            progress.update(len(message.boards))
            record = games.record
            scores = record["white_scores"]
            progress.set_postfix(games=record["games"], white=scores["1.0"], draw=scores["0.5"], black=scores["0.0"], processes=len(workers.playing()))

            if record["games"] % LOG_EVERY_GAMES == 0:
                step = games.positions()
                writer.add_scalar("selfplay/positions an hour", (step - logged_positions) / (now - logged_time) * 3600, step)
                writer.add_scalar("selfplay/processes", len(workers.playing()), step)
                writer.add_scalar("selfplay/plies a game", step / record["games"], step)
                writer.add_scalar("selfplay/white score", (scores["1.0"] + scores["0.5"] / 2) / record["games"], step)
                writer.add_scalar("selfplay/draws", scores["0.5"] / record["games"], step)
                calls, evaluations = workers.served_calls.value, workers.served_positions.value
                if calls > logged_calls:
                    writer.add_scalar("selfplay/network batch", (evaluations - logged_evaluations) / (calls - logged_calls), step)
                    writer.add_scalar("selfplay/network positions a second", (evaluations - logged_evaluations) / (now - logged_time), step)
                logged_time, logged_positions, logged_calls, logged_evaluations = now, step, calls, evaluations
            if games.positions() >= positions and not finishing:
                finish()
    finally:
        workers.terminate()
        writer.close()
        progress.close()

    record = games.record
    print(f"{name}: {record['positions']['train']:,} train and {record['positions']['val']:,} val positions from {record['games']:,} games "
          f"({games.positions() / record['games']:.0f} plies a game), white {record['white_scores']['1.0']}, draws {record['white_scores']['0.5']}, "
          f"black {record['white_scores']['0.0']}, {games.positions() / record['seconds'] * 3600:,.0f} positions an hour")
    print(f"fine-tune on it: python main.py --init-checkpoint {checkpoint_path} --training-set {name} --mix-set full_sf8_30M_mix")


if __name__ == "__main__":
    from tools.checkpoint import latest_checkpoint
    parser = argparse.ArgumentParser(description="the model plays itself with mcts.py's search, and every position is saved as a training set")
    parser.add_argument("checkpoint", nargs="?", type=Path, help="a model.pt, default the run that finished last")
    parser.add_argument("--name", required=True, help="the training set that is made in processed_data/")
    parser.add_argument("--positions", type=int, required=True, help="stop once the set has this many positions, e.g. 1_000_000")
    parser.add_argument("--simulations", type=int, default=SIMULATIONS, help="per move, default %(default)s")
    parser.add_argument("--tablebase", action="store_true", help="end a game once it has at most 5 pieces, with the syzygy tables' result")
    parser.add_argument("--processes", type=int, default=PROCESSES, help=f"at the start, + and - change it while it runs (up to {MAX_PROCESSES}). Default %(default)s")
    parser.add_argument("--games-per-process", type=int, default=GAMES_PER_PROCESS, help="games each process plays at once, their leaves evaluated in one call. Default %(default)s")
    parser.add_argument("--no-tree-reuse", action="store_true", help="start every search from a new root (see TREE_REUSE)")
    parser.add_argument("--no-compile", action="store_true", help="skip torch.compile in the inference process (it takes 10-30 s at the start, and makes the gpu 40%% faster)")
    parser.add_argument("--precision", choices=["bf16", "tf32"], default="bf16", help="of the network in the inference process (see Bf16). Default %(default)s")
    parser.add_argument("--seed", type=int, default=SEED, help="default %(default)s")
    arguments = parser.parse_args()

    main(arguments.checkpoint or latest_checkpoint(), arguments.name, arguments.positions, arguments.simulations, arguments.tablebase,
         arguments.processes, arguments.games_per_process, arguments.seed, not arguments.no_tree_reuse,
         not arguments.no_compile, arguments.precision)

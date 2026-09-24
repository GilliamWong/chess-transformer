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
import argparse
import ctypes
from dataclasses import dataclass
import itertools
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import time
import chess
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import mcts
from data import PROCESSED_DATA_DIR, SELF_PLAY_FILES
from encoding import TOKENS_PER_BOARD, encode_board, encode_move_to_id
from tools.checkpoint import latest_checkpoint
from tools.label_with_stockfish import NO_MOVE
from tools.play.player import CudaGraphModel, load_model
from tools.tablebase import covers, open_tablebase, wdl_score

SIMULATIONS = 64 #+390 elo over the policy alone in the ladder of artifacts/mcts_ladder_2026-09-23.md
TOP_MOVES = 8
SAMPLED_PLIES = 30
MAX_PLIES = 600 #the fifty move rule ends nearly every game long before
VAL_EVERY = 50 #every 50th game goes to val
#one searching process per cpu core, leaving 4 of the 16 free. One of them runs the model on the gpu, where it is about
#four times as fast; more gpu processes share the gpu badly (see CudaGraphModel in tools/play/player.py)
NUM_PROCESSES = 12
GPU_PROCESSES = 1
SEED = 42

CPU = torch.device("cpu")


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

#game number `number` of a run: its random numbers come from the seed and the number alone, so a game can be played again
def play_game(model: nn.Module, device: torch.device, simulations: int, number: int, seed: int = SEED,
              board: chess.Board | None = None, sampled_plies: int = SAMPLED_PLIES, max_plies: int = MAX_PLIES) -> Game:
    random = np.random.default_rng([seed, number])
    board = chess.Board() if board is None else board.copy()
    boards, visit_moves, visit_shares, root_scores, white_to_move = [], [], [], [], []

    while (white_score := final_result(board, max_plies)) is None:
        root = mcts.search(model, board, device, simulations, root_noise=random)
        moves = list(root.children)
        visits = np.array([root.children[move].visit_count for move in moves], dtype=np.float64)

        most_visited = [index for index in np.argsort(-visits, kind="stable")[:TOP_MOVES] if visits[index] > 0]
        position_moves = np.full(TOP_MOVES, NO_MOVE, dtype=np.int16)
        position_shares = np.zeros(TOP_MOVES, dtype=np.float32)
        position_moves[:len(most_visited)] = [encode_move_to_id(moves[index], board.turn) for index in most_visited]
        position_shares[:len(most_visited)] = visits[most_visited] / visits[most_visited].sum()

        boards.append(encode_board(board))
        visit_moves.append(position_moves)
        visit_shares.append(position_shares)
        #the root keeps its score for the side that moved into it, the opponent of the side to move here (see Node)
        root_scores.append(1 - root.value_sum / root.visit_count)
        white_to_move.append(board.turn == chess.WHITE)

        if len(board.move_stack) < sampled_plies:
            move = moves[random.choice(len(moves), p=visits / visits.sum())]
        else:
            move = mcts.most_visited_move(root)
        board.push(move)

    return Game(number, np.array(boards, dtype=np.uint8).reshape(-1, TOKENS_PER_BOARD), np.array(visit_moves, dtype=np.int16).reshape(-1, TOP_MOVES),
                np.array(visit_shares, dtype=np.float32).reshape(-1, TOP_MOVES), np.array(root_scores, dtype=np.float32),
                np.array(white_to_move, dtype=bool), white_score)


#the files of a self play set, on disk: every finished game is added to its split's files, and then the record of what
#has been written. Opening the set again reads the record and cuts off whatever was written after it, the part of a game
#that an interruption left behind
class SelfPlaySet:
    def __init__(self, folder: Path, name: str, settings: dict):
        self.folder = folder
        self.name = name
        self.record_path = folder / f"{name}_selfplay.json"
        if self.record_path.exists():
            self.record = json.loads(self.record_path.read_text())
            if self.record["settings"] != settings:
                raise ValueError(f"{name} was played with {self.record['settings']}, not {settings}. Give another --name or delete its files")
        else:
            if list(folder.glob(f"{name}_*")):
                raise FileExistsError(f"{name} already has files in {folder} but no {self.record_path.name}, give another --name")
            self.record = {"settings": settings, "games": 0, "next_game": 0, "positions": {"train": 0, "val": 0},
                           "white_scores": {"1.0": 0, "0.5": 0, "0.0": 0}, "seconds": 0.0}

        for split in ("train", "val"):
            positions = self.record["positions"][split]
            for path, bytes_per_position in self.files(split).values():
                written = path.stat().st_size if path.exists() else 0
                if written < positions * bytes_per_position:
                    raise ValueError(f"{path.name} is shorter than its {positions:,} recorded positions")
                if written > positions * bytes_per_position:
                    with open(path, "r+b") as file:
                        file.truncate(positions * bytes_per_position)

    #every file of a split, with the bytes one position takes in it
    def files(self, split: str) -> dict[str, tuple[Path, int]]:
        widths = {"boards": TOKENS_PER_BOARD, "visit_moves": TOP_MOVES, "visit_shares": TOP_MOVES, "outcomes": 1, "root_scores": 1}
        dtypes = {"boards": np.uint8, **SELF_PLAY_FILES}
        return {kind: (self.folder / f"{self.name}_{split}_{kind}.bin", widths[kind] * np.dtype(dtypes[kind]).itemsize) for kind in widths}

    def positions(self) -> int:
        return self.record["positions"]["train"] + self.record["positions"]["val"]

    def add(self, game: Game, seconds: float) -> None:
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
        #written whole under another name first, so an interruption never leaves half a record
        partial = self.record_path.with_suffix(".json.part")
        partial.write_text(json.dumps(self.record, indent=4))
        partial.replace(self.record_path)


#the model of a worker process, loaded once for all of its games
MODEL = None

def start_worker(checkpoint_path: str, gpu_processes: int, use_tablebase: bool, started) -> None:
    global MODEL
    torch.set_num_threads(1)
    with started.get_lock():
        worker = started.value
        started.value += 1
    if worker < gpu_processes:
        MODEL = CudaGraphModel(load_model(Path(checkpoint_path), torch.device("cuda")))
    else:
        MODEL = load_model(Path(checkpoint_path), CPU)
    if use_tablebase:
        mcts.tablebase = open_tablebase()

def game_task(task: tuple[int, int, int]) -> Game:
    number, simulations, seed = task
    #CudaGraphModel takes its boards from the cpu as well
    return play_game(MODEL, CPU, simulations, number, seed)


#below normal priority for this process and the workers it starts, which inherit it, so the rest of the computer stays
#responsive for the hours this takes
def lower_priority() -> None:
    if sys.platform != "win32":
        return
    kernel32 = ctypes.windll.kernel32
    #a handle is a pointer: without these types ctypes passes a 32 bit int, and the call fails without a word
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    if not kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), subprocess.BELOW_NORMAL_PRIORITY_CLASS):
        raise ctypes.WinError()


def main(checkpoint_path: Path, name: str, positions: int, simulations: int, use_tablebase: bool, processes: int, gpu_processes: int, seed: int) -> None:
    if simulations < 2:
        raise ValueError("a search of fewer than 2 simulations visits no move, so there would be no visit shares to learn")
    settings = {"checkpoint": str(checkpoint_path), "simulations": simulations, "top_moves": TOP_MOVES, "sampled_plies": SAMPLED_PLIES,
                "max_plies": MAX_PLIES, "root_noise": [mcts.ROOT_NOISE_ALPHA, mcts.ROOT_NOISE_SHARE], "tablebase": use_tablebase, "seed": seed}
    games = SelfPlaySet(PROCESSED_DATA_DIR, name, settings)
    if games.positions() >= positions:
        print(f"{name} already has {games.positions():,} positions")
        return
    if use_tablebase:
        open_tablebase().close() #fails here, not in every worker, when the tables are missing
    gpu_processes = gpu_processes if torch.cuda.is_available() else 0
    lower_priority()

    context = multiprocessing.get_context("spawn")
    started = context.Value("i", 0)
    tasks = ((number, simulations, seed) for number in itertools.count(games.record["next_game"]))
    progress = tqdm(total=positions, initial=games.positions(), unit="pos", smoothing=0.05, dynamic_ncols=True, desc=name)
    last = time.perf_counter()
    with context.Pool(processes, initializer=start_worker, initargs=(str(checkpoint_path), gpu_processes, use_tablebase, started)) as pool:
        for game in pool.imap_unordered(game_task, tasks):
            now = time.perf_counter()
            games.add(game, now - last)
            last = now
            progress.update(len(game.boards))
            scores = games.record["white_scores"]
            progress.set_postfix(games=games.record["games"], white=scores["1.0"], draw=scores["0.5"], black=scores["0.0"])
            if games.positions() >= positions:
                break
    progress.close()

    record = games.record
    print(f"{name}: {record['positions']['train']:,} train and {record['positions']['val']:,} val positions from {record['games']:,} games "
          f"({games.positions() / record['games']:.0f} plies a game), white {record['white_scores']['1.0']}, draws {record['white_scores']['0.5']}, "
          f"black {record['white_scores']['0.0']}, {games.positions() / record['seconds'] * 3600:,.0f} positions an hour")
    print(f"fine-tune on it: python main.py --init-checkpoint {checkpoint_path} --training-set {name} --mix-set full_sf8_30M_mix")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="the model plays itself with mcts.py's search, and every position is saved as a training set")
    parser.add_argument("checkpoint", nargs="?", type=Path, help="a model.pt, default the run that finished last")
    parser.add_argument("--name", required=True, help="the training set that is made in processed_data/")
    parser.add_argument("--positions", type=int, required=True, help="stop once the set has this many positions, e.g. 1_000_000")
    parser.add_argument("--simulations", type=int, default=SIMULATIONS, help="per move, default %(default)s")
    parser.add_argument("--tablebase", action="store_true", help="end a game once it has at most 5 pieces, with the syzygy tables' result")
    parser.add_argument("--processes", type=int, default=NUM_PROCESSES, help="games at once, default %(default)s")
    parser.add_argument("--gpu-processes", type=int, default=GPU_PROCESSES, help="how many of them run the model on the gpu, default %(default)s")
    parser.add_argument("--seed", type=int, default=SEED, help="default %(default)s")
    arguments = parser.parse_args()

    main(arguments.checkpoint or latest_checkpoint(), arguments.name, arguments.positions, arguments.simulations, arguments.tablebase,
         arguments.processes, arguments.gpu_processes, arguments.seed)

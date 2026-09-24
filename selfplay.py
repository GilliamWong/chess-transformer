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
from collections.abc import Iterable, Iterator
import ctypes
from dataclasses import dataclass
import itertools
import json
import multiprocessing
from pathlib import Path
import queue
import signal
import subprocess
import sys
import time
import traceback
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
#every process plays many games at once and sends all of their leaves through the network in one call on the gpu, so
#the python of the searches is what takes the time. Measured with the d128 at 64 simulations, in simulations a second:
#12 x 64 games 46,100 (12 x 128 the same, 12 x 32 34,700, 8 x 64 30,600), against 4,600 for 12 processes asking about
#one position at a time. 12 processes leave 4 of the 16 cores free
PROCESSES = 12
GAMES_PER_PROCESS = 64
SEED = 42
RECORD_ATTEMPTS = 20 #about 10 seconds of trying to write the record

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

#one game being played: its board, its random numbers (from the seed and its number alone, so a game can be played
#again), the search of its current move, and what it has recorded so far
class GameInProgress:
    def __init__(self, number: int, seed: int, board: chess.Board):
        self.number = number
        self.random = np.random.default_rng([seed, number])
        self.board = board.copy()
        self.boards, self.visit_moves, self.visit_shares, self.root_scores, self.white_to_move = [], [], [], [], []
        self.new_search()

    def new_search(self) -> None:
        self.root = mcts.Node(prior=1.0)
        self.simulations = 0
        #the search walks down on the game's own board and is taken back to here after every simulation
        self.root_plies = len(self.board.move_stack)

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
        self.board.push(move)
        self.new_search()

    def to_game(self, white_score: float) -> Game:
        return Game(self.number, np.array(self.boards, dtype=np.uint8).reshape(-1, TOKENS_PER_BOARD),
                    np.array(self.visit_moves, dtype=np.int16).reshape(-1, TOP_MOVES), np.array(self.visit_shares, dtype=np.float32).reshape(-1, TOP_MOVES),
                    np.array(self.root_scores, dtype=np.float32), np.array(self.white_to_move, dtype=bool), white_score)


#plays up to `size` games at once, one simulation of every game's search per round: each walks down its tree, the leaves
#that need the network go through it together in one call, and each backs its score up. The search is mcts.py's, with the
#root noise at its first simulation as in search(); a game that ends makes room for the next number
class SelfPlayBatch:
    def __init__(self, model: nn.Module, device: torch.device, simulations: int, numbers: Iterator[int], size: int, seed: int,
                 board: chess.Board, sampled_plies: int, max_plies: int):
        self.model, self.device, self.simulations = model, device, simulations
        self.numbers, self.seed, self.start_board = numbers, seed, board
        self.sampled_plies, self.max_plies = sampled_plies, max_plies
        self.games: list[GameInProgress] = []
        self.finished: list[Game] = []
        for _ in range(size):
            self.start_next_game()

    def start_next_game(self) -> None:
        for number in self.numbers:
            game = GameInProgress(number, self.seed, self.start_board)
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
            game.simulations += 1
            #the first simulation expanded the root, and no child has been chosen yet
            if game.simulations == 1:
                mcts.add_root_noise(game.root, game.random)
            if game.simulations == self.simulations:
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
               board: chess.Board | None = None, sampled_plies: int = SAMPLED_PLIES, max_plies: int = MAX_PLIES) -> Iterator[Game]:
    batch = SelfPlayBatch(model, device, simulations, iter(numbers), size, seed, board if board is not None else chess.Board(), sampled_plies, max_plies)
    while batch.playing():
        yield from batch.step()

#game number `number` of a run on its own
def play_game(model: nn.Module, device: torch.device, simulations: int, number: int, seed: int = SEED,
              board: chess.Board | None = None, sampled_plies: int = SAMPLED_PLIES, max_plies: int = MAX_PLIES) -> Game:
    return next(play_games(model, device, simulations, [number], 1, seed, board, sampled_plies, max_plies))


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
            if record["settings"] != settings:
                raise ValueError(f"{name} was played with {record['settings']}, not {settings}. Give another --name or delete its files")
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


#one self play process: games number first, first + stride, ..., games_per_process at a time, each sent back through
#results as it ends. Once stop is set it starts no new game, finishes the ones it has and sends None. A failure is sent
#back as its traceback, for the main process to stop with
def worker(checkpoint_path: str, simulations: int, first: int, stride: int, games_per_process: int, seed: int, use_tablebase: bool, stop, results) -> None:
    #ctrl+c reaches every process in the console: the main process decides what it means (see main)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        torch.set_num_threads(1)
        #tensor float 32 for the matrix products: a fifth faster on the gpu, and the priors within 0.001 of full precision
        torch.backends.cuda.matmul.allow_tf32 = True
        if use_tablebase:
            mcts.tablebase = open_tablebase()
        #the boards stay on the cpu either way: CudaGraphModel takes them from there
        if torch.cuda.is_available():
            model = CudaGraphModel(load_model(Path(checkpoint_path), torch.device("cuda")), games_per_process)
        else:
            model = load_model(Path(checkpoint_path), CPU)
        numbers = itertools.takewhile(lambda _: not stop.is_set(), itertools.count(first, stride))
        for game in play_games(model, CPU, simulations, numbers, games_per_process, seed):
            results.put(game)
        results.put(None)
    except Exception:
        results.put(traceback.format_exc())


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


def main(checkpoint_path: Path, name: str, positions: int, simulations: int, use_tablebase: bool, processes: int, games_per_process: int, seed: int) -> None:
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
    lower_priority()

    #every process plays its own numbers, so no two play the same game. Once stop is set they start no new games and
    #finish the ones they have, so every game begun is saved
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    stop = context.Event()
    first = games.record["next_game"]
    workers = [context.Process(target=worker, args=(str(checkpoint_path), simulations, first + index, processes, games_per_process, seed, use_tablebase, stop, results), daemon=True)
               for index in range(processes)]
    for process in workers:
        process.start()
    print(f"{processes} processes playing {games_per_process} games each: the first games end together after a few minutes.\n"
          f"ctrl+c finishes the games being played and saves them, a second ctrl+c stops at once")

    #the first ctrl+c only asks for the finish. An exception here could land in the middle of writing a game
    def interrupt(signal_number, frame) -> None:
        if stop.is_set():
            raise KeyboardInterrupt
        stop.set()
        tqdm.write("finishing the games being played, ctrl+c again to stop at once and lose them")
    signal.signal(signal.SIGINT, interrupt)

    progress = tqdm(total=positions, initial=games.positions(), unit="pos", smoothing=0.05, dynamic_ncols=True, desc=name)
    last = time.perf_counter()
    finished_workers = 0
    try:
        while finished_workers < processes:
            try:
                game = results.get(timeout=60)
            except queue.Empty:
                if any(process.is_alive() for process in workers):
                    continue
                if stop.is_set():
                    break
                raise RuntimeError("every self play process has stopped")
            if game is None:
                finished_workers += 1
                continue
            if isinstance(game, str):
                raise RuntimeError(f"a self play process failed:\n{game}")

            now = time.perf_counter()
            if not games.add(game, now - last):
                tqdm.write(f"could not move the record into place, another program has {games.record_path.name} open. "
                           f"The games are safe (the .part record holds them) and the next game tries again")
            last = now
            progress.update(len(game.boards))
            scores = games.record["white_scores"]
            progress.set_postfix(games=games.record["games"], white=scores["1.0"], draw=scores["0.5"], black=scores["0.0"])
            if games.positions() >= positions and not stop.is_set():
                stop.set()
                progress.set_description(f"{name}, finishing the games being played")
    finally:
        #only after a second ctrl+c or a failure are games still being played: they are lost
        for process in workers:
            process.terminate()
        for process in workers:
            process.join()
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
    parser.add_argument("--processes", type=int, default=PROCESSES, help="default %(default)s")
    parser.add_argument("--games-per-process", type=int, default=GAMES_PER_PROCESS, help="games each process plays at once, their leaves evaluated in one call. Default %(default)s")
    parser.add_argument("--seed", type=int, default=SEED, help="default %(default)s")
    arguments = parser.parse_args()

    main(arguments.checkpoint or latest_checkpoint(), arguments.name, arguments.positions, arguments.simulations, arguments.tablebase,
         arguments.processes, arguments.games_per_process, arguments.seed)

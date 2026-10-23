#self play for expert iteration: every position is saved with what mcts found there, as a training set for train.py.
#   python -m engine.selfplay artifacts/runs/rl_start_d128_4min/model.pt --name selfplay_r1 --positions 1_000_000 --tablebase
#   python -m training.train --init-checkpoint artifacts/runs/rl_start_d128_4min/model.pt --training-set selfplay_r1 --mix-set full_sf8_30M_mix
#the games run in rust (selfplay_rs/): mirror any change to mcts.py's search or the rules there, tests/test_selfplay_rs.py
#holds it to the python reference in tests/reference_selfplay.py
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
try:
    import msvcrt #single key presses, for + and -
except ImportError:
    msvcrt = None
import chess
import numpy as np
from tqdm import tqdm
from engine import mcts
import selfplay_rs
from model.encoding import MOVE_IDS_BY_SQUARES, NUM_MOVE_IDS, TOKENS_PER_BOARD
from tools.label_with_stockfish import NO_MOVE
from tools.tablebase import TABLEBASE_PIECES, open_tablebase, tablebase_path
#every worker imports this file, so nothing at the top may import torch (data.py, tools.checkpoint and inference.py do)

SIMULATIONS = 64 #+390 elo over the policy alone (artifacts/reports/mcts_ladder_2026-09-23.md)
TOP_MOVES = 8
SAMPLED_PLIES = 30
#the move played keeps its subtree: the same target quality for 38 new simulations a move instead of 64
TREE_REUSE = True
MAX_PLIES = 600 #the fifty move rule ends nearly every game long before
VAL_EVERY = 50 #every 50th game goes to val
#rust workers of GAMES_PER_PROCESS games each: with the d128, 2 already keep the gpu busy (~140k evaluations a second)
PROCESSES = 2
MAX_PROCESSES = 8
GAMES_PER_PROCESS = 256
LOG_EVERY_GAMES = 50
SEED = 42
#the network's answers kept by position for transpositions (selfplay_rs' Cache, one a process), about 200 bytes each: a
#third of a d512 search's evaluations repeat a position, and self play's games share their openings
CACHE_POSITIONS = 500_000
#a round's simulations for each leaf it sends to the network: those the cache or the game's end answer at once do not
#wait for it. One leaf a round is still mcts.py's search, in fewer rounds
SIMULATIONS_PER_LEAF = 3
RECORD_ATTEMPTS = 20 #about 10 seconds of retrying the record's move


#every position before a move with what the search found there, and the result from white's side
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


#encoding.py's move ids by from square, to square and promotion (0 for none, else python-chess's piece type), -1 for no move
def move_id_table() -> np.ndarray:
    table = np.full((64, 64, 7), -1, dtype=np.int16)
    for (from_square, to_square, promotion), move_id in MOVE_IDS_BY_SQUARES.items():
        table[from_square, to_square, promotion or 0] = move_id
    return table

#mcts.py's constants are read at call time, so a test can change them. tablebase_pieces: the tables' positions of up to
#that many pieces (tools.tablebase.TABLE_SETS), from syzygy_path's folders rather than the downloaded ones if it is given
def rust_search_settings(use_tablebase: bool, tablebase_pieces: int = TABLEBASE_PIECES, syzygy_path: str | None = None) -> dict:
    return dict(move_id_table=move_id_table().ravel().tolist(), answer_width=ANSWER_WIDTH, c_puct=mcts.C_PUCT, c_puct_factor=mcts.C_PUCT_FACTOR,
                c_puct_base=mcts.C_PUCT_BASE, fpu_reduction=mcts.FPU_REDUCTION, draw=mcts.DRAW, repetitions_for_a_draw=mcts.REPETITIONS_FOR_A_DRAW,
                tablebase_folder=(syzygy_path or tablebase_path(tablebase_pieces)) if use_tablebase else None, tablebase_pieces=tablebase_pieces)

#up to `size` games at once in rust, one simulation of each per round. network takes the leaves' boards and gives back
#ANSWER_WIDTH numbers for each, as the inference process does
class RustSelfPlayBatch:
    def __init__(self, network: Callable[[np.ndarray], np.ndarray], simulations: int, numbers: Iterator[int], size: int, seed: int,
                 board: chess.Board, sampled_plies: int, max_plies: int, tree_reuse: bool, use_tablebase: bool):
        self.network, self.numbers = network, numbers
        self.games = selfplay_rs.SelfPlayGames(
            **rust_search_settings(use_tablebase), start_fen=board.root().fen(), start_moves=[move.uci() for move in board.move_stack], seed=seed,
            simulations=simulations, sampled_plies=sampled_plies, max_plies=max_plies, tree_reuse=tree_reuse, top_moves=TOP_MOVES,
            no_move=NO_MOVE, root_noise_alpha=mcts.ROOT_NOISE_ALPHA, root_noise_share=mcts.ROOT_NOISE_SHARE, cache_positions=CACHE_POSITIONS,
            simulations_per_round=SIMULATIONS_PER_LEAF)
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

#the games with these numbers, `size` at a time, each yielded as it ends
def play_games_in_rust(network: Callable[[np.ndarray], np.ndarray], simulations: int, numbers: Iterable[int], size: int, seed: int = SEED,
                       board: chess.Board | None = None, sampled_plies: int = SAMPLED_PLIES, max_plies: int = MAX_PLIES,
                       tree_reuse: bool = TREE_REUSE, use_tablebase: bool = False) -> Iterator[Game]:
    batch = RustSelfPlayBatch(network, simulations, iter(numbers), size, seed, board if board is not None else chess.Board(), sampled_plies,
                              max_plies, tree_reuse, use_tablebase)
    while batch.playing():
        yield from batch.step()


#a self play set on disk: each game is appended to its split's files, then the record goes to a .part moved over
#<name>_selfplay.json. Reopening takes the fuller record the files hold, and cuts off whatever was written after it
class SelfPlaySet:
    def __init__(self, folder: Path, name: str, settings: dict):
        self.folder = folder
        self.name = name
        self.record_path = folder / f"{name}_selfplay.json"
        self.partial_record_path = folder / f"{name}_selfplay.json.part"

        records = [record for record in map(self.read_record, (self.record_path, self.partial_record_path)) if record is not None]
        for record in records:
            #sets from before tree reuse were played without it
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

    #None when missing or half written
    @staticmethod
    def read_record(path: Path) -> dict | None:
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def files_hold(self, record: dict) -> bool:
        return all((path.stat().st_size if path.exists() else 0) >= record["positions"][split] * bytes_per_position
                   for split in ("train", "val") for path, bytes_per_position in self.files(split).values())

    #every file of a split, with the bytes one position takes in it
    def files(self, split: str) -> dict[str, tuple[Path, int]]:
        from training.data import SELF_PLAY_FILES
        widths = {"boards": TOKENS_PER_BOARD, "visit_moves": TOP_MOVES, "visit_shares": TOP_MOVES, "outcomes": 1, "root_scores": 1}
        dtypes = {"boards": np.uint8, **SELF_PLAY_FILES}
        return {kind: (self.folder / f"{self.name}_{split}_{kind}.bin", widths[kind] * np.dtype(dtypes[kind]).itemsize) for kind in widths}

    def positions(self) -> int:
        return self.record["positions"]["train"] + self.record["positions"]["val"]

    #False when the record could not be moved into place this time
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

    #windows refuses the move while another program has the file open for a moment: retried, and the .part keeps the game safe
    def write_record(self) -> bool:
        for attempt in range(RECORD_ATTEMPTS):
            try:
                self.partial_record_path.write_text(json.dumps(self.record, indent=4))
                self.partial_record_path.replace(self.record_path)
                return True
            except PermissionError:
                time.sleep(0.05 * (attempt + 1))
        return False


#the move logits, then the expected score
ANSWER_WIDTH = NUM_MOVE_IDS + 1

#a worker's place at the inference process: shared memory for its boards (a byte a token) and answers, and a pipe.
#made by the main process, so a worker added later finds it ready
class Slot:
    def __init__(self, games_per_process: int):
        self.boards = shared_memory.SharedMemory(create=True, size=games_per_process * TOKENS_PER_BOARD)
        self.answers = shared_memory.SharedMemory(create=True, size=games_per_process * ANSWER_WIDTH * np.dtype(np.float32).itemsize)
        self.worker_end, self.server_end = multiprocessing.Pipe()

    def close(self) -> None:
        for memory in (self.boards, self.answers):
            memory.close()
            memory.unlink()

def slot_arrays(boards_name: str, answers_name: str, games_per_process: int) -> tuple[list, np.ndarray, np.ndarray]:
    memories = [shared_memory.SharedMemory(name=boards_name), shared_memory.SharedMemory(name=answers_name)]
    boards = np.ndarray((games_per_process, TOKENS_PER_BOARD), dtype=np.uint8, buffer=memories[0].buf)
    answers = np.ndarray((games_per_process, ANSWER_WIDTH), dtype=np.float32, buffer=memories[1].buf)
    return memories, boards, answers


#the network as a worker sees it, through its slot. The answers stay valid until the next call
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


#one cpu-only self play process: sends (worker_id, game) as games end, (worker_id, None) once stopped and finished,
#or its traceback on failure
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


#the inference process and the workers, which can grow and shrink while it runs. A finished worker's slot is reused,
#a failed one's is not (its last request may still be answered into it)
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
        from engine.inference import inference_process
        slot_names = [(slot.boards.name, slot.answers.name) for slot in self.slots]
        self.inference = context.Process(target=inference_process, daemon=True,
                                         args=(checkpoint_path, compile_model, precision, games_per_process, slot_names, [slot.server_end for slot in self.slots],
                                               self.inference_ready, self.inference_stop, self.served_calls, self.served_positions))
        self.inference.start()

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

    def finished(self, worker_id: int) -> None:
        self.done.add(worker_id)
        self.free_slots.append(self.slot_of[worker_id])

    def failed(self, worker_id: int) -> None:
        self.done.add(worker_id)

    #the ones still starting new games
    def playing(self) -> list[int]:
        return [worker_id for worker_id in self.processes if worker_id not in self.done and not self.stops[worker_id].is_set()]

    def can_start(self) -> bool:
        return bool(self.free_slots)

    #the newest one finishes its games and goes
    def stop_one(self) -> None:
        self.stops[self.playing()[-1]].set()

    def stop_all(self) -> None:
        for stop in self.stops.values():
            stop.set()

    def all_done(self) -> bool:
        return len(self.done) == len(self.processes)

    #the ones killed from outside, now counted as failed
    def dead(self) -> list[int]:
        dead = [worker_id for worker_id, process in self.processes.items() if worker_id not in self.done and not process.is_alive()]
        for worker_id in dead:
            self.failed(worker_id)
        return dead

    #games still being played are lost
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


#keys pressed since the last look; none without a windows console
def pressed_keys() -> list[str]:
    keys = []
    try:
        while msvcrt is not None and msvcrt.kbhit():
            keys.append(msvcrt.getwch())
    except OSError:
        pass
    return keys


#windows only; child processes inherit it
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
    from training.data import PROCESSED_DATA_DIR
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
        open_tablebase().close() #fails here rather than in every worker
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

    #imported here, it is slow to import
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=str(RUNS_DIR / "selfplay" / name))
    progress = tqdm(total=positions, initial=games.positions(), unit="pos", smoothing=0.05, dynamic_ncols=True, desc=name)

    #once finishing, every process finishes its games and starts no more, so every game begun is saved
    finishing = False
    def finish() -> None:
        nonlocal finishing
        finishing = True
        workers.stop_all()
        progress.set_description(f"{name}, finishing the games being played")

    #the first ctrl+c only asks for the finish: an exception could land in the middle of writing a game
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
                #on windows a ctrl+c breaks off the wait after interrupt has run, and python does not retry it
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
    print(f"fine-tune on it: python -m training.train --init-checkpoint {checkpoint_path} --training-set {name} --mix-set full_sf8_30M_mix")


if __name__ == "__main__":
    from tools.checkpoint import latest_checkpoint
    parser = argparse.ArgumentParser(description="the model plays itself with mcts.py's search, and every position is saved as a training set")
    parser.add_argument("checkpoint", nargs="?", type=Path, help="a model.pt, default the run that finished last")
    parser.add_argument("--name", required=True, help="the training set that is made in processed_data/")
    parser.add_argument("--positions", type=int, required=True, help="stop once the set has this many positions, e.g. 1_000_000")
    parser.add_argument("--simulations", type=int, default=SIMULATIONS, help="per move, default %(default)s")
    parser.add_argument("--tablebase", action="store_true", help="end a game once it has at most 5 pieces, with the syzygy tables' result")
    parser.add_argument("--processes", type=int, default=PROCESSES, help=f"at the start, + and - change it while it runs (up to {MAX_PROCESSES}). Default %(default)s")
    parser.add_argument("--games-per-process", type=int, default=GAMES_PER_PROCESS, help="games each process plays at once. Default %(default)s")
    parser.add_argument("--no-tree-reuse", action="store_true", help="start every search from a new root (see TREE_REUSE)")
    parser.add_argument("--no-compile", action="store_true", help="skip torch.compile in the inference process (10-30 s at the start, 40%% faster gpu)")
    parser.add_argument("--precision", choices=["fp16", "bf16", "tf32"], default="fp16",
                        help="of the network in the inference process (see inference.fp16_model and Bf16). Default %(default)s")
    parser.add_argument("--seed", type=int, default=SEED, help="default %(default)s")
    arguments = parser.parse_args()

    main(arguments.checkpoint or latest_checkpoint(), arguments.name, arguments.positions, arguments.simulations, arguments.tablebase,
         arguments.processes, arguments.games_per_process, arguments.seed, not arguments.no_tree_reuse,
         not arguments.no_compile, arguments.precision)

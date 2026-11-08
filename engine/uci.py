import argparse
from collections.abc import Iterable
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
import chess
import torch
from engine.inference import cpu_batches
from model.checkpoint import latest_checkpoint, load_model, served_value_sharpness
from engine.engine_search import (ADVANCE_THREADS, CACHE_POSITIONS, ENGINE_FP16_BLOCKS, ENGINE_PRECISION, GC_THREADS, LEAVES, PRECISIONS,
                                  SMART_PRUNING_FACTOR, UNLIMITED, EngineSearch, SearchReport, engine_network, engine_networks,
                                  engine_search_settings, parse_changes, score_to_centipawns)
from engine.mcts import SIMULATIONS
from engine.tablebase import MOST_PIECES, TABLE_SETS, TABLEBASE_PIECES, complete_pieces, open_tablebase, open_tablebase_path

ENGINE_NAME = "Fry"
REPOSITORY = Path(__file__).resolve().parent.parent

MOVE_OVERHEAD_SECONDS = 0.02

def engine_version() -> str:
    version = re.search(r'^version = "([^"]+)"', (REPOSITORY / "pyproject.toml").read_text(), re.MULTILINE).group(1)
    try:
        #git gets no stdin: given the gui's pipe it waits on it
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPOSITORY, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = ""
    return f"{version}-{commit}" if commit else version

def parse_gpus(text: str) -> list[int]:
    if text == "all":
        count = torch.cuda.device_count()
        if not count:
            raise ValueError("there is no cuda device")
        return list(range(count))
    devices = [int(device) for device in text.split(",")]
    if min(devices) < 0 or len(set(devices)) < len(devices):
        raise ValueError(f"{text} is not a list of different cuda devices")
    return devices

#argparse shows an ArgumentTypeError's message but not a ValueError's
def gpus_argument(text: str) -> list[int]:
    try:
        return parse_gpus(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error

GO_NUMBERS = ("wtime", "btime", "winc", "binc", "movestogo", "movetime", "nodes", "depth", "mate")
#with wtime/btime and no movestogo, the game is planned to last this many moves a man above the two kings
MOVES_PER_MAN = 1.5
LEAST_MOVES_LEFT = 10
#a move's weight by the men on the board, kings included: (fewest, most, weight)
PHASE_WEIGHTS = ((26, 32, 0.85), (16, 25, 1.25), (10, 15, 1.0), (3, 9, 0.8))
RESERVE_SECONDS = 1.0
#a search whose score falls may run on by up to MOST_EXTENSION shares (Clock.share), never past MOST_OF_THE_TIME_LEFT_EXTENDED of the time left
MOST_EXTENSION = 2.0
MOST_OF_THE_TIME_LEFT_EXTENDED = 1 / 3
FIRST_MOVE_BONUS = 0.0
BOOK_PLY_BONUS = 0.0
MOST_BOOK_PLIES = 12

#depth and mate are read only to flag a value that is not whole; the search ignores them, as searchmoves
def go_numbers(arguments: list[str]) -> tuple[dict[str, int], list[str]]:
    numbers, bad = {}, []
    for key, value in zip(arguments, arguments[1:]):
        if key in GO_NUMBERS:
            try:
                numbers[key] = int(value)
            except ValueError:
                bad.append(f"{key} {value}")
    return numbers, bad

def phase_weight(men: int) -> float:
    return next((weight for fewest, most, weight in PHASE_WEIGHTS if fewest <= men <= most), 1.0)

#one lock: print writes a line and its newline apart, and the reader thread sends readyok during a search; flushed, as the gui reads a pipe
SENDING = threading.Lock()

def send(line: str) -> None:
    with SENDING:
        print(line, flush=True)

#shakmaty's (the rust search's) limit that python-chess's status leaves out: a side's pawns plus its pieces beyond the start's are at most 8
def standard_material(board: chess.Board, color: chess.Color) -> bool:
    ours = board.occupied_co[color]
    at_the_start = ((board.queens, 1), (board.rooks, 2), (board.knights, 2), (board.bishops & chess.BB_LIGHT_SQUARES, 1),
                    (board.bishops & chess.BB_DARK_SQUARES, 1))
    promoted = sum(max(0, chess.popcount(pieces & ours) - start) for pieces, start in at_the_start)
    return chess.popcount(board.pawns & ours) + promoted <= 8

def parse_position(arguments: list[str]) -> chess.Board:
    moves_index = arguments.index("moves") if "moves" in arguments else len(arguments)
    setup, moves = arguments[:moves_index], arguments[moves_index + 1:]

    if setup[:1] == ["startpos"]:
        board = chess.Board()
    elif setup[:1] == ["fen"]:
        board = chess.Board(" ".join(setup[1:]))
        #python-chess cleans bad castling rights and en passant squares in fen(), which is what the rust search gets
        refused = board.status() & ~(chess.STATUS_BAD_CASTLING_RIGHTS | chess.STATUS_INVALID_EP_SQUARE)
        if refused or not all(standard_material(board, color) for color in chess.COLORS):
            raise ValueError(f"not a position the search can play: {board.fen()}")
    else:
        raise ValueError(f"unknown position: {' '.join(arguments)}")

    for uci_move in moves:
        move = board.parse_uci(uci_move)
        if not move:
            raise ValueError(f"a null move ({uci_move}) in {board.fen()}")
        board.push(move)

    return board

#depth is the length of the expected line: cutechess keeps score, depth and time in the pgn. As lc0's, nodes are the root's visits, those of a kept tree
#included, and nps the speed of this search's own
def info_line(report: SearchReport, seconds: float) -> str:
    nps = round((report.simulations - report.reused) / max(seconds, 1e-3))
    return (f"info depth {len(report.line)} nodes {report.simulations} time {round(seconds * 1000)} nps {nps} "
            f"score cp {score_to_centipawns(report.score)} pv {' '.join(move.uci() for move in report.line)}")

#a new game needs a new Clock
class Clock:
    def __init__(self, moves_per_man: float = MOVES_PER_MAN, least_moves_left: int = LEAST_MOVES_LEFT, reserve_seconds: float = RESERVE_SECONDS,
                 first_move_bonus: float = FIRST_MOVE_BONUS, book_ply_bonus: float = BOOK_PLY_BONUS,
                 move_overhead_seconds: float = MOVE_OVERHEAD_SECONDS):
        self.moves_per_man, self.least_moves_left, self.reserve_seconds = moves_per_man, least_moves_left, reserve_seconds
        self.first_move_bonus, self.book_ply_bonus = first_move_bonus, book_ply_bonus
        self.move_overhead_seconds = move_overhead_seconds
        self.first_search = True
        self.saved = 0.0 #the bank: seconds given to earlier searches and not used
        self.allotted: float | None = None
        self.share = 0.0 #the move's share of the time left, increment included
        self.last_score: int | None = None #centipawns for us
        #(last_score, the seconds this search may add if its score falls below it), None off the clock
        self.extension: tuple[int, float] | None = None

    #(simulations, seconds) for a search of `board`, each None when go does not set it; a go with neither runs `simulations`
    def limits(self, arguments: list[str], board: chess.Board, simulations: int) -> tuple[int | None, float | None]:
        numbers, _ = go_numbers(arguments)
        time_left, increment = ("wtime", "winc") if board.turn == chess.WHITE else ("btime", "binc")
        seconds = self.allotted = self.extension = None
        if "movetime" in numbers:
            seconds = max(0.0, numbers["movetime"] / 1000 - self.move_overhead_seconds)
        elif time_left in numbers:
            seconds_left = numbers[time_left] / 1000 - self.move_overhead_seconds
            bonus = 1.0
            if self.first_search:
                bonus += self.first_move_bonus + self.book_ply_bonus * min(MOST_BOOK_PLIES, board.ply())
                self.first_search = False
            seconds = self.allotted = self.seconds_for_move(seconds_left, numbers.get(increment, 0) / 1000, numbers.get("movestogo"),
                                                            chess.popcount(board.occupied), bonus)
            if self.last_score is not None:
                self.extension = (self.last_score, self.extra_seconds(seconds_left))
        #go nodes 0 plays at once on the root's one simulation, as lc0 does; Searches.start takes at most UNLIMITED
        nodes = min(UNLIMITED, max(1, numbers["nodes"])) if "nodes" in numbers else None
        if nodes is None and seconds is None:
            nodes = simulations
        return nodes, seconds

    def seconds_for_move(self, time_left: float, increment: float, moves_to_go: int | None, men: int, bonus: float = 1.0) -> float:
        spare = max(0.0, time_left - self.reserve_seconds)
        self.saved = min(self.saved, spare)
        #the bank goes into this move, so the moves after are planned without it
        weight = 1.0 if moves_to_go else phase_weight(men)
        weighted_left = moves_to_go or max(self.least_moves_left * weight, sum(self.moves_per_man * phase_weight(k) for k in range(3, men + 1)))
        share = self.share = (spare - self.saved) * weight / weighted_left + increment
        own = share * bonus
        seconds = max(0.0, min(own + self.saved, spare))
        #the bank pays for what the move gets beyond its share and bonus
        self.saved -= max(0.0, seconds - own)
        return seconds

    def extra_seconds(self, time_left: float) -> float:
        return max(0.0, min(MOST_EXTENSION * self.share, (time_left - self.reserve_seconds) * MOST_OF_THE_TIME_LEFT_EXTENDED - (self.allotted or 0.0)))

    def searched(self, seconds: float, score: int) -> None:
        if self.allotted:
            self.saved = max(0.0, self.saved + self.allotted - seconds)
        self.last_score = score

def advance_after(player, board: chess.Board, move: chess.Move) -> None:
    if isinstance(player, EngineSearch):
        played = board.copy()
        played.push(move)
        player.advance(played)

#search_options are EngineSearch's keyword arguments
def load_player(checkpoint_path: Path, gpu: bool, leaves: int, tablebase: bool, search_options: dict,
                precision: str, fp16_blocks: int, gpus: list[int] | None = None) -> EngineSearch:
    if not gpu:
        return EngineSearch(cpu_batches(load_model(checkpoint_path, torch.device("cpu"))), 1, use_tablebase=tablebase, **search_options)
    #the checkpoint's own value sharpness replaces the engine's default; one given by the command line or SearchSettings wins over both
    sharpness = served_value_sharpness(checkpoint_path)
    if sharpness is not None:
        search_options = {**search_options, "settings": {"value_sharpness": sharpness, **search_options.get("settings", {})}}
    #on several gpus the networks wait asleep (inference.GraphedBatches' blocking_wait) where each thread would spin a core
    network =engine_network(checkpoint_path, leaves, precision, fp16_blocks) if gpus is None else (
        engine_networks(checkpoint_path, leaves, precision, fp16_blocks, gpus, len(gpus) > 1))
    return EngineSearch(network, leaves, use_tablebase=tablebase, **search_options)


#a go as the reader saw it, received when it came (the gui times the move from then). stop is set by stop, quit and the input's end (for a go
#that waits for a stop: until_stop)
class Go:
    def __init__(self, arguments: list[str], received: float, until_stop: bool):
        self.arguments, self.received, self.until_stop = arguments, received, until_stop
        self.stop = threading.Event()

#stdin is read on a thread of its own, so that stop, isready and quit reach the engine while it searches; the main thread takes the other
#commands in order (next). While the command queued last is an unfinished go, the reader answers isready itself
class Commands:
    def __init__(self, lines: Iterable[str], infinite_until_stop: bool):
        self.infinite_until_stop = infinite_until_stop
        self.lock = threading.Lock()
        self.unfinished: list[Go] = []
        self.last_go: Go | None = None #the command queued last, if it was a go
        self.queued: queue.Queue = queue.Queue()
        threading.Thread(target=self.read, args=(lines,), daemon=True).start()

    def read(self, lines: Iterable[str]) -> None:
        try:
            for line in lines:
                self.receive(line.split(), time.perf_counter())
        finally:
            with self.lock:
                for go in self.unfinished:
                    if go.until_stop:
                        go.stop.set()
            self.queued.put(None)

    def receive(self, words: list[str], received: float) -> None:
        if not words:
            return
        command, arguments = words[0], words[1:]
        with self.lock:
            if command == "isready" and self.last_go in self.unfinished:
                send("readyok")
            elif command == "stop":
                for go in self.unfinished:
                    go.stop.set()
            else:
                self.last_go = Go(arguments, received, self.infinite_until_stop and "infinite" in arguments) if command == "go" else None
                if self.last_go is not None:
                    self.unfinished.append(self.last_go)
                if command == "quit":
                    for go in self.unfinished:
                        go.stop.set()
                self.queued.put((command, arguments, self.last_go))

    #(command, arguments, its Go for a go), None once the input has ended
    def next(self) -> tuple[str, list[str], Go | None] | None:
        return self.queued.get()

    def finished(self, go: Go) -> None:
        with self.lock:
            self.unfinished.remove(go)

#setoption's whole number options, (lowest, highest); MoveOverhead is in ms, RamLimitMb in MB
SPIN_OPTIONS = {"SyzygyProbeLimit": (min(table_set.pieces for table_set in TABLE_SETS), max(table_set.pieces for table_set in TABLE_SETS)),
                "MoveOverhead": (0, 5000), "CachePositions": (0, 1_000_000_000), "RamLimitMb": (0, 100_000_000), "AdvanceThreads": (0, 256),
                "GcThreads": (0, 64), "Leaves": (1, 1024)}
#setoption's decimal options, as strings since uci has no decimal type, (lowest, highest); SmartPruningFactor 0 is no smart pruning
DECIMAL_OPTIONS = {"SmartPruningFactor": (0.0, 100.0), "FirstMoveBonus": (0.0, 100.0), "BookPlyBonus": (0.0, 100.0)}
GPU_OPTIONS = ("GPUs", "Leaves", "SmartPruningFactor")
BUILD_OPTIONS = ("SyzygyPath", "SyzygyProbeLimit", "CachePositions", "RamLimitMb", "AdvanceThreads", "GcThreads", "GPUs", "Leaves", "SearchSettings",
                 "SmartPruningFactor")

#setoption's "name <name> value <value>", each of any number of words (a path may hold spaces)
def option_name_and_value(arguments: list[str]) -> tuple[str, str]:
    value_at = arguments.index("value") if "value" in arguments else len(arguments)
    return " ".join(arguments[1:value_at]), " ".join(arguments[value_at + 1:])

#the uci options, the command line's choices their defaults (<empty> goes back to the command line's). SyzygyPath: folders of tables, os.pathsep
#between them. SearchSettings: --search-settings' "name=value,..." as changes over the command line's
class Options:
    def __init__(self, tablebase: bool, tablebase_pieces: int, gpu: bool = False, gpus: list[int] | None = None, leaves: int = LEAVES,
                 search_options: dict | None = None, clock_options: dict | None = None):
        search_options, clock_options = search_options or {}, clock_options or {}
        self.tablebase, self.default_pieces = tablebase, tablebase_pieces
        self.syzygy_path: str | None = None #None for the command line's tables
        self.tablebase_pieces = tablebase_pieces
        self.probe_limit: int | None = None #None for SyzygyProbeLimit's default
        self.move_overhead_seconds = MOVE_OVERHEAD_SECONDS
        self.cache_positions = CACHE_POSITIONS
        self.ram_limit_mb = 0
        self.advance_threads = ADVANCE_THREADS
        self.gc_threads = GC_THREADS
        self.gpu, self.command_line_gpus, self.gpus, self.leaves = gpu, gpus, gpus, leaves
        self.command_line_settings = search_options.get("settings", {})
        self.settings: dict | None = None #None for the command line's alone
        self.command_line_factor = search_options.get("smart_pruning_factor", SMART_PRUNING_FACTOR) if search_options.get("smart_pruning", True) else 0.0
        self.smart_pruning_factor: float | None = None #None for the command line's
        self.first_move_bonus = clock_options.get("first_move_bonus", FIRST_MOVE_BONUS)
        self.book_ply_bonus = clock_options.get("book_ply_bonus", BOOK_PLY_BONUS)

    def names(self) -> list[str]:
        names = ["SyzygyPath", *SPIN_OPTIONS, "GPUs", *DECIMAL_OPTIONS, "SearchSettings"]
        return [name for name in names if self.gpu or name not in GPU_OPTIONS]

    def lines(self) -> list[str]:
        defaults = {"SyzygyProbeLimit": MOST_PIECES, "MoveOverhead": round(MOVE_OVERHEAD_SECONDS * 1000), "CachePositions": CACHE_POSITIONS,
                    "RamLimitMb": 0, "AdvanceThreads": ADVANCE_THREADS, "GcThreads": GC_THREADS, "Leaves": self.leaves, "SmartPruningFactor": self.command_line_factor,
                    "FirstMoveBonus": self.first_move_bonus, "BookPlyBonus": self.book_ply_bonus,
                    "GPUs": ",".join(map(str, self.command_line_gpus)) if self.command_line_gpus else "<empty>"}
        lines = []
        for name in self.names():
            if name in SPIN_OPTIONS:
                lowest, highest = SPIN_OPTIONS[name]
                lines.append(f"option name {name} type spin default {defaults[name]} min {lowest} max {highest}")
            elif name in DECIMAL_OPTIONS:
                lines.append(f"option name {name} type string default {defaults[name]:g}")
            else:
                lines.append(f"option name {name} type string default {defaults.get(name, '<empty>')}")
        return lines

    #applies a setoption: what to tell the gui (an info string), None if nothing. A value the search could not take leaves the option as it was
    def set(self, name: str, value: str, built: bool) -> str | None:
        names = self.names()
        name = next((known for known in names if known.lower() == name.lower()), name)
        if name not in names:
            return f"no option {name}"
        if built and name in BUILD_OPTIONS:
            return f"{name} takes effect when the engine restarts: the search is built"
        if name == "SyzygyPath":
            path = os.pathsep.join(filter(None, value.split(os.pathsep))) if value != "<empty>" else ""
            return self.set_tables(path or None, self.probe_limit)
        if name == "GPUs":
            return self.set_gpus(value)
        if name == "SearchSettings":
            try:
                self.settings = parse_changes("" if value == "<empty>" else value, engine_search_settings(self.tablebase or self.syzygy_path is not None))
            except (ValueError, OverflowError) as error: #a whole number setting of inf
                return f"{error}: the search's settings stay as they were"
            return None
        if name in DECIMAL_OPTIONS:
            return self.set_decimal(name, value)
        lowest, highest = SPIN_OPTIONS[name]
        try:
            number = int(value)
        except ValueError:
            number = None
        if number is None or not lowest <= number <= highest:
            return f"{name} is a whole number from {lowest} to {highest}, not {value!r}"
        if name == "SyzygyProbeLimit":
            return self.set_tables(self.syzygy_path, number)
        if name == "MoveOverhead":
            self.move_overhead_seconds = number / 1000
        elif name == "CachePositions":
            self.cache_positions = number
        elif name == "RamLimitMb":
            self.ram_limit_mb = number
        elif name == "GcThreads":
            self.gc_threads = number
        elif name == "Leaves":
            if number == 1 and self.gpus is not None and len(self.gpus) > 1:
                return "1 leaf a round is mcts.py's search, on one gpu: Leaves stays as it was"
            self.leaves = number
        else:
            self.advance_threads = number
        return None

    def set_gpus(self, value: str) -> str | None:
        if value == "<empty>":
            gpus = self.command_line_gpus
        else:
            try:
                gpus = parse_gpus(value)
            except ValueError as error:
                return f"{error}: GPUs stays as it was"
            if max(gpus) >= torch.cuda.device_count():
                return f"there is no cuda device {max(gpus)} ({torch.cuda.device_count()} found): GPUs stays as it was"
        if gpus is not None and len(gpus) > 1 and self.leaves == 1:
            return "several gpus need Leaves above 1: GPUs stays as it was"
        self.gpus = gpus
        return None

    def set_decimal(self, name: str, value: str) -> str | None:
        lowest, highest = DECIMAL_OPTIONS[name]
        try:
            number = float(value)
        except ValueError:
            number = None
        if number is None or not lowest <= number <= highest:
            return f"{name} is a number from {lowest:g} to {highest:g}, not {value!r}"
        if name == "SmartPruningFactor":
            self.smart_pruning_factor = number
        elif name == "FirstMoveBonus":
            self.first_move_bonus = number
        else:
            self.book_ply_bonus = number
        return None

    #the tables of `path` (None for the command line's) up to `limit` (None for the default). A SyzygyPath gives the most pieces it holds up to
    #the limit, at least TABLEBASE_PIECES; the command line's take the limit or --tablebase-pieces. What to tell the gui, None if nothing
    def set_tables(self, path: str | None, limit: int | None) -> str | None:
        if path is None:
            pieces = self.default_pieces if limit is None else limit
        else:
            try:
                pieces = complete_pieces(path, MOST_PIECES if limit is None else limit)
            except OSError as error:
                return f"{error}: the tables stay as they were"
            if pieces < TABLEBASE_PIECES:
                return f"{path} does not hold every table of up to {TABLEBASE_PIECES} pieces: the tables stay as they were"
        problem = self.tables_problem(path, pieces)
        if problem is None:
            self.syzygy_path, self.tablebase_pieces, self.probe_limit = path, pieces, limit
        return problem

    #why the search could not open the tables of `path` (None for the command line's) up to `pieces`, None if it can
    def tables_problem(self, path: str | None, pieces: int) -> str | None:
        try:
            if path is not None:
                open_tablebase_path(path, pieces)[0].close()
            elif self.tablebase:
                open_tablebase(pieces).close()
        except (OSError, ValueError) as error:
            return f"{error}: the tables stay as they were"
        return None

    #(whether the search uses tables, the search options)
    def search(self) -> tuple[bool, dict]:
        options = {"tablebase_pieces": self.tablebase_pieces, "syzygy_path": self.syzygy_path, "cache_positions": self.cache_positions,
                   "memory_limit_mb": self.ram_limit_mb, "advance_threads": self.advance_threads, "gc_threads": self.gc_threads}
        if self.settings is not None:
            options["settings"] = {**self.command_line_settings, **self.settings}
        if self.smart_pruning_factor == 0:
            options["smart_pruning"] = False
        elif self.smart_pruning_factor is not None:
            options.update(smart_pruning=True, smart_pruning_factor=self.smart_pruning_factor)
        return self.tablebase or self.syzygy_path is not None, options

    def clock(self) -> dict:
        return {"first_move_bonus": self.first_move_bonus, "book_ply_bonus": self.book_ply_bonus, "move_overhead_seconds": self.move_overhead_seconds}


def main(checkpoint_path: Path, simulations: int = SIMULATIONS, gpu: bool = False, leaves: int = LEAVES,
         tablebase: bool = False, search_options: dict | None = None, precision: str = ENGINE_PRECISION, fp16_blocks: int = ENGINE_FP16_BLOCKS,
         clock_options: dict | None = None, infinite_until_stop: bool = False, gpus: list[int] | None = None):
    torch.set_num_threads(1) #so that engines running side by side don't fight over cores
    search_options = search_options or {}
    clock_options = clock_options or {}
    options = Options(tablebase, search_options.get("tablebase_pieces", TABLEBASE_PIECES), gpu, gpus, leaves, search_options, clock_options)
    player = None
    board = chess.Board()
    clock = Clock(**options.clock())
    commands = Commands(sys.stdin, infinite_until_stop)
    newest: list[chess.Move] = [] #the move of the newest info line of the search under way, which answers if the search fails

    def load() -> EngineSearch:
        use_tablebase, set_options = options.search()
        return load_player(checkpoint_path, gpu, options.leaves, use_tablebase, {**search_options, **set_options}, precision, fp16_blocks,
                           options.gpus)

    #go's move, None in a game that is over. A go until stop returns once told to stop, also when its search ended by itself, as UCI asks
    def move_for(go: Go) -> chess.Move | None:
        nonlocal player
        newest.clear()
        if not any(board.legal_moves):
            send("info string the game is over, there is no move to play")
            return None
        if player is None:
            player = load()
        for number in go_numbers(go.arguments)[1]:
            send(f"info string go's {number} is not a whole number: left out")
        nodes, seconds = clock.limits(go.arguments, board, simulations)
        if go.until_stop:
            nodes = seconds = None

        def running(report: SearchReport) -> None:
            newest[:] = report.line[:1]
            send(info_line(report, time.perf_counter() - go.received))
        report = player.search(board, nodes, seconds, clock.extension, go.stop, running)
        clock.searched(time.perf_counter() - go.received, score_to_centipawns(report.score))
        move = report.line[0]
        send(info_line(report, time.perf_counter() - go.received))
        if go.until_stop:
            go.stop.wait()
        return move

    #a search that fails still answers, with its newest info line's move or else the first legal move; the gui is told, stderr has the traceback. The
    #player is built again for the next go, as its search may be left half done
    def fallback_move(error: BaseException) -> chess.Move:
        nonlocal player
        traceback.print_exception(error)
        move = newest[0] if newest else next(iter(board.legal_moves))
        send(f"info string the search failed ({error!r}): playing {move.uci()}")
        player = None
        return move

    while (received := commands.next()) is not None:
        command, arguments, go = received

        if command == "uci":
            send(f"id name {ENGINE_NAME} {engine_version()}")
            send("id author William Gong")
            for line in options.lines():
                send(line)
            send("uciok")

        elif command == "isready":
            #loaded here, not at startup, so "uci" is answered at once
            if player is None:
                player = load()
            send("readyok")

        elif command == "setoption":
            told = options.set(*option_name_and_value(arguments), built=player is not None)
            for name, value in options.clock().items():
                setattr(clock, name, value)
            if told is not None:
                send(f"info string {told}")

        elif command == "ucinewgame":
            board = chess.Board()
            clock = Clock(**options.clock())
            if isinstance(player, EngineSearch):
                player.new_game()

        elif command == "position":
            try:
                board = parse_position(arguments)
            except ValueError as error:
                send(f"info string {error}: the last position stays")

        elif command == "go":
            try:
                move = move_for(go)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as error: #pyo3's PanicException, a rust panic, is not an Exception
                move = fallback_move(error)
            send(f"bestmove {move.uci() if move is not None else '0000'}")
            if move is not None:
                advance_after(player, board, move)
            commands.finished(go)

        elif command == "quit":
            break

        #anything else (debug, register, ...) is ignored, as UCI allows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="the model as a UCI engine")
    parser.add_argument("checkpoint", nargs="?", help="a model.pt; default the run that finished last")
    parser.add_argument("--simulations", type=int, default=SIMULATIONS,
                        help="the simulations of a go without movetime, nodes or wtime/btime, default %(default)s")
    parser.add_argument("--gpu", action="store_true", help="run the model on the gpu")
    parser.add_argument("--leaves", type=int, default=LEAVES,
                        help="with --gpu: leaves gathered for each network call, default %(default)s; 1 is mcts.py's search")
    parser.add_argument("--precision", choices=PRECISIONS,
                        help=f"with --gpu: the network's precision (engine_search.PRECISIONS), default {ENGINE_PRECISION}")
    parser.add_argument("--gpus", type=gpus_argument,
                        help="with --gpu: the cuda devices to search on at once, e.g. 0,1,2,3 or all, a thread driving each; "
                             "default the current one. On more than one, for --leaves above 1, the search is another (more leaves out): the "
                             "threads descend in the one tree at once, each pinned to a cpu by the topology and asleep while its gpu works")
    parser.add_argument("--fp16-blocks", type=int,
                        help=f"for the int8 network: the blocks kept in fp16 besides the last, those whose int8 costs the policy most "
                             f"(engine/quantization.py), default {ENGINE_FP16_BLOCKS}; 0 for every other block in int8")
    parser.add_argument("--tablebase", action="store_true", help="positions of at most --tablebase-pieces pieces from the syzygy tables")
    parser.add_argument("--tablebase-pieces", type=int, choices=[table_set.pieces for table_set in TABLE_SETS], default=TABLEBASE_PIECES,
                        help="for --tablebase: the most pieces, default %(default)s; 6 needs python -m engine.tablebase --pieces 6, and without "
                             "the 6 piece dtz tables searches those positions rather than playing the tables' move")
    parser.add_argument("--no-smart-pruning", dest="smart_pruning", action="store_false",
                        help="with --gpu: search all of go's movetime or nodes even once the move is decided (lc0's SmartPruningFactor=0)")
    parser.add_argument("--smart-pruning-factor", type=float, default=SMART_PRUNING_FACTOR,
                        help="with --gpu: smart pruning stops once the most visited move leads by more than the simulations left / "
                             "this, default %(default)s; lc0's 1.33 bets that the second move cannot get all of them")
    parser.add_argument("--first-move-bonus", type=float, default=FIRST_MOVE_BONUS,
                        help="on a clock: the game's first search gets 1 + this + --book-ply-bonus x its ply (at most "
                             f"{MOST_BOOK_PLIES}) times its share, default %(default)s; lc0's 1.8")
    parser.add_argument("--book-ply-bonus", type=float, default=BOOK_PLY_BONUS, help="for --first-move-bonus, default %(default)s; lc0's 0.25")
    parser.add_argument("--search-settings", default="", help="changes to the search's settings (engine_search.py), e.g. c_puct=0.6")
    parser.add_argument("--infinite-until-stop", action="store_true",
                        help="go infinite searches until stop, as UCI and analysis guis ask; without it it runs --simulations, which "
                             "cutechess-cli's tc=inf games need")
    arguments = parser.parse_args()
    if arguments.tablebase_pieces != TABLEBASE_PIECES and not arguments.tablebase:
        parser.error("--tablebase-pieces needs --tablebase")
    if arguments.precision is not None and not arguments.gpu:
        parser.error("--precision needs --gpu")
    if arguments.smart_pruning_factor <= 0:
        parser.error("--smart-pruning-factor is positive; --no-smart-pruning for none")
    if arguments.gpus is not None and not (arguments.gpu and (len(arguments.gpus) == 1 or arguments.leaves > 1)):
        parser.error("--gpus needs --gpu, and on more than one gpu --leaves above 1")
    if arguments.fp16_blocks is not None and not (arguments.gpu and arguments.precision in (None, "int8")):
        parser.error("--fp16-blocks needs --gpu and the int8 network")
    if arguments.fp16_blocks is not None and arguments.fp16_blocks < 0:
        parser.error("--fp16-blocks is 0 or more")
    main(Path(arguments.checkpoint) if arguments.checkpoint else latest_checkpoint(),
         simulations=arguments.simulations, gpu=arguments.gpu, leaves=arguments.leaves, tablebase=arguments.tablebase,
         precision=arguments.precision or ENGINE_PRECISION, infinite_until_stop=arguments.infinite_until_stop, gpus=arguments.gpus,
         fp16_blocks=ENGINE_FP16_BLOCKS if arguments.fp16_blocks is None else arguments.fp16_blocks,
         search_options={"smart_pruning": arguments.smart_pruning, "tablebase_pieces": arguments.tablebase_pieces,
                         "settings": parse_changes(arguments.search_settings, engine_search_settings(arguments.tablebase)),
                         "smart_pruning_factor": arguments.smart_pruning_factor},
         clock_options={"first_move_bonus": arguments.first_move_bonus, "book_ply_bonus": arguments.book_ply_bonus})

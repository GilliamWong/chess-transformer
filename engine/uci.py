#the model as a uci engine, for cutechess-cli matches, tournament managers and the cute chess gui: uci, isready,
#setoption (Options), ucinewgame, position, go, stop, ponderhit and quit, read on a thread of their own (Commands) so
#that stop, isready and quit reach a running search. The search (--select mcts) keeps to go's movetime, nodes and
#wtime/btime; a go without them runs --simulations, go infinite too (cutechess sends it for tc=inf) unless
#--infinite-until-stop. go ponder searches until ponderhit, then as the go asks; its searchmoves, depth and mate are
#ignored. The other selections ignore the clock
#   python -m engine.uci [model.pt | random] [--select mcts --gpu]
import argparse
from collections.abc import Iterable
import json
import os
from pathlib import Path
import queue
import random
import sys
import threading
import time
import chess
import torch
from torch import nn
from engine.inference import cpu_batches
from tools.checkpoint import latest_checkpoint
from engine.engine_search import (ADVANCE_THREADS, CACHE_POSITIONS, ENGINE_FP16_BLOCKS, ENGINE_PRECISION, GROW_LEAVES, LEAVES, PRECISIONS,
                                  SMART_PRUNING_FACTOR, SMART_PRUNING_WAIT, THREADS_PER_GPU, VISITS_PER_LEAF_OUT, EngineSearch, engine_network, engine_networks,
                                  engine_search_settings, parse_changes)
from engine.player import (SAMPLE_PLIES, SELECTIONS, SIMULATIONS, TEMPERATURE, CudaGraphModel, SearchReport, choose_move,
                               load_model, score_to_centipawns)
from tools.tablebase import TABLE_SETS, TABLEBASE_PIECES, open_tablebase, open_tablebase_path

ENGINE_NAME = "ChessTransformer"

#one board takes a few ms on the cpu, less than starting cuda; one thread so parallel games don't fight over cores
DEVICE = torch.device("cpu")
#the time between the search's end and the gui reading bestmove, kept off go's time (uci option MoveOverhead)
MOVE_OVERHEAD_SECONDS = 0.02
#go's arguments that take a number
GO_NUMBERS = ("wtime", "btime", "winc", "binc", "movestogo", "movetime", "nodes", "depth", "mate")
#with wtime/btime and no movestogo, the time left is planned over the moves the game still has, this many a man above
#the two kings: the plan follows the material rather than a move number
MOVES_PER_MAN = 1.5
#the fewest moves the time left is shared over, so the last of it goes gradually
LEAST_MOVES_LEFT = 10
#a move's weight in the plan by the men on the board, kings included, as (fewest, most, weight): the middlegame gets more
PHASE_WEIGHTS = ((26, 32, 0.85), (16, 25, 1.25), (10, 15, 1.0), (3, 9, 0.8))
#kept back from the time left, for a search that runs over
RESERVE_SECONDS = 1.0
#a move gets at most this part of the time left, and never the reserve: no cap, as in lc0
MOST_OF_THE_TIME_LEFT = 1.0
#a search whose score falls below the last one's may run on (engine_search.py FALL_START_CP) by up to this many of its
#move's shares, never past this part of the time left
MOST_EXTENSION = 2.0
MOST_OF_THE_TIME_LEFT_EXTENDED = 1 / 3
#lc0's legacy time manager gives a game's first search on the clock 1 + its first move bonus (lc0's 1.8) + its book ply
#bonus (0.25) x the plies before it, at most MOST_BOOK_PLIES, times its share: it starts without a kept tree. 0 for none
FIRST_MOVE_BONUS = 0.0
BOOK_PLY_BONUS = 0.0
MOST_BOOK_PLIES = 12

#go's numbers (GO_NUMBERS) by name, and those that are not whole numbers ("name value"), which are left out
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

#print writes the line and its newline apart, and the reader thread answers isready while a search sends its lines
SENDING = threading.Lock()

def send(line: str) -> None:
    #flushed: the gui reads a pipe, not a terminal
    with SENDING:
        print(line, flush=True)

#shakmaty's (the rust search's) check of a fen that python-chess's status leaves out: a side's pawns and its pieces
#beyond the start's (a bishop of each colour) are at most 8
def standard_material(board: chess.Board, color: chess.Color) -> bool:
    ours = board.occupied_co[color]
    at_the_start = ((board.queens, 1), (board.rooks, 2), (board.knights, 2), (board.bishops & chess.BB_LIGHT_SQUARES, 1),
                    (board.bishops & chess.BB_DARK_SQUARES, 1))
    promoted = sum(max(0, chess.popcount(pieces & ours) - start) for pieces, start in at_the_start)
    return chess.popcount(board.pawns & ours) + promoted <= 8

#"startpos moves e2e4 e7e5" or "fen <6 fen fields> moves ..."; a ValueError for one the search could not start from: a
#bad fen, a board the rust search refuses (its castling rights and en passant square python-chess cleans in fen()),
#an illegal or a null move
def parse_position(arguments: list[str]) -> chess.Board:
    if "moves" in arguments:
        moves_index = arguments.index("moves")
        setup, moves = arguments[:moves_index], arguments[moves_index + 1:]
    else:
        setup, moves = arguments, []

    if setup[:1] == ["startpos"]:
        board = chess.Board()
    elif setup[:1] == ["fen"]:
        board = chess.Board(" ".join(setup[1:]))
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

def choose_random_move(board: chess.Board) -> chess.Move:
    return random.choice(list(board.legal_moves))

#depth is the length of the expected line; cutechess keeps score, depth and time in the pgn
def info_line(report: SearchReport, seconds: float) -> str:
    return (f"info depth {len(report.line)} nodes {report.simulations} time {round(seconds * 1000)} nps {round(report.simulations / max(seconds, 1e-3))} "
            f"score cp {score_to_centipawns(report.score)} pv {' '.join(move.uci() for move in report.line)}")

#go's limits for the search, and on a clock (wtime/btime) the time a move gets: its weighted share of the time left (less
#the reserve and the bank) over the moves still to play, plus the increment, plus the bank: the time earlier searches
#were given and did not use (smart pruning stops some early), as lc0's legacy time manager spends it. A search whose
#score falls may run on (extension), and the game's first search may get a bonus. The arguments are the constants
#above of the same names, for tools/play/variant_match.py to tune. A new game needs a new Clock
class Clock:
    def __init__(self, moves_per_man: float = MOVES_PER_MAN, least_moves_left: int = LEAST_MOVES_LEFT, reserve_seconds: float = RESERVE_SECONDS,
                 most_of_the_time_left: float = MOST_OF_THE_TIME_LEFT, first_move_bonus: float = FIRST_MOVE_BONUS,
                 book_ply_bonus: float = BOOK_PLY_BONUS, move_overhead_seconds: float = MOVE_OVERHEAD_SECONDS):
        self.moves_per_man, self.least_moves_left = moves_per_man, least_moves_left
        self.reserve_seconds, self.most_of_the_time_left = reserve_seconds, most_of_the_time_left
        self.first_move_bonus, self.book_ply_bonus = first_move_bonus, book_ply_bonus
        self.move_overhead_seconds = move_overhead_seconds
        self.first_search = True #the game's first search on the clock is still to come
        self.saved = 0.0 #the seconds earlier searches were given and did not use
        self.allotted: float | None = None #the seconds the search on the clock was given
        self.share = 0.0 #its move's share of the time left
        self.last_score: int | None = None #the last search's score, centipawns for us
        #(the last search's score, the seconds this search may add if its score falls below it), None off the clock
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
        #go nodes 0 plays at once, the root's one simulation (or a kept tree's move), as lc0 answers at once
        nodes = None if "nodes" not in numbers else max(1, numbers["nodes"])
        if nodes is None and seconds is None:
            nodes = simulations
        return nodes, seconds

    #bonus multiplies the move's share, as lc0's first move bonus does before it adds the bank: the time left pays for it
    def seconds_for_move(self, time_left: float, increment: float, moves_to_go: int | None, men: int, bonus: float = 1.0) -> float:
        spare = max(0.0, time_left - self.reserve_seconds)
        self.saved = min(self.saved, spare)
        #the bank goes into this move, so the moves after are planned without it
        weight = 1.0 if moves_to_go else phase_weight(men)
        weighted_left = moves_to_go or max(self.least_moves_left * weight, sum(self.moves_per_man * phase_weight(k) for k in range(3, men + 1)))
        share = self.share = (spare - self.saved) * weight / weighted_left + increment
        own = share * bonus
        seconds = max(0.0, min(own + self.saved, spare, time_left * self.most_of_the_time_left))
        #the bank pays for what the move gets beyond its share and bonus
        self.saved -= max(0.0, seconds - own)
        return seconds

    #the seconds a search whose score falls may add: MOST_EXTENSION shares, never past MOST_OF_THE_TIME_LEFT_EXTENDED of
    #the time left
    def extra_seconds(self, time_left: float) -> float:
        return max(0.0, min(MOST_EXTENSION * self.share, (time_left - self.reserve_seconds) * MOST_OF_THE_TIME_LEFT_EXTENDED - (self.allotted or 0.0)))

    #after a search, the seconds it took and its score (centipawns for us); what it did not use goes to the bank
    def searched(self, seconds: float, score: int) -> None:
        if self.allotted:
            self.saved = max(0.0, self.saved + self.allotted - seconds)
        self.last_score = score

#after bestmove is sent: the search's tree carried on to the game after the move, off the engine's clock, on a thread of
#its own (EngineSearch.advance); the next position need not follow it
def advance_after(player, board: chess.Board, move: chess.Move) -> None:
    if isinstance(player, EngineSearch):
        played = board.copy()
        played.push(move)
        player.advance(played)

#a file per process, so engines playing side by side never share one. evaluate_elo writes them into the pgn
def log_search(log_folder: Path, board: chess.Board, move: chess.Move, report: SearchReport) -> None:
    record = {
        "fen": board.fen(), "move": move.uci(), "score": report.score, "simulations": report.simulations,
        "line": [line_move.uci() for line_move in report.line],
        "root_moves": [[root_move.uci(), visits] for root_move, visits in report.root_moves]
    }
    with open(log_folder / f"{os.getpid()}.jsonl", "a") as log:
        log.write(json.dumps(record) + "\n")

#boards stay on DEVICE either way: CudaGraphModel takes them from the cpu
def load_engine_model(checkpoint_path: Path, gpu: bool) -> nn.Module:
    if gpu:
        return CudaGraphModel(load_model(checkpoint_path, torch.device("cuda")))
    return load_model(checkpoint_path, DEVICE)

#the search for --select mcts: on the gpu `leaves` leaves a round, the network in `precision` (int8 with fp16_blocks), on
#each of the cuda devices `gpus` if given, a device once for each of its networks (a thread driving each); on the cpu
#one, which is mcts.py's search. options are EngineSearch's keyword arguments (smart pruning, the tables' pieces, changes
#to the search's settings). With several networks they wait for the gpu asleep (inference.GraphedBatches'
#blocking_wait), where each thread would spin a core
def load_engine_search(checkpoint_path: Path, gpu: bool, leaves: int, tablebase: bool, options: dict, precision: str,
                       fp16_blocks: int, gpus: list[int] | None = None) -> EngineSearch:
    if gpu:
        network = engine_network(checkpoint_path, leaves, precision, fp16_blocks) if gpus is None else (
            engine_networks(checkpoint_path, leaves, precision, fp16_blocks, gpus, len(gpus) > 1))
        return EngineSearch(network, leaves, use_tablebase=tablebase, **options)
    return EngineSearch(cpu_batches(load_model(checkpoint_path, DEVICE)), 1, use_tablebase=tablebase, **options)

#what picks the moves: the search for --select mcts, else the model
def load_player(checkpoint_path: Path, selection: str, gpu: bool, leaves: int, tablebase: bool, search_options: dict,
                precision: str, fp16_blocks: int, gpus: list[int] | None = None) -> nn.Module | EngineSearch:
    if selection == "mcts":
        return load_engine_search(checkpoint_path, gpu, leaves, tablebase, search_options, precision, fp16_blocks, gpus)
    return load_engine_model(checkpoint_path, gpu)


#a go as the reader saw it: its arguments and when it came (the gui times the move from then). stop is set at stop or
#quit, or at the end of the input for a go that waits for a stop: go infinite with --infinite-until-stop (until_stop)
#and go ponder, which searches until pondered is set (stop or ponderhit, whose time is kept)
class Go:
    def __init__(self, arguments: list[str], received: float, until_stop: bool):
        self.arguments, self.received, self.until_stop = arguments, received, until_stop
        self.ponder = "ponder" in arguments
        self.stop, self.pondered = threading.Event(), threading.Event()
        self.ponderhit: float | None = None

    def end(self) -> None:
        self.stop.set()
        self.pondered.set()

    def hit(self, received: float) -> None:
        if self.ponder and not self.pondered.is_set():
            self.ponderhit = received
            self.pondered.set()

#stdin read on a thread of its own, so that stop, isready and quit reach the engine while it searches. The main thread
#takes the other commands in order (next; None once the input ends) and keeps every search's state. While the command
#queued last is a go the main thread has not finished, the reader answers isready itself (UCI: at once, the search
#going on); stop and quit end the unfinished gos, ponderhit ends their pondering, and the end of the input the gos that
#wait for a stop
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
                    if go.until_stop or go.ponder:
                        go.end()
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
                    go.end()
            elif command == "ponderhit":
                for go in self.unfinished:
                    go.hit(received)
            else:
                self.last_go = Go(arguments, received, self.infinite_until_stop and "infinite" in arguments) if command == "go" else None
                if self.last_go is not None:
                    self.unfinished.append(self.last_go)
                if command == "quit":
                    for go in self.unfinished:
                        go.end()
                self.queued.put((command, arguments, self.last_go))

    #(command, arguments, its Go for a go), None once the input has ended
    def next(self) -> tuple[str, list[str], Go | None] | None:
        return self.queued.get()

    #the main thread is done with the go, its bestmove sent
    def finished(self, go: Go) -> None:
        with self.lock:
            self.unfinished.remove(go)

#setoption's whole number options, (lowest, highest): the tables' most pieces, MoveOverhead in ms, RamLimitMb in MB
SPIN_OPTIONS = {"SyzygyProbeLimit": (min(table_set.pieces for table_set in TABLE_SETS), max(table_set.pieces for table_set in TABLE_SETS)),
                "MoveOverhead": (0, 5000), "CachePositions": (0, 1_000_000_000), "RamLimitMb": (0, 100_000_000), "AdvanceThreads": (1, 256)}
#the options the search is built with (at isready or the first go): set later, they take effect when the engine
#restarts, not at a new game (Searches takes them when it is made)
BUILD_OPTIONS = ("SyzygyPath", "SyzygyProbeLimit", "CachePositions", "RamLimitMb", "AdvanceThreads")

#setoption's "name <name> value <value>", each of any number of words (a path may hold spaces)
def option_name_and_value(arguments: list[str]) -> tuple[str, str]:
    value_at = arguments.index("value") if "value" in arguments else len(arguments)
    return " ".join(arguments[1:value_at]), " ".join(arguments[value_at + 1:])

#the search's uci options, the command line's choices their defaults. SyzygyPath: folders of syzygy tables, os.pathsep
#between them, or <empty> for the command line's (--tablebase, else none); SyzygyProbeLimit: their most pieces;
#CachePositions: the network's answers the search keeps (engine_search.CACHE_POSITIONS); RamLimitMb: the MB (10^6
#bytes) the search's tree may take, 0 for no limit (Searches' memory_limit_mb, without the cache or a kept tree's copy);
#AdvanceThreads: the threads the kept tree is carried on to the next move on (engine_search.ADVANCE_THREADS)
class Options:
    def __init__(self, tablebase: bool, tablebase_pieces: int):
        self.tablebase, self.default_pieces = tablebase, tablebase_pieces
        self.syzygy_path: str | None = None #None for the command line's tables
        self.tablebase_pieces = tablebase_pieces
        self.move_overhead_seconds = MOVE_OVERHEAD_SECONDS
        self.cache_positions = CACHE_POSITIONS
        self.ram_limit_mb = 0
        self.advance_threads = ADVANCE_THREADS

    #the uci reply's lines
    def lines(self) -> list[str]:
        defaults = {"SyzygyProbeLimit": self.default_pieces, "MoveOverhead": round(MOVE_OVERHEAD_SECONDS * 1000), "CachePositions": CACHE_POSITIONS,
                    "RamLimitMb": 0, "AdvanceThreads": ADVANCE_THREADS}
        return (["option name SyzygyPath type string default <empty>"]
                + [f"option name {name} type spin default {defaults[name]} min {lowest} max {highest}" for name, (lowest, highest) in SPIN_OPTIONS.items()])

    #a setoption once the search is built or not: what to tell the gui (an info string), None if nothing. A value the
    #search could not take (tables that are not all there, a number out of range) leaves the option as it was
    def set(self, name: str, value: str, built: bool) -> str | None:
        name = next((known for known in ("SyzygyPath", *SPIN_OPTIONS) if known.lower() == name.lower()), name)
        if name not in ("SyzygyPath", *SPIN_OPTIONS):
            return f"no option {name}"
        if built and name in BUILD_OPTIONS:
            return f"{name} takes effect when the engine restarts: the search is built"
        if name == "SyzygyPath":
            path = os.pathsep.join(filter(None, value.split(os.pathsep))) if value != "<empty>" else ""
            problem = self.tables_problem(path or None, self.tablebase_pieces)
            if problem is None:
                self.syzygy_path = path or None
            return problem
        lowest, highest = SPIN_OPTIONS[name]
        try:
            number = int(value)
        except ValueError:
            number = None
        if number is None or not lowest <= number <= highest:
            return f"{name} is a whole number from {lowest} to {highest}, not {value!r}"
        if name == "SyzygyProbeLimit":
            problem = self.tables_problem(self.syzygy_path, number)
            if problem is None:
                self.tablebase_pieces = number
            return problem
        if name == "MoveOverhead":
            self.move_overhead_seconds = number / 1000
        elif name == "CachePositions":
            self.cache_positions = number
        elif name == "RamLimitMb":
            self.ram_limit_mb = number
        else:
            self.advance_threads = number
        return None

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

    #load_player's tablebase and the search options these set
    def search(self) -> tuple[bool, dict]:
        return (self.tablebase or self.syzygy_path is not None,
                {"tablebase_pieces": self.tablebase_pieces, "syzygy_path": self.syzygy_path, "cache_positions": self.cache_positions,
                 "memory_limit_mb": self.ram_limit_mb, "advance_threads": self.advance_threads})


def main(checkpoint_path: Path | None = None, random_mover: bool = False, selection: str = "argmax",
         temperature: float = TEMPERATURE, sample_plies: int = SAMPLE_PLIES, simulations: int = SIMULATIONS,
         gpu: bool = False, leaves: int = LEAVES, tablebase: bool = False, search_log: Path | None = None, search_options: dict | None = None,
         precision: str = ENGINE_PRECISION, fp16_blocks: int = ENGINE_FP16_BLOCKS, clock_options: dict | None = None,
         infinite_until_stop: bool = False, gpus: list[int] | None = None):
    torch.set_num_threads(1)
    search_options = search_options or {}
    clock_options = clock_options or {}
    #the options are the search's
    searching = selection == "mcts" and not random_mover
    options = Options(tablebase, search_options.get("tablebase_pieces", TABLEBASE_PIECES))
    player = None
    board = chess.Board()
    clock = Clock(**clock_options)
    commands = Commands(sys.stdin, infinite_until_stop)

    def load() -> nn.Module | EngineSearch:
        use_tablebase, set_options = options.search()
        return load_player(checkpoint_path, selection, gpu, leaves, use_tablebase, {**search_options, **set_options}, precision, fp16_blocks, gpus)

    #go's move, None in a game that is over. A go until stop returns once told to stop, also when its search ended by
    #itself (a proven or only move), as UCI asks; a go ponder searches the opponent's time until ponderhit, then on as
    #the go asks (the tree kept) with its time from the ponderhit, or until stop, for a move the gui throws away
    def move_for(go: Go) -> chess.Move | None:
        nonlocal player
        #timed from when the go came, as the gui times it
        started = go.received
        if not any(board.legal_moves):
            send("info string the game is over, there is no move to play")
            return None
        if player is None and not random_mover:
            player = load()
        if go.ponder:
            pondered = player.search(board, stop=go.pondered) if searching else None
            go.pondered.wait()
            if go.ponderhit is None:
                return pondered.line[0] if pondered else next(iter(board.legal_moves))
            started = go.ponderhit
        if random_mover:
            move = choose_random_move(board)
        elif searching:
            for number in go_numbers(go.arguments)[1]:
                send(f"info string go's {number} is not a whole number: left out")
            nodes, seconds = clock.limits(go.arguments, board, simulations)
            if go.until_stop:
                nodes = seconds = None
            running = lambda report: send(info_line(report, time.perf_counter() - started))
            report = player.search(board, nodes, seconds, clock.extension, go.stop, running)
            clock.searched(time.perf_counter() - started, score_to_centipawns(report.score))
            move = report.line[0]
            send(info_line(report, time.perf_counter() - started))
            if search_log is not None:
                log_search(search_log, board, move, report)
        else:
            move = choose_move(player, board, DEVICE, selection, temperature, sample_plies)
        if go.until_stop:
            go.stop.wait()
        return move

    while (received := commands.next()) is not None:
        command, arguments, go = received

        if command == "uci":
            send(f"id name {'RandomMover' if random_mover else ENGINE_NAME}")
            send("id author William Gong")
            if searching:
                for line in options.lines():
                    send(line)
            send("uciok")

        elif command == "isready":
            #loaded here, not at startup, so "uci" is answered at once
            if player is None and not random_mover:
                player = load()
            send("readyok")

        elif command == "setoption" and searching:
            told = options.set(*option_name_and_value(arguments), built=player is not None)
            clock.move_overhead_seconds = options.move_overhead_seconds
            if told is not None:
                send(f"info string {told}")

        elif command == "ucinewgame":
            board = chess.Board()
            clock = Clock(**clock_options, move_overhead_seconds=options.move_overhead_seconds)
            if isinstance(player, EngineSearch):
                player.new_game()

        elif command == "position":
            try:
                board = parse_position(arguments)
            except ValueError as error:
                send(f"info string {error}: the last position stays")

        elif command == "go":
            move = move_for(go)
            send(f"bestmove {move.uci() if move is not None else '0000'}")
            #not after a ponder miss, whose move is not played
            if move is not None and not (go.ponder and go.ponderhit is None):
                advance_after(player, board, move)
            commands.finished(go)

        elif command == "quit":
            break

        #anything else (debug, register, ...) is ignored, as UCI allows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="the model as a UCI engine")
    parser.add_argument("checkpoint", nargs="?", help="a model.pt, or 'random' for random moves; default the run that finished last")
    parser.add_argument("--select", choices=SELECTIONS, default="argmax", help="how a move is picked, see player.py")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="for --select sample, default %(default)s")
    parser.add_argument("--sample-plies", type=int, default=SAMPLE_PLIES, help="for --select sample: argmax after this many plies, default the whole game")
    parser.add_argument("--simulations", type=int, default=SIMULATIONS,
                        help="for --select mcts: the simulations of a go without movetime, nodes or wtime/btime, default %(default)s")
    parser.add_argument("--gpu", action="store_true", help="run the model on the gpu: faster for one search, slower when many games share it")
    parser.add_argument("--leaves", type=int, default=LEAVES,
                        help="for --select mcts --gpu: leaves gathered for each network call, default %(default)s; 1 is mcts.py's search")
    parser.add_argument("--precision", choices=PRECISIONS,
                        help=f"for --select mcts --gpu: the network's precision (engine_search.PRECISIONS), default {ENGINE_PRECISION}")
    parser.add_argument("--gpus", type=lambda text: [int(device) for device in text.split(",")],
                        help="for --select mcts --gpu: the cuda devices to search on at once, e.g. 0,1,2,3, a thread driving each; "
                             "default the current one. On more than one, for --leaves above 1, the search is another (more leaves out): the "
                             "threads descend in the one tree at once, each pinned to a cpu by the topology and asleep while its gpu works")
    parser.add_argument("--threads-per-gpu", type=int,
                        help=f"for --gpus: the networks on each, a thread driving each, default {THREADS_PER_GPU} on more than one gpu "
                             "and 1 on one; a gpu with more than one is searched as several gpus are")
    parser.add_argument("--fp16-blocks", type=int,
                        help=f"for the int8 network: the blocks kept in fp16 besides the last, those whose int8 costs the policy most "
                             f"(engine/quantization.py), default {ENGINE_FP16_BLOCKS}; 0 for every other block in int8")
    parser.add_argument("--tablebase", action="store_true", help="for --select mcts: positions of at most --tablebase-pieces pieces from the syzygy tables")
    parser.add_argument("--tablebase-pieces", type=int, choices=[table_set.pieces for table_set in TABLE_SETS], default=TABLEBASE_PIECES,
                        help="for --tablebase: the most pieces, default %(default)s; 6 needs python -m tools.tablebase --pieces 6, and without "
                             "the 6 piece dtz tables searches those positions rather than playing the tables' move")
    parser.add_argument("--search-log", type=Path, help="for --select mcts: a folder to add every search to, as <process id>.jsonl")
    parser.add_argument("--grow-leaves", type=int, default=GROW_LEAVES,
                        help="for --select mcts --gpu: about one leaf a round per this many simulations done or left, up to --leaves; 0 for --leaves every round")
    parser.add_argument("--visits-per-leaf-out", type=int, default=VISITS_PER_LEAF_OUT,
                        help="for --select mcts --gpu: at most a leaf out on the network (the rounds out on all of --gpus) per this many "
                             "root visits; 0 for no cap, the default")
    parser.add_argument("--no-smart-pruning", dest="smart_pruning", action="store_false",
                        help="for --select mcts --gpu: search all of go's movetime or nodes even once the move is decided (lc0's SmartPruningFactor=0)")
    parser.add_argument("--smart-pruning-factor", type=float, default=SMART_PRUNING_FACTOR,
                        help="for --select mcts --gpu: smart pruning stops once the most visited move leads by more than the simulations left / "
                             "this, default %(default)s; lc0's 1.33 bets that the second move cannot get all of them")
    parser.add_argument("--smart-pruning-wait", type=float, default=SMART_PRUNING_WAIT,
                        help="for --select mcts --gpu: no smart pruning stop on the visits until this many seconds after a search's first answers, "
                             "default %(default)s; lc0's 0.2")
    parser.add_argument("--stop-at-one-move-left", action="store_true",
                        help="for --select mcts --gpu: stop at once when certainty's proofs leave one root move not lost, as lc0 does")
    parser.add_argument("--first-move-bonus", type=float, default=FIRST_MOVE_BONUS,
                        help="for --select mcts on a clock: the game's first search gets 1 + this + --book-ply-bonus x its ply (at most "
                             f"{MOST_BOOK_PLIES}) times its share, default %(default)s; lc0's 1.8")
    parser.add_argument("--book-ply-bonus", type=float, default=BOOK_PLY_BONUS, help="for --first-move-bonus, default %(default)s; lc0's 0.25")
    parser.add_argument("--search-settings", default="",
                        help="for --select mcts: changes to the search's settings (engine_search.py), e.g. c_puct=0.6")
    parser.add_argument("--infinite-until-stop", action="store_true",
                        help="go infinite searches until stop, as UCI and analysis guis ask; without it it runs --simulations, which "
                             "cutechess-cli's tc=inf games need")
    arguments = parser.parse_args()
    if arguments.tablebase_pieces != TABLEBASE_PIECES and not arguments.tablebase:
        parser.error("--tablebase-pieces needs --tablebase")
    if arguments.precision is not None and not (arguments.select == "mcts" and arguments.gpu):
        parser.error("--precision needs --select mcts --gpu")
    if arguments.smart_pruning_factor <= 0:
        parser.error("--smart-pruning-factor is positive; --no-smart-pruning for none")
    if arguments.threads_per_gpu is not None and (arguments.gpus is None or arguments.threads_per_gpu < 1):
        parser.error("--threads-per-gpu needs --gpus, and is 1 or more")
    #a device once for each of its networks
    network_gpus = None if arguments.gpus is None else [gpu for gpu in arguments.gpus for _ in range(
        arguments.threads_per_gpu or (THREADS_PER_GPU if len(arguments.gpus) > 1 else 1))]
    if network_gpus is not None and not (arguments.select == "mcts" and arguments.gpu and (len(network_gpus) == 1 or arguments.leaves > 1)):
        parser.error("--gpus needs --select mcts --gpu, and on more than one network --leaves above 1")
    if arguments.fp16_blocks is not None and not (arguments.select == "mcts" and arguments.gpu and arguments.precision in (None, "int8")):
        parser.error("--fp16-blocks needs --select mcts --gpu and the int8 network")
    if arguments.fp16_blocks is not None and arguments.fp16_blocks < 0:
        parser.error("--fp16-blocks is 0 or more")
    if arguments.checkpoint == "random":
        main(random_mover=True, infinite_until_stop=arguments.infinite_until_stop)
    else:
        main(Path(arguments.checkpoint) if arguments.checkpoint else latest_checkpoint(),
             selection=arguments.select, temperature=arguments.temperature, sample_plies=arguments.sample_plies,
             simulations=arguments.simulations, gpu=arguments.gpu, leaves=arguments.leaves, tablebase=arguments.tablebase,
             search_log=arguments.search_log, precision=arguments.precision or ENGINE_PRECISION, infinite_until_stop=arguments.infinite_until_stop,
             gpus=network_gpus, fp16_blocks=ENGINE_FP16_BLOCKS if arguments.fp16_blocks is None else arguments.fp16_blocks,
             search_options={"smart_pruning": arguments.smart_pruning, "tablebase_pieces": arguments.tablebase_pieces, "grow_leaves": arguments.grow_leaves,
                             "settings": parse_changes(arguments.search_settings, engine_search_settings(arguments.tablebase)),
                             "smart_pruning_factor": arguments.smart_pruning_factor, "smart_pruning_wait": arguments.smart_pruning_wait,
                             "stop_at_one_move_left": arguments.stop_at_one_move_left, "visits_per_leaf_out": arguments.visits_per_leaf_out},
             clock_options={"first_move_bonus": arguments.first_move_bonus, "book_ply_bonus": arguments.book_ply_bonus})

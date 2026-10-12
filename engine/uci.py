#the model as a uci engine, for cutechess-cli matches and the cute chess gui. Only the commands a gui needs to play:
#uci, isready, ucinewgame, position, go and quit. The search (--select mcts) keeps to go's movetime, nodes and
#wtime/btime; a go without them (go infinite, as cutechess sends for tc=inf) runs --simulations. The other selections
#ignore the clock
#   python -m engine.uci [model.pt | random] [--select mcts --gpu]
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
import chess
import torch
from torch import nn
from engine.inference import cpu_batches
from tools.checkpoint import latest_checkpoint
from engine.engine_search import ENGINE_PRECISION, LEAVES, PRECISIONS, EngineSearch, engine_network, engine_search_settings, parse_changes
from engine.player import (SAMPLE_PLIES, SELECTIONS, SIMULATIONS, TEMPERATURE, CudaGraphModel, SearchReport, choose_move,
                               load_model, score_to_centipawns)
from tools.tablebase import TABLE_SETS, TABLEBASE_PIECES

ENGINE_NAME = "ChessTransformer"

#one board takes a few ms on the cpu, less than starting cuda; one thread so parallel games don't fight over cores
DEVICE = torch.device("cpu")
#the time between the search's end and the gui reading bestmove, kept off go's time
MOVE_OVERHEAD_SECONDS = 0.02
#go's arguments that take a number
GO_NUMBERS = ("wtime", "btime", "winc", "binc", "movestogo", "movetime", "nodes", "depth", "mate")
#with wtime/btime and no movestogo, the time left is planned to last to about this move, and later moves live on the
#increment: lc0's smooth time manager has its clock nearly spent by move ~55 at 60+0.6, while an even share over 30
#more moves kept ~13 s unspent there (artifacts/bt4_tournament_match_20260928)
PLANNED_MOVES = 50
#the fewest moves the time left is shared over, so the last of it goes gradually
LEAST_MOVES_LEFT = 10
#kept back from the time left, for a search that runs over
RESERVE_SECONDS = 1.0
#a move gets at most this part of the time left
MOST_OF_THE_TIME_LEFT = 1 / 5
#Clock's time use: how much the last search moves it, and the least it can be
TIME_USE_WEIGHT = 0.2
LEAST_TIME_USE = 0.3

def send(line: str) -> None:
    #flushed: the gui reads a pipe, not a terminal
    print(line, flush=True)

#"startpos moves e2e4 e7e5" or "fen <6 fen fields> moves ..."
def parse_position(arguments: list[str]) -> chess.Board:
    if "moves" in arguments:
        moves_index = arguments.index("moves")
        setup, moves = arguments[:moves_index], arguments[moves_index + 1:]
    else:
        setup, moves = arguments, []

    if setup[0] == "startpos":
        board = chess.Board()
    elif setup[0] == "fen":
        board = chess.Board(" ".join(setup[1:]))
    else:
        raise ValueError(f"unknown position: {' '.join(arguments)}")

    for uci_move in moves:
        board.push_uci(uci_move)

    return board

def choose_random_move(board: chess.Board) -> chess.Move:
    return random.choice(list(board.legal_moves))

#depth is the length of the expected line; cutechess keeps score, depth and time in the pgn
def info_line(report: SearchReport, seconds: float) -> str:
    return (f"info depth {len(report.line)} nodes {report.simulations} time {round(seconds * 1000)} nps {round(report.simulations / max(seconds, 1e-3))} "
            f"score cp {score_to_centipawns(report.score)} pv {' '.join(move.uci() for move in report.line)}")

#go's limits for the search, and on a clock (wtime/btime) the time a move gets: an even share of the time left (less a
#reserve) over the moves still to play to about move PLANNED_MOVES, plus the increment, divided by the part of their
#time the last searches used. A search smart pruning stops early leaves time on the clock, and so it goes to the moves
#that need it (lc0's smooth time manager does the same). The arguments are the constants above of the same names, for
#tools/play/variant_match.py to tune
class Clock:
    def __init__(self, planned_moves: int = PLANNED_MOVES, least_moves_left: int = LEAST_MOVES_LEFT, reserve_seconds: float = RESERVE_SECONDS,
                 most_of_the_time_left: float = MOST_OF_THE_TIME_LEFT):
        self.planned_moves, self.least_moves_left = planned_moves, least_moves_left
        self.reserve_seconds, self.most_of_the_time_left = reserve_seconds, most_of_the_time_left
        self.time_use = 1.0 #the searches' seconds over the seconds they were given, lately
        self.allotted: float | None = None #the seconds the search on the clock was given

    #(simulations, seconds) for a search of `board`, each None when go does not set it; a go with neither runs `simulations`
    def limits(self, arguments: list[str], board: chess.Board, simulations: int) -> tuple[int | None, float | None]:
        numbers = {key: int(value) for key, value in zip(arguments, arguments[1:]) if key in GO_NUMBERS}
        time_left, increment = ("wtime", "winc") if board.turn == chess.WHITE else ("btime", "binc")
        seconds = self.allotted = None
        if "movetime" in numbers:
            seconds = max(0.0, numbers["movetime"] / 1000 - MOVE_OVERHEAD_SECONDS)
        elif time_left in numbers:
            seconds_left = numbers[time_left] / 1000 - MOVE_OVERHEAD_SECONDS
            seconds = self.allotted = self.seconds_for_move(seconds_left, numbers.get(increment, 0) / 1000, numbers.get("movestogo"), board.fullmove_number)
        nodes = numbers.get("nodes")
        if nodes is None and seconds is None:
            nodes = simulations
        return nodes, seconds

    def seconds_for_move(self, time_left: float, increment: float, moves_to_go: int | None, move_number: int) -> float:
        moves_left = moves_to_go or max(self.least_moves_left, self.planned_moves - move_number)
        share = max(0.0, time_left - self.reserve_seconds) / moves_left + increment
        return max(0.0, min(share / self.time_use, time_left * self.most_of_the_time_left))

    #after a search, the seconds it took
    def searched(self, seconds: float) -> None:
        if self.allotted:
            used = min(1.0, seconds / self.allotted)
            self.time_use = max(LEAST_TIME_USE, (1 - TIME_USE_WEIGHT) * self.time_use + TIME_USE_WEIGHT * used)

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

#the search for --select mcts: on the gpu `leaves` leaves a round, the network in `precision`; on the cpu one, which is
#mcts.py's search. options are EngineSearch's keyword arguments (smart pruning, the tables' pieces, changes to the
#search's settings)
def load_engine_search(checkpoint_path: Path, gpu: bool, leaves: int, tablebase: bool, options: dict, precision: str) -> EngineSearch:
    if gpu:
        return EngineSearch(engine_network(checkpoint_path, leaves, precision), leaves, use_tablebase=tablebase, **options)
    return EngineSearch(cpu_batches(load_model(checkpoint_path, DEVICE)), 1, use_tablebase=tablebase, **options)

#what picks the moves: the search for --select mcts, else the model
def load_player(checkpoint_path: Path, selection: str, gpu: bool, leaves: int, tablebase: bool, search_options: dict,
                precision: str) -> nn.Module | EngineSearch:
    if selection == "mcts":
        return load_engine_search(checkpoint_path, gpu, leaves, tablebase, search_options, precision)
    return load_engine_model(checkpoint_path, gpu)

def main(checkpoint_path: Path | None = None, random_mover: bool = False, selection: str = "argmax",
         temperature: float = TEMPERATURE, sample_plies: int = SAMPLE_PLIES, simulations: int = SIMULATIONS,
         gpu: bool = False, leaves: int = LEAVES, tablebase: bool = False, search_log: Path | None = None, search_options: dict | None = None,
         precision: str = ENGINE_PRECISION):
    torch.set_num_threads(1)
    search_options = search_options or {}
    player = None
    board = chess.Board()
    clock = Clock()

    for line in sys.stdin:
        words = line.split()
        if len(words) == 0:
            continue

        command, arguments = words[0], words[1:]

        if command == "uci":
            send(f"id name {'RandomMover' if random_mover else ENGINE_NAME}")
            send("id author William Gong")
            send("uciok")

        elif command == "isready":
            #loaded here, not at startup, so "uci" is answered at once
            if player is None and not random_mover:
                player = load_player(checkpoint_path, selection, gpu, leaves, tablebase, search_options, precision)
            send("readyok")

        elif command == "ucinewgame":
            board = chess.Board()
            clock = Clock()

        elif command == "position":
            board = parse_position(arguments)

        elif command == "go":
            #timed from the go command, as the gui times it
            started = time.perf_counter()
            if random_mover:
                move = choose_random_move(board)
            else:
                if player is None:
                    player = load_player(checkpoint_path, selection, gpu, leaves, tablebase, search_options, precision)
                if selection == "mcts":
                    nodes, seconds = clock.limits(arguments, board, simulations)
                    report = player.search(board, nodes, seconds)
                    clock.searched(time.perf_counter() - started)
                    move = report.line[0]
                    send(info_line(report, time.perf_counter() - started))
                    if search_log is not None:
                        log_search(search_log, board, move, report)
                else:
                    move = choose_move(player, board, DEVICE, selection, temperature, sample_plies)
            send(f"bestmove {move.uci()}")

        elif command == "quit":
            break

        #anything else (setoption, stop, ...) is ignored, as UCI allows


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
    parser.add_argument("--tablebase", action="store_true", help="for --select mcts: positions of at most --tablebase-pieces pieces from the syzygy tables")
    parser.add_argument("--tablebase-pieces", type=int, choices=[table_set.pieces for table_set in TABLE_SETS], default=TABLEBASE_PIECES,
                        help="for --tablebase: the most pieces, default %(default)s; 6 needs python -m tools.tablebase --pieces 6, and without "
                             "the 6 piece dtz tables searches those positions rather than playing the tables' move")
    parser.add_argument("--search-log", type=Path, help="for --select mcts: a folder to add every search to, as <process id>.jsonl")
    parser.add_argument("--no-smart-pruning", dest="smart_pruning", action="store_false",
                        help="for --select mcts --gpu: search all of go's movetime or nodes even once the move is decided (lc0's SmartPruningFactor=0)")
    parser.add_argument("--search-settings", default="",
                        help="for --select mcts: changes to the search's settings (engine_search.py), e.g. c_puct=0.6")
    arguments = parser.parse_args()
    if arguments.tablebase_pieces != TABLEBASE_PIECES and not arguments.tablebase:
        parser.error("--tablebase-pieces needs --tablebase")
    if arguments.precision is not None and not (arguments.select == "mcts" and arguments.gpu):
        parser.error("--precision needs --select mcts --gpu")
    if arguments.checkpoint == "random":
        main(random_mover=True)
    else:
        main(Path(arguments.checkpoint) if arguments.checkpoint else latest_checkpoint(),
             selection=arguments.select, temperature=arguments.temperature, sample_plies=arguments.sample_plies,
             simulations=arguments.simulations, gpu=arguments.gpu, leaves=arguments.leaves, tablebase=arguments.tablebase,
             search_log=arguments.search_log, precision=arguments.precision or ENGINE_PRECISION,
             search_options={"smart_pruning": arguments.smart_pruning, "tablebase_pieces": arguments.tablebase_pieces,
                             "settings": parse_changes(arguments.search_settings, engine_search_settings(arguments.tablebase))})

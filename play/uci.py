#lets any chess program talk to the model: cutechess-cli for matches against stockfish, the
#cute chess gui for playing it yourself. Those programs start this file as a subprocess and
#send it UCI commands as lines of text on stdin, the replies go back as lines on stdout.
#
#only the part of UCI that a gui needs to play a game is here:
#   uci          -> say who we are, then uciok
#   isready      -> readyok, once the model is loaded
#   position ... -> the position to think about, as a start position plus the moves played so far
#   go ...       -> reply with bestmove. there is no search, so every time setting is ignored
#   quit
from pathlib import Path
import random
import sys
import chess
import torch
from main import CHECKPOINT_PATH
from play.player import choose_move, load_model

ENGINE_NAME = "ChessTransformer"

#one board on the cpu takes a few milliseconds, and starting cuda would take longer than a whole game.
#a single thread so several games can run side by side without fighting over the cores
DEVICE = torch.device("cpu")

def send(line: str) -> None:
    #flush every line: the gui is waiting on the other end of a pipe, not reading a terminal
    print(line, flush=True)

#"position startpos moves e2e4 e7e5" or "position fen <6 fen fields> moves ..."
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

#the bottom rung of the ladder in evaluate_elo.py: a trained model that cannot beat this is broken
def choose_random_move(board: chess.Board) -> chess.Move:
    return random.choice(list(board.legal_moves))

#plays random moves instead of loading a model when random_mover is set
def main(checkpoint_path: Path = CHECKPOINT_PATH, random_mover: bool = False):
    torch.set_num_threads(1)

    model = None
    board = chess.Board()

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
            #loaded here rather than at startup so "uci" is answered straight away
            if model is None and not random_mover:
                model = load_model(checkpoint_path, DEVICE)
            send("readyok")

        elif command == "ucinewgame":
            board = chess.Board()

        elif command == "position":
            board = parse_position(arguments)

        elif command == "go":
            if random_mover:
                move = choose_random_move(board)
            else:
                if model is None:
                    model = load_model(checkpoint_path, DEVICE)
                move = choose_move(model, board, DEVICE)
            send(f"bestmove {move.uci()}")

        elif command == "quit":
            break

        #anything else (setoption, stop, ponderhit, ...) has nothing to act on here, UCI says to ignore it


if __name__ == "__main__":
    #"python -m play.uci" plays model.pt, "python -m play.uci other.pt" plays another checkpoint
    #(so two checkpoints can play each other), "python -m play.uci random" plays random moves
    if len(sys.argv) > 1 and sys.argv[1] == "random":
        main(random_mover=True)
    else:
        main(Path(sys.argv[1]) if len(sys.argv) > 1 else CHECKPOINT_PATH)

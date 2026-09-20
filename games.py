#pgn files to training examples. Kept free of torch on purpose: preprocess.py runs this in
#about 30 processes at once, and importing torch costs each of them half a gigabyte of memory
from pathlib import Path
from dataclasses import dataclass
from encoding import encode_board, encode_move_to_id
import chess.pgn

PGN_DATA_DIR = Path(__file__).parent / "pgn_data"
CARLSEN_PGN = PGN_DATA_DIR / "Carlsen.pgn"
#one pgn per month, 2013-01 to 2025-11. 27 GB, so not in git: see the readme for where to get them
LICHESS_ELITE_PGNS = sorted((PGN_DATA_DIR / "lichess_elite").glob("*.pgn"))

@dataclass
class TrainingExample:
    board: list[int]
    move: int

Game = list[TrainingExample] #one example per position, in the order the game was played

def game_to_examples(game: chess.pgn.Game) -> list[TrainingExample]:
    board = game.board()
    complete_game = []

    for move in game.mainline_moves():
        complete_game.append(TrainingExample(board=encode_board(board), move=encode_move_to_id(move)))
        board.push(move)

    return complete_game

#takes a chess pgn and converts it to training examples. Examples stay grouped by game
#so that split_data can keep every position of a game on the same side of the split.
def pgn_to_examples(pgn_path: Path) -> list[Game]:
    output = []
    with open(pgn_path) as pgn:
        game = chess.pgn.read_game(pgn)
        while game is not None:
            output.append(game_to_examples(game))
            game = chess.pgn.read_game(pgn)

    return output

#a list of games from possibly more than one PGN, good for extensibility
def pgns_to_examples(pgn_paths: list[Path]) -> list[Game]:
    games = []
    for path in pgn_paths:
        games.extend(pgn_to_examples(path))

    return games

from collections.abc import Iterator
from pathlib import Path
from dataclasses import dataclass
from encoding import decode_board, encode_board, encode_move_to_id

import chess.pgn

CARLSEN_PGN = Path(__file__).parent / "Carlsen" / "Carlsen.pgn"

@dataclass
class TrainingExample:
    board: list[int]
    move: int

def game_to_examples(game: chess.pgn.Game) -> list[TrainingExample]:
    board = game.board()
    complete_game = []

    for move in game.mainline_moves():
        complete_game.append(TrainingExample(board=encode_board(board), move=encode_move_to_id(move)))
        board.push(move)

    return complete_game

def pgn_to_examples(pgn_path: Path) -> list[list[TrainingExample]]:
    output = []
    with open(pgn_path) as pgn:
        game = chess.pgn.read_game(pgn)
        while game is not None:
            output.append(game_to_examples(game))
            game = chess.pgn.read_game(pgn)

    return output

def read_games(pgn_path: Path) -> Iterator[chess.pgn.Game]:
    with open(pgn_path) as pgn:
        #read_game returns None once the file is exhausted
        while (game := chess.pgn.read_game(pgn)) is not None:
            yield game

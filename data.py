from pathlib import Path
from dataclasses import dataclass
from encoding import encode_board, encode_move_to_id
from torch.utils.data import Dataset
import torch
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

#takes a chess pgn and converts it to a set of training examples
def pgn_to_examples(pgn_path: Path) -> list[list[TrainingExample]]:
    output = []
    with open(pgn_path) as pgn:
        game = chess.pgn.read_game(pgn)
        while game is not None:
            output.append(game_to_examples(game))
            game = chess.pgn.read_game(pgn)

    return output

class ChessGamesDataset(Dataset):
    """set of examples of chess games that we can use as a dataset"""

    def __init__(self, games: list[list[TrainingExample]]):
        #two tensors, one for the board positions and the other for the moves
        boards = []
        moves = []

        for game in games:
            for example in game:
                boards.append(example.board)
                moves.append(example.move)

        self.boards = torch.tensor(boards, dtype=torch.uint8)
        self.moves = torch.tensor(moves, dtype=torch.int16)

    def __len__(self):
        return len(self.moves)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.boards[index].long(), self.moves[index].long()
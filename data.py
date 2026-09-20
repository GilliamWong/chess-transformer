from pathlib import Path
from dataclasses import dataclass
from encoding import encode_board, encode_move_to_id
from torch.utils.data import DataLoader, Dataset
import torch
import chess.pgn
import random

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

#splits by game rather than by position: consecutive positions of one game are nearly identical,
#so a game on both sides of the split would inflate validation accuracy.
def split_data(games: list[Game], train_fraction: float = 0.8, seed: int = 42) -> tuple[list[Game], list[Game]]:
    if not 0 <= train_fraction <= 1:
        #a percentage like 80 would otherwise silently put every game in train
        raise ValueError(f"train_fraction must be between 0 and 1, got {train_fraction}")

    shuffled_games = games.copy() #copy so we don't modify the input games list
    random.Random(seed).shuffle(shuffled_games) #shuffles in place and returns None

    cut = int(len(shuffled_games) * train_fraction)

    train = shuffled_games[:cut]
    val = shuffled_games[cut:]

    return train, val


class ChessGamesDataset(Dataset):
    """set of examples of chess games that we can use as a dataset"""

    def __init__(self, games: list[Game]):
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


#takes games that are already parsed rather than pgn paths, so the loaders can be built
#(and tested) without reading a whole pgn. Use pgns_to_examples to get the games.
def create_dataloaders(
    games: list[Game],
    train_batch_size: int = 256,
    val_batch_size: int = 512,
    train_fraction: float = 0.8,
    seed: int = 42
) -> tuple[DataLoader, DataLoader]:

    train_data, val_data = split_data(games, train_fraction=train_fraction, seed=seed)

    train_dataset = ChessGamesDataset(train_data)
    val_dataset = ChessGamesDataset(val_data)

    #train is reshuffled every epoch so a batch is not 256 positions of the same few games.
    #val is only measured, so its order does not matter
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    return train_loader, val_loader

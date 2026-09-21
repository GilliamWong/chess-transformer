from pathlib import Path
from dataclasses import dataclass
from encoding import TOKENS_PER_BOARD, encode_board, encode_move_to_id
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler
import numpy as np
import torch
import chess.pgn
import random

#pgns, preprocessed training sets, checkpoints, tensorboard runs and the chess engines: no code lives here
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

PGN_DATA_DIR = ARTIFACTS_DIR / "pgn_data"
CARLSEN_PGN = PGN_DATA_DIR / "Carlsen.pgn"

@dataclass
class TrainingExample:
    board: list[int]
    move: int
    result: int #how the game ended for the side to move: 1 won, 0 draw, -1 lost

#the pgn's Result header, which is from white's side
WHITE_RESULTS = {"1-0": 1, "1/2-1/2": 0, "0-1": -1}

Game = list[TrainingExample] #one example per position, in the order the game was played

def game_to_examples(game: chess.pgn.Game) -> list[TrainingExample]:
    board = game.board()
    complete_game = []
    white_result = WHITE_RESULTS[game.headers["Result"]]

    for move in game.mainline_moves():
        result = white_result if board.turn == chess.WHITE else -white_result
        complete_game.append(TrainingExample(board=encode_board(board), move=encode_move_to_id(move, board.turn), result=result))
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

#training sets too big to parse from pgn on every run, already encoded into arrays on disk.
#see processed_data/scripts_used/ for how they were made
PROCESSED_DATA_DIR = ARTIFACTS_DIR / "processed_data"

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
        #three tensors: the board positions, the moves and how the games ended
        boards = []
        moves = []
        results = []

        for game in games:
            for example in game:
                boards.append(example.board)
                moves.append(example.move)
                results.append(example.result)

        self.boards = torch.tensor(boards, dtype=torch.uint8)
        self.moves = torch.tensor(moves, dtype=torch.int16)
        self.results = torch.tensor(results, dtype=torch.int8)

    #for positions that are already arrays (load_training_set). They can be memory mapped and far too
    #many for a python list of games: the tensors share the arrays' memory, nothing is copied
    @classmethod
    def from_arrays(cls, boards: np.ndarray, moves: np.ndarray, results: np.ndarray) -> "ChessGamesDataset":
        if not len(boards) == len(moves) == len(results):
            raise ValueError(f"{len(boards)} boards but {len(moves)} moves and {len(results)} results")

        dataset = cls([])
        dataset.boards = torch.from_numpy(boards)
        dataset.moves = torch.from_numpy(moves)
        dataset.results = torch.from_numpy(results)

        return dataset

    def __len__(self):
        return len(self.moves)

    #index is one position, or a list of positions that come back as one batch (see datasets_to_dataloaders)
    #the result comes back as the class the value head predicts: 0 lost, 1 draw, 2 won
    def __getitem__(self, index: int | list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.boards[index].long(), self.moves[index].long(), self.results[index].long() + 1


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

    return datasets_to_dataloaders(train_dataset, val_dataset, train_batch_size, val_batch_size)

#the samplers hand the dataset a whole batch of indices at once (batch_size=None turns the loader's own
#batching off), so a batch is one tensor lookup instead of one __getitem__ call per position plus a collate.
#measured on 580k positions: 0.2s per epoch instead of 3.0s, which starts to matter at 70 million
def datasets_to_dataloaders(
    train_dataset: ChessGamesDataset,
    val_dataset: ChessGamesDataset,
    train_batch_size: int = 256,
    val_batch_size: int = 512
) -> tuple[DataLoader, DataLoader]:

    #train is reshuffled every epoch so a batch is not 256 positions of the same few games.
    #val is only measured, so its order does not matter
    train_sampler = BatchSampler(RandomSampler(train_dataset), batch_size=train_batch_size, drop_last=False)
    val_sampler = BatchSampler(SequentialSampler(val_dataset), batch_size=val_batch_size, drop_last=False)

    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=None)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=None)

    return train_loader, val_loader

#opens <name>_{train,val}_{boards,moves,results}.bin from processed_data/ as datasets. The arrays are memory mapped,
#so 71 million positions cost no memory up front: pages are read from disk as batches touch them, and the
#first epoch is a little slower than the rest while the operating system caches the file
def load_training_set(name: str) -> tuple[ChessGamesDataset, ChessGamesDataset]:
    datasets = []
    for split in ("train", "val"):
        boards_path = PROCESSED_DATA_DIR / f"{name}_{split}_boards.bin"
        moves_path = PROCESSED_DATA_DIR / f"{name}_{split}_moves.bin"
        results_path = PROCESSED_DATA_DIR / f"{name}_{split}_results.bin"
        if not boards_path.exists() or not moves_path.exists():
            raise FileNotFoundError(f"{boards_path.name} or {moves_path.name} is missing from {PROCESSED_DATA_DIR}")
        if not results_path.exists():
            raise FileNotFoundError(f"{results_path.name} is missing, scripts_used/add_results.py writes it")

        #the files are raw arrays without a header, in the dtypes ChessGamesDataset stores: uint8 boards, int16 moves, int8 results.
        #mode "c" is copy on write: nothing can change the files, but torch.from_numpy gets the writable array it asks for
        boards = np.memmap(boards_path, dtype=np.uint8, mode="c").reshape(-1, TOKENS_PER_BOARD)
        moves = np.memmap(moves_path, dtype=np.int16, mode="c")
        results = np.memmap(results_path, dtype=np.int8, mode="c")
        datasets.append(ChessGamesDataset.from_arrays(boards, moves, results))

    return datasets[0], datasets[1]

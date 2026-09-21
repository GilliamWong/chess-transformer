from pathlib import Path
from dataclasses import dataclass
from encoding import NUM_PIECE_TYPES, NUM_SQUARES, TOKENS_PER_BOARD, encode_board, encode_move_to_id
from collections.abc import Iterator
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, Sampler, SequentialSampler
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


#a piece code seen from the other side: empty stays empty, the mover's pieces 1-6 and the opponent's 7-12 trade places
SWAPPED_PIECES = torch.tensor([0] + list(range(NUM_PIECE_TYPES + 1, 2 * NUM_PIECE_TYPES + 1)) + list(range(1, NUM_PIECE_TYPES + 1)))

class ChessGamesDataset(Dataset):
    """set of examples of chess games that we can use as a dataset"""

    #history is how many past plies every item describes, see __getitem__
    def __init__(self, games: list[Game], history: int = 0):
        self.history = history
        #three tensors: the board positions, the moves and how the games ended
        boards = []
        moves = []
        results = []
        plies = []

        for game in games:
            for ply, example in enumerate(game):
                boards.append(example.board)
                moves.append(example.move)
                results.append(example.result)
                plies.append(ply)

        self.boards = torch.tensor(boards, dtype=torch.uint8)
        self.moves = torch.tensor(moves, dtype=torch.int16)
        self.results = torch.tensor(results, dtype=torch.int8)
        #how many rows before this one belong to the same game. Games lie one after the other, so that is where
        #the history of a position is found
        self.plies = torch.tensor(plies, dtype=torch.int16)

    #for positions that are already arrays (load_training_set). They can be memory mapped and far too
    #many for a python list of games: the tensors share the arrays' memory, nothing is copied
    @classmethod
    def from_arrays(cls, boards: np.ndarray, moves: np.ndarray, results: np.ndarray, plies: np.ndarray | None = None, history: int = 0) -> "ChessGamesDataset":
        if not len(boards) == len(moves) == len(results):
            raise ValueError(f"{len(boards)} boards but {len(moves)} moves and {len(results)} results")
        if history > 0 and (plies is None or len(plies) != len(boards)):
            raise ValueError("history needs the ply of every position, to know which rows are the same game")

        dataset = cls([], history)
        dataset.boards = torch.from_numpy(boards)
        dataset.moves = torch.from_numpy(moves)
        dataset.results = torch.from_numpy(results)
        if plies is not None:
            dataset.plies = torch.from_numpy(plies)

        return dataset

    def __len__(self):
        return len(self.moves)

    #index is one position, or a list of positions that come back as one batch (see datasets_to_dataloaders)
    #the squares of rows, turned around for the rows where flip is set
    def squares(self, rows: torch.Tensor, flip: bool) -> torch.Tensor:
        squares = self.boards[rows][:, :NUM_SQUARES].long()
        if flip:
            squares = SWAPPED_PIECES[squares.reshape(-1, 8, 8).flip(1).reshape(-1, NUM_SQUARES)]
        return squares

    #what encoding.encode_history gives, for whole batches and from the stored boards: the row k before a position is
    #the position k plies earlier if the position's ply is at least k. Boards are stored from their own side to move,
    #which alternates, so every second one is turned around. (positions, history, NUM_SQUARES)
    def history_of(self, rows: torch.Tensor) -> torch.Tensor:
        changes = torch.zeros((len(rows), self.history, NUM_SQUARES), dtype=torch.long)
        if self.history == 0:
            return changes

        plies = self.plies[rows]
        newer = self.squares(rows, flip=False)
        for back in range(1, self.history + 1):
            known = plies >= back
            older = self.squares((rows - back).clamp(min=0), flip=back % 2 == 1)
            changes[:, back - 1] = torch.where(known.unsqueeze(1) & (older != newer), 1 + older, 0)
            newer = older
        return changes

    #the result comes back as the class the value head predicts: 0 lost, 1 draw, 2 won
    def __getitem__(self, index: int | list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = torch.as_tensor(index)
        history = self.history_of(rows.reshape(-1))
        if rows.ndim == 0:
            history = history[0]
        return self.boards[index].long(), self.moves[index].long(), self.results[index].long() + 1, history


#several training sets behind one index: the positions of the first, then of the second and so on. The sets stay
#separate memory mapped files, joining them into one tensor would read all of them into memory
class MixedDataset(Dataset):
    def __init__(self, datasets: list[ChessGamesDataset]):
        self.datasets = datasets
        self.first_indices = [sum(len(dataset) for dataset in datasets[:number]) for number in range(len(datasets))]

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    #a batch comes back grouped by set rather than in the order of index, which training does not care about
    def __getitem__(self, index: list[int]) -> tuple[torch.Tensor, ...]:
        indices = torch.as_tensor(index)
        parts = []
        for dataset, first_index in zip(self.datasets, self.first_indices):
            in_dataset = indices[(indices >= first_index) & (indices < first_index + len(dataset))] - first_index
            parts.append(dataset[in_dataset.tolist()])

        return tuple(torch.cat(tensors) for tensors in zip(*parts))


#batches for a MixedDataset with the same number of positions from every set, however different their sizes.
#Positions are drawn at random with replacement: over a run that sees a fraction of tens of millions of positions
#that is the same as shuffling, and no set runs out before the others
class EqualSharesBatchSampler(Sampler):
    def __init__(self, dataset: MixedDataset, batch_size: int):
        if batch_size % len(dataset.datasets) != 0:
            raise ValueError(f"a batch of {batch_size} cannot be shared equally between {len(dataset.datasets)} sets")
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        share = self.batch_size // len(self.dataset.datasets)
        for _ in range(len(self)):
            batch = []
            for dataset, first_index in zip(self.dataset.datasets, self.dataset.first_indices):
                batch += (torch.randint(len(dataset), (share,)) + first_index).tolist()
            yield batch


#takes games that are already parsed rather than pgn paths, so the loaders can be built
#(and tested) without reading a whole pgn. Use pgns_to_examples to get the games.
def create_dataloaders(
    games: list[Game],
    train_batch_size: int = 256,
    val_batch_size: int = 512,
    train_fraction: float = 0.8,
    seed: int = 42,
    history: int = 0
) -> tuple[DataLoader, DataLoader]:

    train_data, val_data = split_data(games, train_fraction=train_fraction, seed=seed)

    train_dataset = ChessGamesDataset(train_data, history)
    val_dataset = ChessGamesDataset(val_data, history)

    return datasets_to_dataloaders(train_dataset, val_dataset, train_batch_size, val_batch_size)

#the samplers hand the dataset a whole batch of indices at once (batch_size=None turns the loader's own
#batching off), so a batch is one tensor lookup instead of one __getitem__ call per position plus a collate.
#measured on 580k positions: 0.2s per epoch instead of 3.0s, which starts to matter at 70 million
def datasets_to_dataloaders(
    train_dataset: ChessGamesDataset | MixedDataset,
    val_dataset: ChessGamesDataset,
    train_batch_size: int = 256,
    val_batch_size: int = 512
) -> tuple[DataLoader, DataLoader]:

    #train is reshuffled every epoch so a batch is not 256 positions of the same few games.
    #val is only measured, so its order does not matter
    if isinstance(train_dataset, MixedDataset):
        train_sampler = EqualSharesBatchSampler(train_dataset, train_batch_size)
    else:
        train_sampler = BatchSampler(RandomSampler(train_dataset), batch_size=train_batch_size, drop_last=False)
    val_sampler = BatchSampler(SequentialSampler(val_dataset), batch_size=val_batch_size, drop_last=False)

    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=None)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=None)

    return train_loader, val_loader

#opens <name>_{train,val}_{boards,moves,results}.bin from processed_data/ as datasets. The arrays are memory mapped,
#so 71 million positions cost no memory up front: pages are read from disk as batches touch them, and the
#first epoch is a little slower than the rest while the operating system caches the file
#
#"baseline+fishtest" mixes sets: every train batch takes the same number of positions from each. The val positions
#are those of the first set alone, so the val numbers of a mix can be compared with runs on that set
#
#with history the items also describe that many past plies, which needs <name>_{train,val}_plies.bin
def load_training_set(name: str, history: int = 0) -> tuple[ChessGamesDataset | MixedDataset, ChessGamesDataset]:
    if "+" in name:
        train_datasets, val_datasets = zip(*(load_training_set(one_name, history) for one_name in name.split("+")))
        return MixedDataset(list(train_datasets)), val_datasets[0]

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

        plies = None
        if history > 0:
            plies_path = PROCESSED_DATA_DIR / f"{name}_{split}_plies.bin"
            if not plies_path.exists():
                raise FileNotFoundError(f"{plies_path.name} is missing, scripts_used/add_plies.py writes it")
            plies = np.memmap(plies_path, dtype=np.int16, mode="c")
        datasets.append(ChessGamesDataset.from_arrays(boards, moves, results, plies, history))

    return datasets[0], datasets[1]

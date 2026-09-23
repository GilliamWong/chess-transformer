from pathlib import Path
from encoding import TOKENS_PER_BOARD
from tools.label_with_stockfish import NO_MOVE
from collections.abc import Iterator
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, Sampler, SequentialSampler
import numpy as np
import torch
import torch.nn.functional as F
import warnings

#pgns, preprocessed training sets, checkpoints, tensorboard runs and the chess engines: no code lives here
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

#training sets too big to parse from pgn on every run, already encoded into arrays on disk.
#see processed_data/scripts_used/ for how they were made
PROCESSED_DATA_DIR = ARTIFACTS_DIR / "processed_data"

#what the policy learns to predict: the move that was played, or stockfish's best moves (tools/label_with_stockfish.py)
POLICY_TARGETS = ("human", "stockfish")

#centipawns to the expected score of the game (0 lost, 1 won): sigmoid(EXPECTED_SCORE_PER_CENTIPAWN * centipawns),
#the curve lichess fitted to its games. A pawn is worth a lot at 0 and next to nothing at +800, which centipawns do not show
EXPECTED_SCORE_PER_CENTIPAWN = 0.00368208

class ChessGamesDataset(Dataset):
    """set of examples of chess games that we can use as a dataset"""

    #positions that are already arrays (load_training_set): the boards and the moves that were played. They can be
    #memory mapped and far too many for a python list: the tensors share the arrays' memory, nothing is copied
    def __init__(self, boards: np.ndarray, moves: np.ndarray):
        if len(boards) != len(moves):
            raise ValueError(f"{len(boards)} boards but {len(moves)} moves")

        #the arrays may be read-only memory maps. torch warns that writing through the tensor would be undefined,
        #and nothing here ever writes to them
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            self.boards = torch.from_numpy(boards)
            self.moves = torch.from_numpy(moves)

        #with_stockfish_targets sets these, the policy target is then made from them
        self.stockfish_moves = None
        self.stockfish_evals = None
        self.target_temperature = None
        #and the value target, stockfish's eval of the position, comes from the same labels (with_eval_targets)
        self.position_evals = None

    #from here on the policy target is stockfish's best moves and not the move played. moves and evals are
    #(positions, top moves) the way tools/label_with_stockfish.py writes them, the best move first
    def with_stockfish_targets(self, moves: np.ndarray, evals: np.ndarray, target_temperature: float) -> "ChessGamesDataset":
        if target_temperature <= 0:
            raise ValueError(f"the target temperature must be above 0, got {target_temperature}")
        self.stockfish_moves, self.stockfish_evals = self.stockfish_labels(moves, evals)
        self.target_temperature = target_temperature
        return self

    #the value target is stockfish's eval of the position: the first column of evals, the score after its best move.
    #Independent of the policy target: a set can keep the moves that were played and still learn stockfish's evals
    def with_eval_targets(self, moves: np.ndarray, evals: np.ndarray) -> "ChessGamesDataset":
        _, label_evals = self.stockfish_labels(moves, evals)
        self.position_evals = label_evals[:, 0]
        return self

    #the labels as tensors, once they are known to fit this set, for the policy and the value targets above
    def stockfish_labels(self, moves: np.ndarray, evals: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        if moves.shape != evals.shape or len(moves) != len(self):
            raise ValueError(f"stockfish moves {moves.shape} and evals {evals.shape} for {len(self)} positions")
        #an unlabelled position would come out as a confident target for move 0
        unlabelled = int((moves[:, 0] == NO_MOVE).sum())
        if unlabelled:
            raise ValueError(f"{unlabelled:,} of {len(self):,} positions are not labelled yet, tools.label_with_stockfish finishes them")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            return torch.from_numpy(moves), torch.from_numpy(evals)

    #the value head's target for rows: the eval as the expected score it stands for, a float from 0 to 1
    def value_target_of(self, index: int | list[int]) -> torch.Tensor:
        return torch.sigmoid(EXPECTED_SCORE_PER_CENTIPAWN * self.position_evals[index].float())

    def __len__(self):
        return len(self.moves)

    #the policy target of rows as a few moves and the probability of each, the most probable first: (positions, moves)
    #twice. The move that was played is one move with probability 1. Stockfish's moves share the probability by a
    #softmax over the expected score after each, divided by the temperature: at 0.02 a move that gives up 2% of the
    #expected score gets 1/e of the best move's share. Places without a move (few legal moves) are move 0 with probability 0
    def policy_target(self, index: int | list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.stockfish_moves is None:
            moves = self.moves[index].long().unsqueeze(-1)
            return moves, torch.ones(moves.shape)

        moves = self.stockfish_moves[index].long()
        no_move = moves == NO_MOVE
        expected_scores = torch.sigmoid(EXPECTED_SCORE_PER_CENTIPAWN * self.stockfish_evals[index].float())
        probabilities = torch.softmax((expected_scores / self.target_temperature).masked_fill(no_move, -torch.inf), dim=-1)
        return moves.masked_fill(no_move, 0), probabilities

    #index is one position, or a list of positions that come back as one batch (see datasets_to_dataloaders)
    #boards, the policy target (moves and their probabilities, see policy_target), the value target (see value_target_of)
    def __getitem__(self, index: int | list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target_moves, target_probabilities = self.policy_target(index)
        return self.boards[index].long(), target_moves, target_probabilities, self.value_target_of(index)


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

        #sets labelled with different numbers of stockfish's top moves have target lists of different widths:
        #the narrow ones are filled up with move 0 at probability 0, which is what a missing move is everywhere
        most_moves = max(target_moves.shape[1] for _, target_moves, *_ in parts)
        parts = [
            (boards, F.pad(target_moves, (0, most_moves - target_moves.shape[1])), F.pad(target_probabilities, (0, most_moves - target_moves.shape[1])), *rest)
            for boards, target_moves, target_probabilities, *rest in parts
        ]

        return tuple(torch.cat(tensors) for tensors in zip(*parts))


#batches for a MixedDataset with the same number of positions from every set, however different their sizes.
#Positions are drawn at random with replacement: over a run that sees a fraction of tens of millions of positions
#that is the same as shuffling, and no set runs out before the others
class EqualSharesBatchSampler(Sampler):
    def __init__(self, dataset: MixedDataset, batch_size: int, generator: torch.Generator | None = None):
        if batch_size % len(dataset.datasets) != 0:
            raise ValueError(f"a batch of {batch_size} cannot be shared equally between {len(dataset.datasets)} sets")
        self.dataset = dataset
        self.batch_size = batch_size
        self.generator = generator

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        share = self.batch_size // len(self.dataset.datasets)
        for _ in range(len(self)):
            batch = []
            for dataset, first_index in zip(self.dataset.datasets, self.dataset.first_indices):
                batch += (torch.randint(len(dataset), (share,), generator=self.generator) + first_index).tolist()
            yield batch


#the samplers hand the dataset a whole batch of indices at once (batch_size=None turns the loader's own
#batching off), so a batch is one tensor lookup instead of one __getitem__ call per position plus a collate.
#measured on 580k positions: 0.2s per epoch instead of 3.0s, which starts to matter at 70 million
def datasets_to_dataloaders(
    train_dataset: ChessGamesDataset | MixedDataset,
    val_dataset: ChessGamesDataset,
    train_batch_size: int = 256,
    val_batch_size: int = 512,
    seed: int | None = None
) -> tuple[DataLoader, DataLoader]:

    #train is reshuffled every epoch so a batch is not 256 positions of the same few games.
    #val is only measured, so its order does not matter
    #with a seed the train order has a generator of its own, so it depends on the seed alone and not on how many random
    #numbers were drawn before training (building a model draws a different number for every architecture). Without
    #one it comes from torch's global generator
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    if isinstance(train_dataset, MixedDataset):
        train_sampler = EqualSharesBatchSampler(train_dataset, train_batch_size, generator)
    else:
        train_sampler = BatchSampler(RandomSampler(train_dataset, generator=generator), batch_size=train_batch_size, drop_last=False)
    val_sampler = BatchSampler(SequentialSampler(val_dataset), batch_size=val_batch_size, drop_last=False)

    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=None)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=None)

    return train_loader, val_loader

#opens <name>_{train,val}_{boards,moves}.bin from processed_data/ as datasets. The arrays are memory mapped,
#so 71 million positions cost no memory up front: pages are read from disk as batches touch them, and the
#first epoch is a little slower than the rest while the operating system caches the file
#
#"baseline+fishtest" mixes sets: every train batch takes the same number of positions from each. The val positions
#are those of the first set alone, so the val numbers of a mix can be compared with runs on that set
#
#the value target is stockfish's eval of the position, so every set needs <name>_{train,val}_stockfish_{moves,evals}.bin,
#written by tools/label_with_stockfish.py. policy_target "stockfish" makes the policy target stockfish's best moves
#from the same files
def load_training_set(name: str, policy_target: str = "human", target_temperature: float | None = None) -> tuple[ChessGamesDataset | MixedDataset, ChessGamesDataset]:
    if policy_target not in POLICY_TARGETS:
        raise ValueError(f"policy target {policy_target} is not one of {POLICY_TARGETS}")

    if "+" in name:
        #a set that is named several times is opened once and gets that many shares of every batch
        names = name.split("+")
        loaded = {one_name: load_training_set(one_name, policy_target, target_temperature) for one_name in dict.fromkeys(names)}
        return MixedDataset([loaded[one_name][0] for one_name in names]), loaded[names[0]][1]

    datasets = []
    for split in ("train", "val"):
        boards_path = PROCESSED_DATA_DIR / f"{name}_{split}_boards.bin"
        moves_path = PROCESSED_DATA_DIR / f"{name}_{split}_moves.bin"
        if not boards_path.exists() or not moves_path.exists():
            raise FileNotFoundError(f"{boards_path.name} or {moves_path.name} is missing from {PROCESSED_DATA_DIR}")
        if not stockfish_moves_path(name, split).exists():
            raise FileNotFoundError(f"{stockfish_moves_path(name, split).name} is missing, tools.label_with_stockfish writes it")

        #the files are raw arrays without a header, in the dtypes ChessGamesDataset stores: uint8 boards, int16 moves.
        #mode "r", read-only. Copy on write ("c") looks harmless but makes windows reserve memory for the whole file:
        #with a set named three times an overnight run held 35 GB of reservations for 12 GB of data
        boards = np.memmap(boards_path, dtype=np.uint8, mode="r").reshape(-1, TOKENS_PER_BOARD)
        moves = np.memmap(moves_path, dtype=np.int16, mode="r")
        dataset = ChessGamesDataset(boards, moves)

        #(positions, top moves), and the size of the file says how many top moves that is
        stockfish_moves = np.memmap(stockfish_moves_path(name, split), dtype=np.int16, mode="r")
        stockfish_evals = np.memmap(PROCESSED_DATA_DIR / f"{name}_{split}_stockfish_evals.bin", dtype=np.int16, mode="r")
        if len(stockfish_moves) == 0 or len(stockfish_moves) % len(dataset) != 0:
            raise ValueError(f"{len(stockfish_moves):,} stockfish moves cannot be shared between {len(dataset):,} positions")
        stockfish_moves, stockfish_evals = stockfish_moves.reshape(len(dataset), -1), stockfish_evals.reshape(len(dataset), -1)
        if policy_target == "stockfish":
            if target_temperature is None:
                raise ValueError("stockfish targets need a target temperature")
            dataset.with_stockfish_targets(stockfish_moves, stockfish_evals, target_temperature)
        dataset.with_eval_targets(stockfish_moves, stockfish_evals)
        datasets.append(dataset)

    return datasets[0], datasets[1]


def stockfish_moves_path(name: str, split: str) -> Path:
    return PROCESSED_DATA_DIR / f"{name}_{split}_stockfish_moves.bin"

from pathlib import Path
from encoding import TOKENS_PER_BOARD
from tools.label_with_stockfish import NO_MOVE
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler
import numpy as np
import torch
import warnings

#pgns, preprocessed training sets, checkpoints, tensorboard runs and the chess engines: no code lives here
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

#training sets too big to parse from pgn on every run, already encoded into arrays on disk.
#see processed_data/scripts_used/ for how they were made
PROCESSED_DATA_DIR = ARTIFACTS_DIR / "processed_data"

#centipawns to the expected score of the game (0 lost, 1 won): sigmoid(EXPECTED_SCORE_PER_CENTIPAWN * centipawns),
#the curve lichess fitted to its games. A pawn is worth a lot at 0 and next to nothing at +800, which centipawns do not show
EXPECTED_SCORE_PER_CENTIPAWN = 0.00368208

class ChessGamesDataset(Dataset):
    """set of examples of chess games that we can use as a dataset"""

    #positions that are already arrays (load_training_set): the boards, and stockfish's best moves and the score after
    #each, (positions, top moves) the way tools/label_with_stockfish.py writes them, the best move first. They can be
    #memory mapped and far too many for a python list: the tensors share the arrays' memory, nothing is copied
    def __init__(self, boards: np.ndarray, stockfish_moves: np.ndarray, stockfish_evals: np.ndarray, target_temperature: float):
        if target_temperature <= 0:
            raise ValueError(f"the target temperature must be above 0, got {target_temperature}")
        if stockfish_moves.shape != stockfish_evals.shape or len(stockfish_moves) != len(boards):
            raise ValueError(f"stockfish moves {stockfish_moves.shape} and evals {stockfish_evals.shape} for {len(boards)} positions")
        #an unlabelled position would come out as a confident target for move 0
        unlabelled = int((stockfish_moves[:, 0] == NO_MOVE).sum())
        if unlabelled:
            raise ValueError(f"{unlabelled:,} of {len(boards):,} positions are not labelled yet, tools.label_with_stockfish finishes them")

        #the arrays may be read-only memory maps. torch warns that writing through the tensor would be undefined,
        #and nothing here ever writes to them
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            self.boards = torch.from_numpy(boards)
            self.stockfish_moves = torch.from_numpy(stockfish_moves)
            self.stockfish_evals = torch.from_numpy(stockfish_evals)
        self.target_temperature = target_temperature

    #the value head's target for rows: stockfish's eval of the position (the first column of the evals, the score after
    #its best move) as the expected score it stands for, a float from 0 to 1
    def value_target_of(self, index: int | list[int]) -> torch.Tensor:
        return torch.sigmoid(EXPECTED_SCORE_PER_CENTIPAWN * self.stockfish_evals[index, 0].float())

    def __len__(self):
        return len(self.boards)

    #the policy target of rows as stockfish's moves and the probability of each, the most probable first: (positions, moves)
    #twice. They share the probability by a softmax over the expected score after each, divided by the temperature: at 0.02
    #a move that gives up 2% of the expected score gets 1/e of the best move's share. Places without a move (few legal
    #moves) are move 0 with probability 0
    def policy_target(self, index: int | list[int]) -> tuple[torch.Tensor, torch.Tensor]:
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


#the samplers hand the dataset a whole batch of indices at once (batch_size=None turns the loader's own
#batching off), so a batch is one tensor lookup instead of one __getitem__ call per position plus a collate.
#measured on 580k positions: 0.2s per epoch instead of 3.0s, which starts to matter at 70 million
def datasets_to_dataloaders(
    train_dataset: ChessGamesDataset,
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
    train_sampler = BatchSampler(RandomSampler(train_dataset, generator=generator), batch_size=train_batch_size, drop_last=False)
    val_sampler = BatchSampler(SequentialSampler(val_dataset), batch_size=val_batch_size, drop_last=False)

    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=None)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=None)

    return train_loader, val_loader

#opens <name>_{train,val}_boards.bin from processed_data/ as datasets, with stockfish's labels of every position from
#<name>_{train,val}_stockfish_{moves,evals}.bin (written by tools/label_with_stockfish.py): its best moves, which share
#the policy target at target_temperature, and its eval, the value target. The arrays are memory mapped,
#so 71 million positions cost no memory up front: pages are read from disk as batches touch them, and the
#first epoch is a little slower than the rest while the operating system caches the file
def load_training_set(name: str, target_temperature: float) -> tuple[ChessGamesDataset, ChessGamesDataset]:
    datasets = []
    for split in ("train", "val"):
        boards_path = PROCESSED_DATA_DIR / f"{name}_{split}_boards.bin"
        stockfish_moves_path = PROCESSED_DATA_DIR / f"{name}_{split}_stockfish_moves.bin"
        if not boards_path.exists():
            raise FileNotFoundError(f"{boards_path.name} is missing from {PROCESSED_DATA_DIR}")
        if not stockfish_moves_path.exists():
            raise FileNotFoundError(f"{stockfish_moves_path.name} is missing, tools.label_with_stockfish writes it")

        #the files are raw arrays without a header, in the dtypes ChessGamesDataset stores: uint8 boards, int16 stockfish
        #moves and evals. mode "r", read-only. Copy on write ("c") looks harmless but makes windows reserve memory for the
        #whole file: with a set named three times an overnight run held 35 GB of reservations for 12 GB of data
        boards = np.memmap(boards_path, dtype=np.uint8, mode="r").reshape(-1, TOKENS_PER_BOARD)
        #(positions, top moves), and the size of the file says how many top moves that is
        stockfish_moves = np.memmap(stockfish_moves_path, dtype=np.int16, mode="r")
        stockfish_evals = np.memmap(PROCESSED_DATA_DIR / f"{name}_{split}_stockfish_evals.bin", dtype=np.int16, mode="r")
        if len(stockfish_moves) == 0 or len(stockfish_moves) % len(boards) != 0:
            raise ValueError(f"{len(stockfish_moves):,} stockfish moves cannot be shared between {len(boards):,} positions")
        datasets.append(ChessGamesDataset(boards, stockfish_moves.reshape(len(boards), -1), stockfish_evals.reshape(len(boards), -1), target_temperature))

    return datasets[0], datasets[1]

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

#the value target of a self play position: this much of the game's result, the rest the search's mean score at the root.
#The result is the truth about the model's own play but one noisy label for a whole game, the search's score is a label
#of its own for every position but mostly the network grading itself. Half of each did better than either in AlphaZero
#style training (Oracle's "Lessons from AlphaZero"), and leela blends them too
RESULT_WEIGHT = 0.5
#the policy target of a self play position: the search's visit shares of its top moves raised to 1 / VISIT_TEMPERATURE and
#made to add up to 1 again, 1 the shares as recorded and 0 the most visited move alone. At 64 simulations the recorded
#shares are broad (the most visited move has 46% of the visits) and on average a worse move than the policy's own; sharpened
#they are better. Stockfish at depth 12 on 2,000 positions of selfplay_r1_reuse, in expected score lost: 0.0348 at 1, 0.0226
#at 0.5, 0.0189 at 0.25, 0.0177 at 0.1, against 0.0293 for the start model's own move
VISIT_TEMPERATURE = 1.0

#the policy's blunders: the labelled moves that give away at least this much expected score against stockfish's best
#move. main.blunder_loss pushes the policy's probability off them. Optional files next to a set, written by
#tools/label_blunders.py (per position up to 16 moves that give away at least 0.2, the mildest first; the dataset keeps
#those at or above the threshold, whatever the order):
#   worst_moves    move ids, (positions, slots), NO_MOVE past the labelled moves and for a position not labelled
#   worst_losses   the expected score each gives away (0 = as good as the best move)
BLUNDER_THRESHOLD = 0.2
WORST_FILES = {"worst_moves": np.int16, "worst_losses": np.float16}

#the files selfplay.py writes next to <name>_{split}_boards.bin, per position, and their dtypes:
#   visit_moves    the most visited moves of the search at the root, (positions, top moves), NO_MOVE where there were fewer
#   visit_shares   each one's share of the visits to those moves, adding up to 1
#   outcomes       the game's result for the side to move (0 lost, 0.5 drawn, 1 won)
#   root_scores    the search's mean score at the root for the side to move, from 0 to 1
SELF_PLAY_FILES = {"visit_moves": np.int16, "visit_shares": np.float32, "outcomes": np.float32, "root_scores": np.float32}

#the arrays may be read-only memory maps. torch warns that writing through the tensor would be undefined, and nothing
#here ever writes to them
def shared_tensor(array: np.ndarray) -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
        return torch.from_numpy(array)

class ChessGamesDataset(Dataset):
    """set of examples of chess games that we can use as a dataset"""

    #positions that are already arrays (load_training_set): the boards, and stockfish's best moves and the score after
    #each, (positions, top moves) the way tools/label_with_stockfish.py writes them, the best move first. They can be
    #memory mapped and far too many for a python list: the tensors share the arrays' memory, nothing is copied.
    #worst_moves and worst_losses (WORST_FILES) are the optional blunder labels, both (positions, slots)
    def __init__(self, boards: np.ndarray, stockfish_moves: np.ndarray, stockfish_evals: np.ndarray, target_temperature: float,
                 worst_moves: np.ndarray | None = None, worst_losses: np.ndarray | None = None):
        if target_temperature <= 0:
            raise ValueError(f"the target temperature must be above 0, got {target_temperature}")
        if stockfish_moves.shape != stockfish_evals.shape or len(stockfish_moves) != len(boards):
            raise ValueError(f"stockfish moves {stockfish_moves.shape} and evals {stockfish_evals.shape} for {len(boards)} positions")
        #an unlabelled position would come out as a confident target for move 0
        unlabelled = int((stockfish_moves[:, 0] == NO_MOVE).sum())
        if unlabelled:
            raise ValueError(f"{unlabelled:,} of {len(boards):,} positions are not labelled yet, tools.label_with_stockfish finishes them")
        if (worst_moves is None) != (worst_losses is None):
            raise ValueError("blunder labels need both worst moves and worst losses")
        if worst_moves is not None and (worst_moves.shape != worst_losses.shape or len(worst_moves) != len(boards)):
            raise ValueError(f"worst moves {worst_moves.shape} and losses {worst_losses.shape} for {len(boards)} positions")

        self.boards = shared_tensor(boards)
        self.stockfish_moves = shared_tensor(stockfish_moves)
        self.stockfish_evals = shared_tensor(stockfish_evals)
        self.target_temperature = target_temperature
        self.worst_moves = shared_tensor(worst_moves) if worst_moves is not None else None
        self.worst_losses = shared_tensor(worst_losses) if worst_losses is not None else None

    #everything the set reads from disk
    def tensors(self) -> tuple[torch.Tensor, ...]:
        blunder_labels = (self.worst_moves, self.worst_losses) if self.worst_moves is not None else ()
        return self.boards, self.stockfish_moves, self.stockfish_evals, *blunder_labels

    #the blunders of rows: the labelled moves that give away at least BLUNDER_THRESHOLD, NO_MOVE in the other places.
    #A set without blunder labels has one place per position, always NO_MOVE
    def blunders_of(self, index: int | list[int]) -> torch.Tensor:
        if self.worst_moves is None:
            return torch.full_like(self.stockfish_moves[index][..., :1], NO_MOVE, dtype=torch.long)
        moves = self.worst_moves[index].long()
        #compared in the labels' own precision: float16 stores 0.2 as 0.19995, which is still meant to be at the threshold
        threshold = torch.tensor(BLUNDER_THRESHOLD, dtype=self.worst_losses.dtype)
        return moves.masked_fill(self.worst_losses[index] < threshold, NO_MOVE)

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
    #and the blunders (see blunders_of)
    def __getitem__(self, index: int | list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target_moves, target_probabilities = self.policy_target(index)
        return self.boards[index].long(), target_moves, target_probabilities, self.value_target_of(index), self.blunders_of(index)


#the positions of selfplay.py's games with what the search found at each (SELF_PLAY_FILES), in the same four tensors as
#ChessGamesDataset: the policy learns the search's visit shares, the value head a blend of the result and the root's score
class SelfPlayDataset(Dataset):
    def __init__(self, boards: np.ndarray, visit_moves: np.ndarray, visit_shares: np.ndarray, outcomes: np.ndarray,
                 root_scores: np.ndarray, result_weight: float, visit_temperature: float = VISIT_TEMPERATURE):
        if not 0 <= result_weight <= 1:
            raise ValueError(f"the result weight is a share from 0 to 1, got {result_weight}")
        if visit_temperature < 0:
            raise ValueError(f"the visit temperature is 0 or more, got {visit_temperature}")
        if visit_moves.shape != visit_shares.shape or not len(boards) == len(visit_moves) == len(outcomes) == len(root_scores):
            raise ValueError(f"{len(boards)} boards, visit moves {visit_moves.shape}, shares {visit_shares.shape}, "
                             f"{len(outcomes)} outcomes and {len(root_scores)} root scores do not belong together")

        self.boards = shared_tensor(boards)
        self.visit_moves = shared_tensor(visit_moves)
        self.visit_shares = shared_tensor(visit_shares)
        self.outcomes = shared_tensor(outcomes)
        self.root_scores = shared_tensor(root_scores)
        self.result_weight = result_weight
        self.visit_temperature = visit_temperature

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return self.boards, self.visit_moves, self.visit_shares, self.outcomes, self.root_scores

    def __len__(self):
        return len(self.boards)

    def value_target_of(self, index: int | list[int]) -> torch.Tensor:
        return self.result_weight * self.outcomes[index] + (1 - self.result_weight) * self.root_scores[index]

    #places without a move are move 0 at share 0, as in ChessGamesDataset
    def policy_target(self, index: int | list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        moves = self.visit_moves[index].long()
        no_move = moves == NO_MOVE
        shares = self.visit_shares[index].masked_fill(no_move, 0.0)
        if self.visit_temperature != 1:
            shares = sharpened(shares, self.visit_temperature)
        return moves.masked_fill(no_move, 0), shares

    #the same five tensors as ChessGamesDataset; self play has no blunder labels
    def __getitem__(self, index: int | list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target_moves, target_probabilities = self.policy_target(index)
        no_blunders = torch.full_like(self.visit_moves[index][..., :1], NO_MOVE, dtype=torch.long)
        return self.boards[index].long(), target_moves, target_probabilities, self.value_target_of(index), no_blunders


#visit shares raised to 1 / temperature and made to add up to 1, divided by the largest first so that a low temperature does
#not round them all to 0. At 0 the most visited move alone, shared out between moves with as many visits
def sharpened(shares: torch.Tensor, temperature: float) -> torch.Tensor:
    relative = shares / shares.amax(dim=-1, keepdim=True).clamp_min(1e-30)
    weights = (relative == 1).float() if temperature == 0 else relative ** (1 / temperature)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-30)


TrainingDataset = ChessGamesDataset | SelfPlayDataset


#batches that take a fixed share of their positions from a second loader: fine-tuning on self play keeps some stockfish
#labelled positions in every batch, so the model does not forget what it learned from them. The second loader starts
#over whenever it runs out; one pass over the first is still one epoch
class MixedLoader:
    def __init__(self, main_loader: DataLoader, mix_loader: DataLoader):
        self.main_loader = main_loader
        self.mix_loader = mix_loader
        self.dataset = main_loader.dataset #what the rest of the code reads the size of the set from
        self.mix_batches = iter(())

    def __len__(self):
        return len(self.main_loader)

    def __iter__(self):
        for main_batch in self.main_loader:
            mix_batch = next(self.mix_batches, None)
            if mix_batch is None:
                self.mix_batches = iter(self.mix_loader)
                mix_batch = next(self.mix_batches)
            yield concatenate_batches(main_batch, mix_batch)

#two batches as one. The sets may keep a different number of target moves: the narrower one is padded with move 0 at
#probability 0, which adds nothing to the loss. Blunders are padded with NO_MOVE, which is no blunder
def concatenate_batches(first: tuple[torch.Tensor, ...], second: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    boards, moves, probabilities, values, blunders = zip(first, second)
    width = max(part.shape[1] for part in moves)
    moves = [torch.nn.functional.pad(part, (0, width - part.shape[1])) for part in moves]
    probabilities = [torch.nn.functional.pad(part, (0, width - part.shape[1])) for part in probabilities]
    blunder_width = max(part.shape[1] for part in blunders)
    blunders = [torch.nn.functional.pad(part, (0, blunder_width - part.shape[1]), value=NO_MOVE) for part in blunders]
    return torch.cat(boards), torch.cat(moves), torch.cat(probabilities), torch.cat(values), torch.cat(blunders)


#the samplers hand the dataset a whole batch of indices at once (batch_size=None turns the loader's own
#batching off), so a batch is one tensor lookup instead of one __getitem__ call per position plus a collate.
#measured on 580k positions: 0.2s per epoch instead of 3.0s, which starts to matter at 70 million
def datasets_to_dataloaders(
    train_dataset: TrainingDataset,
    val_dataset: TrainingDataset,
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
#first epoch is a little slower than the rest while the operating system caches the file.
#A self play set (selfplay.py writes SELF_PLAY_FILES instead of stockfish's labels) comes back as SelfPlayDataset,
#its value target blended by result_weight and its policy target sharpened by visit_temperature
def load_training_set(name: str, target_temperature: float, result_weight: float = RESULT_WEIGHT,
                      visit_temperature: float = VISIT_TEMPERATURE) -> tuple[TrainingDataset, TrainingDataset]:
    datasets = []
    for split in ("train", "val"):
        boards_path = PROCESSED_DATA_DIR / f"{name}_{split}_boards.bin"
        stockfish_moves_path = PROCESSED_DATA_DIR / f"{name}_{split}_stockfish_moves.bin"
        if not boards_path.exists():
            raise FileNotFoundError(f"{boards_path.name} is missing from {PROCESSED_DATA_DIR}")

        #the files are raw arrays without a header, in the dtypes ChessGamesDataset stores: uint8 boards, int16 stockfish
        #moves and evals. mode "r", read-only. Copy on write ("c") looks harmless but makes windows reserve memory for the
        #whole file: with a set named three times an overnight run held 35 GB of reservations for 12 GB of data
        boards = np.memmap(boards_path, dtype=np.uint8, mode="r").reshape(-1, TOKENS_PER_BOARD)

        if (PROCESSED_DATA_DIR / f"{name}_{split}_visit_moves.bin").exists():
            arrays = {kind: np.memmap(PROCESSED_DATA_DIR / f"{name}_{split}_{kind}.bin", dtype=dtype, mode="r") for kind, dtype in SELF_PLAY_FILES.items()}
            datasets.append(SelfPlayDataset(boards, arrays["visit_moves"].reshape(len(boards), -1), arrays["visit_shares"].reshape(len(boards), -1),
                                            arrays["outcomes"], arrays["root_scores"], result_weight, visit_temperature))
            continue

        if not stockfish_moves_path.exists():
            raise FileNotFoundError(f"{stockfish_moves_path.name} is missing, tools.label_with_stockfish writes it")
        #(positions, top moves), and the size of the file says how many top moves that is
        stockfish_moves = np.memmap(stockfish_moves_path, dtype=np.int16, mode="r")
        stockfish_evals = np.memmap(PROCESSED_DATA_DIR / f"{name}_{split}_stockfish_evals.bin", dtype=np.int16, mode="r")
        if len(stockfish_moves) == 0 or len(stockfish_moves) % len(boards) != 0:
            raise ValueError(f"{len(stockfish_moves):,} stockfish moves cannot be shared between {len(boards):,} positions")
        #the blunder labels, when the set has them (WORST_FILES)
        worst = {}
        if (PROCESSED_DATA_DIR / f"{name}_{split}_worst_moves.bin").exists():
            worst = {kind: np.memmap(PROCESSED_DATA_DIR / f"{name}_{split}_{kind}.bin", dtype=dtype, mode="r").reshape(len(boards), -1)
                     for kind, dtype in WORST_FILES.items()}
        datasets.append(ChessGamesDataset(boards, stockfish_moves.reshape(len(boards), -1), stockfish_evals.reshape(len(boards), -1), target_temperature,
                                          worst.get("worst_moves"), worst.get("worst_losses")))

    return datasets[0], datasets[1]

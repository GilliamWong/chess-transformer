from pathlib import Path
from encoding import TOKENS_PER_BOARD
from tools.label_with_stockfish import NO_MOVE
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler
import numpy as np
import torch
import warnings

#no code lives here: pgns, training sets, checkpoints, runs, engines
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

#training sets as raw arrays on disk, see processed_data/scripts_used/ for how they were made
PROCESSED_DATA_DIR = ARTIFACTS_DIR / "processed_data"

#centipawns to expected score (0 lost, 1 won): sigmoid(EXPECTED_SCORE_PER_CENTIPAWN * centipawns), lichess's fit
EXPECTED_SCORE_PER_CENTIPAWN = 0.00368208

#self play value target: this much of the game's result, the rest the search's root score
RESULT_WEIGHT = 0.5
#self play policy target: visit shares raised to 1 / VISIT_TEMPERATURE and renormalised (0 = most visited move alone)
VISIT_TEMPERATURE = 1.0

#labelled moves that give away at least this much expected score against stockfish's best count as blunders
BLUNDER_THRESHOLD = 0.2
#optional blunder labels next to a set (tools/label_blunders.py), both (positions, slots), NO_MOVE where empty
WORST_FILES = {"worst_moves": np.int16, "worst_losses": np.float16}

#selfplay.py's files next to <name>_{split}_boards.bin:
#   visit_moves (positions, top moves) NO_MOVE where fewer, visit_shares adding up to 1,
#   outcomes and root_scores for the side to move, 0 to 1
SELF_PLAY_FILES = {"visit_moves": np.int16, "visit_shares": np.float32, "outcomes": np.float32, "root_scores": np.float32}

#the arrays may be read-only memory maps; nothing here writes to them, so torch's warning is noise
def shared_tensor(array: np.ndarray) -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
        return torch.from_numpy(array)

class ChessGamesDataset(Dataset):
    #boards, stockfish's best moves and the score after each as (positions, top moves), best first. The tensors share
    #the (possibly memory mapped) arrays' memory. worst_moves and worst_losses are the optional blunder labels
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

    #labelled moves at or above BLUNDER_THRESHOLD, NO_MOVE elsewhere; one NO_MOVE column without blunder labels
    def blunders_of(self, index: int | list[int]) -> torch.Tensor:
        if self.worst_moves is None:
            return torch.full_like(self.stockfish_moves[index][..., :1], NO_MOVE, dtype=torch.long)
        moves = self.worst_moves[index].long()
        #compared in float16: 0.2 is stored as 0.19995 and still counts as at the threshold
        threshold = torch.tensor(BLUNDER_THRESHOLD, dtype=self.worst_losses.dtype)
        return moves.masked_fill(self.worst_losses[index] < threshold, NO_MOVE)

    #stockfish's eval of the position (first column) as an expected score from 0 to 1
    def value_target_of(self, index: int | list[int]) -> torch.Tensor:
        return torch.sigmoid(EXPECTED_SCORE_PER_CENTIPAWN * self.stockfish_evals[index, 0].float())

    def __len__(self):
        return len(self.boards)

    #(moves, probabilities): a softmax over the expected score after each move / target_temperature.
    #Empty places are move 0 at probability 0
    def policy_target(self, index: int | list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        moves = self.stockfish_moves[index].long()
        no_move = moves == NO_MOVE
        expected_scores = torch.sigmoid(EXPECTED_SCORE_PER_CENTIPAWN * self.stockfish_evals[index].float())
        probabilities = torch.softmax((expected_scores / self.target_temperature).masked_fill(no_move, -torch.inf), dim=-1)
        return moves.masked_fill(no_move, 0), probabilities

    #index is one position or a list of positions returned as one batch: boards, target moves, target probabilities,
    #value target, blunders
    def __getitem__(self, index: int | list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target_moves, target_probabilities = self.policy_target(index)
        return self.boards[index].long(), target_moves, target_probabilities, self.value_target_of(index), self.blunders_of(index)


#selfplay.py's positions: the policy learns the search's visit shares, the value head a blend of result and root score
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

    #empty places are move 0 at share 0
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


#shares ** (1 / temperature), renormalised. Divided by the max first so a low temperature does not underflow to 0;
#at 0 the most visited move(s) alone
def sharpened(shares: torch.Tensor, temperature: float) -> torch.Tensor:
    relative = shares / shares.amax(dim=-1, keepdim=True).clamp_min(1e-30)
    weights = (relative == 1).float() if temperature == 0 else relative ** (1 / temperature)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-30)


TrainingDataset = ChessGamesDataset | SelfPlayDataset


#every batch takes a fixed share from a second loader (stockfish positions while fine-tuning on self play), which
#restarts whenever it runs out; one pass over the main loader is one epoch
class MixedLoader:
    def __init__(self, main_loader: DataLoader, mix_loader: DataLoader):
        self.main_loader = main_loader
        self.mix_loader = mix_loader
        self.dataset = main_loader.dataset #the rest of the code reads the set's size from here
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

#the narrower target is padded with move 0 at probability 0, the blunders with NO_MOVE
def concatenate_batches(first: tuple[torch.Tensor, ...], second: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    boards, moves, probabilities, values, blunders = zip(first, second)
    width = max(part.shape[1] for part in moves)
    moves = [torch.nn.functional.pad(part, (0, width - part.shape[1])) for part in moves]
    probabilities = [torch.nn.functional.pad(part, (0, width - part.shape[1])) for part in probabilities]
    blunder_width = max(part.shape[1] for part in blunders)
    blunders = [torch.nn.functional.pad(part, (0, blunder_width - part.shape[1]), value=NO_MOVE) for part in blunders]
    return torch.cat(boards), torch.cat(moves), torch.cat(probabilities), torch.cat(values), torch.cat(blunders)


#the samplers hand the dataset a whole batch of indices (batch_size=None turns off the loader's own batching):
#one tensor lookup per batch instead of a __getitem__ per position plus a collate, 15x faster
def datasets_to_dataloaders(
    train_dataset: TrainingDataset,
    val_dataset: TrainingDataset,
    train_batch_size: int = 256,
    val_batch_size: int = 512,
    seed: int | None = None
) -> tuple[DataLoader, DataLoader]:

    #with a seed the train order has its own generator, so it doesn't depend on how many random numbers building the
    #model drew
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    train_sampler = BatchSampler(RandomSampler(train_dataset, generator=generator), batch_size=train_batch_size, drop_last=False)
    val_sampler = BatchSampler(SequentialSampler(val_dataset), batch_size=val_batch_size, drop_last=False)

    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=None)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=None)

    return train_loader, val_loader

#opens <name>_{train,val}_*.bin from processed_data/ as memory mapped datasets: stockfish labelled sets as
#ChessGamesDataset, self play sets (SELF_PLAY_FILES) as SelfPlayDataset
def load_training_set(name: str, target_temperature: float, result_weight: float = RESULT_WEIGHT,
                      visit_temperature: float = VISIT_TEMPERATURE) -> tuple[TrainingDataset, TrainingDataset]:
    datasets = []
    for split in ("train", "val"):
        boards_path = PROCESSED_DATA_DIR / f"{name}_{split}_boards.bin"
        stockfish_moves_path = PROCESSED_DATA_DIR / f"{name}_{split}_stockfish_moves.bin"
        if not boards_path.exists():
            raise FileNotFoundError(f"{boards_path.name} is missing from {PROCESSED_DATA_DIR}")

        #raw arrays without a header. Mode "r": copy on write ("c") makes windows reserve memory for the whole file
        #(35 GB of reservations for 12 GB of data in one overnight run)
        boards = np.memmap(boards_path, dtype=np.uint8, mode="r").reshape(-1, TOKENS_PER_BOARD)

        if (PROCESSED_DATA_DIR / f"{name}_{split}_visit_moves.bin").exists():
            arrays = {kind: np.memmap(PROCESSED_DATA_DIR / f"{name}_{split}_{kind}.bin", dtype=dtype, mode="r") for kind, dtype in SELF_PLAY_FILES.items()}
            datasets.append(SelfPlayDataset(boards, arrays["visit_moves"].reshape(len(boards), -1), arrays["visit_shares"].reshape(len(boards), -1),
                                            arrays["outcomes"], arrays["root_scores"], result_weight, visit_temperature))
            continue

        if not stockfish_moves_path.exists():
            raise FileNotFoundError(f"{stockfish_moves_path.name} is missing, tools.label_with_stockfish writes it")
        #the file size says how many top moves per position
        stockfish_moves = np.memmap(stockfish_moves_path, dtype=np.int16, mode="r")
        stockfish_evals = np.memmap(PROCESSED_DATA_DIR / f"{name}_{split}_stockfish_evals.bin", dtype=np.int16, mode="r")
        if len(stockfish_moves) == 0 or len(stockfish_moves) % len(boards) != 0:
            raise ValueError(f"{len(stockfish_moves):,} stockfish moves cannot be shared between {len(boards):,} positions")
        worst = {}
        if (PROCESSED_DATA_DIR / f"{name}_{split}_worst_moves.bin").exists():
            worst = {kind: np.memmap(PROCESSED_DATA_DIR / f"{name}_{split}_{kind}.bin", dtype=dtype, mode="r").reshape(len(boards), -1)
                     for kind, dtype in WORST_FILES.items()}
        datasets.append(ChessGamesDataset(boards, stockfish_moves.reshape(len(boards), -1), stockfish_evals.reshape(len(boards), -1), target_temperature,
                                          worst.get("worst_moves"), worst.get("worst_losses")))

    return datasets[0], datasets[1]

from pathlib import Path
from model.checkpoint import ARTIFACTS_DIR
from model.encoding import EXPECTED_SCORE_PER_CENTIPAWN, NO_MOVE, TOKENS_PER_BOARD
from training.mirror import MirrorRows, draw_mirrors, mirror_moves, read_boards
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, Sampler, SequentialSampler
import numpy as np
import torch
import warnings

PROCESSED_DATA_DIR = ARTIFACTS_DIR / "processed_data"

RESULT_WEIGHT = 0.5
VISIT_TEMPERATURE = 1.0

#expected score a labelled move gives away against stockfish's best, from 0 to 1, at which it is a blunder
BLUNDER_THRESHOLD = 0.3
#(positions, slots), NO_MOVE where a slot is empty
WORST_FILES = {"worst_moves": np.int16, "worst_losses": np.float16}
#top moves (positions, slots), their probabilities (adding up to 1) and the teacher's expected score (positions,)
TEACHER_FILES = {"teacher_moves": np.int16, "teacher_probabilities": np.float16, "teacher_scores": np.float16}
#visit moves (positions, top moves) with NO_MOVE where fewer, shares adding up to 1, the side to move's outcome and root score from 0 to 1
SELF_PLAY_FILES = {"visit_moves": np.int16, "visit_shares": np.float32, "outcomes": np.float32, "root_scores": np.float32}

#the arrays are read-only memory maps that nothing here writes to
def shared_tensor(array: np.ndarray) -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
        return torch.from_numpy(array)


#subclasses define policy_target and value_target_of
class TrainingDataset(Dataset):
    def __init__(self, boards: np.ndarray, worst_moves: np.ndarray | None, worst_losses: np.ndarray | None, blunder_threshold: float):
        if blunder_threshold <= 0:
            raise ValueError(f"the blunder threshold is an expected score lost, above 0, got {blunder_threshold}")
        if (worst_moves is None) != (worst_losses is None):
            raise ValueError("blunder labels need both worst moves and worst losses")
        if worst_moves is not None and (worst_moves.shape != worst_losses.shape or len(worst_moves) != len(boards)):
            raise ValueError(f"worst moves {worst_moves.shape} and losses {worst_losses.shape} for {len(boards)} positions")

        self.boards = shared_tensor(boards)
        self.worst_moves = shared_tensor(worst_moves) if worst_moves is not None else None
        self.worst_losses = shared_tensor(worst_losses) if worst_losses is not None else None
        self.blunder_threshold = blunder_threshold

    def __len__(self):
        return len(self.boards)

    def blunder_tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.worst_moves, self.worst_losses) if self.worst_moves is not None else ()

    #NO_MOVE where the labelled loss is under the threshold; without labels one NO_MOVE column
    def blunders_of(self, index: int | list[int], mirrored: torch.Tensor | None = None) -> torch.Tensor:
        if self.worst_moves is None:
            return torch.full_like(self.boards[index, :1], NO_MOVE, dtype=torch.long)
        #compared in float16, so a loss stored as the threshold counts as a blunder
        below = self.worst_losses[index] < torch.tensor(self.blunder_threshold, dtype=self.worst_losses.dtype)
        return mirror_moves(self.worst_moves[index].long().masked_fill(below, NO_MOVE), mirrored)

    def __getitem__(self, index: int | list[int] | MirrorRows) -> tuple[torch.Tensor, ...]:
        index, boards, mirrored = read_boards(self.boards, index)
        target_moves, target_probabilities = self.policy_target(index, mirrored)
        return boards, target_moves, target_probabilities, self.value_target_of(index), self.blunders_of(index, mirrored)


#stockfish's moves and evals are (positions, top moves), best first; teacher_weight is the teacher's share of the policy and value targets
class ChessGamesDataset(TrainingDataset):
    def __init__(self, boards: np.ndarray, stockfish_moves: np.ndarray, stockfish_evals: np.ndarray, target_temperature: float,
                 worst_moves: np.ndarray | None = None, worst_losses: np.ndarray | None = None, blunder_threshold: float = BLUNDER_THRESHOLD,
                 teacher: dict[str, np.ndarray] | None = None, teacher_weight: float = 0.0):
        if target_temperature <= 0:
            raise ValueError(f"the target temperature must be above 0, got {target_temperature}")
        if not 0 <= teacher_weight <= 1:
            raise ValueError(f"the teacher's share of the targets is between 0 and 1, got {teacher_weight}")
        if teacher_weight and teacher is None:
            raise ValueError("a teacher weight needs the teacher's targets")
        if teacher is not None and not (len(teacher["teacher_moves"]) == len(teacher["teacher_probabilities"]) == len(teacher["teacher_scores"]) == len(boards)):
            raise ValueError(f"the teacher's targets do not cover the {len(boards):,} positions one each")
        if stockfish_moves.shape != stockfish_evals.shape or len(stockfish_moves) != len(boards):
            raise ValueError(f"stockfish moves {stockfish_moves.shape} and evals {stockfish_evals.shape} for {len(boards)} positions")
        unlabelled = int((stockfish_moves[:, 0] == NO_MOVE).sum())
        if unlabelled:
            raise ValueError(f"{unlabelled:,} of {len(boards):,} positions are not labelled yet")
        super().__init__(boards, worst_moves, worst_losses, blunder_threshold)

        self.stockfish_moves = shared_tensor(stockfish_moves)
        self.stockfish_evals = shared_tensor(stockfish_evals)
        self.target_temperature = target_temperature
        self.teacher = {kind: shared_tensor(array) for kind, array in teacher.items()} if teacher_weight else None
        self.teacher_weight = teacher_weight

    #everything the set reads from disk
    def tensors(self) -> tuple[torch.Tensor, ...]:
        teacher = tuple(self.teacher.values()) if self.teacher is not None else ()
        return self.boards, self.stockfish_moves, self.stockfish_evals, *self.blunder_tensors(), *teacher

    def value_target_of(self, index: int | list[int]) -> torch.Tensor:
        stockfish = torch.sigmoid(EXPECTED_SCORE_PER_CENTIPAWN * self.stockfish_evals[index, 0].float())
        if self.teacher is None:
            return stockfish
        return (1 - self.teacher_weight) * stockfish + self.teacher_weight * self.teacher["teacher_scores"][index].float()

    #empty places are move 0 at probability 0; a move in both stockfish's and the teacher's lists gets both shares, which the cross entropy adds up
    def policy_target(self, index: int | list[int], mirrored: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        moves = mirror_moves(self.stockfish_moves[index].long(), mirrored)
        no_move = moves == NO_MOVE
        expected_scores = torch.sigmoid(EXPECTED_SCORE_PER_CENTIPAWN * self.stockfish_evals[index].float())
        probabilities = torch.softmax((expected_scores / self.target_temperature).masked_fill(no_move, -torch.inf), dim=-1)
        moves = moves.masked_fill(no_move, 0)
        if self.teacher is None:
            return moves, probabilities
        return (torch.cat([moves, mirror_moves(self.teacher["teacher_moves"][index].long(), mirrored)], dim=-1),
                torch.cat([(1 - self.teacher_weight) * probabilities, self.teacher_weight * self.teacher["teacher_probabilities"][index].float()], dim=-1))


class SelfPlayDataset(TrainingDataset):
    def __init__(self, boards: np.ndarray, visit_moves: np.ndarray, visit_shares: np.ndarray, outcomes: np.ndarray,
                 root_scores: np.ndarray, result_weight: float, visit_temperature: float = VISIT_TEMPERATURE,
                 worst_moves: np.ndarray | None = None, worst_losses: np.ndarray | None = None, blunder_threshold: float = BLUNDER_THRESHOLD):
        if not 0 <= result_weight <= 1:
            raise ValueError(f"the result weight is a share from 0 to 1, got {result_weight}")
        if visit_temperature < 0:
            raise ValueError(f"the visit temperature is 0 or more, got {visit_temperature}")
        if visit_moves.shape != visit_shares.shape or not len(boards) == len(visit_moves) == len(outcomes) == len(root_scores):
            raise ValueError(f"{len(boards)} boards, visit moves {visit_moves.shape}, shares {visit_shares.shape}, "
                             f"{len(outcomes)} outcomes and {len(root_scores)} root scores do not belong together")
        super().__init__(boards, worst_moves, worst_losses, blunder_threshold)

        self.visit_moves = shared_tensor(visit_moves)
        self.visit_shares = shared_tensor(visit_shares)
        self.outcomes = shared_tensor(outcomes)
        self.root_scores = shared_tensor(root_scores)
        self.result_weight = result_weight
        self.visit_temperature = visit_temperature

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return self.boards, self.visit_moves, self.visit_shares, self.outcomes, self.root_scores, *self.blunder_tensors()

    def value_target_of(self, index: int | list[int]) -> torch.Tensor:
        return self.result_weight * self.outcomes[index] + (1 - self.result_weight) * self.root_scores[index]

    def policy_target(self, index: int | list[int], mirrored: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        moves = mirror_moves(self.visit_moves[index].long(), mirrored)
        no_move = moves == NO_MOVE
        shares = self.visit_shares[index].masked_fill(no_move, 0.0)
        if self.visit_temperature != 1:
            shares = sharpened(shares, self.visit_temperature)
        return moves.masked_fill(no_move, 0), shares


#shares ** (1 / temperature), renormalised; divided by the max first so a low temperature does not underflow, and at 0 only the most visited move(s) remain
def sharpened(shares: torch.Tensor, temperature: float) -> torch.Tensor:
    relative = shares / shares.amax(dim=-1, keepdim=True).clamp_min(1e-30)
    weights = (relative == 1).float() if temperature == 0 else relative ** (1 / temperature)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-30)


#one pass over the main loader is one epoch; the mix loader's position carries over between passes
class MixedLoader:
    def __init__(self, main_loader: DataLoader, mix_loader: DataLoader):
        self.main_loader = main_loader
        self.mix_loader = mix_loader
        self.dataset = main_loader.dataset
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
    return torch.cat(boards), pad_and_cat(moves), pad_and_cat(probabilities), torch.cat(values), pad_and_cat(blunders, NO_MOVE)

def pad_and_cat(parts: tuple[torch.Tensor, ...], value: int = 0) -> torch.Tensor:
    width = max(part.shape[1] for part in parts)
    return torch.cat([torch.nn.functional.pad(part, (0, width - part.shape[1]), value=value) for part in parts])


class MirroringBatches(Sampler):
    def __init__(self, batches: Sampler, seed: int):
        self.batches = batches
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        epoch, self.epoch = self.epoch, self.epoch + 1
        for rows in self.batches:
            yield draw_mirrors(torch.tensor(rows), self.seed, epoch)


def datasets_to_dataloaders(
    train_dataset: TrainingDataset,
    val_dataset: TrainingDataset,
    train_batch_size: int = 256,
    val_batch_size: int = 512,
    seed: int | None = None,
    mirror: bool = False
) -> tuple[DataLoader, DataLoader]:
    if mirror and seed is None:
        raise ValueError("mirroring draws its coins from the seed")

    #a generator of its own keeps the train order independent of how many random numbers building the model drew
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    train_sampler = BatchSampler(RandomSampler(train_dataset, generator=generator), batch_size=train_batch_size, drop_last=False)
    if mirror:
        train_sampler = MirroringBatches(train_sampler, seed)
    val_sampler = BatchSampler(SequentialSampler(val_dataset), batch_size=val_batch_size, drop_last=False)

    #the samplers yield whole batches of indices, so the loaders do no batching of their own
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=None)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=None)

    return train_loader, val_loader

def load_training_set(name: str, target_temperature: float, result_weight: float = RESULT_WEIGHT,
                      visit_temperature: float = VISIT_TEMPERATURE, blunder_threshold: float = BLUNDER_THRESHOLD,
                      teacher_weight: float = 0.0) -> tuple[TrainingDataset, TrainingDataset]:
    def path(split: str, kind: str) -> Path:
        return PROCESSED_DATA_DIR / f"{name}_{split}_{kind}.bin"

    #mode "r": copy on write ("c") makes windows reserve memory for the whole file
    def read(split: str, kind: str, dtype: type) -> np.memmap:
        return np.memmap(path(split, kind), dtype=dtype, mode="r")

    datasets = []
    for split in ("train", "val"):
        boards_path = path(split, "boards")
        if not boards_path.exists():
            raise FileNotFoundError(f"{boards_path.name} is missing from {PROCESSED_DATA_DIR}")
        boards = read(split, "boards", np.uint8).reshape(-1, TOKENS_PER_BOARD)
        worst = {}
        if path(split, "worst_moves").exists():
            worst = {kind: read(split, kind, dtype).reshape(len(boards), -1) for kind, dtype in WORST_FILES.items()}

        if path(split, "visit_moves").exists():
            if teacher_weight:
                raise ValueError(f"{name} is a self-play set; teacher targets are only read for stockfish-labelled sets")
            arrays = {kind: read(split, kind, dtype) for kind, dtype in SELF_PLAY_FILES.items()}
            datasets.append(SelfPlayDataset(boards, arrays["visit_moves"].reshape(len(boards), -1), arrays["visit_shares"].reshape(len(boards), -1),
                                            arrays["outcomes"], arrays["root_scores"], result_weight, visit_temperature,
                                            worst.get("worst_moves"), worst.get("worst_losses"), blunder_threshold))
            continue

        stockfish_moves_path = path(split, "stockfish_moves")
        if not stockfish_moves_path.exists():
            raise FileNotFoundError(f"{stockfish_moves_path.name} is missing: a training set needs its stockfish_moves file")
        #the file size says how many top moves per position
        stockfish_moves = read(split, "stockfish_moves", np.int16)
        stockfish_evals = read(split, "stockfish_evals", np.int16)
        if len(stockfish_moves) == 0 or len(stockfish_moves) % len(boards) != 0:
            raise ValueError(f"{len(stockfish_moves):,} stockfish moves cannot be shared between {len(boards):,} positions")
        teacher = None
        #val keeps stockfish's targets alone, so its loss compares across runs
        weight = teacher_weight if split == "train" else 0.0
        if weight:
            missing = [kind for kind in TEACHER_FILES if not path(split, kind).exists()]
            if missing:
                raise FileNotFoundError(f"{name}_{split}: no {', '.join(missing)}, a teacher weight needs them")
            teacher = {kind: read(split, kind, dtype) for kind, dtype in TEACHER_FILES.items()}
            for kind in ("teacher_moves", "teacher_probabilities"):
                teacher[kind] = teacher[kind].reshape(len(boards), -1)
        datasets.append(ChessGamesDataset(boards, stockfish_moves.reshape(len(boards), -1), stockfish_evals.reshape(len(boards), -1), target_temperature,
                                          worst.get("worst_moves"), worst.get("worst_losses"), blunder_threshold, teacher, weight))

    return datasets[0], datasets[1]

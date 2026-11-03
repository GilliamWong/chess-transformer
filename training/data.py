from pathlib import Path
from model.checkpoint import ARTIFACTS_DIR
from model.encoding import EXPECTED_SCORE_PER_CENTIPAWN, NO_MOVE, TOKENS_PER_BOARD
from training.mirror import MirrorRows, draw_mirrors, mirror_moves, read_boards
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, Sampler, SequentialSampler
import numpy as np
import torch
import warnings

PROCESSED_DATA_DIR = ARTIFACTS_DIR / "processed_data"

#self play value target: this much of the game's result, the rest the search's root score
RESULT_WEIGHT = 0.5
#self play policy target: visit shares raised to 1 / VISIT_TEMPERATURE and renormalised (0 = most visited move alone)
VISIT_TEMPERATURE = 1.0

#a labelled move that gives away at least this much expected score against stockfish's best is a blunder
BLUNDER_THRESHOLD = 0.3
#optional blunder labels next to a set, both (positions, slots) with NO_MOVE where empty
WORST_FILES = {"worst_moves": np.int16, "worst_losses": np.float16}
#optional teacher targets: top moves (positions, slots), their probabilities (adding up to 1) and its expected score (positions,)
TEACHER_FILES = {"teacher_moves": np.int16, "teacher_probabilities": np.float16, "teacher_scores": np.float16}
#selfplay.py's files: visit moves (positions, top moves) with NO_MOVE where fewer, shares adding up to 1, and the side to
#move's outcome and root score from 0 to 1
SELF_PLAY_FILES = {"visit_moves": np.int16, "visit_shares": np.float32, "outcomes": np.float32, "root_scores": np.float32}

#the arrays may be read-only memory maps that nothing here writes to, so torch's warning is noise
def shared_tensor(array: np.ndarray) -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
        return torch.from_numpy(array)


#boards and the optional blunder labels (both arrays or neither, a row per position); subclasses add the other targets
class TrainingDataset(Dataset):
    def __init__(self, boards: np.ndarray, worst_moves: np.ndarray | None, worst_losses: np.ndarray | None, blunder_threshold: float):
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

    #labelled moves at or above the blunder threshold, NO_MOVE elsewhere; one NO_MOVE column without blunder labels.
    #mirrored is a bool per position, see training.mirror
    def blunders_of(self, index: int | list[int], mirrored: torch.Tensor | None = None) -> torch.Tensor:
        if self.worst_moves is None:
            return torch.full_like(self.boards[index, :1], NO_MOVE, dtype=torch.long)
        #compared in float16: 0.2 is stored as 0.19995 and still counts as at the threshold
        below = self.worst_losses[index] < torch.tensor(self.blunder_threshold, dtype=self.worst_losses.dtype)
        return mirror_moves(self.worst_moves[index].long().masked_fill(below, NO_MOVE), mirrored)

    #index is one position, a list of positions returned as one batch, or such a batch with a coin for each position to
    #show it mirrored (training.mirror): boards, target moves, target probabilities, value target, blunders
    def __getitem__(self, index: int | list[int] | MirrorRows) -> tuple[torch.Tensor, ...]:
        index, boards, mirrored = read_boards(self.boards, index)
        target_moves, target_probabilities = self.policy_target(index, mirrored)
        return boards, target_moves, target_probabilities, self.value_target_of(index), self.blunders_of(index, mirrored)


#stockfish's best moves and the evals after each, (positions, top moves) best first. teacher_weight of the policy and
#value targets comes from the teacher's TEACHER_FILES arrays, the rest from stockfish
class ChessGamesDataset(TrainingDataset):
    def __init__(self, boards: np.ndarray, stockfish_moves: np.ndarray, stockfish_evals: np.ndarray, target_temperature: float,
                 worst_moves: np.ndarray | None = None, worst_losses: np.ndarray | None = None, blunder_threshold: float = BLUNDER_THRESHOLD,
                 teacher: dict[str, np.ndarray] | None = None, teacher_weight: float = 0.0):
        if target_temperature <= 0:
            raise ValueError(f"the target temperature must be above 0, got {target_temperature}")
        if not 0 <= teacher_weight <= 1:
            raise ValueError(f"the teacher's share of the targets is between 0 and 1, got {teacher_weight}")
        if teacher_weight and teacher is None:
            raise ValueError("a teacher weight needs the teacher's targets (tools/teacher_targets.py)")
        if teacher is not None and not (len(teacher["teacher_moves"]) == len(teacher["teacher_probabilities"]) == len(teacher["teacher_scores"]) == len(boards)):
            raise ValueError(f"the teacher's targets do not cover the {len(boards):,} positions one each")
        if blunder_threshold <= 0:
            raise ValueError(f"the blunder threshold is an expected score lost, above 0, got {blunder_threshold}")
        if stockfish_moves.shape != stockfish_evals.shape or len(stockfish_moves) != len(boards):
            raise ValueError(f"stockfish moves {stockfish_moves.shape} and evals {stockfish_evals.shape} for {len(boards)} positions")
        #an unlabelled position would come out as a confident target for move 0
        unlabelled = int((stockfish_moves[:, 0] == NO_MOVE).sum())
        if unlabelled:
            raise ValueError(f"{unlabelled:,} of {len(boards):,} positions are not labelled yet, tools.label_with_stockfish finishes them")
        super().__init__(boards, worst_moves, worst_losses, blunder_threshold)

        self.stockfish_moves = shared_tensor(stockfish_moves)
        self.stockfish_evals = shared_tensor(stockfish_evals)
        self.target_temperature = target_temperature
        self.teacher = {kind: shared_tensor(array) for kind, array in teacher.items()} if teacher_weight else None
        self.teacher_weight = teacher_weight

    #everything the set reads from disk
    def tensors(self) -> tuple[torch.Tensor, ...]:
        blunder_labels = (self.worst_moves, self.worst_losses) if self.worst_moves is not None else ()
        teacher = tuple(self.teacher.values()) if self.teacher is not None else ()
        return self.boards, self.stockfish_moves, self.stockfish_evals, *blunder_labels, *teacher

    #stockfish's eval of the position (first column) as an expected score from 0 to 1, mixed with the teacher's
    def value_target_of(self, index: int | list[int]) -> torch.Tensor:
        stockfish = torch.sigmoid(EXPECTED_SCORE_PER_CENTIPAWN * self.stockfish_evals[index, 0].float())
        if self.teacher is None:
            return stockfish
        return (1 - self.teacher_weight) * stockfish + self.teacher_weight * self.teacher["teacher_scores"][index].float()

    #(moves, probabilities): a softmax over the expected score after each move / target_temperature; empty places are
    #move 0 at probability 0. A teacher's moves follow stockfish's: a move in both lists gets both shares, which the
    #cross entropy adds up
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


#selfplay.py's positions: the policy learns the search's visit shares, the value head a blend of result and root score
class SelfPlayDataset(TrainingDataset):
    def __init__(self, boards: np.ndarray, visit_moves: np.ndarray, visit_shares: np.ndarray, outcomes: np.ndarray,
                 root_scores: np.ndarray, result_weight: float, visit_temperature: float = VISIT_TEMPERATURE,
                 worst_moves: np.ndarray | None = None, worst_losses: np.ndarray | None = None, blunder_threshold: float = BLUNDER_THRESHOLD):
        if not 0 <= result_weight <= 1:
            raise ValueError(f"the result weight is a share from 0 to 1, got {result_weight}")
        if visit_temperature < 0:
            raise ValueError(f"the visit temperature is 0 or more, got {visit_temperature}")
        if blunder_threshold <= 0:
            raise ValueError(f"the blunder threshold is an expected score lost, above 0, got {blunder_threshold}")
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
        blunder_labels = (self.worst_moves, self.worst_losses) if self.worst_moves is not None else ()
        return self.boards, self.visit_moves, self.visit_shares, self.outcomes, self.root_scores, *blunder_labels

    def value_target_of(self, index: int | list[int]) -> torch.Tensor:
        return self.result_weight * self.outcomes[index] + (1 - self.result_weight) * self.root_scores[index]

    #empty places are move 0 at share 0
    def policy_target(self, index: int | list[int], mirrored: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        moves = mirror_moves(self.visit_moves[index].long(), mirrored)
        no_move = moves == NO_MOVE
        shares = self.visit_shares[index].masked_fill(no_move, 0.0)
        if self.visit_temperature != 1:
            shares = sharpened(shares, self.visit_temperature)
        return moves.masked_fill(no_move, 0), shares


#shares ** (1 / temperature), renormalised. Divided by the max first so a low temperature does not underflow to 0;
#at 0 the most visited move(s) alone
def sharpened(shares: torch.Tensor, temperature: float) -> torch.Tensor:
    relative = shares / shares.amax(dim=-1, keepdim=True).clamp_min(1e-30)
    weights = (relative == 1).float() if temperature == 0 else relative ** (1 / temperature)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-30)


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
    return torch.cat(boards), pad_and_cat(moves), pad_and_cat(probabilities), torch.cat(values), pad_and_cat(blunders, NO_MOVE)

def pad_and_cat(parts: tuple[torch.Tensor, ...], value: int = 0) -> torch.Tensor:
    width = max(part.shape[1] for part in parts)
    return torch.cat([torch.nn.functional.pad(part, (0, width - part.shape[1]), value=value) for part in parts])


#another sampler's batches, each with the coins that show its rows mirrored at random in this pass over the set; the
#epoch is counted here by the passes started, train.py's loader having none of its own
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


#the samplers hand the dataset a whole batch of indices (batch_size=None turns off the loader's own batching).
#mirror shows the train positions without castling rights mirrored at random (training.mirror), a new draw every pass
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

    #with a seed the train order has its own generator, so it doesn't depend on how many random numbers building the
    #model drew
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    train_sampler = BatchSampler(RandomSampler(train_dataset, generator=generator), batch_size=train_batch_size, drop_last=False)
    if mirror:
        train_sampler = MirroringBatches(train_sampler, seed)
    val_sampler = BatchSampler(SequentialSampler(val_dataset), batch_size=val_batch_size, drop_last=False)

    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=None)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=None)

    return train_loader, val_loader

#opens <name>_{train,val}_*.bin from processed_data/ as memory mapped datasets: stockfish labelled sets as
#ChessGamesDataset, self play sets (SELF_PLAY_FILES) as SelfPlayDataset. teacher_weight > 0 mixes the teacher's targets
#(TEACHER_FILES) into the train split; val keeps stockfish's alone, so val loss compares across runs
def load_training_set(name: str, target_temperature: float, result_weight: float = RESULT_WEIGHT,
                      visit_temperature: float = VISIT_TEMPERATURE, blunder_threshold: float = BLUNDER_THRESHOLD,
                      teacher_weight: float = 0.0) -> tuple[TrainingDataset, TrainingDataset]:
    def path(split: str, kind: str) -> Path:
        return PROCESSED_DATA_DIR / f"{name}_{split}_{kind}.bin"

    #raw arrays without a header. Mode "r": copy on write ("c") makes windows reserve memory for the whole file
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
            raise FileNotFoundError(f"{stockfish_moves_path.name} is missing, tools.label_with_stockfish writes it")
        #the file size says how many top moves per position
        stockfish_moves = read(split, "stockfish_moves", np.int16)
        stockfish_evals = read(split, "stockfish_evals", np.int16)
        if len(stockfish_moves) == 0 or len(stockfish_moves) % len(boards) != 0:
            raise ValueError(f"{len(stockfish_moves):,} stockfish moves cannot be shared between {len(boards):,} positions")
        teacher = None
        weight = teacher_weight if split == "train" else 0.0
        if weight:
            missing = [kind for kind in TEACHER_FILES if not path(split, kind).exists()]
            if missing:
                raise FileNotFoundError(f"{name}_{split}: no {', '.join(missing)}, tools.teacher_targets writes them")
            teacher = {kind: read(split, kind, dtype) for kind, dtype in TEACHER_FILES.items()}
            for kind in ("teacher_moves", "teacher_probabilities"):
                teacher[kind] = teacher[kind].reshape(len(boards), -1)
        datasets.append(ChessGamesDataset(boards, stockfish_moves.reshape(len(boards), -1), stockfish_evals.reshape(len(boards), -1), target_temperature,
                                          worst.get("worst_moves"), worst.get("worst_losses"), blunder_threshold, teacher, weight))

    return datasets[0], datasets[1]

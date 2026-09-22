from pathlib import Path
from dataclasses import dataclass
from encoding import NUM_PIECE_TYPES, NUM_SQUARES, TOKENS_PER_BOARD, encode_board, encode_move_to_id
from tools.label_with_stockfish import NO_MOVE
from collections.abc import Iterator
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, Sampler, SequentialSampler
import numpy as np
import torch
import torch.nn.functional as F
import warnings
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

#what the policy learns to predict: the move that was played, or stockfish's best moves (tools/label_with_stockfish.py)
POLICY_TARGETS = ("human", "stockfish")

#the future target of a square whose future is not in the data: the game ended first. nn.CrossEntropyLoss skips it
FUTURE_UNKNOWN = -100

#centipawns to the expected score of the game (0 lost, 1 won): sigmoid(EXPECTED_SCORE_PER_CENTIPAWN * centipawns),
#the curve lichess fitted to its games. A pawn is worth a lot at 0 and next to nothing at +800, which centipawns do not show
EXPECTED_SCORE_PER_CENTIPAWN = 0.00368208

class ChessGamesDataset(Dataset):
    """set of examples of chess games that we can use as a dataset"""

    #history is how many past plies every item describes, future how many plies ahead its future target looks, see __getitem__
    def __init__(self, games: list[Game], history: int = 0, future: int = 0):
        self.history = history
        self.future = future
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

        #only sets that stockfish has labelled have these (with_stockfish_targets), the policy target is then made from them
        self.stockfish_moves = None
        self.stockfish_evals = None
        self.target_temperature = None

    #for positions that are already arrays (load_training_set). They can be memory mapped and far too
    #many for a python list of games: the tensors share the arrays' memory, nothing is copied
    @classmethod
    def from_arrays(cls, boards: np.ndarray, moves: np.ndarray, results: np.ndarray, plies: np.ndarray | None = None, history: int = 0, future: int = 0) -> "ChessGamesDataset":
        if not len(boards) == len(moves) == len(results):
            raise ValueError(f"{len(boards)} boards but {len(moves)} moves and {len(results)} results")
        if (history > 0 or future > 0) and (plies is None or len(plies) != len(boards)):
            raise ValueError("history and the future target need the ply of every position, to know which rows are the same game")

        dataset = cls([], history, future)
        #the arrays may be read-only memory maps. torch warns that writing through the tensor would be undefined,
        #and nothing here ever writes to them
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            dataset.boards = torch.from_numpy(boards)
            dataset.moves = torch.from_numpy(moves)
            dataset.results = torch.from_numpy(results)
            if plies is not None:
                dataset.plies = torch.from_numpy(plies)

        return dataset

    #from here on the policy target is stockfish's best moves and not the move played. moves and evals are
    #(positions, top moves) the way tools/label_with_stockfish.py writes them, the best move first
    def with_stockfish_targets(self, moves: np.ndarray, evals: np.ndarray, target_temperature: float) -> "ChessGamesDataset":
        if moves.shape != evals.shape or len(moves) != len(self):
            raise ValueError(f"stockfish moves {moves.shape} and evals {evals.shape} for {len(self)} positions")
        #an unlabelled position would come out as a confident target for move 0
        unlabelled = int((moves[:, 0] == NO_MOVE).sum())
        if unlabelled:
            raise ValueError(f"{unlabelled:,} of {len(self):,} positions are not labelled yet, tools.label_with_stockfish finishes them")
        if target_temperature <= 0:
            raise ValueError(f"the target temperature must be above 0, got {target_temperature}")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            self.stockfish_moves = torch.from_numpy(moves)
            self.stockfish_evals = torch.from_numpy(evals)
        self.target_temperature = target_temperature
        return self

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

    #the future head's target: what the board future plies ahead changes on every square, in the codes of history_of
    #(0 unchanged, else 1 + what will be there), seen from the current side to move. A game's last positions have no
    #board that far ahead and get FUTURE_UNKNOWN, which the loss ignores. (positions, NUM_SQUARES)
    def future_of(self, rows: torch.Tensor) -> torch.Tensor:
        if self.future == 0:
            return torch.zeros((len(rows), NUM_SQUARES), dtype=torch.long)

        #the game goes on for at least future more rows if the row that far ahead is that many plies further into a game
        ahead = (rows + self.future).clamp(max=len(self) - 1)
        known = (self.plies[ahead] == self.plies[rows] + self.future) & (rows + self.future < len(self))
        now = self.squares(rows, flip=False)
        later = self.squares(ahead, flip=self.future % 2 == 1)
        changes = torch.where(later != now, 1 + later, 0)
        return changes.masked_fill(~known.unsqueeze(1), FUTURE_UNKNOWN)

    #boards, the policy target (moves and their probabilities, see policy_target), results, history, the future target.
    #the result comes back as the class the value head predicts: 0 lost, 1 draw, 2 won
    def __getitem__(self, index: int | list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = torch.as_tensor(index)
        history = self.history_of(rows.reshape(-1))
        future = self.future_of(rows.reshape(-1))
        if rows.ndim == 0:
            history, future = history[0], future[0]
        target_moves, target_probabilities = self.policy_target(index)
        return self.boards[index].long(), target_moves, target_probabilities, self.results[index].long() + 1, history, future


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

        #a set with the moves that were played has one target move per position, a set with stockfish's has several:
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
    history: int = 0,
    future: int = 0
) -> tuple[DataLoader, DataLoader]:

    train_data, val_data = split_data(games, train_fraction=train_fraction, seed=seed)

    train_dataset = ChessGamesDataset(train_data, history, future)
    val_dataset = ChessGamesDataset(val_data, history, future)

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
#with history the items also describe that many past plies, and with future what changes that many plies ahead; both
#need <name>_{train,val}_plies.bin
#
#policy_target "stockfish" makes the policy target stockfish's best moves, for the sets that have them
#(<name>_{train,val}_stockfish_{moves,evals}.bin, written by tools/label_with_stockfish.py). In a mix the other sets keep
#the move that was played: fishtest's are stockfish's own moves already. Which set got which target is printed
def load_training_set(name: str, history: int = 0, policy_target: str = "human", target_temperature: float | None = None, future: int = 0) -> tuple[ChessGamesDataset | MixedDataset, ChessGamesDataset]:
    if policy_target not in POLICY_TARGETS:
        raise ValueError(f"policy target {policy_target} is not one of {POLICY_TARGETS}")

    if "+" in name:
        #a set that is named several times is opened once and gets that many shares of every batch
        names = name.split("+")
        targets = {one_name: "human" for one_name in names}
        if policy_target == "stockfish":
            targets = {one_name: "stockfish" if stockfish_moves_path(one_name, "train").exists() else "human" for one_name in names}
            if "stockfish" not in targets.values():
                raise FileNotFoundError(f"none of {names} has stockfish labels, tools.label_with_stockfish writes them")
            print(f"policy targets: {targets}")
        loaded = {one_name: load_training_set(one_name, history, targets[one_name], target_temperature, future) for one_name in targets}
        return MixedDataset([loaded[one_name][0] for one_name in names]), loaded[names[0]][1]

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
        #mode "r", read-only. Copy on write ("c") looks harmless but makes windows reserve memory for the whole file:
        #with a set named three times an overnight run held 35 GB of reservations for 12 GB of data
        boards = np.memmap(boards_path, dtype=np.uint8, mode="r").reshape(-1, TOKENS_PER_BOARD)
        moves = np.memmap(moves_path, dtype=np.int16, mode="r")
        results = np.memmap(results_path, dtype=np.int8, mode="r")

        plies = None
        if history > 0 or future > 0:
            plies_path = PROCESSED_DATA_DIR / f"{name}_{split}_plies.bin"
            if not plies_path.exists():
                raise FileNotFoundError(f"{plies_path.name} is missing, scripts_used/add_plies.py writes it")
            plies = np.memmap(plies_path, dtype=np.int16, mode="r")
        dataset = ChessGamesDataset.from_arrays(boards, moves, results, plies, history, future)

        if policy_target == "stockfish":
            if target_temperature is None:
                raise ValueError("stockfish targets need a target temperature")
            if not stockfish_moves_path(name, split).exists():
                raise FileNotFoundError(f"{stockfish_moves_path(name, split).name} is missing, tools.label_with_stockfish writes it")
            #(positions, top moves), and the size of the file says how many top moves that is
            stockfish_moves = np.memmap(stockfish_moves_path(name, split), dtype=np.int16, mode="r")
            stockfish_evals = np.memmap(PROCESSED_DATA_DIR / f"{name}_{split}_stockfish_evals.bin", dtype=np.int16, mode="r")
            if len(stockfish_moves) == 0 or len(stockfish_moves) % len(dataset) != 0:
                raise ValueError(f"{len(stockfish_moves):,} stockfish moves cannot be shared between {len(dataset):,} positions")
            dataset.with_stockfish_targets(stockfish_moves.reshape(len(dataset), -1), stockfish_evals.reshape(len(dataset), -1), target_temperature)
        datasets.append(dataset)

    return datasets[0], datasets[1]


def stockfish_moves_path(name: str, split: str) -> Path:
    return PROCESSED_DATA_DIR / f"{name}_{split}_stockfish_moves.bin"

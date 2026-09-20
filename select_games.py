#picks games out of the shards preprocess.py made and copies their positions into one training set.
#
#   python select_games.py
#
#the choice of games is the part worth thinking about, and it is all in the constants below and in
#choose_games. Writing a different training set means changing those and giving it another NAME:
#the shards hold every game, so this takes minutes and never needs the pgns again.
#
#a training set is four files in processed_data/: <NAME>_train_boards.bin, <NAME>_train_moves.bin
#and the same two for val. load_training_set opens them memory mapped.
import numpy as np
from tqdm import tqdm
from games import LICHESS_ELITE_PGNS
from encoding import TOKENS_PER_BOARD
from preprocess import BOARD_DTYPE, MOVE_DTYPE, PROCESSED_DATA_DIR, load_shard, shard_paths

NAME = "baseline"

CARLSEN_POSITIONS = 724_344
TARGET_POSITIONS = 100 * CARLSEN_POSITIONS #100 times what the first model was trained on

#the whole set is tens of millions of positions, so a small share is already a precise measurement
#and every extra val position is one the model does not get to learn from
VAL_FRACTION = 0.02
SEED = 42

#lichess sorts games by how long they are expected to last: base time + 40 moves of increment
BLITZ_SECONDS = 180   #under this is bullet
RAPID_SECONDS = 480   #under this is blitz
#rapid and classical are everything from RAPID_SECONDS up

def estimated_seconds(games: np.ndarray) -> np.ndarray:
    return games["base_seconds"].astype(np.int64) + 40 * games["increment_seconds"].astype(np.int64)

#returns a True/False per game. Games lost on time are never used: the moves before a flag fell were
#played with seconds on the clock. Slow games are preferred, faster ones are only added if the slow
#ones alone do not reach TARGET_POSITIONS
def choose_games(games: np.ndarray) -> np.ndarray:
    has_clock = games["base_seconds"] >= 0
    finished_normally = ~games["time_forfeit"]
    usable = has_clock & finished_normally

    rapid_or_classical = usable & (estimated_seconds(games) >= RAPID_SECONDS)
    if games["num_positions"][rapid_or_classical].sum() >= TARGET_POSITIONS:
        print("rapid and classical games alone are enough")
        return rapid_or_classical

    print("not enough rapid and classical games, using every time control")
    return usable

#every game of every shard in one table, with the shard it came from
def load_all_games() -> tuple[np.ndarray, np.ndarray]:
    tables = []
    shard_numbers = []
    for shard_number, pgn_path in enumerate(LICHESS_ELITE_PGNS):
        if not shard_paths(pgn_path)[2].exists():
            raise FileNotFoundError(f"{pgn_path.name} has not been preprocessed, run preprocess.py first")
        games = np.load(shard_paths(pgn_path)[2])
        tables.append(games)
        shard_numbers.append(np.full(len(games), shard_number, dtype=np.int32))

    return np.concatenate(tables), np.concatenate(shard_numbers)

def print_summary(games: np.ndarray) -> None:
    seconds = estimated_seconds(games)
    categories = {
        "bullet": (seconds >= 0) & (seconds < BLITZ_SECONDS),
        "blitz": (seconds >= BLITZ_SECONDS) & (seconds < RAPID_SECONDS),
        "rapid and classical": seconds >= RAPID_SECONDS,
        "no clock": games["base_seconds"] < 0,
    }
    print(f"{len(games):,} games, {games['num_positions'].sum():,} positions, {games['time_forfeit'].mean():.0%} lost on time")
    for category, in_category in categories.items():
        kept = in_category & ~games["time_forfeit"]
        print(f"    {category:20}: {in_category.sum():>11,} games, {games['num_positions'][kept].sum():>14,} positions not lost on time")

#copies the positions of the chosen games into one pair of files. Games are read shard by shard and in
#file order, which is far faster than jumping around 180 GB. The loader shuffles, so the order does not matter
def write_positions(games: np.ndarray, shard_numbers: np.ndarray, split: str) -> int:
    boards_path = PROCESSED_DATA_DIR / f"{NAME}_{split}_boards.bin"
    moves_path = PROCESSED_DATA_DIR / f"{NAME}_{split}_moves.bin"

    with open(boards_path, "wb") as boards_file, open(moves_path, "wb") as moves_file:
        for shard_number in tqdm(np.unique(shard_numbers), desc=split):
            boards, moves, _ = load_shard(LICHESS_ELITE_PGNS[shard_number])
            shard_games = np.sort(games[shard_numbers == shard_number], order="first_position")
            for game in shard_games:
                first, last = game["first_position"], game["first_position"] + game["num_positions"]
                boards_file.write(boards[first:last].tobytes())
                moves_file.write(moves[first:last].tobytes())

    return int(games["num_positions"].sum())

#(train boards, train moves, val boards, val moves), memory mapped
def load_training_set(name: str = NAME) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = []
    for split in ("train", "val"):
        boards_path = PROCESSED_DATA_DIR / f"{name}_{split}_boards.bin"
        if not boards_path.exists():
            raise FileNotFoundError(f"{boards_path.name} does not exist, run select_games.py first")
        #mode "c" is copy on write: nothing can change the files, but torch.from_numpy gets the writable array it asks for
        boards = np.memmap(boards_path, dtype=BOARD_DTYPE, mode="c").reshape(-1, TOKENS_PER_BOARD)
        moves = np.memmap(PROCESSED_DATA_DIR / f"{name}_{split}_moves.bin", dtype=MOVE_DTYPE, mode="c")
        if len(boards) != len(moves):
            raise ValueError(f"{name} {split} has {len(boards)} boards but {len(moves)} moves, run select_games.py again")
        arrays += [boards, moves]

    return tuple(arrays)

def main():
    games, shard_numbers = load_all_games()
    print_summary(games)

    chosen = np.flatnonzero(choose_games(games))

    #a random sample of the chosen games, just large enough to reach the target
    random_generator = np.random.default_rng(SEED)
    chosen = random_generator.permutation(chosen)
    positions_so_far = np.cumsum(games["num_positions"][chosen])
    chosen = chosen[:np.searchsorted(positions_so_far, TARGET_POSITIONS) + 1]

    #split by game, never by position, for the same reason as split_data: the order is already random
    num_val_games = int(len(chosen) * VAL_FRACTION)
    if num_val_games == 0:
        raise ValueError(f"only {len(chosen)} games were chosen, too few to set {VAL_FRACTION:.0%} aside for val")
    val_games, train_games = chosen[:num_val_games], chosen[num_val_games:]

    for split, split_games in (("train", train_games), ("val", val_games)):
        num_positions = write_positions(games[split_games], shard_numbers[split_games], split)
        print(f"{split}: {len(split_games):,} games, {num_positions:,} positions")


if __name__ == "__main__":
    main()

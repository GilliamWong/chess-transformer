#turns the lichess elite pgns into arrays on disk, once, so training never has to parse a pgn again.
#parsing all 32 million games takes about 18 hours on one core, so every month goes to its own process.
#
#   python preprocess.py
#
#every month becomes one shard of three files in processed_data/:
#   <month>_boards.bin   uint8, shape (positions, TOKENS_PER_BOARD): encode_board of every position
#   <month>_moves.bin    int16, shape (positions,): encode_move_to_id of the move played from it
#   <month>_games.npy    one row per game (GAME_FIELDS): where its positions are, and what kind of game it was
#the .bin files are raw arrays with no header, their shape comes from the file size (see load_shard).
#
#every game is kept, including the ones a training run might not want. Choosing games (time control,
#rating, no time forfeits) is done afterwards from the games table, which takes seconds instead of hours.
#a month that is already done is skipped, so new months can be added and an interrupted run can be resumed.
import logging
import multiprocessing
import os
from pathlib import Path
import chess.pgn
import numpy as np
from tqdm import tqdm
from games import LICHESS_ELITE_PGNS, game_to_examples
from encoding import TOKENS_PER_BOARD

PROCESSED_DATA_DIR = Path(__file__).parent / "processed_data"

BOARD_DTYPE = np.uint8 #board tokens go up to 80
MOVE_DTYPE = np.int16  #move ids go up to 1967

GAME_FIELDS = np.dtype([
    ("first_position", np.int64),   #index of the game's first position in the shard's boards and moves
    ("num_positions", np.int32),
    ("white_elo", np.int16),
    ("black_elo", np.int16),
    ("base_seconds", np.int32),     #time control "180+2" is base 180 and increment 2, -1 when it is not given
    ("increment_seconds", np.int16),
    ("time_forfeit", np.bool_),     #the game ended because a clock ran out, so the last moves were likely rushed
    ("result", np.int8),            #1 white won, 0 draw, -1 black won. Not used yet, a value head would need it
])

RESULTS = {"1-0": 1, "1/2-1/2": 0, "0-1": -1}

NUM_PROCESSES = max(1, (os.cpu_count() or 2) - 4) #leaves a few cores free so the machine stays usable

def shard_paths(pgn_path: Path) -> tuple[Path, Path, Path]:
    month = pgn_path.stem
    return (
        PROCESSED_DATA_DIR / f"{month}_boards.bin",
        PROCESSED_DATA_DIR / f"{month}_moves.bin",
        PROCESSED_DATA_DIR / f"{month}_games.npy"
    )

#"180+2" -> (180, 2). Lichess writes "-" for games without a clock
def parse_time_control(time_control: str) -> tuple[int, int]:
    base, separator, increment = time_control.partition("+")
    if not (separator and base.isdigit() and increment.isdigit()):
        return -1, -1
    return int(base), int(increment)

def parse_elo(elo: str) -> int:
    return int(elo) if elo.isdigit() else -1

#games we cannot train on: not standard chess, no result, or a move python-chess could not read
def is_usable(game: chess.pgn.Game) -> bool:
    headers = game.headers
    return (
        headers.get("Variant", "Standard") == "Standard"
        and "FEN" not in headers
        and headers.get("Result") in RESULTS
        and len(game.errors) == 0
    )

#parses one month and writes its shard. Returns (games, positions) written.
#positions are appended to the files game by game, so a worker never holds more than one game in memory
def preprocess_month(pgn_path: Path) -> tuple[int, int]:
    #an unreadable move is recorded in game.errors and the game is skipped, the log line adds nothing
    logging.getLogger("chess.pgn").setLevel(logging.CRITICAL)

    boards_path, moves_path, games_path = shard_paths(pgn_path)
    #written under temporary names and renamed at the end, so a shard that exists is always a complete one
    temporary_boards = boards_path.with_suffix(".tmp")
    temporary_moves = moves_path.with_suffix(".tmp")

    games = []
    num_positions = 0

    with open(pgn_path, encoding="utf-8") as pgn, open(temporary_boards, "wb") as boards_file, open(temporary_moves, "wb") as moves_file:
        while (game := chess.pgn.read_game(pgn)) is not None:
            if not is_usable(game):
                continue

            examples = game_to_examples(game)
            if len(examples) == 0:
                continue

            boards_file.write(np.array([example.board for example in examples], dtype=BOARD_DTYPE).tobytes())
            moves_file.write(np.array([example.move for example in examples], dtype=MOVE_DTYPE).tobytes())

            base_seconds, increment_seconds = parse_time_control(game.headers.get("TimeControl", "-"))
            games.append((
                num_positions,
                len(examples),
                parse_elo(game.headers.get("WhiteElo", "")),
                parse_elo(game.headers.get("BlackElo", "")),
                base_seconds,
                increment_seconds,
                game.headers.get("Termination") == "Time forfeit",
                RESULTS[game.headers["Result"]]
            ))
            num_positions += len(examples)

    np.save(games_path.with_suffix(".tmp.npy"), np.array(games, dtype=GAME_FIELDS))

    temporary_boards.replace(boards_path)
    temporary_moves.replace(moves_path)
    #the games table is renamed last: it is the file that marks the shard as done
    games_path.with_suffix(".tmp.npy").replace(games_path)

    return len(games), num_positions

#boards and moves come back memory mapped: nothing is read from disk until it is indexed,
#so opening all 145 shards (about 180 GB) costs no memory
def load_shard(pgn_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boards_path, moves_path, games_path = shard_paths(pgn_path)

    games = np.load(games_path)
    boards = np.memmap(boards_path, dtype=BOARD_DTYPE, mode="r").reshape(-1, TOKENS_PER_BOARD)
    moves = np.memmap(moves_path, dtype=MOVE_DTYPE, mode="r")

    if not len(boards) == len(moves) == games["num_positions"].sum():
        raise ValueError(f"shard of {pgn_path.name} is inconsistent, delete its three files and preprocess it again")

    return boards, moves, games

def main():
    PROCESSED_DATA_DIR.mkdir(exist_ok=True)

    remaining_pgns = [pgn_path for pgn_path in LICHESS_ELITE_PGNS if not shard_paths(pgn_path)[2].exists()]
    print(f"{len(LICHESS_ELITE_PGNS)} pgns, {len(LICHESS_ELITE_PGNS) - len(remaining_pgns)} already done, {NUM_PROCESSES} processes")

    #biggest months first, so the run does not end with one process working through the largest file alone
    remaining_pgns.sort(key=lambda pgn_path: pgn_path.stat().st_size, reverse=True)

    total_games = 0
    total_positions = 0
    with multiprocessing.Pool(NUM_PROCESSES) as pool:
        for num_games, num_positions in tqdm(pool.imap_unordered(preprocess_month, remaining_pgns), total=len(remaining_pgns), desc="months"):
            total_games += num_games
            total_positions += num_positions

    print(f"wrote {total_games} games, {total_positions} positions to {PROCESSED_DATA_DIR.name}/")


#required on windows: every worker process imports this file, and without the guard each would start its own pool
if __name__ == "__main__":
    main()

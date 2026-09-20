import io

import chess.pgn
import numpy as np
import pytest

import preprocess
import select_games
from data import ChessGamesDataset
from encoding import TOKENS_PER_BOARD, decode_board, decode_id_to_move
from games import game_to_examples
from preprocess import GAME_FIELDS, load_shard, parse_time_control, preprocess_month


def write_pgn(pgn_path, games) -> None:
    with open(pgn_path, "w", encoding="utf-8") as pgn:
        for game in games:
            print(game, file=pgn, end="\n\n")


def with_headers(game: chess.pgn.Game, **headers: str) -> chess.pgn.Game:
    """a copy of the game with some headers changed, so the fixture's games stay untouched"""
    copy = chess.pgn.read_game(io.StringIO(str(game)))
    for name, value in headers.items():
        copy.headers[name] = value
    return copy


@pytest.fixture
def shard(carlsen_games, tmp_path, monkeypatch):
    """three real games preprocessed into a temporary folder, with the headers lichess games have"""
    monkeypatch.setattr(preprocess, "PROCESSED_DATA_DIR", tmp_path)
    games = [
        with_headers(carlsen_games[0], TimeControl="180+2", Termination="Normal", WhiteElo="2500", BlackElo="2400", Result="1-0"),
        with_headers(carlsen_games[1], TimeControl="600+0", Termination="Time forfeit", Result="0-1"),
        with_headers(carlsen_games[2], TimeControl="-", Termination="Normal", Result="1/2-1/2"),
    ]
    pgn_path = tmp_path / "month.pgn"
    write_pgn(pgn_path, games)

    assert preprocess_month(pgn_path) == (3, sum(len(game_to_examples(game)) for game in games))
    return pgn_path, games


#the shard has to be exactly what training on the pgn directly would have seen
def test_shard_holds_the_same_examples_as_game_to_examples(shard):
    pgn_path, games = shard

    boards, moves, games_table = load_shard(pgn_path)

    assert boards.dtype == np.uint8 and boards.shape[1] == TOKENS_PER_BOARD
    assert moves.dtype == np.int16
    for row, game in zip(games_table, games):
        examples = game_to_examples(game)
        first, last = row["first_position"], row["first_position"] + row["num_positions"]
        assert boards[first:last].tolist() == [example.board for example in examples]
        assert moves[first:last].tolist() == [example.move for example in examples]


def test_games_table_records_what_kind_of_game_it_was(shard):
    pgn_path, _ = shard

    _, _, games_table = load_shard(pgn_path)

    assert games_table.dtype == GAME_FIELDS
    assert games_table["first_position"][0] == 0
    assert games_table["first_position"][1] == games_table["num_positions"][0]  #games sit back to back
    assert games_table[["white_elo", "black_elo"]][0].tolist() == (2500, 2400)
    assert games_table["base_seconds"].tolist() == [180, 600, -1]
    assert games_table["increment_seconds"].tolist() == [2, 0, -1]
    assert games_table["time_forfeit"].tolist() == [False, True, False]
    assert games_table["result"].tolist() == [1, -1, 0]


def test_unusable_games_are_left_out(carlsen_games, tmp_path, monkeypatch):
    monkeypatch.setattr(preprocess, "PROCESSED_DATA_DIR", tmp_path)
    pgn_path = tmp_path / "month.pgn"
    write_pgn(pgn_path, [
        with_headers(carlsen_games[0], Variant="Chess960"),
        with_headers(carlsen_games[1], Result="*"),             #unfinished
        with_headers(chess.pgn.Game(), Result="1-0"),           #no moves
        with_headers(carlsen_games[2], Result="1-0"),
    ])

    assert preprocess_month(pgn_path)[0] == 1


def test_shard_with_missing_positions_is_rejected(shard):
    pgn_path, _ = shard
    moves_path = preprocess.shard_paths(pgn_path)[1]
    moves_path.write_bytes(moves_path.read_bytes()[:-2])  #one int16 short, as after an interrupted copy

    with pytest.raises(ValueError):
        load_shard(pgn_path)


@pytest.mark.parametrize(
    "time_control, expected",
    [("180+2", (180, 2)), ("600+0", (600, 0)), ("-", (-1, -1)), ("", (-1, -1)), ("1/259200", (-1, -1))],
)
def test_parse_time_control(time_control, expected):
    assert parse_time_control(time_control) == expected


def games_table(rows) -> np.ndarray:
    """(base_seconds, increment_seconds, time_forfeit), every game 100 positions long"""
    table = np.zeros(len(rows), dtype=GAME_FIELDS)
    table["num_positions"] = 100
    for index, (base_seconds, increment_seconds, time_forfeit) in enumerate(rows):
        table[index]["base_seconds"] = base_seconds
        table[index]["increment_seconds"] = increment_seconds
        table[index]["time_forfeit"] = time_forfeit
    return table


#blitz, blitz lost on time, rapid (600s), rapid through its increment (180 + 40 * 10), rapid lost on time, no clock
MIXED_GAMES = [(180, 0, False), (180, 0, True), (600, 0, False), (180, 10, False), (900, 0, True), (-1, -1, False)]


def test_only_slow_games_are_chosen_when_there_are_enough(monkeypatch):
    monkeypatch.setattr(select_games, "TARGET_POSITIONS", 200)

    assert select_games.choose_games(games_table(MIXED_GAMES)).tolist() == [False, False, True, True, False, False]


def test_faster_games_are_added_when_slow_games_are_too_few(monkeypatch):
    monkeypatch.setattr(select_games, "TARGET_POSITIONS", 201)

    #still never a game lost on time or one without a clock
    assert select_games.choose_games(games_table(MIXED_GAMES)).tolist() == [True, False, True, True, False, False]


#the whole path a training run takes: pgns -> shards -> chosen games -> train and val files -> dataset
def test_selected_games_become_a_training_set(carlsen_games, tmp_path, monkeypatch):
    monkeypatch.setattr(preprocess, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(select_games, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(select_games, "NAME", "test")
    monkeypatch.setattr(select_games, "VAL_FRACTION", 0.5)

    usable = dict(TimeControl="600+0", Termination="Normal", Result="1-0")
    months = {
        "first.pgn": [with_headers(carlsen_games[0], **usable), with_headers(carlsen_games[1], TimeControl="600+0", Termination="Time forfeit")],
        "second.pgn": [with_headers(carlsen_games[2], **usable), with_headers(carlsen_games[3], **usable)],
    }
    for name, games in months.items():
        write_pgn(tmp_path / name, games)
        preprocess_month(tmp_path / name)
    monkeypatch.setattr(select_games, "LICHESS_ELITE_PGNS", [tmp_path / name for name in months])

    usable_games = [carlsen_games[0], carlsen_games[2], carlsen_games[3]]
    num_usable_positions = sum(len(game_to_examples(game)) for game in usable_games)
    monkeypatch.setattr(select_games, "TARGET_POSITIONS", num_usable_positions)  #so all three are needed

    select_games.main()
    train_boards, train_moves, val_boards, val_moves = select_games.load_training_set("test")

    #every position of the three usable games, none of the game lost on time, split by whole games
    assert len(train_moves) + len(val_moves) == num_usable_positions
    assert sorted([len(train_moves), len(val_moves)]) == sorted(
        sizes for sizes in _split_sizes([len(game_to_examples(game)) for game in usable_games], len(val_moves))
    )

    dataset = ChessGamesDataset.from_arrays(train_boards, train_moves)
    for index in range(len(dataset)):
        board, move = dataset[index]
        assert decode_id_to_move(move.item()) in decode_board(board.tolist()).legal_moves, index


def _split_sizes(game_lengths, val_size):
    """val is exactly one whole game here (half of three, rounded down), train is the other two"""
    assert val_size in game_lengths
    return [val_size, sum(game_lengths) - val_size]


def test_too_few_games_for_a_val_set_is_an_error(shard, tmp_path, monkeypatch):
    pgn_path, _ = shard
    monkeypatch.setattr(select_games, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(select_games, "LICHESS_ELITE_PGNS", [pgn_path])

    #one usable game, and 2% of one game is no game at all
    with pytest.raises(ValueError):
        select_games.main()

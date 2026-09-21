import chess
import chess.pgn
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import data
from data import (
    ChessGamesDataset,
    TrainingExample,
    create_dataloaders,
    datasets_to_dataloaders,
    game_to_examples,
    load_training_set,
    pgns_to_examples,
    split_data,
)
from encoding import TOKENS_PER_BOARD, decode_board, decode_id_to_move, encode_board, encode_history, encode_move_to_id


def make_board(fill: int) -> list[int]:
    """a recognisable stand-in board, so a test can tell which example it got back"""
    return [fill] * TOKENS_PER_BOARD


#two games of different lengths, so flattening has a game boundary to get wrong
TWO_GAMES = [
    [TrainingExample(make_board(1), 10, 1), TrainingExample(make_board(2), 20, -1)],
    [TrainingExample(make_board(3), 30, 0)],
]


def test_game_to_examples_pairs_each_position_with_the_move_played_from_it():
    game = chess.pgn.Game()
    game.add_line([chess.Move.from_uci(uci) for uci in ("e2e4", "e7e5", "g1f3")])
    game.headers["Result"] = "1-0"

    examples = game_to_examples(game)

    board = chess.Board()
    assert len(examples) == 3
    for example, uci in zip(examples, ("e2e4", "e7e5", "g1f3")):
        move = chess.Move.from_uci(uci)
        assert example.board == encode_board(board)  #the position before the move, not after
        assert example.move == encode_move_to_id(move, board.turn)
        board.push(move)

    #white won: a win for the side to move when white is to move, a loss when black is
    assert [example.result for example in examples] == [1, -1, 1]


def test_dataset_flattens_games_in_order():
    dataset = ChessGamesDataset(TWO_GAMES)

    assert len(dataset) == 3
    for index, (fill, move_id) in enumerate([(1, 10), (2, 20), (3, 30)]):
        board, move, _, _ = dataset[index]
        assert board.tolist() == make_board(fill)
        assert move.item() == move_id


def test_items_have_the_dtype_and_shape_the_model_needs():
    board, move, result, _ = ChessGamesDataset(TWO_GAMES)[0]

    #embedding lookups and CrossEntropyLoss targets both require int64
    assert board.dtype == torch.long and board.shape == (TOKENS_PER_BOARD,)
    assert move.dtype == torch.long and move.shape == ()
    assert result.dtype == torch.long and result.shape == ()


def test_results_come_back_as_the_classes_of_the_value_head():
    _, _, results, _ = ChessGamesDataset(TWO_GAMES)[[0, 1, 2]]

    assert results.tolist() == [2, 0, 1]  #won, lost, draw


def test_largest_values_survive_compact_storage():
    #80 is the largest board token (en passant on h8), 1967 the largest move id
    dataset = ChessGamesDataset([[TrainingExample(make_board(80), 1967, 0)]])

    board, move, _, _ = dataset[0]
    assert board.tolist() == make_board(80)
    assert move.item() == 1967


def test_value_too_large_for_storage_is_rejected():
    #must fail rather than wrap around into a different, valid-looking id
    with pytest.raises(RuntimeError):
        ChessGamesDataset([[TrainingExample(make_board(0), 40000, 0)]])


def test_dataloader_batches_boards_and_moves():
    loader = DataLoader(ChessGamesDataset(TWO_GAMES), batch_size=2, shuffle=False)

    boards, moves, _, _ = next(iter(loader))

    assert boards.shape == (2, TOKENS_PER_BOARD)
    assert moves.tolist() == [10, 20]


#catches boards and moves drifting out of alignment, which shapes and lengths cannot see
def test_every_label_is_legal_in_its_position(carlsen_games):
    games = [game_to_examples(game) for game in carlsen_games]
    dataset = ChessGamesDataset(games)

    assert len(dataset) == sum(len(game) for game in games)
    for index in range(len(dataset)):
        board, move, _, _ = dataset[index]
        position = decode_board(board.tolist())
        assert decode_id_to_move(move.item(), position.turn) in position.legal_moves, index


def make_games(count: int, positions_per_game: int = 1) -> list[list[TrainingExample]]:
    """games whose move ids say which game they are"""
    return [
        [TrainingExample(make_board(0), game_number, 0) for _ in range(positions_per_game)]
        for game_number in range(count)
    ]


def game_numbers(games: list[list[TrainingExample]]) -> list[int]:
    return [game[0].move for game in games]


def test_split_puts_every_game_on_exactly_one_side():
    train, val = split_data(make_games(10), train_fraction=0.8)

    assert len(train) == 8 and len(val) == 2
    assert sorted(game_numbers(train) + game_numbers(val)) == list(range(10))


def test_split_shuffles_before_cutting():
    #an unshuffled cut of a chronological pgn would validate only on the most recent games
    train, _ = split_data(make_games(100))

    assert game_numbers(train) != list(range(80))


def test_split_leaves_the_input_list_alone():
    games = make_games(10)

    split_data(games)

    assert game_numbers(games) == list(range(10))


def test_split_is_the_same_for_the_same_seed():
    #validation numbers are only comparable between runs if the validation set is the same
    first = split_data(make_games(50), seed=7)
    second = split_data(make_games(50), seed=7)
    other_seed = split_data(make_games(50), seed=8)

    assert first == second
    assert first != other_seed


def test_split_rejects_a_percentage():
    with pytest.raises(ValueError):
        split_data(make_games(10), train_fraction=80)


def loader_moves(loader: DataLoader) -> list[int]:
    """every move id the loader yields in one epoch, in the order it yields them"""
    return [move for _, moves, _, _ in loader for move in moves.tolist()]


def write_pgn(pgn_path, games) -> None:
    with open(pgn_path, "w") as pgn:
        for game in games:
            print(game, file=pgn, end="\n\n")


def test_pgns_to_examples_joins_the_games_of_every_pgn_in_order(carlsen_games, tmp_path):
    first_pgn, second_pgn = tmp_path / "first.pgn", tmp_path / "second.pgn"
    write_pgn(first_pgn, carlsen_games[:3])
    write_pgn(second_pgn, carlsen_games[3:])

    games = pgns_to_examples([first_pgn, second_pgn])

    assert games == [game_to_examples(game) for game in carlsen_games]


def test_dataloaders_keep_each_game_on_one_side_of_the_split():
    #3 positions per game, so a split by position rather than by game would cut through a game
    train_loader, val_loader = create_dataloaders(make_games(10, positions_per_game=3), train_fraction=0.8)

    train_games, val_games = set(loader_moves(train_loader)), set(loader_moves(val_loader))

    assert len(train_loader.dataset) == 24 and len(val_loader.dataset) == 6
    assert train_games.isdisjoint(val_games)
    assert train_games | val_games == set(range(10))


def test_dataloaders_use_their_own_batch_sizes():
    train_loader, val_loader = create_dataloaders(make_games(100), train_batch_size=16, val_batch_size=5)

    train_boards, train_moves, _, _ = next(iter(train_loader))
    val_boards, val_moves, _, _ = next(iter(val_loader))

    assert train_boards.shape == (16, TOKENS_PER_BOARD) and train_moves.shape == (16,)
    assert val_boards.shape == (5, TOKENS_PER_BOARD) and val_moves.shape == (5,)


def test_train_loader_reshuffles_and_val_loader_does_not():
    torch.manual_seed(0)
    train_loader, val_loader = create_dataloaders(make_games(100), train_batch_size=16, val_batch_size=16)

    first_epoch, second_epoch = loader_moves(train_loader), loader_moves(train_loader)

    #every example exactly once per epoch, the short last batch included, in a new order each time
    assert sorted(first_epoch) == sorted(second_epoch) == sorted(train_loader.dataset.moves.tolist())
    assert first_epoch != second_epoch
    assert loader_moves(val_loader) == val_loader.dataset.moves.tolist()


def test_dataloaders_pass_the_split_settings_through():
    games = make_games(50)

    default_split, _ = create_dataloaders(games)
    other_split, _ = create_dataloaders(games, train_fraction=0.5, seed=7)

    assert len(default_split.dataset) == 40 and len(other_split.dataset) == 25
    assert set(other_split.dataset.moves.tolist()) == set(game_numbers(split_data(games, 0.5, 7)[0]))


def test_dataset_from_arrays_matches_dataset_from_games():
    from_games = ChessGamesDataset(TWO_GAMES)
    boards = np.array([example.board for game in TWO_GAMES for example in game], dtype=np.uint8)
    moves = np.array([example.move for game in TWO_GAMES for example in game], dtype=np.int16)
    results = np.array([example.result for game in TWO_GAMES for example in game], dtype=np.int8)

    from_arrays = ChessGamesDataset.from_arrays(boards, moves, results)

    assert len(from_arrays) == len(from_games)
    for index in range(len(from_games)):
        for from_array, from_game in zip(from_arrays[index], from_games[index]):
            assert torch.equal(from_array, from_game)
    assert from_arrays[0][0].dtype == torch.long and from_arrays[0][1].dtype == torch.long


def test_dataset_from_arrays_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        ChessGamesDataset.from_arrays(np.zeros((3, TOKENS_PER_BOARD), dtype=np.uint8), np.zeros(2, dtype=np.int16), np.zeros(3, dtype=np.int8))


def test_a_list_of_indices_comes_back_as_one_batch():
    boards, moves, _, _ = ChessGamesDataset(TWO_GAMES)[[2, 0]]

    assert boards.shape == (2, TOKENS_PER_BOARD)
    assert moves.tolist() == [30, 10]


def write_training_set(folder, name, positions_per_split) -> None:
    """raw arrays the way processed_data/ holds them: uint8 boards, int16 moves and int8 results, no header"""
    for split, num_positions in positions_per_split.items():
        boards = np.full((num_positions, TOKENS_PER_BOARD), 5, dtype=np.uint8)
        moves = np.arange(num_positions, dtype=np.int16)
        (folder / f"{name}_{split}_boards.bin").write_bytes(boards.tobytes())
        (folder / f"{name}_{split}_moves.bin").write_bytes(moves.tobytes())
        (folder / f"{name}_{split}_results.bin").write_bytes(np.full(num_positions, -1, dtype=np.int8).tobytes())


def test_training_set_is_loaded_from_raw_arrays(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PROCESSED_DATA_DIR", tmp_path)
    write_training_set(tmp_path, "tiny", {"train": 300, "val": 20})

    train_dataset, val_dataset = load_training_set("tiny")

    assert len(train_dataset) == 300 and len(val_dataset) == 20
    boards, moves, results, _ = train_dataset[[299, 0]]
    assert boards.tolist() == [[5] * TOKENS_PER_BOARD] * 2
    assert moves.tolist() == [299, 0]  #large enough that reading the int16 moves as bytes would give something else
    assert results.tolist() == [0, 0]  #-1 in the file, a lost game


def test_mixed_training_sets_share_every_batch_equally(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PROCESSED_DATA_DIR", tmp_path)
    write_training_set(tmp_path, "large", {"train": 3000, "val": 20})
    write_training_set(tmp_path, "small", {"train": 100, "val": 30})
    #move ids say which set a position is from: large has 0 to 2999, small is moved up to 10000 and above
    small_moves = tmp_path / "small_train_moves.bin"
    small_moves.write_bytes((np.arange(100, dtype=np.int16) + 10000).tobytes())

    train_dataset, val_dataset = load_training_set("large+small")
    train_loader, _ = datasets_to_dataloaders(train_dataset, val_dataset, train_batch_size=16, val_batch_size=16)

    assert len(train_dataset) == 3100 and len(val_dataset) == 20  #val is the first set's
    for _, (boards, moves, results, _) in zip(range(20), train_loader):
        assert boards.shape == (16, TOKENS_PER_BOARD) and results.shape == (16,)
        assert (moves >= 10000).sum() == 8  #half of every batch although small is 3% of the positions


def test_missing_training_set_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PROCESSED_DATA_DIR", tmp_path)

    with pytest.raises(FileNotFoundError):
        load_training_set("does_not_exist")


def test_training_set_with_mismatched_files_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PROCESSED_DATA_DIR", tmp_path)
    write_training_set(tmp_path, "broken", {"train": 300, "val": 20})
    moves_path = tmp_path / "broken_train_moves.bin"
    moves_path.write_bytes(moves_path.read_bytes()[:-2])  #one move short, as after an interrupted copy

    with pytest.raises(ValueError):
        load_training_set("broken")


#the dataset finds the history in the rows before a position, the player takes it from the board's moves.
#Both have to see the same thing, for both colours and through captures, castling, en passant and promotions
def test_history_from_stored_rows_matches_history_from_the_board(carlsen_games):
    games = [game_to_examples(game) for game in carlsen_games]
    dataset = ChessGamesDataset(games, history=3)

    row = 0
    for game in carlsen_games:
        board = game.board()
        for move in game.mainline_moves():
            assert dataset[row][3].tolist() == encode_history(board, 3), (row, board.fen())
            board.push(move)
            row += 1


def test_history_of_a_capture_names_the_squares_and_what_stood_there():
    board = chess.Board()
    for san in ("e4", "d5", "exd5"):
        board.push_san(san)

    #black to move, so the board is seen from black's side: white's pawn went from e4 to d5, which are e5 and d4
    #from there, and it is the opponent's pawn (7) that moved and the mover's own pawn (1) that was taken
    last_ply, = encode_history(board, 1)
    changed = {chess.square_name(square): code for square, code in enumerate(last_ply) if code != 0}
    assert changed == {"e5": 1 + 7, "d4": 1 + 1}


def test_history_is_empty_before_the_first_move_and_without_history():
    dataset = ChessGamesDataset(TWO_GAMES, history=2)

    assert dataset[0][3].tolist() == [[0] * 64, [0] * 64]  #the first position of a game
    assert dataset[2][3].tolist() == [[0] * 64, [0] * 64]  #and of the next game, whatever the row before it holds
    assert ChessGamesDataset(TWO_GAMES)[[0, 1]][3].shape == (2, 0, 64)

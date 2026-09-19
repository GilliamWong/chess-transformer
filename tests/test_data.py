import chess
import chess.pgn
import pytest
import torch
from torch.utils.data import DataLoader

from data import (
    ChessGamesDataset,
    TrainingExample,
    create_dataloaders,
    game_to_examples,
    pgns_to_examples,
    split_data,
)
from encoding import TOKENS_PER_BOARD, decode_board, decode_id_to_move, encode_board, encode_move_to_id


def make_board(fill: int) -> list[int]:
    """a recognisable stand-in board, so a test can tell which example it got back"""
    return [fill] * TOKENS_PER_BOARD


#two games of different lengths, so flattening has a game boundary to get wrong
TWO_GAMES = [
    [TrainingExample(make_board(1), 10), TrainingExample(make_board(2), 20)],
    [TrainingExample(make_board(3), 30)],
]


def test_game_to_examples_pairs_each_position_with_the_move_played_from_it():
    game = chess.pgn.Game()
    game.add_line([chess.Move.from_uci(uci) for uci in ("e2e4", "e7e5", "g1f3")])

    examples = game_to_examples(game)

    board = chess.Board()
    assert len(examples) == 3
    for example, uci in zip(examples, ("e2e4", "e7e5", "g1f3")):
        move = chess.Move.from_uci(uci)
        assert example.board == encode_board(board)  #the position before the move, not after
        assert example.move == encode_move_to_id(move)
        board.push(move)


def test_dataset_flattens_games_in_order():
    dataset = ChessGamesDataset(TWO_GAMES)

    assert len(dataset) == 3
    for index, (fill, move_id) in enumerate([(1, 10), (2, 20), (3, 30)]):
        board, move = dataset[index]
        assert board.tolist() == make_board(fill)
        assert move.item() == move_id


def test_items_have_the_dtype_and_shape_the_model_needs():
    board, move = ChessGamesDataset(TWO_GAMES)[0]

    #embedding lookups and CrossEntropyLoss targets both require int64
    assert board.dtype == torch.long and board.shape == (TOKENS_PER_BOARD,)
    assert move.dtype == torch.long and move.shape == ()


def test_largest_values_survive_compact_storage():
    #80 is the largest board token (en passant on h8), 1967 the largest move id
    dataset = ChessGamesDataset([[TrainingExample(make_board(80), 1967)]])

    board, move = dataset[0]
    assert board.tolist() == make_board(80)
    assert move.item() == 1967


def test_value_too_large_for_storage_is_rejected():
    #must fail rather than wrap around into a different, valid-looking id
    with pytest.raises(RuntimeError):
        ChessGamesDataset([[TrainingExample(make_board(0), 40000)]])


def test_dataloader_batches_boards_and_moves():
    loader = DataLoader(ChessGamesDataset(TWO_GAMES), batch_size=2, shuffle=False)

    boards, moves = next(iter(loader))

    assert boards.shape == (2, TOKENS_PER_BOARD)
    assert moves.tolist() == [10, 20]


#catches boards and moves drifting out of alignment, which shapes and lengths cannot see
def test_every_label_is_legal_in_its_position(carlsen_games):
    games = [game_to_examples(game) for game in carlsen_games]
    dataset = ChessGamesDataset(games)

    assert len(dataset) == sum(len(game) for game in games)
    for index in range(len(dataset)):
        board, move = dataset[index]
        assert decode_id_to_move(move.item()) in decode_board(board.tolist()).legal_moves, index


def make_games(count: int, positions_per_game: int = 1) -> list[list[TrainingExample]]:
    """games whose move ids say which game they are"""
    return [
        [TrainingExample(make_board(0), game_number) for _ in range(positions_per_game)]
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
    return [move for _, moves in loader for move in moves.tolist()]


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

    train_boards, train_moves = next(iter(train_loader))
    val_boards, val_moves = next(iter(val_loader))

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

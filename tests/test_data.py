import chess
import pytest
import torch
from torch.utils.data import DataLoader

from data import ChessGamesDataset, TrainingExample, game_to_examples
from encoding import ENCODING_SIZE, decode_board, decode_id_to_move, encode_board, encode_move_to_id


def make_board(fill: int) -> list[int]:
    """a recognisable stand-in board, so a test can tell which example it got back"""
    return [fill] * ENCODING_SIZE


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
    assert board.dtype == torch.long and board.shape == (ENCODING_SIZE,)
    assert move.dtype == torch.long and move.shape == ()


def test_largest_values_survive_compact_storage():
    #63 is the largest board value (an en passant square), 1967 the largest move id
    dataset = ChessGamesDataset([[TrainingExample(make_board(63), 1967)]])

    board, move = dataset[0]
    assert board.tolist() == make_board(63)
    assert move.item() == 1967


def test_value_too_large_for_storage_is_rejected():
    #must fail rather than wrap around into a different, valid-looking id
    with pytest.raises(RuntimeError):
        ChessGamesDataset([[TrainingExample(make_board(0), 40000)]])


def test_dataloader_batches_boards_and_moves():
    loader = DataLoader(ChessGamesDataset(TWO_GAMES), batch_size=2, shuffle=False)

    boards, moves = next(iter(loader))

    assert boards.shape == (2, ENCODING_SIZE)
    assert moves.tolist() == [10, 20]


#catches boards and moves drifting out of alignment, which shapes and lengths cannot see
def test_every_label_is_legal_in_its_position(carlsen_games):
    games = [game_to_examples(game) for game in carlsen_games]
    dataset = ChessGamesDataset(games)

    assert len(dataset) == sum(len(game) for game in games)
    for index in range(len(dataset)):
        board, move = dataset[index]
        assert decode_id_to_move(move.item()) in decode_board(board.tolist()).legal_moves, index

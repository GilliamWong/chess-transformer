import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import data
from data import EXPECTED_SCORE_PER_CENTIPAWN, ChessGamesDataset, datasets_to_dataloaders, load_training_set
from encoding import TOKENS_PER_BOARD
from tools.label_with_stockfish import NO_EVAL, NO_MOVE

EVAL = 120  #stockfish's score of every position the tests write, in centipawns
TEMPERATURE = 0.02


def write_training_set(folder, name, positions_per_split) -> None:
    """raw arrays the way processed_data/ holds them, no header: uint8 boards, and int16 stockfish moves and evals with
    two places per position, the second empty. Stockfish's best move says which position a row is"""
    for split, num_positions in positions_per_split.items():
        boards = np.full((num_positions, TOKENS_PER_BOARD), 5, dtype=np.uint8)
        stockfish_moves = np.stack([np.arange(num_positions, dtype=np.int16), np.full(num_positions, NO_MOVE, dtype=np.int16)], axis=1)
        stockfish_evals = np.stack([np.full(num_positions, EVAL, dtype=np.int16), np.full(num_positions, NO_EVAL, dtype=np.int16)], axis=1)
        (folder / f"{name}_{split}_boards.bin").write_bytes(boards.tobytes())
        (folder / f"{name}_{split}_stockfish_moves.bin").write_bytes(stockfish_moves.tobytes())
        (folder / f"{name}_{split}_stockfish_evals.bin").write_bytes(stockfish_evals.tobytes())


@pytest.fixture
def tiny(tmp_path, monkeypatch) -> tuple[ChessGamesDataset, ChessGamesDataset]:
    """300 train and 20 val positions"""
    monkeypatch.setattr(data, "PROCESSED_DATA_DIR", tmp_path)
    write_training_set(tmp_path, "tiny", {"train": 300, "val": 20})
    return load_training_set("tiny", TEMPERATURE)


def loader_moves(loader: DataLoader) -> list[int]:
    """every position the loader yields in one epoch, in the order it yields them"""
    return [move for _, moves, _, _ in loader for move in moves[:, 0].tolist()]


def test_training_set_is_loaded_from_raw_arrays(tiny):
    train_dataset, val_dataset = tiny

    assert len(train_dataset) == 300 and len(val_dataset) == 20
    boards, moves, _, values = train_dataset[[299, 0]]
    assert boards.tolist() == [[5] * TOKENS_PER_BOARD] * 2
    assert moves[:, 0].tolist() == [299, 0]  #large enough that reading the int16 moves as bytes would give something else
    #the value target is stockfish's eval of the position as the expected score it stands for
    assert values.tolist() == pytest.approx([1 / (1 + np.exp(-EXPECTED_SCORE_PER_CENTIPAWN * EVAL))] * 2)


def test_items_have_the_dtype_and_shape_the_model_needs(tiny):
    board, target_moves, target_probabilities, value = tiny[0][0]

    #embedding lookups and the gather in policy_loss both require int64
    assert board.dtype == torch.long and board.shape == (TOKENS_PER_BOARD,)
    #the policy target is a list of moves with a probability each, one per place stockfish's labels have
    assert target_moves.dtype == torch.long and target_moves.shape == (2,)
    assert target_probabilities.tolist() == [1.0, 0.0]  #the second place has no move
    assert value.dtype == torch.float and value.shape == ()


def test_a_list_of_indices_comes_back_as_one_batch(tiny):
    boards, moves, _, _ = tiny[0][[2, 0]]

    assert boards.shape == (2, TOKENS_PER_BOARD)
    assert moves[:, 0].tolist() == [2, 0]


def test_dataloaders_use_their_own_batch_sizes(tiny):
    train_loader, val_loader = datasets_to_dataloaders(*tiny, train_batch_size=16, val_batch_size=5)

    train_boards, train_moves, _, _ = next(iter(train_loader))
    val_boards, val_moves, _, _ = next(iter(val_loader))

    assert train_boards.shape == (16, TOKENS_PER_BOARD) and train_moves.shape == (16, 2)
    assert val_boards.shape == (5, TOKENS_PER_BOARD) and val_moves.shape == (5, 2)


def test_train_loader_reshuffles_and_val_loader_does_not(tiny):
    torch.manual_seed(0)
    train_loader, val_loader = datasets_to_dataloaders(*tiny, train_batch_size=16, val_batch_size=16)

    first_epoch, second_epoch = loader_moves(train_loader), loader_moves(train_loader)

    #every position exactly once per epoch, the short last batch included, in a new order each time
    assert sorted(first_epoch) == sorted(second_epoch) == list(range(300))
    assert first_epoch != second_epoch
    assert loader_moves(val_loader) == list(range(20))


def test_train_order_depends_on_the_seed_alone(tiny):
    #building a model draws random numbers, a different amount for every architecture, which used to change the order
    def first_batches(seed: int, draws_before: int) -> list[list[int]]:
        torch.manual_seed(0)
        torch.randn(draws_before)
        train_loader, _ = datasets_to_dataloaders(*tiny, train_batch_size=16, val_batch_size=16, seed=seed)
        return [moves[:, 0].tolist() for _, (_, moves, *_) in zip(range(5), train_loader)]

    assert first_batches(seed=3, draws_before=10) == first_batches(seed=3, draws_before=1000)
    assert first_batches(seed=3, draws_before=10) != first_batches(seed=4, draws_before=10)


def test_missing_training_set_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PROCESSED_DATA_DIR", tmp_path)

    with pytest.raises(FileNotFoundError):
        load_training_set("does_not_exist", TEMPERATURE)


#both targets come from stockfish's labels, a set it has not labelled has neither
def test_training_set_without_stockfish_labels_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PROCESSED_DATA_DIR", tmp_path)
    write_training_set(tmp_path, "unlabelled", {"train": 300, "val": 20})
    for split in ("train", "val"):
        (tmp_path / f"unlabelled_{split}_stockfish_moves.bin").unlink()

    with pytest.raises(FileNotFoundError, match="label_with_stockfish"):
        load_training_set("unlabelled", TEMPERATURE)


def test_training_set_with_mismatched_files_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PROCESSED_DATA_DIR", tmp_path)
    write_training_set(tmp_path, "broken", {"train": 300, "val": 20})
    moves_path = tmp_path / "broken_train_stockfish_moves.bin"
    moves_path.write_bytes(moves_path.read_bytes()[:-2])  #one move short, as after an interrupted copy

    with pytest.raises(ValueError):
        load_training_set("broken", TEMPERATURE)


def test_dataset_rejects_mismatched_lengths():
    labels = np.zeros((2, 2), dtype=np.int16)

    with pytest.raises(ValueError):
        ChessGamesDataset(np.zeros((3, TOKENS_PER_BOARD), dtype=np.uint8), labels, labels, TEMPERATURE)

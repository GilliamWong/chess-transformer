import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from training import data
from training.data import BLUNDER_THRESHOLD, ChessGamesDataset, MixedLoader, SelfPlayDataset, datasets_to_dataloaders, load_training_set
from model.encoding import EXPECTED_SCORE_PER_CENTIPAWN, NO_EVAL, NO_MOVE, TOKENS_PER_BOARD

EVAL = 120  #centipawns
TEMPERATURE = 0.02


@pytest.fixture
def folder(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PROCESSED_DATA_DIR", tmp_path)
    return tmp_path


def write_arrays(folder, name, split, **arrays) -> None:
    for kind, array in arrays.items():
        (folder / f"{name}_{split}_{kind}.bin").write_bytes(array.tobytes())


def write_training_set(folder, name, positions_per_split) -> None:
    for split, positions in positions_per_split.items():
        write_arrays(folder, name, split,
                     boards=np.full((positions, TOKENS_PER_BOARD), 5, dtype=np.uint8),
                     stockfish_moves=np.stack([np.arange(positions, dtype=np.int16), np.full(positions, NO_MOVE, dtype=np.int16)], axis=1),
                     stockfish_evals=np.stack([np.full(positions, EVAL, dtype=np.int16), np.full(positions, NO_EVAL, dtype=np.int16)], axis=1))


@pytest.fixture
def tiny(folder) -> tuple[ChessGamesDataset, ChessGamesDataset]:
    write_training_set(folder, "tiny", {"train": 300, "val": 20})
    return load_training_set("tiny", TEMPERATURE)


def loader_moves(loader: DataLoader) -> list[int]:
    return [move for _, moves, _, _, _ in loader for move in moves[:, 0].tolist()]


def test_training_set_is_loaded_from_raw_arrays(tiny):
    train_dataset, val_dataset = tiny

    assert len(train_dataset) == 300 and len(val_dataset) == 20
    boards, moves, _, values, _ = train_dataset[[299, 0]]
    assert boards.tolist() == [[5] * TOKENS_PER_BOARD] * 2
    assert moves[:, 0].tolist() == [299, 0]  #299 does not fit in a byte
    assert values.tolist() == pytest.approx([1 / (1 + np.exp(-EXPECTED_SCORE_PER_CENTIPAWN * EVAL))] * 2)


def test_items_have_the_dtype_and_shape_the_model_needs(tiny):
    board, target_moves, target_probabilities, value, blunders = tiny[0][0]

    #embedding lookups and the gather in policy_loss both require int64
    assert board.dtype == torch.long and board.shape == (TOKENS_PER_BOARD,)
    assert target_moves.dtype == torch.long and target_moves.shape == (2,)
    assert target_probabilities.tolist() == [1.0, 0.0]
    assert value.dtype == torch.float and value.shape == ()
    assert blunders.dtype == torch.long and blunders.tolist() == [NO_MOVE]


def test_dataloaders_use_their_own_batch_sizes(tiny):
    train_loader, val_loader = datasets_to_dataloaders(*tiny, train_batch_size=16, val_batch_size=5)

    train_boards, train_moves, _, _, _ = next(iter(train_loader))
    val_boards, val_moves, _, _, _ = next(iter(val_loader))

    assert train_boards.shape == (16, TOKENS_PER_BOARD) and train_moves.shape == (16, 2)
    assert val_boards.shape == (5, TOKENS_PER_BOARD) and val_moves.shape == (5, 2)


def test_train_loader_reshuffles_and_val_loader_does_not(tiny):
    torch.manual_seed(0)
    train_loader, val_loader = datasets_to_dataloaders(*tiny, train_batch_size=16, val_batch_size=16)

    first_epoch, second_epoch = loader_moves(train_loader), loader_moves(train_loader)

    assert sorted(first_epoch) == sorted(second_epoch) == list(range(300))
    assert first_epoch != second_epoch
    assert loader_moves(val_loader) == list(range(20))


def test_train_order_depends_on_the_seed_alone(tiny):
    def first_batches(seed: int, draws_before: int) -> list[list[int]]:
        torch.manual_seed(0)
        torch.randn(draws_before)
        train_loader, _ = datasets_to_dataloaders(*tiny, train_batch_size=16, val_batch_size=16, seed=seed)
        return [moves[:, 0].tolist() for _, (_, moves, *_) in zip(range(5), train_loader)]

    assert first_batches(seed=3, draws_before=10) == first_batches(seed=3, draws_before=1000)
    assert first_batches(seed=3, draws_before=10) != first_batches(seed=4, draws_before=10)


@pytest.mark.usefixtures("folder")
def test_missing_training_set_is_reported():
    with pytest.raises(FileNotFoundError):
        load_training_set("does_not_exist", TEMPERATURE)


def test_training_set_without_stockfish_labels_is_reported(folder):
    write_training_set(folder, "unlabelled", {"train": 300, "val": 20})
    for split in ("train", "val"):
        (folder / f"unlabelled_{split}_stockfish_moves.bin").unlink()

    with pytest.raises(FileNotFoundError, match="unlabelled_train_stockfish_moves.bin"):
        load_training_set("unlabelled", TEMPERATURE)


def test_training_set_with_mismatched_files_is_rejected(folder):
    write_training_set(folder, "broken", {"train": 300, "val": 20})
    moves_path = folder / "broken_train_stockfish_moves.bin"
    moves_path.write_bytes(moves_path.read_bytes()[:-2])  #one int16 move short

    with pytest.raises(ValueError):
        load_training_set("broken", TEMPERATURE)


def test_dataset_rejects_mismatched_lengths():
    labels = np.zeros((2, 2), dtype=np.int16)

    with pytest.raises(ValueError):
        ChessGamesDataset(np.zeros((3, TOKENS_PER_BOARD), dtype=np.uint8), labels, labels, TEMPERATURE)


#6 self play batches of 12 outnumber the 5 stockfish ones of 4, which start over; its 2 moves are padded to 8
def test_mixed_batches_take_a_fixed_share_from_each_set(tiny):
    _, stockfish = tiny
    positions = 72
    self_play = SelfPlayDataset(np.full((positions, TOKENS_PER_BOARD), 3, dtype=np.uint8), np.tile(np.arange(8, dtype=np.int16), (positions, 1)),
                                np.full((positions, 8), 1 / 8, dtype=np.float32), np.ones(positions, dtype=np.float32),
                                np.zeros(positions, dtype=np.float32), result_weight=0.5)
    self_play_loader, _ = datasets_to_dataloaders(self_play, self_play, train_batch_size=12, val_batch_size=12, seed=0)
    stockfish_loader, _ = datasets_to_dataloaders(stockfish, stockfish, train_batch_size=4, val_batch_size=4, seed=1)

    batches = list(MixedLoader(self_play_loader, stockfish_loader))

    assert len(batches) == 6
    for boards, moves, probabilities, values, blunders in batches:
        assert boards.shape == (16, TOKENS_PER_BOARD) and moves.shape == probabilities.shape == (16, 8)
        assert probabilities.sum(dim=1).tolist() == pytest.approx([1.0] * 16)
        assert moves[12:, 2:].eq(0).all() and probabilities[12:, 2:].eq(0).all()
        assert values[:12].tolist() == pytest.approx([0.5] * 12)
        assert blunders.shape == (16, 1) and blunders.eq(NO_MOVE).all()


#position 0: a blunder, a move exactly at the threshold and a harmless move; the others nothing labelled
def write_blunder_labels(folder, name, positions):
    moves = np.full((positions, 3), NO_MOVE, dtype=np.int16)
    losses = np.zeros((positions, 3), dtype=np.float16)
    moves[0], losses[0] = [700, 701, 702], [0.6, BLUNDER_THRESHOLD, 0.05]
    write_arrays(folder, name, "train", worst_moves=moves, worst_losses=losses)


def test_blunders_are_the_labelled_moves_that_give_away_enough(folder):
    write_training_set(folder, "blunders", {"train": 4, "val": 2})
    write_blunder_labels(folder, "blunders", 4)

    train_dataset, val_dataset = load_training_set("blunders", TEMPERATURE)

    assert train_dataset[[0, 1]][4].tolist() == [[700, 701, NO_MOVE], [NO_MOVE] * 3]
    #tensors() lists what is read from disk, the labels included
    assert len(train_dataset.tensors()) == 5
    #val has no blunder labels of its own
    assert val_dataset[[0, 1]][4].tolist() == [[NO_MOVE], [NO_MOVE]]

    strict, _ = load_training_set("blunders", TEMPERATURE, blunder_threshold=0.5)
    assert strict[[0]][4].tolist() == [[700, NO_MOVE, NO_MOVE]]
    lenient, _ = load_training_set("blunders", TEMPERATURE, blunder_threshold=0.05)
    assert lenient[[0]][4].tolist() == [[700, 701, 702]]


def test_self_play_sets_take_blunder_labels_too(folder):
    for split, positions in (("train", 4), ("val", 2)):
        write_arrays(folder, "selfplay", split,
                     boards=np.full((positions, TOKENS_PER_BOARD), 3, dtype=np.uint8),
                     visit_moves=np.tile(np.arange(2, dtype=np.int16), (positions, 1)),
                     visit_shares=np.full((positions, 2), 0.5, dtype=np.float32),
                     outcomes=np.ones(positions, dtype=np.float32),
                     root_scores=np.full(positions, 0.5, dtype=np.float32))
    write_blunder_labels(folder, "selfplay", 4)

    train_dataset, val_dataset = load_training_set("selfplay", TEMPERATURE)

    assert isinstance(train_dataset, SelfPlayDataset)
    assert train_dataset[[0, 1]][4].tolist() == [[700, 701, NO_MOVE], [NO_MOVE] * 3]
    assert len(train_dataset.tensors()) == 7
    assert val_dataset[[0, 1]][4].tolist() == [[NO_MOVE], [NO_MOVE]]
    strict, _ = load_training_set("selfplay", TEMPERATURE, blunder_threshold=0.5)
    assert strict[[0]][4].tolist() == [[700, NO_MOVE, NO_MOVE]]


def test_blunder_labels_must_come_in_pairs_of_the_same_shape():
    boards = np.zeros((2, TOKENS_PER_BOARD), dtype=np.uint8)
    labels = np.zeros((2, 2), dtype=np.int16)

    with pytest.raises(ValueError):
        ChessGamesDataset(boards, labels, labels, TEMPERATURE, worst_moves=np.zeros((2, 3), dtype=np.int16))
    with pytest.raises(ValueError):
        ChessGamesDataset(boards, labels, labels, TEMPERATURE, np.zeros((2, 3), dtype=np.int16), np.zeros((2, 2), dtype=np.float16))


def write_teacher_targets(folder, name, positions_per_split):
    for split, positions in positions_per_split.items():
        write_arrays(folder, name, split,
                     teacher_moves=np.tile(np.array([900, 901], dtype=np.int16), (positions, 1)),
                     teacher_probabilities=np.tile(np.array([0.75, 0.25], dtype=np.float16), (positions, 1)),
                     teacher_scores=np.full(positions, 0.9, dtype=np.float16))


def test_teacher_targets_take_their_share_of_the_train_targets(folder):
    write_training_set(folder, "taught", {"train": 4, "val": 2})
    write_teacher_targets(folder, "taught", {"train": 4, "val": 2})
    stockfish_value = torch.sigmoid(torch.tensor(EXPECTED_SCORE_PER_CENTIPAWN * EVAL)).item()

    train_dataset, val_dataset = load_training_set("taught", TEMPERATURE, teacher_weight=0.25)
    _, moves, probabilities, values, _ = train_dataset[[1]]
    assert moves.tolist() == [[1, 0, 900, 901]]
    assert probabilities[0].tolist() == pytest.approx([0.75, 0.0, 0.25 * 0.75, 0.25 * 0.25])
    assert values.item() == pytest.approx(0.75 * stockfish_value + 0.25 * 0.9, abs=1e-3)
    assert len(train_dataset.tensors()) == 6

    _, val_moves, val_probabilities, val_values, _ = val_dataset[[1]]
    assert val_moves.tolist() == [[1, 0]] and val_probabilities[0].tolist() == pytest.approx([1.0, 0.0])
    assert val_values.item() == pytest.approx(stockfish_value, abs=1e-6)

    plain, _ = load_training_set("taught", TEMPERATURE)
    assert plain[[1]][1].tolist() == [[1, 0]]


def test_a_teacher_weight_without_teacher_targets_is_reported(folder):
    write_training_set(folder, "untaught", {"train": 4, "val": 2})
    with pytest.raises(FileNotFoundError, match="teacher_moves"):
        load_training_set("untaught", TEMPERATURE, teacher_weight=0.5)

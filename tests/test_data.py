import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from training import data
from training.data import BLUNDER_THRESHOLD, EXPECTED_SCORE_PER_CENTIPAWN, ChessGamesDataset, MixedLoader, SelfPlayDataset, datasets_to_dataloaders, load_training_set
from model.encoding import TOKENS_PER_BOARD
from tools.label_with_stockfish import NO_EVAL, NO_MOVE

EVAL = 120  #stockfish's score of every position the tests write, in centipawns
TEMPERATURE = 0.02


#raw arrays as in processed_data/, two label places per position (the second empty); the best move is the row number
def write_training_set(folder, name, positions_per_split) -> None:
    for split, num_positions in positions_per_split.items():
        boards = np.full((num_positions, TOKENS_PER_BOARD), 5, dtype=np.uint8)
        stockfish_moves = np.stack([np.arange(num_positions, dtype=np.int16), np.full(num_positions, NO_MOVE, dtype=np.int16)], axis=1)
        stockfish_evals = np.stack([np.full(num_positions, EVAL, dtype=np.int16), np.full(num_positions, NO_EVAL, dtype=np.int16)], axis=1)
        (folder / f"{name}_{split}_boards.bin").write_bytes(boards.tobytes())
        (folder / f"{name}_{split}_stockfish_moves.bin").write_bytes(stockfish_moves.tobytes())
        (folder / f"{name}_{split}_stockfish_evals.bin").write_bytes(stockfish_evals.tobytes())


#300 train and 20 val positions
@pytest.fixture
def tiny(tmp_path, monkeypatch) -> tuple[ChessGamesDataset, ChessGamesDataset]:
    monkeypatch.setattr(data, "PROCESSED_DATA_DIR", tmp_path)
    write_training_set(tmp_path, "tiny", {"train": 300, "val": 20})
    return load_training_set("tiny", TEMPERATURE)


#one epoch's positions in the order the loader yields them
def loader_moves(loader: DataLoader) -> list[int]:
    return [move for _, moves, _, _, _ in loader for move in moves[:, 0].tolist()]


def test_training_set_is_loaded_from_raw_arrays(tiny):
    train_dataset, val_dataset = tiny

    assert len(train_dataset) == 300 and len(val_dataset) == 20
    boards, moves, _, values, _ = train_dataset[[299, 0]]
    assert boards.tolist() == [[5] * TOKENS_PER_BOARD] * 2
    assert moves[:, 0].tolist() == [299, 0]  #large enough that reading the int16 moves as bytes would give something else
    #the value target is stockfish's eval of the position as the expected score it stands for
    assert values.tolist() == pytest.approx([1 / (1 + np.exp(-EXPECTED_SCORE_PER_CENTIPAWN * EVAL))] * 2)


def test_items_have_the_dtype_and_shape_the_model_needs(tiny):
    board, target_moves, target_probabilities, value, blunders = tiny[0][0]

    #embedding lookups and the gather in policy_loss both require int64
    assert board.dtype == torch.long and board.shape == (TOKENS_PER_BOARD,)
    #one target move and probability per place of stockfish's labels
    assert target_moves.dtype == torch.long and target_moves.shape == (2,)
    assert target_probabilities.tolist() == [1.0, 0.0]  #the second place has no move
    assert value.dtype == torch.float and value.shape == ()
    #a set without blunder labels: one place, no blunder
    assert blunders.dtype == torch.long and blunders.tolist() == [NO_MOVE]


def test_a_list_of_indices_comes_back_as_one_batch(tiny):
    boards, moves, _, _, _ = tiny[0][[2, 0]]

    assert boards.shape == (2, TOKENS_PER_BOARD)
    assert moves[:, 0].tolist() == [2, 0]


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

    #every position exactly once per epoch, the short last batch included, in a new order each time
    assert sorted(first_epoch) == sorted(second_epoch) == list(range(300))
    assert first_epoch != second_epoch
    assert loader_moves(val_loader) == list(range(20))


def test_train_order_depends_on_the_seed_alone(tiny):
    #building a model draws a different amount of random numbers for every architecture
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


#8 self play target moves against 2 stockfish ones: the narrower part is padded, and the smaller set starts over
#when it runs out (6 batches take 24 of the 20 stockfish positions)
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
    (folder / f"{name}_train_worst_moves.bin").write_bytes(moves.tobytes())
    (folder / f"{name}_train_worst_losses.bin").write_bytes(losses.tobytes())


def test_blunders_are_the_labelled_moves_that_give_away_enough(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PROCESSED_DATA_DIR", tmp_path)
    write_training_set(tmp_path, "blunders", {"train": 4, "val": 2})
    write_blunder_labels(tmp_path, "blunders", 4)

    train_dataset, val_dataset = load_training_set("blunders", TEMPERATURE)

    blunders = train_dataset[[0, 1]][4]
    assert blunders.tolist() == [[700, 701, NO_MOVE], [NO_MOVE] * 3]
    #the labels are read with the rest of the set, so warming the file cache covers them
    assert len(train_dataset.tensors()) == 5
    #val has no blunder labels of its own
    assert val_dataset[[0, 1]][4].tolist() == [[NO_MOVE], [NO_MOVE]]

    #a higher threshold leaves only the bigger loss, a lower one takes the harmless move too
    strict, _ = load_training_set("blunders", TEMPERATURE, blunder_threshold=0.5)
    assert strict[[0]][4].tolist() == [[700, NO_MOVE, NO_MOVE]]
    lenient, _ = load_training_set("blunders", TEMPERATURE, blunder_threshold=0.05)
    assert lenient[[0]][4].tolist() == [[700, 701, 702]]


#self play sets take blunder labels the same way (tools.label_blunders labels any boards file): a set whose labels
#were ignored would train without the penalty and say nothing
def test_self_play_sets_take_blunder_labels_too(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PROCESSED_DATA_DIR", tmp_path)
    for split, positions in (("train", 4), ("val", 2)):
        (tmp_path / f"selfplay_{split}_boards.bin").write_bytes(np.full((positions, TOKENS_PER_BOARD), 3, dtype=np.uint8).tobytes())
        (tmp_path / f"selfplay_{split}_visit_moves.bin").write_bytes(np.tile(np.arange(2, dtype=np.int16), (positions, 1)).tobytes())
        (tmp_path / f"selfplay_{split}_visit_shares.bin").write_bytes(np.full((positions, 2), 0.5, dtype=np.float32).tobytes())
        (tmp_path / f"selfplay_{split}_outcomes.bin").write_bytes(np.ones(positions, dtype=np.float32).tobytes())
        (tmp_path / f"selfplay_{split}_root_scores.bin").write_bytes(np.full(positions, 0.5, dtype=np.float32).tobytes())
    write_blunder_labels(tmp_path, "selfplay", 4)

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



#teacher targets for every split: moves 900 and 901 at 0.75 / 0.25, expected score 0.9
def write_teacher_targets(folder, name, positions_per_split):
    for split, positions in positions_per_split.items():
        (folder / f"{name}_{split}_teacher_moves.bin").write_bytes(np.tile(np.array([900, 901], dtype=np.int16), (positions, 1)).tobytes())
        (folder / f"{name}_{split}_teacher_probabilities.bin").write_bytes(np.tile(np.array([0.75, 0.25], dtype=np.float16), (positions, 1)).tobytes())
        (folder / f"{name}_{split}_teacher_scores.bin").write_bytes(np.full(positions, 0.9, dtype=np.float16).tobytes())


#the teacher's share of each target: its moves after stockfish's with the weights scaled so all still add up to 1,
#its score mixed into the value; val stays stockfish's alone so val loss compares across runs
def test_teacher_targets_take_their_share_of_the_train_targets(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PROCESSED_DATA_DIR", tmp_path)
    write_training_set(tmp_path, "taught", {"train": 4, "val": 2})
    write_teacher_targets(tmp_path, "taught", {"train": 4, "val": 2})
    stockfish_value = torch.sigmoid(torch.tensor(EXPECTED_SCORE_PER_CENTIPAWN * EVAL)).item()

    train_dataset, val_dataset = load_training_set("taught", TEMPERATURE, teacher_weight=0.25)
    _, moves, probabilities, values, _ = train_dataset[[1]]
    assert moves.tolist() == [[1, 0, 900, 901]]
    assert probabilities[0].tolist() == pytest.approx([0.75, 0.0, 0.25 * 0.75, 0.25 * 0.25])
    assert values.item() == pytest.approx(0.75 * stockfish_value + 0.25 * 0.9, abs=1e-3)
    assert len(train_dataset.tensors()) == 6  #the teacher's files are warmed with the rest

    _, val_moves, val_probabilities, val_values, _ = val_dataset[[1]]
    assert val_moves.tolist() == [[1, 0]] and val_probabilities[0].tolist() == pytest.approx([1.0, 0.0])
    assert val_values.item() == pytest.approx(stockfish_value, abs=1e-6)

    #weight 0 reads no teacher at all: the same targets as a set without one
    plain, _ = load_training_set("taught", TEMPERATURE)
    assert plain[[1]][1].tolist() == [[1, 0]]


#a run asked to distil must not quietly train on stockfish alone
def test_a_teacher_weight_without_teacher_targets_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PROCESSED_DATA_DIR", tmp_path)
    write_training_set(tmp_path, "untaught", {"train": 4, "val": 2})
    with pytest.raises(FileNotFoundError, match="teacher_targets"):
        load_training_set("untaught", TEMPERATURE, teacher_weight=0.5)

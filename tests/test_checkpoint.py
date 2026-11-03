import json

import chess
import pytest
import torch

from model import checkpoint
from model.checkpoint import load_checkpoint, save_checkpoint
from model.model import ChessTransformer
from engine import mcts
from engine.player import load_model

CPU = torch.device("cpu")

#not the default size, so loading only works if the settings come from the file
SETTINGS = {"d_model": 48, "num_layers": 3, "n_heads": 6, "mlp_hidden": 96, "qk_ratio": [0.5, 0.25, 1.0], "kv_heads": 3, "value_bins": 16}


@pytest.mark.parametrize("extra_settings", [{}, {"loops": 1}], ids=["no_extra", "loops_1"])
def test_model_of_any_size_comes_back_with_the_same_weights(tmp_path, extra_settings):
    torch.manual_seed(0)
    model = ChessTransformer(**SETTINGS).eval()
    boards = torch.randint(0, 13, (4, 70))
    save_checkpoint(model, {**SETTINGS, **extra_settings}, tmp_path / "model.pt")

    loaded, loaded_settings = load_checkpoint(tmp_path / "model.pt", CPU)

    assert loaded_settings == SETTINGS
    for loaded_output, output in zip(loaded.eval()(boards), model(boards)):
        assert torch.equal(loaded_output, output)


def test_a_checkpoint_that_runs_its_blocks_more_than_once_is_refused(tmp_path):
    save_checkpoint(ChessTransformer(**SETTINGS), {**SETTINGS, "loops": 2}, tmp_path / "model.pt")

    with pytest.raises(ValueError, match="loops"):
        load_checkpoint(tmp_path / "model.pt", CPU)


def test_player_loads_a_model_that_is_not_the_default_size(tmp_path):
    save_checkpoint(ChessTransformer(**SETTINGS), SETTINGS, tmp_path / "model.pt")

    model = load_model(tmp_path / "model.pt", CPU)

    assert not model.training
    assert mcts.most_visited_move(mcts.search(model, chess.Board(), CPU, 8)) in chess.Board().legal_moves


def test_latest_checkpoint_is_the_run_that_finished_last(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint, "RUNS_DIR", tmp_path)
    runs = {"older": "2026-09-20T10:00:00", "newer": "2026-09-21T09:00:00", "still_training": None, "no_checkpoint": "2026-09-22T09:00:00"}
    for name, finished_at in runs.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "settings.json").write_text(json.dumps({} if finished_at is None else {"finished_at": finished_at}))
        if name != "no_checkpoint":
            (tmp_path / name / "model.pt").write_bytes(b"")

    assert checkpoint.latest_checkpoint() == tmp_path / "newer" / "model.pt"


def test_no_finished_run_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint, "RUNS_DIR", tmp_path)

    with pytest.raises(FileNotFoundError):
        checkpoint.latest_checkpoint()

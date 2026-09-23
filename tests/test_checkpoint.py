import json

import chess
import pytest
import torch

from tools import checkpoint
from tools.checkpoint import load_checkpoint, save_checkpoint
from model import ChessTransformer
from tools.play.player import choose_move, load_model

CPU = torch.device("cpu")

#not the default size, so loading only works if the settings really come from the file
SETTINGS = {"d_model": 48, "num_layers": 3, "n_heads": 6, "qk_ratio": [0.5, 0.25, 1.0], "kv_heads": 3}


def test_model_of_any_size_comes_back_with_the_same_weights(tmp_path):
    torch.manual_seed(0)
    model = ChessTransformer(**SETTINGS).eval()
    boards = torch.randint(0, 13, (4, 70))
    save_checkpoint(model, SETTINGS, tmp_path / "model.pt")

    loaded, loaded_settings = load_checkpoint(tmp_path / "model.pt", CPU)

    assert loaded_settings == SETTINGS
    for loaded_output, output in zip(loaded.eval()(boards), model(boards)):
        assert torch.equal(loaded_output, output)


def test_player_loads_a_model_that_is_not_the_default_size(tmp_path):
    save_checkpoint(ChessTransformer(**SETTINGS), SETTINGS, tmp_path / "model.pt")

    model = load_model(tmp_path / "model.pt", CPU)

    assert not model.training
    assert choose_move(model, chess.Board(), CPU) in chess.Board().legal_moves


def test_bare_weights_are_rejected_with_an_explanation(tmp_path):
    torch.save(ChessTransformer().state_dict(), tmp_path / "old.pt")

    with pytest.raises(ValueError, match="model settings"):
        load_checkpoint(tmp_path / "old.pt", CPU)


#the settings every checkpoint from before the cleanup has, which the model no longer takes
@pytest.mark.parametrize("old_settings", [{"dropout": 0.0}, {"history": 0, "future": 0, "swiglu": True, "dropout": 0.0, "value_target": "eval"}])
def test_checkpoint_from_before_the_cleanup_points_to_the_tag(tmp_path, old_settings):
    torch.save({"model_settings": {**SETTINGS, **old_settings}, "state_dict": {}}, tmp_path / "old.pt")

    with pytest.raises(ValueError, match="pre-cleanup-2026-09-23"):
        load_checkpoint(tmp_path / "old.pt", CPU)


def test_latest_checkpoint_is_the_run_that_finished_last(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint, "RUNS_DIR", tmp_path)
    runs = {"older": "2026-09-20T10:00:00", "newer": "2026-09-21T09:00:00", "still_training": None, "no_checkpoint": "2026-09-22T09:00:00"}
    for name, finished_at in runs.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "settings.json").write_text(json.dumps({} if finished_at is None else {"finished_at": finished_at}))
        if name != "no_checkpoint":
            (tmp_path / name / "model.pt").write_bytes(b"")

    #not the run that is still writing its checkpoint, however new its folder is
    assert checkpoint.latest_checkpoint() == tmp_path / "newer" / "model.pt"


def test_no_finished_run_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint, "RUNS_DIR", tmp_path)

    with pytest.raises(FileNotFoundError):
        checkpoint.latest_checkpoint()

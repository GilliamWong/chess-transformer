import json

import chess
import pytest
import torch

from tools import checkpoint
from tools.checkpoint import load_checkpoint, save_checkpoint, warm_start
from model import ChessTransformer
from tools.play.player import choose_move, load_model

CPU = torch.device("cpu")

#not the default size, so loading only works if the settings really come from the file
SETTINGS = {"d_model": 48, "num_layers": 3, "n_heads": 6, "mlp_hidden": 96, "qk_ratio": [0.5, 0.25, 1.0], "kv_heads": 3, "loops": 2, "value_bins": 16}


def test_model_of_any_size_comes_back_with_the_same_weights(tmp_path):
    torch.manual_seed(0)
    model = ChessTransformer(**SETTINGS).eval()
    boards = torch.randint(0, 13, (4, 70))
    save_checkpoint(model, SETTINGS, tmp_path / "model.pt")

    loaded, loaded_settings = load_checkpoint(tmp_path / "model.pt", CPU)

    assert loaded_settings == SETTINGS
    for loaded_output, output in zip(loaded.eval()(boards), model(boards)):
        assert torch.equal(loaded_output, output)


#the recipe runs (the d512 among them) saved template_maps=0: they still load, the setting dropped
def test_checkpoint_without_template_maps_still_loads(tmp_path):
    torch.manual_seed(0)
    model = ChessTransformer(**SETTINGS).eval()
    boards = torch.randint(0, 13, (4, 70))
    save_checkpoint(model, {**SETTINGS, "template_maps": 0}, tmp_path / "model.pt")

    loaded, loaded_settings = load_checkpoint(tmp_path / "model.pt", CPU)

    assert loaded_settings == SETTINGS
    for loaded_output, output in zip(loaded.eval()(boards), model(boards)):
        assert torch.equal(loaded_output, output)


#checkpoints from before mlp_hidden name mlp_ratio: they load at the width they trained at, where the rounding now in
#model.py would give 320, 672 and 672
@pytest.mark.parametrize("d_model, mlp_ratio, width", [(128, None, 344), (256, None, 688), (512, 1.333, 680)])
def test_a_checkpoint_from_before_mlp_hidden_loads_at_its_old_width(tmp_path, d_model, mlp_ratio, width):
    old = ChessTransformer(d_model=d_model, num_layers=1, n_heads=4, mlp_hidden=width)
    save_checkpoint(old, {"d_model": d_model, "num_layers": 1, "n_heads": 4, "mlp_ratio": mlp_ratio, "value_bins": 128}, tmp_path / "model.pt")

    loaded, loaded_settings = load_checkpoint(tmp_path / "model.pt", CPU)

    assert loaded_settings["mlp_hidden"] == width and "mlp_ratio" not in loaded_settings
    assert loaded.encoder[0].mlp.up_projection.out_features == width


#the warm start of the long d512 runs: a 1-bin checkpoint of an old width goes on with 128 bins. Everything but the
#eval head's last layer has to carry over, and the size has to come from the file
def test_a_warm_start_with_other_bins_renews_only_the_last_layer_of_the_eval_head(tmp_path):
    torch.manual_seed(0)
    old_settings = {"d_model": 32, "num_layers": 2, "n_heads": 2, "mlp_hidden": 40, "value_bins": 1}
    old = ChessTransformer(**old_settings)
    save_checkpoint(old, old_settings, tmp_path / "model.pt")

    model, settings = warm_start(tmp_path / "model.pt", CPU, value_bins=16)

    assert settings == {**old_settings, "value_bins": 16}
    new_weights, old_weights = model.state_dict(), old.state_dict()
    assert new_weights.keys() == old_weights.keys()
    changed = {name for name in new_weights if new_weights[name].shape != old_weights[name].shape or not torch.equal(new_weights[name], old_weights[name])}
    assert changed == {"eval_head.2.weight", "eval_head.2.bias"}
    assert model.eval_head[-1].out_features == 16
    #the same bins: the checkpoint as it is
    same, _ = warm_start(tmp_path / "model.pt", CPU, value_bins=1)
    assert all(torch.equal(tensor, old_weights[name]) for name, tensor in same.state_dict().items())


def test_checkpoint_with_template_maps_points_to_the_commit_that_has_them(tmp_path):
    torch.save({"model_settings": {**SETTINGS, "template_maps": 192}, "state_dict": {}}, tmp_path / "old.pt")

    with pytest.raises(ValueError, match="1a976d9"):
        load_checkpoint(tmp_path / "old.pt", CPU)


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

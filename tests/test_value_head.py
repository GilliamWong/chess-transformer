import pytest
import torch

from main import value_correct, value_error, value_loss, value_target_bins
from model import ChessTransformer, expected_score
from tools.checkpoint import load_checkpoint, save_checkpoint

CPU = torch.device("cpu")


def test_the_target_over_the_bins_is_a_gaussian_around_the_score():
    targets = torch.tensor([0.3, 0.5, 0.999])
    bins = 128

    target = value_target_bins(targets, bins)

    assert target.shape == (3, bins)
    assert torch.allclose(target.sum(dim=-1), torch.ones(3), atol=1e-6)
    centres = (torch.arange(bins) + 0.5) / bins
    #the mass sits on the score's bin and its neighbours, so the target's mean is the score
    assert torch.allclose(target[:2] @ centres, targets[:2], atol=1e-3)
    assert (target[0] > 1e-3).sum() <= 5
    #a score at the edge keeps all of its mass inside the bins
    assert target[2].argmax() == bins - 1


#a head that puts everything on the score's bin has scored it, and a wider spread has scored worse
def test_the_bins_loss_is_lowest_on_the_target_itself():
    targets = torch.tensor([0.3, 0.7])
    bins = 16
    on_target = torch.log(value_target_bins(targets, bins) + 1e-12)
    flat = torch.zeros(2, bins)

    assert value_loss(on_target, targets) < value_loss(flat, targets)
    #the older head: one logit, binary cross entropy
    one_logit = torch.logit(targets).unsqueeze(-1)
    assert value_loss(one_logit, targets) < value_loss(torch.zeros(2, 1), targets)


def test_the_expected_score_is_the_head_output_as_a_score():
    #one logit is the score's logit
    assert torch.allclose(expected_score(torch.logit(torch.tensor([[0.2], [0.9]]))), torch.tensor([0.2, 0.9]))
    #bins: everything on one bin is that bin's centre
    logits = torch.full((1, 4), -1e4)
    logits[0, 2] = 0
    assert torch.allclose(expected_score(logits), torch.tensor([0.625]))

    targets = torch.tensor([0.2, 0.9])
    bins = torch.log(value_target_bins(targets, 64) + 1e-12)
    assert torch.allclose(expected_score(bins), targets, atol=1e-3)
    assert value_correct(bins, targets) == 2
    assert value_error(bins, targets) < 1e-2


#every checkpoint saved before the bins has one logit and no value_bins setting
def test_a_checkpoint_from_before_the_bins_loads_with_one_logit(tmp_path):
    settings = {"d_model": 32, "num_layers": 1, "n_heads": 2}
    #its mlp is the width of before mlp_hidden too
    old = ChessTransformer(**settings, value_bins=1, mlp_hidden=88)
    save_checkpoint(old, settings, tmp_path / "model.pt")

    loaded, loaded_settings = load_checkpoint(tmp_path / "model.pt", CPU)

    assert loaded_settings["value_bins"] == 1
    assert loaded.eval_head[-1].out_features == 1


def test_the_bins_are_a_stored_setting(tmp_path):
    settings = {"d_model": 32, "num_layers": 1, "n_heads": 2, "mlp_hidden": 64, "value_bins": 8}
    save_checkpoint(ChessTransformer(**settings), settings, tmp_path / "model.pt")

    loaded, _ = load_checkpoint(tmp_path / "model.pt", CPU)

    assert loaded.eval_head[-1].out_features == 8


def test_no_bins_is_rejected():
    with pytest.raises(ValueError):
        ChessTransformer(d_model=32, num_layers=1, n_heads=2, value_bins=0)

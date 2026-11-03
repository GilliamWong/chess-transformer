import pytest
import torch

from training.train import value_correct, value_error, value_loss, value_target_bins
from model.model import ChessTransformer, expected_score


def test_the_target_over_the_bins_is_a_gaussian_around_the_score():
    targets = torch.tensor([0.3, 0.5, 0.999])
    bins = 128

    target = value_target_bins(targets, bins)

    assert target.shape == (3, bins)
    assert torch.allclose(target.sum(dim=-1), torch.ones(3), atol=1e-6)
    centres = (torch.arange(bins) + 0.5) / bins
    assert torch.allclose(target[:2] @ centres, targets[:2], atol=1e-3)
    assert (target[0] > 1e-3).sum() <= 5
    #a score at the edge keeps all of its mass inside the bins
    assert target[2].argmax() == bins - 1


def test_the_bins_loss_is_lowest_on_the_target_itself():
    targets = torch.tensor([0.3, 0.7])
    bins = 16
    on_target = torch.log(value_target_bins(targets, bins) + 1e-12)
    flat = torch.zeros(2, bins)

    assert value_loss(on_target, targets) < value_loss(flat, targets)


def test_the_expected_score_is_the_head_output_as_a_score():
    logits = torch.full((1, 4), -1e4)
    logits[0, 2] = 0
    assert torch.allclose(expected_score(logits), torch.tensor([(2 + 0.5) / 4]))

    targets = torch.tensor([0.2, 0.9])
    bin_logits = torch.log(value_target_bins(targets, 64) + 1e-12)
    assert torch.allclose(expected_score(bin_logits), targets, atol=1e-3)
    assert value_correct(bin_logits, targets) == 2
    assert value_error(bin_logits, targets) < 1e-2


@pytest.mark.parametrize("value_bins", [0, 1])
def test_no_bins_is_rejected(value_bins):
    with pytest.raises(ValueError):
        ChessTransformer(d_model=32, num_layers=1, n_heads=2, value_bins=value_bins)

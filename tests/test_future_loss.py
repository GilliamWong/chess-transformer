import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data import FUTURE_UNKNOWN
from main import evaluate_model, future_loss


@pytest.mark.parametrize("known_rows", [0, 1, 3])
def test_future_loss_ignores_unknown_targets_and_handles_an_empty_mask(known_rows):
    logits = torch.zeros(3, 64, 14, requires_grad=True)
    targets = torch.full((3, 64), FUTURE_UNKNOWN, dtype=torch.long)
    targets[:known_rows] = 0

    loss = future_loss(logits, targets)
    assert loss.item() == pytest.approx(math.log(14) if known_rows else 0.0)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[known_rows:]) == 0
    if known_rows:
        assert torch.count_nonzero(logits.grad[:known_rows]) > 0


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("known_targets", [True, False])
def test_validation_future_loss_is_independent_of_batch_grouping(batch_size, known_targets):
    future_logits = torch.zeros(4, 64, 14)
    future_logits[2:, :, 0] = 4
    targets = torch.zeros(4, 64, dtype=torch.long)
    targets[1] = FUTURE_UNKNOWN
    if not known_targets:
        targets.fill_(FUTURE_UNKNOWN)

    class FixedModel(nn.Module):
        def forward(self, boards, history):
            return torch.zeros(len(boards), 2), torch.zeros(len(boards), 3), future_logits[boards[:, 0]]

    dataset = TensorDataset(
        torch.arange(4).unsqueeze(1), torch.zeros(4, 1, dtype=torch.long), torch.ones(4, 1),
        torch.zeros(4, dtype=torch.long), torch.empty(4, 0, 64, dtype=torch.long), targets,
    )
    measured = evaluate_model(FixedModel(), DataLoader(dataset, batch_size=batch_size), nn.CrossEntropyLoss(), torch.device("cpu"))[-1]

    if known_targets:
        #three labelled boards: one uniform prediction and two assigning logit 4 to the correct class
        expected = (math.log(14) + 2 * math.log1p(13 * math.exp(-4))) / 3
        assert measured == pytest.approx(expected, rel=1e-6)
    else:
        assert math.isnan(measured)  #no supervised validation targets means the metric is unavailable

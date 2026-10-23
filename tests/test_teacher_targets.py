import torch

from model.model import expected_score
from tools.teacher_targets import teacher_targets


#the top moves by the teacher's probability, renormalised over those moves, and its expected score
def test_teacher_targets_are_the_top_moves_renormalised():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    score = expected_score(torch.tensor([[0.0, 2.0, 0.0]]))

    moves, probabilities, scores = teacher_targets(torch.cat([logits, score[:, None]], dim=1), moves=2)

    assert moves.tolist() == [[0, 1]]
    assert torch.allclose(probabilities, torch.tensor([[0.625, 0.375]]))
    assert torch.equal(scores, score)

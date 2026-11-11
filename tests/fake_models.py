import numpy as np
import torch
from torch import nn

from engine.inference import cpu_batches
from model.encoding import NUM_MOVE_IDS, NUM_SQUARES, TOKENS_PER_BOARD

#in pawns by board token, positive for the side to move: boards are encoded from its side
PIECE_VALUES = torch.tensor([0, 1, 3, 3, 5, 9, 0, -1, -3, -3, -5, -9, 0], dtype=torch.float)

#about sigmoid(pawns) as a score: all the weight on the end bins of 64
def value_logits(pawns: torch.Tensor) -> torch.Tensor:
    logits = torch.full((len(pawns), 64), -1e4)
    logits[:, 0] = 0
    logits[:, -1] = pawns
    return logits


class UniformModel(nn.Module):
    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(len(boards), NUM_MOVE_IDS), torch.zeros(len(boards), 2)

class BoardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.randn(TOKENS_PER_BOARD, NUM_MOVE_IDS, generator=torch.Generator().manual_seed(0))

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return boards.float() @ self.weights / 100, value_logits(PIECE_VALUES[boards[:, :NUM_SQUARES].long()].sum(dim=1))

#every move the same prior, the score BoardModel's
class EvenModel(nn.Module):
    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(len(boards), NUM_MOVE_IDS), value_logits(PIECE_VALUES[boards[:, :NUM_SQUARES].long()].sum(dim=1))

class CountingNetwork:
    def __init__(self, model):
        self.network = cpu_batches(model)
        self.boards = 0

    def launch(self, boards: np.ndarray) -> np.ndarray:
        self.boards += len(boards)
        return self.network.launch(boards)

    def collect(self, launched: np.ndarray) -> np.ndarray:
        return self.network.collect(launched)

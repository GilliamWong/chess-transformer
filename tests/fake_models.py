import numpy as np
import torch
from torch import nn

from engine.inference import cpu_batches
from model.encoding import NUM_MOVE_IDS

#material in pawns by board token: the side to move's pieces count for it, the other side's against
PIECE_VALUES = torch.tensor([0, 1, 3, 3, 5, 9, 0, -1, -3, -3, -5, -9, 0], dtype=torch.float)


#every move equally likely and every position even: whatever a search finds, it found by searching
class UniformModel(nn.Module):
    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(len(boards), NUM_MOVE_IDS), torch.zeros(len(boards), 1)

#priors of the board's own, and a position scored by its material, so that some games are won
class BoardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.randn(70, NUM_MOVE_IDS, generator=torch.Generator().manual_seed(0))

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return boards.float() @ self.weights / 100, PIECE_VALUES[boards[:, :64].long()].sum(dim=1, keepdim=True)

class CountingNetwork:
    def __init__(self, model):
        self.network = cpu_batches(model)
        self.boards = 0

    def launch(self, boards: np.ndarray) -> np.ndarray:
        self.boards += len(boards)
        return self.network.launch(boards)

    def collect(self, launched: np.ndarray) -> np.ndarray:
        return self.network.collect(launched)

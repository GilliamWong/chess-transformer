#stand-in networks shared by the search and self play tests
import torch
from torch import nn

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

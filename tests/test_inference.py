import chess
import numpy as np
import pytest
import torch

from encoding import encode_board
from inference import GraphedBatches
from model import ChessTransformer
from tools.play.player import CudaGraphModel


#for a batch smaller than the graph and for one board; graphs recorded on zero boards once hit a device-side assert
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
def test_graphs_answer_like_the_model():
    torch.manual_seed(0)
    model = ChessTransformer(d_model=32, num_layers=1, n_heads=2).cuda().eval()
    board, boards = chess.Board(), []
    for san in ("e4", "d5", "exd5"):
        board.push_san(san)
        boards.append(encode_board(board))
    boards = np.array(boards, dtype=np.uint8)
    with torch.no_grad():
        expected = torch.cat(model(torch.from_numpy(boards).long().cuda()), dim=1).float().cpu().numpy()

    graphs = GraphedBatches(model, 4)
    answers = graphs.collect(graphs.launch(boards))
    logits, value = CudaGraphModel(model)(torch.from_numpy(boards[:1]).long())
    torch.cuda.synchronize()

    assert np.allclose(answers, expected, atol=1e-3)
    assert np.allclose(torch.cat([logits, value], dim=1).numpy(), expected[:1], atol=1e-3)

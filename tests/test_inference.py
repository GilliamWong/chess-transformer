import chess
import numpy as np
import pytest
import torch

from encoding import encode_board
from inference import GraphedBatches, answer_rows, graph_sizes
from model import ChessTransformer, expected_score
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
        expected = answer_rows(model(torch.from_numpy(boards).long().cuda())).cpu().numpy()

    graphs = GraphedBatches(model, graph_sizes(4))
    answers = graphs.collect(graphs.launch(boards))
    logits, value = CudaGraphModel(model)(torch.from_numpy(boards[:1]).long())
    torch.cuda.synchronize()

    assert np.allclose(answers, expected, atol=1e-3)
    assert np.allclose(answer_rows((logits, value)).numpy(), expected[:1], atol=1e-3)


#the last number of a row is the score, not a logit: the search backs it up as it is
def test_an_answer_row_ends_with_the_expected_score():
    torch.manual_seed(0)
    model = ChessTransformer(d_model=32, num_layers=1, n_heads=2).eval()
    boards = torch.from_numpy(np.array([encode_board(chess.Board())], dtype=np.uint8)).long()
    with torch.no_grad():
        logits, value_logits = model(boards)
        rows = answer_rows((logits, value_logits))

    assert rows.shape == (1, logits.shape[1] + 1)
    assert 0 < rows[0, -1] < 1
    assert torch.equal(rows[0, -1], expected_score(value_logits)[0])

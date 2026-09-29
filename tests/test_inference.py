import copy

import chess
import numpy as np
import pytest
import torch

from encoding import EN_PASSANT_TOKEN_OFFSET, TOKENS_PER_BOARD, encode_board, encode_move_to_id
from inference import GraphedBatches, ImmediateBatches, MirroredBatches, answer_rows, fp16_model, graph_sizes, small_graph_sizes
from model import VALUE_BINS, ChessTransformer, expected_score, score_spread
from selfplay import ANSWER_WIDTH
from tools.checkpoint import save_checkpoint
from tools.play.player import CudaGraphModel


#every batch up to max_batch has a graph, and past 16 boards pads by fewer than 4 boards or a sixteenth: steps of 16
#cost the engine's rounds of 64 leaves 3% of its nodes/s
@pytest.mark.parametrize("max_batch", [64, 300])
def test_small_graph_sizes_pad_little(max_batch):
    sizes = small_graph_sizes(max_batch)
    assert sizes == sorted(set(sizes)) and sizes[0] == 1 and sizes[-1] >= max_batch
    for batch in range(17, max_batch + 1):
        padded = next(size for size in sizes if size >= batch)
        assert padded - batch < max(4, padded // 16)


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


#the graphs read the model's weights in place, so GraphedBatches must keep a model nothing else holds (as
#engine_network's) alive: freed, its memory went to the next tensors made and every answer came back wrong
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
def test_graphs_keep_their_model_alive():
    torch.manual_seed(0)
    model = ChessTransformer(d_model=32, num_layers=1, n_heads=2).cuda().eval()
    boards = np.array([encode_board(chess.Board())], dtype=np.uint8)
    with torch.no_grad():
        expected = answer_rows(model(torch.from_numpy(boards).long().cuda())).cpu().numpy()

    graphs = GraphedBatches(copy.deepcopy(model), graph_sizes(4))
    #tensors the size of the weights, which take the copy's memory if it was freed
    garbage = [torch.full_like(weight, 1e4) for weight in model.parameters() for _ in range(4)]
    answers = graphs.collect(graphs.launch(boards))

    assert np.allclose(answers, expected, atol=1e-3)


#the d512_selfplay_sf12_10pass run's residual stream passes fp16's 65,504 on the en-passant token: every board came back
#NaN and the engine's search put all its visits on one arbitrary move
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
def test_fp16_model_stays_finite_past_fp16s_range(tmp_path):
    torch.manual_seed(0)
    settings = {"d_model": 32, "num_layers": 1, "n_heads": 2, "mlp_hidden": 64, "value_bins": VALUE_BINS}
    model = ChessTransformer(**settings).eval()
    with torch.no_grad():
        #the token of a board without en passant, so on every board
        model.embed.weight[EN_PASSANT_TOKEN_OFFSET, 0] = 2e5
    save_checkpoint(model, settings, tmp_path / "model.pt")
    board, boards = chess.Board(), []
    for san in ("e4", "c5", "Nf3"):
        board.push_san(san)
        boards.append(encode_board(board))
    boards = torch.tensor(boards).cuda()

    #fp16_model sets these for the whole process
    flags = torch.backends.cuda.matmul.allow_fp16_accumulation, torch.backends.cuda.matmul.allow_tf32
    try:
        with torch.no_grad():
            logits, value_logits = fp16_model(tmp_path / "model.pt")(boards)
            expected_logits, expected_value_logits = model.cuda()(boards)
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation, torch.backends.cuda.matmul.allow_tf32 = flags

    assert torch.isfinite(logits).all() and torch.isfinite(value_logits).all()
    policy_distance = 0.5 * (torch.softmax(logits.float(), dim=-1) - torch.softmax(expected_logits, dim=-1)).abs().sum(dim=-1)
    assert policy_distance.max() < 0.01
    assert torch.allclose(expected_score(value_logits.float()), expected_score(expected_value_logits), atol=0.01)


#fused_blocks.FusedBlock must compute what model.TransformerBlock does, and SummedSquareFeatures what SquareFeatures
#does: with random templates and square features (zeros at init would hide them), grouped key/value heads and
#query/key widths of 8, 16 and 24 (padded to 16 and 32 in the kernel)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
def test_fused_blocks_compute_what_the_model_does():
    from fused_blocks import fuse_blocks
    torch.manual_seed(0)
    model = ChessTransformer(d_model=128, num_layers=3, n_heads=4, kv_heads=2, qk_ratio=[0.25, 0.5, 0.75], value_bins=16).cuda().eval()
    with torch.no_grad():
        model.templates.normal_()
        model.square_features.table.normal_(std=0.1)
    board, boards = chess.Board(), []
    for san in "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6 f3 Be7".split():
        board.push_san(san)
        boards.append(encode_board(board))
    boards = torch.tensor(boards).cuda()

    with torch.no_grad():
        logits, value_logits = model(boards)
        fused_logits, fused_value_logits = fuse_blocks(copy.deepcopy(model))(boards)

    assert torch.allclose(fused_logits, logits, atol=1e-2)
    assert torch.equal(fused_logits.argmax(-1), logits.argmax(-1))
    assert torch.allclose(expected_score(fused_value_logits), expected_score(value_logits), atol=1e-3)


#fused_blocks.linear in fp16 as the engine runs it, accumulating in fp16, with the bias and residual added (the
#attention's output projection) or silu(gate) * up (the mlp), on rows and columns that leave part of a tile empty
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
@pytest.mark.parametrize("gated", [False, True])
def test_linear_kernel_in_fp16(gated):
    from fused_blocks import linear
    torch.manual_seed(0)
    width = 200
    x = torch.randn(3, 70, 512, device="cuda").half()
    weight = (torch.randn(512, 2 * width if gated else width, device="cuda") / 512**0.5).half()
    bias = None if gated else torch.randn(width, device="cuda").half()
    residual = None if gated else torch.randn(3, 70, width, device="cuda").half()

    out = linear(x, weight, bias, residual, gated)

    projected = x.float() @ weight.float()
    if gated:
        gate, up = projected.chunk(2, dim=-1)
        expected = torch.nn.functional.silu(gate) * up
    else:
        expected = projected + bias.float() + residual.float()
    assert out.dtype == torch.float16 and out.shape == expected.shape
    assert torch.allclose(out.float(), expected, atol=0.03, rtol=0.01)


#a tile's extra rows (fused_blocks.linear_extra_rows) are a matmul of their own on the same weights: every row comes
#out the same bits whichever tile it falls in, in plain, gated and fp32 accumulating matmuls
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
@pytest.mark.parametrize("gated, accumulate_fp32", [(False, False), (True, False), (False, True)])
def test_extra_rows_leave_every_row_the_same(monkeypatch, gated, accumulate_fp32):
    import fused_blocks
    torch.manual_seed(0)
    x = torch.randn(1000, 512, device="cuda").half()
    weight = (torch.randn(512, 400 if gated else 200, device="cuda") / 512**0.5).half()
    bias = None if gated else torch.randn(200, device="cuda").half()

    outs = []
    for extra in (0, 16, 32, 64):
        monkeypatch.setattr(fused_blocks, "linear_extra_rows", lambda M, column_tiles, multiprocessors, extra=extra: extra)
        outs.append(fused_blocks.linear(x, weight, bias, None, gated, accumulate_fp32))

    assert all(torch.equal(out, outs[0]) for out in outs[1:])


#at 64 positions (4,480 rows) a 512 column matmul's 140 tiles of 128 rows take two waves of an RTX 4090's 128
#multiprocessors and 128 tiles of 144 rows one; tiles that fill their waves take no extra rows
def test_extra_rows_save_a_wave():
    fused_blocks = pytest.importorskip("fused_blocks")
    assert fused_blocks.linear_extra_rows(64 * 70, 4, 128) == 16
    assert fused_blocks.linear_extra_rows(56 * 70, 4, 128) == 0


#the policy head's from and to projections (fused_blocks.HalfInputLinear): fp16 inputs accumulating in fp32 and
#giving fp32, as close to the fp32 matmul as fp16's rounding of the weights allows
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
def test_linear_kernel_accumulating_in_fp32():
    from fused_blocks import linear
    torch.manual_seed(0)
    x = torch.randn(3, 64, 512, device="cuda").half()
    weight = (torch.randn(512, 200, device="cuda") / 512**0.5).half()
    bias = torch.randn(200, device="cuda")

    out = linear(x, weight, bias, accumulate_fp32=True)

    assert out.dtype == torch.float32
    assert torch.allclose(out, x.float() @ weight.float() + bias, atol=1e-4)


#a row ends with the spread of the value head's distribution and then the score, not a logit: the search backs it up
#as it is. The spread is the distribution's standard deviation: 0 with all its mass in one bin, 1/sqrt(12) for the
#uniform distribution from 0 to 1
def test_an_answer_row_ends_with_the_spread_and_the_expected_score():
    torch.manual_seed(0)
    model = ChessTransformer(d_model=32, num_layers=1, n_heads=2).eval()
    boards = torch.from_numpy(np.array([encode_board(chess.Board())], dtype=np.uint8)).long()
    with torch.no_grad():
        logits, value_logits = model(boards)
        rows = answer_rows((logits, value_logits))

    assert rows.shape == (1, logits.shape[1] + 2)
    assert 0 < rows[0, -1] < 1
    assert torch.equal(rows[0, -1], expected_score(value_logits)[0])
    assert torch.equal(rows[0, -2], score_spread(value_logits)[0])
    one_bin = torch.full((1, VALUE_BINS), -1e4)
    one_bin[0, 40] = 0.0
    assert score_spread(one_bin)[0].item() == pytest.approx(0.0, abs=1e-3)
    assert score_spread(torch.zeros(1, VALUE_BINS))[0].item() == pytest.approx(12 ** -0.5, rel=1e-3)


#MirroredBatches' answer for a board without castling rights is the average of the network's answers for the board and
#for its mirror image as python-chess makes it (Board.transform(flip_horizontal)), each legal move's logit averaged with
#its mirrored move's; a board that can still castle passes through unchanged. The network is an arbitrary function of
#the tokens, so a wrong square, en passant square or move id in the mirroring changes the answer
def test_mirrored_batches_average_a_board_with_its_mirror_image():
    weights = np.random.default_rng(0).normal(size=(TOKENS_PER_BOARD, ANSWER_WIDTH))
    answer = lambda boards: (np.asarray(boards, dtype=np.float64) @ weights / 100).astype(np.float32)
    boards = [chess.Board("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1"), chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
              chess.Board("2r2rk1/pp3ppp/2n1pn2/3p4/3P4/2N1PN2/PP3PPP/2R2RK1 w - - 0 15")]
    network = MirroredBatches(ImmediateBatches(answer))
    answers = network.collect(network.launch(np.array([encode_board(board) for board in boards], dtype=np.uint8)))

    for board, answered in zip(boards, answers):
        plain = answer([encode_board(board)])[0]
        if board.castling_rights:
            np.testing.assert_array_equal(answered, plain)
            continue
        image = board.transform(chess.flip_horizontal)
        mirrored = answer([encode_board(image)])[0]
        for move in board.legal_moves:
            image_move = chess.Move(move.from_square ^ 7, move.to_square ^ 7, move.promotion)
            assert image_move in image.legal_moves
            expected = (plain[encode_move_to_id(move, board.turn)] + mirrored[encode_move_to_id(image_move, image.turn)]) / 2
            assert answered[encode_move_to_id(move, board.turn)] == pytest.approx(expected, rel=1e-6)
        assert answered[-1] == pytest.approx((plain[-1] + mirrored[-1]) / 2, rel=1e-6)

import contextlib
import copy
import math
import sys
import time
from types import SimpleNamespace

import chess
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from model.encoding import EN_PASSANT_TOKEN_OFFSET, NUM_SQUARES, TOKENS_PER_BOARD, encode_board
import selfplay_rs
from engine import inference
from engine.inference import ExactSizeCompiled, GraphedBatches, answer_rows, cpu_batches, fp16_model, graph_sizes, small_graph_sizes
from engine.selfplay import ANSWER_WIDTH, rust_search_settings
from fake_models import BoardModel
from selfplay_rs import LEAN_COLUMNS, LEAN_ROW
from model.model import VALUE_BINS, ChessTransformer, expected_score
from model.checkpoint import save_checkpoint
from tests.test_model import mix_in_the_input

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")


def tiny_model() -> ChessTransformer:
    torch.manual_seed(0)
    return ChessTransformer(d_model=32, num_layers=1, n_heads=2).eval()


def boards_after(moves: str) -> np.ndarray:
    board, boards = chess.Board(), []
    for san in moves.split():
        board.push_san(san)
        boards.append(encode_board(board))
    return np.array(boards, dtype=np.uint8)


def model_answers(model: nn.Module, boards: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return answer_rows(model(torch.from_numpy(boards).long().cuda())).cpu().numpy()


#fp16_model sets torch's matmul flags for the whole process
@contextlib.contextmanager
def restoring_matmul_flags():
    flags = torch.backends.cuda.matmul.allow_fp16_accumulation, torch.backends.cuda.matmul.allow_tf32
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation, torch.backends.cuda.matmul.allow_tf32 = flags


@pytest.mark.parametrize("max_batch, chunk", [(64, None), (300, None), (64, 64), (256, 64), (300, 48)])
def test_small_graph_sizes_pad_little(max_batch, chunk):
    sizes = small_graph_sizes(max_batch, chunk)
    assert sizes == sorted(set(sizes)) and sizes[0] == 1 and sizes[-1] >= max_batch
    every = chunk or max_batch
    for batch in range(1, max_batch + 1):
        padded = next(size for size in sizes if size >= batch)
        assert math.ceil(padded / every) == math.ceil(batch / every)
        last = padded - (padded - 1) // every * every
        if last > 16:
            assert padded - batch < max(4, last // 16)
    assert all(whole in sizes for whole in range(every, max_batch + 1, every))


@pytest.mark.parametrize("l2_mb, width, lanes, chunk", [(96, 768, 2, 64), (96, 768, 1, 128), (48, 768, 2, 64), (48, 256, 2, 96),
                                                        (48, 128, 2, 192), (6, 768, 1, 64)])
def test_chunk_positions_fit_the_l2(monkeypatch, l2_mb, width, lanes, chunk):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(L2_cache_size=l2_mb << 20))
    assert inference.chunk_positions(width, lanes) == chunk


@gpu
def test_graphs_answer_like_the_model():
    model = tiny_model().cuda()
    boards = boards_after("e4 d5 exd5")
    expected = model_answers(model, boards)

    graphs = GraphedBatches(model, graph_sizes(4))
    answers = graphs.collect(graphs.launch(boards))
    torch.cuda.synchronize()

    assert np.allclose(answers, expected, atol=1e-3)


def test_exact_size_compiled_runs_each_size_through_its_compile(monkeypatch):
    compiled_for = []
    monkeypatch.setattr(torch, "compile", lambda model, dynamic: lambda boards: compiled_for.append((len(boards), dynamic)) or model(boards))
    model = ExactSizeCompiled(tiny_model(), 4)
    boards = torch.tensor([encode_board(chess.Board())] * 4)
    with torch.no_grad():
        model(boards)
        model(boards[:2])

    assert compiled_for == [(4, False), (2, True)]


#its compiles hold on one gpu only, as a network's do
class OnGpu(nn.Module):
    def __init__(self, gpu: int):
        super().__init__()
        self.gpu = gpu
        self.linear = nn.Linear(4, 4)

    def forward(self, boards: torch.Tensor) -> torch.Tensor:
        return self.linear(boards) + self.gpu


#dynamo caps a frame's compiles at its recompile_limit (8) across instances: every gpu's network must compile each size
def test_the_network_of_every_gpu_is_compiled(monkeypatch):
    compiles = []
    compile = torch.compile
    monkeypatch.setattr(torch, "compile", lambda model, dynamic: compile(model, dynamic=dynamic, backend=lambda graph, inputs: compiles.append(1) or graph.forward))
    #ExactSizeCompiled.forward counts 8 gpus; torch's own checks see the real count
    count = torch.cuda.device_count
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8 if sys._getframe(1).f_code is ExactSizeCompiled.forward.__code__ else count())
    boards = torch.ones(4, 4)
    try:
        for gpu in range(8):
            network = ExactSizeCompiled(OnGpu(gpu), 4)
            with torch.no_grad():
                for size in (1, 2, 4):
                    network(boards[:size])
    finally:
        torch._dynamo.reset()

    assert len(compiles) == 8 * 3


@gpu
def test_chunked_graphs_answer_like_the_model():
    model = tiny_model().cuda()
    boards = boards_after("e4 c5 Nf3 d6 d4 cxd4 Nxd4")
    expected = model_answers(model, boards)

    graphs = GraphedBatches(model, [8], chunk=3)
    answers = graphs.collect(graphs.launch(boards))

    assert np.allclose(answers, expected, atol=1e-3)


@gpu
def test_graphs_keep_their_model_alive():
    model = tiny_model().cuda()
    boards = np.array([encode_board(chess.Board())], dtype=np.uint8)
    expected = model_answers(model, boards)

    graphs = GraphedBatches(copy.deepcopy(model), graph_sizes(4))
    #tensors the size of the weights, which take the copy's memory if it was freed
    garbage = [torch.full_like(weight, 1e4) for weight in model.parameters() for _ in range(4)]
    answers = graphs.collect(graphs.launch(boards))

    assert np.allclose(answers, expected, atol=1e-3)


@gpu
def test_more_batches_out_than_lanes_are_refused():
    model = tiny_model().cuda()
    boards = np.array([encode_board(chess.Board())], dtype=np.uint8)
    graphs = GraphedBatches(model, [1], in_flight=1)
    launched = graphs.launch(boards)
    with pytest.raises(RuntimeError, match="more batches in flight than lanes"):
        graphs.launch(boards)
    graphs.collect(launched)
    graphs.collect(graphs.launch(boards))


@gpu
def test_batches_are_collected_in_any_order():
    boards = boards_after("e4 d5 exd5")
    graphs = GraphedBatches(tiny_model().cuda(), [1, 2])
    expected = graphs.collect(graphs.launch(boards[1:])).copy()

    first, second = graphs.launch(boards[:1]), graphs.launch(boards[1:])
    graphs.collect(second)
    third = graphs.launch(boards[1:])
    graphs.collect(first)

    assert np.array_equal(graphs.collect(third), expected)


@gpu
def test_blocking_wait_sleeps_through_the_batch():
    model = tiny_model().cuda()
    boards = np.array([encode_board(chess.Board())], dtype=np.uint8)
    spinning = GraphedBatches(model, [1], in_flight=1)
    expected = spinning.collect(spinning.launch(boards)).copy()

    graphs = GraphedBatches(model, [1], in_flight=1, blocking_wait=True)
    #a few tenths of a second of gpu work ahead of the batch on its lane
    with torch.cuda.stream(graphs.lanes[0].stream):
        torch.cuda._sleep(500_000_000)
    launched = graphs.launch(boards)
    started, started_cpu = time.perf_counter(), time.thread_time()
    answers = graphs.collect(launched)
    waited, cpu = time.perf_counter() - started, time.thread_time() - started_cpu

    assert np.array_equal(answers, expected)
    assert waited > 0.05 and cpu < 0.25 * waited


@gpu
def test_boards_written_into_the_lane_answer_as_launched():
    model = tiny_model().cuda()
    searches = selfplay_rs.Searches(**rust_search_settings(False))
    searches.start(0, chess.STARTING_FEN, [], 100)
    searches.finish(cpu_batches(BoardModel()).evaluate_boards(np.frombuffer(searches.descend(), dtype=np.uint8).reshape(-1, TOKENS_PER_BOARD)))
    copied, written = GraphedBatches(model, [8, 16], in_flight=2), GraphedBatches(model, [8, 16], in_flight=2)
    for _ in range(2):
        boards = np.frombuffer(searches.descend(12, 12, 36), dtype=np.uint8).reshape(-1, TOKENS_PER_BOARD)
        lane = written.free_lane()
        lane.boards_array[:len(boards)] = boards
        expected = copied.collect(copied.launch(boards)).copy()
        answers = written.collect(written.launch_filled(lane, len(boards)))
        assert len(boards) == 12 and np.array_equal(answers, expected)
        searches.finish(expected)


@gpu
@pytest.mark.parametrize("blocking_wait", [False, True])
def test_lean_lanes_answer_the_columns_of_whole_rows(blocking_wait):
    model = tiny_model().cuda()
    searches = selfplay_rs.Searches(**rust_search_settings(False))
    searches.start(0, chess.STARTING_FEN, [], 200)
    whole, lean = GraphedBatches(model, [8, 16], 2, blocking_wait=blocking_wait), GraphedBatches(model, [8, 16], 2, blocking_wait=blocking_wait, lean=True)
    for _ in range(6):
        lane = lean.free_lane()
        written, _ = searches.descend_into(lane.rows_array, 12, 12, 36, legal_columns=True)
        boards = lane.rows_array[:written, :TOKENS_PER_BOARD].copy()
        columns = lane.rows_array[:written, TOKENS_PER_BOARD:].copy().view("<i2").astype(np.int64)
        rows = whole.collect(whole.launch(boards)).copy()
        answers = lean.collect(lean.launch_filled(lane, written)).copy()
        assert written and np.array_equal(answers, np.take_along_axis(rows, columns, axis=1))
        searches.finish(answers)


@gpu
def test_batches_in_flight_together_answer_as_alone():
    torch.manual_seed(0)
    #big enough for the two batches to overlap
    model = ChessTransformer(d_model=256, num_layers=4, n_heads=4).cuda().eval()
    random = np.random.default_rng(0)
    board, boards = chess.Board(), []
    while len(boards) < 512:
        board = chess.Board() if board.is_game_over() else board
        board.push(random.choice(list(board.legal_moves)))
        boards.append(encode_board(board))
    batches = np.array(boards, dtype=np.uint8).reshape(2, 256, -1)

    alone = GraphedBatches(model, [256], in_flight=1)
    expected = [alone.collect(alone.launch(batch)).copy() for batch in batches]
    together = GraphedBatches(model, [256])
    for _ in range(20):
        launched = [together.launch(batch) for batch in batches]
        answers = [together.collect(batch_launched).copy() for batch_launched in launched]
        assert np.array_equal(answers, expected)

    columns = random.integers(0, ANSWER_WIDTH, size=(2, 256, LEAN_COLUMNS))
    rows = np.zeros((2, 256, LEAN_ROW), dtype=np.uint8)
    rows[:, :, :TOKENS_PER_BOARD] = batches
    rows[:, :, TOKENS_PER_BOARD:] = columns.astype("<i2").view(np.uint8).reshape(2, 256, -1)
    expected = [np.take_along_axis(rows_alone, batch_columns, axis=1) for rows_alone, batch_columns in zip(expected, columns)]
    lean = GraphedBatches(model, [256], lean=True)
    for _ in range(20):
        launched = []
        for batch_rows in rows:
            lane = lean.free_lane()
            lane.rows_array[:256] = batch_rows
            launched.append(lean.launch_filled(lane, 256))
        answers = [lean.collect(batch_launched).copy() for batch_launched in launched]
        assert np.array_equal(answers, expected)


@gpu
def test_fp16_model_stays_finite_past_fp16s_range(tmp_path):
    torch.manual_seed(0)
    settings = {"d_model": 32, "num_layers": 1, "n_heads": 2, "mlp_hidden": 64, "value_bins": VALUE_BINS}
    model = ChessTransformer(**settings).eval()
    mix_in_the_input(model)
    with torch.no_grad():
        #the token of a board without en passant, so on every board
        model.embed.weight[EN_PASSANT_TOKEN_OFFSET, 0] = 2e5
    save_checkpoint(model, settings, tmp_path / "model.pt")
    boards = torch.from_numpy(boards_after("e4 c5 Nf3")).long().cuda()

    with restoring_matmul_flags(), torch.no_grad():
        rows = answer_rows(fp16_model(tmp_path / "model.pt")(boards))
        expected_logits, expected_value_logits = model.cuda()(boards)

    assert torch.isfinite(rows).all()
    policy_distance = 0.5 * (torch.softmax(rows[:, :-1], dim=-1) - torch.softmax(expected_logits, dim=-1)).abs().sum(dim=-1)
    assert policy_distance.max() < 0.01
    assert torch.allclose(rows[:, -1], expected_score(expected_value_logits), atol=0.01)


@gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_the_input_embedding_is_the_models(dtype):
    from model.fused_blocks import InputEmbedding
    torch.manual_seed(0)
    #a width that fills no power of two
    model = ChessTransformer(d_model=96, num_layers=1, n_heads=4).cuda().eval()
    with torch.no_grad():
        model.square_features.table.normal_()
    boards = torch.from_numpy(boards_after("e4 Nf6 e5 d5")).long().cuda()
    model.half_except_heads(dtype)

    with torch.no_grad():
        found = InputEmbedding(model)(boards)
        features = F.pad(model.square_features.table.double()[model.square_features.indices(boards) + model.square_features.row_offsets].sum(-2),
                         (0, 0, 0, TOKENS_PER_BOARD - NUM_SQUARES))
        expected = (model.embed.weight.double()[boards] + model.position_embed.double() + features.to(dtype).double()).to(dtype)

    #fp16: a feature sum's rounding may differ by a step from fp64's
    rtol, atol = (1e-3, 1e-2) if dtype == torch.float16 else (1e-6, 1e-5)
    assert found.dtype == dtype and torch.allclose(found, expected, rtol=rtol, atol=atol)


@gpu
def test_the_board_embedding_finds_the_feature_rows_of_square_features(monkeypatch):
    from model import fused_blocks
    monkeypatch.setattr(fused_blocks, "fused_ends", lambda device: True)
    model = tiny_model().cuda()
    with torch.no_grad():
        #whole-number tables make the sums exact
        for weight in (model.embed.weight, model.position_embed, model.square_features.table):
            weight.copy_(torch.randint(-512, 512, weight.shape))
    fens = ["Q6k/1Q6/8/8/8/8/1q6/K6q w - - 0 1", "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            "7K/3N1N2/2n3n1/8/8/2B1b3/8/k7 b - - 0 1", "8/8/8/8/8/8/8/K6k w - - 0 1"]
    random, board = np.random.default_rng(0), chess.Board()
    while len(fens) < 600:
        board = chess.Board() if board.is_game_over() else board
        board.push(random.choice(list(board.legal_moves)))
        fens.append(board.fen())
    boards = torch.tensor([encode_board(chess.Board(fen)) for fen in fens]).cuda()

    with torch.no_grad():
        found = fused_blocks.InputEmbedding(model)(boards)
        features = F.pad(model.square_features(boards), (0, 0, 0, TOKENS_PER_BOARD - NUM_SQUARES))
        expected = model.embed(boards) + model.position_embed + features

    assert torch.equal(found, expected)


@gpu
#71 boards take several eval-head programs
@pytest.mark.parametrize("boards", [1, 5, 71])
def test_the_fused_heads_compute_the_models_heads(monkeypatch, boards):
    from model import fused_blocks
    monkeypatch.setattr(fused_blocks, "fused_ends", lambda device: True)
    torch.manual_seed(0)
    d = 64
    model = ChessTransformer(d_model=d, num_layers=1, n_heads=2, value_bins=16).cuda().eval()
    with torch.no_grad():
        #sixteenths and gains of powers of two, so that fp16 holds the matmul's weights
        model.from_projection.weight.copy_(torch.randint(-2, 3, (d, d)) / 16)
        model.to_projection.weight.copy_(torch.randint(-2, 3, (d, d)) / 16)
        model.ln.weight.copy_(2.0 ** torch.randint(-1, 2, (d,)))
    model.half_except_heads(torch.float16)
    stream = torch.randn(boards, TOKENS_PER_BOARD, d, device="cuda").half()
    fused = fused_blocks.FusedTransformer(model)

    with torch.no_grad():
        rows = fused.answers(stream)
        y = F.rms_norm(stream.double(), (d,), model.ln.weight.double(), model.ln.eps)
        linear = lambda layer, x: F.linear(x, layer.weight.double(), layer.bias.double())
        squares = y[:, :NUM_SQUARES]
        logits = linear(model.from_projection, squares) @ linear(model.to_projection, squares).transpose(1, 2) / math.sqrt(d)
        logits = logits[:, model.from_squares, model.to_squares]
        logits[:, model.promotion_move_ids] += linear(model.promotion_projection, squares)[:, model.promotion_to_squares, model.promotion_pieces]
        compressed = F.gelu(linear(model.board_down_project[0], y)).reshape(boards, -1)
        value_logits = linear(model.eval_head[2], F.gelu(linear(model.eval_head[0], compressed)))

    assert rows.shape == (boards, logits.shape[1] + 1)
    assert torch.allclose(rows[:, :-1].double(), logits, rtol=1e-5, atol=1e-5)
    assert torch.allclose(rows[:, -1], expected_score(value_logits), rtol=1e-5, atol=1e-6)


@gpu
@pytest.mark.parametrize("capability", [(8, 0), (8, 6), (8, 9), (12, 0)])
def test_the_fused_ends_only_on_sm_120(monkeypatch, capability):
    from model.fused_blocks import FusedTransformer
    torch.manual_seed(0)
    model = ChessTransformer(d_model=64, num_layers=1, n_heads=2, value_bins=16).cuda().eval()
    boards = torch.tensor([encode_board(chess.Board())] * 3).cuda()
    with torch.no_grad():
        expected = answer_rows(model(boards))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    fused = FusedTransformer(model.half_except_heads(torch.float16))
    with torch.no_grad():
        outputs = fused(boards)

    is_sm_120 = capability == (12, 0)
    assert fused.fused_heads == fused.input_embedding.from_boards == is_sm_120
    assert isinstance(outputs, torch.Tensor) == is_sm_120
    assert torch.allclose(answer_rows(outputs), expected, atol=1e-2)


#a kernel using griddepcontrol does not compile for gpus older than sm_90
@pytest.mark.parametrize("capability", [(8, 6), (9, 0)])
def test_dependent_launch_only_from_sm_90(monkeypatch, capability):
    import triton
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from model.fused_blocks import dependent_launch, dependent_launch_wait

    @triton.jit
    def copy_kernel(x_ptr, out_ptr, PDL: tl.constexpr):
        dependent_launch_wait(PDL)
        tl.store(out_ptr, tl.load(x_ptr))

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    launch = dependent_launch(torch.device("cuda"))
    source = ASTSource(fn=copy_kernel, signature={"x_ptr": "*fp32", "out_ptr": "*fp32", "PDL": "constexpr"}, constexprs={"PDL": launch["PDL"]})
    ptx = triton.compile(source, target=GPUTarget("cuda", 10 * capability[0] + capability[1], 32)).asm["ptx"]

    assert launch["PDL"] == launch["launch_pdl"] == (capability >= (9, 0)) == ("griddepcontrol" in ptx)


def test_downs_mix_the_inputs_of_neighbours_of_one_kind():
    from model.fused_blocks import link_input_mixes

    class Fp16Block:
        def __init__(self):
            self.residual_scale, self.input_scale = object(), object()

    class Int8Block(Fp16Block):
        pass

    blocks = [Fp16Block() for _ in range(4)]
    link_input_mixes(blocks)
    blocks[:2] = [Int8Block(), Int8Block()]
    link_input_mixes(blocks)

    assert [block.input_mixed for block in blocks] == [False, True, False, True]
    assert [block.next_scales for block in blocks] == [(blocks[1].residual_scale, blocks[1].input_scale), None,
                                                       (blocks[3].residual_scale, blocks[3].input_scale), None]


@gpu
@pytest.mark.parametrize("fp16", [False, True])
def test_fused_blocks_compute_what_the_model_does(tmp_path, fp16):
    from model.fused_blocks import FusedTransformer
    torch.manual_seed(0)
    #qk widths of 8, 16 and 24, which the kernel pads to 16 and 32
    settings = {"d_model": 128, "num_layers": 3, "n_heads": 4, "kv_heads": 2, "qk_ratio": [0.25, 0.5, 0.75], "mlp_hidden": 320, "value_bins": 16}
    model = ChessTransformer(**settings).cuda().eval()
    mix_in_the_input(model)
    with torch.no_grad():
        #random templates and features: zeros at init would hide them
        model.templates.normal_()
        model.square_features.table.normal_(std=0.1)
    save_checkpoint(model, settings, tmp_path / "model.pt")
    boards = torch.from_numpy(boards_after("e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6 f3 Be7")).long().cuda()

    with restoring_matmul_flags(), torch.no_grad():
        logits, value_logits = model(boards)
        fused = fp16_model(tmp_path / "model.pt") if fp16 else FusedTransformer(copy.deepcopy(model))
        rows = answer_rows(fused(boards))

    assert torch.allclose(rows[:, :-1], logits, atol=1e-2)
    assert torch.equal(rows[:, :-1].argmax(-1), logits.argmax(-1))
    assert torch.allclose(rows[:, -1], expected_score(value_logits), atol=1e-3)


@gpu
@pytest.mark.parametrize("cuda", [False, True])
@pytest.mark.parametrize("heads, kv_heads, dq", [(24, 12, 24), (24, 12, 8), (4, 4, 16), (6, 2, 16)])
def test_template_attention_in_fp16(monkeypatch, cuda, heads, kv_heads, dq):
    from model import attention_cuda
    from model.fused_blocks import BIAS_POSITIONS, template_attention
    #cuda: the CUDA kernel where it takes the shapes, else the Triton kernels
    monkeypatch.setattr(attention_cuda, "ENABLED", cuda)
    torch.manual_seed(0)
    B, T, dv, templates_count, eps = BIAS_POSITIONS + 5, TOKENS_PER_BOARD, 32, 8, 1e-5
    projected = torch.randn(B, T, heads * dq + kv_heads * dq + kv_heads * dv + heads * templates_count, device="cuda").half()
    templates = (torch.randn(templates_count, NUM_SQUARES, NUM_SQUARES, device="cuda") / 2).half()
    q_gain, k_gain = (1 + torch.randn(2, dq, device="cuda") / 10).half()

    out = template_attention(projected, templates, q_gain, k_gain, heads, kv_heads, dq, dv, templates_count, eps)

    q, k, v, c = projected.float().split([heads * dq, kv_heads * dq, kv_heads * dv, heads * templates_count], dim=-1)
    q = F.rms_norm(q.reshape(B, T, heads, dq), (dq,), q_gain.float(), eps).transpose(1, 2)
    k = F.rms_norm(k.reshape(B, T, kv_heads, dq), (dq,), k_gain.float(), eps).transpose(1, 2).repeat_interleave(heads // kv_heads, 1)
    v = v.reshape(B, T, kv_heads, dv).transpose(1, 2).repeat_interleave(heads // kv_heads, 1)
    bias = torch.einsum("biht,tij->bhij", c[:, :NUM_SQUARES].reshape(B, NUM_SQUARES, heads, templates_count), templates.float())
    scores = q @ k.transpose(-2, -1) / math.sqrt(dq) + F.pad(bias, (0, T - NUM_SQUARES, 0, T - NUM_SQUARES))
    expected = (torch.softmax(scores, -1) @ v).transpose(1, 2).reshape(B, T, heads * dv)
    assert out.dtype == torch.float16 and out.shape == expected.shape
    assert torch.allclose(out.float(), expected, atol=1e-2)


@gpu
def test_template_attention_in_cuda_graphs(monkeypatch):
    from model import attention_cuda, fused_blocks
    from model.fused_blocks import GRAPHED_BIAS_POSITIONS, template_attention
    monkeypatch.setattr(attention_cuda, "ENABLED", False)
    torch.manual_seed(0)
    heads, kv_heads, dq, dv, templates_count, eps = 24, 12, 24, 32, 8, 1e-5
    B = 2 * GRAPHED_BIAS_POSITIONS + 5
    projected = torch.randn(B, TOKENS_PER_BOARD, heads * dq + kv_heads * dq + kv_heads * dv + heads * templates_count, device="cuda").half()
    templates = (torch.randn(templates_count, NUM_SQUARES, NUM_SQUARES, device="cuda") / 2).half()
    q_gain, k_gain = (1 + torch.randn(2, dq, device="cuda") / 10).half()
    attend = lambda: template_attention(projected, templates, q_gain, k_gain, heads, kv_heads, dq, dv, templates_count, eps)
    in_chunks = attend()
    with monkeypatch.context() as patch:
        patch.setattr(fused_blocks, "BIAS_POSITIONS", B)
        expected = attend()

    streams = [torch.cuda.Stream() for _ in range(2)]
    graphs, answers = [], []
    for stream in streams:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            answers.append(attend())
        graphs.append(graph)
    mismatches = torch.zeros(len(streams), dtype=torch.int64, device="cuda")
    torch.cuda.synchronize()
    #no wait between replays, so that the two graphs overlap in every way
    for _ in range(1000):
        for lane, (graph, stream, answer) in enumerate(zip(graphs, streams, answers)):
            with torch.cuda.stream(stream):
                graph.replay()
                mismatches[lane] += (answer != expected).any()
    torch.cuda.synchronize()

    assert torch.equal(in_chunks, expected) and mismatches.tolist() == [0, 0]


@gpu
@pytest.mark.parametrize("gated", [False, True])
def test_linear_kernel_in_fp16(gated):
    from model.fused_blocks import linear
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


@gpu
@pytest.mark.parametrize("gated, accumulate_fp32", [(False, False), (True, False), (False, True)])
def test_extra_rows_leave_every_row_the_same(monkeypatch, gated, accumulate_fp32):
    from model import fused_blocks
    torch.manual_seed(0)
    x = torch.randn(1000, 512, device="cuda").half()
    weight = (torch.randn(512, 400 if gated else 200, device="cuda") / 512**0.5).half()
    bias = None if gated else torch.randn(200, device="cuda").half()

    outs = []
    for extra in (0, 16, 32, 64):
        monkeypatch.setattr(fused_blocks, "linear_extra_rows", lambda M, rows, column_tiles, multiprocessors, extra=extra: extra)
        outs.append(fused_blocks.linear(x, weight, bias, None, gated, accumulate_fp32))

    assert all(torch.equal(out, outs[0]) for out in outs[1:])


#64 positions' 140 tiles take two waves of 128 multiprocessors, so 16 extra rows make 128 tiles; 56 positions fill their wave
def test_extra_rows_save_a_wave():
    from model import fused_blocks
    assert fused_blocks.linear_extra_rows(64 * 70, 128, 4, 128) == 16
    assert fused_blocks.linear_extra_rows(56 * 70, 128, 4, 128) == 0


@gpu
@pytest.mark.parametrize("gated, accumulate_fp32", [(False, False), (True, False), (False, True)])
def test_linear_tiles_give_the_same_bits(gated, accumulate_fp32):
    from model import fused_blocks
    torch.manual_seed(0)
    x = torch.randn(3, 70, 512, device="cuda").half()
    weight = (torch.randn(512, 400 if gated else 200, device="cuda") / 512**0.5).half()
    bias = None if gated else torch.randn(200, device="cuda").half()
    residual = None if gated or accumulate_fp32 else torch.randn(3, 70, 200, device="cuda").half()

    tiles = dict.fromkeys(tile for by_rows in [*fused_blocks.LINEAR_TILES.values(), fused_blocks.UNTIMED_LINEAR_TILES] for _, tile in by_rows)
    outs = [fused_blocks.launch_linear(x, weight, bias, residual, gated, accumulate_fp32, None, None, None, tile) for tile in tiles]

    assert len(outs) > 1 and all(torch.equal(out, outs[0]) for out in outs[1:])


#the thresholds are rows per multiprocessor, so they scale with the multiprocessor count
def test_tiles_are_picked_by_rows_per_multiprocessor():
    from model import fused_blocks
    tiles = [(1, "a board each"), (8, "eight"), (math.inf, "more")]

    assert [fused_blocks.tile_for(tiles, rows, 70) for rows in (70, 71, 560, 561, 10**6)] == ["a board each", "eight", "eight", "more", "more"]
    assert [fused_blocks.tile_for(tiles, rows, 170) for rows in (170, 171, 1360, 1361)] == ["a board each", "eight", "eight", "more"]


#sm_80 (A100), 86 (A40), 89 (RTX 4090, L40S) and 90 (H100) take one tile per matmul at every size
@pytest.mark.parametrize("capability, multiprocessors, expected", [
    ((8, 0), 108, {"projection": (128, 64, 128, 4, 3), "gate_up": (128, 32, 64, 4, 3), "down": (128, 64, 128, 4, 3)}),
    ((8, 6), 84, {"projection": (128, 64, 128, 4, 3), "gate_up": (128, 64, 64, 4, 3), "down": (64, 128, 128, 4, 3)}),
    ((8, 9), 142, {"projection": (128, 64, 128, 4, 3), "gate_up": (128, 64, 64, 4, 3), "down": (128, 64, 128, 4, 3)}),
    ((9, 0), 132, {"projection": (128, 128, 64, 8, 3), "gate_up": (128, 64, 64, 4, 2), "down": (128, 128, 64, 8, 3, 128)})])
def test_other_gpus_keep_their_tiles_at_every_size(monkeypatch, capability, multiprocessors, expected):
    from model import fused_blocks, int8_blocks
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device=None: SimpleNamespace(multi_processor_count=multiprocessors))
    device = torch.device("cuda")

    for boards in (1, 2, 4, 8, 16, 64, 128):
        assert fused_blocks.linear_tile(device, 70 * boards) == (128, 128, 4, 3)
        assert {matmul: int8_blocks.int8_tile(device, matmul, 70 * boards) for matmul in expected} == expected


@gpu
def test_linear_kernel_accumulating_in_fp32():
    from model.fused_blocks import linear
    torch.manual_seed(0)
    x = torch.randn(3, 64, 512, device="cuda").half()
    weight = (torch.randn(512, 200, device="cuda") / 512**0.5).half()
    bias = torch.randn(200, device="cuda")

    out = linear(x, weight, bias, accumulate_fp32=True)

    assert out.dtype == torch.float32
    assert torch.allclose(out, x.float() @ weight.float() + bias, atol=1e-4)


def test_an_answer_row_ends_with_the_expected_score():
    model = tiny_model()
    boards = torch.tensor([encode_board(chess.Board())])
    with torch.no_grad():
        logits, value_logits = model(boards)
        rows = answer_rows((logits, value_logits))

    assert rows.shape == (1, logits.shape[1] + 1)
    assert 0 < rows[0, -1] < 1
    assert torch.equal(rows[0, -1], expected_score(value_logits)[0])

import copy
import math
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
from tools.checkpoint import save_checkpoint
from engine.player import CudaGraphModel
from tests.test_model import mix_in_the_input


#every batch up to max_batch has a graph, and past 16 boards a forward pads by fewer than 4 boards or a sixteenth: steps
#of 16 cost the engine's rounds of 64 leaves 3% of its nodes/s. Past a chunk a batch runs whole chunks and pads only its
#last forward, never into another: a graph of 133 ran 128 boards as forwards of 128 and 5
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


#a forward's boards are as many as the gpu's L2 holds for every lane's at once (a d768 with two lanes: 64 on the 5090's
#96 MB), and never so few that a small-L2 gpu runs its batches as forwards of a few boards
@pytest.mark.parametrize("l2_mb, width, lanes, chunk", [(96, 768, 2, 64), (96, 768, 1, 128), (48, 768, 2, 64), (48, 256, 2, 96),
                                                        (48, 128, 2, 192), (6, 768, 1, 64)])
def test_chunk_positions_fit_the_l2(monkeypatch, l2_mb, width, lanes, chunk):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(L2_cache_size=l2_mb << 20))
    assert inference.chunk_positions(width, lanes) == chunk


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


#engine_network's compile: a batch of the exact size runs through the forward compiled for it, any other through the
#dynamic one
def test_exact_size_compiled_runs_each_size_through_its_compile(monkeypatch):
    compiled_for = []
    monkeypatch.setattr(torch, "compile", lambda model, dynamic: lambda boards: compiled_for.append((len(boards), dynamic)) or model(boards))
    model = ExactSizeCompiled(ChessTransformer(d_model=32, num_layers=1, n_heads=2).eval(), 4)
    boards = torch.tensor([encode_board(chess.Board())] * 4)
    with torch.no_grad():
        model(boards)
        model(boards[:2])

    assert compiled_for == [(4, False), (2, True)]


#a model whose compiles hold on one gpu only, as a network's do on the gpu its weights are on
class OnGpu(nn.Module):
    def __init__(self, gpu: int):
        super().__init__()
        self.gpu = gpu
        self.linear = nn.Linear(4, 4)

    def forward(self, boards: torch.Tensor) -> torch.Tensor:
        return self.linear(boards) + self.gpu


#dynamo keeps one frame's compiles for every instance of a module, and compiles no more past its recompile_limit (8):
#each gpu's network adds 3 (sizes 1, 2 on and the exact size), and from the third of 8 gpus on the networks ran
#uncompiled. Every one of them is compiled for every size
def test_the_network_of_every_gpu_is_compiled(monkeypatch):
    compiles = []
    compile = torch.compile
    monkeypatch.setattr(torch, "compile", lambda model, dynamic: compile(model, dynamic=dynamic, backend=lambda graph, inputs: compiles.append(1) or graph.forward))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
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


#a batch bigger than a chunk runs as forwards of that many in its graph: every board's answer in its place, the last
#chunk part padding
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
def test_chunked_graphs_answer_like_the_model():
    torch.manual_seed(0)
    model = ChessTransformer(d_model=32, num_layers=1, n_heads=2).cuda().eval()
    board, boards = chess.Board(), []
    for san in "e4 c5 Nf3 d6 d4 cxd4 Nxd4".split():
        board.push_san(san)
        boards.append(encode_board(board))
    boards = np.array(boards, dtype=np.uint8)
    with torch.no_grad():
        expected = answer_rows(model(torch.from_numpy(boards).long().cuda())).cpu().numpy()

    graphs = GraphedBatches(model, [8], chunk=3)
    answers = graphs.collect(graphs.launch(boards))

    assert np.allclose(answers, expected, atol=1e-3)


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


#a launch on a lane whose batch is not yet collected would write over its answers, so it is refused
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
def test_more_batches_out_than_lanes_are_refused():
    model = ChessTransformer(d_model=32, num_layers=1, n_heads=2).cuda().eval()
    boards = np.array([encode_board(chess.Board())], dtype=np.uint8)
    graphs = GraphedBatches(model, [1], in_flight=1)
    launched = graphs.launch(boards)
    with pytest.raises(RuntimeError, match="more batches in flight than lanes"):
        graphs.launch(boards)
    graphs.collect(launched)
    graphs.collect(graphs.launch(boards))


#batches may be collected in any order: a launch takes whichever lane is free, not the next in turn
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
def test_batches_are_collected_in_any_order():
    torch.manual_seed(0)
    model = ChessTransformer(d_model=32, num_layers=1, n_heads=2).cuda().eval()
    board, boards = chess.Board(), []
    for san in ("e4", "d5", "exd5"):
        board.push_san(san)
        boards.append(encode_board(board))
    boards = np.array(boards, dtype=np.uint8)
    graphs = GraphedBatches(model, [1, 2])
    expected = graphs.collect(graphs.launch(boards[1:])).copy()

    first, second = graphs.launch(boards[:1]), graphs.launch(boards[1:])
    graphs.collect(second)
    third = graphs.launch(boards[1:])
    graphs.collect(first)

    assert np.array_equal(graphs.collect(third), expected)


#blocking_wait's collect sleeps through a batch the gpu is still working on, where the copy would spin its core the whole
#time, and answers as the copy does
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
def test_blocking_wait_sleeps_through_the_batch():
    model = ChessTransformer(d_model=32, num_layers=1, n_heads=2).cuda().eval()
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


#boards a search wrote into the free lane's boards_array (Searches.descend_into) answer as launch's copy of them does, on
#either lane
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
def test_boards_written_into_the_lane_answer_as_launched():
    model = ChessTransformer(d_model=32, num_layers=1, n_heads=2).cuda().eval()
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


#a lean lane answers a search's rows (Searches.descend_into's legal_columns) with exactly the columns of the answer rows a
#lane of whole rows gives the same boards, round after round on either lane, waiting by event or by the copy
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
@pytest.mark.parametrize("blocking_wait", [False, True])
def test_lean_lanes_answer_the_columns_of_whole_rows(blocking_wait):
    model = ChessTransformer(d_model=32, num_layers=1, n_heads=2).cuda().eval()
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


#two batches out at once in two lanes come back with the bits each gets alone: lanes recorded on one stream into one
#memory pool wrote over each other's activations. A model big enough that the two batches overlap
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
def test_batches_in_flight_together_answer_as_alone():
    torch.manual_seed(0)
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
        assert all(np.array_equal(answer, expected_answer) for answer, expected_answer in zip(answers, expected))

    #lean lanes too: rows of the boards and some columns of their answers, the columns of the rows alone
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
        assert all(np.array_equal(answer, expected_answer) for answer, expected_answer in zip(answers, expected))


#the d512_selfplay_sf12_10pass run's residual stream passes fp16's 65,504 on the en-passant token: every board came back
#NaN and the engine's search put all its visits on one arbitrary move
#an RMSNorm squaring that stream in fp16 would overflow the same way
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
def test_fp16_model_stays_finite_past_fp16s_range(tmp_path):
    torch.manual_seed(0)
    settings = {"d_model": 32, "num_layers": 1, "n_heads": 2, "mlp_hidden": 64, "value_bins": VALUE_BINS}
    model = ChessTransformer(**settings).eval()
    mix_in_the_input(model)
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
            rows = answer_rows(fp16_model(tmp_path / "model.pt")(boards))
            expected_logits, expected_value_logits = model.cuda()(boards)
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation, torch.backends.cuda.matmul.allow_tf32 = flags

    assert torch.isfinite(rows).all()
    policy_distance = 0.5 * (torch.softmax(rows[:, :-1], dim=-1) - torch.softmax(expected_logits, dim=-1)).abs().sum(dim=-1)
    assert policy_distance.max() < 0.01
    assert torch.allclose(rows[:, -1], expected_score(expected_value_logits), atol=0.01)


#the input embedding that every fused block reads, made by one kernel, is the model's: each token's embedding and
#position rows, and the sum of the squares' feature rows in the table's precision (the metadata tokens have none). A
#width that fills no power of two, more tokens than a program takes
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_the_input_embedding_is_the_models(dtype):
    from model.fused_blocks import InputEmbedding
    torch.manual_seed(0)
    model = ChessTransformer(d_model=96, num_layers=1, n_heads=4).cuda().eval()
    with torch.no_grad():
        model.square_features.table.normal_()
    board, boards = chess.Board(), []
    for san in "e4 Nf6 e5 d5".split():
        board.push_san(san)
        boards.append(encode_board(board))
    boards = torch.tensor(boards).cuda()
    model.half_except_heads(dtype)

    with torch.no_grad():
        found = InputEmbedding(model)(boards)
        features = F.pad(model.square_features.table.double()[model.square_features.indices(boards) + model.square_features.row_offsets].sum(-2),
                         (0, 0, 0, TOKENS_PER_BOARD - NUM_SQUARES))
        expected = (model.embed.weight.double()[boards] + model.position_embed.double() + features.to(dtype).double()).to(dtype)

    #fp16: a feature sum's rounding may differ by a step from fp64's
    rtol, atol = (1e-3, 1e-2) if dtype == torch.float16 else (1e-6, 1e-5)
    assert found.dtype == dtype and torch.allclose(found, expected, rtol=rtol, atol=atol)


#board_embedding_kernel (fused_blocks.InputEmbedding where fused_ends) finds each square's 23 feature rows itself, as
#SquareFeatures.indices does: on positions of a random game and ones with promoted pieces, several queens, en passant and
#kings in corners, with tables of whole numbers, whose sums are exact in any order, so that no wrong row can hide
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
def test_the_board_embedding_finds_the_feature_rows_of_square_features(monkeypatch):
    from model import fused_blocks
    monkeypatch.setattr(fused_blocks, "fused_ends", lambda device: True)
    torch.manual_seed(0)
    model = ChessTransformer(d_model=32, num_layers=1, n_heads=2).cuda().eval()
    with torch.no_grad():
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


#fused_blocks.heads, the heads after the last block in three kernels (FusedTransformer's where fused_ends), give the
#model's heads on the stream to fp32's rounding when fp16 holds the weights of their one fp16 matmul exactly
#(fused_blocks.fused_head_weights; their rounding is the only other error): the move logits with the promotions' and the
#expected score, of 16 bins and of the older head's one. On one board, a few and more than a program of the eval head
#takes, with each width of the matmul's tiles
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
@pytest.mark.parametrize("value_bins", [16, 1])
@pytest.mark.parametrize("boards", [1, 5, 71])
def test_the_fused_heads_compute_the_models_heads(monkeypatch, value_bins, boards):
    from model import fused_blocks
    from model.fused_blocks import FusedTransformer
    monkeypatch.setattr(fused_blocks, "fused_ends", lambda device: True)
    torch.manual_seed(0)
    d = 64
    model = ChessTransformer(d_model=d, num_layers=1, n_heads=2, value_bins=value_bins).cuda().eval()
    with torch.no_grad():
        #sixteenths and gains of powers of two, so that fp16 holds the matmul's weights
        model.from_projection.weight.copy_(torch.randint(-2, 3, (d, d)) / 16)
        model.to_projection.weight.copy_(torch.randint(-2, 3, (d, d)) / 16)
        model.ln.weight.copy_(2.0 ** torch.randint(-1, 2, (d,)))
    model.half_except_heads(torch.float16)
    stream = torch.randn(boards, TOKENS_PER_BOARD, d, device="cuda").half()
    fused = FusedTransformer(model)

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


#the board embedding and the fused heads only where they were timed (fused_ends): an A100, A40 or L40S (sm_80, 86, 89)
#keeps today's kernels and outputs, the model's logits and value logits, which answer as the model does
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
@pytest.mark.parametrize("capability", [(8, 0), (8, 6), (8, 9), (12, 0)])
def test_the_fused_ends_only_on_the_gpus_they_were_timed_on(monkeypatch, capability):
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

    timed = capability == (12, 0)
    assert fused.fused_heads == fused.input_embedding.from_boards == timed
    assert isinstance(outputs, torch.Tensor) == timed
    assert torch.allclose(answer_rows(outputs), expected, atol=1e-2)


#programmatic dependent launch only from sm_90: a kernel built for older gpus (hyak's A100s and A40s) with griddepcontrol
#does not compile
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


#a block's down mixes the next block's input only when both are of one kind, and linking again (quantize_blocks after
#FusedTransformer) clears the old links: a block left expecting a mixed input would skip its mix
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


#fused_blocks.FusedBlock must compute what model.TransformerBlock does, InputEmbedding what the model's embedding does
#and FusedTransformer's heads (its answer rows) what the heads do: with random templates and square features (zeros at
#init would hide them), grouped key/value heads and query/key widths of 8, 16 and 24 (padded to 16 and 32 in the
#kernel), and the input skip away from its start. In fp32, and in fp16 as the engine runs it (fp16_model)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
@pytest.mark.parametrize("fp16", [False, True])
def test_fused_blocks_compute_what_the_model_does(tmp_path, fp16):
    from model.fused_blocks import FusedTransformer
    torch.manual_seed(0)
    settings = {"d_model": 128, "num_layers": 3, "n_heads": 4, "kv_heads": 2, "qk_ratio": [0.25, 0.5, 0.75], "mlp_hidden": 320, "value_bins": 16}
    model = ChessTransformer(**settings).cuda().eval()
    mix_in_the_input(model)
    with torch.no_grad():
        model.templates.normal_()
        model.square_features.table.normal_(std=0.1)
    save_checkpoint(model, settings, tmp_path / "model.pt")
    board, boards = chess.Board(), []
    for san in "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6 f3 Be7".split():
        board.push_san(san)
        boards.append(encode_board(board))
    boards = torch.tensor(boards).cuda()

    #fp16_model sets these for the whole process
    flags = torch.backends.cuda.matmul.allow_fp16_accumulation, torch.backends.cuda.matmul.allow_tf32
    try:
        with torch.no_grad():
            logits, value_logits = model(boards)
            fused = fp16_model(tmp_path / "model.pt") if fp16 else FusedTransformer(copy.deepcopy(model))
            rows = answer_rows(fused(boards))
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation, torch.backends.cuda.matmul.allow_tf32 = flags

    assert torch.allclose(rows[:, :-1], logits, atol=1e-2)
    assert torch.equal(rows[:, :-1].argmax(-1), logits.argmax(-1))
    assert torch.allclose(rows[:, -1], expected_score(value_logits), atol=1e-3)


#fused_blocks.template_attention in fp16 as the engine runs it, against the attention in fp32 at the d768's shapes:
#every query row (the squares' with the template bias, the metadata tokens' without), one, two and three query heads
#to a key/value head (three fill no power of two of rows), query/key widths padded in the kernel or not, and more
#positions than one bias buffer takes
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
@pytest.mark.parametrize("heads, kv_heads, dq", [(24, 12, 24), (24, 12, 8), (4, 4, 16), (6, 2, 16)])
def test_template_attention_in_fp16(heads, kv_heads, dq):
    from model.fused_blocks import BIAS_POSITIONS, template_attention
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


#template_attention recorded in cuda graphs makes the next chunk's bias on a second stream: it must wait until the square
#rows two chunks back have read its buffer, the square rows until their bias is made, and the graph for both streams.
#Two graphs at the d768's shapes, of more positions than the two buffers take, replayed at once on their own streams
#give the bits of one chunk, as eager calls in chunks do
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
def test_template_attention_in_cuda_graphs(monkeypatch):
    from model import fused_blocks
    from model.fused_blocks import GRAPHED_BIAS_POSITIONS, template_attention
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


#fused_blocks.linear in fp16 as the engine runs it, accumulating in fp16, with the bias and residual added (the
#attention's output projection) or silu(gate) * up (the mlp), on rows and columns that leave part of a tile empty
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
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


#a tile's extra rows (fused_blocks.linear_extra_rows) are a matmul of their own on the same weights: every row comes
#out the same bits whichever tile it falls in, in plain, gated and fp32 accumulating matmuls
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
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


#at 64 positions (4,480 rows) a 512 column matmul's 140 tiles of 128 rows take two waves of an RTX 4090's 128
#multiprocessors and 128 tiles of 144 rows one; tiles that fill their waves take no extra rows
def test_extra_rows_save_a_wave():
    fused_blocks = pytest.importorskip("model.fused_blocks")
    assert fused_blocks.linear_extra_rows(64 * 70, 128, 4, 128) == 16
    assert fused_blocks.linear_extra_rows(56 * 70, 128, 4, 128) == 0


#every tile of linear's tables gives the bits of the others (none changes the depth's order or the instructions) in
#plain (with the residual), gated and fp32 accumulating matmuls, on three boards' rows that leave tiles part empty
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
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


#a tile is picked by the rows a multiprocessor's share of the matmul is, so a card with more of them picks as one with
#fewer does at as many rows each: a 170 multiprocessor RTX 5090 takes the RTX 5070 Ti's (70) tile for 70/170 its rows
def test_tiles_are_picked_by_rows_per_multiprocessor():
    fused_blocks = pytest.importorskip("model.fused_blocks")
    tiles = [(1, "a board each"), (8, "eight"), (math.inf, "more")]

    assert [fused_blocks.tile_for(tiles, rows, 70) for rows in (70, 71, 560, 561, 10**6)] == ["a board each", "eight", "eight", "more", "more"]
    assert [fused_blocks.tile_for(tiles, rows, 170) for rows in (170, 171, 1360, 1361)] == ["a board each", "eight", "eight", "more"]


#sm_120's tiles by rows stay on sm_120: an A100 (sm_80), A40 (sm_86) and RTX 4090 / L40S (sm_89) keep their own tiles
#at every size, and a gpu not timed (an H100, sm_90) sm_120's for any rows, as before the tables went by rows
@pytest.mark.parametrize("capability, multiprocessors, expected", [
    ((8, 0), 108, {"projection": (128, 64, 128, 4, 3), "gate_up": (128, 32, 64, 4, 3), "down": (128, 64, 128, 4, 3)}),
    ((8, 6), 84, {"projection": (128, 64, 128, 4, 3), "gate_up": (128, 64, 64, 4, 3), "down": (64, 128, 128, 4, 3)}),
    ((8, 9), 142, {"projection": (128, 64, 128, 4, 3), "gate_up": (128, 64, 64, 4, 3), "down": (128, 64, 128, 4, 3)}),
    ((9, 0), 132, {"projection": (128, 128, 64, 8, 3), "gate_up": (128, 64, 64, 4, 2), "down": (128, 128, 64, 8, 3, 128)})])
def test_other_gpus_keep_their_tiles_at_every_size(monkeypatch, capability, multiprocessors, expected):
    fused_blocks = pytest.importorskip("model.fused_blocks")
    int8_blocks = pytest.importorskip("model.int8_blocks")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device=None: SimpleNamespace(multi_processor_count=multiprocessors))
    device = torch.device("cuda")

    for boards in (1, 2, 4, 8, 16, 64, 128):
        assert fused_blocks.linear_tile(device, 70 * boards) == (128, 128, 4, 3)
        assert {matmul: int8_blocks.int8_tile(device, matmul, 70 * boards) for matmul in expected} == expected


#the heads' matmul (fused_blocks.FusedTransformer): fp16 inputs accumulating in fp32 and giving fp32, as close to the
#fp32 matmul as fp16's rounding of the weights allows
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
def test_linear_kernel_accumulating_in_fp32():
    from model.fused_blocks import linear
    torch.manual_seed(0)
    x = torch.randn(3, 64, 512, device="cuda").half()
    weight = (torch.randn(512, 200, device="cuda") / 512**0.5).half()
    bias = torch.randn(200, device="cuda")

    out = linear(x, weight, bias, accumulate_fp32=True)

    assert out.dtype == torch.float32
    assert torch.allclose(out, x.float() @ weight.float() + bias, atol=1e-4)


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

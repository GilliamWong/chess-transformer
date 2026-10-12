import copy
import math

import chess
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from model.encoding import EN_PASSANT_TOKEN_OFFSET, NUM_SQUARES, TOKENS_PER_BOARD, encode_board
from engine import inference
from engine.inference import ExactSizeCompiled, GraphedBatches, answer_rows, fp16_model, graph_sizes, small_graph_sizes
from model.model import VALUE_BINS, ChessTransformer, expected_score
from tools.checkpoint import save_checkpoint
from engine.player import CudaGraphModel
from tests.test_model import mix_in_the_input


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


#a batch bigger than CHUNK_POSITIONS runs as forwards of that many in its graph: every board's answer in its place, the
#last chunk part padding
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
def test_chunked_graphs_answer_like_the_model(monkeypatch):
    monkeypatch.setattr(inference, "CHUNK_POSITIONS", 3)
    torch.manual_seed(0)
    model = ChessTransformer(d_model=32, num_layers=1, n_heads=2).cuda().eval()
    board, boards = chess.Board(), []
    for san in "e4 c5 Nf3 d6 d4 cxd4 Nxd4".split():
        board.push_san(san)
        boards.append(encode_board(board))
    boards = np.array(boards, dtype=np.uint8)
    with torch.no_grad():
        expected = answer_rows(model(torch.from_numpy(boards).long().cuda())).cpu().numpy()

    graphs = GraphedBatches(model, [8])
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
            logits, value_logits = fp16_model(tmp_path / "model.pt")(boards)
            expected_logits, expected_value_logits = model.cuda()(boards)
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation, torch.backends.cuda.matmul.allow_tf32 = flags

    assert torch.isfinite(logits).all() and torch.isfinite(value_logits).all()
    policy_distance = 0.5 * (torch.softmax(logits.float(), dim=-1) - torch.softmax(expected_logits, dim=-1)).abs().sum(dim=-1)
    assert policy_distance.max() < 0.01
    assert torch.allclose(expected_score(value_logits.float()), expected_score(expected_value_logits), atol=0.01)


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
#and FusedTransformer's one matmul what the heads do: with random templates and square features (zeros at init would
#hide them), grouped key/value heads and query/key widths of 8, 16 and 24 (padded to 16 and 32 in the kernel), and the
#input skip away from its start. In fp32, and in fp16 as the engine runs it (fp16_model)
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
            fused_logits, fused_value_logits = fused(boards)
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation, torch.backends.cuda.matmul.allow_tf32 = flags

    assert torch.allclose(fused_logits.float(), logits, atol=1e-2)
    assert torch.equal(fused_logits.argmax(-1), logits.argmax(-1))
    assert torch.allclose(expected_score(fused_value_logits.float()), expected_score(value_logits), atol=1e-3)


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
        monkeypatch.setattr(fused_blocks, "linear_extra_rows", lambda M, column_tiles, multiprocessors, extra=extra: extra)
        outs.append(fused_blocks.linear(x, weight, bias, None, gated, accumulate_fp32))

    assert all(torch.equal(out, outs[0]) for out in outs[1:])


#at 64 positions (4,480 rows) a 512 column matmul's 140 tiles of 128 rows take two waves of an RTX 4090's 128
#multiprocessors and 128 tiles of 144 rows one; tiles that fill their waves take no extra rows
def test_extra_rows_save_a_wave():
    fused_blocks = pytest.importorskip("model.fused_blocks")
    assert fused_blocks.linear_extra_rows(64 * 70, 4, 128) == 16
    assert fused_blocks.linear_extra_rows(56 * 70, 4, 128) == 0


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

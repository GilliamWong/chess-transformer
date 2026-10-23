#the int8 network (model/int8_blocks.py, engine/quantization.py): its kernels against fp32 references, GPTQ's
#rounding, the weights' cache, and the int8 model against the fp32 one
import copy
import shutil
from types import SimpleNamespace

import chess
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from model.encoding import encode_board
from model.model import ChessTransformer, expected_score
from engine import quantization
from engine.quantization import gptq
from tests.test_model import mix_in_the_input
from tools.checkpoint import save_checkpoint

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")


#with uncorrelated inputs (a diagonal Hessian) no column's error can be made up for by another: GPTQ rounds to nearest
def test_gptq_without_correlated_inputs_rounds_to_nearest():
    torch.manual_seed(0)
    weight = torch.randn(40, 64, dtype=torch.float64)
    integers, scale = gptq(weight, torch.diag(torch.rand(64, dtype=torch.float64) + 0.5))

    assert torch.allclose(scale.double(), weight.abs().amax(dim=1) / 127, rtol=1e-6)
    assert torch.equal(integers, (weight / scale.double()[:, None]).round().to(torch.int8))


#with correlated inputs GPTQ's matmul output is nearer the exact one than rounding to nearest, in int8's range
def test_gptq_beats_rounding_to_nearest_on_correlated_inputs():
    torch.manual_seed(0)
    inputs = torch.randn(2000, 8, dtype=torch.float64) @ torch.randn(8, 64, dtype=torch.float64) + 0.1 * torch.randn(2000, 64, dtype=torch.float64)
    weight = torch.randn(40, 64, dtype=torch.float64)
    integers, scale = gptq(weight, inputs.t() @ inputs)
    nearest = (weight / scale.double()[:, None]).round()

    error = lambda rounded: (inputs @ (rounded * scale.double()[:, None] - weight).t()).pow(2).sum()
    assert integers.abs().max() <= 127 and error(integers.double()) < 0.5 * error(nearest)


def dequantised(integers: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return integers.double() * scale.double()[:, None]


#int8_linear's output is the dequantised operands' matmul (exact in int32) with each block matmul's tiles of every gpu
#and epilogue (the projection's bias, gate_up's silu(gate) * up, down's residual over its 1,024 inputs, and the next
#block's input mixed in), to fp16's rounding: on one board's rows (less than a tile) and three boards', and on columns
#that leave part of a tile empty, by its Triton kernel (launch_int8_linear)
@gpu
@pytest.mark.parametrize("matmul", ["projection", "gate_up", "down", "mixing down"])
@pytest.mark.parametrize("boards", [1, 3])
def test_int8_linear_is_the_dequantised_matmul(matmul, boards):
    from model.int8_blocks import INT8_TILES, launch_int8_linear
    torch.manual_seed(0)
    down, gated = matmul.endswith("down"), matmul == "gate_up"
    width, depth = 200, 1024 if down else 256
    x = torch.randint(-127, 128, (boards, 70, depth), device="cuda", dtype=torch.int8)
    x_scale = torch.rand(boards, 70, device="cuda") / 100
    weight = torch.randint(-127, 128, (2 * width if gated else width, depth), device="cuda", dtype=torch.int8)
    weight_scale = torch.rand(len(weight), device="cuda") / 100
    bias = torch.randn(width, device="cuda").half() if matmul == "projection" else None
    residual = torch.randn(boards, 70, width, device="cuda").half() if down else None
    mix = (torch.randn(boards, 70, width, device="cuda").half(), torch.tensor(0.9, device="cuda").half(),
           torch.tensor(0.3, device="cuda").half()) if matmul == "mixing down" else ()

    projected = dequantised(x.reshape(-1, depth), x_scale.reshape(-1)) @ dequantised(weight, weight_scale).t()
    if matmul == "projection":
        expected = projected + bias.double()
    elif matmul == "gate_up":
        gate, up = projected.chunk(2, dim=-1)
        expected = F.silu(gate) * up
    else:
        expected = projected + residual.reshape(-1, width).double()
    if mix:
        original, residual_scale, input_scale = mix
        expected = residual_scale.double() * expected.half().double() + input_scale.double() * original.reshape(-1, width).double()
    for tile in dict.fromkeys(tile for tiles in INT8_TILES.values() for _, tile in tiles["down" if down else matmul]):
        out = launch_int8_linear(x, x_scale, weight, weight_scale, bias, residual, gated, *(mix or (None, None, None)), tile)
        assert out.dtype == torch.float16 and out.shape == (boards, 70, width)
        assert torch.allclose(out.reshape(-1, width).double(), expected, atol=1e-3, rtol=2e-3), tile


#a block matmul's operands at the d768's depths: (x, x_scale, weight, weight_scale, bias, residual, gated, mix)
def int8_operands(matmul: str, rows: int, width: int = 196) -> tuple:
    torch.manual_seed(rows)
    down, gated = matmul.endswith("down"), matmul == "gate_up"
    depth = 1024 if down else 768
    x = torch.randint(-127, 128, (rows, depth), device="cuda", dtype=torch.int8)
    x_scale = torch.rand(rows, device="cuda") / 100
    weight = torch.randint(-127, 128, (2 * width if gated else width, depth), device="cuda", dtype=torch.int8)
    weight_scale = torch.rand(len(weight), device="cuda") / 100
    bias = torch.randn(width, device="cuda").half() if matmul == "projection" else None
    residual = torch.randn(rows, width, device="cuda").half() if down else None
    mix = (torch.randn(rows, width, device="cuda").half(), torch.tensor([0.9], device="cuda").half(),
           torch.tensor([0.3], device="cuda").half()) if matmul == "mixing down" else ()
    return x, x_scale, weight, weight_scale, bias, residual, gated, mix


def triton_int8_linear(monkeypatch, matmul: str, operands: tuple) -> torch.Tensor:
    from model import int8_cuda
    from model.int8_blocks import int8_linear
    x, x_scale, weight, weight_scale, bias, residual, gated, mix = operands
    with monkeypatch.context() as patch:
        patch.setattr(int8_cuda, "ENABLED", False)
        return int8_linear(x, x_scale, weight, weight_scale, bias, residual, gated, "down" if matmul.endswith("down") else matmul, *mix)


#this gpu's key in int8_cuda.CONFIGS
def gpu_key() -> tuple[int, int, int]:
    return (*torch.cuda.get_device_capability(), torch.cuda.get_device_properties(0).multi_processor_count)


#the (most rows, config) entries of a kind for every gpu timed with this one's compute capability (the kernels run on
#any multiprocessor count; it only changes which is fastest)
def configs_of_this_capability(kind: str) -> list[tuple[int, tuple]]:
    from model import int8_cuda
    found = [entry for key, table in int8_cuda.CONFIGS.items() if key[:2] == gpu_key()[:2] for entry in table[kind]]
    if not found:
        pytest.skip("no small-batch CUDA kernels timed on a gpu like this one")
    return found


#the small-batch CUDA kernels (model/int8_cuda.py) give int8_linear's Triton kernel's bits with every config timed on
#this compute capability and every epilogue: on as many rows as a config takes and on fewer than one tile, columns
#leaving part of a tile empty
@gpu
@pytest.mark.parametrize("matmul", ["projection", "gate_up", "down", "mixing down"])
def test_the_small_batch_cuda_kernels_give_int8_linears_bits(matmul, monkeypatch):
    from model import int8_cuda
    kind = "down" if matmul.endswith("down") else matmul
    for largest, config in configs_of_this_capability(kind):
        monkeypatch.setattr(int8_cuda, "CONFIGS", {gpu_key(): {kind: [(largest, config)]}})
        monkeypatch.setattr(int8_cuda, "KERNELS", {})
        for rows in (5, largest):
            operands = int8_operands(matmul, rows)
            x, x_scale, weight, weight_scale, bias, residual, gated, mix = operands
            found = int8_cuda.small_int8_linear(x, x_scale, weight, weight_scale, bias, residual, gated, *(mix or (None, None, None)))
            assert found is not None and torch.equal(found, triton_int8_linear(monkeypatch, matmul, operands)), (config, rows)
            assert int8_cuda.takes_rows(x.device, (kind,), rows, x.shape[-1]) and not int8_cuda.takes_rows(x.device, (kind,), largest + 1, x.shape[-1])


#recorded in cuda graphs on two streams and replayed at once (the engine's two lanes), a CUDA kernel reads the graph's
#input buffer and answers as it does outside a graph
@gpu
def test_the_small_batch_cuda_kernels_run_in_cuda_graphs_on_two_lanes(monkeypatch):
    from model import int8_cuda
    largest, config = configs_of_this_capability("down")[0]
    monkeypatch.setattr(int8_cuda, "CONFIGS", {gpu_key(): {"down": [(largest, config)]}})
    monkeypatch.setattr(int8_cuda, "KERNELS", {})
    _, x_scale, weight, weight_scale, _, residual, _, _ = int8_operands("down", largest)
    inputs = [int8_operands("down", largest)[0] for _ in range(2)]
    inputs[1] = inputs[1].flip(0).contiguous()
    run = lambda x: int8_cuda.small_int8_linear(x, x_scale, weight, weight_scale, None, residual, False, None, None, None)
    expected = [run(x) for x in inputs]
    assert all(answer is not None for answer in expected)
    lanes, buffers, graphs, outputs = [torch.cuda.Stream() for _ in range(2)], [x.clone() for x in inputs], [], []
    for lane, buffer in zip(lanes, buffers):
        graphs.append(torch.cuda.CUDAGraph())
        with torch.cuda.graph(graphs[-1], stream=lane):
            outputs.append(run(buffer))
    for order in (1, -1):
        for buffer, x in zip(buffers, inputs[::order]):
            buffer.copy_(x)
        torch.cuda.synchronize()
        for _ in range(20):
            for lane, graph in zip(lanes, graphs):
                with torch.cuda.stream(lane):
                    graph.replay()
        torch.cuda.synchronize()
        assert all(torch.equal(out, answer) for out, answer in zip(outputs, expected[::order]))


#when NVRTC fails the CUDA kernels are off with a warning and int8_linear runs its Triton kernel
@gpu
def test_int8_linear_falls_back_to_triton_when_the_cuda_kernels_do_not_compile(monkeypatch):
    from model import int8_cuda
    from model.int8_blocks import int8_linear
    largest, config = configs_of_this_capability("projection")[0]

    def failing(*arguments):
        raise RuntimeError("NVRTC compilation failed")
    monkeypatch.setattr(int8_cuda, "compile_cubin", failing)
    monkeypatch.setattr(int8_cuda, "CONFIGS", {gpu_key(): {"projection": [(largest, config)]}})
    monkeypatch.setattr(int8_cuda, "KERNELS", {})
    operands = int8_operands("projection", largest)
    x, x_scale, weight, weight_scale, bias, residual, gated, mix = operands
    with pytest.warns(UserWarning, match="Triton kernel runs instead"):
        out = int8_linear(x, x_scale, weight, weight_scale, bias, residual, gated, "projection")
    assert torch.equal(out, triton_int8_linear(monkeypatch, "projection", operands))


#gpus the kernels were not timed on never compile them, int8_linear runs Triton as before: sm_80 (the A100), sm_86 and
#sm_89 with this gpu's multiprocessors, and this compute capability with another number of them
@gpu
@pytest.mark.parametrize("untimed", [(8, 0), (8, 6), (8, 9), "multiprocessors"])
def test_untimed_gpus_never_use_the_cuda_kernels(untimed, monkeypatch):
    from model import int8_cuda

    def compiling(*arguments):
        raise AssertionError("compiled")
    monkeypatch.setattr(int8_cuda, "compile_cubin", compiling)
    monkeypatch.setattr(int8_cuda, "KERNELS", {})
    timed = {"projection": [(70, (5, 4, 1, 12, 1))]}
    if untimed == "multiprocessors":
        monkeypatch.setattr(int8_cuda, "CONFIGS", {(*gpu_key()[:2], gpu_key()[2] + 1): timed})
    else:
        monkeypatch.setattr(int8_cuda, "CONFIGS", {(12, 0, gpu_key()[2]): timed})
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: untimed)
    x, x_scale, weight, weight_scale, bias, residual, gated, _ = int8_operands("projection", 70)
    assert int8_cuda.small_int8_linear(x, x_scale, weight, weight_scale, bias, residual, gated, None, None, None) is None
    assert not int8_cuda.takes_rows(x.device, ("projection",), 70, 768)


#norm_quant rounds the RMSNorm (with gain) of each row to int8 steps of its absmax / 127; mix_norm_quant first mixes
#in the block's input (residual_scale * x + input_scale * original), and gives the mix too
@gpu
def test_norm_quant_rounds_the_rmsnorm():
    from model.int8_blocks import mix_norm_quant, norm_quant
    torch.manual_seed(0)
    x, original = torch.randn(2, 70, 192, device="cuda").half() * 4, torch.randn(2, 70, 192, device="cuda").half()
    gain = (torch.rand(192, device="cuda") + 0.5).half()
    residual_scale, input_scale = torch.tensor([0.8], device="cuda").half(), torch.tensor([0.5], device="cuda").half()

    for mix in (False, True):
        if mix:
            mixed, integers, scale = mix_norm_quant(x, original, residual_scale, input_scale, gain, 1e-5)
            expected_mix = 0.8 * x.float() + 0.5 * original.float()
            assert mixed.dtype == torch.float16 and torch.allclose(mixed.float(), expected_mix, atol=4e-3, rtol=1e-3)
            normed = F.rms_norm(mixed.float(), (192,), gain.float(), 1e-5)
        else:
            integers, scale = norm_quant(x, gain, 1e-5)
            normed = F.rms_norm(x.float(), (192,), gain.float(), 1e-5)
        step = normed.abs().amax(dim=-1) / 127
        assert integers.dtype == torch.int8 and torch.allclose(scale, step, rtol=1e-5)
        assert ((integers.float() * scale[..., None] - normed).abs() <= 0.501 * step[..., None]).all()


#rotate_quant rounds each row times the block-diagonal Hadamard matrix to int8 steps of its absmax / 127, written wider
#with zeros: with blocks that fill a power of two of slots (the d768's 1,024 of 992) and blocks that do not
@gpu
@pytest.mark.parametrize("hidden, width", [(992, 1024), (320, 384)])
def test_rotate_quant_rounds_the_rotated_rows(hidden, width):
    from model.int8_blocks import HADAMARD_BLOCK, hadamard, rotate_quant
    torch.manual_seed(0)
    h = torch.randn(3, 70, hidden, device="cuda").half()

    integers, scale = rotate_quant(h, hadamard(HADAMARD_BLOCK).cuda().half(), width)

    rotation = torch.block_diag(*[hadamard(HADAMARD_BLOCK)] * (width // HADAMARD_BLOCK)).cuda()
    rotated = F.pad(h.float(), (0, width - hidden)) @ rotation
    step = rotated.abs().amax(dim=-1) / 127
    assert integers.shape == (3, 70, width) and (integers[..., hidden:] == 0).all()
    assert torch.allclose(scale, step, rtol=1e-5)
    assert ((integers.float() * scale[..., None] - rotated).abs() <= 0.501 * step[..., None]).all()


#rotate_quant's rows a program change no bits, with rows that leave its last program part empty
@gpu
def test_rotate_quant_rows_change_no_bits(monkeypatch):
    from model import int8_blocks
    from model.int8_blocks import HADAMARD_BLOCK, hadamard, rotate_quant
    torch.manual_seed(0)
    h = torch.randn(5, 70, 992, device="cuda").half()
    signs = hadamard(HADAMARD_BLOCK).cuda().half()

    found = []
    for rows in (1, 2, 4, 8, 16):
        monkeypatch.setattr(int8_blocks, "rotate_rows", lambda device, M, rows=rows: rows)
        found.append(rotate_quant(h, signs, 1024))

    assert all(torch.equal(integers, found[0][0]) and torch.equal(scale, found[0][1]) for integers, scale in found[1:])


#rotate_quant's rows a program: sm_120's fewer at up to 4 rows a multiprocessor, and the untimed gpus' (sm_80, sm_86 and
#sm_89: an A100, A40 and L40S) today's 8 at any batch
@pytest.mark.parametrize("capability", [(8, 0), (8, 6), (8, 9), (12, 0)])
def test_rotate_rows_by_compute_capability(monkeypatch, capability):
    from model import int8_blocks
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device=None: SimpleNamespace(multi_processor_count=70))

    rows = [int8_blocks.rotate_rows(torch.device("cuda"), boards * 70) for boards in (1, 4, 8, 64)]

    assert rows == ([2, 2, 8, 8] if capability == (12, 0) else [8, 8, 8, 8])


SMALL_SETTINGS ={"d_model": 128, "num_layers": 3, "n_heads": 4, "kv_heads": 2, "qk_ratio": [0.25, 0.5, 0.75], "mlp_hidden": 320, "value_bins": 16}


#a small model saved in tmp_path, its input skip mixed in, and real positions to check it on
def small_checkpoint(tmp_path, settings: dict = SMALL_SETTINGS) -> tuple[ChessTransformer, torch.Tensor]:
    torch.manual_seed(0)
    model = ChessTransformer(**settings).eval()
    mix_in_the_input(model)
    with torch.no_grad():
        model.templates.normal_()
        model.square_features.table.normal_(std=0.1)
        #sharp enough priors that a wrong scale or rotation shows
        model.from_projection.weight.mul_(8)
    save_checkpoint(model, settings, tmp_path / "model.pt")
    board, boards = chess.Board(), []
    for san in "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6 f3 Be7".split():
        board.push_san(san)
        boards.append(encode_board(board))
    return model, torch.tensor(boards)


#SMALL_SETTINGS with more blocks, their query widths in turn
def deeper(layers: int) -> dict:
    ratios = SMALL_SETTINGS["qk_ratio"]
    return {**SMALL_SETTINGS, "num_layers": layers, "qk_ratio": [ratios[layer % len(ratios)] for layer in range(layers)]}


#small_checkpoint with five blocks, one of them rounding badly in int8: an input channel of its mlp's gate 100x the
#others, so that its gate weights' row scales leave every other channel a step or two of int8
def outlier_checkpoint(tmp_path, layer: int) -> tuple[ChessTransformer, torch.Tensor]:
    settings = deeper(5)
    model, boards = small_checkpoint(tmp_path, settings)
    with torch.no_grad():
        model.encoder[layer].mlp.gate_projection.weight[:, 0] *= 100
    save_checkpoint(model, settings, tmp_path / "model.pt")
    return model, boards


@pytest.fixture
def small_calibration(monkeypatch, tmp_path):
    monkeypatch.setattr(quantization, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(quantization, "CALIBRATION_POSITIONS", 64)


#the int8 model (inference.int8_model without fp16 blocks: every block but the last in int8) gives the fp32 model's
#priors and score to int8's rounding, with the same top moves; the fp16 model is nearer
@gpu
def test_the_int8_model_computes_what_the_model_does(tmp_path, small_calibration):
    from engine.inference import answer_rows, fp16_model, int8_model
    from model.int8_blocks import Int8Block
    model, boards = small_checkpoint(tmp_path)
    boards = boards.cuda()
    flags = torch.backends.cuda.matmul.allow_fp16_accumulation, torch.backends.cuda.matmul.allow_tf32
    try:
        with torch.no_grad():
            expected_logits, expected_values = model.cuda()(boards)
            int8 = int8_model(tmp_path / "model.pt", 0)
            rows = answer_rows(int8(boards))
            fp16_rows = answer_rows(fp16_model(tmp_path / "model.pt")(boards))
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation, torch.backends.cuda.matmul.allow_tf32 = flags

    assert [isinstance(block, Int8Block) for block in int8.encoder] == [True, True, False]
    distance = lambda found: 0.5 * (torch.softmax(found[:, :-1], dim=-1) - torch.softmax(expected_logits, dim=-1)).abs().sum(dim=-1)
    assert distance(rows).max() < 0.005 and distance(fp16_rows).max() < distance(rows).max()
    assert torch.equal(rows[:, :-1].argmax(-1), expected_logits.argmax(-1))
    assert torch.allclose(rows[:, -1], expected_score(expected_values), atol=5e-3)


#an int8 block's down mixing the next int8 block's input gives the bits the next block's own mix gives
@gpu
def test_down_mixes_the_next_blocks_input_to_the_bit(tmp_path, small_calibration):
    from engine.inference import answer_rows, int8_model
    _, boards = small_checkpoint(tmp_path)
    flags = torch.backends.cuda.matmul.allow_fp16_accumulation, torch.backends.cuda.matmul.allow_tf32
    try:
        with torch.no_grad():
            int8 = int8_model(tmp_path / "model.pt", 0)
            mixed_in_down = answer_rows(int8(boards.cuda()))
            linked = [block.input_mixed for block in int8.encoder[:2]]
            for block in int8.encoder[:2]:
                block.next_scales, block.input_mixed = None, False
            mixed_by_each = answer_rows(int8(boards.cuda()))
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation, torch.backends.cuda.matmul.allow_tf32 = flags

    assert linked == [False, True] and torch.equal(mixed_in_down, mixed_by_each)


#programmatic dependent launch changes no bits: the int8 model (every kernel of fused_blocks and int8_blocks) recorded as
#cuda graphs with it answers as without it
@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0), reason="no sm_90 gpu")
def test_dependent_launch_changes_no_bits(tmp_path, small_calibration, monkeypatch):
    from engine.inference import GraphedBatches, int8_model
    from model import fused_blocks, int8_blocks
    _, boards = small_checkpoint(tmp_path)
    boards = boards.numpy().astype("uint8")
    flags = torch.backends.cuda.matmul.allow_fp16_accumulation, torch.backends.cuda.matmul.allow_tf32
    try:
        int8 = int8_model(tmp_path / "model.pt", 0)
        dependent = GraphedBatches(int8, [len(boards)], in_flight=1)
        answers = dependent.collect(dependent.launch(boards)).copy()
        for module in (fused_blocks, int8_blocks):
            monkeypatch.setattr(module, "dependent_launch", lambda device: {"PDL": False, "launch_pdl": False})
        one_after_another = GraphedBatches(int8, [len(boards)], in_flight=1)
        expected = one_after_another.collect(one_after_another.launch(boards))
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation, torch.backends.cuda.matmul.allow_tf32 = flags

    assert np.array_equal(answers, expected)


#the tiles picked by rows (INT8_TILES, fused_blocks.LINEAR_TILES) change no answer of the int8 network: at every
#graph size to 64 boards its graphs give the bits of graphs whose matmuls all take their tables' tiles for any rows
@gpu
def test_tiles_by_rows_change_no_answer(tmp_path, small_calibration, monkeypatch):
    from engine.inference import GraphedBatches, int8_model, small_graph_sizes
    from model import fused_blocks, int8_blocks
    _, boards = small_checkpoint(tmp_path)
    boards = np.resize(boards.numpy().astype("uint8"), (64, boards.shape[1]))
    sizes = small_graph_sizes(64)
    flags = torch.backends.cuda.matmul.allow_fp16_accumulation, torch.backends.cuda.matmul.allow_tf32
    try:
        int8 = int8_model(tmp_path / "model.pt", 0)
        by_rows = GraphedBatches(int8, sizes, in_flight=1)
        answers = [by_rows.collect(by_rows.launch(boards[:size])).copy() for size in sizes]
        for module in (fused_blocks, int8_blocks):
            monkeypatch.setattr(module, "tile_for", lambda tiles, M, multiprocessors: tiles[-1][1])
        any_rows = GraphedBatches(int8, sizes, in_flight=1)
        expected = [any_rows.collect(any_rows.launch(boards[:size])).copy() for size in sizes]
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation, torch.backends.cuda.matmul.allow_tf32 = flags

    assert all(np.array_equal(answer, expected_answer) for answer, expected_answer in zip(answers, expected))


#engine_network in int8: the int8 model (with the engine's fp16 blocks) compiled and recorded as cuda graphs answers as
#it does uncompiled
@gpu
def test_the_engine_network_runs_in_int8(tmp_path, small_calibration):
    from engine.engine_search import ENGINE_FP16_BLOCKS, engine_network
    from engine.inference import answer_rows, int8_model
    _, boards = small_checkpoint(tmp_path)
    flags = torch.backends.cuda.matmul.allow_fp16_accumulation, torch.backends.cuda.matmul.allow_tf32
    try:
        network = engine_network(tmp_path / "model.pt", 4, "int8", lean=False)
        answers = network.collect(network.launch(boards[:3].numpy().astype("uint8")))
        with torch.no_grad():
            expected = answer_rows(int8_model(tmp_path / "model.pt", ENGINE_FP16_BLOCKS)(boards[:3].cuda())).cpu().numpy()
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation, torch.backends.cuda.matmul.allow_tf32 = flags

    #fp16's rounding differs a little once compiled; a graph reading the wrong buffers is off by whole logits
    assert abs(answers - expected).max() < 0.05


#the int8 model in cuda graphs on two lanes answers to the bit as without the small-batch CUDA kernels, with configs for
#the small model's depths (128, and down's 384) up to 2 boards: the 4-board graph runs Triton beside them
@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0), reason="no sm_90 gpu")
def test_the_int8_model_gives_the_same_bits_with_the_small_batch_cuda_kernels(tmp_path, small_calibration, monkeypatch):
    from engine.inference import GraphedBatches, int8_model
    from model import int8_cuda
    _, boards = small_checkpoint(tmp_path)
    boards = boards.numpy().astype("uint8")
    configs = {"projection": [(140, (5, 4, 1, 2, 1))], "gate_up": [(140, (5, 1, 1, 2, 1))], "down": [(140, (3, 4, 1, 6, 1))]}
    monkeypatch.setattr(int8_cuda, "CONFIGS", {gpu_key(): configs})
    monkeypatch.setattr(int8_cuda, "KERNELS", {})
    launched_kinds = set()
    launch = int8_cuda.Kernels.launch

    def counting(self, config, epilogue, *arguments):
        launched_kinds.add(epilogue)
        return launch(self, config, epilogue, *arguments)
    monkeypatch.setattr(int8_cuda.Kernels, "launch", counting)
    flags = torch.backends.cuda.matmul.allow_fp16_accumulation, torch.backends.cuda.matmul.allow_tf32
    try:
        answers = {}
        for enabled in (True, False):
            monkeypatch.setattr(int8_cuda, "ENABLED", enabled)
            network = GraphedBatches(int8_model(tmp_path / "model.pt", 0), [1, 2, 4], in_flight=2)
            pending = [network.launch(boards[:count]) for count in (1, 2)]
            answers[enabled] = [network.collect(launched).copy() for launched in pending] + [network.collect(network.launch(boards[:3])).copy()]
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation, torch.backends.cuda.matmul.allow_tf32 = flags

    #every epilogue ran in CUDA: the projection's, gate_up's, down's and down's with the next block's input mixed in
    assert launched_kinds == {0, 1, 2, 3}
    assert all(np.array_equal(found, expected) for found, expected in zip(answers[True], answers[False]))


def failing(*arguments):
    raise AssertionError("computed again")


#the weights are computed once per checkpoint content and rounding settings and read back as they were saved: a copy of
#the checkpoint elsewhere reads them too, a checkpoint with another weight or another rounding setting gets its own, and
#computing them again gives the same bits (GPTQ in fp64)
@gpu
def test_the_int8_weights_are_cached_by_the_checkpoints_content(tmp_path, small_calibration, monkeypatch):
    model, _ = small_checkpoint(tmp_path)
    computed = quantization.int8_weights(tmp_path / "model.pt", 0)
    assert sorted(computed) == [0, 1] and sorted(computed[0]) == ["down", "gate_up", "projection"]
    (tmp_path / "copy").mkdir()
    shutil.copy(tmp_path / "model.pt", tmp_path / "copy" / "model.pt")
    changed = copy.deepcopy(model)
    with torch.no_grad():
        changed.encoder[0].Q.weight[0, 0] += 1
    save_checkpoint(changed, SMALL_SETTINGS, tmp_path / "changed.pt")
    recomputed = quantization.int8_weights(tmp_path / "changed.pt", 0)
    quantization.cached_weights_path(tmp_path / "changed.pt", 0).unlink()
    recomputed_again = quantization.int8_weights(tmp_path / "changed.pt", 0)

    monkeypatch.setattr(quantization, "compute_int8_weights", failing)
    cached = quantization.int8_weights(tmp_path / "copy" / "model.pt", 0)

    assert len(list(quantization.CACHE.iterdir())) == 2
    for layer, matmuls in computed.items():
        for name, (integers, scale) in matmuls.items():
            assert integers.dtype == torch.int8 and scale.dtype == torch.float32
            assert torch.equal(cached[layer][name][0], integers) and torch.equal(cached[layer][name][1], scale)
            assert all(torch.equal(a, b) for a, b in zip(recomputed[layer][name], recomputed_again[layer][name]))
    assert not torch.equal(recomputed[0]["projection"][0], computed[0]["projection"][0])
    path = quantization.cached_weights_path(tmp_path / "model.pt", 0)
    monkeypatch.setattr(quantization, "DAMPENING", 2 * quantization.DAMPENING)
    assert quantization.cached_weights_path(tmp_path / "model.pt", 0) != path


#the weights without the blocks kept in fp16 are cached by their count, beside the all-int8 weights they are taken from
#(rounded once) and keeping their choice (measured once); the all-int8 weights keep the name of the files cached before
#blocks could be kept in fp16, so those are read rather than rounded again
@gpu
def test_the_weights_without_fp16_blocks_are_cached_by_their_count(tmp_path, small_calibration, monkeypatch):
    path = tmp_path / "model.pt"
    small_checkpoint(tmp_path, deeper(4))
    one = quantization.int8_weights(path, 1)
    monkeypatch.setattr(quantization, "compute_int8_weights", failing)
    two = quantization.int8_weights(path, 2)
    monkeypatch.setattr(quantization, "block_errors", failing)
    one_again, all_int8 = quantization.int8_weights(path, 1), quantization.int8_weights(path, 0)

    assert sorted(all_int8) == [0, 1, 2] and len(one) == 2 and len(two) == 1 and set(two) < set(one) == set(one_again)
    for layer, matmuls in one.items():
        for name, pair in matmuls.items():
            assert all(torch.equal(tensor, cached) and torch.equal(tensor, expected)
                       for tensor, cached, expected in zip(pair, one_again[layer][name], all_int8[layer][name]))
    assert sorted(quantization.CACHE.iterdir()) == sorted(quantization.cached_weights_path(path, count) for count in (0, 1, 2))
    #production's calibration settings
    monkeypatch.undo()
    assert quantization.cached_weights_path(path, 0).name.endswith("_2199eb67a639465a.pt")


#the block kept in fp16 is the one whose int8 costs the policy most, here the one whose gate rounds badly, and keeping it
#brings the priors near the fp32 model's: between int8 blocks, and just before the last (the two fp16 blocks linked)
@gpu
@pytest.mark.parametrize("outlier", [2, 3])
def test_the_block_whose_int8_costs_most_stays_in_fp16(tmp_path, small_calibration, outlier):
    from engine.inference import answer_rows, int8_model
    model, boards = outlier_checkpoint(tmp_path, outlier)
    boards = boards.cuda()
    flags = torch.backends.cuda.matmul.allow_fp16_accumulation, torch.backends.cuda.matmul.allow_tf32
    try:
        errors = quantization.block_errors(tmp_path / "model.pt", quantization.int8_weights(tmp_path / "model.pt", 0))
        with torch.no_grad():
            expected_logits, _ = model.cuda()(boards)
            kept_logits = answer_rows(int8_model(tmp_path / "model.pt", 1)(boards))[:, :-1]
            all_int8_logits = answer_rows(int8_model(tmp_path / "model.pt", 0)(boards))[:, :-1]
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation, torch.backends.cuda.matmul.allow_tf32 = flags

    assert sorted(errors) == [0, 1, 2, 3] and max(errors, key=errors.get) == outlier
    assert sorted(quantization.int8_weights(tmp_path / "model.pt", 1)) == [layer for layer in range(4) if layer != outlier]
    distance = lambda found: 0.5 * (torch.softmax(found.float(), dim=-1) - torch.softmax(expected_logits, dim=-1)).abs().sum(dim=-1)
    assert distance(kept_logits).max() < 0.005 and distance(kept_logits).mean() < 0.5 * distance(all_int8_logits).mean()


#an fp16 block between int8 blocks mixes its own input and computes as the fp16 model's block does, and the int8 block
#after it mixes its own: the model answers to the bit as its blocks run one after another, each mixing its own input,
#the int8 ones taken from the all-int8 model and the fp16 ones from the fp16 model
@gpu
def test_fp16_blocks_between_int8_blocks_compute_as_the_fp16_models(tmp_path, small_calibration):
    from torch import nn
    from engine.inference import answer_rows, fp16_model, int8_model
    from model.fused_blocks import FusedBlock
    from model.int8_blocks import Int8Block
    _, boards = outlier_checkpoint(tmp_path, 2)
    boards = boards.cuda()
    flags = torch.backends.cuda.matmul.allow_fp16_accumulation, torch.backends.cuda.matmul.allow_tf32
    try:
        with torch.no_grad():
            kept = int8_model(tmp_path / "model.pt", 1)
            answers = answer_rows(kept(boards))
            all_int8, one_by_one = int8_model(tmp_path / "model.pt", 0), fp16_model(tmp_path / "model.pt")
            blocks = [one_by_one.encoder[layer] if layer in (2, 4) else all_int8.encoder[layer] for layer in range(5)]
            for block in blocks:
                block.next_scales, block.input_mixed = None, False
            one_by_one.encoder = nn.ModuleList(blocks)
            expected = answer_rows(one_by_one(boards))
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation, torch.backends.cuda.matmul.allow_tf32 = flags

    assert [type(block) for block in kept.encoder] == [Int8Block, Int8Block, FusedBlock, Int8Block, FusedBlock]
    assert torch.equal(answers, expected)

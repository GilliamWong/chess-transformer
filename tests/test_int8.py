#the int8 network (model/int8_blocks.py, engine/quantization.py): its kernels against fp32 references, GPTQ's
#rounding, the weights' cache, and the int8 model against the fp32 one
import copy
import shutil

import chess
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


#int8_linear's output is the dequantised operands' matmul (exact in int32) with linear_epilogue's bias, residual or
#silu(gate) * up, to fp16's rounding, on rows and columns that leave part of a tile empty
@gpu
@pytest.mark.parametrize("gated", [False, True])
def test_int8_linear_is_the_dequantised_matmul(gated):
    from model.int8_blocks import PROJECTION_TILE, GATE_UP_TILE, int8_linear
    torch.manual_seed(0)
    width = 200
    x = torch.randint(-127, 128, (3, 70, 256), device="cuda", dtype=torch.int8)
    x_scale = torch.rand(3, 70, device="cuda") / 100
    weight = torch.randint(-127, 128, (2 * width if gated else width, 256), device="cuda", dtype=torch.int8)
    weight_scale = torch.rand(len(weight), device="cuda") / 100
    bias = None if gated else torch.randn(width, device="cuda").half()
    residual = None if gated else torch.randn(3, 70, width, device="cuda").half()

    out = int8_linear(x, x_scale, weight, weight_scale, bias, residual, gated, list(GATE_UP_TILE if gated else PROJECTION_TILE))

    projected = dequantised(x.reshape(-1, 256), x_scale.reshape(-1)) @ dequantised(weight, weight_scale).t()
    if gated:
        gate, up = projected.chunk(2, dim=-1)
        expected = F.silu(gate) * up
    else:
        expected = projected + bias.double() + residual.reshape(-1, width).double()
    assert out.dtype == torch.float16 and out.shape == (3, 70, width)
    assert torch.allclose(out.reshape(-1, width).double(), expected, atol=1e-3, rtol=2e-3)


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


SMALL_SETTINGS = {"d_model": 128, "num_layers": 3, "n_heads": 4, "kv_heads": 2, "qk_ratio": [0.25, 0.5, 0.75], "mlp_hidden": 320, "value_bins": 16}


#a small model saved in tmp_path, its input skip mixed in, and real positions to check it on
def small_checkpoint(tmp_path) -> tuple[ChessTransformer, torch.Tensor]:
    torch.manual_seed(0)
    model = ChessTransformer(**SMALL_SETTINGS).eval()
    mix_in_the_input(model)
    with torch.no_grad():
        model.templates.normal_()
        model.square_features.table.normal_(std=0.1)
        #sharp enough priors that a wrong scale or rotation shows
        model.from_projection.weight.mul_(8)
    save_checkpoint(model, SMALL_SETTINGS, tmp_path / "model.pt")
    board, boards = chess.Board(), []
    for san in "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6 f3 Be7".split():
        board.push_san(san)
        boards.append(encode_board(board))
    return model, torch.tensor(boards)


@pytest.fixture
def small_calibration(monkeypatch, tmp_path):
    monkeypatch.setattr(quantization, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(quantization, "CALIBRATION_POSITIONS", 64)


#the int8 model (inference.int8_model: every block but the last in int8) gives the fp32 model's priors and score to
#int8's rounding, with the same top moves; the fp16 model is nearer
@gpu
def test_the_int8_model_computes_what_the_model_does(tmp_path, small_calibration):
    from engine.inference import fp16_model, int8_model
    from model.int8_blocks import Int8Block
    model, boards = small_checkpoint(tmp_path)
    boards = boards.cuda()
    flags = torch.backends.cuda.matmul.allow_fp16_accumulation, torch.backends.cuda.matmul.allow_tf32
    try:
        with torch.no_grad():
            expected_logits, expected_values = model.cuda()(boards)
            int8 = int8_model(tmp_path / "model.pt")
            logits, values = int8(boards)
            fp16_logits, _ = fp16_model(tmp_path / "model.pt")(boards)
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation, torch.backends.cuda.matmul.allow_tf32 = flags

    assert [isinstance(block, Int8Block) for block in int8.encoder] == [True, True, False]
    distance = lambda found: 0.5 * (torch.softmax(found.float(), dim=-1) - torch.softmax(expected_logits, dim=-1)).abs().sum(dim=-1)
    assert distance(logits).max() < 0.02 and distance(fp16_logits).max() < distance(logits).max()
    assert torch.equal(logits.argmax(-1), expected_logits.argmax(-1))
    assert torch.allclose(expected_score(values.float()), expected_score(expected_values), atol=5e-3)


#the weights are computed once per checkpoint content and read back as they were saved: a copy of the checkpoint
#elsewhere reads them too, a checkpoint with another weight gets its own, and computing them again gives the same
#bits (GPTQ in fp64)
@gpu
def test_the_int8_weights_are_cached_by_the_checkpoints_content(tmp_path, small_calibration, monkeypatch):
    model, _ = small_checkpoint(tmp_path)
    computed = quantization.int8_weights(tmp_path / "model.pt")
    assert sorted(computed) == [0, 1] and sorted(computed[0]) == ["down", "gate_up", "projection"]
    (tmp_path / "copy").mkdir()
    shutil.copy(tmp_path / "model.pt", tmp_path / "copy" / "model.pt")
    changed = copy.deepcopy(model)
    with torch.no_grad():
        changed.encoder[0].Q.weight[0, 0] += 1
    save_checkpoint(changed, SMALL_SETTINGS, tmp_path / "changed.pt")
    recomputed = quantization.int8_weights(tmp_path / "changed.pt")
    for path in quantization.CACHE.iterdir():
        if path.name.startswith(quantization.checkpoint_digest(tmp_path / "changed.pt")):
            path.unlink()
    recomputed_again = quantization.int8_weights(tmp_path / "changed.pt")

    def failing(*arguments):
        raise AssertionError("computed again")
    monkeypatch.setattr(quantization, "compute_int8_weights", failing)
    cached = quantization.int8_weights(tmp_path / "copy" / "model.pt")

    assert len(list(quantization.CACHE.iterdir())) == 2
    for layer, matmuls in computed.items():
        for name, (integers, scale) in matmuls.items():
            assert integers.dtype == torch.int8 and scale.dtype == torch.float32
            assert torch.equal(cached[layer][name][0], integers) and torch.equal(cached[layer][name][1], scale)
            assert all(torch.equal(a, b) for a, b in zip(recomputed[layer][name], recomputed_again[layer][name]))
    assert not torch.equal(recomputed[0]["projection"][0], computed[0]["projection"][0])

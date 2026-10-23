#the engine's blocks with their matmuls in int8 but combine_heads: weights scaled per output channel, inputs per token
import math

import torch
import triton
import triton.language as tl
from torch import nn
from triton.language.extra import libdevice

from model.fused_blocks import (GROUP_M, FusedBlock, dependent_launch, dependent_launch_wait, linear, linear_epilogue, link_input_mixes,
                                template_attention, tile_for)

#the mlp's hidden is rotated in blocks of this many to spread its outlier channels, down's weights to match (H H^T = 32 I)
HADAMARD_BLOCK = 32
#the down matmul reads its inputs (the mlp's hidden width, padded with zeros) BLOCK_K at a time
DOWN_INPUT_MULTIPLE = 128
#int8 matmul tiles (rows, columns, depth, warps, stages[, registers a thread may use]) by compute capability and matmul,
#as fused_blocks.tile_for lists, within 99 KB of shared memory; a tile changes only the speed (the int32 sums are exact).
#sm_120's were timed in the engine's network on an RTX 5070 Ti and an RTX 5090, one batch and two at once: below 16 rows
#a multiprocessor the narrower tiles fill more of them. The others' are the fastest at 32 to 128 boards on an A100, A40
#and L40S. Only sm_120's down is capped (timed there): two blocks share a multiprocessor
INT8_TILES = {(12, 0): {"projection": [(4, (64, 64, 64, 4, 4)), (math.inf, (128, 128, 64, 8, 3))],
                        "gate_up": [(4, (64, 32, 64, 4, 4)), (8, (128, 32, 64, 4, 3)), (math.inf, (128, 64, 64, 4, 2))],
                        "down": [(2, (128, 32, 128, 4, 4)), (4, (64, 64, 64, 4, 4)), (12, (128, 64, 128, 8, 3)),
                                 (math.inf, (128, 128, 64, 8, 3, 128))]},
              (8, 9): {"projection": [(math.inf, (128, 64, 128, 4, 3))], "gate_up": [(math.inf, (128, 64, 64, 4, 3))],
                       "down": [(math.inf, (128, 64, 128, 4, 3))]},
              (8, 6): {"projection": [(math.inf, (128, 64, 128, 4, 3))], "gate_up": [(math.inf, (128, 64, 64, 4, 3))],
                       "down": [(math.inf, (64, 128, 128, 4, 3))]},
              (8, 0): {"projection": [(math.inf, (128, 64, 128, 4, 3))], "gate_up": [(math.inf, (128, 32, 64, 4, 3))],
                       "down": [(math.inf, (128, 64, 128, 4, 3))]}}
#the rows a program of norm_quant_kernel takes and its warps
NORM_ROWS, NORM_WARPS = 1, 2
#the rows a program of rotate_quant_kernel takes by compute capability, [(the most rows a multiprocessor's share may be,
#rows), ...]: fewer at a few boards on sm_120 (timed on an RTX 5070 Ti and 5090), UNTIMED_ROTATE_ROWS on other gpus.
#Any rows give the same bits
ROTATE_ROWS = {(12, 0): [(4, 2), (math.inf, 8)]}
UNTIMED_ROTATE_ROWS = 8


#a matmul's tile for M rows on the gpu, by its compute capability (one not timed takes sm_120's for any rows)
def int8_tile(device: torch.device, matmul: str, M: int) -> tuple[int, ...]:
    capability = torch.cuda.get_device_capability(device)
    tiles = INT8_TILES[capability][matmul] if capability in INT8_TILES else INT8_TILES[(12, 0)][matmul][-1:]
    return tile_for(tiles, M, torch.cuda.get_device_properties(device).multi_processor_count)


#rotate_quant_kernel's rows a program for M rows on the gpu
def rotate_rows(device: torch.device, M: int) -> int:
    shares = ROTATE_ROWS.get(torch.cuda.get_device_capability(device))
    if shares is None:
        return UNTIMED_ROTATE_ROWS
    share = M / torch.cuda.get_device_properties(device).multi_processor_count
    return next(rows for most, rows in shares if share <= most)


#Sylvester's Hadamard matrix of +-1, size a power of two: symmetric, and its square is size times the identity
def hadamard(size: int) -> torch.Tensor:
    matrix = torch.ones(1, 1)
    while matrix.shape[0] < size:
        matrix = torch.cat([torch.cat([matrix, matrix], 1), torch.cat([matrix, -matrix], 1)], 0)
    return matrix


#the int8 inputs' width for the down matmul
def down_inputs(hidden: int) -> int:
    return triton.cdiv(hidden, DOWN_INPUT_MULTIPLE) * DOWN_INPUT_MULTIPLE


#x @ w^T (x M x K, w N x K, gated: the gate's rows then up's) dequantised, then linear_epilogue; loads past the end wrap
@triton.jit
def int8_linear_kernel(x_ptr, x_scale_ptr, w_ptr, w_scale_ptr, bias_ptr, residual_ptr, original_ptr, residual_scale_ptr, input_scale_ptr, out_ptr,
                       M, N, K, HAS_BIAS: tl.constexpr, HAS_RESIDUAL: tl.constexpr, GATED: tl.constexpr, MIX: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
                       PDL: tl.constexpr):
    dependent_launch_wait(PDL)
    pid = tl.program_id(0)
    grid_m, grid_n = tl.cdiv(M, BLOCK_M), tl.cdiv(N, BLOCK_N)
    in_group = GROUP_M * grid_n
    first_m = pid // in_group * GROUP_M
    group_rows = min(grid_m - first_m, GROUP_M)
    pid_m = first_m + pid % in_group % group_rows
    pid_n = pid % in_group // group_rows
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    depth = tl.arange(0, BLOCK_K)
    row_ok, col_ok = rows < M, cols < N
    x_ptrs = x_ptr + (rows % M)[:, None] * K + depth[None, :]
    w_ptrs = w_ptr + (cols % N)[None, :] * K + depth[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs)
        acc = tl.dot(x, tl.load(w_ptrs), acc, out_dtype=tl.int32)
        if GATED:
            up = tl.dot(x, tl.load(w_ptrs + N * K), up, out_dtype=tl.int32)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    x_scale = tl.load(x_scale_ptr + rows, mask=row_ok, other=0.0)[:, None]
    out = acc.to(tl.float32) * x_scale * tl.load(w_scale_ptr + cols, mask=col_ok, other=0.0)[None, :]
    up_out = out
    if GATED:
        up_out = up.to(tl.float32) * x_scale * tl.load(w_scale_ptr + N + cols, mask=col_ok, other=0.0)[None, :]
    linear_epilogue(out, up_out, rows, cols, M, N, bias_ptr, residual_ptr, original_ptr, residual_scale_ptr, input_scale_ptr, out_ptr, HAS_BIAS,
                    HAS_RESIDUAL, GATED, MIX)


#int8_linear with a tile of INT8_TILES'
def launch_int8_linear(x: torch.Tensor, x_scale: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor | None,
                       residual: torch.Tensor | None, gated: bool, original: torch.Tensor | None, residual_scale: torch.Tensor | None,
                       input_scale: torch.Tensor | None, tile: tuple[int, ...]) -> torch.Tensor:
    block_m, block_n, block_k, warps, stages, *registers = tile
    rows, K = weight.shape
    N = rows // 2 if gated else rows
    if K % block_k != 0:
        raise ValueError(f"int8_linear_kernel reads the {K} inputs {block_k} at a time")
    x = x.contiguous()
    M = x.numel() // K
    out = torch.empty(*x.shape[:-1], N, device=x.device, dtype=torch.float16)
    mix = original is not None
    #the pointers of unused inputs are never read
    bias_ = bias if bias is not None else out
    residual_ = residual.contiguous() if residual is not None else out
    mix_ = (original.contiguous(), residual_scale, input_scale) if mix else (out, out, out)
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    int8_linear_kernel[grid](x, x_scale, weight, weight_scale, bias_, residual_, *mix_, out, M, N, K, HAS_BIAS=bias is not None,
                             HAS_RESIDUAL=residual is not None, GATED=gated, MIX=mix, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
                             GROUP_M=GROUP_M, num_warps=warps, num_stages=stages, **({"maxnreg": registers[0]} if registers else {}),
                             **dependent_launch(x.device))
    return out


#an opaque op for torch.compile, as fused_blocks.linear; gives fp16. matmul names its INT8_TILES (projection, gate_up
#or down). With original (the input embedding) and the next block's residual_scale and input_scale it gives the next
#block's input (linear_epilogue)
@torch.library.custom_op("chess::int8_linear", mutates_args=())
def int8_linear(x: torch.Tensor, x_scale: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor | None,
                residual: torch.Tensor | None, gated: bool, matmul: str, original: torch.Tensor | None = None,
                residual_scale: torch.Tensor | None = None, input_scale: torch.Tensor | None = None) -> torch.Tensor:
    tile = int8_tile(x.device, matmul, x.numel() // weight.shape[1])
    return launch_int8_linear(x, x_scale, weight, weight_scale, bias, residual, gated, original, residual_scale, input_scale, tile)


@int8_linear.register_fake
def _(x, x_scale, weight, weight_scale, bias, residual, gated, matmul, original=None, residual_scale=None, input_scale=None):
    return x.new_empty(*x.shape[:-1], weight.shape[0] // 2 if gated else weight.shape[0], dtype=torch.float16)


#rows RMS-normalised (with gain) and rounded to int8, a row's scale its absmax / 127; MIX first mixes in the block's input
@triton.jit
def norm_quant_kernel(x_ptr, original_ptr, residual_scale_ptr, input_scale_ptr, gain_ptr, mixed_ptr, q_ptr, scale_ptr, M, D, eps,
                      MIX: tl.constexpr, ROWS: tl.constexpr, BLOCK_D: tl.constexpr, PDL: tl.constexpr):
    dependent_launch_wait(PDL)
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    d = tl.arange(0, BLOCK_D)
    row_ok = rows < M
    mask = row_ok[:, None] & (d < D)[None, :]
    offsets = rows[:, None] * D + d[None, :]
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    if MIX:
        original = tl.load(original_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mixed = (tl.load(residual_scale_ptr).to(tl.float32) * x + tl.load(input_scale_ptr).to(tl.float32) * original).to(tl.float16)
        tl.store(mixed_ptr + offsets, mixed, mask=mask)
        x = mixed.to(tl.float32)
    gain = tl.load(gain_ptr + d, mask=d < D, other=0.0).to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x, 1) / D + eps)[:, None] * gain[None, :]
    absmax = tl.maximum(tl.max(tl.abs(y), 1), 1e-20)
    tl.store(q_ptr + offsets, libdevice.rint(y * (127.0 / absmax)[:, None]).to(tl.int8), mask=mask)
    tl.store(scale_ptr + rows, absmax / 127.0, mask=row_ok)


def launch_norm_quant(x, original, residual_scale, input_scale, gain, mixed, eps):
    D = x.shape[-1]
    M = x.numel() // D
    q = torch.empty(x.shape, device=x.device, dtype=torch.int8)
    scale = torch.empty(x.shape[:-1], device=x.device, dtype=torch.float32)
    mix = original is not None
    norm_quant_kernel[(triton.cdiv(M, NORM_ROWS),)](x, original if mix else x, residual_scale if mix else gain, input_scale if mix else gain, gain,
                                                    mixed if mix else x, q, scale, M, D, eps, MIX=mix, ROWS=NORM_ROWS,
                                                    BLOCK_D=triton.next_power_of_2(D), num_warps=NORM_WARPS, **dependent_launch(x.device))
    return q, scale


@torch.library.custom_op("chess::norm_quant", mutates_args=())
def norm_quant(x: torch.Tensor, gain: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    return launch_norm_quant(x.contiguous(), None, None, None, gain, None, eps)


@norm_quant.register_fake
def _(x, gain, eps):
    return x.new_empty(x.shape, dtype=torch.int8), x.new_empty(x.shape[:-1], dtype=torch.float32)


#the block's input mix, and its norm_quant
@torch.library.custom_op("chess::mix_norm_quant", mutates_args=())
def mix_norm_quant(x: torch.Tensor, original: torch.Tensor, residual_scale: torch.Tensor, input_scale: torch.Tensor, gain: torch.Tensor,
                   eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.contiguous()
    mixed = torch.empty_like(x, dtype=torch.float16)
    q, scale = launch_norm_quant(x, original.contiguous(), residual_scale, input_scale, gain, mixed, eps)
    return mixed, q, scale


@mix_norm_quant.register_fake
def _(x, original, residual_scale, input_scale, gain, eps):
    return x.new_empty(x.shape, dtype=torch.float16), x.new_empty(x.shape, dtype=torch.int8), x.new_empty(x.shape[:-1], dtype=torch.float32)


#h's rows times the block-diagonal Hadamard matrix, rounded to int8 WIDTH wide; read as rows of BLOCK for one matmul
@triton.jit
def rotate_quant_kernel(h_ptr, hadamard_ptr, q_ptr, scale_ptr, M, N, WIDTH: tl.constexpr, ROWS: tl.constexpr, BLOCK: tl.constexpr,
                        SLOTS: tl.constexpr, PDL: tl.constexpr):
    dependent_launch_wait(PDL)
    r = tl.arange(0, ROWS * SLOTS)
    tokens = tl.program_id(0) * ROWS + r // SLOTS
    i = tl.arange(0, BLOCK)
    cols = (r % SLOTS)[:, None] * BLOCK + i[None, :]
    stored = (tokens < M)[:, None] & (cols < WIDTH)
    h = tl.load(h_ptr + tokens[:, None] * N + cols, mask=stored & (cols < N), other=0.0)
    rotated = tl.dot(h, tl.load(hadamard_ptr + i[:, None] * BLOCK + i[None, :]))
    absmax = tl.maximum(tl.max(tl.reshape(tl.max(tl.abs(rotated), 1), (ROWS, SLOTS)), 1), 1e-20)
    inverse_step = tl.reshape(tl.broadcast_to((127.0 / absmax)[:, None], (ROWS, SLOTS)), (ROWS * SLOTS,))
    tl.store(q_ptr + tokens[:, None] * WIDTH + cols, libdevice.rint(rotated * inverse_step[:, None]).to(tl.int8), mask=stored)
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    tl.store(scale_ptr + rows, absmax / 127.0, mask=rows < M)


@torch.library.custom_op("chess::rotate_quant", mutates_args=())
def rotate_quant(h: torch.Tensor, hadamard_signs: torch.Tensor, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    h = h.contiguous()
    N = h.shape[-1]
    M = h.numel() // N
    q = torch.empty(*h.shape[:-1], width, device=h.device, dtype=torch.int8)
    scale = torch.empty(h.shape[:-1], device=h.device, dtype=torch.float32)
    block = hadamard_signs.shape[0]
    rows = rotate_rows(h.device, M)
    rotate_quant_kernel[(triton.cdiv(M, rows),)](h, hadamard_signs, q, scale, M, N, WIDTH=width, ROWS=rows, BLOCK=block,
                                                 SLOTS=triton.next_power_of_2(width // block), num_warps=4, **dependent_launch(h.device))
    return q, scale


@rotate_quant.register_fake
def _(h, hadamard_signs, width):
    return h.new_empty(*h.shape[:-1], width, dtype=torch.int8), h.new_empty(h.shape[:-1], dtype=torch.float32)


class Int8Block(nn.Module):
    #takes over a FusedBlock with its int8 weights; output_scale multiplies down's, as fp16 inference shrinks the residual
    def __init__(self, block: FusedBlock, weights: dict[str, tuple[torch.Tensor, torch.Tensor]], output_scale: float):
        super().__init__()
        self.heads, self.kv_heads, self.dq, self.dv, self.templates_count = block.heads, block.kv_heads, block.dq, block.dv, block.templates_count
        self.ln1, self.ln2 = block.ln1, block.ln2
        self.residual_scale, self.input_scale = block.residual_scale, block.input_scale
        self.q_gain, self.k_gain, self.eps = block.q_gain, block.k_gain, block.eps
        self.projection_bias = block.projection_bias
        #fp16: combine_heads' input would need a scale per head
        self.combine_heads, self.combine_heads_bias = block.combine_heads, block.combine_heads_bias
        device = block.projection.device
        for name, (integers, scale) in weights.items():
            self.register_buffer(name, integers.to(device).contiguous())
            self.register_buffer(f"{name}_scale", (scale * output_scale if name == "down" else scale).to(device, torch.float32).contiguous())
        self.register_buffer("hadamard_signs", hadamard(HADAMARD_BLOCK).to(device, torch.float16))
        #set by link_input_mixes: the next block's (residual_scale, input_scale) when down mixes its input, and whether this
        #block's input comes mixed
        self.next_scales: tuple[nn.Parameter, nn.Parameter] | None = None
        self.input_mixed = False

    def forward(self, x: torch.Tensor, templates: torch.Tensor, original_board_embed: torch.Tensor) -> torch.Tensor:
        if self.input_mixed:
            normed, scale = norm_quant(x, self.ln1.weight, self.ln1.eps)
        else:
            x, normed, scale = mix_norm_quant(x, original_board_embed, self.residual_scale, self.input_scale, self.ln1.weight, self.ln1.eps)
        projected = int8_linear(normed, scale, self.projection, self.projection_scale, self.projection_bias, None, False, "projection")
        attended = template_attention(projected, templates, self.q_gain, self.k_gain, self.heads, self.kv_heads, self.dq, self.dv, self.templates_count,
                                      self.eps)
        x = linear(attended, self.combine_heads, self.combine_heads_bias, residual=x)
        normed, scale = norm_quant(x, self.ln2.weight, self.ln2.eps)
        hidden = int8_linear(normed, scale, self.gate_up, self.gate_up_scale, None, None, True, "gate_up")
        rotated, scale = rotate_quant(hidden, self.hadamard_signs, self.down.shape[1])
        if self.next_scales is None:
            return int8_linear(rotated, scale, self.down, self.down_scale, None, x, False, "down")
        return int8_linear(rotated, scale, self.down, self.down_scale, None, x, False, "down", original_board_embed, *self.next_scales)


#the fused model (FusedTransformer) with the blocks of the layers in weights swapped for Int8Blocks
def quantize_blocks(model: nn.Module, weights: dict[int, dict], output_scale: float) -> nn.Module:
    blocks = [Int8Block(block, weights[layer], output_scale) if layer in weights else block for layer, block in enumerate(model.encoder)]
    link_input_mixes(blocks)
    model.encoder = nn.ModuleList(blocks)
    return model

#the engine's blocks with their matmuls on normalised inputs (Q|K|V|template mix and gate|up) and the mlp's down in
#int8, for speed: an RTX 5070 Ti multiplies int8 at twice fp16's rate (int32 sums, exact). The weights are int8 with a
#scale per output channel (engine/quantization.py rounds them), the inputs int8 with a scale per token found as they
#are made: the RMSNorm rounds its output, and the mlp's hidden activations are rotated by a Hadamard matrix in blocks
#of HADAMARD_BLOCK first, which spreads their outlier channels over the block (the down weights are rotated to match,
#H H^T being HADAMARD_BLOCK times the identity: (h H)(W H / HADAMARD_BLOCK)^T = h W^T). combine_heads stays fp16: its
#input needs a scale per head
import torch
import triton
import triton.language as tl
from torch import nn
from triton.language.extra import libdevice

from model.fused_blocks import GROUP_M, FusedBlock, linear, linear_epilogue, template_attention

HADAMARD_BLOCK = 32
#the down matmul reads its inputs (the mlp's hidden width, padded with zeros) BLOCK_K at a time
DOWN_INPUT_MULTIPLE = 128
#tiles of the int8 matmuls: BLOCK_M rows x BLOCK_N columns (a gated matmul's of the gate and as many of up), BLOCK_K
#deep, warps, pipeline stages. Each fits in the 99 KB of shared memory an sm_86 / sm_89 / sm_120 block can have
PROJECTION_TILE = (128, 128, 64, 8, 3)
GATE_UP_TILE = (128, 64, 64, 4, 2)
DOWN_TILE = (128, 128, 64, 4, 3)
#rows a program of norm_quant_kernel and of rotate_quant_kernel takes, and its warps
NORM_ROWS, NORM_WARPS = 1, 4
ROTATE_ROWS = 32


#Sylvester's Hadamard matrix of +-1, size a power of two: symmetric, and its square is size times the identity
def hadamard(size: int) -> torch.Tensor:
    matrix = torch.ones(1, 1)
    while matrix.shape[0] < size:
        matrix = torch.cat([torch.cat([matrix, matrix], 1), torch.cat([matrix, -matrix], 1)], 0)
    return matrix


#the int8 inputs' width for the down matmul
def down_inputs(hidden: int) -> int:
    return triton.cdiv(hidden, DOWN_INPUT_MULTIPLE) * DOWN_INPUT_MULTIPLE


#out = (x @ w^T) dequantised by x's scale per row and w's per column, then linear_epilogue's bias, silu(gate) * up and
#residual. x is M x K int8, w N x K int8 (gated: the gate's N rows then up's). Rows and columns past the end are read
#wrapped round, whole loads without masks, and never stored
@triton.jit
def int8_linear_kernel(x_ptr, x_scale_ptr, w_ptr, w_scale_ptr, bias_ptr, residual_ptr, out_ptr, M, N, K,
                       HAS_BIAS: tl.constexpr, HAS_RESIDUAL: tl.constexpr, GATED: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
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
    linear_epilogue(out, up_out, rows, cols, M, N, bias_ptr, residual_ptr, out_ptr, HAS_BIAS, HAS_RESIDUAL, GATED)


#an opaque op for torch.compile, as fused_blocks.linear; gives fp16. tile is one of the *_TILE above
@torch.library.custom_op("chess::int8_linear", mutates_args=())
def int8_linear(x: torch.Tensor, x_scale: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor | None,
                residual: torch.Tensor | None, gated: bool, tile: list[int]) -> torch.Tensor:
    block_m, block_n, block_k, warps, stages = tile
    rows, K = weight.shape
    N = rows // 2 if gated else rows
    if K % block_k != 0:
        raise ValueError(f"int8_linear_kernel reads the {K} inputs {block_k} at a time")
    x = x.contiguous()
    M = x.numel() // K
    out = torch.empty(*x.shape[:-1], N, device=x.device, dtype=torch.float16)
    #the pointers of unused inputs are never read
    bias_ = bias if bias is not None else out
    residual_ = residual.contiguous() if residual is not None else out
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    int8_linear_kernel[grid](x, x_scale, weight, weight_scale, bias_, residual_, out, M, N, K, HAS_BIAS=bias is not None,
                             HAS_RESIDUAL=residual is not None, GATED=gated, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=GROUP_M,
                             num_warps=warps, num_stages=stages)
    return out


@int8_linear.register_fake
def _(x, x_scale, weight, weight_scale, bias, residual, gated, tile):
    return x.new_empty(*x.shape[:-1], weight.shape[0] // 2 if gated else weight.shape[0], dtype=torch.float16)


#rows of x RMS-normalised (with gain) and rounded to int8 with a scale per row, their absmax / 127. With MIX, x is
#first the block's input mix residual_scale * x + input_scale * original, which is written out in fp16 too
@triton.jit
def norm_quant_kernel(x_ptr, original_ptr, residual_scale_ptr, input_scale_ptr, gain_ptr, mixed_ptr, q_ptr, scale_ptr, M, D, eps,
                      MIX: tl.constexpr, ROWS: tl.constexpr, BLOCK_D: tl.constexpr):
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
                                                    BLOCK_D=triton.next_power_of_2(D), num_warps=NORM_WARPS)
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


#rows of h (M x N fp16) times the block-diagonal Hadamard matrix of +-1 (blocks of hadamard_signs' size), rounded to
#int8 with a scale per row, written WIDTH wide with zeros past N. A first pass finds each row's absmax and a second
#rotates again and rounds: the rows are read twice, from L2, rather than held
@triton.jit
def rotate_quant_kernel(h_ptr, hadamard_ptr, q_ptr, scale_ptr, M, N, WIDTH: tl.constexpr, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    i = tl.arange(0, BLOCK)
    row_ok = rows < M
    hadamard = tl.load(hadamard_ptr + i[:, None] * BLOCK + i[None, :])
    absmax = tl.full((ROWS,), 1e-20, tl.float32)
    for start in range(0, WIDTH, BLOCK):
        cols = start + i
        h = tl.load(h_ptr + rows[:, None] * N + cols[None, :], mask=row_ok[:, None] & (cols < N)[None, :], other=0.0)
        absmax = tl.maximum(absmax, tl.max(tl.abs(tl.dot(h, hadamard)), 1))
    for start in range(0, WIDTH, BLOCK):
        cols = start + i
        h = tl.load(h_ptr + rows[:, None] * N + cols[None, :], mask=row_ok[:, None] & (cols < N)[None, :], other=0.0)
        q = libdevice.rint(tl.dot(h, hadamard) * (127.0 / absmax)[:, None]).to(tl.int8)
        tl.store(q_ptr + rows[:, None] * WIDTH + cols[None, :], q, mask=row_ok[:, None])
    tl.store(scale_ptr + rows, absmax / 127.0, mask=row_ok)


@torch.library.custom_op("chess::rotate_quant", mutates_args=())
def rotate_quant(h: torch.Tensor, hadamard_signs: torch.Tensor, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    h = h.contiguous()
    N = h.shape[-1]
    M = h.numel() // N
    q = torch.empty(*h.shape[:-1], width, device=h.device, dtype=torch.int8)
    scale = torch.empty(h.shape[:-1], device=h.device, dtype=torch.float32)
    rotate_quant_kernel[(triton.cdiv(M, ROTATE_ROWS),)](h, hadamard_signs, q, scale, M, N, WIDTH=width, ROWS=ROTATE_ROWS, BLOCK=hadamard_signs.shape[0],
                                                        num_warps=4)
    return q, scale


@rotate_quant.register_fake
def _(h, hadamard_signs, width):
    return h.new_empty(*h.shape[:-1], width, dtype=torch.int8), h.new_empty(h.shape[:-1], dtype=torch.float32)


class Int8Block(nn.Module):
    #takes over a FusedBlock's norms, attention and combine_heads, with the int8 weights and their scales of its
    #projection, gate_up and down (engine/quantization.py); output_scale multiplies down's, as fp16 inference shrinks
    #what the blocks write into the residual stream
    def __init__(self, block: FusedBlock, weights: dict[str, tuple[torch.Tensor, torch.Tensor]], output_scale: float):
        super().__init__()
        self.heads, self.kv_heads, self.dq, self.dv, self.templates_count = block.heads, block.kv_heads, block.dq, block.dv, block.templates_count
        self.ln1, self.ln2 = block.ln1, block.ln2
        self.residual_scale, self.input_scale = block.residual_scale, block.input_scale
        self.q_gain, self.k_gain, self.eps = block.q_gain, block.k_gain, block.eps
        self.projection_bias = block.projection_bias
        self.combine_heads, self.combine_heads_bias = block.combine_heads, block.combine_heads_bias
        device = block.projection.device
        for name, (integers, scale) in weights.items():
            self.register_buffer(name, integers.to(device).contiguous())
            self.register_buffer(f"{name}_scale", (scale * output_scale if name == "down" else scale).to(device, torch.float32).contiguous())
        self.register_buffer("hadamard_signs", hadamard(HADAMARD_BLOCK).to(device, torch.float16))

    def forward(self, x: torch.Tensor, templates: torch.Tensor, original_board_embed: torch.Tensor) -> torch.Tensor:
        x, normed, scale = mix_norm_quant(x, original_board_embed, self.residual_scale, self.input_scale, self.ln1.weight, self.ln1.eps)
        projected = int8_linear(normed, scale, self.projection, self.projection_scale, self.projection_bias, None, False, PROJECTION_TILE)
        attended = template_attention(projected, templates, self.q_gain, self.k_gain, self.heads, self.kv_heads, self.dq, self.dv, self.templates_count,
                                      self.eps)
        x = linear(attended, self.combine_heads, self.combine_heads_bias, residual=x)
        normed, scale = norm_quant(x, self.ln2.weight, self.ln2.eps)
        hidden = int8_linear(normed, scale, self.gate_up, self.gate_up_scale, None, None, True, GATE_UP_TILE)
        rotated, scale = rotate_quant(hidden, self.hadamard_signs, self.down.shape[1])
        return int8_linear(rotated, scale, self.down, self.down_scale, None, x, False, DOWN_TILE)


#the fused model (fuse_blocks) with the blocks of the layers in weights swapped for Int8Blocks
def quantize_blocks(model: nn.Module, weights: dict[int, dict], output_scale: float) -> nn.Module:
    model.encoder = nn.ModuleList(Int8Block(block, weights[layer], output_scale) if layer in weights else block
                                  for layer, block in enumerate(model.encoder))
    return model

#the engine's transformer blocks for inference on the gpu: the same arithmetic as model.TransformerBlock, arranged
#for speed (tests/test_inference.py checks they agree, so a change to the block must be made here too). Q, K, V and
#the template mix are one matmul, the mlp's gate and up another, and attention is one Triton kernel: it RMS-normalises
#the queries and keys (QK-norm), adds the template bias, takes the softmax and mixes the values without writing the
#scores, the bias or any transposed copy to memory, and each query head reads its group's key/value head in place
#instead of a copy (repeat_interleave). The matmuls are a Triton kernel too, with the bias, the mlp's silu(gate) * up
#and the residual add in the same kernel
import math

import torch
import triton
import triton.language as tl
from torch import nn

from model import NUM_SQUARES, SquareFeatures, TransformerBlock

#an instance of the kernel takes BLOCK_M query rows of one head for CHUNK positions; fastest on an RTX 4090 of the
#tried 16/32/64 rows, chunks of 4/8/16 and 2/4/8 warps (41 us a block of 64 positions, the first version 57 us)
BLOCK_M, CHUNK, WARPS = 16, 4, 2


#an instance reads its rows of the templates once and keeps them for its whole chunk of positions. The keys are the 64
#squares (template bias) and the metadata tokens (none), 64 + 16 columns rather than 70 padded to 128
@triton.jit
def template_attention_kernel(q_ptr, k_ptr, v_ptr, c_ptr, templates_ptr, q_gain_ptr, k_gain_ptr, out_ptr,
                              B, stride_b, stride_t, out_stride_b, out_stride_t, eps, scale,
                              T: tl.constexpr, HEADS_PER_KV: tl.constexpr, DQ: tl.constexpr, DV: tl.constexpr, TEMPLATES: tl.constexpr,
                              BLOCK_M: tl.constexpr, BLOCK_META: tl.constexpr, BLOCK_DQ: tl.constexpr, CHUNK: tl.constexpr):
    m, h, chunk = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    g = h // HEADS_PER_KV
    rows = m * BLOCK_M + tl.arange(0, BLOCK_M)
    squares = tl.arange(0, 64)
    meta = 64 + tl.arange(0, BLOCK_META)
    d = tl.arange(0, BLOCK_DQ)
    dv = tl.arange(0, DV)
    row_ok, meta_ok, d_ok, square_rows = rows < T, meta < T, d < DQ, rows < 64

    q_gain = tl.load(q_gain_ptr + d, mask=d_ok, other=0.0).to(tl.float32)
    k_gain = tl.load(k_gain_ptr + d, mask=d_ok, other=0.0).to(tl.float32)
    #TEMPLATES x BLOCK_M x 64
    tt = tl.arange(0, TEMPLATES)
    templates = tl.load(templates_ptr + tt[:, None, None] * 4096 + rows[None, :, None] * 64 + squares[None, None, :],
                        mask=square_rows[None, :, None], other=0.0).to(tl.float32)

    for i in range(CHUNK):
        b = chunk * CHUNK + i
        if b < B:
            #q, k, v and the template coefficients are column slices of one projection, so they share its strides
            base = b * stride_b
            q = tl.load(q_ptr + base + rows[:, None] * stride_t + h * DQ + d[None, :], mask=row_ok[:, None] & d_ok[None, :], other=0.0).to(tl.float32)
            q = (q * tl.rsqrt(tl.sum(q * q, 1) / DQ + eps)[:, None] * q_gain[None, :]).to(tl.float16)
            k_sq = tl.load(k_ptr + base + squares[:, None] * stride_t + g * DQ + d[None, :], mask=d_ok[None, :], other=0.0).to(tl.float32)
            k_sq = k_sq * tl.rsqrt(tl.sum(k_sq * k_sq, 1) / DQ + eps)[:, None] * k_gain[None, :]
            k_meta = tl.load(k_ptr + base + meta[:, None] * stride_t + g * DQ + d[None, :], mask=meta_ok[:, None] & d_ok[None, :], other=0.0).to(tl.float32)
            k_meta = k_meta * tl.rsqrt(tl.sum(k_meta * k_meta, 1) / DQ + eps)[:, None] * k_gain[None, :]
            s_sq = tl.dot(q, tl.trans(k_sq.to(tl.float16))) * scale
            s_meta = tl.dot(q, tl.trans(k_meta.to(tl.float16))) * scale

            #bias[i, j] = the sum over templates t of coefficient[i, t] * template[t, i, j]
            coefficients = tl.load(c_ptr + base + rows[None, :] * stride_t + h * TEMPLATES + tt[:, None], mask=square_rows[None, :], other=0.0).to(tl.float32)
            s_sq += tl.sum(coefficients[:, :, None] * templates, 0)

            s_meta = tl.where(meta_ok[None, :], s_meta, float("-inf"))
            top = tl.maximum(tl.max(s_sq, 1), tl.max(s_meta, 1))
            p_sq = tl.exp(s_sq - top[:, None])
            p_meta = tl.exp(s_meta - top[:, None])
            total = tl.sum(p_sq, 1) + tl.sum(p_meta, 1)
            v_sq = tl.load(v_ptr + base + squares[:, None] * stride_t + g * DV + dv[None, :])
            v_meta = tl.load(v_ptr + base + meta[:, None] * stride_t + g * DV + dv[None, :], mask=meta_ok[:, None], other=0.0)
            out = tl.dot(p_sq.to(tl.float16), v_sq.to(tl.float16)) + tl.dot(p_meta.to(tl.float16), v_meta.to(tl.float16))
            out = out / total[:, None]
            tl.store(out_ptr + b * out_stride_b + rows[:, None] * out_stride_t + h * DV + dv[None, :], out.to(out_ptr.dtype.element_ty), mask=row_ok[:, None])


#an opaque op for torch.compile: compiling through the kernel call made it ~15x slower
@torch.library.custom_op("chess::template_attention", mutates_args=())
def template_attention(projected: torch.Tensor, templates: torch.Tensor, q_gain: torch.Tensor, k_gain: torch.Tensor, heads: int, kv_heads: int,
                       dq: int, dv: int, templates_count: int, eps: float) -> torch.Tensor:
    projected = projected.contiguous()
    B, T, _ = projected.shape
    q, k, v, c = projected.split([heads * dq, kv_heads * dq, kv_heads * dv, heads * templates_count], dim=-1)
    out = torch.empty(B, T, heads * dv, device=projected.device, dtype=projected.dtype)
    grid = (triton.cdiv(T, BLOCK_M), heads, triton.cdiv(B, CHUNK))
    template_attention_kernel[grid](q, k, v, c, templates, q_gain, k_gain, out, B, projected.stride(0), projected.stride(1), out.stride(0), out.stride(1),
                                    eps, 1 / math.sqrt(dq), T=T, HEADS_PER_KV=heads // kv_heads, DQ=dq, DV=dv, TEMPLATES=templates_count,
                                    BLOCK_M=BLOCK_M, BLOCK_META=max(16, triton.next_power_of_2(T - NUM_SQUARES)),
                                    BLOCK_DQ=max(16, triton.next_power_of_2(dq)), CHUNK=CHUNK, num_warps=WARPS)
    return out


@template_attention.register_fake
def _(projected, templates, q_gain, k_gain, heads, kv_heads, dq, dv, templates_count, eps):
    B, T, _ = projected.shape
    return projected.new_empty(B, T, heads * dv)


#a tile of the blocks' matmuls is LINEAR_ROWS x LINEAR_COLUMNS, LINEAR_DEPTH deep (a gated matmul's is LINEAR_COLUMNS / 2
#columns of the gate and the same of up), and GROUP_M row tiles go through the columns together so their rows stay in
#L2. The fastest on an RTX 4090 at 64 positions of 32 to 128 rows, 64 to 256 columns, 2 to 4 stages and 4 or 8 warps;
#smaller tiles for smaller batches gained under 1% of a search
LINEAR_ROWS, LINEAR_COLUMNS, LINEAR_DEPTH, LINEAR_WARPS, LINEAR_STAGES, GROUP_M = 128, 128, 32, 4, 3, 8


#out = x @ w (w is in_features x out_features), then in the same kernel the bias, silu(gate) * up (gated: w's first
#half of columns is the gate, the second up) and the residual added. fp16 inputs accumulate in fp16, as cuBLAS does
#with allow_fp16_accumulation (twice fp32 accumulation's speed on a geforce card); the elementwise work is in fp32
@triton.jit
def linear_kernel(x_ptr, w_ptr, bias_ptr, residual_ptr, out_ptr, M, N, K, stride_w,
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
    x_ptrs = x_ptr + rows[:, None] * K + depth[None, :]
    w_ptrs = w_ptr + depth[:, None] * stride_w + cols[None, :]

    acc_type: tl.constexpr = tl.float16 if x_ptr.dtype.element_ty == tl.float16 else tl.float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_type)
    up = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_type)
    for _ in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs, mask=row_ok[:, None], other=0.0)
        acc = tl.dot(x, tl.load(w_ptrs, mask=col_ok[None, :], other=0.0), acc, out_dtype=acc_type)
        if GATED:
            up = tl.dot(x, tl.load(w_ptrs + N, mask=col_ok[None, :], other=0.0), up, out_dtype=acc_type)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K * stride_w

    out = acc.to(tl.float32)
    if HAS_BIAS:
        out += tl.load(bias_ptr + cols, mask=col_ok, other=0.0).to(tl.float32)[None, :]
    if GATED:
        out = out * tl.sigmoid(out) * up.to(tl.float32)
    out_offsets = rows[:, None] * N + cols[None, :]
    if HAS_RESIDUAL:
        out += tl.load(residual_ptr + out_offsets, mask=row_ok[:, None] & col_ok[None, :], other=0.0).to(tl.float32)
    tl.store(out_ptr + out_offsets, out.to(out_ptr.dtype.element_ty), mask=row_ok[:, None] & col_ok[None, :])


#an opaque op for torch.compile, as template_attention
@torch.library.custom_op("chess::linear", mutates_args=())
def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None, residual: torch.Tensor | None = None,
           gated: bool = False) -> torch.Tensor:
    K, width = weight.shape
    N = width // 2 if gated else width
    if K % LINEAR_DEPTH != 0:
        raise ValueError(f"linear_kernel reads the {K} inputs {LINEAR_DEPTH} at a time")
    x = x.contiguous()
    out = x.new_empty(*x.shape[:-1], N)
    M = x.numel() // K
    columns = LINEAR_COLUMNS // 2 if gated else LINEAR_COLUMNS
    #the pointers of unused inputs are never read
    bias_ = bias if bias is not None else weight
    residual_ = residual.contiguous() if residual is not None else out
    grid = (triton.cdiv(M, LINEAR_ROWS) * triton.cdiv(N, columns),)
    linear_kernel[grid](x, weight, bias_, residual_, out, M, N, K, weight.stride(0), HAS_BIAS=bias is not None, HAS_RESIDUAL=residual is not None,
                        GATED=gated, BLOCK_M=LINEAR_ROWS, BLOCK_N=columns, BLOCK_K=LINEAR_DEPTH, GROUP_M=GROUP_M, num_warps=LINEAR_WARPS,
                        num_stages=LINEAR_STAGES)
    return out


@linear.register_fake
def _(x, weight, bias=None, residual=None, gated=False):
    return x.new_empty(*x.shape[:-1], weight.shape[1] // 2 if gated else weight.shape[1])


#a linear layer's weight as linear_kernel reads it, in_features x out_features
def transposed(weight: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(weight.detach().t().contiguous())


class FusedBlock(nn.Module):
    #takes over a trained block's weights (copied into the joined, transposed matmuls, or shared)
    def __init__(self, block: TransformerBlock):
        super().__init__()
        self.heads, self.kv_heads = block.n_heads, block.kv_heads
        self.dq, self.dv, self.templates_count = block.d_qk_head, block.d_head, block.position_templates
        self.ln1, self.ln2 = block.ln1, block.ln2
        self.q_gain, self.k_gain = block.q_norm.weight, block.k_norm.weight
        self.eps = block.q_norm.eps if block.q_norm.eps is not None else torch.finfo(torch.float32).eps
        projections = (block.Q, block.K, block.V, block.template_mix)
        mlp = block.mlp
        self.projection = transposed(torch.cat([projection.weight for projection in projections]))
        self.projection_bias = nn.Parameter(torch.cat([projection.bias for projection in projections]).detach())
        self.combine_heads = transposed(block.combine_heads.weight)
        self.combine_heads_bias = block.combine_heads.bias
        self.gate_up = transposed(torch.cat([mlp.gate_projection.weight, mlp.up_projection.weight]))
        self.down = transposed(mlp.down_projection.weight)

    def forward(self, x: torch.Tensor, templates: torch.Tensor) -> torch.Tensor:
        attended = template_attention(linear(self.ln1(x), self.projection, self.projection_bias), templates, self.q_gain, self.k_gain, self.heads,
                                      self.kv_heads, self.dq, self.dv, self.templates_count, self.eps)
        x = linear(attended, self.combine_heads, self.combine_heads_bias, residual=x)
        return linear(linear(self.ln2(x), self.gate_up, gated=True), self.down, residual=x)


#squares a program of feature_sum_kernel sums
FEATURE_SQUARES = 8


#embedding_bag's sum of each square's feature rows, in fp32 and in the same order, as one kernel: torch's took 62 us
#of a 1.6 ms batch of 64, torch.compile's gather and sum 71 us
@triton.jit
def feature_sum_kernel(rows_ptr, table_ptr, out_ptr, squares, D, FEATURES: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr):
    square = tl.program_id(0) * BLOCK_S + tl.arange(0, BLOCK_S)
    d = tl.arange(0, BLOCK_D)
    square_ok, d_ok = square < squares, d < D
    total = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)
    for feature in range(FEATURES):
        row = tl.load(rows_ptr + square * FEATURES + feature, mask=square_ok, other=0)
        total += tl.load(table_ptr + row[:, None] * D + d[None, :], mask=square_ok[:, None] & d_ok[None, :], other=0.0).to(tl.float32)
    tl.store(out_ptr + square[:, None] * D + d[None, :], total.to(out_ptr.dtype.element_ty), mask=square_ok[:, None] & d_ok[None, :])


@torch.library.custom_op("chess::feature_sum", mutates_args=())
def feature_sum(rows: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    rows, table = rows.contiguous(), table.contiguous()
    features, D = rows.shape[-1], table.shape[1]
    out = table.new_empty(*rows.shape[:-1], D)
    squares = rows.numel() // features
    feature_sum_kernel[(triton.cdiv(squares, FEATURE_SQUARES),)](rows, table, out, squares, D, FEATURES=features, BLOCK_S=FEATURE_SQUARES,
                                                                  BLOCK_D=triton.next_power_of_2(D))
    return out


@feature_sum.register_fake
def _(rows, table):
    return table.new_empty(*rows.shape[:-1], table.shape[1])


#model.SquareFeatures with its rows summed by feature_sum_kernel
class SummedSquareFeatures(nn.Module):
    def __init__(self, features: SquareFeatures):
        super().__init__()
        self.features = features

    def forward(self, board_state: torch.Tensor) -> torch.Tensor:
        return feature_sum(self.features.indices(board_state) + self.features.row_offsets, self.features.table)


#the model with its blocks swapped for FusedBlocks and its square features summed as above, for inference only;
#embeddings and heads unchanged
def fuse_blocks(model: nn.Module) -> nn.Module:
    model.encoder = nn.ModuleList(FusedBlock(block) for block in model.encoder)
    model.square_features = SummedSquareFeatures(model.square_features)
    return model

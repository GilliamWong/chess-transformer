#the engine's transformer blocks for inference on the gpu: the same arithmetic as model.TransformerBlock, arranged
#for speed (tests/test_inference.py checks they agree, so a change to the block must be made here too). Q, K, V and
#the template mix are one matmul, the mlp's gate and up another, and attention is a Triton kernel: it RMS-normalises
#the queries and keys (QK-norm), adds the template bias (made by a kernel of its own), takes the softmax and mixes the
#values without writing the scores or any transposed copy to memory, and reads each key/value head once for all the
#query heads of its group instead of a copy for each (repeat_interleave). The matmuls are a Triton kernel too, with
#the bias, the mlp's silu(gate) * up and the residual add in the same kernel
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn

from model.model import NUM_SQUARES, ChessTransformer, TransformerBlock

#warps an attention instance has for each 64 of its square rows, and for its metadata rows
SQUARE_WARPS, META_WARPS = 2, 2
#the most warps Triton gives an instance
MAX_WARPS = 32
#rows of template_bias_kernel's matmuls: (position, head) pairs
BIAS_PAIRS = 64
#positions whose template bias is made and read at a time: a buffer that stays in L2
BIAS_POSITIONS = 32


#bias[(b, h), i] = c[(b, h), i] @ templates[:, i] for each query square i, on tensor cores (in the attention it was slow)
@triton.jit
def template_bias_kernel(c_ptr, templates_ptr, bias_ptr, pairs_count, stride_b, stride_t,
                         HEADS: tl.constexpr, TEMPLATES: tl.constexpr, BLOCK_TEMPLATES: tl.constexpr, PAIRS: tl.constexpr):
    i, p = tl.program_id(0), tl.program_id(1)
    pairs = p * PAIRS + tl.arange(0, PAIRS)
    pair_ok = pairs < pairs_count
    b, h = pairs // HEADS, pairs % HEADS
    t = tl.arange(0, BLOCK_TEMPLATES)
    squares = tl.arange(0, 64)
    coefficients = tl.load(c_ptr + b[:, None] * stride_b + i * stride_t + h[:, None] * TEMPLATES + t[None, :],
                           mask=pair_ok[:, None] & (t < TEMPLATES)[None, :], other=0.0)
    templates = tl.load(templates_ptr + t[:, None] * 4096 + i * 64 + squares[None, :], mask=(t < TEMPLATES)[:, None], other=0.0)
    #ieee: fp32 inputs (the tests' fp32 model) would otherwise be rounded to tf32
    bias = tl.dot(coefficients, templates, input_precision="ieee")
    tl.store(bias_ptr + pairs[:, None] * 4096 + i * 64 + squares[None, :], bias.to(bias_ptr.dtype.element_ty), mask=pair_ok[:, None])


#the keys are the 64 squares (template bias) and the metadata tokens (none), 64 + 16 columns rather than 70 padded to 128
@triton.jit
def template_attention_kernel(q_ptr, k_ptr, v_ptr, bias_ptr, q_gain_ptr, k_gain_ptr, out_ptr,
                              stride_b, stride_t, out_stride_b, out_stride_t, eps, scale,
                              T: tl.constexpr, HEADS: tl.constexpr, HEADS_PER_KV: tl.constexpr, DQ: tl.constexpr, DV: tl.constexpr,
                              FIRST_ROW: tl.constexpr, ROWS_PER_HEAD: tl.constexpr, BLOCK_ROWS: tl.constexpr, BLOCK_META: tl.constexpr,
                              BLOCK_DQ: tl.constexpr):
    #all the query heads of key/value head g at once, so its keys and values are read and normalised once
    g, b = tl.program_id(0), tl.program_id(1)
    r = tl.arange(0, BLOCK_ROWS)
    head = g * HEADS_PER_KV + r // ROWS_PER_HEAD
    token = FIRST_ROW + r % ROWS_PER_HEAD
    squares = tl.arange(0, 64)
    meta = 64 + tl.arange(0, BLOCK_META)
    d = tl.arange(0, BLOCK_DQ)
    dv = tl.arange(0, DV)
    row_ok, meta_ok, d_ok = (r < HEADS_PER_KV * ROWS_PER_HEAD) & (token < T), meta < T, d < DQ

    q_gain = tl.load(q_gain_ptr + d, mask=d_ok, other=0.0).to(tl.float32)
    k_gain = tl.load(k_gain_ptr + d, mask=d_ok, other=0.0).to(tl.float32)
    #q, k and v are column slices of one projection, so they share its strides
    base = b * stride_b
    q = tl.load(q_ptr + base + token[:, None] * stride_t + head[:, None] * DQ + d[None, :], mask=row_ok[:, None] & d_ok[None, :], other=0.0).to(tl.float32)
    q = (q * tl.rsqrt(tl.sum(q * q, 1) / DQ + eps)[:, None] * q_gain[None, :]).to(tl.float16)
    k_sq = tl.load(k_ptr + base + squares[:, None] * stride_t + g * DQ + d[None, :], mask=d_ok[None, :], other=0.0).to(tl.float32)
    k_sq = k_sq * tl.rsqrt(tl.sum(k_sq * k_sq, 1) / DQ + eps)[:, None] * k_gain[None, :]
    k_meta = tl.load(k_ptr + base + meta[:, None] * stride_t + g * DQ + d[None, :], mask=meta_ok[:, None] & d_ok[None, :], other=0.0).to(tl.float32)
    k_meta = k_meta * tl.rsqrt(tl.sum(k_meta * k_meta, 1) / DQ + eps)[:, None] * k_gain[None, :]
    s_sq = tl.dot(q, tl.trans(k_sq.to(tl.float16))) * scale
    s_meta = tl.dot(q, tl.trans(k_meta.to(tl.float16))) * scale
    if FIRST_ROW == 0:
        #the group's heads are consecutive, so their bias is one block
        s_sq += tl.load(bias_ptr + (b * HEADS + g * HEADS_PER_KV) * 4096 + r[:, None] * 64 + squares[None, :], mask=row_ok[:, None], other=0.0).to(tl.float32)

    s_meta = tl.where(meta_ok[None, :], s_meta, float("-inf"))
    top = tl.maximum(tl.max(s_sq, 1), tl.max(s_meta, 1))
    p_sq = tl.exp(s_sq - top[:, None])
    p_meta = tl.exp(s_meta - top[:, None])
    total = tl.sum(p_sq, 1) + tl.sum(p_meta, 1)
    v_sq = tl.load(v_ptr + base + squares[:, None] * stride_t + g * DV + dv[None, :])
    v_meta = tl.load(v_ptr + base + meta[:, None] * stride_t + g * DV + dv[None, :], mask=meta_ok[:, None], other=0.0)
    out = tl.dot(p_sq.to(tl.float16), v_sq.to(tl.float16)) + tl.dot(p_meta.to(tl.float16), v_meta.to(tl.float16))
    out = out / total[:, None]
    tl.store(out_ptr + b * out_stride_b + token[:, None] * out_stride_t + head[:, None] * DV + dv[None, :], out.to(out_ptr.dtype.element_ty),
             mask=row_ok[:, None])


#an opaque op for torch.compile: compiling through the kernel call made it ~15x slower
@torch.library.custom_op("chess::template_attention", mutates_args=())
def template_attention(projected: torch.Tensor, templates: torch.Tensor, q_gain: torch.Tensor, k_gain: torch.Tensor, heads: int, kv_heads: int,
                       dq: int, dv: int, templates_count: int, eps: float) -> torch.Tensor:
    projected = projected.contiguous()
    B, T, _ = projected.shape
    q, k, v, c = projected.split([heads * dq, kv_heads * dq, kv_heads * dv, heads * templates_count], dim=-1)
    out = torch.empty(B, T, heads * dv, device=projected.device, dtype=projected.dtype)
    bias = torch.empty(min(B, BIAS_POSITIONS), heads, NUM_SQUARES, NUM_SQUARES, device=projected.device, dtype=projected.dtype)
    heads_per_kv, block_meta = heads // kv_heads, max(16, triton.next_power_of_2(T - NUM_SQUARES))
    square_rows = triton.next_power_of_2(heads_per_kv * NUM_SQUARES)
    square_warps = SQUARE_WARPS * square_rows // NUM_SQUARES
    if square_warps > MAX_WARPS:
        raise ValueError(f"template_attention takes at most {MAX_WARPS // SQUARE_WARPS} query heads a key/value head, not {heads_per_kv}")
    shapes = dict(T=T, HEADS=heads, HEADS_PER_KV=heads_per_kv, DQ=dq, DV=dv, BLOCK_META=block_meta, BLOCK_DQ=max(16, triton.next_power_of_2(dq)))
    scalars = (projected.stride(0), projected.stride(1), out.stride(0), out.stride(1), eps, 1 / math.sqrt(dq))
    for start in range(0, B, BIAS_POSITIONS):
        positions = slice(start, min(B, start + BIAS_POSITIONS))
        count = positions.stop - start
        template_bias_kernel[(NUM_SQUARES, triton.cdiv(count * heads, BIAS_PAIRS))](c[positions], templates, bias, count * heads, projected.stride(0),
                                                                                 projected.stride(1), HEADS=heads, TEMPLATES=templates_count,
                                                                                 BLOCK_TEMPLATES=max(16, triton.next_power_of_2(templates_count)),
                                                                                 PAIRS=BIAS_PAIRS)
        template_attention_kernel[(kv_heads, count)](q[positions], k[positions], v[positions], bias, q_gain, k_gain, out[positions], *scalars, FIRST_ROW=0,
                                                     ROWS_PER_HEAD=NUM_SQUARES, BLOCK_ROWS=square_rows,
                                                     num_warps=square_warps, **shapes)
    template_attention_kernel[(kv_heads, B)](q, k, v, bias, q_gain, k_gain, out, *scalars, FIRST_ROW=NUM_SQUARES, ROWS_PER_HEAD=block_meta,
                                             BLOCK_ROWS=triton.next_power_of_2(heads_per_kv * block_meta), num_warps=META_WARPS, **shapes)
    return out


@template_attention.register_fake
def _(projected, templates, q_gain, k_gain, heads, kv_heads, dq, dv, templates_count, eps):
    B, T, _ = projected.shape
    return projected.new_empty(B, T, heads * dv)


#a tile of the blocks' matmuls is LINEAR_ROWS (+ linear_extra_rows()) x LINEAR_COLUMNS, LINEAR_DEPTH deep (a gated
#matmul's is LINEAR_COLUMNS / 2 columns of the gate and the same of up), and GROUP_M row tiles go through the columns
#together so their rows stay in L2. The fastest on an RTX 4090 at 64 positions of 32 to 128 rows, 64 to 256 columns,
#2 to 4 stages and 4 or 8 warps; smaller tiles for smaller batches gained under 1% of a search
LINEAR_ROWS, LINEAR_COLUMNS, LINEAR_DEPTH, LINEAR_WARPS, LINEAR_STAGES, GROUP_M = 128, 128, 32, 4, 3, 8
#a tile's extra rows each cost about this many of its LINEAR_ROWS (they are a smaller matmul of their own)
EXTRA_ROW_COST = 1.5


#the rows (0, 16, 32 or 64) a tile takes after its LINEAR_ROWS: a matmul runs in waves of a tile per multiprocessor,
#and a few tiles more than a whole number of waves cost a whole wave. At 64 positions (4,480 rows) a 512 column
#matmul has 140 tiles of 128 rows for 128 multiprocessors, or 128 tiles of 144
def linear_extra_rows(M: int, column_tiles: int, multiprocessors: int) -> int:
    def cost(extra: int) -> float:
        waves = triton.cdiv(triton.cdiv(M, LINEAR_ROWS + extra) * column_tiles, multiprocessors)
        return waves * (LINEAR_ROWS + EXTRA_ROW_COST * extra)
    return min((0, 16, 32, 64), key=cost)


#a tile's bias, silu(gate) * up and residual, in fp32, and its store. MIX stores the next block's input instead: the
#stream as stored, mixed with the input embedding (original) as TransformerBlock mixes it
@triton.jit
def linear_epilogue(acc, up, rows, cols, M, N, bias_ptr, residual_ptr, original_ptr, residual_scale_ptr, input_scale_ptr, out_ptr,
                    HAS_BIAS: tl.constexpr, HAS_RESIDUAL: tl.constexpr, GATED: tl.constexpr, MIX: tl.constexpr):
    row_ok, col_ok = rows < M, cols < N
    out = acc.to(tl.float32)
    if HAS_BIAS:
        out += tl.load(bias_ptr + cols, mask=col_ok, other=0.0).to(tl.float32)[None, :]
    if GATED:
        out = out * tl.sigmoid(out) * up.to(tl.float32)
    offsets = rows[:, None] * N + cols[None, :]
    mask = row_ok[:, None] & col_ok[None, :]
    if HAS_RESIDUAL:
        out += tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    if MIX:
        stream = out.to(out_ptr.dtype.element_ty).to(tl.float32)
        original = tl.load(original_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        out = tl.load(residual_scale_ptr).to(tl.float32) * stream + tl.load(input_scale_ptr).to(tl.float32) * original
    tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


#out = x @ w (w is in_features x out_features), then in the same kernel the bias, silu(gate) * up (gated: w's first
#half of columns is the gate, the second up) and the residual added. ACC_TYPE is the accumulators' type: fp16 for the
#blocks, as cuBLAS did with allow_fp16_accumulation (twice fp32 accumulation's speed on a geforce card)
@triton.jit
def linear_kernel(x_ptr, w_ptr, bias_ptr, residual_ptr, out_ptr, M, N, K, stride_w,
                  HAS_BIAS: tl.constexpr, HAS_RESIDUAL: tl.constexpr, GATED: tl.constexpr, ACC_TYPE: tl.constexpr,
                  BLOCK_M: tl.constexpr, EXTRA_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    TILE_M: tl.constexpr = BLOCK_M + EXTRA_M
    grid_m, grid_n = tl.cdiv(M, TILE_M), tl.cdiv(N, BLOCK_N)
    in_group = GROUP_M * grid_n
    first_m = pid // in_group * GROUP_M
    group_rows = min(grid_m - first_m, GROUP_M)
    pid_m = first_m + pid % in_group % group_rows
    pid_n = pid % in_group // group_rows
    rows = pid_m * TILE_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    depth = tl.arange(0, BLOCK_K)
    col_ok = cols < N
    x_ptrs = x_ptr + rows[:, None] * K + depth[None, :]
    w_ptrs = w_ptr + depth[:, None] * stride_w + cols[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    up = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    #the tile's EXTRA_M rows after its BLOCK_M, a matmul of their own on the same weights (a row's sums are the same
    #bits in either)
    if EXTRA_M > 0:
        extra_rows = pid_m * TILE_M + BLOCK_M + tl.arange(0, EXTRA_M)
        extra_ptrs = x_ptr + extra_rows[:, None] * K + depth[None, :]
        extra_acc = tl.zeros((EXTRA_M, BLOCK_N), dtype=ACC_TYPE)
        extra_up = tl.zeros((EXTRA_M, BLOCK_N), dtype=ACC_TYPE)

    for _ in range(0, K, BLOCK_K):
        w = tl.load(w_ptrs, mask=col_ok[None, :], other=0.0)
        x = tl.load(x_ptrs, mask=(rows < M)[:, None], other=0.0)
        acc = tl.dot(x, w, acc, out_dtype=ACC_TYPE)
        if GATED:
            w_up = tl.load(w_ptrs + N, mask=col_ok[None, :], other=0.0)
            up = tl.dot(x, w_up, up, out_dtype=ACC_TYPE)
        if EXTRA_M > 0:
            x_extra = tl.load(extra_ptrs, mask=(extra_rows < M)[:, None], other=0.0)
            extra_acc = tl.dot(x_extra, w, extra_acc, out_dtype=ACC_TYPE)
            if GATED:
                extra_up = tl.dot(x_extra, w_up, extra_up, out_dtype=ACC_TYPE)
            extra_ptrs += BLOCK_K
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K * stride_w

    linear_epilogue(acc, up, rows, cols, M, N, bias_ptr, residual_ptr, out_ptr, out_ptr, out_ptr, out_ptr, HAS_BIAS, HAS_RESIDUAL, GATED, False)
    if EXTRA_M > 0:
        linear_epilogue(extra_acc, extra_up, extra_rows, cols, M, N, bias_ptr, residual_ptr, out_ptr, out_ptr, out_ptr, out_ptr, HAS_BIAS, HAS_RESIDUAL,
                        GATED, False)


#an opaque op for torch.compile, as template_attention. fp16 inputs accumulate in fp16 and give fp16, or with
#accumulate_fp32 accumulate in fp32 and give fp32; fp32 inputs accumulate in fp32 (tf32 products)
@torch.library.custom_op("chess::linear", mutates_args=())
def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None, residual: torch.Tensor | None = None,
           gated: bool = False, accumulate_fp32: bool = False) -> torch.Tensor:
    K, width = weight.shape
    N = width // 2 if gated else width
    if K % LINEAR_DEPTH != 0:
        raise ValueError(f"linear_kernel reads the {K} inputs {LINEAR_DEPTH} at a time")
    x = x.contiguous()
    out = x.new_empty(*x.shape[:-1], N, dtype=torch.float32 if accumulate_fp32 else x.dtype)
    acc_type = tl.float16 if x.dtype == torch.float16 and not accumulate_fp32 else tl.float32
    M = x.numel() // K
    columns = LINEAR_COLUMNS // 2 if gated else LINEAR_COLUMNS
    extra = linear_extra_rows(M, triton.cdiv(N, columns), torch.cuda.get_device_properties(x.device).multi_processor_count)
    #the pointers of unused inputs are never read
    bias_ = bias if bias is not None else weight
    residual_ = residual.contiguous() if residual is not None else out
    grid = (triton.cdiv(M, LINEAR_ROWS + extra) * triton.cdiv(N, columns),)
    linear_kernel[grid](x, weight, bias_, residual_, out, M, N, K, weight.stride(0), HAS_BIAS=bias is not None, HAS_RESIDUAL=residual is not None,
                        GATED=gated, ACC_TYPE=acc_type, BLOCK_M=LINEAR_ROWS, EXTRA_M=extra, BLOCK_N=columns, BLOCK_K=LINEAR_DEPTH, GROUP_M=GROUP_M,
                        num_warps=LINEAR_WARPS, num_stages=LINEAR_STAGES)
    return out


@linear.register_fake
def _(x, weight, bias=None, residual=None, gated=False, accumulate_fp32=False):
    return x.new_empty(*x.shape[:-1], weight.shape[1] // 2 if gated else weight.shape[1], dtype=torch.float32 if accumulate_fp32 else x.dtype)


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
        self.residual_scale, self.input_scale = block.residual_scale, block.input_scale
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

    def forward(self, x: torch.Tensor, templates: torch.Tensor, original_board_embed: torch.Tensor) -> torch.Tensor:
        x = self.residual_scale * x + self.input_scale * original_board_embed
        attended = template_attention(linear(self.ln1(x), self.projection, self.projection_bias), templates, self.q_gain, self.k_gain, self.heads,
                                      self.kv_heads, self.dq, self.dv, self.templates_count, self.eps)
        x = linear(attended, self.combine_heads, self.combine_heads_bias, residual=x)
        return linear(linear(self.ln2(x), self.gate_up, gated=True), self.down, residual=x)


#tokens and columns a program of input_embedding_kernel takes: whole rows in one program were slower
EMBEDDING_TOKENS, EMBEDDING_COLUMNS = 8, 256


#the input embedding in one kernel, so that every block reads one buffer: inductor recomputed or copied it for each
@triton.jit
def input_embedding_kernel(boards_ptr, rows_ptr, embed_ptr, position_ptr, table_ptr, out_ptr, tokens, D, T: tl.constexpr, SQUARES: tl.constexpr,
                           FEATURES: tl.constexpr, BLOCK_TOKENS: tl.constexpr, BLOCK_D: tl.constexpr):
    token = tl.program_id(0) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    board, place = token // T, token % T
    token_ok, d_ok = token < tokens, d < D
    square_ok = token_ok & (place < SQUARES)
    features = tl.zeros((BLOCK_TOKENS, BLOCK_D), dtype=tl.float32)
    for feature in range(FEATURES):
        row = tl.load(rows_ptr + (board * SQUARES + place) * FEATURES + feature, mask=square_ok, other=0)
        features += tl.load(table_ptr + row[:, None] * D + d[None, :], mask=square_ok[:, None] & d_ok[None, :], other=0.0).to(tl.float32)
    code = tl.load(boards_ptr + token, mask=token_ok, other=0)
    mask = token_ok[:, None] & d_ok[None, :]
    total = tl.load(embed_ptr + code[:, None] * D + d[None, :], mask=mask, other=0.0).to(tl.float32)
    total += tl.load(position_ptr + place[:, None] * D + d[None, :], mask=mask, other=0.0).to(tl.float32)
    #the features' sum in the table's precision, as SquareFeatures gives it
    total += features.to(out_ptr.dtype.element_ty).to(tl.float32)
    tl.store(out_ptr + token[:, None] * D + d[None, :], total.to(out_ptr.dtype.element_ty), mask=mask)


#boards B x T, rows B x 64 x features (each square's rows in the feature table): B x T x D
@torch.library.custom_op("chess::input_embedding", mutates_args=())
def input_embedding(boards: torch.Tensor, rows: torch.Tensor, embed: torch.Tensor, position: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    boards, rows = boards.contiguous(), rows.contiguous()
    B, T = boards.shape
    D = table.shape[1]
    out = table.new_empty(B, T, D)
    grid = (triton.cdiv(B * T, EMBEDDING_TOKENS), triton.cdiv(D, EMBEDDING_COLUMNS))
    input_embedding_kernel[grid](boards, rows, embed.contiguous(), position.contiguous(), table.contiguous(), out, B * T, D, T=T, SQUARES=NUM_SQUARES,
                                 FEATURES=rows.shape[-1], BLOCK_TOKENS=EMBEDDING_TOKENS, BLOCK_D=EMBEDDING_COLUMNS)
    return out


@input_embedding.register_fake
def _(boards, rows, embed, position, table):
    return table.new_empty(*boards.shape, table.shape[1])


#ChessTransformer's input embedding (embed, position_embed and the square features) by input_embedding_kernel
class InputEmbedding(nn.Module):
    def __init__(self, model: ChessTransformer):
        super().__init__()
        self.embed, self.position_embed, self.features = model.embed, model.position_embed, model.square_features

    def forward(self, board_state: torch.Tensor) -> torch.Tensor:
        rows = self.features.indices(board_state) + self.features.row_offsets
        return input_embedding(board_state, rows, self.embed.weight, self.position_embed, self.features.table)


#logits[i, j] = (W_f x_i + b_f) . (W_t x_j + b_t) / sqrt(d) = a_i . x_j + s_i: from and to as one d x d matmul (a)
def head_weights(model: ChessTransformer) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        from_weight, from_bias = model.from_projection.weight.double(), model.from_projection.bias.double()
        to_weight, to_bias = model.to_projection.weight.double(), model.to_projection.bias.double()
        promotion, down = model.promotion_projection, model.board_down_project[0]
        scale = 1 / math.sqrt(model.d_model)
        narrow_weight = torch.cat([(to_bias @ from_weight)[None] * scale, promotion.weight.double(), down.weight.double()])
        narrow_bias = torch.cat([(from_bias @ to_bias)[None] * scale, promotion.bias.double(), down.bias.double()])
        return ((from_weight.T @ to_weight * scale).half(), (to_weight.T @ from_bias * scale).float(), narrow_weight.float(),
                narrow_bias.float())


#ChessTransformer.forward with FusedBlocks and head_weights' matmuls, for inference: a change to it must be made here too
class FusedTransformer(nn.Module):
    def __init__(self, model: ChessTransformer):
        super().__init__()
        self.input_embedding = InputEmbedding(model)
        self.templates, self.ln, self.loops = model.templates, model.ln, model.loops
        self.encoder = nn.ModuleList(FusedBlock(block) for block in model.encoder)
        self.board_down_activation, self.eval_head = model.board_down_project[1], model.eval_head
        for name in ("from_squares", "to_squares", "promotion_move_ids", "promotion_to_squares", "promotion_pieces"):
            self.register_buffer(name, getattr(model, name))
        self.a_weight, self.a_bias, self.narrow_weight, self.narrow_bias = (nn.Parameter(tensor, requires_grad=False) for tensor in head_weights(model))
        promotions = model.promotion_projection.out_features
        self.promotion_columns, self.down_columns = slice(1, 1 + promotions), slice(1 + promotions, None)

    def forward(self, board_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embed_state = self.input_embedding(board_state)
        original_board_embed = embed_state
        for _ in range(self.loops):
            for block in self.encoder:
                embed_state = block(embed_state, self.templates, original_board_embed)
        model_output = self.ln(embed_state)

        with torch.autocast(device_type=board_state.device.type, enabled=False):
            #on every token: the metadata tokens' a is not needed, but slicing out the squares' would copy them
            a = linear(model_output.half(), self.a_weight, self.a_bias, accumulate_fp32=True)
            model_output = model_output.float()
            narrow = F.linear(model_output, self.narrow_weight, self.narrow_bias) #B T: s, the promotion logits, board_down_project's
            squares = narrow[:, :NUM_SQUARES]
            logits = a[:, :NUM_SQUARES] @ model_output[:, :NUM_SQUARES].transpose(1, 2) + squares[..., :1]
            combined_logits = logits[:, self.from_squares, self.to_squares]
            combined_logits[:, self.promotion_move_ids] += squares[..., self.promotion_columns][:, self.promotion_to_squares, self.promotion_pieces]
            compressed_board = self.board_down_activation(narrow[..., self.down_columns]).reshape(board_state.shape[0], -1)
            return combined_logits, self.eval_head(compressed_board)

#the engine's transformer blocks for inference on the gpu: the same arithmetic as model.TransformerBlock, arranged
#for speed (tests/test_inference.py checks they agree, so a change to the block must be made here too). Q, K, V and
#the template mix are one matmul, the mlp's gate and up another, and attention is a Triton kernel: it RMS-normalises
#the queries and keys (QK-norm), adds the template bias (made by a kernel of its own), takes the softmax and mixes the
#values without writing the scores or any transposed copy to memory, and reads each key/value head once for all the
#query heads of its group instead of a copy for each (repeat_interleave). The matmuls are a Triton kernel too, with
#the bias, the mlp's silu(gate) * up, the residual add and the next block's input mix in the same kernel
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn
from triton.language.extra import libdevice
from triton.language.extra.cuda import gdc_wait

from model.encoding import TOKENS_PER_BOARD
from model.model import EDGE, KING, KNIGHT, NO_PIECE, NUM_SQUARES, PAWN, QUEEN, THEIRS, ChessTransformer, TransformerBlock, value_bin_edges

#warps an attention instance has for each 64 of its square rows, and for its metadata rows
SQUARE_WARPS, META_WARPS = 2, 2
#the most warps Triton gives an instance
MAX_WARPS = 32
#rows of template_bias_kernel's matmuls: (position, head) pairs
BIAS_PAIRS = 64
#positions whose template bias is made and read at a time: a buffer that stays in L2
BIAS_POSITIONS = 32
#in a cuda graph, half as many into two buffers: the next chunk's bias is made on a second stream as one is attended
GRAPHED_BIAS_POSITIONS = 16


#programmatic dependent launch (sm_90 and newer): a kernel starts as the blocks of the one before it finish and waits
#for that one before touching memory. No early trigger: kernels launched early slowed small batches
@triton.jit
def dependent_launch_wait(PDL: tl.constexpr):
    if PDL:
        gdc_wait()


#the launch keywords of a kernel that starts with dependent_launch_wait: griddepcontrol is sm_90 ptx
def dependent_launch(device: torch.device) -> dict:
    supported = torch.cuda.get_device_capability(device) >= (9, 0)
    return {"PDL": supported, "launch_pdl": supported}


#bias[(b, h), i] = c[(b, h), i] @ templates[:, i] for each query square i, on tensor cores (in the attention it was slow)
@triton.jit
def template_bias_kernel(c_ptr, templates_ptr, bias_ptr, pairs_count, stride_b, stride_t,
                         HEADS: tl.constexpr, TEMPLATES: tl.constexpr, BLOCK_TEMPLATES: tl.constexpr, PAIRS: tl.constexpr,
                         PDL: tl.constexpr):
    dependent_launch_wait(PDL)
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
                              BLOCK_DQ: tl.constexpr, PDL: tl.constexpr):
    dependent_launch_wait(PDL)
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


#template_attention's second stream for each stream a cuda graph is captured on
SIDE_STREAMS: dict[torch.cuda.Stream, torch.cuda.Stream] = {}


#an opaque op for torch.compile: compiling through the kernel call made it ~15x slower
@torch.library.custom_op("chess::template_attention", mutates_args=())
def template_attention(projected: torch.Tensor, templates: torch.Tensor, q_gain: torch.Tensor, k_gain: torch.Tensor, heads: int, kv_heads: int,
                       dq: int, dv: int, templates_count: int, eps: float) -> torch.Tensor:
    projected = projected.contiguous()
    B, T, _ = projected.shape
    q, k, v, c = projected.split([heads * dq, kv_heads * dq, kv_heads * dv, heads * templates_count], dim=-1)
    out = torch.empty(B, T, heads * dv, device=projected.device, dtype=projected.dtype)
    heads_per_kv, block_meta = heads // kv_heads, max(16, triton.next_power_of_2(T - NUM_SQUARES))
    square_rows = triton.next_power_of_2(heads_per_kv * NUM_SQUARES)
    square_warps = SQUARE_WARPS * square_rows // NUM_SQUARES
    if square_warps > MAX_WARPS:
        raise ValueError(f"template_attention takes at most {MAX_WARPS // SQUARE_WARPS} query heads a key/value head, not {heads_per_kv}")
    shapes = dict(T=T, HEADS=heads, HEADS_PER_KV=heads_per_kv, DQ=dq, DV=dv, BLOCK_META=block_meta, BLOCK_DQ=max(16, triton.next_power_of_2(dq)))
    launch = dependent_launch(projected.device)
    scalars = (projected.stride(0), projected.stride(1), out.stride(0), out.stride(1), eps, 1 / math.sqrt(dq))

    def make_bias(positions: slice, bias: torch.Tensor) -> None:
        count = positions.stop - positions.start
        template_bias_kernel[(NUM_SQUARES, triton.cdiv(count * heads, BIAS_PAIRS))](c[positions], templates, bias, count * heads, projected.stride(0),
                                                                                 projected.stride(1), HEADS=heads, TEMPLATES=templates_count,
                                                                                 BLOCK_TEMPLATES=max(16, triton.next_power_of_2(templates_count)),
                                                                                 PAIRS=BIAS_PAIRS, **launch)

    def attend_squares(positions: slice, bias: torch.Tensor) -> None:
        template_attention_kernel[(kv_heads, positions.stop - positions.start)](q[positions], k[positions], v[positions], bias, q_gain, k_gain,
                                                                                out[positions], *scalars, FIRST_ROW=0, ROWS_PER_HEAD=NUM_SQUARES,
                                                                                BLOCK_ROWS=square_rows, num_warps=square_warps, **shapes, **launch)

    #the metadata rows read no bias
    def attend_metadata(bias: torch.Tensor) -> None:
        template_attention_kernel[(kv_heads, B)](q, k, v, bias, q_gain, k_gain, out, *scalars, FIRST_ROW=NUM_SQUARES, ROWS_PER_HEAD=block_meta,
                                                 BLOCK_ROWS=triton.next_power_of_2(heads_per_kv * block_meta), num_warps=META_WARPS, **shapes, **launch)

    #uncaptured, a second stream's waits and twice the launches halved the speed of callers without cuda graphs
    if not torch.cuda.is_current_stream_capturing():
        bias = torch.empty(min(B, BIAS_POSITIONS), heads, NUM_SQUARES, NUM_SQUARES, device=projected.device, dtype=projected.dtype)
        for start in range(0, B, BIAS_POSITIONS):
            positions = slice(start, min(B, start + BIAS_POSITIONS))
            make_bias(positions, bias)
            attend_squares(positions, bias)
        attend_metadata(bias)
        return out

    chunks = [slice(start, min(B, start + GRAPHED_BIAS_POSITIONS)) for start in range(0, B, GRAPHED_BIAS_POSITIONS)]
    biases = [torch.empty(min(B, GRAPHED_BIAS_POSITIONS), heads, NUM_SQUARES, NUM_SQUARES, device=projected.device, dtype=projected.dtype)
              for _ in range(min(2, len(chunks)))]
    main = torch.cuda.current_stream(projected.device)
    if main not in SIDE_STREAMS:
        SIDE_STREAMS[main] = torch.cuda.Stream(projected.device)
    side = SIDE_STREAMS[main]
    side.wait_stream(main)
    make_bias(chunks[0], biases[0])
    for index, positions in enumerate(chunks):
        if index + 1 < len(chunks):
            with torch.cuda.stream(side):
                #the buffer chunk index - 1's square rows read
                if index > 0:
                    side.wait_event(read)
                make_bias(chunks[index + 1], biases[(index + 1) % 2])
            made = side.record_event()
        attend_squares(positions, biases[index % 2])
        if index + 2 < len(chunks):
            read = main.record_event()
        if index + 1 < len(chunks):
            main.wait_event(made)
    with torch.cuda.stream(side):
        attend_metadata(biases[0])
    main.wait_stream(side)
    return out


@template_attention.register_fake
def _(projected, templates, q_gain, k_gain, heads, kv_heads, dq, dv, templates_count, eps):
    B, T, _ = projected.shape
    return projected.new_empty(B, T, heads * dv)


#a tile of the blocks' matmuls is its rows (+ linear_extra_rows()) x its columns, LINEAR_DEPTH deep (a gated matmul's
#is columns / 2 columns of the gate and the same of up), and GROUP_M row tiles go through the columns together so their
#rows stay in L2. LINEAR_ROWS x LINEAR_COLUMNS with LINEAR_WARPS and LINEAR_STAGES was the fastest on an RTX 4090 at 64
#positions of 32 to 128 rows, 64 to 256 columns, 2 to 4 stages and 4 or 8 warps
LINEAR_ROWS, LINEAR_COLUMNS, LINEAR_DEPTH, LINEAR_WARPS, LINEAR_STAGES, GROUP_M = 128, 128, 32, 4, 3, 8
#linear's tiles (rows, columns, warps, stages) by compute capability, as tile_for lists; a gpu not timed takes them all
#at LINEAR_ROWS x LINEAR_COLUMNS. sm_120's were timed as int8_blocks.INT8_TILES' (on the int8 network's fp16 matmuls).
#Any tile gives the same bits: it changes neither the depth's order nor the instructions
UNTIMED_LINEAR_TILES = [(math.inf, (LINEAR_ROWS, LINEAR_COLUMNS, LINEAR_WARPS, LINEAR_STAGES))]
LINEAR_TILES = {(12, 0): [(2, (64, 32, 4, 5)), (8, (64, 64, 4, 4)), (12, (128, 64, 4, 3)), (16, (64, 128, 4, 3))] + UNTIMED_LINEAR_TILES}
#a tile's extra rows each cost about this many of its rows (they are a smaller matmul of their own)
EXTRA_ROW_COST = 1.5


#a tile from tiles, [(the most rows a multiprocessor's share may be, tile), ...] with the last for any: a matmul runs
#in waves of tiles over the multiprocessors, so a tile for a share times out the same on a card with more of them. On
#a 70 multiprocessor card (RTX 5070 Ti) a board's 70 rows are one per multiprocessor
def tile_for(tiles: list[tuple[float, tuple[int, ...]]], M: int, multiprocessors: int) -> tuple[int, ...]:
    share = M / multiprocessors
    return next(tile for most, tile in tiles if share <= most)


#linear's tile for a matmul of M rows on the gpu
def linear_tile(device: torch.device, M: int) -> tuple[int, ...]:
    tiles = LINEAR_TILES.get(torch.cuda.get_device_capability(device), UNTIMED_LINEAR_TILES)
    return tile_for(tiles, M, torch.cuda.get_device_properties(device).multi_processor_count)


#the rows (0, 16, 32 or 64) a tile of rows takes after them: a matmul runs in waves of a tile per multiprocessor, and
#a few tiles more than a whole number of waves cost a whole wave. At 64 positions (4,480 rows) a 512 column matmul has
#140 tiles of 128 rows for 128 multiprocessors, or 128 tiles of 144
def linear_extra_rows(M: int, rows: int, column_tiles: int, multiprocessors: int) -> int:
    def cost(extra: int) -> float:
        waves = triton.cdiv(triton.cdiv(M, rows + extra) * column_tiles, multiprocessors)
        return waves * (rows + EXTRA_ROW_COST * extra)
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
def linear_kernel(x_ptr, w_ptr, bias_ptr, residual_ptr, original_ptr, residual_scale_ptr, input_scale_ptr, out_ptr, M, N, K, stride_w,
                  HAS_BIAS: tl.constexpr, HAS_RESIDUAL: tl.constexpr, GATED: tl.constexpr, MIX: tl.constexpr, ACC_TYPE: tl.constexpr,
                  BLOCK_M: tl.constexpr, EXTRA_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
                  PDL: tl.constexpr):
    dependent_launch_wait(PDL)
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

    linear_epilogue(acc, up, rows, cols, M, N, bias_ptr, residual_ptr, original_ptr, residual_scale_ptr, input_scale_ptr, out_ptr, HAS_BIAS,
                    HAS_RESIDUAL, GATED, MIX)
    if EXTRA_M > 0:
        linear_epilogue(extra_acc, extra_up, extra_rows, cols, M, N, bias_ptr, residual_ptr, original_ptr, residual_scale_ptr, input_scale_ptr, out_ptr,
                        HAS_BIAS, HAS_RESIDUAL, GATED, MIX)


#linear with a tile of LINEAR_TILES'
def launch_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, residual: torch.Tensor | None, gated: bool,
                  accumulate_fp32: bool, original: torch.Tensor | None, residual_scale: torch.Tensor | None, input_scale: torch.Tensor | None,
                  tile: tuple[int, ...]) -> torch.Tensor:
    rows, columns, warps, stages = tile
    K, width = weight.shape
    N = width // 2 if gated else width
    if K % LINEAR_DEPTH != 0:
        raise ValueError(f"linear_kernel reads the {K} inputs {LINEAR_DEPTH} at a time")
    x = x.contiguous()
    out = x.new_empty(*x.shape[:-1], N, dtype=torch.float32 if accumulate_fp32 else x.dtype)
    acc_type = tl.float16 if x.dtype == torch.float16 and not accumulate_fp32 else tl.float32
    M = x.numel() // K
    columns = columns // 2 if gated else columns
    extra = linear_extra_rows(M, rows, triton.cdiv(N, columns), torch.cuda.get_device_properties(x.device).multi_processor_count)
    #the pointers of unused inputs are never read
    bias_ = bias if bias is not None else weight
    residual_ = residual.contiguous() if residual is not None else out
    mix = original is not None
    mix_ = (original.contiguous(), residual_scale, input_scale) if mix else (out, out, out)
    grid = (triton.cdiv(M, rows + extra) * triton.cdiv(N, columns),)
    linear_kernel[grid](x, weight, bias_, residual_, *mix_, out, M, N, K, weight.stride(0), HAS_BIAS=bias is not None, HAS_RESIDUAL=residual is not None,
                        GATED=gated, MIX=mix, ACC_TYPE=acc_type, BLOCK_M=rows, EXTRA_M=extra, BLOCK_N=columns, BLOCK_K=LINEAR_DEPTH, GROUP_M=GROUP_M,
                        num_warps=warps, num_stages=stages, **dependent_launch(x.device))
    return out


#an opaque op for torch.compile, as template_attention. fp16 inputs accumulate in fp16 and give fp16, or with
#accumulate_fp32 accumulate in fp32 and give fp32; fp32 inputs accumulate in fp32 (tf32 products). With original (the
#input embedding) and the next block's residual_scale and input_scale it gives the next block's input (linear_epilogue)
@torch.library.custom_op("chess::linear", mutates_args=())
def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None, residual: torch.Tensor | None = None,
           gated: bool = False, accumulate_fp32: bool = False, original: torch.Tensor | None = None, residual_scale: torch.Tensor | None = None,
           input_scale: torch.Tensor | None = None) -> torch.Tensor:
    tile = linear_tile(x.device, x.numel() // weight.shape[0])
    return launch_linear(x, weight, bias, residual, gated, accumulate_fp32, original, residual_scale, input_scale, tile)


@linear.register_fake
def _(x, weight, bias=None, residual=None, gated=False, accumulate_fp32=False, original=None, residual_scale=None, input_scale=None):
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
        #set by link_input_mixes
        self.next_scales: tuple[nn.Parameter, nn.Parameter] | None = None
        self.input_mixed = False

    def forward(self, x: torch.Tensor, templates: torch.Tensor, original_board_embed: torch.Tensor) -> torch.Tensor:
        if not self.input_mixed:
            x = self.residual_scale * x + self.input_scale * original_board_embed
        attended = template_attention(linear(self.ln1(x), self.projection, self.projection_bias), templates, self.q_gain, self.k_gain, self.heads,
                                      self.kv_heads, self.dq, self.dv, self.templates_count, self.eps)
        x = linear(attended, self.combine_heads, self.combine_heads_bias, residual=x)
        hidden = linear(self.ln2(x), self.gate_up, gated=True)
        if self.next_scales is None:
            return linear(hidden, self.down, residual=x)
        return linear(hidden, self.down, residual=x, original=original_board_embed, residual_scale=self.next_scales[0], input_scale=self.next_scales[1])


#each block's down mixes the input of the block after it (next_scales, input_mixed) when both are of one kind: an fp16
#block after an int8 one (the last, or one kept in fp16 for its int8 error) mixes its own input, as an int8 block after
#an fp16 one does
def link_input_mixes(blocks: list[nn.Module]) -> None:
    for block in blocks:
        block.next_scales, block.input_mixed = None, False
    for block, following in zip(blocks, blocks[1:]):
        if type(block) is type(following):
            block.next_scales, following.input_mixed = (following.residual_scale, following.input_scale), True


#tokens and columns a program of input_embedding_kernel takes: whole rows in one program were slower
EMBEDDING_TOKENS, EMBEDDING_COLUMNS = 8, 256


#the input embedding in one kernel, so that every block reads one buffer: inductor recomputed or copied it for each
@triton.jit
def input_embedding_kernel(boards_ptr, rows_ptr, embed_ptr, position_ptr, table_ptr, out_ptr, tokens, D, T: tl.constexpr, SQUARES: tl.constexpr,
                           FEATURES: tl.constexpr, BLOCK_TOKENS: tl.constexpr, BLOCK_D: tl.constexpr, PDL: tl.constexpr):
    dependent_launch_wait(PDL)
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
                                 FEATURES=rows.shape[-1], BLOCK_TOKENS=EMBEDDING_TOKENS, BLOCK_D=EMBEDDING_COLUMNS, **dependent_launch(boards.device))
    return out


@input_embedding.register_fake
def _(boards, rows, embed, position, table):
    return table.new_empty(*boards.shape, table.shape[1])


#the gpus the board embedding (board_embedding_kernel) and the fused heads (heads) were timed on, an RTX 5070 Ti and a
#5090 (sm_120, 70 and 170 multiprocessors): other gpus keep the kernels before them until timed
FUSED_ENDS_CAPABILITIES = {(12, 0)}


def fused_ends(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device) in FUSED_ENDS_CAPABILITIES


#tokens and columns a program of board_embedding_kernel takes, by the batch's tokens: more programs for one or two boards
#(the fastest of 2-16 tokens and 128-1024 columns on an RTX 5070 Ti and 5090); whole rows in one program were slower
def embedding_block(tokens: int) -> tuple[int, int]:
    return (4, 128) if tokens < 4 * TOKENS_PER_BOARD else (8, 256)


#where the square is from the first square holding the piece CODE (a second queen is rare), or NO_PIECE
@triton.jit
def piece_offset_row(codes, squares, place, CODE: tl.constexpr, NO_PIECE: tl.constexpr):
    piece = tl.min(tl.where(codes == CODE, squares[None, :], 64), 1)
    return tl.where(piece < 64, (place // 8 - piece // 8 + 7) * 15 + place % 8 - piece % 8 + 7, NO_PIECE)


#a direction's column of tokens x 8
@triton.jit
def direction_column(values, directions, DIRECTION: tl.constexpr):
    return tl.sum(tl.where(directions[None, :] == DIRECTION, values, 0), 1)


#the code where found first holds along each direction (0 if it never does) and how far out (model.first_along)
@triton.jit
def first_along(along, found, steps):
    distance = tl.min(tl.where(found, steps[None, None, :], 8), 2)
    return tl.sum(tl.where(steps[None, None, :] == distance[:, :, None], along, 0), 2), distance


#feature FEATURE's table row for each token
@triton.jit
def feature_row(table_ptr, offsets_ptr, row, FEATURE: tl.constexpr, d, D, mask):
    return tl.load(table_ptr + (row + tl.load(offsets_ptr + FEATURE))[:, None] * D + d[None, :], mask=mask, other=0.0).to(tl.float32)


#input_embedding_kernel finding each square's 23 feature rows itself, as SquareFeatures.indices does (a change to it must
#be made here too), from rays (each square's squares out along the 8 directions, nearest first, past the edge 64) and
#jumps (its knight jumps)
@triton.jit
def board_embedding_kernel(boards_ptr, rays_ptr, jumps_ptr, offsets_ptr, embed_ptr, position_ptr, table_ptr, out_ptr, tokens, D, T: tl.constexpr,
                           PAWN: tl.constexpr, KNIGHT: tl.constexpr, QUEEN: tl.constexpr, KING: tl.constexpr, THEIRS: tl.constexpr,
                           EDGE: tl.constexpr, NO_PIECE: tl.constexpr, BLOCK_TOKENS: tl.constexpr, BLOCK_D: tl.constexpr, PDL: tl.constexpr):
    dependent_launch_wait(PDL)
    token = tl.program_id(0) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    board, place = token // T, token % T
    token_ok, d_ok = token < tokens, d < D
    square_ok = token_ok & (place < 64)
    board_ptr = boards_ptr + board * T
    squares, directions, steps = tl.arange(0, 64), tl.arange(0, 8), tl.arange(0, 8)
    codes = tl.load(board_ptr[:, None] + squares[None, :], mask=square_ok[:, None], other=0)
    #the codes out along each direction, EDGE past the edge (and on the 8th step, which pads the 7)
    ray = tl.load(rays_ptr + (place[:, None, None] * 8 + directions[None, :, None]) * 8 + steps[None, None, :], mask=square_ok[:, None, None], other=64)
    along = tl.load(board_ptr[:, None, None] + ray, mask=ray < 64, other=EDGE)
    first, distance = first_along(along, (along != 0) & (along != EDGE), steps)
    lines = tl.where(first > 0, 1 + (first - 1) * 7 + distance, 0)
    neighbours = tl.sum(tl.where(steps[None, None, :] == 0, along, 0), 2)
    jump = tl.load(jumps_ptr + place[:, None] * 8 + directions[None, :], mask=square_ok[:, None], other=64)
    jumped = tl.load(board_ptr[:, None] + jump, mask=jump < 64, other=EDGE)
    knights = tl.minimum(tl.sum((jumped == KNIGHT).to(tl.int32), 1), 2) * 3 + tl.minimum(tl.sum((jumped == KNIGHT + THEIRS).to(tl.int32), 1), 2)
    #the nearest pawn up and down the file (directions 0 and 1)
    pawn, pawn_distance = first_along(along, (along == PAWN) | (along == PAWN + THEIRS), steps)
    pawns = tl.where(pawn > 0, 1 + (pawn == PAWN + THEIRS).to(tl.int32) * 7 + pawn_distance, 0)

    #the rows summed in SquareFeatures.indices' order
    mask = square_ok[:, None] & d_ok[None, :]
    features = tl.zeros((BLOCK_TOKENS, BLOCK_D), dtype=tl.float32)
    features += feature_row(table_ptr, offsets_ptr, piece_offset_row(codes, squares, place, KING, NO_PIECE), 0, d, D, mask)
    features += feature_row(table_ptr, offsets_ptr, piece_offset_row(codes, squares, place, KING + THEIRS, NO_PIECE), 1, d, D, mask)
    features += feature_row(table_ptr, offsets_ptr, piece_offset_row(codes, squares, place, QUEEN, NO_PIECE), 2, d, D, mask)
    features += feature_row(table_ptr, offsets_ptr, piece_offset_row(codes, squares, place, QUEEN + THEIRS, NO_PIECE), 3, d, D, mask)
    for direction in tl.static_range(8):
        features += feature_row(table_ptr, offsets_ptr, direction_column(neighbours, directions, direction), 4 + direction, d, D, mask)
    for direction in tl.static_range(8):
        features += feature_row(table_ptr, offsets_ptr, direction_column(lines, directions, direction), 12 + direction, d, D, mask)
    features += feature_row(table_ptr, offsets_ptr, knights, 20, d, D, mask)
    features += feature_row(table_ptr, offsets_ptr, direction_column(pawns, directions, 0), 21, d, D, mask)
    features += feature_row(table_ptr, offsets_ptr, direction_column(pawns, directions, 1), 22, d, D, mask)

    code = tl.load(boards_ptr + token, mask=token_ok, other=0)
    mask = token_ok[:, None] & d_ok[None, :]
    total = tl.load(embed_ptr + code[:, None] * D + d[None, :], mask=mask, other=0.0).to(tl.float32)
    total += tl.load(position_ptr + place[:, None] * D + d[None, :], mask=mask, other=0.0).to(tl.float32)
    #the features' sum in the table's precision, as SquareFeatures gives it
    total += features.to(out_ptr.dtype.element_ty).to(tl.float32)
    tl.store(out_ptr + token[:, None] * D + d[None, :], total.to(out_ptr.dtype.element_ty), mask=mask)


#boards B x T: B x T x D. rays 64 x 8 x 8 and jumps 64 x 8 (board_embedding_kernel), offsets the features' first rows
@torch.library.custom_op("chess::board_embedding", mutates_args=())
def board_embedding(boards: torch.Tensor, rays: torch.Tensor, jumps: torch.Tensor, offsets: torch.Tensor, embed: torch.Tensor, position: torch.Tensor,
                    table: torch.Tensor) -> torch.Tensor:
    boards = boards.contiguous()
    B, T = boards.shape
    D = table.shape[1]
    out = table.new_empty(B, T, D)
    block_tokens, block_d = embedding_block(B * T)
    grid = (triton.cdiv(B * T, block_tokens), triton.cdiv(D, block_d))
    board_embedding_kernel[grid](boards, rays, jumps, offsets, embed.contiguous(), position.contiguous(), table.contiguous(), out, B * T, D, T=T, PAWN=PAWN,
                                 KNIGHT=KNIGHT, QUEEN=QUEEN, KING=KING, THEIRS=THEIRS, EDGE=EDGE, NO_PIECE=NO_PIECE, BLOCK_TOKENS=block_tokens,
                                 BLOCK_D=block_d, **dependent_launch(boards.device))
    return out


@board_embedding.register_fake
def _(boards, rays, jumps, offsets, embed, position, table):
    return table.new_empty(*boards.shape, table.shape[1])


#ChessTransformer's input embedding (embed, position_embed and the square features) by board_embedding_kernel, or on gpus
#without fused_ends by input_embedding_kernel on SquareFeatures.indices' rows
class InputEmbedding(nn.Module):
    def __init__(self, model: ChessTransformer):
        super().__init__()
        self.embed, self.position_embed, self.features = model.embed, model.position_embed, model.square_features
        self.from_boards = fused_ends(model.embed.weight.device)
        if self.from_boards:
            self.register_buffer("rays", F.pad(self.features.rays, (0, 1), value=NUM_SQUARES).int().contiguous())
            self.register_buffer("jumps", self.features.knight_jumps.int().contiguous())

    def forward(self, board_state: torch.Tensor) -> torch.Tensor:
        if self.from_boards:
            return board_embedding(board_state, self.rays, self.jumps, self.features.row_offsets, self.embed.weight, self.position_embed,
                                   self.features.table)
        rows = self.features.indices(board_state) + self.features.row_offsets
        return input_embedding(board_state, rows, self.embed.weight, self.position_embed, self.features.table)


#the heads after the last block in three kernels, from its stream to the engine's answer rows (a board's move logits,
#then its expected score). With x a row of the stream, r its 1 / rms and g the final norm's gain, the normed row is
#y = r g x: the heads fold g into their weights and r into their epilogues, and read the stream as it is
#(fused_head_weights). Their fp32 matmuls are fp16 tensor core matmuls of numbers split in two (split), many times
#faster than Triton's fp32 dot

#the narrow matmul's columns: board_down_project's, then a square's terms (s and t, a move's terms of its from and to
#square, and the 4 promotion logits), padded
NARROW_COLUMNS = 32
DOWN_COLUMNS = 16
#where the promotions' logits are among a square's terms, and where r is
PROMOTION_TERM, R_TERM = 2, 6
#per square, its terms and r, padded
SQUARE_TERMS = 8
#a number is split into its fp16 rounding and the rest times LO_SCALE, also fp16, which keeps the rest out of fp16's
#subnormals: their products are exact in fp32, and the products of two rests are below fp32's rounding
LO_SCALE = 2048.0
#the metadata rows the narrow matmul takes after the squares, at least TOKENS_PER_BOARD - 64
META_ROWS = 16
#the stream's columns head_matmul_kernel reads at a time
HEAD_DEPTH = 64
#moves a program of head_answer_kernel writes
MOVE_BLOCK = 256
#the eval head's hidden units a program of head_answer_kernel takes
VALUE_UNITS = 16


#nn.GELU's erf form
@triton.jit
def gelu(x):
    return 0.5 * x * (1 + libdevice.erf(x * 0.7071067811865476))


#an fp32 tensor as fp16 hi + lo / LO_SCALE
@triton.jit
def split(x, LO_SCALE: tl.constexpr):
    hi = x.to(tl.float16)
    return hi, ((x - hi.to(tl.float32)) * LO_SCALE).to(tl.float16)


#A_COLUMNS of a = r x A on a board's squares (r from the squares' sums of squares, as they are read), and their part of
#a . x for every pair of squares
@triton.jit
def head_a_tile(board_ptr, a_ptr, partials_ptr, tile, D, eps, A_COLUMNS: tl.constexpr, DEPTH: tl.constexpr, LO_SCALE: tl.constexpr):
    squares = tl.arange(0, 64)
    depth = tl.arange(0, DEPTH)
    columns = tile * A_COLUMNS + tl.arange(0, A_COLUMNS)
    x_ptrs = board_ptr + squares[:, None] * D + depth[None, :]
    a_ptrs = a_ptr + depth[:, None] * D + columns[None, :]
    acc = tl.zeros((64, A_COLUMNS), dtype=tl.float32)
    #summed across the rows once, after the loop
    squared = tl.zeros((64, DEPTH), dtype=tl.float32)
    for _ in range(0, D, DEPTH):
        x = tl.load(x_ptrs).to(tl.float16)
        acc = tl.dot(x, tl.load(a_ptrs), acc)
        squared += x.to(tl.float32) * x.to(tl.float32)
        x_ptrs += DEPTH
        a_ptrs += DEPTH * D
    hi, lo = split(acc * tl.rsqrt(tl.sum(squared, 1) / D + eps)[:, None], LO_SCALE)
    x = tl.trans(tl.load(board_ptr + squares[:, None] * D + columns[None, :]).to(tl.float16))
    tl.store(partials_ptr + squares[:, None] * 64 + squares[None, :], tl.dot(hi, x) + tl.dot(lo, x) * (1 / LO_SCALE))


#ROWS rows of a board's stream from FIRST_ROW (those before T) times the narrow weights' COLUMNS from FIRST, and their r
@triton.jit
def narrow_rows(board_ptr, narrow_hi_ptr, narrow_lo_ptr, D, eps, T: tl.constexpr, FIRST_ROW: tl.constexpr, ROWS: tl.constexpr, FIRST: tl.constexpr,
                COLUMNS: tl.constexpr, NARROW: tl.constexpr, DEPTH: tl.constexpr, LO_SCALE: tl.constexpr):
    rows, depth = FIRST_ROW + tl.arange(0, ROWS), tl.arange(0, DEPTH)
    row_ptrs = board_ptr + rows[:, None] * D + depth[None, :]
    weight_offsets = depth[:, None] * NARROW + FIRST + tl.arange(0, COLUMNS)[None, :]
    hi_acc, lo_acc = tl.zeros((ROWS, COLUMNS), dtype=tl.float32), tl.zeros((ROWS, COLUMNS), dtype=tl.float32)
    squared = tl.zeros((ROWS, DEPTH), dtype=tl.float32)
    for _ in range(0, D, DEPTH):
        x = tl.load(row_ptrs, mask=(rows < T)[:, None], other=0.0).to(tl.float16)
        hi_acc = tl.dot(x, tl.load(narrow_hi_ptr + weight_offsets), hi_acc)
        lo_acc = tl.dot(x, tl.load(narrow_lo_ptr + weight_offsets), lo_acc)
        squared += x.to(tl.float32) * x.to(tl.float32)
        row_ptrs += DEPTH
        weight_offsets += DEPTH * NARROW
    r = tl.rsqrt(tl.sum(squared, 1) / D + eps)
    return r[:, None] * (hi_acc + lo_acc * (1 / LO_SCALE)), r


#board_down_project's output for ROWS tokens of a board from FIRST_ROW, split (hi then lo)
@triton.jit
def head_board_down(board_ptr, narrow_hi_ptr, narrow_lo_ptr, narrow_bias_ptr, down_ptr, lo_offset, D, eps, T: tl.constexpr, FIRST_ROW: tl.constexpr,
                    ROWS: tl.constexpr, NARROW: tl.constexpr, DOWN: tl.constexpr, DEPTH: tl.constexpr, LO_SCALE: tl.constexpr):
    rows, columns = FIRST_ROW + tl.arange(0, ROWS), tl.arange(0, DOWN)
    down, _ = narrow_rows(board_ptr, narrow_hi_ptr, narrow_lo_ptr, D, eps, T, FIRST_ROW, ROWS, 0, DOWN, NARROW, DEPTH, LO_SCALE)
    hi, lo = split(gelu(down + tl.load(narrow_bias_ptr + columns)[None, :]), LO_SCALE)
    offsets = rows[:, None] * DOWN + columns[None, :]
    tl.store(down_ptr + offsets, hi, mask=(rows < T)[:, None])
    tl.store(down_ptr + lo_offset + offsets, lo, mask=(rows < T)[:, None])


#a board's square terms and r
@triton.jit
def head_square_terms(board_ptr, narrow_hi_ptr, narrow_lo_ptr, narrow_bias_ptr, terms_ptr, D, eps, T: tl.constexpr, NARROW: tl.constexpr,
                      DOWN: tl.constexpr, TERMS: tl.constexpr, R_TERM: tl.constexpr, DEPTH: tl.constexpr, LO_SCALE: tl.constexpr):
    squares, columns = tl.arange(0, 64), tl.arange(0, NARROW - DOWN)
    terms, r = narrow_rows(board_ptr, narrow_hi_ptr, narrow_lo_ptr, D, eps, T, 0, 64, DOWN, NARROW - DOWN, NARROW, DEPTH, LO_SCALE)
    terms += tl.load(narrow_bias_ptr + DOWN + columns)[None, :]
    tl.store(terms_ptr + squares[:, None] * TERMS + columns[None, :], terms, mask=(columns < R_TERM)[None, :])
    tl.store(terms_ptr + squares * TERMS + R_TERM, r)


#program (tile, board): every tile but the last three takes a's columns head_a_tile does (partials, summed by
#head_answer_kernel), the last three the narrow matmul: board_down_project's output on the squares, then on the
#metadata tokens, then the square terms
@triton.jit
def head_matmul_kernel(x_ptr, a_ptr, narrow_hi_ptr, narrow_lo_ptr, narrow_bias_ptr, partials_ptr, squares_ptr, board_down_ptr, B, D, eps,
                       T: tl.constexpr, A_COLUMNS: tl.constexpr, DEPTH: tl.constexpr, NARROW: tl.constexpr, DOWN: tl.constexpr,
                       TERMS: tl.constexpr, R_TERM: tl.constexpr, META: tl.constexpr, LO_SCALE: tl.constexpr, PDL: tl.constexpr):
    dependent_launch_wait(PDL)
    tile, b = tl.program_id(0), tl.program_id(1)
    a_tiles = tl.num_programs(0) - 3
    board_ptr = x_ptr + b * T * D
    down_ptr = board_down_ptr + b * T * DOWN
    if tile < a_tiles:
        head_a_tile(board_ptr, a_ptr, partials_ptr + (b * a_tiles + tile) * 4096, tile, D, eps, A_COLUMNS, DEPTH, LO_SCALE)
    elif tile == a_tiles:
        head_board_down(board_ptr, narrow_hi_ptr, narrow_lo_ptr, narrow_bias_ptr, down_ptr, B * T * DOWN, D, eps, T, 0, 64, NARROW, DOWN, DEPTH, LO_SCALE)
    elif tile == a_tiles + 1:
        head_board_down(board_ptr, narrow_hi_ptr, narrow_lo_ptr, narrow_bias_ptr, down_ptr, B * T * DOWN, D, eps, T, 64, META, NARROW, DOWN, DEPTH,
                        LO_SCALE)
    else:
        head_square_terms(board_ptr, narrow_hi_ptr, narrow_lo_ptr, narrow_bias_ptr, squares_ptr + b * 64 * TERMS, D, eps, T, NARROW, DOWN, TERMS, R_TERM,
                          DEPTH, LO_SCALE)


#BLOCK moves of a board from the first: r_t (a_f . x_t) + t_t + s_f, and the promotion's logit
@triton.jit
def head_moves(partials_ptr, terms_ptr, moves_ptr, rows_ptr, first, A_TILES: tl.constexpr, MOVES: tl.constexpr, BLOCK: tl.constexpr,
               TERMS: tl.constexpr, R_TERM: tl.constexpr):
    move = first + tl.arange(0, BLOCK)
    move_ok = move < MOVES
    #from * 64 + to, and the place of the promotion's logit among the board's square terms (-1 for none)
    pair = tl.load(moves_ptr + move, mask=move_ok, other=0)
    promotion = tl.load(moves_ptr + MOVES + move, mask=move_ok, other=-1)
    product = tl.zeros((BLOCK,), dtype=tl.float32)
    for tile in tl.static_range(A_TILES):
        product += tl.load(partials_ptr + tile * 4096 + pair)
    from_terms, to_terms = terms_ptr + pair // 64 * TERMS, terms_ptr + pair % 64 * TERMS
    logit = tl.load(to_terms + R_TERM) * product + tl.load(to_terms + 1) + tl.load(from_terms)
    logit += tl.load(terms_ptr + promotion, mask=promotion >= 0, other=0.0)
    tl.store(rows_ptr + move, logit, mask=move_ok)


#UNITS of the eval head's hidden layer for BOARDS boards (its inputs, board_down, and weights split: each hi then lo), and
#their part of its output layer
@triton.jit
def head_value_tile(board_down_ptr, hidden_ptr, hidden_bias_ptr, out_ptr, value_partials_ptr, unit_tile, board_tile, B, INPUTS: tl.constexpr,
                    HIDDEN: tl.constexpr, BINS: tl.constexpr, BLOCK_BINS: tl.constexpr, BOARDS: tl.constexpr, UNITS: tl.constexpr,
                    DEPTH: tl.constexpr, LO_SCALE: tl.constexpr):
    boards = board_tile * BOARDS + tl.arange(0, BOARDS)
    board_ok = boards < B
    units = unit_tile * UNITS + tl.arange(0, UNITS)
    depth = tl.arange(0, DEPTH)
    acc, rest = tl.zeros((BOARDS, UNITS), dtype=tl.float32), tl.zeros((BOARDS, UNITS), dtype=tl.float32)
    for start in range(0, INPUTS, DEPTH):
        inputs = start + depth
        inputs_ok = inputs < INPUTS
        downs = board_down_ptr + boards[:, None] * INPUTS + inputs[None, :]
        down_hi = tl.load(downs, mask=board_ok[:, None] & inputs_ok[None, :], other=0.0)
        down_lo = tl.load(downs + B * INPUTS, mask=board_ok[:, None] & inputs_ok[None, :], other=0.0)
        weights = hidden_ptr + units[None, :] * INPUTS + inputs[:, None]
        weight_hi = tl.load(weights, mask=inputs_ok[:, None], other=0.0)
        weight_lo = tl.load(weights + HIDDEN * INPUTS, mask=inputs_ok[:, None], other=0.0)
        acc = tl.dot(down_hi, weight_hi, acc)
        rest = tl.dot(down_lo, weight_hi, tl.dot(down_hi, weight_lo, rest))
    hidden_hi, hidden_lo = split(gelu(acc + rest * (1 / LO_SCALE) + tl.load(hidden_bias_ptr + units)[None, :]), LO_SCALE)
    bins = tl.arange(0, BLOCK_BINS)
    weights = out_ptr + bins[None, :] * HIDDEN + units[:, None]
    out_hi = tl.load(weights, mask=(bins < BINS)[None, :], other=0.0)
    out_lo = tl.load(weights + BINS * HIDDEN, mask=(bins < BINS)[None, :], other=0.0)
    value = tl.dot(hidden_hi, out_hi) + (tl.dot(hidden_lo, out_hi) + tl.dot(hidden_hi, out_lo)) * (1 / LO_SCALE)
    tl.store(value_partials_ptr + (unit_tile * B + boards[:, None]) * BLOCK_BINS + bins[None, :], value, mask=board_ok[:, None])


#programs 0 to boards * move programs - 1 each write MOVE_BLOCK moves of a board (head_moves), the rest each take a tile
#of the eval head (head_value_tile: value_partials, summed by head_value_kernel)
@triton.jit
def head_answer_kernel(partials_ptr, squares_ptr, moves_ptr, board_down_ptr, hidden_ptr, hidden_bias_ptr, out_ptr, value_partials_ptr, rows_ptr, B,
                       A_TILES: tl.constexpr, MOVES: tl.constexpr, MOVE_BLOCK: tl.constexpr, MOVE_PROGRAMS: tl.constexpr, TERMS: tl.constexpr,
                       R_TERM: tl.constexpr, INPUTS: tl.constexpr, HIDDEN: tl.constexpr, BINS: tl.constexpr, BLOCK_BINS: tl.constexpr,
                       BOARDS: tl.constexpr, UNITS: tl.constexpr, DEPTH: tl.constexpr, LO_SCALE: tl.constexpr, PDL: tl.constexpr):
    dependent_launch_wait(PDL)
    pid = tl.program_id(0)
    if pid < B * MOVE_PROGRAMS:
        b = pid // MOVE_PROGRAMS
        head_moves(partials_ptr + b * A_TILES * 4096, squares_ptr + b * 64 * TERMS, moves_ptr, rows_ptr + b * (MOVES + 1), pid % MOVE_PROGRAMS * MOVE_BLOCK,
                   A_TILES, MOVES, MOVE_BLOCK, TERMS, R_TERM)
    else:
        tile = pid - B * MOVE_PROGRAMS
        board_tiles = tl.cdiv(B, BOARDS)
        head_value_tile(board_down_ptr, hidden_ptr, hidden_bias_ptr, out_ptr, value_partials_ptr, tile // board_tiles, tile % board_tiles, B, INPUTS,
                        HIDDEN, BINS, BLOCK_BINS, BOARDS, UNITS, DEPTH, LO_SCALE)


#a board's expected score from the eval head's output (value_partials summed, then its bias): the bins' softmax times
#their centres, or one bin's sigmoid (model.expected_score)
@triton.jit
def head_value_kernel(value_partials_ptr, out_bias_ptr, centres_ptr, rows_ptr, B, UNIT_TILES: tl.constexpr, BLOCK_TILES: tl.constexpr,
                      MOVES: tl.constexpr, BINS: tl.constexpr, BLOCK_BINS: tl.constexpr, PDL: tl.constexpr):
    dependent_launch_wait(PDL)
    b = tl.program_id(0)
    tiles, bins = tl.arange(0, BLOCK_TILES), tl.arange(0, BLOCK_BINS)
    bin_ok = bins < BINS
    partials = tl.load(value_partials_ptr + (tiles[:, None] * B + b) * BLOCK_BINS + bins[None, :], mask=(tiles < UNIT_TILES)[:, None], other=0.0)
    value = tl.sum(partials, 0) + tl.load(out_bias_ptr + bins, mask=bin_ok, other=0.0)
    if BINS == 1:
        score = tl.sigmoid(tl.sum(tl.where(bins == 0, value, 0.0), 0))
    else:
        value = tl.where(bin_ok, value, float("-inf"))
        exponent = libdevice.exp(value - tl.max(value, 0))
        probability = exponent / tl.sum(exponent, 0)
        score = tl.sum(probability * tl.load(centres_ptr + bins, mask=bin_ok, other=0.0), 0)
    tl.store(rows_ptr + b * (MOVES + 1) + MOVES, score)


#a's columns a program of head_matmul_kernel takes: fewer for fewer boards, so that a small batch runs on more
#multiprocessors (the fastest of 32-128 on an RTX 5090 and 5070 Ti)
def head_a_columns(boards: int, d_model: int) -> int:
    columns = 32 if boards <= 4 else 64
    while d_model % columns:
        columns //= 2
    return columns


#the eval head's inputs a program of head_answer_kernel reads at a time for this many boards (the fastest of 32-128)
def value_depth(boards: int) -> int:
    return 32 if boards >= 64 else 128


#an opaque op for torch.compile, as linear. The stream B x T x d (fp16, or rounded to it) in, the answer rows out;
#hidden and out are the eval head's two layers' weights split (2 x out_features x in_features)
@torch.library.custom_op("chess::heads", mutates_args=())
def heads(stream: torch.Tensor, a_weight: torch.Tensor, narrow_hi: torch.Tensor, narrow_lo: torch.Tensor, narrow_bias: torch.Tensor,
          moves: torch.Tensor, hidden: torch.Tensor, hidden_bias: torch.Tensor, out: torch.Tensor, out_bias: torch.Tensor, centres: torch.Tensor,
          eps: float) -> torch.Tensor:
    stream = stream.contiguous()
    B, T, D = stream.shape
    move_count = moves.shape[1]
    _, hidden_units, inputs = hidden.shape
    bins = out.shape[1]
    depth = HEAD_DEPTH if D % HEAD_DEPTH == 0 else HEAD_DEPTH // 2
    if D % depth != 0 or hidden_units % VALUE_UNITS != 0:
        raise ValueError(f"the fused heads read d_model {D} {depth} at a time and the eval head's {hidden_units} units {VALUE_UNITS} at a time")
    a_columns = head_a_columns(B, D)
    a_tiles = D // a_columns
    block_bins, value_boards = max(16, triton.next_power_of_2(bins)), min(64, max(16, triton.next_power_of_2(B)))
    unit_tiles, board_tiles = hidden_units // VALUE_UNITS, triton.cdiv(B, value_boards)
    f32 = dict(device=stream.device, dtype=torch.float32)
    partials, squares = torch.empty(B, a_tiles, 64, 64, **f32), torch.empty(B, 64, SQUARE_TERMS, **f32)
    board_down, value_partials = torch.empty(2, B, inputs, device=stream.device, dtype=torch.float16), torch.empty(unit_tiles, B, block_bins, **f32)
    rows = torch.empty(B, move_count + 1, **f32)
    launch = dependent_launch(stream.device)
    head_matmul_kernel[(a_tiles + 3, B)](stream, a_weight, narrow_hi, narrow_lo, narrow_bias, partials, squares, board_down, B, D, eps, T=T,
                                         A_COLUMNS=a_columns, DEPTH=depth, NARROW=NARROW_COLUMNS, DOWN=DOWN_COLUMNS, TERMS=SQUARE_TERMS,
                                         R_TERM=R_TERM, META=META_ROWS, LO_SCALE=LO_SCALE, num_warps=4, **launch)
    move_programs = triton.cdiv(move_count, MOVE_BLOCK)
    head_answer_kernel[(B * move_programs + unit_tiles * board_tiles,)](
        partials, squares, moves, board_down, hidden, hidden_bias, out, value_partials, rows, B, A_TILES=a_tiles, MOVES=move_count,
        MOVE_BLOCK=MOVE_BLOCK, MOVE_PROGRAMS=move_programs, TERMS=SQUARE_TERMS, R_TERM=R_TERM, INPUTS=inputs, HIDDEN=hidden_units, BINS=bins,
        BLOCK_BINS=block_bins, BOARDS=value_boards, UNITS=VALUE_UNITS, DEPTH=value_depth(value_boards), LO_SCALE=LO_SCALE, num_warps=4, **launch)
    head_value_kernel[(B,)](value_partials, out_bias, centres, rows, B, UNIT_TILES=unit_tiles, BLOCK_TILES=triton.next_power_of_2(unit_tiles),
                            MOVES=move_count, BINS=bins, BLOCK_BINS=block_bins, num_warps=4, **launch)
    return rows


@heads.register_fake
def _(stream, a_weight, narrow_hi, narrow_lo, narrow_bias, moves, hidden, hidden_bias, out, out_bias, centres, eps):
    return stream.new_empty(stream.shape[0], moves.shape[1] + 1, dtype=torch.float32)


#a weight as split gives it, hi then lo
def split_weight(weight: torch.Tensor) -> torch.Tensor:
    hi = weight.half()
    return torch.stack([hi, ((weight.double() - hi.double()) * LO_SCALE).half()])


#the fused heads' weights: logits[f, t] = (W_f y_f + b_f) . (W_t y_t + b_t) / sqrt(d) = r_f r_t x_f A x_t + t_t + s_f
#for the normed rows y = r g x, A = diag(g) W_f^T W_t diag(g) / sqrt(d) in fp16. The narrow weights (d x NARROW_COLUMNS,
#split) give board_down_project's output, s_f = y_f . W_f^T b_t / sqrt(d) + b_f . b_t / sqrt(d), t_t = y_t . W_t^T b_f
#/ sqrt(d) and the promotion logits
def fused_head_weights(model: ChessTransformer) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        gain = model.ln.weight.double()
        from_weight, from_bias = model.from_projection.weight.double(), model.from_projection.bias.double()
        to_weight, to_bias = model.to_projection.weight.double(), model.to_projection.bias.double()
        promotion, down = model.promotion_projection, model.board_down_project[0]
        scale = 1 / math.sqrt(model.d_model)
        a = gain[:, None] * (from_weight.T @ to_weight * scale) * gain[None, :]
        narrow = torch.cat([down.weight.double(), (to_bias @ from_weight)[None] * scale, (from_bias @ to_weight)[None] * scale,
                            promotion.weight.double()])
        narrow_bias = torch.cat([down.bias.double(), (from_bias @ to_bias)[None] * scale, torch.zeros(1, dtype=torch.float64, device=gain.device),
                                 promotion.bias.double()])
        padding = NARROW_COLUMNS - len(narrow)
        narrow = F.pad(narrow * gain[None, :], (0, 0, 0, padding)).T.contiguous()
        return a.half(), split_weight(narrow), F.pad(narrow_bias, (0, padding)).float()


#per move from * 64 + to, and the place of its promotion's logit among its board's square terms (-1 for none)
def move_table(model: ChessTransformer) -> torch.Tensor:
    promotions = torch.full_like(model.from_squares, -1)
    promotions[model.promotion_move_ids] = model.promotion_to_squares * SQUARE_TERMS + PROMOTION_TERM + model.promotion_pieces
    return torch.stack([model.from_squares * NUM_SQUARES + model.to_squares, promotions]).int()


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


#ChessTransformer.forward with FusedBlocks, for inference: a change to it must be made here too. On the gpus of
#fused_ends the heads are their kernels (heads), which give the answer rows rather than the model's logits and value
#logits (inference.answer_rows takes either); on others head_weights' matmuls
class FusedTransformer(nn.Module):
    def __init__(self, model: ChessTransformer):
        super().__init__()
        self.d_model = model.d_model
        self.input_embedding = InputEmbedding(model)
        self.templates, self.ln, self.loops = model.templates, model.ln, model.loops
        self.encoder = nn.ModuleList(FusedBlock(block) for block in model.encoder)
        link_input_mixes(list(self.encoder))
        self.fused_heads = fused_ends(model.embed.weight.device)
        if self.fused_heads:
            self.eps = model.ln.eps if model.ln.eps is not None else torch.finfo(torch.float32).eps
            a_weight, narrow, narrow_bias = fused_head_weights(model)
            hidden, out = model.eval_head[0], model.eval_head[2]
            edges = value_bin_edges(out.out_features, out.weight.device)
            buffers = {"a_weight": a_weight, "narrow": narrow, "narrow_bias": narrow_bias, "moves": move_table(model), "hidden": split_weight(hidden.weight),
                       "hidden_bias": hidden.bias.detach(), "out": split_weight(out.weight), "out_bias": out.bias.detach(),
                       "centres": (edges[:-1] + edges[1:]) / 2}
            for name, tensor in buffers.items():
                self.register_buffer(name, tensor)
            return
        self.board_down_activation, self.eval_head = model.board_down_project[1], model.eval_head
        for name in ("from_squares", "to_squares", "promotion_move_ids", "promotion_to_squares", "promotion_pieces"):
            self.register_buffer(name, getattr(model, name))
        self.a_weight, self.a_bias, self.narrow_weight, self.narrow_bias = (nn.Parameter(tensor, requires_grad=False) for tensor in head_weights(model))
        promotions = model.promotion_projection.out_features
        self.promotion_columns, self.down_columns = slice(1, 1 + promotions), slice(1 + promotions, None)

    def forward(self, board_state: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embed_state = self.input_embedding(board_state)
        original_board_embed = embed_state
        for _ in range(self.loops):
            for block in self.encoder:
                embed_state = block(embed_state, self.templates, original_board_embed)
        if self.fused_heads:
            return self.answers(embed_state)
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

    #the fused heads on the last block's stream: the answer rows
    def answers(self, embed_state: torch.Tensor) -> torch.Tensor:
        return heads(embed_state, self.a_weight, self.narrow[0], self.narrow[1], self.narrow_bias, self.moves, self.hidden, self.hidden_bias, self.out,
                     self.out_bias, self.centres, self.eps)

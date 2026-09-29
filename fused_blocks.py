#the engine's transformer blocks for inference on the gpu: the same arithmetic as model.TransformerBlock, arranged
#for speed (tests/test_inference.py checks they agree, so a change to the block must be made here too). Q, K, V and
#the template mix are one matmul, the mlp's gate and up another, and attention is one Triton kernel: it RMS-normalises
#the queries and keys (QK-norm), adds the template bias, takes the softmax and mixes the values without writing the
#scores, the bias or any transposed copy to memory, and each query head reads its group's key/value head in place
#instead of a copy (repeat_interleave)
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn

from model import NUM_SQUARES, TransformerBlock

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


class FusedBlock(nn.Module):
    #takes over a trained block's weights (copied into the joined matmuls, or shared)
    def __init__(self, block: TransformerBlock):
        super().__init__()
        self.heads, self.kv_heads = block.n_heads, block.kv_heads
        self.dq, self.dv, self.templates_count = block.d_qk_head, block.d_head, block.position_templates
        self.ln1, self.ln2, self.combine_heads, self.down = block.ln1, block.ln2, block.combine_heads, block.mlp.down_projection
        self.q_gain, self.k_gain = block.q_norm.weight, block.k_norm.weight
        self.eps = block.q_norm.eps if block.q_norm.eps is not None else torch.finfo(torch.float32).eps
        weight = block.Q.weight
        with torch.no_grad():
            projections = (block.Q, block.K, block.V, block.template_mix)
            self.projection = nn.Linear(block.d_model, sum(p.out_features for p in projections), device=weight.device, dtype=weight.dtype)
            self.projection.weight.copy_(torch.cat([projection.weight for projection in projections]))
            self.projection.bias.copy_(torch.cat([projection.bias for projection in projections]))
            mlp = block.mlp
            self.gate_up = nn.Linear(block.d_model, 2 * mlp.up_projection.out_features, bias=False, device=weight.device, dtype=weight.dtype)
            self.gate_up.weight.copy_(torch.cat([mlp.gate_projection.weight, mlp.up_projection.weight]))

    def forward(self, x: torch.Tensor, templates: torch.Tensor) -> torch.Tensor:
        attended = template_attention(self.projection(self.ln1(x)), templates, self.q_gain, self.k_gain, self.heads, self.kv_heads, self.dq,
                                      self.dv, self.templates_count, self.eps)
        x = x + self.combine_heads(attended)
        gate, up = self.gate_up(self.ln2(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(gate) * up)


#the model with its blocks swapped for FusedBlocks, for inference only; embeddings, square features and heads unchanged
def fuse_blocks(model: nn.Module) -> nn.Module:
    model.encoder = nn.ModuleList(FusedBlock(block) for block in model.encoder)
    return model

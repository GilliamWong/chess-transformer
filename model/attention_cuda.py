#template attention of the d768's shapes as one CUDA kernel (NVRTC, launched as model/int8_cuda.py launches its kernels):
#a CTA takes a position and 8 query heads (4 key/value heads), makes their template bias itself on tensor cores 16
#squares at a time into shared memory (two squares' 8 heads times their 8 template rows: the bias kernel's operands, so
#its bits) and attends with it, the metadata rows last: the bias kernel and its buffers, the second stream, the chunks of
#positions and the metadata kernel go. Used only on the gpus it was timed on; fused_blocks.template_attention runs its
#Triton kernels everywhere else
import ctypes
import math
import sys
import threading
import warnings

import torch

from model import int8_cuda

SOURCE = r"""
typedef unsigned int u32;
typedef unsigned short u16;
#define DEV __device__ __forceinline__

DEV void mma(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
DEV float half_to_float(u32 h) { float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"((u16)h)); return f; }
//the two halves of a register, low then high, as floats
DEV float low_half(u32 pair) { return half_to_float(pair & 0xffff); }
DEV float high_half(u32 pair) { return half_to_float(pair >> 16); }
DEV u32 pack(float lo, float hi) { u32 r; asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(lo), "f"(hi)); return r; }
DEV float ex2_approx(float x) { float y; asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float div_full(float a, float b) { float y; asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }
DEV float rsqrt_approx(float x) { float y; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float quad_sum(float x) { x += __shfl_xor_sync(0xffffffff, x, 1); return x + __shfl_xor_sync(0xffffffff, x, 2); }
DEV float quad_max(float x) { x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, 1)); return fmaxf(x, __shfl_xor_sync(0xffffffff, x, 2)); }
#define NEG_INF __int_as_float(0xff800000)

//a board: 64 squares then 6 metadata tokens. A CTA takes 8 of the 24 query heads: a warp a head, 2 heads to a key/value
//head
constexpr int TOKENS = 70, SQUARES = 64, DV = 32, TEMPLATES = 8, HEADS = 24, SLAB = 16, PHASES = SQUARES / SLAB + 1;
constexpr int HB = @HB@, GROUPS = HB / 2, THREADS = HB * 32;
//the keys in n8 tiles: 72, the 6 metadata tokens after the squares and 2 that are never real; in key steps of 16, KEYS
constexpr int KEY_TILES = 9, KEYS = @KEYS@;
//halves in a value row (a key's 32 values, padded against bank conflicts) and in a bias row; a head has BIAS_ROWS rows of
//them (the 16 squares of a slab and a pad) so that a warp's reads and the tensor cores' writes both fall in different banks
constexpr int VALUE_ROW = @VALUE_ROW@, BIAS_ROW = @BIAS_ROW@, BIAS_ROWS = @BIAS_ROWS@;

//a board's row of projected is q (24 heads x DQ) | k (12 x DQ) | v (12 x DV) | the template mix (24 x TEMPLATES)
//and the dims of a head, in blocks of 8: a tensor core takes 16 at a step, the second block of an odd count being zeros
template <int DQ> struct Row {
    static constexpr int WIDTH = 36 * DQ + 576, Q = 0, K = 24 * DQ, V = 36 * DQ, C = 36 * DQ + 384;
    static constexpr int BLOCKS = DQ / 8, STEPS = (BLOCKS + 1) / 2;
};

//the raw queries of the 16 rows from token first: each block of 8 dims of row g and of row g + 8
template <int DQ>
DEV void load_queries(const u16* board, int head, int first, int g, int t, u32 (&raw)[Row<DQ>::BLOCKS][2])
{
    #pragma unroll
    for (int block = 0; block < Row<DQ>::BLOCKS; block++)
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            const int token = first + g + 8 * half;
            raw[block][half] = token < TOKENS ? *reinterpret_cast<const u32*>(board + token * Row<DQ>::WIDTH + Row<DQ>::Q + head * DQ + block * 8 + 2 * t) : 0u;
        }
}

//what a warp's part of a slab's bias reads: a pair of its squares (the warp's number), for each 8 keys the squares' 8
//templates (as B operands, the depth 0-7 for one square and 8-15 for the other, in pairs of a register) and the 8 heads' 8
//template coefficients (as A operands, the rows 0-7 for one square and 8-15 for the other). templates are
//[square][key % 8][template pair][key / 8], each entry a pair of templates, so that a thread's operands for a square are
//32 contiguous bytes
struct BiasOperands {
    u32 templates[SQUARES / 8][2], coefficients[2];
};

template <int DQ>
DEV void load_bias_operands(BiasOperands& operands, const u16* board, const u32* templates, int slab, int warp, int g, int t)
{
    const int square = slab * SLAB + 2 * warp;
    const u16* mix = board + Row<DQ>::C + (blockIdx.x * HB + g) * TEMPLATES + 2 * t;
    operands.coefficients[0] = *reinterpret_cast<const u32*>(mix + square * Row<DQ>::WIDTH);
    operands.coefficients[1] = *reinterpret_cast<const u32*>(mix + (square + 1) * Row<DQ>::WIDTH);
    #pragma unroll
    for (int other = 0; other < 2; other++) {
        const uint4* from = reinterpret_cast<const uint4*>(templates + (((square + other) * 8 + g) * (TEMPLATES / 2) + t) * (SQUARES / 8));
        const uint4 first = from[0], second = from[1];
        const u32 tiles[SQUARES / 8] = {first.x, first.y, first.z, first.w, second.x, second.y, second.z, second.w};
        #pragma unroll
        for (int tile = 0; tile < SQUARES / 8; tile++)
            operands.templates[tile][other] = tiles[tile];
    }
}

//a slab's bias for the 8 heads on the tensor cores, rounded to fp16 as the bias kernel stores it
DEV void make_bias(const BiasOperands& operands, u16* to_bias, int warp, int g, int t)
{
    #pragma unroll
    for (int tile = 0; tile < SQUARES / 8; tile++) {
        float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        mma(c, operands.coefficients[0], 0, 0, operands.coefficients[1], operands.templates[tile][0], operands.templates[tile][1]);
        u16* to = to_bias + (g * BIAS_ROWS + 2 * warp) * BIAS_ROW + tile * 8 + 2 * t;
        *reinterpret_cast<u32*>(to) = pack(c[0], c[1]);
        *reinterpret_cast<u32*>(to + BIAS_ROW) = pack(c[2], c[3]);
    }
}

//out[b, token, head * DV + :] for the 8 heads from blockIdx.x * 8 of position blockIdx.y
template <int DQ>
__global__ void __launch_bounds__(THREADS, 2) template_attention_cuda(
    const u16* __restrict__ projected, const u32* __restrict__ templates, const u16* __restrict__ q_gain,
    const u16* __restrict__ k_gain, u16* __restrict__ out, float eps, float scale)
{
    constexpr int BLOCKS = Row<DQ>::BLOCKS, STEPS = Row<DQ>::STEPS, WIDTH = Row<DQ>::WIDTH;
    extern __shared__ __align__(16) u16 shared[];
    u16* values = shared;                                          //[group][KEYS][VALUE_ROW], keys padded with zeros
    u16* bias = values + GROUPS * KEYS * VALUE_ROW;                  //[head][BIAS_ROWS][BIAS_ROW]: a slab's squares, keys last
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int b = blockIdx.y, head = blockIdx.x * HB + warp, group = warp >> 1, kv = blockIdx.x * GROUPS + group;

    //the gains of the dims this thread holds: d = block * 8 + 2 t + e
    //the queries' are held except at 24 dims, where registers run out and they are read where used
    constexpr bool HOLD_QUERY_GAINS = DQ <= 16;
    float q_gains[BLOCKS][2], k_gains[BLOCKS][2];
    #pragma unroll
    for (int block = 0; block < BLOCKS; block++)
        #pragma unroll
        for (int e = 0; e < 2; e++) {
            q_gains[block][e] = half_to_float(q_gain[block * 8 + 2 * t + e]);
            k_gains[block][e] = half_to_float(k_gain[block * 8 + 2 * t + e]);
        }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    const u16* board = projected + (size_t)b * TOKENS * WIDTH;

    //every load of the prologue at once: the values of the CTA's key/value heads, copied to shared memory as the rest goes on
    //(the keys past the board zero), the keys of this warp's, the first queries and the first slab's bias operands
    #pragma unroll
    for (int round = 0; round < GROUPS * KEYS * 4 / THREADS; round++) {
        const int item = threadIdx.x + round * THREADS;
        const int part = item & 3, key = item / 4 % KEYS, row_group = item / 4 / KEYS;
        u16* to = values + ((row_group * KEYS + key) * VALUE_ROW + part * 8);
        if (key < TOKENS)
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((u32)__cvta_generic_to_shared(to)),
                         "l"(board + key * WIDTH + Row<DQ>::V + (blockIdx.x * GROUPS + row_group) * DV + part * 8));
        else
            *reinterpret_cast<uint4*>(to) = make_uint4(0, 0, 0, 0);
    }
    u32 raw_keys[KEY_TILES][BLOCKS];
    #pragma unroll
    for (int n = 0; n < KEY_TILES; n++)
        #pragma unroll
        for (int block = 0; block < BLOCKS; block++) {
            const int key = n * 8 + g;
            raw_keys[n][block] = key < TOKENS ? *reinterpret_cast<const u32*>(board + key * WIDTH + Row<DQ>::K + kv * DQ + block * 8 + 2 * t) : 0u;
        }
    u32 raw_queries[BLOCKS][2];
    load_queries<DQ>(board, head, 0, g, t, raw_queries);
    BiasOperands operands;
    load_bias_operands<DQ>(operands, board, templates, 0, warp, g, t);

    //the keys RMS-normalised, as the B operands of the scores
    u32 keys[KEY_TILES][BLOCKS];
    #pragma unroll
    for (int n = 0; n < KEY_TILES; n++) {
        float x[BLOCKS][2];
        float squares = 0.0f;
        #pragma unroll
        for (int block = 0; block < BLOCKS; block++) {
            x[block][0] = low_half(raw_keys[n][block]);
            x[block][1] = high_half(raw_keys[n][block]);
            squares = __fmaf_rn(x[block][0], x[block][0], squares);
            squares = __fmaf_rn(x[block][1], x[block][1], squares);
        }
        const float inverse = rsqrt_approx(quad_sum(squares) / DQ + eps);
        #pragma unroll
        for (int block = 0; block < BLOCKS; block++)
            keys[n][block] = pack(x[block][0] * inverse * k_gains[block][0], x[block][1] * inverse * k_gains[block][1]);
    }

    //the values are in shared memory by the first barrier
    asm volatile("cp.async.wait_all;" ::: "memory");

    //the squares 16 at a time, then the metadata tokens
    #pragma unroll 1
    for (int phase = 0; phase < PHASES; phase++) {
        const bool metadata = phase == PHASES - 1;
        if (!metadata) {
            make_bias(operands, bias, warp, g, t);
            __syncthreads();
        }
        //the next slab's queries and bias operands load as this one is attended
        u32 next_queries[BLOCKS][2];
        if (!metadata) {
            load_queries<DQ>(board, head, phase + 1 < PHASES - 1 ? (phase + 1) * SLAB : SQUARES, g, t, next_queries);
            if (phase + 1 < PHASES - 1)
                load_bias_operands<DQ>(operands, board, templates, phase + 1, warp, g, t);
        }

        //this warp's head: rows g and g + 8 of the tile
        const int first = metadata ? SQUARES : phase * SLAB;
        int token[2];
        bool row_ok[2];
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            token[half] = first + g + 8 * half;
            row_ok[half] = token[half] < TOKENS;
        }
        //the queries RMS-normalised
        u32 queries[BLOCKS][2];
        float y[BLOCKS][2][2];
        float squares[2] = {0.0f, 0.0f};
        #pragma unroll
        for (int block = 0; block < BLOCKS; block++)
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                y[block][half][0] = low_half(raw_queries[block][half]);
                y[block][half][1] = high_half(raw_queries[block][half]);
                squares[half] = __fmaf_rn(y[block][half][0], y[block][half][0], squares[half]);
                squares[half] = __fmaf_rn(y[block][half][1], y[block][half][1], squares[half]);
            }
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            const float inverse = rsqrt_approx(quad_sum(squares[half]) / DQ + eps);
            #pragma unroll
            for (int block = 0; block < BLOCKS; block++) {
                const u32 read = HOLD_QUERY_GAINS ? 0u : *reinterpret_cast<const u32*>(q_gain + block * 8 + 2 * t);
                queries[block][half] = pack(y[block][half][0] * inverse * (HOLD_QUERY_GAINS ? q_gains[block][0] : low_half(read)),
                                            y[block][half][1] * inverse * (HOLD_QUERY_GAINS ? q_gains[block][1] : high_half(read)));
            }
        }

        //scores against the squares and the metadata tokens, 8 keys a tile
        float scores[KEY_TILES][4];
        #pragma unroll
        for (int n = 0; n < KEY_TILES; n++) {
            scores[n][0] = scores[n][1] = scores[n][2] = scores[n][3] = 0.0f;
            #pragma unroll
            for (int s = 0; s < STEPS; s++) {
                const bool pair = 2 * s + 1 < BLOCKS;
                mma(scores[n], queries[2 * s][0], queries[2 * s][1], pair ? queries[2 * s + 1][0] : 0u, pair ? queries[2 * s + 1][1] : 0u,
                    keys[n][2 * s], pair ? keys[n][2 * s + 1] : 0u);
            }
        }
        float top[2] = {NEG_INF, NEG_INF};
        const u16* head_bias = bias + warp * BIAS_ROWS * BIAS_ROW;
        #pragma unroll
        for (int n = 0; n < KEY_TILES; n++)
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                //the metadata rows and keys read no bias
                const u32 pair = !metadata && n < SQUARES / 8 ? *reinterpret_cast<const u32*>(head_bias + (g + 8 * half) * BIAS_ROW + n * 8 + 2 * t) : 0u;
                #pragma unroll
                for (int e = 0; e < 2; e++) {
                    float s = scores[n][2 * half + e] * scale;
                    if (!metadata && n < SQUARES / 8)
                        s += e ? high_half(pair) : low_half(pair);
                    if (n * 8 + 2 * t + e >= TOKENS)
                        s = NEG_INF;
                    scores[n][2 * half + e] = s;
                    top[half] = fmaxf(top[half], s);
                }
            }
        #pragma unroll
        for (int half = 0; half < 2; half++)
            top[half] = quad_max(top[half]);
        float total[2] = {0.0f, 0.0f};
        #pragma unroll
        for (int n = 0; n < KEY_TILES; n++)
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const float p = ex2_approx((scores[n][e] - top[e >> 1]) * 1.44269504088896340736f);
                scores[n][e] = p;
                total[e >> 1] += p;
            }
        #pragma unroll
        for (int half = 0; half < 2; half++)
            total[half] = quad_sum(total[half]);

        float acc[DV / 8][4];
        #pragma unroll
        for (int n = 0; n < DV / 8; n++)
            acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
        #pragma unroll
        for (int k = 0; k < 5; k++) {
            //the last 16 keys' second tile (keys 72 to 79) is never real
            const int next = k < 4 ? 2 * k + 1 : 0;
            const u32 p0 = pack(scores[2 * k][0], scores[2 * k][1]), p1 = pack(scores[2 * k][2], scores[2 * k][3]);
            const u32 p2 = k < 4 ? pack(scores[next][0], scores[next][1]) : 0u, p3 = k < 4 ? pack(scores[next][2], scores[next][3]) : 0u;
            //the B operands of two 8-column tiles at once: the 8 x 8 blocks of 8 keys and 8 values, transposed
            #pragma unroll
            for (int n = 0; n < DV / 8; n += 2) {
                u32 b0, b1, b2, b3;
                const u16* rows = values + (group * KEYS + k * 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * VALUE_ROW + n * 8 + (lane >> 4) * 8;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                             : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"((u32)__cvta_generic_to_shared(rows)));
                mma(acc[n], p0, p1, p2, p3, b0, b1);
                mma(acc[n + 1], p0, p1, p2, p3, b2, b3);
            }
        }
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            if (!row_ok[half])
                continue;
            const float inverse = div_full(1.0f, total[half]);
            u16* to = out + ((size_t)b * TOKENS + token[half]) * (HEADS * DV) + head * DV + 2 * t;
            #pragma unroll
            for (int n = 0; n < DV / 8; n++)
                *reinterpret_cast<u32*>(to + n * 8) = pack(acc[n][2 * half] * inverse, acc[n][2 * half + 1] * inverse);
        }
        #pragma unroll
        for (int block = 0; block < BLOCKS; block++)
            #pragma unroll
            for (int half = 0; half < 2; half++)
                raw_queries[block][half] = next_queries[block][half];
        //the next slab's bias goes where this one's was
        if (!metadata)
            __syncthreads();
    }
}
"""

#what the source and the launch share: query heads a CTA takes (a warp each), keys a key/value head's values are kept for, and
#the halves of a key's row of values and of a row and the rows of a head's bias in shared memory
HEADS_PER_CTA, KEYS, VALUE_ROW, BIAS_ROW, BIAS_ROWS = 8, 80, 40, 72, 17
for name, value in {"HB": HEADS_PER_CTA, "KEYS": KEYS, "VALUE_ROW": VALUE_ROW, "BIAS_ROW": BIAS_ROW, "BIAS_ROWS": BIAS_ROWS}.items():
    SOURCE = SOURCE.replace(f"@{name}@", str(value))
THREADS = HEADS_PER_CTA * 32
#a key/value head's values for each 2 query heads, and a slab of bias for each
SHARED_BYTES = 2 * (HEADS_PER_CTA // 2 * KEYS * VALUE_ROW + HEADS_PER_CTA * BIAS_ROWS * BIAS_ROW)

#by (compute capability, multiprocessors): the gpus where the kernel was timed against the Triton kernels, with the fewest
#positions it takes and the share of its CTA slots its last wave must fill (a few CTAs past a wave take as long as another)
TIMED = {(12, 0, 70): (36, 0.55)}
#CTAs a position makes (24 query heads over HEADS_PER_CTA), and the most that share a multiprocessor
CTAS_PER_POSITION = 24 // HEADS_PER_CTA
CTAS_PER_MULTIPROCESSOR = 2
DQS = (8, 16, 24)
#off: the Triton kernels everywhere (the tests of those)
ENABLED = True
CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8


def kernel_name(dq: int) -> str:
    return f"template_attention_cuda<{dq}>"


class Kernels:
    #the kernels of one gpu, compiled and loaded into torch's context on it; fewest_positions and least_fill as TIMED's
    def __init__(self, device: torch.device, fewest_positions: int, least_fill: float):
        self.fewest_positions, self.least_fill = fewest_positions, least_fill
        self.slots = CTAS_PER_MULTIPROCESSOR * torch.cuda.get_device_properties(device).multi_processor_count
        self.driver = ctypes.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
        self.driver.cuLaunchKernelEx.argtypes = [ctypes.POINTER(int8_cuda.LaunchConfig), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        cubin, lowered = int8_cuda.compile_cubin(SOURCE, [kernel_name(dq) for dq in DQS], torch.cuda.get_device_capability(device))
        with torch.cuda.device(device):
            torch.empty(0, device=device)
            self.module = ctypes.c_void_p()
            self.check(self.driver.cuModuleLoadData(ctypes.byref(self.module), cubin))
        self.functions = {}
        for dq in DQS:
            function = ctypes.c_void_p()
            self.check(self.driver.cuModuleGetFunction(ctypes.byref(function), self.module, lowered[kernel_name(dq)].encode()))
            self.check(self.driver.cuFuncSetAttribute(function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, SHARED_BYTES))
            self.functions[dq] = function

    def check(self, result: int) -> None:
        if result != 0:
            message = ctypes.c_char_p()
            self.driver.cuGetErrorString(result, ctypes.byref(message))
            raise RuntimeError(f"CUDA driver: {message.value.decode() if message.value else result}")

    #whether this many positions are timed faster here than the Triton kernels
    def takes(self, positions: int) -> bool:
        ctas = CTAS_PER_POSITION * positions
        waves = -(-ctas // self.slots)
        return positions >= self.fewest_positions and ctas >= self.least_fill * waves * self.slots

    def launch(self, projected: torch.Tensor, templates: torch.Tensor, q_gain: torch.Tensor, k_gain: torch.Tensor, out: torch.Tensor, dq: int,
               eps: float) -> None:
        attribute = (int8_cuda.LaunchAttribute * 1)()
        attribute[0].id = int8_cuda.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
        attribute[0].value[0] = 1
        stream = torch.cuda.current_stream(projected.device).cuda_stream
        launch = int8_cuda.LaunchConfig(CTAS_PER_POSITION, projected.shape[0], 1, THREADS, 1, 1, SHARED_BYTES, stream, attribute, 1)
        arguments = [ctypes.c_void_p(tensor.data_ptr()) for tensor in (projected, templates, q_gain, k_gain, out)]
        arguments += [ctypes.c_float(eps), ctypes.c_float(1 / math.sqrt(dq))]
        pointers = (ctypes.c_void_p * len(arguments))(*[ctypes.cast(ctypes.byref(argument), ctypes.c_void_p) for argument in arguments])
        self.check(self.driver.cuLaunchKernelEx(ctypes.byref(launch), self.functions[dq], pointers, None))


#Kernels by gpu index, or None where they are not used
KERNELS: dict[int, Kernels | None] = {}


def kernels(device: torch.device) -> Kernels | None:
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in KERNELS:
        KERNELS[index] = None
        timed = TIMED.get((*torch.cuda.get_device_capability(index), torch.cuda.get_device_properties(index).multi_processor_count))
        if timed is not None:
            try:
                KERNELS[index] = Kernels(torch.device("cuda", index), *timed)
            except (OSError, RuntimeError, AttributeError) as error:
                warnings.warn(f"the CUDA template attention is off, the Triton kernels run instead: {error}")
    return KERNELS[index]


#the templates [template][square][key] as the kernel's bias operands read them, [square][key % 8][template // 2][key // 8] of
#pairs of templates, made once for as long as the tensor lives unchanged (the entry holds the tensor, so that its memory is
#not another's), or None while a cuda graph records and it is not made. The models of every gpu pair theirs at once, on threads
PAIRED: dict[int, tuple[torch.Tensor, int, torch.Tensor]] = {}
PAIRED_LOCK = threading.Lock()
#the most kept: the oldest go first
PAIRED_MOST = 64


#the count of in-place updates of a tensor (an inference tensor, made in torch.inference_mode, has none)
def version(tensor: torch.Tensor) -> int:
    return 0 if tensor.is_inference() else tensor._version


def paired_templates(templates: torch.Tensor) -> torch.Tensor | None:
    found = PAIRED.get(templates.data_ptr())
    if found is not None and found[1] == version(templates) and found[0].shape == templates.shape:
        return found[2]
    if torch.cuda.is_current_stream_capturing():
        return None
    with torch.no_grad():
        paired = templates.reshape(4, 2, 64, 8, 8).permute(2, 4, 0, 3, 1).contiguous()
    with PAIRED_LOCK:
        PAIRED[templates.data_ptr()] = (templates, version(templates), paired)
        while len(PAIRED) > PAIRED_MOST:
            del PAIRED[next(iter(PAIRED))]
    return paired


#whether the kernel takes these templates on this gpu, then compiled and the templates paired now, not while a cuda graph records
def prepare(templates: torch.Tensor) -> None:
    if templates.is_cuda and templates.dtype == torch.float16 and tuple(templates.shape) == (8, 64, 64) and kernels(templates.device) is not None:
        paired_templates(templates)


#template_attention's answer from the CUDA kernel, or None where it takes neither these shapes nor this gpu
def cuda_template_attention(projected: torch.Tensor, templates: torch.Tensor, q_gain: torch.Tensor, k_gain: torch.Tensor, heads: int, kv_heads: int,
                            dq: int, dv: int, templates_count: int, eps: float) -> torch.Tensor | None:
    if not (ENABLED and projected.is_cuda):
        return None
    B, T, _ = projected.shape
    shapes = (projected.dtype == templates.dtype == q_gain.dtype == k_gain.dtype == torch.float16 and heads == 24 and kv_heads == 12 and dv == 32
              and templates_count == 8 and T == 70 and dq in DQS and tuple(templates.shape) == (8, 64, 64))
    found = kernels(projected.device) if shapes else None
    if found is None or not found.takes(B):
        return None
    paired = paired_templates(templates)
    projected, q_gain, k_gain = projected.contiguous(), q_gain.contiguous(), k_gain.contiguous()
    #the kernel's vector loads
    if paired is None or projected.data_ptr() % 16:
        return None
    out = torch.empty(B, T, heads * dv, device=projected.device, dtype=projected.dtype)
    found.launch(projected, paired, q_gain, k_gain, out, dq, eps)
    return out

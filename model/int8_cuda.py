#the int8 matmuls of small batches (the engine's 1 to 8 boards) as CUDA kernels: compiled once per process by NVRTC
#(the copy torch ships) and launched through the CUDA driver on torch's stream, so inside its cuda graphs too. A CTA
#splits K over its warps and adds their int32 sums in shared memory (integer atomics: exact in any order), then runs
#int8_linear_kernel's epilogue with the same fp32 operations: the answers are int8_linear's bits. Used only on the gpus
#CONFIGS was timed on; int8_linear's Triton kernel does the rest, and everything when the kernels do not compile
import ctypes
import sys
import warnings

import torch

SOURCE = r"""
typedef unsigned int u32;
typedef unsigned short u16;
#define DEV __device__ __forceinline__

DEV void mma(int* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
DEV float half_to_float(u16 h) { float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }
DEV u16 float_to_half(float f) { u16 h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h; }
//as Triton lowers tl.sigmoid
DEV float ex2_approx(float x) { float y; asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float div_full(float a, float b) { float y; asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }

//out[M, N] = epilogue(x[M, K] @ w[N, K]^T), x and w int8. A CTA takes MT m16 tiles x WN * NT n8 tiles: a warp NT of
//them, WN warps across and WK warps down K, each CK 64-deep chunks. Thread t of a quad loads 16 contiguous bytes of a
//row's chunk for both operands (the chunk's k order permuted alike in x and w, which leaves the int32 sums unchanged).
//EPI: 0 bias, 1 silu(gate) * up (w's rows: gate then up), 2 residual, 3 residual then the next block's input mix. The
//weights are read before the wait for the previous kernel
template <int MT, int NT, int WN, int WK, int CK, int EPI>
__global__ void __launch_bounds__(WN * WK * 32) small_int8(
    const signed char* __restrict__ x, const float* __restrict__ x_scale, const signed char* __restrict__ w,
    const float* __restrict__ w_scale, const u16* __restrict__ bias, const u16* __restrict__ residual,
    const u16* __restrict__ original, const u16* __restrict__ residual_scale, const u16* __restrict__ input_scale,
    u16* __restrict__ out, int M, int N, int K)
{
    constexpr int G = EPI == 1 ? 2 : 1;
    constexpr int BM = MT * 16, BN = WN * NT * 8, THREADS = WN * WK * 32;
    //a row of partial sums padded against bank conflicts
    constexpr int ROW = BN + 4;
    constexpr int GROUPS = BN / 4, VECTORS = BM * GROUPS, ROUNDS = (VECTORS + THREADS - 1) / THREADS;
    __shared__ __align__(16) int partial[G * BM * ROW];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int kw = warp % WK, nw = warp / WK;
    const int n0 = blockIdx.x * BN, m0 = blockIdx.y * BM;
    const int k0 = kw * CK * 64 + t * 16;

    uint4 wv[CK][G][NT];
    #pragma unroll
    for (int c = 0; c < CK; c++)
        #pragma unroll
        for (int h = 0; h < G; h++)
            #pragma unroll
            for (int j = 0; j < NT; j++) {
                int col = min(n0 + nw * NT * 8 + j * 8 + g, N - 1);
                wv[c][h][j] = __ldg(reinterpret_cast<const uint4*>(w + (size_t)(h * N + col) * K + k0 + c * 64));
            }
    //each thread's groups of 4 outputs, their weight scales and bias
    int rows_[ROUNDS], cols_[ROUNDS];
    bool valid[ROUNDS];
    float4 ws[ROUNDS][G];
    uint2 extra[ROUNDS], mix[ROUNDS];
    #pragma unroll
    for (int k = 0; k < ROUNDS; k++) {
        const int v = threadIdx.x + k * THREADS;
        rows_[k] = v / GROUPS;
        cols_[k] = (v % GROUPS) * 4;
        valid[k] = v < VECTORS && m0 + rows_[k] < M && n0 + cols_[k] < N;
        #pragma unroll
        for (int h = 0; h < G; h++)
            ws[k][h] = valid[k] ? __ldg(reinterpret_cast<const float4*>(w_scale + h * N + n0 + cols_[k])) : make_float4(0, 0, 0, 0);
        if (EPI == 0)
            extra[k] = valid[k] ? __ldg(reinterpret_cast<const uint2*>(bias + n0 + cols_[k])) : make_uint2(0, 0);
    }
    for (int i = threadIdx.x * 4; i < G * BM * ROW; i += THREADS * 4)
        *reinterpret_cast<int4*>(partial + i) = make_int4(0, 0, 0, 0);

    asm volatile("griddepcontrol.wait;" ::: "memory");
    //rows past M read row M - 1: their sums are never stored
    uint4 xv[CK][MT][2];
    #pragma unroll
    for (int c = 0; c < CK; c++)
        #pragma unroll
        for (int i = 0; i < MT; i++)
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                int row = min(m0 + i * 16 + half * 8 + g, M - 1);
                xv[c][i][half] = *reinterpret_cast<const uint4*>(x + (size_t)row * K + k0 + c * 64);
            }
    float xs[ROUNDS];
    #pragma unroll
    for (int k = 0; k < ROUNDS; k++) {
        const size_t offset = (size_t)(m0 + rows_[k]) * N + n0 + cols_[k];
        xs[k] = valid[k] ? x_scale[m0 + rows_[k]] : 0.0f;
        if (EPI >= 2)
            extra[k] = valid[k] ? *reinterpret_cast<const uint2*>(residual + offset) : make_uint2(0, 0);
        if (EPI == 3)
            mix[k] = valid[k] ? *reinterpret_cast<const uint2*>(original + offset) : make_uint2(0, 0);
    }
    //the zeroed partial sums
    __syncthreads();

    int acc[G][MT][NT][4];
    #pragma unroll
    for (int h = 0; h < G; h++)
        #pragma unroll
        for (int i = 0; i < MT; i++)
            #pragma unroll
            for (int j = 0; j < NT; j++)
                acc[h][i][j][0] = acc[h][i][j][1] = acc[h][i][j][2] = acc[h][i][j][3] = 0;
    #pragma unroll
    for (int c = 0; c < CK; c++)
        #pragma unroll
        for (int step = 0; step < 2; step++)
            #pragma unroll
            for (int i = 0; i < MT; i++) {
                u32 a0 = step ? xv[c][i][0].z : xv[c][i][0].x, a2 = step ? xv[c][i][0].w : xv[c][i][0].y;
                u32 a1 = step ? xv[c][i][1].z : xv[c][i][1].x, a3 = step ? xv[c][i][1].w : xv[c][i][1].y;
                #pragma unroll
                for (int h = 0; h < G; h++)
                    #pragma unroll
                    for (int j = 0; j < NT; j++) {
                        u32 b0 = step ? wv[c][h][j].z : wv[c][h][j].x, b1 = step ? wv[c][h][j].w : wv[c][h][j].y;
                        mma(acc[h][i][j], a0, a1, a2, a3, b0, b1);
                    }
            }
    #pragma unroll
    for (int h = 0; h < G; h++)
        #pragma unroll
        for (int i = 0; i < MT; i++)
            #pragma unroll
            for (int j = 0; j < NT; j++) {
                int* p = partial + (h * BM + i * 16 + g) * ROW + nw * NT * 8 + j * 8 + t * 2;
                if (WK > 1) {
                    atomicAdd(p, acc[h][i][j][0]);
                    atomicAdd(p + 1, acc[h][i][j][1]);
                    atomicAdd(p + 8 * ROW, acc[h][i][j][2]);
                    atomicAdd(p + 8 * ROW + 1, acc[h][i][j][3]);
                } else {
                    *reinterpret_cast<int2*>(p) = make_int2(acc[h][i][j][0], acc[h][i][j][1]);
                    *reinterpret_cast<int2*>(p + 8 * ROW) = make_int2(acc[h][i][j][2], acc[h][i][j][3]);
                }
            }
    __syncthreads();

    //int8_linear_kernel's epilogue: (sum * x_scale) * w_scale, then fma with the bias or residual, rounded as it rounds
    #pragma unroll
    for (int k = 0; k < ROUNDS; k++) {
        if (!valid[k]) continue;
        int4 sums[G];
        #pragma unroll
        for (int h = 0; h < G; h++)
            sums[h] = *reinterpret_cast<int4*>(partial + (h * BM + rows_[k]) * ROW + cols_[k]);
        const int sum_gate[4] = {sums[0].x, sums[0].y, sums[0].z, sums[0].w};
        const int sum_up[4] = {sums[G - 1].x, sums[G - 1].y, sums[G - 1].z, sums[G - 1].w};
        const float scale_gate[4] = {ws[k][0].x, ws[k][0].y, ws[k][0].z, ws[k][0].w};
        const float scale_up[4] = {ws[k][G - 1].x, ws[k][G - 1].y, ws[k][G - 1].z, ws[k][G - 1].w};
        const u16 extras[4] = {(u16)(extra[k].x & 0xffff), (u16)(extra[k].x >> 16), (u16)(extra[k].y & 0xffff), (u16)(extra[k].y >> 16)};
        const u16 mixes[4] = {(u16)(mix[k].x & 0xffff), (u16)(mix[k].x >> 16), (u16)(mix[k].y & 0xffff), (u16)(mix[k].y >> 16)};
        u16 result[4];
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            float value = __fmul_rn((float)sum_gate[e], xs[k]);
            if (EPI == 0) {
                value = __fmaf_rn(value, scale_gate[e], half_to_float(extras[e]));
            } else if (EPI == 1) {
                const float gate = __fmul_rn(value, scale_gate[e]);
                const float up = __fmul_rn(__fmul_rn((float)sum_up[e], xs[k]), scale_up[e]);
                const float sigmoid = div_full(1.0f, __fadd_rn(ex2_approx(__fmul_rn(gate, -1.44269502162933349609375f)), 1.0f));
                value = __fmul_rn(__fmul_rn(gate, sigmoid), up);
            } else {
                value = __fmaf_rn(value, scale_gate[e], half_to_float(extras[e]));
                if (EPI == 3) {
                    const float stream = half_to_float(float_to_half(value));
                    value = __fmaf_rn(stream, half_to_float(residual_scale[0]), __fmul_rn(half_to_float(mixes[e]), half_to_float(input_scale[0])));
                }
            }
            result[e] = float_to_half(value);
        }
        *reinterpret_cast<uint2*>(out + (size_t)(m0 + rows_[k]) * N + n0 + cols_[k]) =
            make_uint2(result[0] | ((u32)result[1] << 16), result[2] | ((u32)result[3] << 16));
    }
}
"""

#(MT, NT, WN, WK, CK) by (compute capability, multiprocessors) and kind, by the most rows each takes: the fastest of a
#sweep at 70 to 560 rows on an RTX 5090, kept where it beat int8_linear's best Triton tile (16 to 128 rows) in the
#network too. On the RTX 5070 Ti (70) it only matched them
CONFIGS = {(12, 0, 170): {"projection": [(70, (3, 4, 1, 12, 1)), (140, (3, 4, 1, 12, 1))],
                          "gate_up": [(70, (3, 2, 1, 12, 1)), (140, (3, 2, 2, 6, 2))],
                          "down": [(70, (3, 3, 1, 8, 2)), (140, (3, 4, 1, 8, 2))]}}
EPILOGUES = {"projection": 0, "gate_up": 1, "down": 2, "mixing down": 3}
#off: int8_linear's Triton kernel everywhere (the tests of that kernel)
ENABLED = True

CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6


class LaunchAttribute(ctypes.Structure):
    _fields_ = [("id", ctypes.c_int), ("pad", ctypes.c_char * 4), ("value", ctypes.c_uint * 16)]


class LaunchConfig(ctypes.Structure):
    _fields_ = [("grid_x", ctypes.c_uint), ("grid_y", ctypes.c_uint), ("grid_z", ctypes.c_uint), ("block_x", ctypes.c_uint),
                ("block_y", ctypes.c_uint), ("block_z", ctypes.c_uint), ("shared_bytes", ctypes.c_uint), ("stream", ctypes.c_void_p),
                ("attributes", ctypes.POINTER(LaunchAttribute)), ("attribute_count", ctypes.c_uint)]


def kernel_name(config: tuple, epilogue: int) -> str:
    return f"small_int8<{', '.join(str(value) for value in config)}, {epilogue}>"


def nvrtc_library() -> ctypes.CDLL:
    major = torch.version.cuda.split(".")[0]
    names = [f"nvrtc64_{major}0_0.dll"] if sys.platform == "win32" else [f"libnvrtc.so.{major}", "libnvrtc.so"]
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
    raise OSError(f"none of {names} loads")


#source -> cubin for sm_<capability>, and each name expression's lowered (mangled) name
def compile_cubin(source: str, names: list[str], capability: tuple[int, int]) -> tuple[bytes, dict[str, str]]:
    nvrtc = nvrtc_library()
    nvrtc.nvrtcGetErrorString.restype = ctypes.c_char_p

    def check(result: int) -> None:
        if result != 0:
            raise RuntimeError(f"NVRTC: {nvrtc.nvrtcGetErrorString(result).decode()}")
    program = ctypes.c_void_p()
    check(nvrtc.nvrtcCreateProgram(ctypes.byref(program), source.encode(), b"int8_cuda.cu", 0, None, None))
    try:
        for name in names:
            check(nvrtc.nvrtcAddNameExpression(program, name.encode()))
        options = [f"--gpu-architecture=sm_{capability[0]}{capability[1]}".encode(), b"-std=c++17"]
        if nvrtc.nvrtcCompileProgram(program, len(options), (ctypes.c_char_p * len(options))(*options)) != 0:
            size = ctypes.c_size_t()
            nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(size))
            log = ctypes.create_string_buffer(size.value)
            nvrtc.nvrtcGetProgramLog(program, log)
            raise RuntimeError(f"NVRTC compilation failed:\n{log.value.decode(errors='replace')}")
        size = ctypes.c_size_t()
        check(nvrtc.nvrtcGetCUBINSize(program, ctypes.byref(size)))
        cubin = ctypes.create_string_buffer(size.value)
        check(nvrtc.nvrtcGetCUBIN(program, cubin))
        lowered = {}
        for name in names:
            pointer = ctypes.c_char_p()
            check(nvrtc.nvrtcGetLoweredName(program, name.encode(), ctypes.byref(pointer)))
            lowered[name] = pointer.value.decode()
        return cubin.raw, lowered
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(program))


class Kernels:
    #the configs of one gpu, compiled and loaded into torch's context on it
    def __init__(self, device: torch.device, configs: dict[str, list[tuple[int, tuple]]]):
        self.configs = configs
        self.driver = ctypes.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
        self.driver.cuLaunchKernelEx.argtypes = [ctypes.POINTER(LaunchConfig), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        wanted = sorted({(config, EPILOGUES[epilogue]) for kind, table in configs.items() for _, config in table
                         for epilogue in (["down", "mixing down"] if kind == "down" else [kind])})
        names = [kernel_name(config, epilogue) for config, epilogue in wanted]
        cubin, lowered = compile_cubin(SOURCE, names, torch.cuda.get_device_capability(device))
        with torch.cuda.device(device):
            torch.empty(0, device=device)
            self.module = ctypes.c_void_p()
            self.check(self.driver.cuModuleLoadData(ctypes.byref(self.module), cubin))
        self.functions = {}
        for config, epilogue in wanted:
            function = ctypes.c_void_p()
            self.check(self.driver.cuModuleGetFunction(ctypes.byref(function), self.module, lowered[kernel_name(config, epilogue)].encode()))
            self.functions[config, epilogue] = function

    def check(self, result: int) -> None:
        if result != 0:
            message = ctypes.c_char_p()
            self.driver.cuGetErrorString(result, ctypes.byref(message))
            raise RuntimeError(f"CUDA driver: {message.value.decode() if message.value else result}")

    #tensors: the kernel's pointer arguments in order (None for unused ones); with programmatic dependent launch
    def launch(self, config: tuple, epilogue: int, tensors: list, M: int, N: int, K: int) -> None:
        mt, nt, wn, wk, ck = config
        attribute = (LaunchAttribute * 1)()
        attribute[0].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
        attribute[0].value[0] = 1
        stream = torch.cuda.current_stream(tensors[0].device).cuda_stream
        launch = LaunchConfig(-(-N // (wn * nt * 8)), -(-M // (mt * 16)), 1, wn * wk * 32, 1, 1, 0, stream, attribute, 1)
        arguments = [ctypes.c_void_p(tensor.data_ptr() if tensor is not None else 0) for tensor in tensors]
        arguments += [ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)]
        pointers = (ctypes.c_void_p * len(arguments))(*[ctypes.cast(ctypes.byref(argument), ctypes.c_void_p) for argument in arguments])
        self.check(self.driver.cuLaunchKernelEx(ctypes.byref(launch), self.functions[config, epilogue], pointers, None))


#Kernels by gpu index, or None where they are not used
KERNELS: dict[int, Kernels | None] = {}


def kernels(device: torch.device) -> Kernels | None:
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in KERNELS:
        KERNELS[index] = None
        configs = CONFIGS.get((*torch.cuda.get_device_capability(index), torch.cuda.get_device_properties(index).multi_processor_count))
        if configs is not None:
            try:
                KERNELS[index] = Kernels(torch.device("cuda", index), configs)
            except (OSError, RuntimeError, AttributeError) as error:
                warnings.warn(f"the small-batch int8 CUDA kernels are off, int8_linear's Triton kernel runs instead: {error}")
    return KERNELS[index]


#int8_linear's answer from a CUDA kernel, or None when none is set for this gpu, kind and number of rows
def small_int8_linear(x: torch.Tensor, x_scale: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor | None,
                      residual: torch.Tensor | None, gated: bool, original: torch.Tensor | None, residual_scale: torch.Tensor | None,
                      input_scale: torch.Tensor | None) -> torch.Tensor | None:
    if not ENABLED or not x.is_cuda:
        return None
    found = kernels(x.device)
    if found is None:
        return None
    rows, K = weight.shape
    N = rows // 2 if gated else rows
    M = x.numel() // K
    kind = "gate_up" if gated else "down" if residual is not None else "projection"
    config = next((config for largest, config in found.configs.get(kind, ()) if M <= largest), None)
    #a config splits K exactly, and the epilogue reads and writes 4 columns at once
    if config is None or K != config[3] * config[4] * 64 or N % 4:
        return None
    out = torch.empty(*x.shape[:-1], N, device=x.device, dtype=torch.float16)
    epilogue = EPILOGUES["mixing down" if original is not None else kind]
    tensors = [x.contiguous(), x_scale.contiguous(), weight.contiguous(), weight_scale.contiguous(), bias,
               residual.contiguous() if residual is not None else None, original.contiguous() if original is not None else None, residual_scale,
               input_scale, out]
    #the kernel's vector loads: 16 bytes of x, w and the weight scales, 8 of the bias, residual and original
    if any(tensor is not None and tensor.data_ptr() % 16 for tensor in tensors[:8]):
        return None
    found.launch(config, epilogue, tensors, M, N, K)
    return out

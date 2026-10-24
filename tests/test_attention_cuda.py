#the template attention as a CUDA kernel (model/attention_cuda.py), against the attention in fp32 and the Triton kernels it
#stands in for
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from model import attention_cuda

HEADS, KV_HEADS, DV, TEMPLATES, TOKENS, EPS = 24, 12, 32, 8, 70, 1e-5

#the gpu the kernel was written for: its compute capability
sm_120 = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0), reason="no sm_120 gpu")


def gpu_key() -> tuple[int, int, int]:
    return (*torch.cuda.get_device_capability(), torch.cuda.get_device_properties(0).multi_processor_count)


#this gpu timed for every number of positions, with fresh kernels
@pytest.fixture
def timed_here(monkeypatch):
    monkeypatch.setattr(attention_cuda, "TIMED", {gpu_key(): (1, 0.0)})
    monkeypatch.setattr(attention_cuda, "KERNELS", {})


def operands(positions: int, dq: int, seed: int = 0) -> tuple:
    torch.manual_seed(seed)
    projected = torch.randn(positions, TOKENS, HEADS * dq + KV_HEADS * dq + KV_HEADS * DV + HEADS * TEMPLATES, device="cuda").half()
    templates = (torch.randn(TEMPLATES, 64, 64, device="cuda") / 2).half()
    q_gain, k_gain = (1 + torch.randn(2, dq, device="cuda") / 10).half()
    return projected, templates, q_gain, k_gain, HEADS, KV_HEADS, dq, DV, TEMPLATES, EPS


def cuda_attention(arguments: tuple) -> torch.Tensor:
    found = attention_cuda.cuda_template_attention(*arguments)
    assert found is not None
    return found


def triton_attention(monkeypatch, arguments: tuple) -> torch.Tensor:
    from model.fused_blocks import template_attention
    with monkeypatch.context() as patch:
        patch.setattr(attention_cuda, "ENABLED", False)
        return template_attention(*arguments)


def reference(arguments: tuple) -> torch.Tensor:
    projected, templates, q_gain, k_gain, heads, kv_heads, dq, dv, templates_count, eps = arguments
    B, T, _ = projected.shape
    q, k, v, c = projected.float().split([heads * dq, kv_heads * dq, kv_heads * dv, heads * templates_count], dim=-1)
    q = F.rms_norm(q.reshape(B, T, heads, dq), (dq,), q_gain.float(), eps).transpose(1, 2)
    k = F.rms_norm(k.reshape(B, T, kv_heads, dq), (dq,), k_gain.float(), eps).transpose(1, 2).repeat_interleave(heads // kv_heads, 1)
    v = v.reshape(B, T, kv_heads, dv).transpose(1, 2).repeat_interleave(heads // kv_heads, 1)
    bias = torch.einsum("biht,tij->bhij", c[:, :64].reshape(B, 64, heads, templates_count), templates.float())
    scores = q @ k.transpose(-2, -1) / math.sqrt(dq) + F.pad(bias, (0, T - 64, 0, T - 64))
    return (torch.softmax(scores, -1) @ v).transpose(1, 2).reshape(B, T, heads * dv)


#the kernel computes the attention in fp32 to fp16's rounding at every query width of the d768 and for numbers of positions
#that leave a CTA's slots part empty (1, 5), fill them (64) and spill into another wave (100)
@sm_120
@pytest.mark.parametrize("dq", [8, 16, 24])
@pytest.mark.parametrize("positions", [1, 5, 37, 64, 100])
def test_the_cuda_attention_is_the_attention_in_fp32(timed_here, dq, positions):
    arguments = operands(positions, dq)
    out = cuda_attention(arguments)
    expected = reference(arguments)
    assert out.dtype == torch.float16 and out.shape == expected.shape
    assert torch.allclose(out.float(), expected, atol=1e-2)
    assert not torch.isnan(out).any()


#and what the Triton kernels give, to a few fp16 steps: a change of what the engine plays only through them
@sm_120
@pytest.mark.parametrize("dq", [8, 16, 24])
def test_the_cuda_attention_follows_the_triton_kernels(timed_here, monkeypatch, dq):
    arguments = operands(64, dq, seed=1)
    out = cuda_attention(arguments)
    assert (out.float() - triton_attention(monkeypatch, arguments).float()).abs().max() < 4e-3


#the template bias is a square's 8 coefficients times its 8 template rows: with every coefficient zero the attention is the
#plain one, and templates of zeros are the same
@sm_120
def test_the_cuda_attention_reads_the_templates_and_their_coefficients(timed_here):
    projected, templates, *rest = operands(8, 16)
    without_bias = cuda_attention((projected, torch.zeros_like(templates), *rest))
    mixed = projected.clone()
    mixed[..., -HEADS * TEMPLATES:] = 0
    assert torch.equal(without_bias, cuda_attention((mixed, templates, *rest)))
    assert (cuda_attention((projected, templates, *rest)).float() - without_bias.float()).abs().max() > 0.05


#recorded in cuda graphs on two streams and replayed at once, the kernel gives eager calls' bits every time: nothing it reads
#or writes is shared between launches
@sm_120
def test_the_cuda_attention_in_cuda_graphs(timed_here):
    arguments = operands(64, 24, seed=2)
    expected = cuda_attention(arguments)
    streams = [torch.cuda.Stream() for _ in range(2)]
    graphs, answers = [], []
    for stream in streams:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            answers.append(cuda_attention(arguments))
        graphs.append(graph)
    mismatches = torch.zeros(2, dtype=torch.int64, device="cuda")
    torch.cuda.synchronize()
    for _ in range(200):
        for lane, (graph, stream, answer) in enumerate(zip(graphs, streams, answers)):
            with torch.cuda.stream(stream):
                graph.replay()
                mismatches[lane] += (answer != expected).any()
    torch.cuda.synchronize()
    assert mismatches.tolist() == [0, 0]


#the number of positions that take the kernel on a gpu timed with (fewest, least_fill): at 70 multiprocessors (140 slots)
#the 36 to 46 positions that fill a wave, and the 52 and more that fill more than the least of a second, not the 47 to 51
#whose last wave is a few CTAs
def test_positions_take_the_kernel_by_the_waves_they_fill():
    here = SimpleNamespace(fewest_positions=36, least_fill=0.55, slots=140)
    taken = [positions for positions in range(1, 260) if attention_cuda.Kernels.takes(here, positions)]
    assert taken == [*range(36, 47), *range(52, 260)]
    #a wave of exactly the slots is whole
    exact = SimpleNamespace(fewest_positions=1, least_fill=0.95, slots=3 * attention_cuda.CTAS_PER_POSITION)
    assert [positions for positions in range(1, 12) if attention_cuda.Kernels.takes(exact, positions)] == [3, 6, 9]


#gpus the kernel was not timed on never compile it and give the Triton kernels' work: other compute capabilities (the
#A100's sm_80, sm_86, sm_89) with this gpu's multiprocessors, and this capability with another number of them
@sm_120
@pytest.mark.parametrize("untimed", [(8, 0), (8, 6), (8, 9), "multiprocessors"])
def test_untimed_gpus_never_use_the_cuda_attention(monkeypatch, untimed):
    def compiling(*arguments):
        raise AssertionError("compiled")
    monkeypatch.setattr(attention_cuda.int8_cuda, "compile_cubin", compiling)
    monkeypatch.setattr(attention_cuda, "KERNELS", {})
    if untimed == "multiprocessors":
        monkeypatch.setattr(attention_cuda, "TIMED", {(*gpu_key()[:2], gpu_key()[2] + 1): (1, 0.0)})
    else:
        monkeypatch.setattr(attention_cuda, "TIMED", {(12, 0, gpu_key()[2]): (1, 0.0)})
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: untimed)
    assert attention_cuda.cuda_template_attention(*operands(64, 16)) is None
    assert attention_cuda.kernels(torch.device("cuda")) is None


#when NVRTC fails the kernel is off with a warning, and template_attention runs its Triton kernels
@sm_120
def test_the_triton_kernels_run_when_the_cuda_attention_does_not_compile(timed_here, monkeypatch):
    from model.fused_blocks import template_attention

    def failing(*arguments):
        raise RuntimeError("NVRTC compilation failed")
    monkeypatch.setattr(attention_cuda.int8_cuda, "compile_cubin", failing)
    arguments = operands(64, 16)
    with pytest.warns(UserWarning, match="Triton kernels run instead"):
        out = template_attention(*arguments)
    assert torch.equal(out, triton_attention(monkeypatch, arguments))


#shapes and dtypes it does not take go to the Triton kernels: other head counts, an fp32 input, an input past a 16 byte
#boundary (the kernel's vector loads)
@sm_120
def test_the_cuda_attention_leaves_other_inputs_to_the_triton_kernels(timed_here):
    projected, templates, q_gain, k_gain, heads, kv_heads, dq, dv, templates_count, eps = operands(64, 16)
    assert attention_cuda.cuda_template_attention(projected, templates, q_gain, k_gain, heads, kv_heads, dq, dv, templates_count, eps) is not None
    four = torch.randn(64, TOKENS, 4 * 16 + 4 * 16 + 4 * DV + 4 * TEMPLATES, device="cuda").half()
    assert attention_cuda.cuda_template_attention(four, templates, q_gain, k_gain, 4, 4, 16, DV, TEMPLATES, eps) is None
    assert attention_cuda.cuda_template_attention(projected.float(), templates.float(), q_gain.float(), k_gain.float(), heads, kv_heads, dq, dv,
                                                  templates_count, eps) is None
    #two halves past the start of a buffer
    shifted = torch.cat([projected.new_zeros(2), projected.flatten()])[2:].view_as(projected)
    assert shifted.data_ptr() % 16 and torch.equal(shifted, projected)
    assert attention_cuda.cuda_template_attention(shifted, templates, q_gain, k_gain, heads, kv_heads, dq, dv, templates_count, eps) is None


#the templates the kernel reads are the model's rearranged: [square][key % 8][template // 2][key // 8] of pairs of
#templates, made once for as long as the tensor is unchanged and again after an update of it, not while a graph records
@sm_120
def test_paired_templates_are_the_templates_rearranged_and_follow_them(monkeypatch):
    monkeypatch.setattr(attention_cuda, "PAIRED", {})
    templates = (torch.randn(TEMPLATES, 64, 64, device="cuda") / 2).half()
    paired = attention_cuda.paired_templates(templates)
    assert paired is attention_cuda.paired_templates(templates)

    halves = paired.view(64, 8, 4, 8, 2)
    for square, key, template in [(0, 0, 0), (3, 17, 5), (63, 63, 7), (40, 9, 2)]:
        assert halves[square, key % 8, template // 2, key // 8, template % 2] == templates[template, square, key]

    templates.add_(1)
    updated = attention_cuda.paired_templates(templates)
    assert updated is not paired and halves.shape == updated.view(64, 8, 4, 8, 2).shape
    assert updated.view(64, 8, 4, 8, 2)[3, 1, 2, 4, 1] == templates[5, 3, 33]

    #a model made in torch.inference_mode has tensors that count no updates
    with torch.inference_mode():
        fixed = (torch.randn(TEMPLATES, 64, 64, device="cuda") / 2).half()
    assert attention_cuda.paired_templates(fixed) is attention_cuda.paired_templates(fixed)

    other = (torch.randn(TEMPLATES, 64, 64, device="cuda") / 2).half()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.zeros(1, device="cuda")
        assert attention_cuda.paired_templates(other) is None


#models loaded on threads (--gpus) pair their templates at once: every one is kept up to the most, the oldest go after
@sm_120
def test_templates_paired_on_threads_are_all_kept_up_to_the_most(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.setattr(attention_cuda, "PAIRED", {})
    monkeypatch.setattr(attention_cuda, "PAIRED_MOST", 6)
    many = [(torch.randn(TEMPLATES, 64, 64, device="cuda") / 2).half() for _ in range(10)]
    with ThreadPoolExecutor(4) as pool:
        paired = list(pool.map(attention_cuda.paired_templates, many))
    assert len(attention_cuda.PAIRED) == 6
    assert all(attention_cuda.paired_templates(templates) is not None for templates in many)
    assert all(torch.equal(found.view(64, 8, 4, 8, 2)[5, 3, 1, 6, 0], templates[2, 5, 6 * 8 + 3]) for found, templates in zip(paired, many))


#the fused model pairs its templates and compiles the kernels when it is built, not while a cuda graph records them
@sm_120
def test_building_the_fused_model_prepares_the_cuda_attention(timed_here, monkeypatch):
    from model.fused_blocks import FusedTransformer
    from model.model import ChessTransformer
    monkeypatch.setattr(attention_cuda, "PAIRED", {})
    model = ChessTransformer(d_model=128, num_layers=2, n_heads=4, kv_heads=2, qk_ratio=[0.25, 0.5], mlp_hidden=320, value_bins=16).cuda().eval()
    fused = FusedTransformer(model.half_except_heads(torch.float16))
    assert attention_cuda.PAIRED[fused.templates.data_ptr()][0] is fused.templates
    assert attention_cuda.KERNELS[torch.cuda.current_device()] is not None

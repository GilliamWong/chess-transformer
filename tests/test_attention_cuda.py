import math
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from model import attention_cuda

HEADS, KV_HEADS, DV, TEMPLATES, TOKENS, EPS = 24, 12, 32, 8, 70, 1e-5

sm_80 = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0), reason="no gpu of sm_80 or newer")


def gpu_key() -> tuple[int, int, int]:
    return (*torch.cuda.get_device_capability(), torch.cuda.get_device_properties(0).multi_processor_count)


@pytest.fixture
def timed_here(monkeypatch):
    monkeypatch.setattr(attention_cuda, "TIMED", {gpu_key(): attention_cuda.Gate(1, 0.0)})
    monkeypatch.setattr(attention_cuda, "KERNELS", {})


def random_templates() -> torch.Tensor:
    return (torch.randn(TEMPLATES, 64, 64, device="cuda") / 2).half()


def operands(positions: int, dq: int, seed: int = 0) -> tuple:
    torch.manual_seed(seed)
    projected = torch.randn(positions, TOKENS, HEADS * dq + KV_HEADS * dq + KV_HEADS * DV + HEADS * TEMPLATES, device="cuda").half()
    templates = random_templates()
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


@sm_80
@pytest.mark.parametrize("dq", [8, 16, 24])
@pytest.mark.parametrize("positions", [1, 5, 37, 64, 100])
def test_the_cuda_attention_is_the_attention_in_fp32(timed_here, dq, positions):
    arguments = operands(positions, dq)
    out = cuda_attention(arguments)
    expected = reference(arguments)
    assert out.dtype == torch.float16 and out.shape == expected.shape
    assert torch.allclose(out.float(), expected, atol=1e-2)


@sm_80
@pytest.mark.parametrize("dq", [8, 16, 24])
def test_the_cuda_attention_follows_the_triton_kernels(timed_here, monkeypatch, dq):
    arguments = operands(64, dq, seed=1)
    out = cuda_attention(arguments)
    assert (out.float() - triton_attention(monkeypatch, arguments).float()).abs().max() < 4e-3


#a gpu whose gate changes the phases a CTA takes with the positions gives a position the same bits in every batch. On the others the
#split is not used, and for sm_120 the compiler joins a product and a sum of the two kernels differently: an fp16 rounding step apart
@sm_80
@pytest.mark.parametrize("per_cta", [1, 2, 3])
def test_phases_split_over_ctas_give_the_attention_of_one_cta_for_all(monkeypatch, per_cta):
    own = attention_cuda.TIMED.get(gpu_key())
    splits_here = own is not None and len(own.phases) > 1
    cases = [operands(positions, dq, seed=3) for positions in (1, 37) for dq in (8, 24)]
    monkeypatch.setattr(attention_cuda, "TIMED", {gpu_key(): attention_cuda.Gate(1, 0.0)})
    monkeypatch.setattr(attention_cuda, "KERNELS", {})
    whole = [cuda_attention(arguments) for arguments in cases]
    monkeypatch.setattr(attention_cuda, "TIMED", {gpu_key(): attention_cuda.Gate(1, 0.0, ((math.inf, per_cta),))})
    monkeypatch.setattr(attention_cuda, "KERNELS", {})
    for arguments, expected in zip(cases, whole):
        found = cuda_attention(arguments)
        assert torch.equal(found, expected) if splits_here else (found.float() - expected.float()).abs().max() <= 2e-3


@sm_80
def test_the_cuda_attention_reads_the_templates_and_their_coefficients(timed_here):
    projected, templates, *rest = operands(8, 16)
    without_bias = cuda_attention((projected, torch.zeros_like(templates), *rest))
    mixed = projected.clone()
    mixed[..., -HEADS * TEMPLATES:] = 0
    assert torch.equal(without_bias, cuda_attention((mixed, templates, *rest)))
    assert (cuda_attention((projected, templates, *rest)).float() - without_bias.float()).abs().max() > 0.05


#two graphs replayed at once on two streams must give the eager bits: nothing is shared between launches
@sm_80
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


#at 140 slots: 36 to 46 positions fill a wave, 52 and more fill a second past least_fill, 47 to 51 leave a few CTAs in the last
def test_positions_take_the_kernel_by_the_waves_they_fill():
    takes = attention_cuda.Kernels.takes
    here = SimpleNamespace(fewest_positions=36, least_fill=0.55, slots=140, phases=attention_cuda.ALL_PHASES)
    assert [positions for positions in range(1, 260) if takes(here, positions)] == [*range(36, 47), *range(52, 260)]
    #a wave of exactly the slots is whole
    exact = SimpleNamespace(fewest_positions=1, least_fill=0.95, slots=3 * attention_cuda.CTAS_PER_POSITION, phases=attention_cuda.ALL_PHASES)
    assert [positions for positions in range(1, 12) if takes(exact, positions)] == [3, 6, 9]
    #up to 2 positions, a CTA a phase: 5 times the CTAs
    split = SimpleNamespace(fewest_positions=1, least_fill=0.95, slots=attention_cuda.PHASES * attention_cuda.CTAS_PER_POSITION,
                            phases=((2, 1), (math.inf, attention_cuda.PHASES)))
    assert [positions for positions in range(1, 12) if takes(split, positions)] == [1, 2, 5, 10]


#a gate whose phases stop short of some count of positions would fail that launch
def test_every_gate_has_phases_for_any_count_of_positions():
    assert all(gate.phases[-1][0] == math.inf for gate in attention_cuda.TIMED.values())


@sm_80
@pytest.mark.parametrize("untimed", [(8, 0), (8, 6), (8, 9), "multiprocessors"])
def test_untimed_gpus_never_use_the_cuda_attention(monkeypatch, untimed):
    def compiling(*arguments):
        raise AssertionError("compiled")
    monkeypatch.setattr(attention_cuda.int8_cuda, "compile_cubin", compiling)
    monkeypatch.setattr(attention_cuda, "KERNELS", {})
    if untimed == "multiprocessors":
        monkeypatch.setattr(attention_cuda, "TIMED", {(*gpu_key()[:2], gpu_key()[2] + 1): attention_cuda.Gate(1, 0.0)})
    else:
        monkeypatch.setattr(attention_cuda, "TIMED", {(12, 0, gpu_key()[2]): attention_cuda.Gate(1, 0.0)})
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: untimed)
    assert attention_cuda.cuda_template_attention(*operands(64, 16)) is None
    assert attention_cuda.kernels(torch.device("cuda")) is None


@sm_80
def test_the_triton_kernels_run_when_the_cuda_attention_does_not_compile(timed_here, monkeypatch):
    from model.fused_blocks import template_attention

    def failing(*arguments):
        raise RuntimeError("NVRTC compilation failed")
    monkeypatch.setattr(attention_cuda.int8_cuda, "compile_cubin", failing)
    arguments = operands(64, 16)
    with pytest.warns(UserWarning, match="Triton kernels run instead"):
        out = template_attention(*arguments)
    assert torch.equal(out, triton_attention(monkeypatch, arguments))


@sm_80
def test_the_cuda_attention_leaves_other_inputs_to_the_triton_kernels(timed_here):
    arguments = operands(64, 16)
    projected, templates, q_gain, k_gain, *rest = arguments
    cuda_attention(arguments)
    four = torch.randn(64, TOKENS, 4 * 16 + 4 * 16 + 4 * DV + 4 * TEMPLATES, device="cuda").half()
    assert attention_cuda.cuda_template_attention(four, templates, q_gain, k_gain, 4, 4, 16, DV, TEMPLATES, EPS) is None
    in_fp32 = [tensor.float() for tensor in arguments[:4]]
    assert attention_cuda.cuda_template_attention(*in_fp32, *rest) is None
    #two halves past a buffer's start: not 16 byte aligned, which the kernel's vector loads need
    shifted = torch.cat([projected.new_zeros(2), projected.flatten()])[2:].view_as(projected)
    assert shifted.data_ptr() % 16 and torch.equal(shifted, projected)
    assert attention_cuda.cuda_template_attention(shifted, templates, q_gain, k_gain, *rest) is None


#layout [square][key % 8][template // 2][key // 8] of template pairs
@sm_80
def test_paired_templates_are_the_templates_rearranged_and_follow_them(monkeypatch):
    monkeypatch.setattr(attention_cuda, "PAIRED", {})
    templates = random_templates()
    paired = attention_cuda.paired_templates(templates)
    assert paired is attention_cuda.paired_templates(templates)

    halves = paired.view(64, 8, 4, 8, 2)
    for square, key, template in [(0, 0, 0), (3, 17, 5), (63, 63, 7), (40, 9, 2)]:
        assert halves[square, key % 8, template // 2, key // 8, template % 2] == templates[template, square, key]

    templates.add_(1)
    updated = attention_cuda.paired_templates(templates)
    assert updated is not paired
    assert updated.view(64, 8, 4, 8, 2)[3, 1, 2, 4, 1] == templates[5, 3, 33]

    #a model made in torch.inference_mode has tensors that count no updates
    with torch.inference_mode():
        fixed = random_templates()
    assert attention_cuda.paired_templates(fixed) is attention_cuda.paired_templates(fixed)

    other = random_templates()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.zeros(1, device="cuda")
        assert attention_cuda.paired_templates(other) is None


@sm_80
def test_templates_paired_on_threads_are_all_kept_up_to_the_most(monkeypatch):
    monkeypatch.setattr(attention_cuda, "PAIRED", {})
    monkeypatch.setattr(attention_cuda, "PAIRED_MOST", 6)
    many = [random_templates() for _ in range(10)]
    with ThreadPoolExecutor(4) as pool:
        paired = list(pool.map(attention_cuda.paired_templates, many))
    assert len(attention_cuda.PAIRED) == 6
    assert all(attention_cuda.paired_templates(templates) is not None for templates in many)
    assert all(torch.equal(found.view(64, 8, 4, 8, 2)[5, 3, 1, 6, 0], templates[2, 5, 6 * 8 + 3]) for found, templates in zip(paired, many))


@sm_80
def test_building_the_fused_model_prepares_the_cuda_attention(timed_here, monkeypatch):
    from model.fused_blocks import FusedTransformer
    from model.model import ChessTransformer
    monkeypatch.setattr(attention_cuda, "PAIRED", {})
    model = ChessTransformer(d_model=128, num_layers=2, n_heads=4, kv_heads=2, qk_ratio=[0.25, 0.5], mlp_hidden=320, value_bins=16).cuda().eval()
    fused = FusedTransformer(model.half_except_heads(torch.float16))
    assert attention_cuda.PAIRED[fused.templates.data_ptr()][0] is fused.templates
    assert attention_cuda.KERNELS[torch.cuda.current_device()] is not None

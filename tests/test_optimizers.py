import copy

import pytest
import torch

from training.muon import BatchedMuon
from training.train import WEIGHT_DECAY, MasterWeights, blunder_loss, compile_muon_step, make_optimizers, policy_loss, value_loss
from model.encoding import NO_MOVE
from model.model import ChessTransformer

CPU = torch.device("cpu")
BLOCK_LINEAR_LAYERS = ["Q", "K", "V", "combine_heads", "mlp.up_projection", "mlp.gate_projection", "mlp.down_projection", "template_mix"]


def small_model() -> ChessTransformer:
    torch.manual_seed(0)
    return ChessTransformer(d_model=32, num_layers=2, n_heads=4, position_templates=2, qk_ratio=[0.5, 1.0], kv_heads=2)


def names_in(optimizer: torch.optim.Optimizer, model: ChessTransformer) -> set[str]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return {names[id(parameter)] for group in optimizer.param_groups for parameter in group["params"]}


#every parameter goes to exactly one of the two
def test_muon_takes_the_weight_matrices_inside_the_blocks_and_adamw_the_rest():
    model = small_model()

    adamw, muon = make_optimizers(model, 1e-3, CPU)

    assert isinstance(adamw, torch.optim.AdamW) and isinstance(muon, torch.optim.Muon)
    in_muon, in_adamw = names_in(muon, model), names_in(adamw, model)
    assert in_muon == {f"encoder.{block}.{layer}.weight" for block in range(2) for layer in BLOCK_LINEAR_LAYERS}
    assert in_muon.isdisjoint(in_adamw)
    assert in_muon | in_adamw == {name for name, _ in model.named_parameters()}
    assert {"embed.weight", "position_embed", "templates", "from_projection.weight", "encoder.0.Q.bias", "encoder.0.ln1.weight"} <= in_adamw
    #the same decay and learning rate for both
    assert muon.param_groups[0]["weight_decay"] == adamw.param_groups[0]["weight_decay"] == WEIGHT_DECAY
    assert muon.param_groups[0]["lr"].item() == pytest.approx(adamw.param_groups[0]["lr"])

    #the sweep settings reach both
    adamw, muon = make_optimizers(model, 1e-3, CPU, weight_decay=0.1, muon_momentum=0.9)
    assert muon.param_groups[0]["weight_decay"] == adamw.param_groups[0]["weight_decay"] == 0.1
    assert muon.param_groups[0]["momentum"] == 0.9


#QK-norm's gains set how sharp attention can be and the input skip's scales how big the residual stream is, so weight
#decay must not pull them to zero
def test_the_qk_norm_gains_and_input_skip_scales_are_not_decayed():
    model = small_model()
    adamw, muon = make_optimizers(model, 1e-3, CPU, weight_decay=0.1)

    names = {id(parameter): name for name, parameter in model.named_parameters()}
    undecayed = {names[id(parameter)] for group in adamw.param_groups if group["weight_decay"] == 0 for parameter in group["params"]}
    qk_gains = {f"encoder.{block}.{norm}.weight" for block in range(2) for norm in ("q_norm", "k_norm")}
    skip_scales = {f"encoder.{block}.{scale}" for block in range(2) for scale in ("residual_scale", "input_scale")}
    assert undecayed == qk_gains | skip_scales
    assert names_in(muon, model) | names_in(adamw, model) == set(names.values())


#catches the embeddings left at the shared learning rate, their decay sped up along with it, or the schedule moving
#them apart from the rest
def test_the_embeddings_get_their_own_learning_rate():
    model = small_model()
    adamw, _ = make_optimizers(model, 1e-3, CPU, weight_decay=0.1, embed_lr_multiplier=10)

    names = {id(parameter): name for name, parameter in model.named_parameters()}
    embeddings = next(group for group in adamw.param_groups if any(names[id(parameter)] == "embed.weight" for parameter in group["params"]))
    assert {names[id(parameter)] for parameter in embeddings["params"]} == {"embed.weight", "position_embed", "square_features.table"}
    assert embeddings["lr"] == pytest.approx(1e-2)
    assert embeddings["lr"] * embeddings["weight_decay"] == pytest.approx(1e-3 * 0.1)

    scheduler = torch.optim.lr_scheduler.LambdaLR(adamw, lambda step: 1 / (1 + step))
    for _ in range(3):
        scheduler.step()
    assert embeddings["lr"] == pytest.approx(10 * adamw.param_groups[0]["lr"])


#Muon's learning rate is a tensor; the schedule has to move it just like AdamW's float
def test_the_schedule_moves_both_learning_rates_alike():
    model = small_model()
    optimizers = make_optimizers(model, 1e-3, CPU)
    schedulers = [torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (1 + step)) for optimizer in optimizers]

    for step in range(1, 4):
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        for optimizer in optimizers:
            optimizer.step()
        for scheduler in schedulers:
            scheduler.step()

        adamw_lr, muon_lr = (optimizer.param_groups[0]["lr"] for optimizer in optimizers)
        assert isinstance(muon_lr, torch.Tensor)
        assert adamw_lr == pytest.approx(1e-3 / (1 + step))
        assert muon_lr.item() == pytest.approx(adamw_lr)


#both optimizers really step their weights
def test_training_with_muon_learns_a_batch():
    model = small_model()
    boards = torch.randint(0, 13, (16, 70))
    target_moves = torch.randint(0, 1968, (16, 1))
    target_probabilities = torch.ones(16, 1)
    optimizers = make_optimizers(model, 3e-3, CPU)
    weights_before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    losses = []
    for _ in range(30):
        loss = policy_loss(model(boards)[0], target_moves, target_probabilities)
        losses.append(loss.item())
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

    assert losses[-1] < losses[0] / 2
    for name in ["encoder.0.Q.weight", "encoder.1.mlp.down_projection.weight", "embed.weight", "from_projection.weight"]:
        assert not torch.equal(dict(model.named_parameters())[name], weights_before[name]), name


#a step of train.train's: the bf16 copy's gradients reach the fp32 weights whole, and the copy then has the stepped
#weights. A copy that kept its own weights, or optimizers stepping it directly, would train in bf16 alone
def test_master_weights_step_the_fp32_model_and_the_bf16_copy_follows():
    model = small_model()
    master_weights = MasterWeights(model)
    compute_model = master_weights.compute_model
    assert compute_model.encoder[0].Q.weight.dtype == torch.bfloat16 and compute_model.eval_head[0].weight.dtype == torch.float32
    #from the start too: a resumed run starts from a new copy, whose heads must not be rounded to bf16
    for parameter, compute_parameter in zip(model.parameters(), compute_model.parameters()):
        assert torch.equal(compute_parameter, parameter.to(compute_parameter.dtype))
    optimizers = make_optimizers(model, 3e-3, CPU)
    query_weights_before = model.encoder[0].Q.weight.detach().clone()

    master_weights.zero_grad()
    logits, value_logits = compute_model(torch.randint(0, 13, (16, 70)))
    (policy_loss(logits, torch.randint(0, 1968, (16, 1)), torch.ones(16, 1)) + value_loss(value_logits, torch.rand(16))).backward()
    master_weights.gradients_to_master()
    for parameter, compute_parameter in zip(model.parameters(), compute_model.parameters()):
        assert parameter.grad.dtype == torch.float32 and torch.equal(parameter.grad, compute_parameter.grad.float())
    for optimizer in optimizers:
        optimizer.step()
    master_weights.weights_to_compute()

    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert not torch.equal(model.encoder[0].Q.weight, query_weights_before)
    for parameter, compute_parameter in zip(model.parameters(), compute_model.parameters()):
        assert torch.equal(compute_parameter, parameter.to(compute_parameter.dtype))


def test_the_blunder_loss_is_minus_log_of_the_probability_left_to_other_moves():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.2]]))

    #no blunders: nothing to push off
    assert blunder_loss(logits, torch.tensor([[NO_MOVE, NO_MOVE]])).item() == pytest.approx(0.0, abs=1e-6)
    #moves 1 and 2 are blunders: they hold 0.5 of the probability together
    assert blunder_loss(logits, torch.tensor([[1, 2]])).item() == pytest.approx(-torch.log(torch.tensor(0.5)).item(), rel=1e-5)


def test_the_blunder_loss_pushes_the_blunder_down_and_the_rest_up():
    logits = torch.zeros(1, 4, requires_grad=True)

    blunder_loss(logits, torch.tensor([[2, NO_MOVE]])).backward()

    assert logits.grad[0, 2] > 0
    assert (logits.grad[0, [0, 1, 3]] < 0).all()


#tall, wide and square matrices, two or more of a shape, as the model's blocks have them
MUON_SHAPES = [(24, 40), (40, 24), (32, 32), (24, 40), (40, 24), (32, 32), (8, 40)]
#a weight decay big enough to show in a step
MUON_SETTINGS = dict(weight_decay=0.5, momentum=0.9, adjust_lr_fn="match_rms_adamw")


#torch's Muon and BatchedMuon on the same weights, each with its own copies
def muon_pair(device: torch.device = CPU, nesterov: bool = True) -> list[tuple[torch.optim.Muon, list[torch.nn.Parameter]]]:
    torch.manual_seed(0)
    weights = [torch.randn(shape, device=device) for shape in MUON_SHAPES]
    pair = []
    for optimizer_class in (torch.optim.Muon, BatchedMuon):
        parameters = [torch.nn.Parameter(weight.clone()) for weight in weights]
        pair.append((optimizer_class(parameters, lr=torch.tensor(1e-2, device=device), nesterov=nesterov, **MUON_SETTINGS), parameters))
    return pair


#the same gradients for every list of parameters, three orders of magnitude apart between matrices
def give_same_gradients(parameter_lists: list[list[torch.nn.Parameter]], generator: torch.Generator) -> None:
    for parameters in zip(*parameter_lists):
        gradient = torch.randn(parameters[0].shape, generator=generator) * 10 ** (3 * torch.rand(1, generator=generator).item())
        for parameter in parameters:
            parameter.grad = gradient.to(parameter.device, copy=True)


#how far the weights are from the reference's, against how far the reference moved them from where they started
def assert_moved_alike(expected_parameters, actual_parameters, starts, tolerance: float) -> None:
    for expected, actual, start in zip(expected_parameters, actual_parameters, starts):
        assert (actual - expected).norm() <= tolerance * (expected - start).norm()


#catches an update that differs from torch's by more than bf16 rounding: one norm shared by a shape's matrices (the
#gradients are scaled apart for it), a tall matrix not transposed, the decay or the shape scaling of the learning rate
#left out, a learning rate tensor that the schedule's changes do not reach, a momentum buffer that differs
@pytest.mark.parametrize("nesterov", [True, False])
def test_batched_muon_steps_like_torchs_muon(nesterov):
    (reference, reference_parameters), (batched, batched_parameters) = muon_pair(nesterov=nesterov)
    starts = [parameter.detach().clone() for parameter in reference_parameters]
    generator = torch.Generator().manual_seed(1)

    for step in range(6):
        for optimizer in (reference, batched):
            optimizer.param_groups[0]["lr"].fill_(1e-2 / (1 + step))
        give_same_gradients([reference_parameters, batched_parameters], generator)
        reference.step()
        batched.step()

        assert_moved_alike(reference_parameters, batched_parameters, starts, 1e-2)
    for expected, actual in zip(reference_parameters, batched_parameters):
        assert torch.equal(reference.state[expected]["momentum_buffer"], batched.state[actual]["momentum_buffer"])


#a run resumes from the Muon state it saved: the saved layout is torch's, and either class continues from the other's
def test_batched_muon_and_torchs_muon_load_each_others_saved_state():
    for saving_class, loading_class in ((torch.optim.Muon, BatchedMuon), (BatchedMuon, torch.optim.Muon)):
        torch.manual_seed(0)
        saved_parameters = [torch.nn.Parameter(torch.randn(shape)) for shape in MUON_SHAPES]
        saving = saving_class(saved_parameters, lr=torch.tensor(1e-2), **MUON_SETTINGS)
        generator = torch.Generator().manual_seed(1)
        for _ in range(3):
            give_same_gradients([saved_parameters], generator)
            saving.step()
        state = copy.deepcopy(saving.state_dict())

        loaded_parameters = [torch.nn.Parameter(parameter.detach().clone()) for parameter in saved_parameters]
        loading = loading_class(loaded_parameters, lr=torch.tensor(5e-3), **MUON_SETTINGS)
        loading.load_state_dict(state)
        assert loading.state_dict()["param_groups"][0].keys() == state["param_groups"][0].keys()
        assert all(state_of.keys() == {"momentum_buffer"} for state_of in loading.state_dict()["state"].values())
        assert loading.param_groups[0]["lr"].item() == pytest.approx(1e-2)
        for saved, loaded in zip(saved_parameters, loaded_parameters):
            assert torch.equal(saving.state[saved]["momentum_buffer"], loading.state[loaded]["momentum_buffer"])

        #both go on with the same gradients: without the momentum the steps would differ
        starts = [parameter.detach().clone() for parameter in saved_parameters]
        for _ in range(3):
            give_same_gradients([saved_parameters, loaded_parameters], generator)
            saving.step()
            loading.step()
        assert_moved_alike(saved_parameters, loaded_parameters, starts, 1e-2)


#the compiled step is a cuda graph in the run, and the matrices go through cuBLAS in batches: a different rounding, so the
#tolerance is that of bf16 rather than of the cpu test
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
def test_the_compiled_step_of_batched_muon_matches_torchs_muon_on_the_gpu():
    (reference, reference_parameters), (batched, batched_parameters) = muon_pair(torch.device("cuda"))
    compile_muon_step([batched])
    starts = [parameter.detach().clone() for parameter in reference_parameters]
    generator = torch.Generator().manual_seed(1)

    for step in range(5):
        for optimizer in (reference, batched):
            optimizer.param_groups[0]["lr"].fill_(1e-2 / (1 + step))
        give_same_gradients([reference_parameters, batched_parameters], generator)
        reference.step()
        batched.step()

        assert_moved_alike(reference_parameters, batched_parameters, starts, 5e-2)

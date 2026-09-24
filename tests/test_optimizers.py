import pytest
import torch

from main import WEIGHT_DECAY, make_optimizers, policy_loss
from model import ChessTransformer

CPU = torch.device("cpu")
BLOCK_LINEAR_LAYERS = ["Q", "K", "V", "combine_heads", "mlp.up_projection", "mlp.gate_projection", "mlp.down_projection", "template_mix"]


def small_model() -> ChessTransformer:
    torch.manual_seed(0)
    return ChessTransformer(d_model=32, num_layers=2, n_heads=4, position_templates=2, qk_ratio=[0.5, 1.0], kv_heads=2)


def names_in(optimizer: torch.optim.Optimizer, model: ChessTransformer) -> set[str]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return {names[id(parameter)] for group in optimizer.param_groups for parameter in group["params"]}


#the linear layers inside the blocks go to Muon; the embeddings, the template bank, the heads, the biases and the
#layer norms to AdamW, and every parameter to exactly one of the two
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


#Muon's learning rate is a tensor (the compiled step reads it); the schedule has to move it just like AdamW's float
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


#both optimizers really step their weights: the model learns a fixed batch
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

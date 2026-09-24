"""Real two- and four-process Gloo tests; no GPUs or repository training data required."""

import copy
import json
import time
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import data
import main as single
from encoding import BOARD_VOCAB_SIZE, TOKENS_PER_BOARD
from model import ChessTransformer
from multi_gpu import train_ddp
from tools.budget import Budget
from tools.checkpoint import load_checkpoint


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_shuffled_shards_have_equal_full_batches_and_do_not_overlap(world_size):
    dataset = list(range(67))
    samplers = [train_ddp.TrainingSampler(dataset, 16, rank, world_size, 42) for rank in range(world_size)]
    batches = [[batch.tolist() for batch in sampler] for sampler in samplers]
    shards = [[index for batch in rank_batches for index in batch] for rank_batches in batches]
    assert all(len(rank_batches) == 4 for rank_batches in batches)
    assert all(len(batch) == 16 // world_size for rank_batches in batches for batch in rank_batches)
    assert len({index for shard in shards for index in shard}) == 64
    assert [batch.tolist() for batch in samplers[0]] == batches[0]
    samplers[0].epoch = 1
    assert [batch.tolist() for batch in samplers[0]] != batches[0]


@pytest.mark.parametrize("dataset,batch_size,world_size", [
    (list(range(32)), 7, 2), (list(range(8)), 16, 2),
])
def test_invalid_shards_fail_before_training(dataset, batch_size, world_size):
    with pytest.raises(ValueError):
        train_ddp.TrainingSampler(dataset, batch_size, 0, world_size, 42)


def training_loss(model, batch):
    policy, value = model(batch[0])
    return single.policy_loss(policy, batch[1], batch[2]) + 0.8 * single.value_loss(value, batch[3])


def compare_update(rank, compiled=False):
    torch.manual_seed(123)
    model = ChessTransformer(d_model=16, num_layers=1, n_heads=2, position_templates=2)
    with torch.no_grad():
        model.templates.normal_(std=0.1)
    distributed = DDP(model)
    reference = copy.deepcopy(model)
    if compiled:
        distributed = torch.compile(distributed, backend="aot_eager")
    generator = torch.Generator().manual_seed(18)
    batch = (
        torch.randint(BOARD_VOCAB_SIZE, (8, TOKENS_PER_BOARD), generator=generator),
        torch.arange(16).reshape(8, 2), torch.tensor([[0.7, 0.3]] * 8),
        torch.rand(8, generator=generator),
    )
    world_size = dist.get_world_size()
    local_size = 8 // world_size
    local = tuple(tensor[rank * local_size:(rank + 1) * local_size] for tensor in batch)
    loss = training_loss(distributed, local)
    loss.backward()

    expected = training_loss(reference, batch)
    expected.backward()
    averaged_loss = loss.detach().clone()
    dist.all_reduce(averaged_loss)
    torch.testing.assert_close(averaged_loss / world_size, expected)
    for actual, wanted in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, wanted.grad, rtol=2e-4, atol=2e-6)
    for current in (model, reference):
        nn.utils.clip_grad_norm_(current.parameters(), 1.0)
        torch.optim.SGD(current.parameters(), lr=0.01).step()
    for actual, wanted in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, wanted, rtol=2e-5, atol=2e-6)


def check_shared_clock(rank):
    #the shared budget takes rank zero's clock every CLOCK_EVERY steps and keeps it in between, on every rank alike
    budget = Budget(16, minutes=1)
    clock = Budget(16, minutes=1) if rank == 0 else None
    optimizer = torch.optim.SGD([nn.Parameter(torch.ones(1))], lr=0.1)
    scheduler = train_ddp.make_scheduler(optimizer, budget, "cosine")
    now = [100.0 + 1000 * rank]
    with patch("tools.budget.time.perf_counter", side_effect=lambda: now[0]), patch.object(train_ddp, "CLOCK_EVERY", 2):
        for step, expected_seconds in enumerate((0, 30, 30, 90), start=1):
            now[0] += 30 if rank == 0 else 300  #the other ranks have deliberately different clocks
            optimizer.step()
            train_ddp.step_budget(budget, clock, torch.device("cpu"))
            scheduler.step()
            assert budget.steps == step
            assert budget.positions == 16 * step
            assert budget.seconds == expected_seconds
            assert budget.finished() == (step == 4)
            states = [None] * dist.get_world_size()
            dist.all_gather_object(states, (budget.progress(), scheduler.get_last_lr()))
            assert all(state == states[0] for state in states[1:])
            if step == 2:
                if rank == 0:
                    clock.pause()
                now[0] += 600  #validation is outside the budget
                if rank == 0:
                    clock.resume()


def check_shared_evaluation():
    #the ranks' shares of the val batches add up to main.py's numbers on the same batches, also when some ranks get none
    rows = np.random.default_rng(5)
    size = 11
    val_dataset = data.ChessGamesDataset(
        rows.integers(0, BOARD_VOCAB_SIZE, (size, TOKENS_PER_BOARD)).astype(np.uint8),
        rows.integers(0, 1968, (size, 2)).astype(np.int16), rows.integers(-300, 300, (size, 2)).astype(np.int16),
        single.TARGET_TEMPERATURE)
    _, val_loader = data.datasets_to_dataloaders(val_dataset, val_dataset, val_batch_size=4, seed=0)  #batches of 4, 4 and 3
    torch.manual_seed(7)
    model = ChessTransformer(d_model=16, num_layers=1, n_heads=2, position_templates=2)
    for max_batches in (None, 2):
        expected = single.evaluate_model(model, val_loader, torch.device("cpu"), max_batches)
        assert train_ddp.evaluate(model, val_loader, torch.device("cpu"), max_batches) == pytest.approx(expected, rel=1e-6)


def write_set(folder, name):
    for split, size in (("train", 32), ("val", 10)):
        arrays = dict(
            boards=np.full((size, TOKENS_PER_BOARD), 5, dtype=np.uint8),
            stockfish_moves=np.tile(np.array([0, 1], dtype=np.int16), (size, 1)),
            stockfish_evals=np.tile(np.array([120, -60], dtype=np.int16), (size, 1)),
        )
        for field, array in arrays.items():
            (folder / f"{name}_{split}_{field}.bin").write_bytes(array.tobytes())


def distributed_worker(rank, folder, world_size):
    torch.set_num_threads(1)
    folder = Path(folder)
    dist.init_process_group("gloo", init_method=(folder / "rendezvous").as_uri(),
                            rank=rank, world_size=world_size, timeout=timedelta(seconds=90))
    try:
        compare_update(rank)
        compare_update(rank, compiled=True)
        check_shared_evaluation()
        single.WARMUP_STEPS = 1
        check_shared_clock(rank)

        #Exercise the actual production entry point, with only data/output locations
        #and the warmup/evaluation intervals reduced for this tiny integration run.
        data.PROCESSED_DATA_DIR = folder
        train_ddp.RUNS_DIR = folder / "runs"
        import tools.compare_runs as reports
        reports.RUNS_DIR = train_ddp.RUNS_DIR
        reports.PROGRESS_PAGE = folder / "progress.html"
        single.BATCH_SIZE = 8  #10 validation rows includes an incomplete final batch
        single.EVAL_EVERY, single.EVAL_BATCHES = 2, 1
        args = train_ddp.parse_args([
            "--device", "cpu", "--no-compile", "--no-tensorboard", "--name", "smoke",
            "--positions", "192", "--batch-size", "16", "--d-model", "16", "--layers", "1",
            "--heads", "2", "--templates", "2",
            "--training-set", "a",
        ])
        train_ddp.run(args, torch.device("cpu"))
        model, settings = load_checkpoint(folder / "runs" / "smoke" / "model.pt", torch.device("cpu"))
        assert settings["d_model"] == 16 and settings["position_templates"] == 2
        assert isinstance(model, ChessTransformer)
        states = [None] * dist.get_world_size()
        dist.all_gather_object(states, {name: tensor.tolist() for name, tensor in model.state_dict().items()})
        assert all(state == states[0] for state in states[1:])
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires the Gloo backend")
@pytest.mark.parametrize("world_size", [2, 4])
def test_distributed_gradients_budgets_and_complete_training(tmp_path, world_size):
    write_set(tmp_path, "a")
    context = mp.spawn(distributed_worker, args=(str(tmp_path), world_size), nprocs=world_size, join=False)
    deadline = time.monotonic() + 180
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail(f"{world_size}-process training did not finish within 180 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
    run_dir = tmp_path / "runs" / "smoke"
    assert list((tmp_path / "runs").iterdir()) == [run_dir]
    assert len(list(run_dir.glob("events.out.tfevents.*"))) == 1
    settings = json.loads((run_dir / "settings.json").read_text())
    assert settings["world_size"] == world_size
    assert settings["positions_seen"] == 192
    assert settings["batch_size"] == 16 and settings["per_gpu_batch_size"] == 16 // world_size
    assert settings["target_temperature"] == single.TARGET_TEMPERATURE
    assert settings["optimizer"] == "muon"
    assert settings["finished_at"] and settings["final_val_value_loss"] >= 0
    assert (tmp_path / "progress.html").exists()

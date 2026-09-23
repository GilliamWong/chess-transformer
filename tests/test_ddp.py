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
from data import FUTURE_UNKNOWN, MixedDataset
from encoding import BOARD_VOCAB_SIZE, NUM_HISTORY_CODES, TOKENS_PER_BOARD
from model import ChessTransformer
from multi_gpu import train_ddp
from tools.budget import Budget
from tools.checkpoint import load_checkpoint


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_shuffled_shards_have_equal_full_batches_and_do_not_overlap(world_size):
    dataset = list(range(67))
    samplers = [train_ddp.TrainingSampler(dataset, 16, rank, world_size, 42) for rank in range(world_size)]
    batches = [list(sampler) for sampler in samplers]
    shards = [[index for batch in rank_batches for index in batch] for rank_batches in batches]
    assert all(len(rank_batches) == 4 for rank_batches in batches)
    assert all(len(batch) == 16 // world_size for rank_batches in batches for batch in rank_batches)
    assert len({index for shard in shards for index in shard}) == 64
    assert list(samplers[0]) == batches[0]
    samplers[0].epoch = 1
    assert list(samplers[0]) != batches[0]


def test_mixture_keeps_repeated_shares_and_independent_reproducible_streams():
    dataset = MixedDataset([list(range(1000))] * 3 + [list(range(2000))])
    first = train_ddp.TrainingSampler(dataset, 32, 0, 2, 42)
    second = train_ddp.TrainingSampler(dataset, 32, 1, 2, 42)
    left, right = next(iter(first)), next(iter(second))
    for batch in (left, right):
        assert len(batch) == 16
        assert [sum(start <= index < stop for index in batch)
                for start, stop in ((0, 1000), (1000, 2000), (2000, 3000), (3000, 5000))] == [4] * 4
    assert left != right
    torch.manual_seed(999)
    assert next(iter(first)) == left  #model randomness must not perturb sampling
    first.epoch = 1
    assert next(iter(first)) != left


@pytest.mark.parametrize("dataset,batch_size,world_size", [
    (list(range(32)), 7, 2), (list(range(8)), 16, 2),
    (MixedDataset([list(range(100))] * 3), 16, 2),
    (MixedDataset([[], list(range(100))]), 16, 2),
])
def test_invalid_shards_fail_before_training(dataset, batch_size, world_size):
    with pytest.raises(ValueError):
        train_ddp.TrainingSampler(dataset, batch_size, 0, world_size, 42)


def compare_update(rank, future_mode, compiled=False):
    torch.manual_seed(123)
    model = ChessTransformer(d_model=16, num_layers=1, n_heads=2, position_templates=2,
                             history=1, future=int(future_mode != "disabled"), swiglu=True, dropout=0, value_target="result")
    with torch.no_grad():
        model.templates.normal_(std=0.1)
    distributed = DDP(model)
    reference = copy.deepcopy(model)
    if compiled:
        distributed = torch.compile(distributed, backend="aot_eager")
    generator = torch.Generator().manual_seed(18)
    future = torch.randint(NUM_HISTORY_CODES, (8, 64), generator=generator)
    #One rank has no supervision; the other has uneven supervision within its batch.
    future[:4] = FUTURE_UNKNOWN
    future[4, :32] = FUTURE_UNKNOWN
    if future_mode == "unknown":
        future.fill_(FUTURE_UNKNOWN)
    batch = (
        torch.randint(BOARD_VOCAB_SIZE, (8, TOKENS_PER_BOARD), generator=generator),
        torch.arange(16).reshape(8, 2), torch.tensor([[0.7, 0.3]] * 8),
        torch.arange(8) % 3, torch.randint(NUM_HISTORY_CODES, (8, 1, 64), generator=generator), future,
    )
    world_size = dist.get_world_size()
    local_size = 8 // world_size
    local = tuple(tensor[rank * local_size:(rank + 1) * local_size] for tensor in batch)
    loss = train_ddp.distributed_loss(distributed(local[0], local[4]), local, 0.8, 0.5)
    loss.backward()

    policy, value, future_logits = reference(batch[0], batch[4])
    expected = single.policy_loss(policy, batch[1], batch[2]) + 0.8 * single.value_loss(value, batch[3])
    if future_logits is not None:
        expected = expected + 0.5 * single.future_loss(future_logits, batch[5])
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
    budget = Budget(16, minutes=1)
    optimizer = torch.optim.SGD([nn.Parameter(torch.ones(1))], lr=0.1)
    scheduler = train_ddp.make_scheduler(optimizer, budget, "cosine")
    now = [100.0 + 1000 * rank]
    with patch("tools.budget.time.perf_counter", side_effect=lambda: now[0]):
        for step, expected_seconds in enumerate((0, 30, 60), start=1):
            now[0] += 30 if rank == 0 else 300  #rank one has a deliberately different clock
            optimizer.step()
            train_ddp.step_budget(budget, torch.device("cpu"))
            scheduler.step()
            assert budget.steps == step
            assert budget.positions == 16 * step
            assert budget.seconds == expected_seconds
            assert budget.finished() == (step == 3)
            states = [None] * dist.get_world_size()
            dist.all_gather_object(states, (budget.progress(), scheduler.get_last_lr()))
            assert all(state == states[0] for state in states[1:])
            if step == 2:
                if rank == 0:
                    budget.pause()
                now[0] += 600  #validation is outside the budget
                if rank == 0:
                    budget.resume()


def write_set(folder, name, labelled):
    for split, size in (("train", 32), ("val", 10)):
        arrays = dict(
            boards=np.full((size, TOKENS_PER_BOARD), 5, dtype=np.uint8),
            moves=np.arange(size, dtype=np.int16), results=np.zeros(size, dtype=np.int8),
            plies=(np.arange(size) % 5).astype(np.int16),
        )
        if labelled:
            arrays.update(stockfish_moves=np.tile(np.array([0, 1], dtype=np.int16), (size, 1)),
                          stockfish_evals=np.tile(np.array([120, -60], dtype=np.int16), (size, 1)))
        for field, array in arrays.items():
            (folder / f"{name}_{split}_{field}.bin").write_bytes(array.tobytes())


def distributed_worker(rank, folder, world_size):
    torch.set_num_threads(1)
    folder = Path(folder)
    dist.init_process_group("gloo", init_method=(folder / "rendezvous").as_uri(),
                            rank=rank, world_size=world_size, timeout=timedelta(seconds=90))
    try:
        for mode in ("disabled", "uneven", "unknown"):
            compare_update(rank, mode)
        compare_update(rank, "uneven", compiled=True)
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
            "--heads", "2", "--templates", "2", "--history", "1", "--future", "2", "--swiglu",
            "--training-set", "a+a+a+b", "--policy-target", "stockfish", "--value-target", "result",  #b has no evals
        ])
        train_ddp.run(args, torch.device("cpu"))
        model, settings = load_checkpoint(folder / "runs" / "smoke" / "model.pt", torch.device("cpu"))
        assert settings["future"] == 2 and settings["history"] == 1
        assert isinstance(model, ChessTransformer)
        states = [None] * dist.get_world_size()
        dist.all_gather_object(states, {name: tensor.tolist() for name, tensor in model.state_dict().items()})
        assert all(state == states[0] for state in states[1:])
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires the Gloo backend")
@pytest.mark.parametrize("world_size", [2, 4])
def test_distributed_gradients_budgets_and_complete_training(tmp_path, world_size):
    write_set(tmp_path, "a", labelled=True)
    write_set(tmp_path, "b", labelled=False)
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
    assert settings["policy_target"] == "stockfish"
    assert settings["finished_at"] and settings["final_val_future_loss"] >= 0
    assert (tmp_path / "progress.html").exists()

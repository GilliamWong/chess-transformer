import copy
from contextlib import contextmanager
from itertools import islice
import json
import time
from datetime import timedelta
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from training import compare_runs as reports
from training import data
from training import train as single
from model.encoding import BLACK_QUEENSIDE_INDEX, BOARD_VOCAB_SIZE, CASTLING_TOKEN_OFFSET, NUM_PIECE_CODES, NUM_SQUARES, TOKENS_PER_BOARD, WHITE_KINGSIDE_INDEX
from model.model import ChessTransformer
from training import train_ddp
from training.muon import BatchedMuon
from model.checkpoint import load_checkpoint


CPU = torch.device("cpu")
needs_gloo = pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires the Gloo backend")


#every square holds a piece code, which the square features index by
def random_boards(rows: np.random.Generator, size: int) -> np.ndarray:
    squares = rows.integers(0, NUM_PIECE_CODES, (size, NUM_SQUARES))
    metadata = rows.integers(0, BOARD_VOCAB_SIZE, (size, TOKENS_PER_BOARD - NUM_SQUARES))
    return np.concatenate([squares, metadata], axis=1).astype(np.uint8)


@contextmanager
def process_group(rank, folder, world_size):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=(folder / "rendezvous").as_uri(),
                            rank=rank, world_size=world_size, timeout=timedelta(seconds=90))
    try:
        yield
    finally:
        dist.destroy_process_group()


def use_folder(folder):
    data.PROCESSED_DATA_DIR = folder
    train_ddp.RUNS_DIR = reports.RUNS_DIR = folder / "runs"
    reports.PROGRESS_PAGE = folder / "progress.html"


def assert_same_on_every_rank(value):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    assert all(other == values[0] for other in values[1:])


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
    batch = (
        torch.from_numpy(random_boards(np.random.default_rng(18), 8)).long(),
        torch.arange(16).reshape(8, 2), torch.tensor([[0.7, 0.3]] * 8),
        torch.rand(8, generator=torch.Generator().manual_seed(18)),
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


def check_shared_evaluation():
    rows = np.random.default_rng(5)
    size = 11
    val_dataset = data.ChessGamesDataset(
        random_boards(rows, size),
        rows.integers(0, 1968, (size, 2)).astype(np.int16), rows.integers(-300, 300, (size, 2)).astype(np.int16),
        single.TARGET_TEMPERATURE)
    _, val_loader = data.datasets_to_dataloaders(val_dataset, val_dataset, val_batch_size=4, seed=0)  #batches of 4, 4 and 3: some ranks get none
    torch.manual_seed(7)
    model = ChessTransformer(d_model=16, num_layers=1, n_heads=2, position_templates=2)
    for max_batches in (None, 2):
        expected = single.evaluate_model(model, val_loader, CPU, max_batches)
        assert train_ddp.evaluate(model, val_loader, CPU, max_batches) == pytest.approx(expected, rel=1e-6)


def write_bins(folder, name, split, **arrays):
    for field, array in arrays.items():
        (folder / f"{name}_{split}_{field}.bin").write_bytes(array.tobytes())


def write_set(folder, name):
    for split, size in (("train", 32), ("val", 10)):
        write_bins(folder, name, split,
                   boards=np.full((size, TOKENS_PER_BOARD), 5, dtype=np.uint8),
                   stockfish_moves=np.tile(np.array([0, 1], dtype=np.int16), (size, 1)),
                   stockfish_evals=np.tile(np.array([120, -60], dtype=np.int16), (size, 1)))


def distributed_worker(rank, folder, world_size):
    with process_group(rank, folder, world_size):
        compare_update(rank)
        compare_update(rank, compiled=True)
        check_shared_evaluation()
        single.WARMUP_STEPS = 1

        use_folder(folder)
        single.BATCH_SIZE = 8  #the 10 validation rows end in an incomplete batch
        single.EVAL_EVERY, single.EVAL_BATCHES = 2, 1
        args = train_ddp.parse_args([
            "--device", "cpu", "--no-compile", "--no-tensorboard", "--name", "smoke",
            "--positions", "192", "--batch-size", "16", "--d-model", "16", "--layers", "1",
            "--heads", "2", "--templates", "2",
            "--training-set", "a",
        ])
        train_ddp.run(args, CPU)
        model, settings = load_checkpoint(folder / "runs" / "smoke" / "model.pt", CPU)
        assert settings["d_model"] == 16 and settings["position_templates"] == 2
        assert isinstance(model, ChessTransformer)
        assert_same_on_every_rank({name: tensor.tolist() for name, tensor in model.state_dict().items()})


def write_varied_set(folder, name, seed=11, castling_free=False):
    #castling_free: no row has a castling right, so every row can be mirrored
    rows = np.random.default_rng(seed)
    for split, size in (("train", 32), ("val", 10)):
        boards = random_boards(rows, size)
        if castling_free:
            boards[:, WHITE_KINGSIDE_INDEX:BLACK_QUEENSIDE_INDEX + 1] = CASTLING_TOKEN_OFFSET
        write_bins(folder, name, split,
                   boards=boards,
                   stockfish_moves=rows.integers(0, 1968, (size, 2)).astype(np.int16),
                   stockfish_evals=rows.integers(-300, 300, (size, 2)).astype(np.int16),
                   worst_moves=np.where(rows.random((size, 4)) < 0.5, rows.integers(0, 1968, (size, 4)), -1).astype(np.int16),
                   worst_losses=np.full((size, 4), 0.5, dtype=np.float16))


class Preempted(Exception):
    pass


RESUME_FLAGS = ["--device", "cpu", "--no-compile", "--no-tensorboard", "--positions", "320", "--batch-size", "16",
                "--d-model", "16", "--layers", "1", "--heads", "2", "--templates", "2", "--training-set", "b",
                "--eval-every", "3", "--resume", "--checkpoint-minutes", "0", "--snapshot-every", "5"]


def prepare_resume_run(folder):
    use_folder(folder)
    single.BATCH_SIZE, single.EVAL_BATCHES = 8, 1
    single.WARMUP_STEPS = 2  #the stop comes after the warmup, which the schedule needs


def run_training(flags, name):
    train_ddp.run(train_ddp.parse_args(flags + ["--name", name]), CPU)


def run_until_preempted(flags, name):
    batches_on_device = single.batches_on_device
    def stopped_before_step_8(batches, device):
        for number, batch in enumerate(batches_on_device(batches, device)):
            if number == 7:
                raise Preempted
            yield batch
    with patch.object(single, "batches_on_device", stopped_before_step_8), pytest.raises(Preempted):
        run_training(flags, name)


def resume_worker(rank, folder, world_size, mix, mirror):
    with process_group(rank, folder, world_size):
        prepare_resume_run(folder)
        #20 steps of 2 batches an epoch: the stop after step 7 is in the middle of the fourth epoch
        #with a mix set, c gives 4 of the 16 and has 8 batches an epoch: the stop is in its first and the resume carries into its second
        flags = RESUME_FLAGS + (["--mix-set", "c", "--mix-share", "0.25"] if mix else [])
        if mirror:
            run_training(flags, "plain")  #the same run without mirroring
            flags += ["--mirror"]
        with patch.object(train_ddp, "TrainingSampler", wraps=train_ddp.TrainingSampler) as samplers:
            run_training(flags, "whole")
        assert [call.args[-1] for call in samplers.call_args_list] == [mirror] * (2 if mix else 1)

        run_until_preempted(flags, "stopped")
        dist.barrier()
        run_training(flags, "stopped")  #the requeued job
        run_training(flags, "stopped")  #requeued once more, after the end


def saved_weights(runs, run, checkpoint):
    return torch.load(runs / run / checkpoint)["state_dict"]


@needs_gloo
@pytest.mark.parametrize("mix, mirror", [(False, False), (True, False), (True, True)])
def test_a_stopped_and_resumed_run_ends_with_the_weights_of_one_that_never_stopped(tmp_path, mix, mirror):
    write_varied_set(tmp_path, "b", castling_free=mirror)
    write_varied_set(tmp_path, "c", seed=12, castling_free=mirror)
    mp.spawn(resume_worker, args=(tmp_path, 2, mix, mirror), nprocs=2)
    runs = tmp_path / "runs"
    snapshots = ["snapshot_80.pt", "snapshot_160.pt", "snapshot_240.pt", "snapshot_320.pt"]  #every 5 of the 20 steps
    for run in ("whole", "stopped"):
        assert sorted(path.name for path in (runs / run).glob("snapshot_*.pt")) == sorted(snapshots)
    for checkpoint in ("final.pt", "model.pt", *snapshots):
        whole = saved_weights(runs, "whole", checkpoint)
        stopped = saved_weights(runs, "stopped", checkpoint)
        assert whole.keys() == stopped.keys()
        assert all(torch.equal(whole[name], stopped[name]) for name in whole), checkpoint
    settings = json.loads((runs / "stopped" / "settings.json").read_text())
    assert [entry["step"] for entry in settings["resumed"]] == [7]
    assert settings["positions_seen"] == 320 and settings["blunder_labels"]
    assert settings["mirror"] == mirror and [entry["mirror"] for entry in settings["resumed"]] == [mirror]
    if mirror:
        plain = saved_weights(runs, "plain", "final.pt")
        whole = saved_weights(runs, "whole", "final.pt")
        assert not all(torch.equal(plain[name], whole[name]) for name in whole)
    assert not (runs / "stopped" / train_ddp.RESUME_FILE).exists()
    if mix:
        assert settings["mix_set"] == "c" and settings["mix_share"] == 0.25 and settings["mix_blunder_labels"]
        assert settings["final_mix_val_loss"] > 0 and settings["final_mix_val_loss"] != settings["final_val_loss"]


def micro_batch_worker(rank, folder, world_size):
    with process_group(rank, folder, world_size):
        prepare_resume_run(folder)
        run_training(RESUME_FLAGS, "whole")
        run_until_preempted(RESUME_FLAGS, "parts")
        dist.barrier()
        run_training(RESUME_FLAGS + ["--micro-batch", "4"], "parts")


@needs_gloo
def test_a_run_resumed_in_micro_batches_trains_like_one_in_whole_batches(tmp_path):
    write_varied_set(tmp_path, "b")
    mp.spawn(micro_batch_worker, args=(tmp_path, 2), nprocs=2)
    runs = tmp_path / "runs"
    def weights(run, checkpoint):
        state = saved_weights(runs, run, checkpoint)
        return torch.cat([tensor.flatten().float() for tensor in state.values() if tensor.is_floating_point()])
    #bf16 rounds the parts' gradients differently from the whole batch's, so the tolerance is a share of how far training moved the weights
    whole, parts = weights("whole", "final.pt"), weights("parts", "final.pt")
    assert (parts - whole).norm() < 0.03 * (whole - weights("whole", "snapshot_80.pt")).norm()
    settings = json.loads((runs / "parts" / "settings.json").read_text())
    assert settings["micro_batches"] == 1 and settings["resumed"][0]["micro_batches"] == 2


def test_mixed_training_batches_are_the_training_sets_rows_then_the_mix_sets():
    def dataset(token, positions):
        rows = np.random.default_rng(token)
        return data.ChessGamesDataset(np.full((positions, TOKENS_PER_BOARD), token, dtype=np.uint8), rows.integers(0, 1968, (positions, 2)).astype(np.int16),
                                      rows.integers(-300, 300, (positions, 2)).astype(np.int16), single.TARGET_TEMPERATURE)
    main_sampler = train_ddp.TrainingSampler(dataset(3, 10), 3, 0, 1, 0)
    mix_sampler = train_ddp.TrainingSampler(dataset(7, 6), 2, 0, 1, 1)
    main_loader = torch.utils.data.DataLoader(main_sampler.dataset, batch_size=None, sampler=main_sampler)
    mix_loader = torch.utils.data.DataLoader(mix_sampler.dataset, batch_size=None, sampler=mix_sampler)

    batches = list(islice(train_ddp.training_batches(main_loader, mix_loader), 7))

    for boards, *_ in batches:
        assert boards.shape == (5, TOKENS_PER_BOARD)
        assert boards[:3].eq(3).all() and boards[3:].eq(7).all()
    #3 batches an epoch in each: the seventh is the third epoch's first
    assert main_sampler.epoch == mix_sampler.epoch == 2


@needs_gloo
@pytest.mark.parametrize("world_size", [2, 4])
def test_distributed_gradients_budgets_and_complete_training(tmp_path, world_size):
    write_set(tmp_path, "a")
    context = mp.spawn(distributed_worker, args=(tmp_path, world_size), nprocs=world_size, join=False)
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


#shapes with 4, 2 and 1 matrices once the tall ones are transposed: none splits evenly between 3 processes, and the last
#has fewer matrices than there are processes
SHARDED_MUON_SHAPES = [(24, 40), (40, 24), (32, 32), (24, 40), (40, 24), (32, 32), (8, 40)]


def sharded_muon_worker(rank, folder, world_size):
    with process_group(rank, folder, world_size):
        torch.manual_seed(0)
        weights = [torch.randn(shape) for shape in SHARDED_MUON_SHAPES]
        reference_parameters = [nn.Parameter(weight.clone()) for weight in weights]
        sharded_parameters = [nn.Parameter(weight.clone()) for weight in weights]
        settings = dict(weight_decay=0.5, momentum=0.9, adjust_lr_fn="match_rms_adamw")
        reference = torch.optim.Muon(reference_parameters, lr=torch.tensor(1e-2), **settings)
        sharded = BatchedMuon(sharded_parameters, lr=torch.tensor(1e-2), **settings)
        assert sharded.world_size == world_size > 1

        #every rank has the same gradients, as after DDP's all-reduce
        generator = torch.Generator().manual_seed(1)
        for _ in range(3):
            for expected, actual in zip(reference_parameters, sharded_parameters):
                expected.grad = torch.randn(expected.shape, generator=generator) * 10 ** (3 * torch.rand(1, generator=generator).item())
                actual.grad = expected.grad.clone()
            reference.step()
            sharded.step()

        for expected, actual, weight in zip(reference_parameters, sharded_parameters, weights):
            assert (actual - expected).norm() <= 1e-2 * (expected - weight).norm()
        #DDP training's loss would not show ranks ending with different weights
        assert_same_on_every_rank([parameter.tolist() for parameter in sharded_parameters])


@needs_gloo
@pytest.mark.parametrize("world_size", [2, 3])
def test_muon_matrices_shared_out_between_processes_step_like_one_process_and_alike_on_every_rank(tmp_path, world_size):
    mp.spawn(sharded_muon_worker, args=(tmp_path, world_size), nprocs=world_size)

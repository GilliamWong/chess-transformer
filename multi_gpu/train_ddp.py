"""Separate DDP trainer. Example (Linux, two GPUs), from the repository root:

uv run torchrun --standalone --nproc-per-node=2 -m multi_gpu.train_ddp --minutes 30

--batch-size and --positions are global, across all processes. The model, targets,
loss definitions, validation and checkpoint format are shared with main.py.
"""

import argparse
import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler, Sampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import main as single
from data import MixedDataset
from model import ChessTransformer
from tools.budget import Budget
from tools.checkpoint import RUNS_DIR, save_checkpoint


class TrainingSampler(Sampler):
    """Equal local batches, with epoch shuffling or independently drawn mixture shares."""

    def __init__(self, dataset, batch_size: int, rank: int, world_size: int, seed: int):
        if batch_size <= 0 or batch_size % world_size:
            raise ValueError("global batch size must be positive and divisible by the process count")
        if len(dataset) < batch_size:
            raise ValueError("the training set must contain at least one global batch")
        self.dataset = dataset
        self.batch_size = batch_size // world_size
        self.global_batch_size = batch_size
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.epoch = 0
        if isinstance(dataset, MixedDataset):
            if self.batch_size % len(dataset.datasets):
                raise ValueError("per-process batch size must be divisible by the number of mixture shares")
            if any(len(part) == 0 for part in dataset.datasets):
                raise ValueError("every mixture share needs a nonempty training set")

    def __len__(self):
        return len(self.dataset) // self.global_batch_size

    def __iter__(self):
        if not isinstance(self.dataset, MixedDataset):
            sampler = DistributedSampler(
                self.dataset, num_replicas=self.world_size, rank=self.rank,
                shuffle=True, seed=self.seed, drop_last=True,
            )
            sampler.set_epoch(self.epoch)
            yield from BatchSampler(sampler, self.batch_size, drop_last=True)
            return

        #Replacement sampling matches EqualSharesBatchSampler. A private generator
        #keeps dropout/model RNG and other ranks from repeating the same draw stream.
        generator = torch.Generator().manual_seed(self.seed + self.epoch * self.world_size + self.rank)
        share = self.batch_size // len(self.dataset.datasets)
        for _ in range(len(self)):
            batch = []
            for dataset, offset in zip(self.dataset.datasets, self.dataset.first_indices):
                batch.extend((torch.randint(len(dataset), (share,), generator=generator) + offset).tolist())
            yield batch


def endless_batches(loader: DataLoader):
    while True:
        yield from loader
        loader.sampler.epoch += 1


def step_budget(budget: Budget, device: torch.device) -> None:
    """Rank zero owns elapsed time; all ranks use its progress for LR and stopping."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if budget.steps == 0:
        dist.barrier()  #all first-step compilation is outside the timed budget
    if dist.get_rank() == 0:
        budget.step_done()
    else:
        budget.steps += 1
    seconds = torch.tensor(budget.seconds, dtype=torch.float64, device=device)
    dist.broadcast(seconds, src=0)
    budget.seconds = seconds.item()


def make_scheduler(optimizer, budget: Budget, schedule: str):
    warmup_end = 0.0

    def multiplier(step):
        nonlocal warmup_end
        if step <= single.WARMUP_STEPS:
            warmup_end = budget.progress()
            return single.lr_multiplier(step, 0.0, schedule)
        progress = (budget.progress() - warmup_end) / max(1 - warmup_end, 1e-12)
        return single.lr_multiplier(step, progress, schedule)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def evaluate(model, loader, device, max_batches=None):
    #Use the unwrapped model: rank-zero-only DDP.forward could wait for other ranks.
    #The original validation loader retains exactly the single-GPU validation cohort.
    metrics = torch.zeros(4, dtype=torch.float64, device=device)
    if dist.get_rank() == 0:
        values = single.evaluate_model(model, loader, device, max_batches)
        metrics.copy_(torch.tensor(values, dtype=torch.float64, device=device))
    dist.broadcast(metrics, src=0)
    return metrics.tolist()


METRIC_NAMES = ("val_loss", "val_accuracy", "val_value_loss", "val_value_accuracy")


def log_validation(writer, metrics, step, final=False):
    tags = ("loss/val", "accuracy/val", "loss/val_value", "accuracy/val_value")
    for name, tag, value in zip(METRIC_NAMES, tags, metrics):
        writer.add_scalar(f"final/{name}" if final else tag, value, step)


def train(model, raw_model, train_loader, val_loader, optimizer, scheduler, budget, device,
          value_weight, writer=None, save_best=None):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    best_val_loss = float("inf")  #the best trained checkpoint, even if the first update regresses
    loss_total = torch.zeros((), device=device)
    interval_steps = 0
    progress = tqdm(total=budget.total, desc="train", dynamic_ncols=True, disable=rank != 0,
                    unit="s" if budget.timed else "it")
    model.train()
    try:
        for cpu_batch in endless_batches(train_loader):
            batch = tuple(tensor.to(device, non_blocking=True) for tensor in cpu_batch)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, value_logits = model(batch[0])
                loss = single.policy_loss(logits, batch[1], batch[2]) + value_weight * single.value_loss(value_logits, batch[3])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            before = budget.seconds
            step_budget(budget, device)
            scheduler.step()
            progress.update(budget.seconds - before if budget.timed else 1)
            loss_total += loss.detach()
            interval_steps += 1
            finished = budget.finished()  #one shared decision, before rank zero pauses its clock

            if budget.steps % single.EVAL_EVERY == 0 or finished:
                if rank == 0:
                    budget.pause()
                dist.all_reduce(loss_total)
                train_loss = loss_total.item() / (interval_steps * world_size)
                metrics = evaluate(raw_model, val_loader, device, single.EVAL_BATCHES)
                if rank == 0:
                    tqdm.write(f"step {budget.steps} ({budget.positions:,} positions, {budget.seconds / 60:.1f} min): "
                               f"train loss {train_loss:.3f}, val loss {metrics[0]:.3f}, val accuracy {metrics[1]:.1%}")
                    if writer is not None:
                        writer.add_scalar("loss/train", train_loss, budget.steps)
                        writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], budget.steps)
                        log_validation(writer, metrics, budget.steps)
                    if metrics[0] < best_val_loss:
                        best_val_loss = metrics[0]
                        if save_best is not None:
                            save_best()
                model.train()
                loss_total.zero_()
                interval_steps = 0
                dist.barrier()  #checkpoint/logging time is excluded on every process
                if rank == 0 and not finished:
                    budget.resume()
            if finished:
                break
    finally:
        progress.close()


def run(args, device: torch.device):
    """Run inside an initialized process group (also used by CPU integration tests)."""
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if args.batch_size <= 0 or args.batch_size % world_size:
        raise ValueError("--batch-size must be positive and divisible by the process count")
    budget = (Budget(args.batch_size, positions=args.positions) if args.positions is not None
              else Budget(args.batch_size, minutes=args.minutes))
    if not budget.timed and budget.max_steps <= single.WARMUP_STEPS:
        raise ValueError(f"the position budget must exceed {single.WARMUP_STEPS} global warmup steps")
    if budget.timed and (not math.isfinite(args.minutes) or args.minutes < 1):
        raise ValueError("--minutes must be finite and at least 1")
    torch.manual_seed(args.seed)

    #Only one process per host warms the OS file cache. Read-only mappings let
    #peers share cached pages without copying the data.
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    train_dataset, val_dataset = single.load_training_set(args.training_set, args.policy_target, args.target_temperature)
    if local_rank == 0:
        single.warm_file_cache(train_dataset, val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=None, sampler=BatchSampler(
        SequentialSampler(val_dataset), single.BATCH_SIZE, drop_last=False))
    dist.barrier()
    if not len(val_loader.dataset):
        raise ValueError("validation needs at least one position")
    sampler = TrainingSampler(train_dataset, args.batch_size, rank, world_size, args.seed)
    train_loader = DataLoader(train_dataset, batch_size=None, sampler=sampler,
                              pin_memory=device.type == "cuda",
                              generator=torch.Generator().manual_seed(args.seed + rank))

    model_settings = dict(d_model=args.d_model, num_layers=args.layers, n_heads=args.heads,
                          position_templates=args.templates, mlp_ratio=args.mlp_ratio,
                          qk_ratio=args.qk_ratio[0] if len(args.qk_ratio) == 1 else args.qk_ratio, kv_heads=args.kv_heads)
    raw_model = ChessTransformer(**model_settings).to(device)
    if args.init_from is not None:
        #weights only: the optimizer and the schedule start afresh, so this is a restart with new settings, not a resume
        raw_model.load_state_dict(torch.load(args.init_from, map_location=device)["state_dict"])
    model = DDP(raw_model, device_ids=[device.index] if device.type == "cuda" else None)
    if not args.no_compile:
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr, fused=device.type == "cuda")
    scheduler = make_scheduler(optimizer, budget, args.schedule)
    metrics = evaluate(raw_model, val_loader, device, single.EVAL_BATCHES)
    writer = None
    try:
        if rank == 0:
            budget_label = (f"{budget.describe()}_{world_size}{'gpu' if device.type == 'cuda' else 'cpu'}"
                            if budget.timed else budget.describe())
            run_name = args.name or (single.folder_name(args.note) if args.note else
                                     f"ddp{world_size}_d{args.d_model}_L{args.layers}_h{args.heads}_"
                                     f"{budget.describe()}_{datetime.now():%Y%m%d_%H-%M-%S_%f}")
            run_dir = RUNS_DIR / run_name
            run_dir.mkdir(parents=True)  #a named run must not overwrite an earlier run
            writer = SummaryWriter(str(run_dir))
            settings = dict(
                model_settings=model_settings, parameters=sum(p.numel() for p in raw_model.parameters()),
                peak_learn_rate=args.lr, schedule=args.schedule, seed=args.seed,
                value_loss_weight=args.value_weight,
                budget=budget_label, batch_size=args.batch_size, per_gpu_batch_size=sampler.batch_size,
                training_set=args.training_set, policy_target=args.policy_target,
                target_temperature=args.target_temperature if args.policy_target == "stockfish" else None,
                note=args.note, commit=single.current_commit(), trainer="ddp", world_size=world_size,
                init_from=str(args.init_from) if args.init_from is not None else None,
                device=device.type, backend=dist.get_backend(), compile=not args.no_compile,
            )
            settings_path = run_dir / "settings.json"
            settings_path.write_text(json.dumps(settings, indent=4), encoding="utf-8")
            print(f"DDP: {world_size} processes on {device.type}, global batch {args.batch_size}, "
                  f"local batch {sampler.batch_size}; untrained val loss {metrics[0]:.3f}")
            print(f"logging to {run_dir}")
            if not args.no_tensorboard:
                from tools.tensorboard_server import start_tensorboard
                start_tensorboard()

        def save_best():
            save_checkpoint(raw_model, model_settings, run_dir / "model.pt")

        train(model, raw_model, train_loader, val_loader, optimizer, scheduler, budget, device,
              args.value_weight, writer, save_best if rank == 0 else None)
        if rank == 0:
            save_checkpoint(raw_model, model_settings, run_dir / "final.pt")  #model.pt is the best evaluation's
        metrics = evaluate(raw_model, val_loader, device)
        if rank == 0:
            log_validation(writer, metrics, budget.steps, final=True)
            settings.update({f"final_{name}": value for name, value in zip(METRIC_NAMES, metrics)})
            settings.update(train_seconds=budget.seconds, positions_seen=budget.positions,
                            gpu_seconds=budget.seconds * world_size if device.type == "cuda" else 0,
                            finished_at=datetime.now().isoformat(timespec="seconds"))
            settings_path.write_text(json.dumps(settings, indent=4), encoding="utf-8")
            from tools.compare_runs import write_progress_page
            print(f"final val loss {metrics[0]:.3f}, accuracy {metrics[1]:.1%}; "
                  f"progress page: {write_progress_page()}")
        dist.barrier()
    finally:
        if writer is not None:
            writer.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    budgets = parser.add_mutually_exclusive_group()
    budgets.add_argument("--minutes", type=float, default=single.TRAIN_MINUTES_BUDGET,
                         help="wall minutes of training, excluding first-step compilation and evaluation")
    budgets.add_argument("--positions", type=int, help="positions across all GPUs, rounded down to full global batches")
    parser.add_argument("--batch-size", type=int, default=single.BATCH_SIZE, help="GLOBAL batch size, default %(default)s")
    parser.add_argument("--name", help="new run folder name; existing folders are rejected")
    parser.add_argument("--init-from", type=Path, help="model.pt to start the weights from (same architecture flags); optimizer and schedule start afresh")
    parser.add_argument("--note", default="")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--no-compile", action="store_true", help="use eager execution for debugging")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="CPU/Gloo is for distributed smoke tests")
    parser.add_argument("--d-model", type=int, default=single.D_MODEL)
    parser.add_argument("--layers", type=int, default=single.NUM_LAYERS)
    parser.add_argument("--heads", type=int, default=single.N_HEADS)
    parser.add_argument("--templates", type=int, default=single.POSITION_TEMPLATES)
    parser.add_argument("--mlp-ratio", type=float)
    parser.add_argument("--kv-heads", type=int, help="key/value heads, each shared by a group of heads (grouped-query attention)")
    parser.add_argument("--qk-ratio", type=float, nargs="+", default=[1.0], help="width of the queries and keys as a multiple of the values', one for every block or one per block")
    parser.add_argument("--lr", type=float, default=single.PEAK_LEARN_RATE)
    parser.add_argument("--schedule", choices=("cosine", "hold", "constant"), default=single.SCHEDULE)
    parser.add_argument("--seed", type=int, default=single.SEED)
    parser.add_argument("--training-set", default=single.TRAINING_SET)
    parser.add_argument("--policy-target", choices=("human", "stockfish"), default=single.POLICY_TARGET)
    parser.add_argument("--target-temperature", type=float, default=single.TARGET_TEMPERATURE)
    parser.add_argument("--value-weight", type=float, default=single.VALUE_LOSS_WEIGHT)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
        raise SystemExit("launch from the repository root with: uv run torchrun --standalone --nproc-per-node=2 -m multi_gpu.train_ddp [options]")
    local_rank = int(os.environ["LOCAL_RANK"])
    if args.device == "cuda":
        if not dist.is_nccl_available():
            raise SystemExit("GPU DDP requires a CUDA PyTorch build with NCCL (use Linux; native Windows lacks NCCL)")
        if local_rank >= torch.cuda.device_count():
            raise SystemExit("--nproc-per-node must not exceed the number of visible CUDA GPUs")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(minutes=60))
    try:
        run(args, device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

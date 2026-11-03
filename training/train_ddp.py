"""Multi-GPU DDP trainer, from the repository root on Linux:
uv run torchrun --standalone --nproc-per-node=2 -m training.train_ddp --positions 100000000
--batch-size and --positions are global, across all processes."""

import argparse
import contextlib
import json
import os
import time
from datetime import datetime, timedelta
from itertools import islice
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import BatchSampler, DataLoader, Sampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from training import train as single
from training.data import concatenate_batches
from training.mirror import draw_mirrors
from model.model import ChessTransformer, mlp_hidden_width
from training.budget import Budget
from model.checkpoint import RUNS_DIR, save_checkpoint
from training.compare_runs import write_progress_page
from training.tensorboard_server import start_tensorboard

RESUME_FILE = "resume.pt"


#the mirror coins depend on (seed, epoch, row) alone, so a resumed run draws what an unstopped one would, at any process count
class TrainingSampler(Sampler):
    def __init__(self, dataset, batch_size: int, rank: int, world_size: int, seed: int, mirror: bool = False):
        if batch_size <= 0 or batch_size % world_size:
            raise ValueError("global batch size must be positive and divisible by the process count")
        if len(dataset) < batch_size:
            raise ValueError("the training set must contain at least one global batch")
        self.dataset = dataset
        self.batch_size = batch_size // world_size
        self.global_batch_size = batch_size
        self.rank, self.seed = rank, seed
        self.mirror = mirror
        self.epoch = 0
        self.skip = 0

    def __len__(self):
        return len(self.dataset) // self.global_batch_size

    def __iter__(self):
        #int32: as a python list the order would take gigabytes per process
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.dataset), generator=generator, dtype=torch.int32)
        first, self.skip = self.skip, 0
        for start in range(first * self.global_batch_size, len(self) * self.global_batch_size, self.global_batch_size):
            local_start = start + self.rank * self.batch_size
            rows = order[local_start:local_start + self.batch_size].long()
            yield draw_mirrors(rows, self.seed, self.epoch) if self.mirror else rows


def endless_batches(loader: DataLoader):
    while True:
        yield from loader
        loader.sampler.epoch += 1


def training_batches(loader: DataLoader, mix_loader: DataLoader | None):
    if mix_loader is None:
        yield from endless_batches(loader)
        return
    mix_batches = endless_batches(mix_loader)
    for batch in endless_batches(loader):
        yield concatenate_batches(batch, next(mix_batches))


#clock is rank zero's own timer (None elsewhere)
def step_budget(budget: Budget, clock: Budget | None) -> None:
    if budget.steps == 0:
        dist.barrier()  #first-step compilation stays outside the clock
    budget.steps += 1
    if clock is not None:
        clock.step_done()


METRIC_TAGS = {"val_loss": "loss/val", "val_accuracy": "accuracy/val", "val_value_loss": "loss/val_value",
               "val_value_accuracy": "accuracy/val_value", "val_value_error": "error/val_value"}


#every rank measures every world_size-th batch; takes the unwrapped model, whose forward waits for no other rank
def evaluate(model, loader, device, max_batches=None):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    count = len(loader) if max_batches is None else min(max_batches, len(loader))
    own_batches = [indices for number, indices in enumerate(islice(loader.sampler, count)) if number % world_size == rank]
    totals = torch.zeros(len(METRIC_TAGS) + 1, dtype=torch.float64, device=device)
    if own_batches:
        positions = sum(len(indices) for indices in own_batches)
        own_loader = DataLoader(loader.dataset, sampler=own_batches, batch_size=None)
        averages = single.evaluate_model(model, own_loader, device, show_progress=rank == 0)
        totals.copy_(torch.tensor([average * positions for average in averages] + [positions], dtype=torch.float64))
    dist.all_reduce(totals)
    return (totals[:-1] / totals[-1]).tolist()


def log_validation(writer, metrics, step, final=False):
    for (name, tag), value in zip(METRIC_TAGS.items(), metrics):
        writer.add_scalar(f"final/{name}" if final else tag, value, step)


#writer, save_best, save_resume(best_val_loss) and save_snapshot are rank zero's, None elsewhere
def train(model, master_weights, train_loader, val_loader, optimizers, schedulers, budget, clock, device, value_weight,
          eval_every, writer, save_best, blunder_weight, best_val_loss, save_resume, resume_every_seconds, mix_loader,
          save_snapshot, snapshot_every, micro_batches):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    totals = torch.zeros(3, device=device)
    interval_steps = 0
    last_resume_save = time.monotonic()
    progress = tqdm(total=budget.max_steps, initial=budget.steps, desc="train", dynamic_ncols=True,
                    disable=rank != 0, unit="it")
    model.train()
    try:
        for batch in single.batches_on_device(training_batches(train_loader, mix_loader), device):
            master_weights.zero_grad()
            #the parts' gradients add up here and DDP averages them across ranks on the last part's backward
            loss = penalty = torch.zeros((), device=device)
            for part, micro_batch in enumerate(zip(*(tensor.chunk(micro_batches) for tensor in batch))):
                with contextlib.nullcontext() if part == micro_batches - 1 else model.no_sync():
                    logits, value_logits = model(micro_batch[0])
                    part_loss = single.policy_loss(logits, micro_batch[1], micro_batch[2]) + value_weight * single.value_loss(value_logits, micro_batch[3])
                    part_penalty = single.blunder_loss(logits, micro_batch[4]) if blunder_weight else torch.zeros((), device=device)
                    ((part_loss + blunder_weight * part_penalty) / micro_batches).backward()
                loss = loss + part_loss.detach() / micro_batches
                penalty = penalty + part_penalty.detach() / micro_batches
            master_weights.gradients_to_master()
            gradient_norm = torch.nn.utils.clip_grad_norm_(master_weights.parameters, 1.0)
            #every rank steps all the weights; Muon shares its orthogonalising out between the ranks
            for optimizer in optimizers:
                optimizer.step()
            master_weights.weights_to_compute()

            step_budget(budget, clock)
            for scheduler in schedulers:
                scheduler.step()
            progress.update(1)
            totals += torch.stack([loss, penalty, gradient_norm]).float()
            interval_steps += 1
            finished = budget.finished()  #one shared decision, before rank zero pauses its clock

            if budget.steps % eval_every == 0 or finished:
                dist.all_reduce(totals)
                #waits for the gpu, so the pause starts when the steps are done
                train_loss, train_penalty, gradient_norm = (totals / (interval_steps * world_size)).tolist()
                if rank == 0:
                    clock.pause()
                metrics = evaluate(master_weights.model, val_loader, device, single.EVAL_BATCHES)
                if rank == 0:
                    tqdm.write(f"step {budget.steps} ({budget.positions:,} positions, {clock.seconds / 60:.1f} min): "
                               f"train loss {train_loss:.3f}, val loss {metrics[0]:.3f}, val accuracy {metrics[1]:.1%}")
                    writer.add_scalar("loss/train", train_loss, budget.steps)
                    if blunder_weight:
                        writer.add_scalar("loss/train_blunder", train_penalty, budget.steps)
                    writer.add_scalar("gradient_norm/train", gradient_norm, budget.steps)
                    writer.add_scalar("learning_rate", schedulers[0].get_last_lr()[0], budget.steps)
                    log_validation(writer, metrics, budget.steps)
                    if metrics[0] < best_val_loss:
                        best_val_loss = metrics[0]
                        save_best()
                model.train()
                totals.zero_()
                interval_steps = 0
                dist.barrier()  #checkpoint/logging time is excluded on every process
                if rank == 0 and not finished:
                    clock.resume()
            if save_snapshot is not None and budget.steps % snapshot_every == 0:
                save_snapshot()
            if finished:
                break
            if save_resume is not None and time.monotonic() - last_resume_save >= resume_every_seconds:
                save_resume(best_val_loss)
                last_resume_save = time.monotonic()
    finally:
        progress.close()


def save_resume_state(path: Path, raw_model, optimizers, schedulers, budget: Budget, clock: Budget, best_val_loss: float):
    state = dict(steps=budget.steps, seconds=clock.seconds, best_val_loss=best_val_loss, model=raw_model.state_dict(),
                 optimizers=[optimizer.state_dict() for optimizer in optimizers],
                 schedulers=[scheduler.state_dict() for scheduler in schedulers])
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def run(args, device: torch.device):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if args.batch_size <= 0 or args.batch_size % world_size:
        raise ValueError("--batch-size must be positive and divisible by the process count")
    local_batch = args.batch_size // world_size
    if args.micro_batch is not None and (args.micro_batch <= 0 or local_batch % args.micro_batch):
        raise ValueError(f"--micro-batch must divide the local batch of {local_batch} positions, got {args.micro_batch}")
    micro_batches = local_batch // args.micro_batch if args.micro_batch is not None else 1
    budget = Budget(args.batch_size, positions=args.positions)
    clock = Budget(args.batch_size, positions=args.positions) if rank == 0 else None
    if budget.max_steps <= single.WARMUP_STEPS:
        raise ValueError(f"the position budget must exceed {single.WARMUP_STEPS} global warmup steps")
    if not 0 < args.decay_fraction <= 1:
        raise ValueError(f"--decay-fraction is a share of the run, above 0 and at most 1, got {args.decay_fraction}")
    resumed = None
    if args.resume:
        if args.name is None:
            raise ValueError("--resume needs --name")
        #each rank reads the state itself: the run folder is on a filesystem every rank sees
        settings_path = RUNS_DIR / args.name / "settings.json"
        if settings_path.exists() and "finished_at" in json.loads(settings_path.read_text(encoding="utf-8")):
            if rank == 0:
                print(f"{args.name} has finished already, nothing to resume")
            return
        resume_path = RUNS_DIR / args.name / RESUME_FILE
        if resume_path.exists():
            resumed = torch.load(resume_path, map_location=device, weights_only=True)
    torch.manual_seed(args.seed)

    mix_batch_size = 0
    if args.mix_set is not None:
        if not 0 < args.mix_share < 1:
            raise ValueError(f"--mix-share is the share of every batch from {args.mix_set}, between 0 and 1, got {args.mix_share}")
        mix_batch_size = round(args.batch_size * args.mix_share / world_size) * world_size
        if not 0 < mix_batch_size < args.batch_size:
            raise ValueError(f"a mix share of {args.mix_share} leaves no whole batch of {world_size} processes on one side of {args.batch_size}")

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    def load(name):
        return single.load_training_set(name, args.target_temperature, args.result_weight, args.visit_temperature, args.blunder_threshold)
    train_dataset, val_dataset = load(args.training_set)
    mix_train_dataset, mix_val_dataset = load(args.mix_set) if args.mix_set is not None else (None, None)
    if local_rank == 0:
        single.warm_file_cache(*(dataset for dataset in (train_dataset, val_dataset, mix_train_dataset, mix_val_dataset) if dataset is not None))
    def val_loader_of(dataset):
        return DataLoader(dataset, batch_size=None, sampler=BatchSampler(SequentialSampler(dataset), single.BATCH_SIZE, drop_last=False))
    val_loader = val_loader_of(val_dataset)
    dist.barrier()
    if not len(val_dataset):
        raise ValueError("validation needs at least one position")
    sampler = TrainingSampler(train_dataset, args.batch_size - mix_batch_size, rank, world_size, args.seed, args.mirror)
    #not pinned here: batches_on_device pins them in its thread
    train_loader = DataLoader(train_dataset, batch_size=None, sampler=sampler,
                              generator=torch.Generator().manual_seed(args.seed + rank))
    mix_sampler = mix_loader = None
    if args.mix_set is not None:
        #its own order: the same seed would draw the same rows of both sets
        mix_sampler = TrainingSampler(mix_train_dataset, mix_batch_size, rank, world_size, args.seed + 1, args.mirror)
        mix_loader = DataLoader(mix_train_dataset, batch_size=None, sampler=mix_sampler)

    model_settings = dict(d_model=args.d_model, num_layers=args.layers, n_heads=args.heads,
                          position_templates=args.templates, mlp_hidden=mlp_hidden_width(args.d_model, args.mlp_ratio),
                          qk_ratio=args.qk_ratio[0] if len(args.qk_ratio) == 1 else args.qk_ratio, kv_heads=args.kv_heads,
                          value_bins=single.VALUE_BINS)
    raw_model = ChessTransformer(**model_settings).to(device)
    if resumed is not None:
        raw_model.load_state_dict(resumed["model"])
    master_weights = single.MasterWeights(raw_model)
    #DDP averages the compute model's gradients in bf16, as its bf16_compress_hook would
    model = DDP(master_weights.compute_model, device_ids=[device.index] if device.type == "cuda" else None)
    if not args.no_compile:
        model = torch.compile(model)
    optimizers = single.make_optimizers(raw_model, args.lr, device, args.weight_decay, args.muon_momentum, args.embed_lr_multiplier)
    #the schedulers first: making one sets its optimizer's learning rate for step 0, which the saved states then replace
    schedulers = [single.make_scheduler(optimizer, budget, args.schedule, args.decay_fraction) for optimizer in optimizers]
    best_val_loss = float("inf")
    if resumed is not None:
        for optimizer, state in zip(optimizers, resumed["optimizers"]):
            optimizer.load_state_dict(state)
        for scheduler, state in zip(schedulers, resumed["schedulers"]):
            scheduler.load_state_dict(state)
        budget.steps = resumed["steps"]
        if clock is not None:
            clock.steps, clock.seconds = resumed["steps"], resumed["seconds"]
        #one batch of each set a step, so the step fixes the epoch and the place in its order
        sampler.epoch, sampler.skip = divmod(resumed["steps"], len(sampler))
        if mix_sampler is not None:
            mix_sampler.epoch, mix_sampler.skip = divmod(resumed["steps"], len(mix_sampler))
        best_val_loss = resumed["best_val_loss"]
    if device.type == "cuda" and not args.no_compile:
        single.compile_muon_step(optimizers)  #after loading: the graph captures the learning rate tensor it finds
    metrics = evaluate(raw_model, val_loader, device, single.EVAL_BATCHES)
    writer = None
    try:
        if rank == 0:
            run_name = args.name or (single.folder_name(args.note) if args.note else
                                     f"ddp{world_size}_d{args.d_model}_L{args.layers}_h{args.heads}_"
                                     f"{budget.describe()}_{datetime.now():%Y%m%d_%H-%M-%S_%f}")
            run_dir = RUNS_DIR / run_name
            settings_path = run_dir / "settings.json"
            #purge_step hides what an earlier job logged past the state it left, whose steps this job trains again
            run_dir.mkdir(parents=True, exist_ok=args.resume)
            writer = SummaryWriter(str(run_dir), purge_step=budget.steps + 1 if args.resume else None)
            if resumed is not None:
                settings = json.loads(settings_path.read_text(encoding="utf-8"))
                settings["resumed"] = settings.get("resumed", []) + [dict(step=budget.steps, at=datetime.now().isoformat(timespec="seconds"),
                                                                          world_size=world_size, micro_batches=micro_batches, mirror=args.mirror)]
            else:
                settings = dict(
                    model_settings=model_settings, parameters=sum(p.numel() for p in raw_model.parameters()),
                    peak_learn_rate=args.lr, schedule=args.schedule, decay_fraction=args.decay_fraction, embed_lr_multiplier=args.embed_lr_multiplier, seed=args.seed,
                    value_loss_weight=args.value_weight, blunder_weight=args.blunder_weight, blunder_threshold=args.blunder_threshold,
                    weight_decay=args.weight_decay, muon_momentum=args.muon_momentum, mirror=args.mirror,
                    blunder_labels=train_dataset.worst_moves is not None,
                    budget=budget.describe(), batch_size=args.batch_size, per_gpu_batch_size=local_batch, micro_batches=micro_batches,
                    training_set=args.training_set, target_temperature=args.target_temperature,
                    result_weight=args.result_weight, visit_temperature=args.visit_temperature,
                    mix_set=args.mix_set, mix_share=mix_batch_size / args.batch_size if args.mix_set is not None else None,
                    mix_blunder_labels=mix_train_dataset.worst_moves is not None if args.mix_set is not None else None,
                    note=args.note, commit=single.current_commit(), trainer="ddp", world_size=world_size,
                    device=device.type, backend=dist.get_backend(), compile=not args.no_compile,
                    eval_every=args.eval_every, optimizer="muon", snapshot_every=args.snapshot_every,
                )
            settings_path.write_text(json.dumps(settings, indent=4), encoding="utf-8")
            mix_note = f" ({mix_batch_size} of it from {args.mix_set})" if args.mix_set is not None else ""
            parts_note = f" in {micro_batches} parts" if micro_batches > 1 else ""
            start = (f"resumed at step {budget.steps:,}, val loss {metrics[0]:.3f}" if resumed is not None
                     else f"untrained val loss {metrics[0]:.3f}")
            print(f"DDP: {world_size} processes on {device.type}, global batch {args.batch_size}{mix_note}, "
                  f"local batch {local_batch}{parts_note}; {start}")
            print(f"logging to {run_dir}")
            if not args.no_tensorboard:
                start_tensorboard()

        def save_best():
            save_checkpoint(raw_model, model_settings, run_dir / "model.pt")

        def save_resume(best_val_loss):
            save_resume_state(run_dir / RESUME_FILE, raw_model, optimizers, schedulers, budget, clock, best_val_loss)

        def save_snapshot():
            save_checkpoint(raw_model, model_settings, run_dir / f"snapshot_{budget.positions:_}.pt")

        train(model, master_weights, train_loader, val_loader, optimizers, schedulers, budget, clock, device,
              value_weight=args.value_weight, eval_every=args.eval_every, writer=writer,
              save_best=save_best if rank == 0 else None, blunder_weight=args.blunder_weight, best_val_loss=best_val_loss,
              save_resume=save_resume if rank == 0 and args.resume else None, resume_every_seconds=args.checkpoint_minutes * 60,
              mix_loader=mix_loader, save_snapshot=save_snapshot if rank == 0 and args.snapshot_every else None,
              snapshot_every=args.snapshot_every, micro_batches=micro_batches)
        if rank == 0:
            save_checkpoint(raw_model, model_settings, run_dir / "final.pt")
        metrics = evaluate(raw_model, val_loader, device)
        mix_metrics = evaluate(raw_model, val_loader_of(mix_val_dataset), device) if args.mix_set is not None else None
        if rank == 0:
            log_validation(writer, metrics, budget.steps, final=True)
            settings.update({f"final_{name}": value for name, value in zip(METRIC_TAGS, metrics)})
            if mix_metrics is not None:
                for name, value in zip(METRIC_TAGS, mix_metrics):
                    writer.add_scalar(f"final/mix_{name}", value, budget.steps)
                    settings[f"final_mix_{name}"] = value
            settings.update(train_seconds=clock.seconds, positions_seen=budget.positions,
                            gpu_seconds=clock.seconds * world_size if device.type == "cuda" else 0,
                            finished_at=datetime.now().isoformat(timespec="seconds"))
            settings_path.write_text(json.dumps(settings, indent=4), encoding="utf-8")
            (run_dir / RESUME_FILE).unlink(missing_ok=True)  #finished_at now tells a requeued job there is nothing left
            print(f"final val loss {metrics[0]:.3f}, accuracy {metrics[1]:.1%}; "
                  f"progress page: {write_progress_page()}")
        dist.barrier()
    finally:
        if writer is not None:
            writer.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--positions", type=int, required=True, help="positions across all GPUs, rounded down to full global batches")
    parser.add_argument("--batch-size", type=int, default=single.BATCH_SIZE, help="GLOBAL batch size, default %(default)s")
    parser.add_argument("--micro-batch", type=int,
                        help="positions a GPU puts through at once: the local batch goes in equal parts whose gradients add up "
                             "(gradient accumulation), the same step in less memory. Default the whole local batch")
    parser.add_argument("--name", help="new run folder name; existing folders are rejected (except by --resume)")
    parser.add_argument("--resume", action="store_true",
                        help="carry on the --name run from its last saved state, or start it")
    parser.add_argument("--checkpoint-minutes", type=float, default=10,
                        help="with --resume: wall minutes between two saved states. Default %(default)s")
    parser.add_argument("--snapshot-every", type=int, default=0,
                        help="steps between two copies of the weights kept in the run folder (snapshot_<positions>.pt), 0 for none")
    parser.add_argument("--note", default="")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--no-compile", action="store_true", help="use eager execution for debugging")
    parser.add_argument("--eval-every", type=int, default=single.EVAL_EVERY,
                        help="steps between two looks at the val set, default %(default)s")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="CPU/Gloo is for distributed smoke tests")
    parser.add_argument("--d-model", type=int, default=single.D_MODEL)
    parser.add_argument("--layers", type=int, default=single.NUM_LAYERS)
    parser.add_argument("--heads", type=int, default=single.N_HEADS)
    parser.add_argument("--templates", type=int, default=single.POSITION_TEMPLATES)
    parser.add_argument("--mlp-ratio", type=float)
    parser.add_argument("--kv-heads", type=int, help="key/value heads, each shared by a group of heads (grouped-query attention)")
    parser.add_argument("--qk-ratio", type=float, nargs="+", default=[1.0], help="query/key width as a multiple of the values', one for every block or one per block")
    parser.add_argument("--lr", type=float, default=single.PEAK_LEARN_RATE)
    parser.add_argument("--schedule", choices=("cosine", "hold", "hold_sqrt"), default=single.SCHEDULE)
    parser.add_argument("--decay-fraction", type=float, default=single.DECAY_FRACTION, help="for hold and hold_sqrt: the share of the run the decay takes")
    parser.add_argument("--embed-lr-multiplier", type=float, default=single.EMBED_LR_MULTIPLIER, help="the input embeddings' learning rate as a multiple of the rest's")
    parser.add_argument("--seed", type=int, default=single.SEED)
    parser.add_argument("--training-set", default=single.TRAINING_SET)
    parser.add_argument("--mix-set", help="a second training set that fills --mix-share of every batch")
    parser.add_argument("--mix-share", type=float, default=single.MIX_SHARE, help="share of every batch from --mix-set, default %(default)s")
    parser.add_argument("--result-weight", type=float, default=single.RESULT_WEIGHT,
                        help="self play sets: share of the value target from the game's result, the rest the search's score. Default %(default)s")
    parser.add_argument("--visit-temperature", type=float, default=single.VISIT_TEMPERATURE,
                        help="self play sets: sharpens the visit shares the policy learns, 0 = the most visited move. Default %(default)s")
    parser.add_argument("--target-temperature", type=float, default=single.TARGET_TEMPERATURE)
    parser.add_argument("--value-weight", type=float, default=single.VALUE_LOSS_WEIGHT)
    parser.add_argument("--blunder-weight", type=float, default=single.BLUNDER_WEIGHT,
                        help="weight of the push off labelled blunders, 0 for none")
    parser.add_argument("--blunder-threshold", type=float, default=single.BLUNDER_THRESHOLD,
                        help="a labelled move losing at least this much expected score is a blunder")
    parser.add_argument("--weight-decay", type=float, default=single.WEIGHT_DECAY, help="for Muon and AdamW both")
    parser.add_argument("--muon-momentum", type=float, default=single.MUON_MOMENTUM)
    parser.add_argument("--mirror", action="store_true",
                        help="show train positions (the mix set's too) without castling rights mirrored left-right at random, "
                             "a new draw every epoch that a --resume run repeats exactly. Val is never mirrored")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
        raise SystemExit("launch from the repository root with: uv run torchrun --standalone --nproc-per-node=2 -m training.train_ddp [options]")
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

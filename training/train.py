import argparse
import copy
import json
import math
import queue
import re
import subprocess
import threading
from collections.abc import Callable, Iterator, Sequence
from datetime import datetime
from itertools import islice
from pathlib import Path
from training.data import BLUNDER_THRESHOLD, RESULT_WEIGHT, VISIT_TEMPERATURE, MixedLoader, datasets_to_dataloaders, load_training_set
from model.checkpoint import ARTIFACTS_DIR, RUNS_DIR, load_checkpoint, save_checkpoint
from training.muon import BatchedMuon
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from training.budget import Budget
from training.compare_runs import write_progress_page
from model.encoding import NO_MOVE
from training.tensorboard_server import start_tensorboard
from model.model import VALUE_BINS, ChessTransformer, expected_score, mlp_hidden_width, value_bin_edges

D_MODEL = 128
NUM_LAYERS = 6
N_HEADS = 4
POSITION_TEMPLATES = 8
#the width of the gaussian target over the bins, in bin widths
VALUE_TARGET_SPREAD = 0.75

PEAK_LEARN_RATE = 6e-3
FINE_TUNE_LEARN_RATE = 1e-3
WARMUP_STEPS = 500
SCHEDULE = "cosine"
DECAY_FRACTION = 0.3
EMBED_LR_MULTIPLIER = 1.0
BATCH_SIZE = 1024

WEIGHT_DECAY = 0.1
MUON_MOMENTUM = 0.95

TRAIN_MINUTES_BUDGET = 4

EVAL_EVERY = 100
#reading the loss for the progress bar makes the cpu wait for the gpu
POSTFIX_EVERY = 20
EVAL_BATCHES = 20
SEED = 42

VALUE_LOSS_WEIGHT = 0.5

BLUNDER_WEIGHT = 4.0

#a move that gives up this much expected score gets 1/e of the best move's share of the policy target
TARGET_TEMPERATURE = 0.02

TRAINING_SET = "full_sf8_30M_mix"

MIX_SHARE = 0.25


def endless_batches(data_loader: DataLoader | MixedLoader) -> Iterator[tuple[torch.Tensor, ...]]:
    while True:
        yield from data_loader

def batches_on_device(batches: Iterator[tuple[torch.Tensor, ...]], device: torch.device, ahead: int = 4) -> Iterator[tuple[torch.Tensor, ...]]:
    ready = queue.Queue(maxsize=ahead)

    def to_device(tensor: torch.Tensor) -> torch.Tensor:
        #pinned memory is what lets the copy overlap the gpu's work
        return tensor.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else tensor.to(device)

    def fill() -> None:
        for batch in batches:
            ready.put(tuple(to_device(tensor) for tensor in batch))

    threading.Thread(target=fill, daemon=True).start()
    while True:
        yield ready.get()


#cross entropy against several target moves with a probability each, (B, moves); probability 0 adds nothing
def policy_loss(logits: torch.Tensor, target_moves: torch.Tensor, target_probabilities: torch.Tensor) -> torch.Tensor:
    log_probabilities = torch.log_softmax(logits, dim=-1).gather(1, target_moves)
    return -(target_probabilities * log_probabilities).sum(dim=-1).mean()

#the share of each bin under a gaussian around every target score, (B, bins) adding up to 1; the part outside 0 to 1 is left out
def value_target_bins(targets: torch.Tensor, bins: int) -> torch.Tensor:
    edges = value_bin_edges(bins, targets.device)
    spread = VALUE_TARGET_SPREAD / bins
    cumulative = 0.5 * (1 + torch.erf((edges - targets.unsqueeze(-1)) / (spread * math.sqrt(2))))
    return (cumulative[:, 1:] - cumulative[:, :-1]) / (cumulative[:, -1:] - cumulative[:, :1])

#targets are expected scores from 0 to 1
def value_loss(value_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    value_logits = value_logits.float()
    bins = value_logits.shape[-1]
    if bins == 1:
        return F.binary_cross_entropy_with_logits(value_logits.squeeze(-1), targets)
    return -(value_target_bins(targets, bins) * torch.log_softmax(value_logits, dim=-1)).sum(dim=-1).mean()

#unlikelihood of a position's blunders together; blunder_moves is (B, places), NO_MOVE where none
def blunder_loss(logits: torch.Tensor, blunder_moves: torch.Tensor) -> torch.Tensor:
    blunders = blunder_moves != NO_MOVE
    probabilities = torch.softmax(logits.float(), dim=-1).gather(1, blunder_moves.clamp(min=0)) * blunders
    return -torch.log1p(-probabilities.sum(dim=-1).clamp(max=1 - 1e-6)).mean()

def value_correct(value_logits: torch.Tensor, targets: torch.Tensor) -> int:
    return int(((expected_score(value_logits) > 0.5) == (targets > 0.5)).sum().item())

def value_error(value_logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (expected_score(value_logits) - targets).abs().sum().item()


#model is master_weights' compute model, maybe compiled
def train(
    model: nn.Module,
    master_weights: "MasterWeights",
    save_best_checkpoint: Callable[[], None],
    train_data_loader: DataLoader | MixedLoader,
    val_data_loader: DataLoader,
    optimizers: Sequence[torch.optim.Optimizer],
    schedulers: Sequence[torch.optim.lr_scheduler.LRScheduler],
    device: torch.device,
    best_val_loss: float,
    budget: Budget,
    writer: SummaryWriter,
    value_loss_weight: float,
    eval_every: int,
    blunder_weight: float,
    save_snapshot: Callable[[], None] | None,
    snapshot_every: int) -> None:

    #kept on the device: adding to it doesn't make the cpu wait for the gpu the way loss.item() does
    loss_total = torch.zeros((), device=device)
    penalty_total = torch.zeros((), device=device)
    gradient_norm_total = torch.zeros((), device=device)
    total_examples = 0
    interval_steps = 0

    model.train()

    progress = tqdm(total=budget.total, desc="train", dynamic_ncols=True, unit="s" if budget.timed else "it",
                    bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f} [{elapsed}<{remaining}{postfix}]")

    for batch, target_moves, target_probabilities, values, blunders in batches_on_device(endless_batches(train_data_loader), device):
        logits, eval_logits = model(batch)

        loss = policy_loss(logits, target_moves, target_probabilities) + value_loss_weight * value_loss(eval_logits, values)
        penalty = blunder_loss(logits, blunders) if blunder_weight else torch.zeros((), device=device)

        master_weights.zero_grad()

        (loss + blunder_weight * penalty).backward()

        master_weights.gradients_to_master()
        gradient_norm_total += torch.nn.utils.clip_grad_norm_(master_weights.parameters, 1.0)
        interval_steps += 1

        for optimizer in optimizers:
            optimizer.step()
        master_weights.weights_to_compute()
        progress.update(budget.step_done())
        for scheduler in schedulers:
            scheduler.step()
        step = budget.steps

        loss_total += loss.detach() * len(batch)
        penalty_total += penalty.detach() * len(batch)
        total_examples += len(batch)

        if step % POSTFIX_EVERY == 0:
            progress.set_postfix(loss=f"{loss_total.item() / total_examples:.3f}")

        if step % eval_every == 0 or budget.finished():
            train_loss = loss_total.item() / total_examples #waits for the gpu, so the pause starts when the steps are done
            budget.pause()
            val_loss, val_accuracy, val_value_loss, val_value_accuracy, val_value_error = evaluate_model(model, val_data_loader, device, max_batches=EVAL_BATCHES)
            model.train()

            tqdm.write(f"step {step} ({budget.positions:,} positions, {budget.seconds / 60:.1f} min): train loss {train_loss:.3f}, val loss {val_loss:.3f}, val accuracy {val_accuracy:.1%}, value loss {val_value_loss:.3f}, value accuracy {val_value_accuracy:.1%}, value error {val_value_error:.4f}")

            writer.add_scalar("loss/train", train_loss, step)
            if blunder_weight:
                writer.add_scalar("loss/train_blunder", penalty_total.item() / total_examples, step)
            writer.add_scalar("loss/val", val_loss, step)
            writer.add_scalar("accuracy/val", val_accuracy, step)
            writer.add_scalar("loss/val_value", val_value_loss, step)
            writer.add_scalar("accuracy/val_value", val_value_accuracy, step)
            writer.add_scalar("error/val_value", val_value_error, step)
            writer.add_scalar("learning_rate", schedulers[0].get_last_lr()[0], step)
            writer.add_scalar("gradient_norm/train", gradient_norm_total.item() / interval_steps, step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_best_checkpoint()

            loss_total.zero_()
            penalty_total.zero_()
            gradient_norm_total.zero_()
            total_examples = 0
            interval_steps = 0
            budget.resume()

        if save_snapshot is not None and step % snapshot_every == 0:
            save_snapshot()

        if budget.finished():
            break

    progress.close()


#move loss and accuracy, then value loss, accuracy and mean error, over the val loader or its first max_batches
def evaluate_model(
    model: nn.Module,
    test_data_loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
    show_progress: bool = True) -> tuple[float, float, float, float, float]:

    epoch_loss_total = 0.0
    total_correct_preds = 0
    value_loss_total = 0.0
    value_correct_preds = 0
    value_error_total = 0.0
    total_examples = 0

    model.eval()

    with torch.no_grad():

        num_batches = len(test_data_loader) if max_batches is None else min(max_batches, len(test_data_loader))
        progress = tqdm(islice(test_data_loader, num_batches), total=num_batches, desc="val", leave=False, dynamic_ncols=True, disable=not show_progress)

        for batch, target_moves, target_probabilities, values, _ in progress:
            target_moves = target_moves.to(device)
            target_probabilities = target_probabilities.to(device)
            values = values.to(device)
            batch = batch.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, value_logits = model(batch)

                model_preds = torch.softmax(logits, dim=-1).argmax(dim=-1)

                loss = policy_loss(logits, target_moves, target_probabilities)
                batch_value_loss = value_loss(value_logits, values)

            #accuracy is against the first target move, the best one
            epoch_loss_total += loss.item() * len(batch)
            total_correct_preds += (model_preds == target_moves[:, 0]).sum().item()
            value_loss_total += batch_value_loss.item() * len(batch)
            value_correct_preds += value_correct(value_logits, values)
            value_error_total += value_error(value_logits, values)
            total_examples += len(batch)

            progress.set_postfix(
                loss=f"{epoch_loss_total / total_examples:.3f}",
                accuracy=f"{total_correct_preds / total_examples:.1%}"
            )

    return epoch_loss_total / total_examples, total_correct_preds / total_examples, value_loss_total / total_examples, value_correct_preds / total_examples, value_error_total / total_examples


def load_dataloaders(training_set: str = TRAINING_SET, target_temperature: float = TARGET_TEMPERATURE, seed: int = SEED, batch_size: int = BATCH_SIZE,
                     result_weight: float = RESULT_WEIGHT, visit_temperature: float = VISIT_TEMPERATURE,
                     blunder_threshold: float = BLUNDER_THRESHOLD, teacher_weight: float = 0.0, mirror: bool = False) -> tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = load_training_set(training_set, target_temperature, result_weight, visit_temperature, blunder_threshold, teacher_weight)
    warm_file_cache(train_dataset, val_dataset)
    return datasets_to_dataloaders(train_dataset, val_dataset, train_batch_size=batch_size, val_batch_size=batch_size, seed=seed, mirror=mirror)

def warm_file_cache(*datasets) -> None:
    tensors = [tensor for dataset in datasets for tensor in dataset.tensors()]
    total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    chunk = 1 << 28

    progress = tqdm(total=total_bytes, desc="reading the training files into memory", unit="B", unit_scale=True, dynamic_ncols=True, leave=False)
    for tensor in tensors:
        flat = tensor.reshape(-1).view(torch.uint8)
        for start in range(0, len(flat), chunk):
            flat[start:start + chunk].sum() #touching the pages is what reads them
            progress.update(min(chunk, len(flat) - start))
    progress.close()


#progress is how much of the run after the warmup is done, from 0 to 1
def lr_multiplier(step: int, progress: float, schedule: str = SCHEDULE, decay_fraction: float = DECAY_FRACTION) -> float:
    if step <= WARMUP_STEPS:
        return (step + 1) / WARMUP_STEPS
    if schedule == "cosine":
        return 0.5 * (1 + math.cos(math.pi * progress))
    if schedule == "hold":
        return min(1.0, (1 - progress) / decay_fraction)
    if schedule == "hold_sqrt":
        return 1 - math.sqrt(min(1.0, max(0.0, (progress - 1 + decay_fraction) / decay_fraction)))
    if schedule == "constant":
        return 1.0
    raise ValueError(f"unknown schedule {schedule}")


def make_optimizers(model: ChessTransformer, learn_rate: float, device: torch.device, weight_decay: float = WEIGHT_DECAY,
                    muon_momentum: float = MUON_MOMENTUM, embed_lr_multiplier: float = EMBED_LR_MULTIPLIER) -> list[torch.optim.Optimizer]:
    if embed_lr_multiplier <= 0:
        raise ValueError(f"the embeddings' learning rate multiplier must be above 0, got {embed_lr_multiplier}")
    block_matrices = [parameter for parameter in model.encoder.parameters() if parameter.ndim == 2]
    #weight decay would pull the QK-norm gains and the input skip's scales towards 0
    qk_gains = [norm.weight for block in model.encoder for norm in (block.q_norm, block.k_norm)]
    skip_scales = [scale for block in model.encoder for scale in (block.residual_scale, block.input_scale)]
    undecayed = qk_gains + skip_scales
    embeddings = [model.embed.weight, model.position_embed, model.square_features.table]
    elsewhere = {id(parameter) for parameter in block_matrices + undecayed + embeddings}
    rest = [parameter for parameter in model.parameters() if id(parameter) not in elsewhere]
    #a tensor: compile_muon_step's graph would bake a float in and recompile whenever the schedule changes it
    muon = BatchedMuon(block_matrices, lr=torch.tensor(learn_rate, device=device), weight_decay=weight_decay, momentum=muon_momentum, adjust_lr_fn="match_rms_adamw") #type: ignore[arg-type]
    #the embeddings' weight decay is divided by their multiplier, so a step decays them as much as the rest
    groups = [{"params": rest}, {"params": undecayed, "weight_decay": 0.0},
              {"params": embeddings, "lr": learn_rate * embed_lr_multiplier, "weight_decay": weight_decay / embed_lr_multiplier}]
    #AdamW comes first: the training loop logs its learning rate
    return [torch.optim.AdamW(groups, lr=learn_rate, weight_decay=weight_decay, fused=device.type == "cuda"), muon]

#call after making the schedulers, which wrap optimizer.step and need it to still be a method
def compile_muon_step(optimizers: list[torch.optim.Optimizer]) -> None:
    for optimizer in optimizers:
        if isinstance(optimizer, torch.optim.Muon):
            optimizer.step = torch.compile(optimizer.step, mode="reduce-overhead") #type: ignore[method-assign]

#a bf16 copy of the model trains while the optimizers step the fp32 model, whose weights keep updates too small for bf16; checkpoints save model
class MasterWeights:
    def __init__(self, model: ChessTransformer):
        self.model = model
        self.compute_model = copy.deepcopy(model).half_except_heads()
        self.parameters = list(model.parameters())
        self.compute_parameters = list(self.compute_model.parameters())
        for parameter in self.parameters:
            parameter.grad = torch.zeros_like(parameter)

    def zero_grad(self) -> None:
        self.compute_model.zero_grad()

    def gradients_to_master(self) -> None:
        torch._foreach_copy_([parameter.grad for parameter in self.parameters], [parameter.grad for parameter in self.compute_parameters]) #type: ignore

    @torch.no_grad()
    def weights_to_compute(self) -> None:
        torch._foreach_copy_(self.compute_parameters, self.parameters) #type: ignore


def current_commit() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("+" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return ""


def folder_name(note: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "_", note).strip("_")[:60]

def main(
    positions_budget: int | None = None,
    minutes_budget: float = TRAIN_MINUTES_BUDGET,
    run_name: str | None = None,
    note: str = "",
    d_model: int = D_MODEL,
    num_layers: int = NUM_LAYERS,
    n_heads: int = N_HEADS,
    peak_learn_rate: float | None = None,
    position_templates: int = POSITION_TEMPLATES,
    training_set: str = TRAINING_SET,
    value_loss_weight: float = VALUE_LOSS_WEIGHT,
    schedule: str = SCHEDULE,
    decay_fraction: float = DECAY_FRACTION,
    seed: int = SEED,
    mlp_ratio: float | None = None,
    target_temperature: float = TARGET_TEMPERATURE,
    qk_ratio: float | list[float] = 1.0,
    kv_heads: int | None = None,
    batch_size: int = BATCH_SIZE,
    init_checkpoint: Path | None = None,
    mix_set: str | None = None,
    mix_share: float = MIX_SHARE,
    result_weight: float = RESULT_WEIGHT,
    visit_temperature: float = VISIT_TEMPERATURE,
    eval_every: int = EVAL_EVERY,
    compile_mode: str = "default",
    blunder_weight: float = BLUNDER_WEIGHT,
    value_bins: int = VALUE_BINS,
    blunder_threshold: float = BLUNDER_THRESHOLD,
    weight_decay: float = WEIGHT_DECAY,
    muon_momentum: float = MUON_MOMENTUM,
    snapshot_every: int = 0,
    teacher_weight: float = 0.0,
    embed_lr_multiplier: float = EMBED_LR_MULTIPLIER,
    mirror: bool = False):

    initial_model = None
    if init_checkpoint is not None:
        initial_model, model_settings = load_checkpoint(init_checkpoint, torch.device("cpu"))
    else:
        model_settings = {"d_model": d_model, "num_layers": num_layers, "n_heads": n_heads, "position_templates": position_templates, "mlp_hidden": mlp_hidden_width(d_model, mlp_ratio), "qk_ratio": qk_ratio, "kv_heads": kv_heads, "value_bins": value_bins}
    if mix_set is not None and not 0 < mix_share < 1:
        raise ValueError(f"--mix-share is the share of every batch from {mix_set}, between 0 and 1, got {mix_share}")
    if peak_learn_rate is None:
        peak_learn_rate = FINE_TUNE_LEARN_RATE if init_checkpoint is not None else PEAK_LEARN_RATE

    budget = Budget(batch_size, positions=positions_budget) if positions_budget is not None else Budget(batch_size, minutes=minutes_budget)
    if run_name is None:
        run_name = folder_name(note) if note else f"d{model_settings['d_model']}_L{model_settings['num_layers']}_h{model_settings['n_heads']}_lr{peak_learn_rate:g}_{budget.describe()}_{datetime.now():%H-%M-%S}"
        #runs started in the same second (several gpus) must not share a folder: mkdir claims the name
        for number in range(1, 100):
            try:
                (RUNS_DIR / run_name).mkdir(parents=True)
                break
            except FileExistsError:
                run_name = run_name.rsplit("_", 1)[0] + f"_{number}" if number > 1 else f"{run_name}_1"

    if not budget.timed and budget.max_steps <= WARMUP_STEPS:
        raise ValueError(f"{positions_budget:,} positions is {budget.max_steps} steps, it has to be more than the {WARMUP_STEPS} warmup steps")
    if budget.timed and budget.max_seconds < 60:
        raise ValueError(f"{minutes_budget} minutes is too short, the warmup alone takes a good part of a minute")

    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}")

    mix_batch_size = round(batch_size * mix_share) if mix_set is not None else 0
    train_data_loader, val_data_loader = load_dataloaders(training_set, target_temperature, seed, batch_size - mix_batch_size, result_weight, visit_temperature, blunder_threshold, teacher_weight, mirror)
    print(f"{training_set}: {len(train_data_loader.dataset):,} train positions, {len(val_data_loader.dataset):,} val positions")
    mix_val_loader = None
    if mix_set is not None:
        mix_train_loader, mix_val_loader = load_dataloaders(mix_set, target_temperature, seed + 1, mix_batch_size, mirror=mirror)
        train_data_loader = MixedLoader(train_data_loader, mix_train_loader)
        print(f"{mix_set}: {mix_batch_size} of every {batch_size} positions")

    model = (initial_model if initial_model is not None else ChessTransformer(**model_settings)).to(device)
    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"model: {model_settings}, {num_parameters:,} parameters")

    master_weights = MasterWeights(model)
    compiled_model = torch.compile(master_weights.compute_model, mode=compile_mode)
    optimizers = make_optimizers(model, peak_learn_rate, device, weight_decay, muon_momentum, embed_lr_multiplier)

    warmup_end = 0.0
    def current_lr_multiplier(step: int) -> float:
        nonlocal warmup_end
        if step <= WARMUP_STEPS:
            warmup_end = budget.progress()
            return lr_multiplier(step, 0.0, schedule, decay_fraction)
        return lr_multiplier(step, (budget.progress() - warmup_end) / (1 - warmup_end), schedule, decay_fraction)

    schedulers = [torch.optim.lr_scheduler.LambdaLR(optimizer, current_lr_multiplier) for optimizer in optimizers]
    if device.type == "cuda":
        compile_muon_step(optimizers)

    val_loss, val_accuracy, val_value_loss, val_value_accuracy, val_value_error = evaluate_model(compiled_model, val_data_loader, device, max_batches=EVAL_BATCHES)
    print(f"{'starting point' if init_checkpoint is not None else 'untrained'}: val loss {val_loss:.3f}, val accuracy {val_accuracy:.1%}, value loss {val_value_loss:.3f}, value accuracy {val_value_accuracy:.1%}, value error {val_value_error:.4f}")

    print(f"budget: {budget.describe()}, batches of {batch_size} positions, evaluating every {eval_every} steps")
    run_dir = RUNS_DIR / run_name
    print(f"logging to {run_dir.relative_to(ARTIFACTS_DIR.parent)}")
    writer = SummaryWriter(log_dir=str(run_dir))

    run_settings = {
        "model_settings": model_settings,
        "parameters": num_parameters,
        "peak_learn_rate": peak_learn_rate,
        "schedule": schedule,
        "decay_fraction": decay_fraction,
        "embed_lr_multiplier": embed_lr_multiplier,
        "seed": seed,
        "value_loss_weight": value_loss_weight,
        "blunder_weight": blunder_weight,
        "blunder_threshold": blunder_threshold,
        "weight_decay": weight_decay,
        "muon_momentum": muon_momentum,
        "teacher_weight": teacher_weight,
        "mirror": mirror,
        "blunder_labels": getattr(train_data_loader.dataset, "worst_moves", None) is not None,
        "budget": budget.describe(),
        "batch_size": batch_size,
        "training_set": training_set,
        "target_temperature": target_temperature,
        "init_checkpoint": str(init_checkpoint) if init_checkpoint is not None else None,
        "mix_set": mix_set,
        "mix_share": mix_share if mix_set is not None else None,
        "result_weight": result_weight,
        "visit_temperature": visit_temperature,
        "eval_every": eval_every,
        "compile_mode": compile_mode,
        "snapshot_every": snapshot_every,
        "optimizer": "muon",
        "note": note,
        "commit": current_commit()
    }
    (run_dir / "settings.json").write_text(json.dumps(run_settings, indent=4))

    #from model, not compiled_model: the compiled wrapper prefixes every name with "_orig_mod."
    def save_best_checkpoint() -> None:
        save_checkpoint(model, model_settings, run_dir / "model.pt")

    def save_snapshot() -> None:
        save_checkpoint(model, model_settings, run_dir / f"snapshot_{budget.positions:_}.pt")

    train(compiled_model, master_weights, save_best_checkpoint, train_data_loader, val_data_loader, optimizers, schedulers, device, best_val_loss=val_loss, budget=budget, writer=writer, value_loss_weight=value_loss_weight,
          eval_every=eval_every, blunder_weight=blunder_weight, save_snapshot=save_snapshot if snapshot_every else None, snapshot_every=snapshot_every)

    save_checkpoint(model, model_settings, run_dir / "final.pt")

    val_loss, val_accuracy, val_value_loss, val_value_accuracy, val_value_error = evaluate_model(compiled_model, val_data_loader, device)
    print(f"final, on all {len(val_data_loader.dataset):,} val positions: val loss {val_loss:.3f}, val accuracy {val_accuracy:.1%}, value loss {val_value_loss:.3f}, value accuracy {val_value_accuracy:.1%}, value error {val_value_error:.4f}")

    writer.add_scalar("final/val_loss", val_loss, budget.steps)
    writer.add_scalar("final/val_accuracy", val_accuracy, budget.steps)
    writer.add_scalar("final/val_value_loss", val_value_loss, budget.steps)
    writer.add_scalar("final/val_value_accuracy", val_value_accuracy, budget.steps)
    writer.add_scalar("final/val_value_error", val_value_error, budget.steps)

    if mix_val_loader is not None:
        mix_val_loss, mix_val_accuracy, mix_val_value_loss, mix_val_value_accuracy, mix_val_value_error = evaluate_model(compiled_model, mix_val_loader, device)
        print(f"final, on all {len(mix_val_loader.dataset):,} val positions of {mix_set}: val loss {mix_val_loss:.3f}, val accuracy {mix_val_accuracy:.1%}, value loss {mix_val_value_loss:.3f}, value accuracy {mix_val_value_accuracy:.1%}, value error {mix_val_value_error:.4f}")
        writer.add_scalar("final/mix_val_loss", mix_val_loss, budget.steps)
        writer.add_scalar("final/mix_val_accuracy", mix_val_accuracy, budget.steps)
        run_settings.update(final_mix_val_loss=mix_val_loss, final_mix_val_accuracy=mix_val_accuracy,
                            final_mix_val_value_loss=mix_val_value_loss, final_mix_val_value_accuracy=mix_val_value_accuracy,
                            final_mix_val_value_error=mix_val_value_error)
    writer.close()

    run_settings.update(
        final_val_loss=val_loss,
        final_val_accuracy=val_accuracy,
        final_val_value_loss=val_value_loss,
        final_val_value_accuracy=val_value_accuracy,
        final_val_value_error=val_value_error,
        train_seconds=budget.seconds,
        positions_seen=budget.positions,
        finished_at=datetime.now().isoformat(timespec="seconds")
    )
    (run_dir / "settings.json").write_text(json.dumps(run_settings, indent=4))

    print(f"progress page: {write_progress_page()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train the chess transformer")
    budget_flags = parser.add_mutually_exclusive_group()
    budget_flags.add_argument("--minutes", type=float, default=TRAIN_MINUTES_BUDGET, help="minutes of training, compiling and evaluating not counted, default %(default)s")
    budget_flags.add_argument("--positions", type=int, help="train on this many positions instead of for a number of minutes")
    parser.add_argument("--note", default="", help="what this run tries, shown on the progress page")
    parser.add_argument("--name", help="name of this run in tensorboard, default the date and time")
    parser.add_argument("--no-tensorboard", action="store_true", help="do not start tensorboard or open the browser")
    parser.add_argument("--d-model", type=int, default=D_MODEL, help="width of the model, default %(default)s")
    parser.add_argument("--layers", type=int, default=NUM_LAYERS, help="number of transformer blocks, default %(default)s")
    parser.add_argument("--heads", type=int, default=N_HEADS, help="attention heads, d-model has to divide by it, default %(default)s")
    parser.add_argument("--lr", type=float, help=f"peak learning rate, default {PEAK_LEARN_RATE:g} ({FINE_TUNE_LEARN_RATE:g} with --init-checkpoint)")
    parser.add_argument("--templates", type=int, default=POSITION_TEMPLATES, help="attention bias templates the blocks mix, default %(default)s")
    parser.add_argument("--training-set", default=TRAINING_SET, help="a labelled training set in processed_data/, default %(default)s")
    parser.add_argument("--value-weight", type=float, default=VALUE_LOSS_WEIGHT, help="weight of the value head's loss next to the move loss, default %(default)s")
    parser.add_argument("--blunder-weight", type=float, default=BLUNDER_WEIGHT, help="weight of the push off labelled blunders, 0 for none. Default %(default)s")
    parser.add_argument("--blunder-threshold", type=float, default=BLUNDER_THRESHOLD, help="a labelled move losing at least this much expected score is a blunder. Default %(default)s")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="for Muon and AdamW both, default %(default)s")
    parser.add_argument("--muon-momentum", type=float, default=MUON_MOMENTUM, help="default %(default)s")
    parser.add_argument("--schedule", choices=["cosine", "hold", "hold_sqrt", "constant"], default=SCHEDULE,
                        help="the learning rate after the warmup: cosine to zero over the run; hold the peak, then a straight line to zero over the last --decay-fraction; "
                             "hold_sqrt the same with 1 - sqrt(t), t the share of the decay done; constant the peak throughout. Default %(default)s")
    parser.add_argument("--decay-fraction", type=float, default=DECAY_FRACTION, help="for hold and hold_sqrt: the share of the run the decay takes, default %(default)s")
    parser.add_argument("--embed-lr-multiplier", type=float, default=EMBED_LR_MULTIPLIER, help="the input embeddings' (token, position, square features) learning rate as a multiple of the rest's. Default %(default)s")
    parser.add_argument("--seed", type=int, default=SEED, help="default %(default)s")
    parser.add_argument("--mlp-ratio", type=float, help="hidden size of the swiglu mlp as a multiple of d-model, rounded down to a multiple of 32. Default 8/3")
    parser.add_argument("--target-temperature", type=float, default=TARGET_TEMPERATURE, help="how stockfish's moves share the target, see TARGET_TEMPERATURE. Default %(default)s")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="positions per step, for train and val. Default %(default)s")
    parser.add_argument("--kv-heads", type=int, help="key/value heads, each shared by a group of heads (grouped-query attention). Default one per head")
    parser.add_argument("--value-bins", type=int, default=VALUE_BINS, help="bins of the expected score the eval head predicts a distribution over, 1 for a single logit. Default %(default)s")
    parser.add_argument("--qk-ratio", type=float, nargs="+", default=[1.0], help="query/key width as a multiple of the values', one for every block or one per block, e.g. 0.25 0.5 0.5 0.75. Default 1")
    parser.add_argument("--init-checkpoint", type=Path, help="fine-tune from this model.pt; the model is the checkpoint's size and the size flags are ignored")
    parser.add_argument("--mix-set", help="a second training set that fills --mix-share of every batch")
    parser.add_argument("--mix-share", type=float, default=MIX_SHARE, help="share of every batch from --mix-set, default %(default)s")
    parser.add_argument("--result-weight", type=float, default=RESULT_WEIGHT, help="self play sets: share of the value target from the game's result, the rest the search's score. Default %(default)s")
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY, help="steps between two looks at the val set, default %(default)s")
    parser.add_argument("--snapshot-every", type=int, default=0, help="steps between two copies of the weights kept in the run folder (snapshot_<positions>.pt), 0 for none")
    parser.add_argument("--compile-mode", choices=["default", "max-autotune-no-cudagraphs"], default="default", help="torch.compile's mode, default %(default)s")
    parser.add_argument("--teacher-weight", type=float, default=0.0, help="share of the train targets from a bigger model's; val keeps stockfish's. Default %(default)s")
    parser.add_argument("--visit-temperature", type=float, default=VISIT_TEMPERATURE, help="self play sets: sharpens the visit shares, 1 as recorded, 0 the most visited move alone. Default %(default)s")
    parser.add_argument("--mirror", action="store_true", help="show train positions without castling rights mirrored left-right at random, a new draw every pass over the set. Val is never mirrored")
    arguments = vars(parser.parse_args())

    if not arguments.pop("no_tensorboard"):
        start_tensorboard()

    qk_ratio = arguments["qk_ratio"]
    arguments["qk_ratio"] = qk_ratio[0] if len(qk_ratio) == 1 else qk_ratio
    renamed = {"minutes": "minutes_budget", "positions": "positions_budget", "name": "run_name", "layers": "num_layers", "heads": "n_heads",
               "lr": "peak_learn_rate", "templates": "position_templates", "value_weight": "value_loss_weight"}
    main(**{renamed.get(flag, flag): value for flag, value in arguments.items()})

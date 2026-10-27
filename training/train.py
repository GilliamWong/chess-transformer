import argparse
import copy
import json
import queue
import re
import subprocess
import threading
from collections.abc import Callable, Iterator, Sequence
from datetime import datetime
from itertools import islice
from pathlib import Path
from training.data import ARTIFACTS_DIR, BLUNDER_THRESHOLD, RESULT_WEIGHT, VISIT_TEMPERATURE, MixedLoader, datasets_to_dataloaders, load_training_set
from training.muon import BatchedMuon
import torch
import torch.nn.functional as F
import math
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tools.budget import Budget
from tools.checkpoint import RUNS_DIR, load_checkpoint, save_checkpoint
from tools.compare_runs import write_progress_page
from tools.label_with_stockfish import NO_MOVE
from tools.tensorboard_server import start_tensorboard
from model.model import ChessTransformer, expected_score, mlp_hidden_width, value_bin_edges

#the model, each of these has a command line flag (python -m training.train --help)
D_MODEL = 128
#at equal compute a d128 reached val loss 1.904 with 6 layers, 1.937 with 4 and 1.901 with 8
NUM_LAYERS = 6
N_HEADS = 4
POSITION_TEMPLATES = 8
#the eval head's bins over the expected score, 1 for the older single logit
VALUE_BINS = 128
#the target over the bins is a gaussian around the score this many bin widths wide
VALUE_TARGET_SPREAD = 0.75

PEAK_LEARN_RATE = 6e-3
#the peak for a fine-tune (--init-checkpoint): at PEAK_LEARN_RATE the warmup undid much of what the start model knew
FINE_TUNE_LEARN_RATE = 1e-3
WARMUP_STEPS = 500
#after the warmup:
#   cosine    half a cosine from the peak to zero over the whole run
#   hold      the peak, then a straight line to zero over the last DECAY_FRACTION of the run
#   constant  the peak throughout
SCHEDULE = "cosine"
#4-minute d128 runs: 0.2 was the worst of 0.2 / 0.4 / 0.6, and 0.6 tied cosine
DECAY_FRACTION = 0.3
#the input embeddings' learning rate (token, position, square features) as a multiple of everything else's
EMBED_LR_MULTIPLIER = 1.0
BATCH_SIZE = 1024

#for Muon and AdamW both: 0.1 beat 0.01 by 0.007 val loss on two seeds (d128 L6, 10M positions)
WEIGHT_DECAY = 0.1
#torch's Muon default
MUON_MOMENTUM = 0.95

#minutes of pure training (compiling and evaluating not counted) unless --minutes or --positions is given
TRAIN_MINUTES_BUDGET = 4

EVAL_EVERY = 100  #steps between two looks at the val set
POSTFIX_EVERY = 20 #reading the loss for the progress bar makes the cpu wait for the gpu
EVAL_BATCHES = 20 #val batches per look; the whole val set is used at the end
SEED = 42 #of the first weights and of the training order

#at 1 the 128-bin eval head cost the policy 0.02 val loss on a d128, at 0.5 nothing, for the same value error
VALUE_LOSS_WEIGHT = 0.5

#weight of blunder_loss: +27 Elo at 4 against 0 on a d128 (8 was no better). A set without blunder labels trains as if 0
BLUNDER_WEIGHT = 4.0

#a move that gives up this much expected score gets 1/e of the best move's share of the policy target
TARGET_TEMPERATURE = 0.02

#30M positions of random whole games of the stockfish labelled full_sf8_20260922, in its mix:
#lichess elite 68%, fishtest 29%, otb 3%
TRAINING_SET = "full_sf8_30M_mix"

#with --mix-set, this share of every batch comes from that set
MIX_SHARE = 0.25


def endless_batches(data_loader: DataLoader | MixedLoader) -> Iterator[tuple[torch.Tensor, ...]]:
    while True:
        yield from data_loader

#a background thread copies the next batches to the device while the gpu works: 39,500 -> 44,200 positions/s
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


#cross entropy against several target moves with a probability each, (B, moves); places at probability 0 add nothing
def policy_loss(logits: torch.Tensor, target_moves: torch.Tensor, target_probabilities: torch.Tensor) -> torch.Tensor:
    log_probabilities = torch.log_softmax(logits, dim=-1).gather(1, target_moves)
    return -(target_probabilities * log_probabilities).sum(dim=-1).mean()

#the share of each bin under a gaussian around every target score, (B, bins) adding up to 1: the part of the
#gaussian outside 0 to 1 is left out
def value_target_bins(targets: torch.Tensor, bins: int) -> torch.Tensor:
    edges = value_bin_edges(bins, targets.device)
    spread = VALUE_TARGET_SPREAD / bins
    cumulative = 0.5 * (1 + torch.erf((edges - targets.unsqueeze(-1)) / (spread * math.sqrt(2))))
    return (cumulative[:, 1:] - cumulative[:, :-1]) / (cumulative[:, -1:] - cumulative[:, :1])

#the target is an expected score from 0 to 1: one logit learns its logit, bins learn the gaussian around it
def value_loss(value_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    value_logits = value_logits.float()
    bins = value_logits.shape[-1]
    if bins == 1:
        return F.binary_cross_entropy_with_logits(value_logits.squeeze(-1), targets)
    return -(value_target_bins(targets, bins) * torch.log_softmax(value_logits, dim=-1)).sum(dim=-1).mean()

#unlikelihood: -log(1 - the probability given to a position's blunders together), 0 without any.
#blunder_moves is (B, places), NO_MOVE where there is none
def blunder_loss(logits: torch.Tensor, blunder_moves: torch.Tensor) -> torch.Tensor:
    blunders = blunder_moves != NO_MOVE
    probabilities = torch.softmax(logits.float(), dim=-1).gather(1, blunder_moves.clamp(min=0)) * blunders
    return -torch.log1p(-probabilities.sum(dim=-1).clamp(max=1 - 1e-6)).mean()

#positions where the value head gets whether the side to move is ahead (score above a half)
def value_correct(value_logits: torch.Tensor, targets: torch.Tensor) -> int:
    return int(((expected_score(value_logits) > 0.5) == (targets > 0.5)).sum().item())

#how far the head's score is from the target, added up: comparable between heads, which their losses are not
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
    value_loss_weight: float = VALUE_LOSS_WEIGHT,
    eval_every: int = EVAL_EVERY,
    blunder_weight: float = BLUNDER_WEIGHT,
    save_snapshot: Callable[[], None] | None = None,
    snapshot_every: int = 0) -> None:

    #kept on the device: adding to it doesn't make the cpu wait for the gpu the way loss.item() does
    loss_total = torch.zeros((), device=device)
    penalty_total = torch.zeros((), device=device)
    gradient_norm_total = torch.zeros((), device=device)
    total_examples = 0
    interval_steps = 0

    model.train()

    #dynamic_ncols: otherwise a resized terminal turns every update into a new line.
    #the bar counts steps, or seconds when the budget is in minutes
    progress = tqdm(total=budget.total, desc="train", dynamic_ncols=True, unit="s" if budget.timed else "it",
                    bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f} [{elapsed}<{remaining}{postfix}]")

    for batch, target_moves, target_probabilities, values, blunders in batches_on_device(endless_batches(train_data_loader), device):
        logits, eval_logits = model(batch)

        loss = policy_loss(logits, target_moves, target_probabilities) + value_loss_weight * value_loss(eval_logits, values)
        penalty = blunder_loss(logits, blunders) if blunder_weight else torch.zeros((), device=device)

        master_weights.zero_grad()

        (loss + blunder_weight * penalty).backward()

        master_weights.gradients_to_master()
        #before clipping: a runaway (an attention head collapsing, say) shows here first
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

        #the last step is always evaluated, so the final weights can become the checkpoint
        if step % eval_every == 0 or budget.finished():
            train_loss = loss_total.item() / total_examples #waits for the gpu, so the pause starts when the steps are done
            budget.pause() #evaluating doesn't count towards a budget in minutes
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


#move loss and accuracy, then value loss, accuracy and mean error, over the val loader or its first max_batches.
#the val loader isn't shuffled, so these are the same positions every time
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
        #show_progress=False where several processes evaluate at once (training/train_ddp.py)
        progress = tqdm(islice(test_data_loader, num_batches), total=num_batches, desc="val", leave=False, dynamic_ncols=True, disable=not show_progress)

        #no blunder penalty: val loss stays the move loss alone
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

            #accuracy is against the first target move, stockfish's best
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

    return (epoch_loss_total / total_examples), (total_correct_preds / total_examples), (value_loss_total / total_examples), (value_correct_preds / total_examples), (value_error_total / total_examples)


def load_dataloaders(training_set: str = TRAINING_SET, target_temperature: float = TARGET_TEMPERATURE, seed: int = SEED, batch_size: int = BATCH_SIZE,
                     result_weight: float = RESULT_WEIGHT, visit_temperature: float = VISIT_TEMPERATURE,
                     blunder_threshold: float = BLUNDER_THRESHOLD, teacher_weight: float = 0.0) -> tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = load_training_set(training_set, target_temperature, result_weight, visit_temperature, blunder_threshold, teacher_weight)
    warm_file_cache(train_dataset, val_dataset)
    return datasets_to_dataloaders(
        train_dataset,
        val_dataset,
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        seed=seed
    )

#reads every file once, front to back, so the OS has them cached before the first batch: random rows from an
#uncached file are a disk read each, and such runs trained a quarter slower
def warm_file_cache(*datasets) -> None:
    tensors = [tensor for dataset in datasets for tensor in dataset.tensors()]
    total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    progress = tqdm(total=total_bytes, desc="reading the training files into memory", unit="B", unit_scale=True, dynamic_ncols=True, leave=False)
    for tensor in tensors:
        flat = tensor.reshape(-1).view(torch.uint8) if tensor.dtype != torch.uint8 else tensor.reshape(-1)
        for start in range(0, len(flat), 1 << 28):
            flat[start:start + (1 << 28)].sum() #touching the pages is what reads them
            progress.update(min(1 << 28, len(flat) - start))
    progress.close()


#progress is how much of the run after the warmup is done, from 0 to 1
def lr_multiplier(step: int, progress: float, schedule: str = SCHEDULE) -> float:
    if step <= WARMUP_STEPS:
        return (step + 1) / WARMUP_STEPS

    if schedule == "cosine":
        multiplier = 0.5 * (1 + math.cos(math.pi * progress))
    elif schedule == "hold":
        multiplier = min(1.0, (1 - progress) / DECAY_FRACTION)
    elif schedule == "constant":
        multiplier = 1.0
    else:
        raise ValueError(f"unknown schedule {schedule}")

    return multiplier


#Muon for the weight matrices inside the blocks, AdamW for the rest; match_rms_adamw lets both share one learning rate.
#d128: val loss 1.9414 against 1.9929 with AdamW alone. AdamW comes first, the training loop logs its learning rate
def make_optimizers(model: ChessTransformer, learn_rate: float, device: torch.device, weight_decay: float = WEIGHT_DECAY,
                    muon_momentum: float = MUON_MOMENTUM, embed_lr_multiplier: float = EMBED_LR_MULTIPLIER) -> list[torch.optim.Optimizer]:
    if embed_lr_multiplier <= 0:
        raise ValueError(f"the embeddings' learning rate multiplier must be above 0, got {embed_lr_multiplier}")
    block_matrices = [parameter for parameter in model.encoder.parameters() if parameter.ndim == 2]
    #QK-norm's gains alone set how sharp attention can be: decayed at 0.1 they fell to 0.3 and the heads went near uniform
    qk_gains = [norm.weight for block in model.encoder for norm in (block.q_norm, block.k_norm)]
    #the input skip's scales (model.TransformerBlock): decayed, the residual stream's scale would be pulled towards 0
    skip_scales = [scale for block in model.encoder for scale in (block.residual_scale, block.input_scale)]
    undecayed = qk_gains + skip_scales
    #the input embeddings at their own learning rate. AdamW decays by lr * weight_decay a step, so their decay is divided
    #by the multiplier: decaying them faster as well would be a second change
    embeddings = [model.embed.weight, model.position_embed, model.square_features.table]
    elsewhere = {id(parameter) for parameter in block_matrices + undecayed + embeddings}
    rest = [parameter for parameter in model.parameters() if id(parameter) not in elsewhere]
    #a tensor: compile_muon_step's graph would bake a float in and recompile whenever the schedule changes it
    muon = BatchedMuon(block_matrices, lr=torch.tensor(learn_rate, device=device), weight_decay=weight_decay, momentum=muon_momentum, adjust_lr_fn="match_rms_adamw") #type: ignore[arg-type]
    groups = [{"params": rest}, {"params": undecayed, "weight_decay": 0.0},
              {"params": embeddings, "lr": learn_rate * embed_lr_multiplier, "weight_decay": weight_decay / embed_lr_multiplier}]
    return [torch.optim.AdamW(groups, lr=learn_rate, weight_decay=weight_decay, fused=device.type == "cuda"), muon]

#Muon's step as one compiled cuda graph: eager 28 ms next to an 80 ms d256 training step, graphed 5 ms.
#With several processes the gather inside the step splits it into two graphs (training.muon).
#Call after making the schedulers, which wrap optimizer.step and need it to still be a method
def compile_muon_step(optimizers: list[torch.optim.Optimizer]) -> None:
    for optimizer in optimizers:
        if isinstance(optimizer, torch.optim.Muon):
            optimizer.step = torch.compile(optimizer.step, mode="reduce-overhead") #type: ignore[method-assign]

#a bf16 copy of the model trains (a d512 step 6% faster than autocast) while the optimizers step the fp32 model, which
#keeps the updates too small for a bf16 weight (bf16 weights alone: +0.020 val loss on a d128). Checkpoints save model
class MasterWeights:
    def __init__(self, model: ChessTransformer):
        self.model = model
        self.compute_model = copy.deepcopy(model).half_except_heads()
        self.parameters = list(model.parameters())
        self.compute_parameters = list(self.compute_model.parameters())
        #allocated once, every step copies into them
        for parameter in self.parameters:
            parameter.grad = torch.zeros_like(parameter)

    def zero_grad(self) -> None:
        self.compute_model.zero_grad()

    #after backward, before clipping and the optimizers' step
    def gradients_to_master(self) -> None:
        torch._foreach_copy_([parameter.grad for parameter in self.parameters], [parameter.grad for parameter in self.compute_parameters]) #type: ignore

    #after the optimizers' step
    @torch.no_grad()
    def weights_to_compute(self) -> None:
        torch._foreach_copy_(self.compute_parameters, self.parameters) #type: ignore


#the commit, with a + when there were uncommitted changes
def current_commit() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("+" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return ""


#"d256 L7 swiglu 4/3x, 24M positions (runpod)" -> "d256_L7_swiglu_4_3x_24M_positions_runpod", at most 60 characters
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
    loops: int = 1,
    blunder_weight: float = BLUNDER_WEIGHT,
    value_bins: int = VALUE_BINS,
    blunder_threshold: float = BLUNDER_THRESHOLD,
    weight_decay: float = WEIGHT_DECAY,
    muon_momentum: float = MUON_MOMENTUM,
    snapshot_every: int = 0,
    teacher_weight: float = 0.0,
    embed_lr_multiplier: float = EMBED_LR_MULTIPLIER):

    #a fine-tune is the size the checkpoint says, whatever the flags say
    initial_model = None
    if init_checkpoint is not None:
        initial_model, model_settings = load_checkpoint(init_checkpoint, torch.device("cpu"))
    else:
        #saved in the checkpoint so the model can be rebuilt when loading
        model_settings = {"d_model": d_model, "num_layers": num_layers, "n_heads": n_heads, "position_templates": position_templates, "mlp_hidden": mlp_hidden_width(d_model, mlp_ratio), "qk_ratio": qk_ratio, "kv_heads": kv_heads, "loops": loops, "value_bins": value_bins}
    if mix_set is not None and not 0 < mix_share < 1:
        raise ValueError(f"--mix-share is the share of every batch from {mix_set}, between 0 and 1, got {mix_share}")
    if peak_learn_rate is None:
        peak_learn_rate = FINE_TUNE_LEARN_RATE if init_checkpoint is not None else PEAK_LEARN_RATE

    if positions_budget is not None:
        budget = Budget(batch_size, positions=positions_budget)
    else:
        budget = Budget(batch_size, minutes=minutes_budget)
    if run_name is None:
        run_name = folder_name(note) if note else f"d{model_settings['d_model']}_L{model_settings['num_layers']}_h{model_settings['n_heads']}_lr{peak_learn_rate:g}_{budget.describe()}_{datetime.now():%H-%M-%S}"
        #runs started in the same second (several gpus) must not share a folder: mkdir claims the name
        for number in range(1, 100):
            try:
                (RUNS_DIR / run_name).mkdir(parents=True)
                break
            except FileExistsError:
                run_name = run_name.rsplit("_", 1)[0] + f"_{number}" if number > 1 else f"{run_name}_1"

    #a run this short would end during the warmup
    if not budget.timed and budget.max_steps <= WARMUP_STEPS:
        raise ValueError(f"{positions_budget:,} positions is {budget.max_steps} steps, it has to be more than the {WARMUP_STEPS} warmup steps")
    if budget.timed and budget.max_seconds < 60:
        raise ValueError(f"{minutes_budget} minutes is too short, the warmup alone takes a good part of a minute")

    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}")

    mix_batch_size = round(batch_size * mix_share) if mix_set is not None else 0
    train_data_loader, val_data_loader = load_dataloaders(training_set, target_temperature, seed, batch_size - mix_batch_size, result_weight, visit_temperature, blunder_threshold, teacher_weight)
    print(f"{training_set}: {len(train_data_loader.dataset):,} train positions, {len(val_data_loader.dataset):,} val positions")
    mix_val_loader = None
    if mix_set is not None:
        mix_train_loader, mix_val_loader = load_dataloaders(mix_set, target_temperature, seed + 1, mix_batch_size)
        train_data_loader = MixedLoader(train_data_loader, mix_train_loader)
        print(f"{mix_set}: {mix_batch_size} of every {batch_size} positions")

    model = (initial_model if initial_model is not None else ChessTransformer(**model_settings)).to(device)
    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"model: {model_settings}, {num_parameters:,} parameters")

    master_weights = MasterWeights(model)
    compute_model = master_weights.compute_model
    compiled_model = torch.compile(compute_model) if compile_mode == "default" else torch.compile(compute_model, mode=compile_mode)
    optimizers = make_optimizers(model, peak_learn_rate, device, weight_decay, muon_momentum, embed_lr_multiplier)

    #the schedule runs over the part of the budget left when the warmup ends
    warmup_end = 0.0
    def current_lr_multiplier(step: int) -> float:
        nonlocal warmup_end
        if step <= WARMUP_STEPS:
            warmup_end = budget.progress()
            return lr_multiplier(step, 0.0, schedule)
        return lr_multiplier(step, (budget.progress() - warmup_end) / (1 - warmup_end), schedule)

    schedulers = [torch.optim.lr_scheduler.LambdaLR(optimizer, current_lr_multiplier) for optimizer in optimizers]
    if device.type == "cuda":
        compile_muon_step(optimizers)

    #untrained: loss about ln(1968) = 7.6, value loss ln(2) = 0.69 and half right. Anything else means broken data or metrics
    val_loss, val_accuracy, val_value_loss, val_value_accuracy, val_value_error = evaluate_model(compiled_model, val_data_loader, device, max_batches=EVAL_BATCHES)
    print(f"{'starting point' if init_checkpoint is not None else 'untrained'}: val loss {val_loss:.3f}, val accuracy {val_accuracy:.1%}, value loss {val_value_loss:.3f}, value accuracy {val_value_accuracy:.1%}, value error {val_value_error:.4f}")

    print(f"budget: {budget.describe()}, batches of {batch_size} positions, evaluating every {eval_every} steps")
    run_dir = RUNS_DIR / run_name
    print(f"logging to {run_dir.relative_to(ARTIFACTS_DIR.parent)}")
    writer = SummaryWriter(log_dir=str(run_dir))

    #compare_runs.py reads this; the results are added when the run finishes
    run_settings = {
        "model_settings": model_settings,
        "parameters": num_parameters,
        "peak_learn_rate": peak_learn_rate,
        "schedule": schedule,
        "embed_lr_multiplier": embed_lr_multiplier,
        "seed": seed,
        "value_loss_weight": value_loss_weight,
        "blunder_weight": blunder_weight,
        "blunder_threshold": blunder_threshold,
        "weight_decay": weight_decay,
        "muon_momentum": muon_momentum,
        "teacher_weight": teacher_weight,
        #the blunder penalty only acts on a set with blunder labels
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

    #model.pt is the best evaluation's, usually but not always the last
    save_checkpoint(model, model_settings, run_dir / "final.pt")

    #the number to quote: the final weights on the whole val set
    val_loss, val_accuracy, val_value_loss, val_value_accuracy, val_value_error = evaluate_model(compiled_model, val_data_loader, device)
    print(f"final, on all {len(val_data_loader.dataset):,} val positions: val loss {val_loss:.3f}, val accuracy {val_accuracy:.1%}, value loss {val_value_loss:.3f}, value accuracy {val_value_accuracy:.1%}, value error {val_value_error:.4f}")

    writer.add_scalar("final/val_loss", val_loss, budget.steps)
    writer.add_scalar("final/val_accuracy", val_accuracy, budget.steps)
    writer.add_scalar("final/val_value_loss", val_value_loss, budget.steps)
    writer.add_scalar("final/val_value_accuracy", val_value_accuracy, budget.steps)
    writer.add_scalar("final/val_value_error", val_value_error, budget.steps)

    #how much of the mix set the model still knows
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

    #the standard matches: the policy with avoid-repetition, and the search
    measure = f"python -m tools.play.evaluate_elo --checkpoint {(run_dir / 'model.pt').relative_to(ARTIFACTS_DIR.parent)} --elo 1700 --games 200"
    print(f"measure it: {measure} --select avoid-repetition --concurrency 12\n       and: {measure} --select mcts --simulations 64")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train the chess transformer")
    #int() accepts underscores, so --positions 12_300_000 works
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
    parser.add_argument("--schedule", choices=["cosine", "hold", "constant"], default=SCHEDULE, help="the learning rate after the warmup, see SCHEDULE. Default %(default)s")
    parser.add_argument("--embed-lr-multiplier", type=float, default=EMBED_LR_MULTIPLIER, help="the input embeddings' learning rate as a multiple of the rest's, see make_optimizers. Default %(default)s")
    parser.add_argument("--seed", type=int, default=SEED, help="default %(default)s")
    parser.add_argument("--mlp-ratio", type=float, help="hidden size of the swiglu mlp as a multiple of d-model, rounded down to a multiple of 32. Default 8/3")
    parser.add_argument("--target-temperature", type=float, default=TARGET_TEMPERATURE, help="how stockfish's moves share the target, see TARGET_TEMPERATURE. Default %(default)s")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="positions per step, for train and val. Default %(default)s")
    parser.add_argument("--kv-heads", type=int, help="key/value heads, each shared by a group of heads (grouped-query attention). Default one per head")
    parser.add_argument("--loops", type=int, default=1, help="how many times the whole stack of blocks runs with the same weights. Default %(default)s")
    parser.add_argument("--value-bins", type=int, default=VALUE_BINS, help="bins of the expected score the eval head predicts a distribution over, 1 for a single logit. Default %(default)s")
    parser.add_argument("--qk-ratio", type=float, nargs="+", default=[1.0], help="query/key width as a multiple of the values', one for every block or one per block, e.g. 0.25 0.5 0.5 0.75. Default 1")
    parser.add_argument("--init-checkpoint", type=Path, help="fine-tune from this model.pt; the model is the checkpoint's size and the size flags are ignored")
    parser.add_argument("--mix-set", help="a second training set that fills --mix-share of every batch, e.g. full_sf8_30M_mix when fine-tuning on self play")
    parser.add_argument("--mix-share", type=float, default=MIX_SHARE, help="share of every batch from --mix-set, default %(default)s")
    parser.add_argument("--result-weight", type=float, default=RESULT_WEIGHT, help="self play sets: share of the value target from the game's result, the rest the search's score. Default %(default)s")
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY, help="steps between two looks at the val set, default %(default)s")
    parser.add_argument("--snapshot-every", type=int, default=0, help="steps between two copies of the weights kept in the run folder (snapshot_<positions>.pt), 0 for none")
    parser.add_argument("--compile-mode", choices=["default", "max-autotune-no-cudagraphs"], default="default", help="torch.compile's mode, default %(default)s")
    parser.add_argument("--teacher-weight", type=float, default=0.0, help="share of the train targets from a bigger model's (tools/teacher_targets.py); val keeps stockfish's. Default %(default)s")
    parser.add_argument("--visit-temperature", type=float, default=VISIT_TEMPERATURE, help="self play sets: sharpens the visit shares, 1 as recorded, 0 the most visited move alone. Default %(default)s")
    arguments = parser.parse_args()

    if not arguments.no_tensorboard:
        start_tensorboard()

    main(positions_budget=arguments.positions,
         minutes_budget=arguments.minutes,
         run_name=arguments.name,
         note=arguments.note,
         d_model=arguments.d_model,
         num_layers=arguments.layers,
         n_heads=arguments.heads,
         peak_learn_rate=arguments.lr,
         position_templates=arguments.templates,
         training_set=arguments.training_set,
         value_loss_weight=arguments.value_weight,
         schedule=arguments.schedule,
         seed=arguments.seed,
         mlp_ratio=arguments.mlp_ratio,
         target_temperature=arguments.target_temperature,
         qk_ratio=arguments.qk_ratio[0] if len(arguments.qk_ratio) == 1 else arguments.qk_ratio,
         kv_heads=arguments.kv_heads,
         batch_size=arguments.batch_size,
         init_checkpoint=arguments.init_checkpoint,
         mix_set=arguments.mix_set,
         mix_share=arguments.mix_share,
         result_weight=arguments.result_weight,
         visit_temperature=arguments.visit_temperature,
         eval_every=arguments.eval_every,
         snapshot_every=arguments.snapshot_every,
         compile_mode=arguments.compile_mode,
         loops=arguments.loops,
         blunder_weight=arguments.blunder_weight,
         value_bins=arguments.value_bins,
         blunder_threshold=arguments.blunder_threshold,
         weight_decay=arguments.weight_decay,
         muon_momentum=arguments.muon_momentum,
         teacher_weight=arguments.teacher_weight,
         embed_lr_multiplier=arguments.embed_lr_multiplier)

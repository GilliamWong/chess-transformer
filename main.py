#for reference: the first model (carlsen pgn only) trained for 10 epochs of 580,949 positions, about 4 minutes.
#   the same number of positions: 5,673 steps   (--positions 5_800_000)
#   the same training time:       12,000 steps  (--positions 12_300_000, or simply --minutes 4), at about 58 steps a second
import argparse
import json
import queue
import re
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from datetime import datetime
from itertools import islice
from pathlib import Path
from data import ARTIFACTS_DIR, RESULT_WEIGHT, VISIT_TEMPERATURE, MixedLoader, datasets_to_dataloaders, load_training_set
import torch
import torch.nn.functional as F
import math
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tools.budget import Budget
from tools.checkpoint import RUNS_DIR, load_checkpoint, save_checkpoint
from tools.label_with_stockfish import NO_MOVE
from model import ChessTransformer

#the model, each of these has a command line flag (python main.py --help)
D_MODEL = 128
#6, not 4: at the same training compute a d128 with 6 layers reached val loss 1.904 against 1.937 with 4 (and 1.901 with 8)
NUM_LAYERS = 6
N_HEADS = 4
POSITION_TEMPLATES = 8

PEAK_LEARN_RATE = 6e-3
#the peak for a fine-tune (--init-checkpoint), from weights that already work. At PEAK_LEARN_RATE the first fine-tune on
#self play fell from 72.8% to 61.4% top move accuracy on its val set during the warmup and spent the rest of the run
#climbing back (71.3% at the end)
FINE_TUNE_LEARN_RATE = 1e-3
WARMUP_STEPS = 500
#what the learning rate does after the warmup:
#   cosine    falls from the peak to zero along half a cosine, over the whole run
#   hold      stays at the peak, then falls to zero in a straight line over the last DECAY_FRACTION of the run.
#             The part at the peak does not depend on how long the run is, so a run can later be made longer from there
#   constant  stays at the peak. Only to see what the decay is worth, the weights never settle
SCHEDULE = "cosine"
DECAY_FRACTION = 0.2
BATCH_SIZE = 1024

#for Muon and AdamW both (see make_optimizers). AdamW's own default, which every run before Muon trained with; torch's
#Muon defaults to 0.1
WEIGHT_DECAY = 0.01

#minutes of pure training (compiling and evaluating are not counted), unless --minutes or --positions is given.
#runs with the same budget are comparable, so architecture experiments should all use this one
TRAIN_MINUTES_BUDGET = 4

EVAL_EVERY = 100  #steps between two looks at the val set
POSTFIX_EVERY = 20 #steps between two updates of the loss next to the progress bar, reading it makes the cpu wait for the gpu
EVAL_BATCHES = 20 #only this many val batches for those looks, the whole val set is kept for the end
SEED = 42 #of the model's first weights and of the order of the training positions

#the value head learns stockfish's eval of the position, as the expected score it stands for (see value_target_of in data.py)
VALUE_LOSS_WEIGHT = 1.0

#weight of blunder_loss, the push off the moves that give away at least data.BLUNDER_THRESHOLD against stockfish's best.
#Measured on a d128 (8M positions, 800 games against stockfish 1700 per run, two seeds): +27 Elo at 4 against 0, while the
#move loss got worse (the policy no longer matches stockfish's top moves as exactly). 8 was no better than 4 in Elo and
#cost 1.1 points of accuracy; only the 4 worst moves at weight 1 did little (the policy already gives them almost nothing).
#A set without blunder labels (data.WORST_FILES) trains as if this were 0
BLUNDER_WEIGHT = 4.0

#the policy learns stockfish's best moves, which tools/label_with_stockfish.py has written next to every position.
#How they share the probability is TARGET_TEMPERATURE, in expected score (0 lost to 1 won): a move that gives up this
#much gets 1/e of the best move's share. See policy_target in data.py
TARGET_TEMPERATURE = 0.02

#the name of a training set in processed_data/ (see load_training_set in data.py).
#full_sf8_30M_mix is 30M positions of random whole games of the stockfish labelled full_sf8_20260922, in its mix:
#lichess elite 68% (a third of it blitz), fishtest 29%, otb 3%
TRAINING_SET = "full_sf8_30M_mix"

#with --mix-set, this share of every batch comes from that set: fine-tuning on self play keeps stockfish labelled
#positions in the batches so the model does not forget them
MIX_SHARE = 0.25


def endless_batches(data_loader: DataLoader | MixedLoader) -> Iterator[tuple[torch.Tensor, ...]]:
    while True:
        yield from data_loader

#the same batches, already on the device. A background thread reads the next few and copies them over while the gpu
#works on the current one; reading them only when the step asks for them left the gpu waiting 4-5 ms of every 25.
#measured: 39,500 -> 44,200 positions a second
def batches_on_device(batches: Iterator[tuple[torch.Tensor, ...]], device: torch.device, ahead: int = 4) -> Iterator[tuple[torch.Tensor, ...]]:
    ready = queue.Queue(maxsize=ahead)

    def to_device(tensor: torch.Tensor) -> torch.Tensor:
        #pinned memory is what lets the copy run while the gpu computes
        return tensor.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else tensor.to(device)

    def fill() -> None:
        for batch in batches:
            ready.put(tuple(to_device(tensor) for tensor in batch))

    #a daemon, so it does not keep the program alive once training stops asking for batches
    threading.Thread(target=fill, daemon=True).start()
    while True:
        yield ready.get()


#cross entropy against a target that is a few moves with a probability each (B, moves) instead of one move.
#With a single move at probability 1 it is nn.CrossEntropyLoss. Places at probability 0 add nothing, whatever their move
def policy_loss(logits: torch.Tensor, target_moves: torch.Tensor, target_probabilities: torch.Tensor) -> torch.Tensor:
    log_probabilities = torch.log_softmax(logits, dim=-1).gather(1, target_moves)
    return -(target_probabilities * log_probabilities).sum(dim=-1).mean()

#the value head's loss: the target is an expected score from 0 to 1 and the head's one output is its logit,
#so binary cross entropy
def value_loss(value_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(value_logits.squeeze(-1).float(), targets)

#unlikelihood: -log(1 - the probability the policy gives a position's blunders together), 0 for a position without any.
#Cross entropy already pushes every move but the targets down in proportion to its probability; this pushes the
#blunders down on top of that, the ones the policy likes the most the hardest. blunder_moves is (B, places), NO_MOVE
#where there is none
def blunder_loss(logits: torch.Tensor, blunder_moves: torch.Tensor) -> torch.Tensor:
    blunders = blunder_moves != NO_MOVE
    probabilities = torch.softmax(logits.float(), dim=-1).gather(1, blunder_moves.clamp(min=0)) * blunders
    return -torch.log1p(-probabilities.sum(dim=-1).clamp(max=1 - 1e-6)).mean()

#how many positions the value head gets right: the side to move ahead or not (score above a half)
def value_correct(value_logits: torch.Tensor, targets: torch.Tensor) -> int:
    return int(((value_logits.squeeze(-1) > 0) == (targets > 0.5)).sum().item())


def train(
    model: nn.Module,
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
    blunder_weight: float = BLUNDER_WEIGHT) -> None:

    #kept on the device: adding to it does not make the cpu wait for the gpu the way loss.item() does.
    #loss/train stays the policy and value loss the older runs logged, the blunder penalty is logged on its own
    loss_total = torch.zeros((), device=device)
    penalty_total = torch.zeros((), device=device)
    total_examples = 0

    model.train()

    #dynamic_ncols: a bar wider than a resized terminal wraps, and every update becomes a new line
    #the bar counts steps, or seconds of training when the budget is in minutes
    progress = tqdm(total=budget.total, desc="train", dynamic_ncols=True, unit="s" if budget.timed else "it",
                    bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f} [{elapsed}<{remaining}{postfix}]")

    #target_moves and target_probabilities are the policy target: stockfish's best few moves (policy_loss)
    #values is the value head's target: stockfish's eval as an expected score (value_loss)
    #blunders are the moves the policy is pushed off (blunder_loss)
    for batch, target_moves, target_probabilities, values, blunders in batches_on_device(endless_batches(train_data_loader), device):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, eval_logits = model(batch)

            loss = policy_loss(logits, target_moves, target_probabilities) + value_loss_weight * value_loss(eval_logits, values)
            penalty = blunder_loss(logits, blunders) if blunder_weight else torch.zeros((), device=device)

        for optimizer in optimizers:
            optimizer.zero_grad()

        (loss + blunder_weight * penalty).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        for optimizer in optimizers:
            optimizer.step()
        progress.update(budget.step_done())
        for scheduler in schedulers:
            scheduler.step()
        step = budget.steps

        loss_total += loss.detach() * len(batch)
        penalty_total += penalty.detach() * len(batch)
        total_examples += len(batch)

        #the running average, a single batch's loss jumps around too much to read
        if step % POSTFIX_EVERY == 0:
            progress.set_postfix(loss=f"{loss_total.item() / total_examples:.3f}")

        #the last step is always evaluated, so the final weights get their chance to be the checkpoint
        if step % eval_every == 0 or budget.finished():
            train_loss = loss_total.item() / total_examples #also waits for the gpu, so the pause starts when the steps are really done
            budget.pause() #evaluating does not count towards a budget in minutes
            val_loss, val_accuracy, val_value_loss, val_value_accuracy = evaluate_model(model, val_data_loader, device, max_batches=EVAL_BATCHES)
            model.train()

            #tqdm.write rather than print, which would cut through the progress bar
            tqdm.write(f"step {step} ({budget.positions:,} positions, {budget.seconds / 60:.1f} min): train loss {train_loss:.3f}, val loss {val_loss:.3f}, val accuracy {val_accuracy:.1%}, value loss {val_value_loss:.3f}, value accuracy {val_value_accuracy:.1%}")

            #the part of the name before the slash groups the plots: both losses end up next to each other
            writer.add_scalar("loss/train", train_loss, step)
            if blunder_weight:
                writer.add_scalar("loss/train_blunder", penalty_total.item() / total_examples, step)
            writer.add_scalar("loss/val", val_loss, step)
            writer.add_scalar("accuracy/val", val_accuracy, step)
            writer.add_scalar("loss/val_value", val_value_loss, step)
            writer.add_scalar("accuracy/val_value", val_value_accuracy, step)
            writer.add_scalar("learning_rate", schedulers[0].get_last_lr()[0], step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_best_checkpoint()

            loss_total.zero_()
            penalty_total.zero_()
            total_examples = 0
            budget.resume()

        if budget.finished():
            break

    progress.close()


#returns average loss and accuracy for the model on the test set, or on its first max_batches batches: first of the
#moves, then of the value head (value_loss and value_correct).
#The move loss stays on its own so it can be compared with runs from before the value head.
#the val loader is not shuffled, so those are the same positions every time and the numbers can be compared
def evaluate_model(
    model: nn.Module,
    test_data_loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
    show_progress: bool = True) -> tuple[float, float, float, float]:

    epoch_loss_total = 0.0
    total_correct_preds = 0
    value_loss_total = 0.0
    value_correct_preds = 0
    total_examples = 0

    model.eval()

    with torch.no_grad():

        num_batches = len(test_data_loader) if max_batches is None else min(max_batches, len(test_data_loader))
        #islice stops after num_batches, leave=False clears the bar afterwards. show_progress=False where several
        #processes evaluate at once (multi_gpu/train_ddp.py), one bar is enough
        progress = tqdm(islice(test_data_loader, num_batches), total=num_batches, desc="val", leave=False, dynamic_ncols=True, disable=not show_progress)

        #the blunders are left out: the val numbers stay the move loss alone, comparable with every earlier run
        for batch, target_moves, target_probabilities, values, _ in progress:
            target_moves = target_moves.to(device)
            target_probabilities = target_probabilities.to(device)
            values = values.to(device)
            batch = batch.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                #should be of dim B 1 output_dim
                logits, value_logits = model(batch)

                model_preds = torch.softmax(logits, dim=-1).argmax(dim=-1)

                loss = policy_loss(logits, target_moves, target_probabilities)
                batch_value_loss = value_loss(value_logits, values)

            #accuracy is against the first target move, the one stockfish likes best
            epoch_loss_total += loss.item() * len(batch)
            total_correct_preds += (model_preds == target_moves[:, 0]).sum().item()
            value_loss_total += batch_value_loss.item() * len(batch)
            value_correct_preds += value_correct(value_logits, values)
            total_examples += len(batch)

            progress.set_postfix(
                loss=f"{epoch_loss_total / total_examples:.3f}",
                accuracy=f"{total_correct_preds / total_examples:.1%}"
            )

    return (epoch_loss_total / total_examples), (total_correct_preds / total_examples), (value_loss_total / total_examples), (value_correct_preds / total_examples)


def load_dataloaders(training_set: str = TRAINING_SET, target_temperature: float = TARGET_TEMPERATURE, seed: int = SEED, batch_size: int = BATCH_SIZE,
                     result_weight: float = RESULT_WEIGHT, visit_temperature: float = VISIT_TEMPERATURE) -> tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = load_training_set(training_set, target_temperature, result_weight, visit_temperature)
    warm_file_cache(train_dataset, val_dataset)
    return datasets_to_dataloaders(
        train_dataset,
        val_dataset,
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        seed=seed
    )

#reads every file of the sets once, front to back, so that the operating system has them in memory before the first
#batch. Batches read random rows: from a file that is not in memory yet each row is a disk read, and a run on a set
#that was just written or pushed out of memory by other work trained a quarter slower until enough of it had been
#touched. Reading 12 GB in order takes about a minute; nothing to read takes seconds
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


#Muon moves a weight matrix equally far along every direction its gradient points in: newton-schulz iterations set
#all singular values of the momentum to about 1 (in a d256's gradients the strongest direction is 20-90 times the
#median one). That is a step for a linear layer, so Muon only takes the weight matrices inside the blocks, 87% of a
#d256's parameters; the embeddings, the template bank, the heads, the biases and the layer norms stay with AdamW.
#match_rms_adamw scales Muon's steps to the size AdamW's usually have, so both share the learning rate and its schedule.
#Measured against AdamW for everything (d128, 4 layers, 4 minutes, the same start and data): val loss 1.9414 against
#1.9929 with 8% fewer positions, as much as twice the training time gave.
#AdamW comes first in the list, the training loop logs its learning rate
def make_optimizers(model: ChessTransformer, learn_rate: float, device: torch.device) -> list[torch.optim.Optimizer]:
    block_matrices = [parameter for parameter in model.encoder.parameters() if parameter.ndim == 2]
    in_muon = {id(parameter) for parameter in block_matrices}
    rest = [parameter for parameter in model.parameters() if id(parameter) not in in_muon]
    #a tensor, because compile_muon_step's graph reads it: a float would be baked into the graph and recompiled every
    #time the schedule changes it
    muon = torch.optim.Muon(block_matrices, lr=torch.tensor(learn_rate, device=device), weight_decay=WEIGHT_DECAY, adjust_lr_fn="match_rms_adamw") #type: ignore[arg-type]
    #fused does the whole update in one gpu kernel, it only exists for parameters that are on the gpu
    return [torch.optim.AdamW(rest, lr=learn_rate, weight_decay=WEIGHT_DECAY, fused=device.type == "cuda"), muon]

#Muon's step as one compiled cuda graph. Eager, its newton-schulz iterations are ~20 small matrix products a matrix,
#one kernel launch each, and the step took 28 ms next to a d256 training step of ~80 ms; graphed 5 ms.
#Only once the schedulers exist: a scheduler wraps optimizer.step and needs it to still be a method
def compile_muon_step(optimizers: list[torch.optim.Optimizer]) -> None:
    for optimizer in optimizers:
        if isinstance(optimizer, torch.optim.Muon):
            optimizer.step = torch.compile(optimizer.step, mode="reduce-overhead") #type: ignore[method-assign]


#the commit the code was at, with a + when there were uncommitted changes on top of it
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
    blunder_weight: float = BLUNDER_WEIGHT):

    #fine-tuning starts from a checkpoint's weights, and the model is the size the checkpoint says, whatever the flags say
    initial_model = None
    if init_checkpoint is not None:
        initial_model, model_settings = load_checkpoint(init_checkpoint, torch.device("cpu"))
    else:
        #everything ChessTransformer is built from: saved in the checkpoint so it can be rebuilt when loading
        model_settings = {"d_model": d_model, "num_layers": num_layers, "n_heads": n_heads, "position_templates": position_templates, "mlp_ratio": mlp_ratio, "qk_ratio": qk_ratio, "kv_heads": kv_heads, "loops": loops}
    if mix_set is not None and not 0 < mix_share < 1:
        raise ValueError(f"--mix-share is the share of every batch from {mix_set}, between 0 and 1, got {mix_share}")
    if peak_learn_rate is None:
        peak_learn_rate = FINE_TUNE_LEARN_RATE if init_checkpoint is not None else PEAK_LEARN_RATE

    #a budget in positions is only used when one is asked for
    if positions_budget is not None:
        budget = Budget(batch_size, positions=positions_budget)
    else:
        budget = Budget(batch_size, minutes=minutes_budget)
    if run_name is None:
        #the note names the run in tensorboard, made safe for a folder; without one, the settings that differ between experiments
        run_name = folder_name(note) if note else f"d{model_settings['d_model']}_L{model_settings['num_layers']}_h{model_settings['n_heads']}_lr{peak_learn_rate:g}_{budget.describe()}_{datetime.now():%H-%M-%S}"
        #runs started in the same second (several gpus at once) must not share a folder: the folder is claimed here,
        #and only the claim that made it decides the name
        for number in range(1, 100):
            try:
                (RUNS_DIR / run_name).mkdir(parents=True)
                break
            except FileExistsError:
                run_name = run_name.rsplit("_", 1)[0] + f"_{number}" if number > 1 else f"{run_name}_1"

    #a run this short would end during the warmup, without ever reaching PEAK_LEARN_RATE
    if not budget.timed and budget.max_steps <= WARMUP_STEPS:
        raise ValueError(f"{positions_budget:,} positions is {budget.max_steps} steps, it has to be more than the {WARMUP_STEPS} warmup steps")
    if budget.timed and budget.max_seconds < 60:
        raise ValueError(f"{minutes_budget} minutes is too short, the warmup alone takes a good part of a minute")

    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}")

    #with a mix set, its share of every batch is taken off the training set's
    mix_batch_size = round(batch_size * mix_share) if mix_set is not None else 0
    train_data_loader, val_data_loader = load_dataloaders(training_set, target_temperature, seed, batch_size - mix_batch_size, result_weight, visit_temperature)
    print(f"{training_set}: {len(train_data_loader.dataset):,} train positions, {len(val_data_loader.dataset):,} val positions")
    mix_val_loader = None
    if mix_set is not None:
        mix_train_loader, mix_val_loader = load_dataloaders(mix_set, target_temperature, seed + 1, mix_batch_size)
        train_data_loader = MixedLoader(train_data_loader, mix_train_loader)
        print(f"{mix_set}: {mix_batch_size} of every {batch_size} positions")

    model = (initial_model if initial_model is not None else ChessTransformer(**model_settings)).to(device)
    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"model: {model_settings}, {num_parameters:,} parameters")

    #max-autotune-no-cudagraphs tries fused matrix product kernels for every shape and keeps the fastest: a slower first compile
    compiled_model = torch.compile(model) if compile_mode == "default" else torch.compile(model, mode=compile_mode)
    optimizers = make_optimizers(model, peak_learn_rate, device)

    #the schedule runs over the part of the budget that is left when the warmup ends
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

    #before any training the loss should be about ln(1968) = 7.6 and the accuracy about 0,
    #anything else means the data or the metrics are wrong
    #the value head starts at about ln(2) = 0.69 and half right
    val_loss, val_accuracy, val_value_loss, val_value_accuracy = evaluate_model(compiled_model, val_data_loader, device, max_batches=EVAL_BATCHES)
    print(f"{'starting point' if init_checkpoint is not None else 'untrained'}: val loss {val_loss:.3f}, val accuracy {val_accuracy:.1%}, value loss {val_value_loss:.3f}, value accuracy {val_value_accuracy:.1%}")

    print(f"budget: {budget.describe()}, batches of {batch_size} positions, evaluating every {eval_every} steps")
    run_dir = RUNS_DIR / run_name
    print(f"logging to {run_dir.relative_to(ARTIFACTS_DIR.parent)}")
    writer = SummaryWriter(log_dir=str(run_dir))

    #compare_runs.py reads this. The results are added once the run has finished
    run_settings = {
        "model_settings": model_settings,
        "parameters": num_parameters,
        "peak_learn_rate": peak_learn_rate,
        "schedule": schedule,
        "seed": seed,
        "value_loss_weight": value_loss_weight,
        #the blunder push only acts on a set with blunder labels: whether this one has them is recorded next to the weight
        "blunder_weight": blunder_weight,
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
        "optimizer": "muon", #runs before 2026-09-24 trained everything with AdamW
        "note": note,
        "commit": current_commit()
    }
    (run_dir / "settings.json").write_text(json.dumps(run_settings, indent=4))

    #saved from model, not compiled_model: the compiled wrapper prefixes every name with "_orig_mod."
    def save_best_checkpoint() -> None:
        save_checkpoint(model, model_settings, run_dir / "model.pt")

    train(compiled_model, save_best_checkpoint, train_data_loader, val_data_loader, optimizers, schedulers, device, best_val_loss=val_loss, budget=budget, writer=writer, value_loss_weight=value_loss_weight,
          eval_every=eval_every, blunder_weight=blunder_weight)

    #the final weights as well: model.pt is the best evaluation's, which is usually but not always the last one
    save_checkpoint(model, model_settings, run_dir / "final.pt")

    #the number to quote for this run: the final weights on the whole val set
    val_loss, val_accuracy, val_value_loss, val_value_accuracy = evaluate_model(compiled_model, val_data_loader, device)
    print(f"final, on all {len(val_data_loader.dataset):,} val positions: val loss {val_loss:.3f}, val accuracy {val_accuracy:.1%}, value loss {val_value_loss:.3f}, value accuracy {val_value_accuracy:.1%}")

    writer.add_scalar("final/val_loss", val_loss, budget.steps)
    writer.add_scalar("final/val_accuracy", val_accuracy, budget.steps)
    writer.add_scalar("final/val_value_loss", val_value_loss, budget.steps)
    writer.add_scalar("final/val_value_accuracy", val_value_accuracy, budget.steps)

    #how much of the mix set the model still knows: comparable with the final val numbers of a run trained on that set
    if mix_val_loader is not None:
        mix_val_loss, mix_val_accuracy, mix_val_value_loss, mix_val_value_accuracy = evaluate_model(compiled_model, mix_val_loader, device)
        print(f"final, on all {len(mix_val_loader.dataset):,} val positions of {mix_set}: val loss {mix_val_loss:.3f}, val accuracy {mix_val_accuracy:.1%}, value loss {mix_val_value_loss:.3f}, value accuracy {mix_val_value_accuracy:.1%}")
        writer.add_scalar("final/mix_val_loss", mix_val_loss, budget.steps)
        writer.add_scalar("final/mix_val_accuracy", mix_val_accuracy, budget.steps)
        run_settings.update(final_mix_val_loss=mix_val_loss, final_mix_val_accuracy=mix_val_accuracy,
                            final_mix_val_value_loss=mix_val_value_loss, final_mix_val_value_accuracy=mix_val_value_accuracy)
    writer.close()

    run_settings.update(
        final_val_loss=val_loss,
        final_val_accuracy=val_accuracy,
        final_val_value_loss=val_value_loss,
        final_val_value_accuracy=val_value_accuracy,
        train_seconds=budget.seconds,
        positions_seen=budget.positions,
        finished_at=datetime.now().isoformat(timespec="seconds")
    )
    (run_dir / "settings.json").write_text(json.dumps(run_settings, indent=4))

    #the table and chart of every run so far, imported here because compare_runs imports this file
    from tools.compare_runs import write_progress_page
    print(f"progress page: {write_progress_page()}")

    #the matches every model has been measured with (artifacts/mcts_ladder_2026-09-23.md), the policy alone and the search.
    #The policy with avoid-repetition, evaluate_elo's default since 2026-09-24 (earlier logged results are argmax)
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
    parser.add_argument("--training-set", default=TRAINING_SET, help="name of a training set in processed_data/ that tools.label_with_stockfish has labelled. Default %(default)s")
    parser.add_argument("--value-weight", type=float, default=VALUE_LOSS_WEIGHT, help="weight of the value head's loss next to the move loss, default %(default)s")
    parser.add_argument("--blunder-weight", type=float, default=BLUNDER_WEIGHT, help="weight of the push off labelled blunders (blunder_loss), 0 for none. "
                                                                                     "Only a set with blunder labels has any. Default %(default)s")
    parser.add_argument("--schedule", choices=["cosine", "hold", "constant"], default=SCHEDULE, help="what the learning rate does after the warmup, see SCHEDULE in main.py. Default %(default)s")
    parser.add_argument("--seed", type=int, default=SEED, help="runs that differ only in this show how much of a difference between two runs is chance. Default %(default)s")
    parser.add_argument("--mlp-ratio", type=float, help="hidden size of the swiglu mlp in every block as a multiple of d-model, default 8/3")
    parser.add_argument("--target-temperature", type=float, default=TARGET_TEMPERATURE, help="how stockfish's moves share the target, see TARGET_TEMPERATURE in main.py. Default %(default)s")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="positions per step, for train and val. Lower it for a model that does not fit, default %(default)s")
    parser.add_argument("--kv-heads", type=int, help="key/value heads, each shared by a group of heads (grouped-query attention). Default one for every head")
    parser.add_argument("--loops", type=int, default=1, help="how many times the whole stack of blocks runs, the same weights every time (looped depth). Default %(default)s")
    parser.add_argument("--qk-ratio", type=float, nargs="+", default=[1.0], help="width of the queries and keys as a multiple of the values', rounded to whole dims per head. One for every block or one per block, e.g. --qk-ratio 0.25 0.5 0.5 0.75. Default 1")
    parser.add_argument("--init-checkpoint", type=Path, help="start from this model.pt's weights (fine-tuning); the model is the checkpoint's size and the size flags are ignored")
    parser.add_argument("--mix-set", help="a second training set that fills --mix-share of every batch, e.g. full_sf8_30M_mix when fine-tuning on self play")
    parser.add_argument("--mix-share", type=float, default=MIX_SHARE, help="share of every batch from --mix-set, default %(default)s")
    parser.add_argument("--result-weight", type=float, default=RESULT_WEIGHT, help="for a self play set: how much of the value target is the game's result, the rest the search's score. Default %(default)s")
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY, help="steps between two looks at the val set, default %(default)s. A long run can look less often: each look costs about EVAL_BATCHES / 3 steps")
    parser.add_argument("--compile-mode", choices=["default", "max-autotune-no-cudagraphs"], default="default", help="torch.compile's mode, see train_speed measurements. Default %(default)s")
    parser.add_argument("--visit-temperature", type=float, default=VISIT_TEMPERATURE, help="for a self play set: sharpens the visit shares the policy learns, "
                                                                                           "1 as recorded, 0 the most visited move alone (see VISIT_TEMPERATURE in data.py). Default %(default)s")
    arguments = parser.parse_args()

    if not arguments.no_tensorboard:
        #imported here because tools/tensorboard_server.py imports RUNS_DIR from this file
        from tools.tensorboard_server import start_tensorboard
        start_tensorboard()

    main(arguments.positions, arguments.minutes, arguments.name, arguments.note, arguments.d_model, arguments.layers, arguments.heads, arguments.lr, arguments.templates, arguments.training_set, arguments.value_weight, arguments.schedule, arguments.seed, arguments.mlp_ratio, arguments.target_temperature, arguments.qk_ratio[0] if len(arguments.qk_ratio) == 1 else arguments.qk_ratio, arguments.kv_heads, arguments.batch_size,
         init_checkpoint=arguments.init_checkpoint, mix_set=arguments.mix_set, mix_share=arguments.mix_share, result_weight=arguments.result_weight,
         visit_temperature=arguments.visit_temperature,
         eval_every=arguments.eval_every, compile_mode=arguments.compile_mode, loops=arguments.loops,
         blunder_weight=arguments.blunder_weight)

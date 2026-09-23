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
from collections.abc import Callable, Iterator
from datetime import datetime
from itertools import islice
from pathlib import Path
from data import FUTURE_UNKNOWN, ARTIFACTS_DIR, CARLSEN_PGN, pgns_to_examples, create_dataloaders, datasets_to_dataloaders, load_training_set
import torch
import torch.nn.functional as F
import math
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tools.budget import Budget
from tools.checkpoint import RUNS_DIR, save_checkpoint
from model import ChessTransformer

#the model, each of these has a command line flag (python main.py --help)
D_MODEL = 128
NUM_LAYERS = 4
N_HEADS = 4
POSITION_TEMPLATES = 16
HISTORY = 0 #past plies the model is shown, as what changed on the board in each
FUTURE = 0 #plies ahead the future head predicts the board's changes, 0 for no future head
FUTURE_LOSS_WEIGHT = 0.5

PEAK_LEARN_RATE = 6e-3
WARMUP_STEPS = 500
#what the learning rate does after the warmup:
#   cosine    falls from the peak to zero along half a cosine, over the whole run
#   hold      stays at the peak, then falls to zero in a straight line over the last DECAY_FRACTION of the run.
#             The part at the peak does not depend on how long the run is, so a run can later be made longer from there
#   constant  stays at the peak. Only to see what the decay is worth, the weights never settle
SCHEDULE = "cosine"
DECAY_FRACTION = 0.2
BATCH_SIZE = 1024

#minutes of pure training (compiling and evaluating are not counted), unless --minutes or --positions is given.
#runs with the same budget are comparable, so architecture experiments should all use this one
TRAIN_MINUTES_BUDGET = 4

EVAL_EVERY = 100  #steps between two looks at the val set
POSTFIX_EVERY = 20 #steps between two updates of the loss next to the progress bar, reading it makes the cpu wait for the gpu
EVAL_BATCHES = 20 #only this many val batches for those looks, the whole val set is kept for the end
SEED = 42 #of the model's first weights and of the order of the training positions. The carlsen train/val split always uses this one
DROPOUT = 0.0

VALUE_LOSS_WEIGHT = 1.0
#what the value head learns to predict: "eval", stockfish's score of the position (the sets that have stockfish labels),
#or "result", how the game ended. See VALUE_TARGETS in data.py
VALUE_TARGET = "eval"

#what the policy learns to predict: "human" is the move that was played, "stockfish" is stockfish's best moves for the
#sets that tools/label_with_stockfish.py has labelled. How they share the probability is TARGET_TEMPERATURE, in expected
#score (0 lost to 1 won): a move that gives up this much gets 1/e of the best move's share. See policy_target in data.py
POLICY_TARGET = "stockfish"
TARGET_TEMPERATURE = 0.02

#"carlsen" parses pgn_data/Carlsen.pgn, which is in the repository and takes 20 seconds.
#anything else is the name of a training set in processed_data/ (see load_training_set in data.py).
#full_sf8_30M_mix is 30M positions of random whole games of the stockfish labelled full_sf8_20260922, in its mix:
#lichess elite 68% (a third of it blitz), fishtest 29%, otb 3%
TRAINING_SET = "full_sf8_30M_mix"


def endless_batches(data_loader: DataLoader) -> Iterator[tuple[torch.Tensor, ...]]:
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

#the value head's loss for either target (see VALUE_TARGETS in data.py): an expected score from 0 to 1 is a float and
#the head's one output is its logit, so binary cross entropy; the game's result is a class, so cross entropy
def value_loss(value_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if targets.is_floating_point():
        return F.binary_cross_entropy_with_logits(value_logits.squeeze(-1).float(), targets)
    return F.cross_entropy(value_logits, targets)

#how many positions the value head gets right: the side to move ahead or not (score above a half), or the result's class
def value_correct(value_logits: torch.Tensor, targets: torch.Tensor) -> int:
    if targets.is_floating_point():
        return int(((value_logits.squeeze(-1) > 0) == (targets > 0.5)).sum().item())
    return int((value_logits.argmax(dim=-1) == targets).sum().item())


def train(
    model: nn.Module,
    save_best_checkpoint: Callable[[], None],
    train_data_loader: DataLoader,
    val_data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    best_val_loss: float,
    budget: Budget,
    writer: SummaryWriter,
    value_loss_weight: float = VALUE_LOSS_WEIGHT,
    future_loss_weight: float = FUTURE_LOSS_WEIGHT) -> None:

    #kept on the device: adding to it does not make the cpu wait for the gpu the way loss.item() does
    loss_total = torch.zeros((), device=device)
    total_examples = 0

    model.train()

    #dynamic_ncols: a bar wider than a resized terminal wraps, and every update becomes a new line
    #the bar counts steps, or seconds of training when the budget is in minutes
    progress = tqdm(total=budget.total, desc="train", dynamic_ncols=True, unit="s" if budget.timed else "it",
                    bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f} [{elapsed}<{remaining}{postfix}]")

    #target_moves and target_probabilities are the policy target: the move played, or stockfish's best few (policy_loss)
    #values is the value head's target: stockfish's eval as an expected score, or how the game ended (value_loss)
    #history is what changed on the board in the plies before each position, empty for a model that is not shown any
    #future is what will have changed some plies ahead, the future head's target (all FUTURE_UNKNOWN without the head)
    for batch, target_moves, target_probabilities, values, history, future in batches_on_device(endless_batches(train_data_loader), device):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, eval_logits, future_logits = model(batch, history)

            loss = policy_loss(logits, target_moves, target_probabilities) + value_loss_weight * value_loss(eval_logits, values)
            if future_logits is not None:
                loss = loss + future_loss_weight * future_loss(future_logits, future)

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        progress.update(budget.step_done())
        scheduler.step()
        step = budget.steps

        loss_total += loss.detach() * len(batch)
        total_examples += len(batch)

        #the running average, a single batch's loss jumps around too much to read
        if step % POSTFIX_EVERY == 0:
            progress.set_postfix(loss=f"{loss_total.item() / total_examples:.3f}")

        #the last step is always evaluated, so the final weights get their chance to be the checkpoint
        if step % EVAL_EVERY == 0 or budget.finished():
            train_loss = loss_total.item() / total_examples #also waits for the gpu, so the pause starts when the steps are really done
            budget.pause() #evaluating does not count towards a budget in minutes
            val_loss, val_accuracy, val_value_loss, val_value_accuracy, val_future_loss = evaluate_model(model, val_data_loader, device, max_batches=EVAL_BATCHES)
            model.train()

            #tqdm.write rather than print, which would cut through the progress bar
            tqdm.write(f"step {step} ({budget.positions:,} positions, {budget.seconds / 60:.1f} min): train loss {train_loss:.3f}, val loss {val_loss:.3f}, val accuracy {val_accuracy:.1%}, value loss {val_value_loss:.3f}, value accuracy {val_value_accuracy:.1%}, future loss {val_future_loss:.3f}")

            #the part of the name before the slash groups the plots: both losses end up next to each other
            writer.add_scalar("loss/train", train_loss, step)
            writer.add_scalar("loss/val", val_loss, step)
            writer.add_scalar("accuracy/val", val_accuracy, step)
            writer.add_scalar("loss/val_value", val_value_loss, step)
            writer.add_scalar("accuracy/val_value", val_value_accuracy, step)
            writer.add_scalar("loss/val_future", val_future_loss, step)
            writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_best_checkpoint()

            loss_total.zero_()
            total_examples = 0
            budget.resume()

        if budget.finished():
            break

    progress.close()


#cross entropy of the future head over every square whose future is known: logits are (B, squares, codes),
#targets (B, squares) with FUTURE_UNKNOWN where the game ended first; no known targets gives zero loss
def future_loss(future_logits: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(future_logits.flatten(0, 1), future.flatten(), ignore_index=FUTURE_UNKNOWN, reduction="sum") / (future != FUTURE_UNKNOWN).sum().clamp_min(1)


#returns average loss and accuracy for the model on the test set, or on its first max_batches batches: first of the
#moves, then of the value head (value_loss and value_correct), then the future head's loss (0 without the head, NaN if
#no targets are known).
#The move loss stays on its own so it can be compared with runs from before the other heads.
#the val loader is not shuffled, so those are the same positions every time and the numbers can be compared
def evaluate_model(
    model: nn.Module,
    test_data_loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None) -> tuple[float, float, float, float, float]:

    epoch_loss_total = 0.0
    total_correct_preds = 0
    value_loss_total = 0.0
    value_correct_preds = 0
    future_loss_total = 0.0
    future_target_count = 0
    total_examples = 0

    model.eval()

    with torch.no_grad():

        num_batches = len(test_data_loader) if max_batches is None else min(max_batches, len(test_data_loader))
        #islice stops after num_batches, leave=False clears the bar afterwards
        progress = tqdm(islice(test_data_loader, num_batches), total=num_batches, desc="val", leave=False, dynamic_ncols=True)

        for batch, target_moves, target_probabilities, values, history, future in progress:
            target_moves = target_moves.to(device)
            target_probabilities = target_probabilities.to(device)
            values = values.to(device)
            batch = batch.to(device)
            history = history.to(device)
            future = future.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                #should be of dim B 1 output_dim
                logits, value_logits, future_logits = model(batch, history)

                model_preds = torch.softmax(logits, dim=-1).argmax(dim=-1)

                loss = policy_loss(logits, target_moves, target_probabilities)
                batch_value_loss = value_loss(value_logits, values)
                if future_logits is not None:
                    known_squares = (future != FUTURE_UNKNOWN).sum().item()
                    future_loss_total += future_loss(future_logits, future).item() * known_squares
                    future_target_count += known_squares

            #accuracy is against the first target move: the move played, or the one stockfish likes best
            epoch_loss_total += loss.item() * len(batch)
            total_correct_preds += (model_preds == target_moves[:, 0]).sum().item()
            value_loss_total += batch_value_loss.item() * len(batch)
            value_correct_preds += value_correct(value_logits, values)
            total_examples += len(batch)

            progress.set_postfix(
                loss=f"{epoch_loss_total / total_examples:.3f}",
                accuracy=f"{total_correct_preds / total_examples:.1%}"
            )

    val_future_loss = future_loss_total / future_target_count if future_target_count else (0.0 if future_logits is None else math.nan)
    return (epoch_loss_total / total_examples), (total_correct_preds / total_examples), (value_loss_total / total_examples), (value_correct_preds / total_examples), val_future_loss


def load_dataloaders(training_set: str = TRAINING_SET, history: int = 0, policy_target: str = POLICY_TARGET, target_temperature: float = TARGET_TEMPERATURE, future: int = 0, value_target: str = VALUE_TARGET, seed: int = SEED, batch_size: int = BATCH_SIZE) -> tuple[DataLoader, DataLoader]:
    if training_set == "carlsen":
        if policy_target != "human" or value_target != "result":
            raise ValueError("the carlsen games are parsed from their pgn and have no stockfish labels")
        print("parsing games, this takes a while")
        games = pgns_to_examples([CARLSEN_PGN])
        return create_dataloaders(
            games,
            train_batch_size=batch_size,
            val_batch_size=batch_size,
            seed=SEED,
            history=history,
            future=future
        )

    train_dataset, val_dataset = load_training_set(training_set, history, policy_target, target_temperature, future, value_target)
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
    #a set named several times in a mix is the same object several times, and is read once
    sets = {id(one_dataset): one_dataset for dataset in datasets for one_dataset in getattr(dataset, "datasets", [dataset])}
    tensors = [tensor for one_dataset in sets.values() for tensor in (one_dataset.boards, one_dataset.moves, one_dataset.results, one_dataset.plies, one_dataset.stockfish_moves, one_dataset.stockfish_evals) if tensor is not None]
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
    peak_learn_rate: float = PEAK_LEARN_RATE,
    position_templates: int = POSITION_TEMPLATES,
    training_set: str = TRAINING_SET,
    value_loss_weight: float = VALUE_LOSS_WEIGHT,
    history: int = HISTORY,
    schedule: str = SCHEDULE,
    swiglu: bool = False,
    seed: int = SEED,
    mlp_ratio: float | None = None,
    policy_target: str = POLICY_TARGET,
    target_temperature: float = TARGET_TEMPERATURE,
    future: int = FUTURE,
    future_loss_weight: float = FUTURE_LOSS_WEIGHT,
    value_target: str = VALUE_TARGET,
    qk_ratio: float | list[float] = 1.0,
    kv_heads: int | None = None,
    batch_size: int = BATCH_SIZE):

    #a budget in positions is only used when one is asked for
    if positions_budget is not None:
        budget = Budget(batch_size, positions=positions_budget)
    else:
        budget = Budget(batch_size, minutes=minutes_budget)
    if run_name is None:
        #the note names the run in tensorboard, made safe for a folder; without one, the settings that differ between experiments
        run_name = folder_name(note) if note else f"d{d_model}_L{num_layers}_h{n_heads}_lr{peak_learn_rate:g}_{budget.describe()}_{datetime.now():%H-%M-%S}"
        #runs started in the same second (several gpus at once) must not share a folder: the folder is claimed here,
        #and only the claim that made it decides the name
        for number in range(1, 100):
            try:
                (RUNS_DIR / run_name).mkdir(parents=True)
                break
            except FileExistsError:
                run_name = run_name.rsplit("_", 1)[0] + f"_{number}" if number > 1 else f"{run_name}_1"

    #everything ChessTransformer is built from: saved in the checkpoint so it can be rebuilt when loading
    model_settings = {"d_model": d_model, "num_layers": num_layers, "n_heads": n_heads, "position_templates": position_templates, "history": history, "future": future, "swiglu": swiglu, "mlp_ratio": mlp_ratio, "dropout": DROPOUT, "value_target": value_target, "qk_ratio": qk_ratio, "kv_heads": kv_heads}

    #a run this short would end during the warmup, without ever reaching PEAK_LEARN_RATE
    if not budget.timed and budget.max_steps <= WARMUP_STEPS:
        raise ValueError(f"{positions_budget:,} positions is {budget.max_steps} steps, it has to be more than the {WARMUP_STEPS} warmup steps")
    if budget.timed and budget.max_seconds < 60:
        raise ValueError(f"{minutes_budget} minutes is too short, the warmup alone takes a good part of a minute")

    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}")

    train_data_loader, val_data_loader = load_dataloaders(training_set, history, policy_target, target_temperature, future, value_target, seed, batch_size)
    print(f"{training_set}: {len(train_data_loader.dataset):,} train positions, {len(val_data_loader.dataset):,} val positions, policy target {policy_target}, value target {value_target}")

    model = ChessTransformer(**model_settings).to(device)
    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"model: {model_settings}, {num_parameters:,} parameters")

    compiled_model = torch.compile(model)
    #fused does the whole update in one gpu kernel, it only exists for parameters that are on the gpu
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_learn_rate, fused=(device.type == "cuda"))

    #the schedule runs over the part of the budget that is left when the warmup ends
    warmup_end = 0.0
    def current_lr_multiplier(step: int) -> float:
        nonlocal warmup_end
        if step <= WARMUP_STEPS:
            warmup_end = budget.progress()
            return lr_multiplier(step, 0.0, schedule)
        return lr_multiplier(step, (budget.progress() - warmup_end) / (1 - warmup_end), schedule)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, current_lr_multiplier)

    #before any training the loss should be about ln(1968) = 7.6 and the accuracy about 0,
    #anything else means the data or the metrics are wrong
    #the value head starts at about ln(2) = 0.69 (eval) or ln(3) = 1.1 (result) and half or a third right
    val_loss, val_accuracy, val_value_loss, val_value_accuracy, val_future_loss = evaluate_model(compiled_model, val_data_loader, device, max_batches=EVAL_BATCHES)
    print(f"untrained: val loss {val_loss:.3f}, val accuracy {val_accuracy:.1%}, value loss {val_value_loss:.3f}, value accuracy {val_value_accuracy:.1%}")

    print(f"budget: {budget.describe()}, batches of {batch_size} positions, evaluating every {EVAL_EVERY} steps")
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
        #with "eval" the value loss is binary cross entropy against stockfish's eval, not comparable with a "result" run's
        "value_target": value_target,
        "future_loss_weight": future_loss_weight,
        "budget": budget.describe(),
        "batch_size": batch_size,
        "training_set": training_set,
        #with "stockfish" the val loss and accuracy are against stockfish's moves, not comparable with a "human" run's
        "policy_target": policy_target,
        "target_temperature": target_temperature if policy_target == "stockfish" else None,
        "note": note,
        "commit": current_commit()
    }
    (run_dir / "settings.json").write_text(json.dumps(run_settings, indent=4))

    #saved from model, not compiled_model: the compiled wrapper prefixes every name with "_orig_mod."
    def save_best_checkpoint() -> None:
        save_checkpoint(model, model_settings, run_dir / "model.pt")

    train(compiled_model, save_best_checkpoint, train_data_loader, val_data_loader, optimizer, scheduler, device, best_val_loss=val_loss, budget=budget, writer=writer, value_loss_weight=value_loss_weight, future_loss_weight=future_loss_weight)

    #the final weights as well: model.pt is the best evaluation's, which is usually but not always the last one
    save_checkpoint(model, model_settings, run_dir / "final.pt")

    #the number to quote for this run: the final weights on the whole val set
    val_loss, val_accuracy, val_value_loss, val_value_accuracy, val_future_loss = evaluate_model(compiled_model, val_data_loader, device)
    print(f"final, on all {len(val_data_loader.dataset):,} val positions: val loss {val_loss:.3f}, val accuracy {val_accuracy:.1%}, value loss {val_value_loss:.3f}, value accuracy {val_value_accuracy:.1%}, future loss {val_future_loss:.3f}")

    writer.add_scalar("final/val_loss", val_loss, budget.steps)
    writer.add_scalar("final/val_accuracy", val_accuracy, budget.steps)
    writer.add_scalar("final/val_value_loss", val_value_loss, budget.steps)
    writer.add_scalar("final/val_value_accuracy", val_value_accuracy, budget.steps)
    writer.close()

    run_settings.update(
        final_val_loss=val_loss,
        final_val_accuracy=val_accuracy,
        final_val_value_loss=val_value_loss,
        final_val_value_accuracy=val_value_accuracy,
        final_val_future_loss=val_future_loss,
        train_seconds=budget.seconds,
        positions_seen=budget.positions,
        finished_at=datetime.now().isoformat(timespec="seconds")
    )
    (run_dir / "settings.json").write_text(json.dumps(run_settings, indent=4))

    #the table and chart of every run so far, imported here because compare_runs imports this file
    from tools.compare_runs import write_progress_page
    print(f"progress page: {write_progress_page()}")


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
    parser.add_argument("--lr", type=float, default=PEAK_LEARN_RATE, help="peak learning rate, default %(default)s")
    parser.add_argument("--templates", type=int, default=POSITION_TEMPLATES, help="attention bias templates the blocks mix, default %(default)s")
    parser.add_argument("--training-set", default=TRAINING_SET, help="name of a training set in processed_data/, several joined with + to mix them in equal shares (val is the first one's), or carlsen. Default %(default)s")
    parser.add_argument("--value-weight", type=float, default=VALUE_LOSS_WEIGHT, help="weight of the value head's loss next to the move loss, default %(default)s")
    parser.add_argument("--value-target", choices=["eval", "result"], default=VALUE_TARGET, help="what the value head predicts: stockfish's eval of the position, or how the game ended. Default %(default)s")
    parser.add_argument("--history", type=int, default=HISTORY, help="past plies the model is shown, default %(default)s")
    parser.add_argument("--schedule", choices=["cosine", "hold", "constant"], default=SCHEDULE, help="what the learning rate does after the warmup, see SCHEDULE in main.py. Default %(default)s")
    parser.add_argument("--swiglu", action="store_true", help="a swiglu mlp in every block instead of the gelu one")
    parser.add_argument("--seed", type=int, default=SEED, help="runs that differ only in this show how much of a difference between two runs is chance. Default %(default)s")
    parser.add_argument("--mlp-ratio", type=float, help="hidden size of the mlp in every block as a multiple of d-model, default 4 (8/3 with --swiglu)")
    parser.add_argument("--policy-target", choices=["human", "stockfish"], default=POLICY_TARGET, help="the move that was played, or stockfish's best moves for the sets that have them (baseline_sf). Default %(default)s")
    parser.add_argument("--target-temperature", type=float, default=TARGET_TEMPERATURE, help="how stockfish's moves share the target, see TARGET_TEMPERATURE in main.py. Default %(default)s")
    parser.add_argument("--future", type=int, default=FUTURE, help="plies ahead the future head predicts the board's changes, 0 for no future head, default %(default)s")
    parser.add_argument("--future-weight", type=float, default=FUTURE_LOSS_WEIGHT, help="weight of the future head's loss, default %(default)s")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="positions per step, for train and val. Lower it for a model that does not fit, default %(default)s")
    parser.add_argument("--kv-heads", type=int, help="key/value heads, each shared by a group of heads (grouped-query attention). Default one for every head")
    parser.add_argument("--qk-ratio", type=float, nargs="+", default=[1.0], help="width of the queries and keys as a multiple of the values', rounded to whole dims per head. One for every block or one per block, e.g. --qk-ratio 0.25 0.5 0.5 0.75. Default 1")
    arguments = parser.parse_args()

    if not arguments.no_tensorboard:
        #imported here because tools/tensorboard_server.py imports RUNS_DIR from this file
        from tools.tensorboard_server import start_tensorboard
        start_tensorboard()

    main(arguments.positions, arguments.minutes, arguments.name, arguments.note, arguments.d_model, arguments.layers, arguments.heads, arguments.lr, arguments.templates, arguments.training_set, arguments.value_weight, arguments.history, arguments.schedule, arguments.swiglu, arguments.seed, arguments.mlp_ratio, arguments.policy_target, arguments.target_temperature, arguments.future, arguments.future_weight, arguments.value_target, arguments.qk_ratio[0] if len(arguments.qk_ratio) == 1 else arguments.qk_ratio, arguments.kv_heads, arguments.batch_size)

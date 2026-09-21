#for reference: the first model (carlsen pgn only) trained for 10 epochs of 580,949 positions, about 4 minutes.
#   the same number of positions: 5,673 steps   (--positions 5_800_000)
#   the same training time:       12,000 steps  (--positions 12_300_000, or simply --minutes 4), at about 58 steps a second
import argparse
import json
import queue
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from datetime import datetime
from itertools import islice
from pathlib import Path
from data import ARTIFACTS_DIR, CARLSEN_PGN, pgns_to_examples, create_dataloaders, datasets_to_dataloaders, load_training_set
import torch
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

PEAK_LEARN_RATE = 6e-3
WARMUP_STEPS = 500
BATCH_SIZE = 1024

#minutes of pure training (compiling and evaluating are not counted), unless --minutes or --positions is given.
#runs with the same budget are comparable, so architecture experiments should all use this one
TRAIN_MINUTES_BUDGET = 4

EVAL_EVERY = 100  #steps between two looks at the val set
POSTFIX_EVERY = 20 #steps between two updates of the loss next to the progress bar, reading it makes the cpu wait for the gpu
EVAL_BATCHES = 20 #only this many val batches for those looks, the whole val set is kept for the end
SEED = 42
DROPOUT = 0.0

VALUE_LOSS_WEIGHT = 1.0

#"carlsen" parses pgn_data/Carlsen.pgn, which is in the repository and takes 20 seconds.
#anything else is the name of a training set in processed_data/ (see load_training_set in data.py)
TRAINING_SET = "baseline"


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


def train(
    model: nn.Module,
    save_best_checkpoint: Callable[[], None],
    train_data_loader: DataLoader,
    val_data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    best_val_loss: float,
    budget: Budget,
    writer: SummaryWriter,
    value_loss_weight: float = VALUE_LOSS_WEIGHT) -> None:

    #kept on the device: adding to it does not make the cpu wait for the gpu the way loss.item() does
    loss_total = torch.zeros((), device=device)
    total_examples = 0

    model.train()

    #dynamic_ncols: a bar wider than a resized terminal wraps, and every update becomes a new line
    #the bar counts steps, or seconds of training when the budget is in minutes
    progress = tqdm(total=budget.total, desc="train", dynamic_ncols=True, unit="s" if budget.timed else "it",
                    bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f} [{elapsed}<{remaining}{postfix}]")

    #results is how each position's game ended for the side to move: 0 lost, 1 draw, 2 won
    #history is what changed on the board in the plies before each position, empty for a model that is not shown any
    for batch, labels, results, history in batches_on_device(endless_batches(train_data_loader), device):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, value_logits = model(batch, history)

            loss = loss_function(logits, labels) + value_loss_weight * loss_function(value_logits, results)

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        progress.update(budget.step_done())
        scheduler.step()
        step = budget.steps

        loss_total += loss.detach() * len(labels)
        total_examples += len(labels)

        #the running average, a single batch's loss jumps around too much to read
        if step % POSTFIX_EVERY == 0:
            progress.set_postfix(loss=f"{loss_total.item() / total_examples:.3f}")

        #the last step is always evaluated, so the final weights get their chance to be the checkpoint
        if step % EVAL_EVERY == 0 or budget.finished():
            train_loss = loss_total.item() / total_examples #also waits for the gpu, so the pause starts when the steps are really done
            budget.pause() #evaluating does not count towards a budget in minutes
            val_loss, val_accuracy, val_value_loss, val_value_accuracy = evaluate_model(model, val_data_loader, loss_function, device, max_batches=EVAL_BATCHES)
            model.train()

            #tqdm.write rather than print, which would cut through the progress bar
            tqdm.write(f"step {step} ({budget.positions:,} positions, {budget.seconds / 60:.1f} min): train loss {train_loss:.3f}, val loss {val_loss:.3f}, val accuracy {val_accuracy:.1%}, value loss {val_value_loss:.3f}, value accuracy {val_value_accuracy:.1%}")

            #the part of the name before the slash groups the plots: both losses end up next to each other
            writer.add_scalar("loss/train", train_loss, step)
            writer.add_scalar("loss/val", val_loss, step)
            writer.add_scalar("accuracy/val", val_accuracy, step)
            writer.add_scalar("loss/val_value", val_value_loss, step)
            writer.add_scalar("accuracy/val_value", val_value_accuracy, step)
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


#returns average loss and accuracy for the model on the test set, or on its first max_batches batches: first of the
#moves, then of the value head. The move loss stays on its own so it can be compared with runs from before the value head.
#the val loader is not shuffled, so those are the same positions every time and the numbers can be compared
def evaluate_model(
    model: nn.Module, 
    test_data_loader: DataLoader, 
    loss_function: nn.Module, 
    device: torch.device,
    max_batches: int | None = None) -> tuple[float, float, float, float]:

    epoch_loss_total = 0.0
    total_correct_preds = 0
    value_loss_total = 0.0
    value_correct_preds = 0
    total_examples = 0

    model.eval()

    with torch.no_grad():

        num_batches = len(test_data_loader) if max_batches is None else min(max_batches, len(test_data_loader))
        #islice stops after num_batches, leave=False clears the bar afterwards
        progress = tqdm(islice(test_data_loader, num_batches), total=num_batches, desc="val", leave=False, dynamic_ncols=True)

        for batch, labels, results, history in progress:
            labels = labels.to(device)
            results = results.to(device)
            batch = batch.to(device)
            history = history.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                #should be of dim B 1 output_dim
                logits, value_logits = model(batch, history)

                model_preds = torch.softmax(logits, dim=-1).argmax(dim=-1)

                loss = loss_function(logits, labels)
                value_loss = loss_function(value_logits, results)

            epoch_loss_total += loss.item() * len(labels)
            total_correct_preds += (model_preds == labels).sum().item()
            value_loss_total += value_loss.item() * len(labels)
            value_correct_preds += (value_logits.argmax(dim=-1) == results).sum().item()
            total_examples += len(labels)

            progress.set_postfix(
                loss=f"{epoch_loss_total / total_examples:.3f}",
                accuracy=f"{total_correct_preds / total_examples:.1%}"
            )

    return (epoch_loss_total / total_examples), (total_correct_preds / total_examples), (value_loss_total / total_examples), (value_correct_preds / total_examples)


def load_dataloaders(training_set: str = TRAINING_SET, history: int = 0) -> tuple[DataLoader, DataLoader]:
    if training_set == "carlsen":
        print("parsing games, this takes a while")
        games = pgns_to_examples([CARLSEN_PGN])
        return create_dataloaders(
            games,
            train_batch_size=BATCH_SIZE,
            val_batch_size=BATCH_SIZE,
            seed=SEED,
            history=history
        )

    train_dataset, val_dataset = load_training_set(training_set, history)
    return datasets_to_dataloaders(
        train_dataset,
        val_dataset,
        train_batch_size=BATCH_SIZE,
        val_batch_size=BATCH_SIZE
    )

#progress is how much of the run after the warmup is done, from 0 to 1
def lr_multiplier(step: int, progress: float) -> float:
    if step <= WARMUP_STEPS:
        return (step + 1) / WARMUP_STEPS

    multiplier = 0.5 * (1 + math.cos(math.pi * progress))

    return multiplier


#the commit the code was at, with a + when there were uncommitted changes on top of it
def current_commit() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("+" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return ""


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
    history: int = HISTORY):

    #a budget in positions is only used when one is asked for
    if positions_budget is not None:
        budget = Budget(BATCH_SIZE, positions=positions_budget)
    else:
        budget = Budget(BATCH_SIZE, minutes=minutes_budget)
    if run_name is None:
        #the settings that differ between experiments, so the plots in tensorboard label themselves
        run_name = f"d{d_model}_L{num_layers}_h{n_heads}_lr{peak_learn_rate:g}_{budget.describe()}_{datetime.now():%H-%M-%S}"

    #everything ChessTransformer is built from: saved in the checkpoint so it can be rebuilt when loading
    model_settings = {"d_model": d_model, "num_layers": num_layers, "n_heads": n_heads, "position_templates": position_templates, "history": history, "dropout": DROPOUT}

    #a run this short would end during the warmup, without ever reaching PEAK_LEARN_RATE
    if not budget.timed and budget.max_steps <= WARMUP_STEPS:
        raise ValueError(f"{positions_budget:,} positions is {budget.max_steps} steps, it has to be more than the {WARMUP_STEPS} warmup steps")
    if budget.timed and budget.max_seconds < 60:
        raise ValueError(f"{minutes_budget} minutes is too short, the warmup alone takes a good part of a minute")

    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}")

    train_data_loader, val_data_loader = load_dataloaders(training_set, history)
    print(f"{training_set}: {len(train_data_loader.dataset):,} train positions, {len(val_data_loader.dataset):,} val positions")

    model = ChessTransformer(**model_settings).to(device)
    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"model: {model_settings}, {num_parameters:,} parameters")
    loss_function = nn.CrossEntropyLoss()

    compiled_model = torch.compile(model)
    #fused does the whole update in one gpu kernel, it only exists for parameters that are on the gpu
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_learn_rate, fused=(device.type == "cuda"))

    #the cosine runs over the part of the budget that is left when the warmup ends
    warmup_end = 0.0
    def current_lr_multiplier(step: int) -> float:
        nonlocal warmup_end
        if step <= WARMUP_STEPS:
            warmup_end = budget.progress()
            return lr_multiplier(step, 0.0)
        return lr_multiplier(step, (budget.progress() - warmup_end) / (1 - warmup_end))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, current_lr_multiplier)

    #before any training the loss should be about ln(1968) = 7.6 and the accuracy about 0,
    #anything else means the data or the metrics are wrong
    #the value head starts at about ln(3) = 1.1 and a third right
    val_loss, val_accuracy, val_value_loss, val_value_accuracy = evaluate_model(compiled_model, val_data_loader, loss_function, device, max_batches=EVAL_BATCHES)
    print(f"untrained: val loss {val_loss:.3f}, val accuracy {val_accuracy:.1%}, value loss {val_value_loss:.3f}, value accuracy {val_value_accuracy:.1%}")

    print(f"budget: {budget.describe()}, batches of {BATCH_SIZE} positions, evaluating every {EVAL_EVERY} steps")
    run_dir = RUNS_DIR / run_name
    print(f"logging to {run_dir.relative_to(ARTIFACTS_DIR.parent)}")
    writer = SummaryWriter(log_dir=str(run_dir))

    #compare_runs.py reads this. The results are added once the run has finished
    run_settings = {
        "model_settings": model_settings,
        "parameters": num_parameters,
        "peak_learn_rate": peak_learn_rate,
        "value_loss_weight": value_loss_weight,
        "budget": budget.describe(),
        "batch_size": BATCH_SIZE,
        "training_set": training_set,
        "note": note,
        "commit": current_commit()
    }
    (run_dir / "settings.json").write_text(json.dumps(run_settings, indent=4))

    #saved from model, not compiled_model: the compiled wrapper prefixes every name with "_orig_mod."
    def save_best_checkpoint() -> None:
        save_checkpoint(model, model_settings, run_dir / "model.pt")

    train(compiled_model, save_best_checkpoint, train_data_loader, val_data_loader, loss_function, optimizer, scheduler, device, best_val_loss=val_loss, budget=budget, writer=writer, value_loss_weight=value_loss_weight)

    #the number to quote for this run: the final weights on the whole val set
    val_loss, val_accuracy, val_value_loss, val_value_accuracy = evaluate_model(compiled_model, val_data_loader, loss_function, device)
    print(f"final, on all {len(val_data_loader.dataset):,} val positions: val loss {val_loss:.3f}, val accuracy {val_accuracy:.1%}, value loss {val_value_loss:.3f}, value accuracy {val_value_accuracy:.1%}")

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
    parser.add_argument("--history", type=int, default=HISTORY, help="past plies the model is shown, default %(default)s")
    arguments = parser.parse_args()

    if not arguments.no_tensorboard:
        #imported here because tools/tensorboard_server.py imports RUNS_DIR from this file
        from tools.tensorboard_server import start_tensorboard
        start_tensorboard()

    main(arguments.positions, arguments.minutes, arguments.name, arguments.note, arguments.d_model, arguments.layers, arguments.heads, arguments.lr, arguments.templates, arguments.training_set, arguments.value_weight, arguments.history)

#for reference: the first model (carlsen pgn only) trained for 10 epochs of 580,949 positions, about 4 minutes.
#   the same number of positions: 5,673 steps   (TRAIN_POSITIONS_BUDGET = 5_800_000)
#   the same training time:       12,000 steps  (TRAIN_POSITIONS_BUDGET = 12_300_000), at about 58 steps a second
import argparse
import json
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
from tools.checkpoint import save_checkpoint
from model import ChessTransformer

#the model, each of these has a command line flag (python main.py --help)
D_MODEL = 128
NUM_LAYERS = 4
N_HEADS = 4

PEAK_LEARN_RATE = 6e-3
WARMUP_STEPS = 500
BATCH_SIZE = 1024

#positions to train on, unless --positions is given. A step is one batch, so steps = positions // BATCH_SIZE
TRAIN_POSITIONS_BUDGET = 15_000_000

EVAL_EVERY = 100  #steps between two looks at the val set
EVAL_BATCHES = 20 #only this many val batches for those looks, the whole val set is kept for the end
SEED = 42
DROPOUT = 0.0

#"carlsen" parses pgn_data/Carlsen.pgn, which is in the repository and takes 20 seconds.
#anything else is the name of a training set in processed_data/ (see load_training_set in data.py)
TRAINING_SET = "baseline"

#the best checkpoint of the latest run. Every run also keeps its own in runs/<name>/model.pt
CHECKPOINT_PATH = ARTIFACTS_DIR / "model.pt"
RUNS_DIR = ARTIFACTS_DIR / "runs" #one folder of tensorboard logs per run

def endless_batches(data_loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for batch, labels in data_loader:
            yield batch, labels


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
    max_steps: int,
    writer: SummaryWriter) -> None:

    loss_total = 0.0
    total_examples = 0

    model.train()

    #dynamic_ncols: a bar wider than a resized terminal wraps, and every update becomes a new line
    progress = tqdm(range(1, max_steps + 1), desc="train", dynamic_ncols=True)

    for step, (batch, labels) in zip(progress, endless_batches(train_data_loader)):
        labels = labels.to(device)
        batch = batch.to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(batch)

            loss = loss_function(logits, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        scheduler.step()

        loss_total += loss.item() * len(labels)
        total_examples += len(labels)

        #the running average, a single batch's loss jumps around too much to read
        progress.set_postfix(loss=f"{loss_total / total_examples:.3f}")

        #the last step is always evaluated, so the final weights get their chance to be the checkpoint
        if step % EVAL_EVERY == 0 or step == max_steps:
            train_loss = loss_total / total_examples
            val_loss, val_accuracy = evaluate_model(model, val_data_loader, loss_function, device, max_batches=EVAL_BATCHES)
            model.train()

            #tqdm.write rather than print, which would cut through the progress bar
            tqdm.write(f"step {step}/{max_steps} ({step * BATCH_SIZE:,} positions): train loss {train_loss:.3f}, val loss {val_loss:.3f}, val accuracy {val_accuracy:.1%}")

            #the part of the name before the slash groups the plots: both losses end up next to each other
            writer.add_scalar("loss/train", train_loss, step)
            writer.add_scalar("loss/val", val_loss, step)
            writer.add_scalar("accuracy/val", val_accuracy, step)
            writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_best_checkpoint()

            loss_total = 0.0
            total_examples = 0


#returns average loss and accuracy for the model on the test set, or on its first max_batches batches.
#the val loader is not shuffled, so those are the same positions every time and the numbers can be compared
def evaluate_model(
    model: nn.Module, 
    test_data_loader: DataLoader, 
    loss_function: nn.Module, 
    device: torch.device,
    max_batches: int | None = None) -> tuple[float, float]:

    epoch_loss_total = 0.0
    total_correct_preds = 0
    total_examples = 0

    model.eval()

    with torch.no_grad():

        num_batches = len(test_data_loader) if max_batches is None else min(max_batches, len(test_data_loader))
        #islice stops after num_batches, leave=False clears the bar afterwards
        progress = tqdm(islice(test_data_loader, num_batches), total=num_batches, desc="val", leave=False, dynamic_ncols=True)

        for batch, labels in progress:
            labels = labels.to(device)
            batch = batch.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                #should be of dim B 1 output_dim
                logits = model(batch)

                model_preds = torch.softmax(logits, dim=-1).argmax(dim=-1)

                loss = loss_function(logits, labels)

            epoch_loss_total += loss.item() * len(labels)
            total_correct_preds += (model_preds == labels).sum().item()
            total_examples += len(labels)

            progress.set_postfix(
                loss=f"{epoch_loss_total / total_examples:.3f}",
                accuracy=f"{total_correct_preds / total_examples:.1%}"
            )

    return (epoch_loss_total / total_examples), (total_correct_preds / total_examples)


def load_dataloaders() -> tuple[DataLoader, DataLoader]:
    if TRAINING_SET == "carlsen":
        print("parsing games, this takes a while")
        games = pgns_to_examples([CARLSEN_PGN])
        return create_dataloaders(
            games,
            train_batch_size=BATCH_SIZE,
            val_batch_size=BATCH_SIZE,
            seed=SEED
        )

    train_dataset, val_dataset = load_training_set(TRAINING_SET)
    return datasets_to_dataloaders(
        train_dataset,
        val_dataset,
        train_batch_size=BATCH_SIZE,
        val_batch_size=BATCH_SIZE
    )

def lr_multiplier(step: int, max_steps: int) -> float:
    if step <= WARMUP_STEPS:
        return (step + 1) / WARMUP_STEPS

    progress = (step - WARMUP_STEPS) / (max_steps - WARMUP_STEPS)

    multiplier = 0.5 * (1 + math.cos(math.pi * progress))

    return multiplier


def main(
    positions_budget: int = TRAIN_POSITIONS_BUDGET,
    run_name: str | None = None,
    d_model: int = D_MODEL,
    num_layers: int = NUM_LAYERS,
    n_heads: int = N_HEADS,
    peak_learn_rate: float = PEAK_LEARN_RATE):

    max_steps = positions_budget // BATCH_SIZE
    if run_name is None:
        #the settings that differ between experiments, so the plots in tensorboard label themselves
        run_name = f"d{d_model}_L{num_layers}_h{n_heads}_lr{peak_learn_rate:g}_{positions_budget / 1e6:g}M_{datetime.now():%H-%M-%S}"

    #everything ChessTransformer is built from: saved in the checkpoint so it can be rebuilt when loading
    model_settings = {"d_model": d_model, "num_layers": num_layers, "n_heads": n_heads, "dropout": DROPOUT}

    #a run this short would end during the warmup, without ever reaching PEAK_LEARN_RATE
    if max_steps <= WARMUP_STEPS:
        raise ValueError(f"{positions_budget:,} positions is {max_steps} steps, it has to be more than the {WARMUP_STEPS} warmup steps")

    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}")

    train_data_loader, val_data_loader = load_dataloaders()
    print(f"{TRAINING_SET}: {len(train_data_loader.dataset):,} train positions, {len(val_data_loader.dataset):,} val positions")

    model = ChessTransformer(**model_settings).to(device)
    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"model: {model_settings}, {num_parameters:,} parameters")
    loss_function = nn.CrossEntropyLoss()

    compiled_model = torch.compile(model)
    #fused does the whole update in one gpu kernel, it only exists for parameters that are on the gpu
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_learn_rate, fused=(device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_multiplier(step, max_steps))

    #before any training the loss should be about ln(1968) = 7.6 and the accuracy about 0,
    #anything else means the data or the metrics are wrong
    val_loss, val_accuracy = evaluate_model(compiled_model, val_data_loader, loss_function, device, max_batches=EVAL_BATCHES)
    print(f"untrained: val loss {val_loss:.3f}, val accuracy {val_accuracy:.1%}")

    print(f"{max_steps:,} steps of {BATCH_SIZE} positions, evaluating every {EVAL_EVERY}")
    run_dir = RUNS_DIR / run_name
    print(f"logging to {run_dir.relative_to(ARTIFACTS_DIR.parent)}")
    writer = SummaryWriter(log_dir=str(run_dir))

    #compare_runs.py reads this. The results are added once the run has finished
    run_settings = {
        "model_settings": model_settings,
        "parameters": num_parameters,
        "peak_learn_rate": peak_learn_rate,
        "positions_budget": positions_budget,
        "batch_size": BATCH_SIZE,
        "training_set": TRAINING_SET
    }
    (run_dir / "settings.json").write_text(json.dumps(run_settings, indent=4))

    #saved from model, not compiled_model: the compiled wrapper prefixes every name with "_orig_mod."
    def save_best_checkpoint() -> None:
        save_checkpoint(model, model_settings, CHECKPOINT_PATH)
        save_checkpoint(model, model_settings, run_dir / "model.pt")

    start_time = time.time()
    train(compiled_model, save_best_checkpoint, train_data_loader, val_data_loader, loss_function, optimizer, scheduler, device, best_val_loss=val_loss, max_steps=max_steps, writer=writer)

    #the number to quote for this run: the final weights on the whole val set
    val_loss, val_accuracy = evaluate_model(compiled_model, val_data_loader, loss_function, device)
    print(f"final, on all {len(val_data_loader.dataset):,} val positions: val loss {val_loss:.3f}, val accuracy {val_accuracy:.1%}")

    writer.add_scalar("final/val_loss", val_loss, max_steps)
    writer.add_scalar("final/val_accuracy", val_accuracy, max_steps)
    writer.close()

    run_settings.update(final_val_loss=val_loss, final_val_accuracy=val_accuracy, train_seconds=time.time() - start_time)
    (run_dir / "settings.json").write_text(json.dumps(run_settings, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train the chess transformer")
    #int() accepts underscores, so --positions 12_300_000 works
    parser.add_argument("--positions", type=int, default=TRAIN_POSITIONS_BUDGET, help="positions to train on, default %(default)s")
    parser.add_argument("--name", help="name of this run in tensorboard, default the date and time")
    parser.add_argument("--no-tensorboard", action="store_true", help="do not start tensorboard or open the browser")
    parser.add_argument("--d-model", type=int, default=D_MODEL, help="width of the model, default %(default)s")
    parser.add_argument("--layers", type=int, default=NUM_LAYERS, help="number of transformer blocks, default %(default)s")
    parser.add_argument("--heads", type=int, default=N_HEADS, help="attention heads, d-model has to divide by it, default %(default)s")
    parser.add_argument("--lr", type=float, default=PEAK_LEARN_RATE, help="peak learning rate, default %(default)s")
    arguments = parser.parse_args()

    if not arguments.no_tensorboard:
        #imported here because tools/tensorboard_server.py imports RUNS_DIR from this file
        from tools.tensorboard_server import start_tensorboard
        start_tensorboard()

    main(arguments.positions, arguments.name, arguments.d_model, arguments.layers, arguments.heads, arguments.lr)

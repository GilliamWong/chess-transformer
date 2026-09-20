#for reference: the first model (carlsen pgn only) trained for 10 epochs of 580,949 positions, about 4 minutes.
#   the same number of positions: 5,673 steps   (TRAIN_POSITIONS_BUDGET = 5_800_000)
#   the same training time:       12,000 steps  (TRAIN_POSITIONS_BUDGET = 12_300_000), at about 58 steps a second
import argparse
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from collections.abc import Iterator
from datetime import datetime
from itertools import islice
from pathlib import Path
from data import CARLSEN_PGN, pgns_to_examples, create_dataloaders, datasets_to_dataloaders, load_training_set
import torch
import math
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import ChessTransformer

PEAK_LEARN_RATE = 1e-2
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

CHECKPOINT_PATH = Path(__file__).parent / "model.pt" #weights from the evaluation with the best val loss
RUNS_DIR = Path(__file__).parent / "runs" #one folder of tensorboard logs per run: tensorboard --logdir runs
TENSORBOARD_PORT = 6006

def endless_batches(data_loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for batch, labels in data_loader:
            yield batch, labels


def train(
    model: nn.Module,
    checkpoint_model: nn.Module,
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
                #saved from checkpoint_model, not the compiled model: the compiled wrapper prefixes every
                #name with "_orig_mod." and play/player.py could not load that
                torch.save(checkpoint_model.state_dict(), CHECKPOINT_PATH)
                tqdm.write(f"  saved to {CHECKPOINT_PATH.name}")

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


def tensorboard_is_running() -> bool:
    with socket.socket() as connection:
        return connection.connect_ex(("localhost", TENSORBOARD_PORT)) == 0

#starts tensorboard unless one is already serving, and opens it in the browser once it answers.
#it is a separate process, so it keeps serving the plots after training has finished
def start_tensorboard() -> None:
    if not tensorboard_is_running():
        RUNS_DIR.mkdir(exist_ok=True)
        subprocess.Popen(
            [sys.executable, "-m", "tensorboard.main", "--logdir", str(RUNS_DIR), "--port", str(TENSORBOARD_PORT)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    def open_when_ready() -> None:
        for _ in range(60):
            if tensorboard_is_running():
                webbrowser.open(f"http://localhost:{TENSORBOARD_PORT}")
                return
            time.sleep(0.5)

    #in the background, tensorboard takes a few seconds to come up and training should not wait for it
    threading.Thread(target=open_when_ready, daemon=True).start()


def main(positions_budget: int = TRAIN_POSITIONS_BUDGET, run_name: str | None = None):
    max_steps = positions_budget // BATCH_SIZE
    if run_name is None:
        run_name = datetime.now().strftime("%m-%d_%H-%M-%S")

    #a run this short would end during the warmup, without ever reaching PEAK_LEARN_RATE
    if max_steps <= WARMUP_STEPS:
        raise ValueError(f"{positions_budget:,} positions is {max_steps} steps, it has to be more than the {WARMUP_STEPS} warmup steps")

    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}")

    train_data_loader, val_data_loader = load_dataloaders()
    print(f"{TRAINING_SET}: {len(train_data_loader.dataset):,} train positions, {len(val_data_loader.dataset):,} val positions")

    model = ChessTransformer(dropout=DROPOUT).to(device)
    loss_function = nn.CrossEntropyLoss()

    compiled_model = torch.compile(model)
    #fused does the whole update in one gpu kernel, it only exists for parameters that are on the gpu
    optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LEARN_RATE, fused=(device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_multiplier(step, max_steps))

    #before any training the loss should be about ln(1968) = 7.6 and the accuracy about 0,
    #anything else means the data or the metrics are wrong
    val_loss, val_accuracy = evaluate_model(compiled_model, val_data_loader, loss_function, device, max_batches=EVAL_BATCHES)
    print(f"untrained: val loss {val_loss:.3f}, val accuracy {val_accuracy:.1%}")

    print(f"{max_steps:,} steps of {BATCH_SIZE} positions, evaluating every {EVAL_EVERY}")
    print(f"logging to runs/{run_name}, watch with: tensorboard --logdir runs")
    writer = SummaryWriter(log_dir=str(RUNS_DIR / run_name))
    train(compiled_model, model, train_data_loader, val_data_loader, loss_function, optimizer, scheduler, device, best_val_loss=val_loss, max_steps=max_steps, writer=writer)

    #the number to quote for this run: the final weights on the whole val set
    val_loss, val_accuracy = evaluate_model(compiled_model, val_data_loader, loss_function, device)
    print(f"final, on all {len(val_data_loader.dataset):,} val positions: val loss {val_loss:.3f}, val accuracy {val_accuracy:.1%}")

    writer.add_scalar("final/val_loss", val_loss, max_steps)
    writer.add_scalar("final/val_accuracy", val_accuracy, max_steps)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train the chess transformer")
    #int() accepts underscores, so --positions 12_300_000 works
    parser.add_argument("--positions", type=int, default=TRAIN_POSITIONS_BUDGET, help="positions to train on, default %(default)s")
    parser.add_argument("--name", help="name of this run in tensorboard, default the date and time")
    parser.add_argument("--no-tensorboard", action="store_true", help="do not start tensorboard or open the browser")
    arguments = parser.parse_args()

    if not arguments.no_tensorboard:
        start_tensorboard()

    main(arguments.positions, arguments.name)

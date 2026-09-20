from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from data import CARLSEN_PGN, pgns_to_examples, create_dataloaders, datasets_to_dataloaders, load_training_set
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import ChessTransformer

LEARN_RATE = 1e-3
BATCH_SIZE = 1024

TRAIN_POSITIONS_BUDGET = 10_000_000
MAX_STEPS = TRAIN_POSITIONS_BUDGET // BATCH_SIZE #a step is one batch, one update of the weights

EVAL_EVERY = 1000  #steps between two looks at the val set
EVAL_BATCHES = 100 #only this many val batches for those looks, the whole val set is kept for the end
SEED = 42
#dropout guards against memorising a small dataset. With tens of millions of positions the model sees
#each one about once, so there is nothing to memorise and dropout only slows the learning down.
#worth turning back on (0.1) when training on "carlsen" alone
DROPOUT = 0.0

#"carlsen" parses pgn_data/Carlsen.pgn, which is in the repository and takes 20 seconds.
#anything else is the name of a training set in processed_data/ (see load_training_set in data.py)
TRAINING_SET = "baseline"

CHECKPOINT_PATH = Path(__file__).parent / "model.pt" #weights from the evaluation with the best val loss

def endless_batches(data_loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for batch, labels in data_loader:
            yield batch, labels

#trains the model for MAX_STEPS steps. Every EVAL_EVERY steps it looks at the val set, reports,
#and saves checkpoint_model if the val loss is the best so far
def train(
    model: nn.Module,
    checkpoint_model: nn.Module,
    train_data_loader: DataLoader,
    val_data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    best_val_loss: float) -> None:

    #set model.train()
    #take batches from the dataloader until we have done MAX_STEPS steps
    #for every batch from the dataloader and move to the gpu, then:
    #zero out gradients, put the data through the model
    #put model outputs through the loss function
    #call loss.backwards
    #then take a step with the optimizer
    #every EVAL_EVERY steps log the loss, evaluate, and save if it is the best so far

    loss_total = 0.0
    total_examples = 0

    model.train()

    progress = tqdm(range(1, MAX_STEPS + 1), desc="train")

    for step, (batch, labels) in zip(progress, endless_batches(train_data_loader)):
        labels = labels.to(device)
        batch = batch.to(device)

        #the matrix multiplies run in bfloat16, which is about twice as fast on the gpu. The weights stay
        #in float32. bfloat16 has the same range as float32, so unlike float16 it needs no GradScaler
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(batch)

            loss = loss_function(logits, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        loss_total += loss.item() * len(labels)
        total_examples += len(labels)

        #the running average, a single batch's loss jumps around too much to read
        progress.set_postfix(loss=f"{loss_total / total_examples:.3f}")

        #the last step is always evaluated, so the final weights get their chance to be the checkpoint
        if step % EVAL_EVERY == 0 or step == MAX_STEPS:
            train_loss = loss_total / total_examples
            val_loss, val_accuracy = evaluate_model(model, val_data_loader, loss_function, device, max_batches=EVAL_BATCHES)
            #evaluate_model leaves the model in eval mode, and nothing else would switch it back
            model.train()

            #tqdm.write rather than print, which would cut through the progress bar
            tqdm.write(f"step {step}/{MAX_STEPS} ({step * BATCH_SIZE:,} positions): train loss {train_loss:.3f}, val loss {val_loss:.3f}, val accuracy {val_accuracy:.1%}")

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
        progress = tqdm(islice(test_data_loader, num_batches), total=num_batches, desc="val", leave=False)

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


def main():
    #sets up the model, creates the dataset and dataloader, creates loss and optimizers
    #and also runs the training loop for MAX_STEPS steps

    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}")

    train_data_loader, val_data_loader = load_dataloaders()
    print(f"{TRAINING_SET}: {len(train_data_loader.dataset):,} train positions, {len(val_data_loader.dataset):,} val positions")

    model = ChessTransformer(dropout=DROPOUT).to(device)
    loss_function = nn.CrossEntropyLoss()

    compiled_model = torch.compile(model)
    #fused does the whole update in one gpu kernel, it only exists for parameters that are on the gpu
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARN_RATE, fused=(device.type == "cuda"))

    #before any training the loss should be about ln(1968) = 7.6 and the accuracy about 0,
    #anything else means the data or the metrics are wrong
    #on the same val batches as the evaluations during training, so it is the loss they have to beat
    val_loss, val_accuracy = evaluate_model(compiled_model, val_data_loader, loss_function, device, max_batches=EVAL_BATCHES)
    print(f"untrained: val loss {val_loss:.3f}, val accuracy {val_accuracy:.1%}")

    print(f"{MAX_STEPS:,} steps of {BATCH_SIZE} positions, evaluating every {EVAL_EVERY}")
    train(compiled_model, model, train_data_loader, val_data_loader, loss_function, optimizer, device, best_val_loss=val_loss)

    #the number to quote for this run: the final weights on the whole val set
    val_loss, val_accuracy = evaluate_model(compiled_model, val_data_loader, loss_function, device)
    print(f"final, on all {len(val_data_loader.dataset):,} val positions: val loss {val_loss:.3f}, val accuracy {val_accuracy:.1%}")


if __name__ == "__main__":
    main()

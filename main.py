from pathlib import Path
from data import CARLSEN_PGN, pgns_to_examples, create_dataloaders
import torch
from torch import nn
from torch.utils.data import DataLoader
from model import ChessTransformer

NUM_EPOCHS = 10
LEARN_RATE = 1e-3
BATCH_SIZE = 1024
SEED = 42

CHECKPOINT_PATH = Path(__file__).parent / "model.pt" #weights of the epoch with the best val loss

#trains the model for one epoch and returns averasge loss over that epoch
def train_one_epoch(
    model: nn.Module, 
    train_data_loader: DataLoader, 
    loss_function: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device) -> float:

    #set model.train()
    #loop through the dataloader
    #for every batch from the dataloader and move to the gpu, then:
    #zero out gradients, put the data through the model
    #put model outputs through the loss function
    #call loss.backwards
    #then take a step with the optimizer
    #log the loss and repeat the loop until the epoch is finished
    #then return the average loss

    epoch_loss_total = 0.0
    total_examples = 0

    model.train()

    for batch, labels in train_data_loader:
        labels = labels.to(device)
        batch = batch.to(device)

        logits = model(batch)

        loss = loss_function(logits, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        epoch_loss_total += loss.item() * len(labels)
        total_examples += len(labels)

    return epoch_loss_total / total_examples
    

#returns average loss and accuracy for the model on the test set
def evaluate_model(
    model: nn.Module, 
    test_data_loader: DataLoader, 
    loss_function: nn.Module, 
    device: torch.device) -> tuple[float, float]:

    epoch_loss_total = 0.0
    total_correct_preds = 0
    total_examples = 0

    model.eval()

    with torch.no_grad():

        for batch, labels in test_data_loader:
            labels = labels.to(device)
            batch = batch.to(device)

            #should be of dim B 1 output_dim
            logits = model(batch)

            model_preds = torch.softmax(logits, dim=-1).argmax(dim=-1)

            loss = loss_function(logits, labels)

            epoch_loss_total += loss.item() * len(labels)
            total_correct_preds += (model_preds == labels).sum().item()
            total_examples += len(labels)

    return (epoch_loss_total / total_examples), (total_correct_preds / total_examples)


def main():
    #sets up the model, creates the dataset and dataloader, creates loss and optimizers
    #and also runs the training loop for some number of epochs

    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}")

    print("parsing games, this takes a while")
    games = pgns_to_examples([CARLSEN_PGN])
    train_data_loader, val_data_loader = create_dataloaders(
        games,
        train_batch_size=BATCH_SIZE,
        val_batch_size=BATCH_SIZE,
        seed=SEED
    )
    print(f"{len(games)} games, {len(train_data_loader.dataset)} train positions, {len(val_data_loader.dataset)} val positions")

    model = ChessTransformer().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARN_RATE)

    #before any training the loss should be about ln(1968) = 7.6 and the accuracy about 0,
    #anything else means the data or the metrics are wrong
    val_loss, val_accuracy = evaluate_model(model, val_data_loader, loss_function, device)
    print(f"untrained: val loss {val_loss:.3f}, val accuracy {val_accuracy:.1%}")

    best_val_loss = val_loss

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_data_loader, loss_function, optimizer, device)
        val_loss, val_accuracy = evaluate_model(model, val_data_loader, loss_function, device)

        print(f"epoch {epoch}/{NUM_EPOCHS}: train loss {train_loss:.3f}, val loss {val_loss:.3f}, val accuracy {val_accuracy:.1%}")

        #once the model overfits, val loss rises while train loss keeps falling,
        #so the last epoch is not necessarily the best one
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  saved to {CHECKPOINT_PATH.name}")


if __name__ == "__main__":
    main()

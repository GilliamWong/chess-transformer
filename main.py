from data import CARLSEN_PGN, pgns_to_examples, create_dataloaders
import torch
from torch import nn
from torch.utils.data import DataLoader
from model import ChessTransformer

NUM_EPOCHS = 10
LEARN_RATE = 1e-3
BATCH_SIZE = 1024
SEED = 42

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



if __name__ == "__main__":
    main()

from dataclasses import dataclass
from torch import no_grad, max
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import torch.nn as nn
from .device import init_device


@dataclass
class TrainConfig:
    """
    Attributes:
        batch_size (int): the number of samples processed in one forward/backward pass during training.
        epochs (int): the maximum number of training iterations for the model.
        learning_rate (float): ontrols the step size at which the model updates its weights during training
    """
    batch_size: int
    epochs: int
    learning_rate: float


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.CrossEntropyLoss,
        epochs: int
    ) -> None:        
        device = init_device()
        model = model.to(device)
        best_val_loss = float('inf')
        patience = 5
        trigger_times = 0
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            # adjust weights and biases
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            # test against validation data
            model.eval()
            val_loss = 0.0
            with no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)

                    val_loss += loss.item()
            val_loss /= len(val_loader)
            print(f'epoch {epoch + 1}: validation Loss: {val_loss}')
            # verify that our accuracy is improving, otherwise stop early
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print('early stopping!')
                    break
        # output model's final accuracy against validation data
        correct = 0
        total = 0
        with no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        print(f"accuracy: {100 * correct / total:.2f}%")

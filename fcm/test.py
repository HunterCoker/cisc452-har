from dataclasses import dataclass
from torch import no_grad, max
from torch.utils.data import DataLoader
import torch.nn as nn
from .device import init_device


@dataclass
class TestConfig:
    """
    Attributes:
        model_path (str): the path of the model you want to test relative to the root of this project.
    """
    model_path: str


def test_model(
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.CrossEntropyLoss
    ) -> None:
        device = init_device()
        model = model.to(device)
        model.eval()

        test_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item()

                _, predicted = max(outputs, dim=1)
                correct_predictions += (predicted == y_batch).sum().item()
                total_samples += y_batch.size(0)

        avg_loss = test_loss / len(data_loader)
        accuracy = (correct_predictions / total_samples) * 100

        print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
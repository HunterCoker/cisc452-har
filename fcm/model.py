from torch import save, load, tensor
import torch.nn as nn
import os


class Model(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int
    ) -> None:
        super(Model, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_size)
        )

    def forward(self, x: tensor):
        # x shape: (batch_size, seq_length, input_size)
        x = x.permute(0, 2, 1)  # (batch_size, input_size, seq_length)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # (batch_size, seq_length_new, features)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # take the output from the last time step
        x = self.fc(x)
        return x
    
    def export(self, name: str) -> None:
        save(self.state_dict(), os.path.join("models/", name))

    def load(self, filepath) -> bool:
        self.load_state_dict(load(filepath))


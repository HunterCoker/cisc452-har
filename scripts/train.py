import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# for preprocessing dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import numpy as np
# for loss and optimizer
import torch.nn as nn
import torch.optim as optim
# project code
from fcm import Model, train_model, TrainConfig, WISDMDataset

config = TrainConfig(
    batch_size=64,
    epochs=50,
    learning_rate=0.0005
)

if __name__ == '__main__':
    # load the dataset and split it for validation
    dataset = WISDMDataset(train=True)
    train_indices, val_indices = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42)

    train_dataset = Subset(dataset, train_indices)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )

    val_dataset = Subset(dataset, val_indices)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    # configure model
    input_size = dataset.num_features
    model = Model(
        input_size=input_size,
        output_size=dataset.nclasses
    )

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=config.learning_rate
    )

    # train the model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=config.epochs
    )

    while True:
        ans = input('would you like to save this model? [y/n] ')
        if ans == 'y':
            name = input("provide a name for the model: ")
            model.export(name)
            break
        elif ans == 'n':
            break
        else:
            print(f'\'{ans}\' is not a valid response...')


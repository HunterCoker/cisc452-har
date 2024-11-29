import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# for preprocessing dataset
from torch.utils.data import DataLoader
# for loss and optimizer
import torch.nn as nn
import torch.optim as optim
# project code
from fcm import Model, test_model, TestConfig, WISDMDataset


# the names of models correspond to its accuracy on the validation dataset during training
#   95.65 -> 93.82 is an example of good generalization
#   98.07 -> 91.72 is an example of overfitting
config = TestConfig(
    model_path="models/95.65"
)


if __name__ == '__main__':
    dataset = WISDMDataset(train=False)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=True
    )

    input_size = dataset.num_features
    model = Model(
        input_size=input_size,
        output_size=dataset.nclasses
    )
    model.load(config.model_path)

    criterion = nn.CrossEntropyLoss()

    test_model(
        model=model,
        data_loader=data_loader,
        criterion=criterion
    )
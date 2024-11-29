from torch.utils.data import Dataset
from torch import tensor, float32, long
import pandas as pd
import numpy as np
import os


DATASET_PATH          = "datasets/UCI HAR Dataset/"
TRAIN_PATH            = os.path.join(DATASET_PATH, "train")
TEST_PATH             = os.path.join(DATASET_PATH, "test")
INERTIAL_SIGNALS_PATH = os.path.join(TRAIN_PATH, "Inertial Signals")


class WISDMDataset(Dataset):
    def __init__(
        self,
        train: bool
    ) -> None:
        self.train = train
        self.set_type = 'train' if train else 'test'

        # activity_labels = pd.read_csv(
        #     os.path.join(DATASET_PATH, "activity_labels.txt"),
        #     header=None,
        #     sep='\s+',
        #     names=["id", "activity"]
        # )
        # self.activity_labels_dict = dict(zip(activity_labels["id"], activity_labels["activity"]))
        self.nclasses = 6

        # features = pd.read_csv(
        #     os.path.join(DATASET_PATH, "features.txt"),
        #     header=None,
        #     sep='\s+',
        #     names=["index", "feature"]
        # )
        # self.features_list = features["feature"].tolist()
        self.num_features = 9

        self.signal_types = [
            'body_acc_x',
            'body_acc_y',
            'body_acc_z',
            'body_gyro_x',
            'body_gyro_y',
            'body_gyro_z',
            'total_acc_x',
            'total_acc_y',
            'total_acc_z'
        ]
        self.X, self.y = self._load_data(TRAIN_PATH if train else TEST_PATH)

    def _load_data(self, data_path):
        X_signals = []
        for signal in self.signal_types:
            filename = os.path.join(
                data_path,
                'Inertial Signals',
                f'{signal}_{self.set_type}.txt'
            )
            X_signal = np.loadtxt(filename).astype(np.float32)
            X_signals.append(X_signal)
        # stack signals along the last axis to get shape [num_samples, 128, 9]
        X = np.stack(X_signals, axis=-1)

        # load labels and convert to zero-based indexing
        y_path = os.path.join(data_path, f'y_{self.set_type}.txt')
        y = np.loadtxt(y_path).astype(int) - 1  # Labels are 1-based in the dataset
        
        return X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_sample, y_sample = self.X[idx], self.y[idx]
        if self.train:
            # add Gaussian noise to increase the diversity of training data
            noise = np.random.normal(0, 0.01, X_sample.shape)
            X_sample += noise
        return tensor(X_sample, dtype=float32), tensor(y_sample, dtype=long)
"""
prepare data for ssl training

Techniques:
- BYOL
- SimCLR
- MoCoV3
- DINO
"""

import torch
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
import torchvision.transforms as T
import pickle

import transforms as RST

from lightly.transforms.byol_transform import BYOLTransform


def loadRML22(filepath):
    """
    Load RadioML 22 https://github.com/venkateshsathya/RML22
    """
    with open(filepath, 'rb') as f:
        Xd = pickle.load(f, encoding='latin1')
    mods = [item[0] for item in Xd.keys()]
    snrs = [item[1] for item in Xd.keys()]
    X = []
    y = []
    name2label = {k: v for v, k in enumerate(sorted(set(mods)))}
    for mod, snr in zip(mods, snrs):
        b = Xd[(mod, snr)]
        X.append(b)
        y.extend([name2label[mod], ] * b.shape[0])
    X = np.vstack(X)
    y = np.array(y)
    return X, y


class IQDataset:
    def __init__(self, x, y, transform_x=None, transform_y=None):
        self.x = x
        self.y = y
        self.transform_x = transform_x
        self.transform_y = transform_y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        if self.transform_x:
            x = self.transform_x(x)
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if self.transform_y:
            y = self.transform_y(y)
        elif isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        return x, y


class BYOLRData:
    def __init__(self, X, y, nsplits=2, test_size=0.4, batch_size=64, num_workers=4):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.sss = StratifiedShuffleSplit(
            n_splits=nsplits, test_size=test_size, random_state=0)

    def get_transforms(self, signal_length):
        txforms = T.Compose([
            T.Lambda(lambda x: torch.from_numpy(x).float()),
            RST.RandomResample(max_rate=8),
            RST.RandomDCShift(max_shift=3),
            RST.RandomAmplitudeScale(scale_factor=0.1),
            # RST.RandomPhaseShift(),
            # RST.RandomFrequencyShift(),
            RST.RandomZeroMasking(max_mask_size=32, dim=-1),
            RST.RandomTimeShift(max_shift=16),
            # RST.TimeReversal(),
            # RST.RandomAddNoise(std=0.001)
            RST.RandomTimeCrop(crop_size=signal_length),  # without time crop we face the problem of different lengths # noqa
        ])

        return BYOLTransform(
            view_1_transform=txforms,
            view_2_transform=txforms
        )

    def get_datasets(self, signal_length=128):
        dev_idxs, val_idxs = next(self.sss.split(self.X, self.y))
        byol_dataset = IQDataset(
            self.X[dev_idxs], self.y[dev_idxs], transform_x=self.get_transforms(signal_length))
        train_dataset = IQDataset(self.X[dev_idxs], self.y[dev_idxs])
        val_dataset = IQDataset(self.X[val_idxs], self.y[val_idxs])

        return byol_dataset, train_dataset, val_dataset

    def get_dataloaders(self, signal_length=128):
        byol_dataset, train_dataset, val_dataset = self.get_datasets(
            signal_length)
        byol_loader = DataLoader(
            byol_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return byol_loader, train_loader, val_loader


class SimCLRData:
    pass


class MoCoV3Data:
    pass


class DINOData:
    pass

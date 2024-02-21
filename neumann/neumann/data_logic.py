import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


class RandomValuation(torch.utils.data.Dataset):
    """CLEVRHans dataset. 
    The implementations is mainly from https://github.com/ml-research/NeSyConceptLearner/blob/main/src/pretrain-slot-attention/data.py.
    """

    def __init__(self, dataset, split, atoms, n_data=2000):
        super().__init__()
        self.atoms = atoms
        self.n_data = n_data
        self.dataset = dataset
        assert split in {
            "train",
            "val",
            "test",
        }  # note: test isn't very useful since it doesn't have ground-truth scene information
        self.split = split


    def __getitem__(self, item):
        return torch.rand((len(self.atoms), ))

    def __len__(self):
        return self.n_data

class ZeroValuation(torch.utils.data.Dataset):
    """CLEVRHans dataset.
    The implementations is mainly from https://github.com/ml-research/NeSyConceptLearner/blob/main/src/pretrain-slot-attention/data.py.
    """

    def __init__(self, dataset, split, atoms, n_data=2000):
        super().__init__()
        self.atoms = atoms
        self.n_data = n_data
        self.dataset = dataset
        assert split in {
            "train",
            "val",
            "test",
        }  # note: test isn't very useful since it doesn't have ground-truth scene information
        self.split = split


    def __getitem__(self, item):
        return torch.zeros((len(self.atoms), ))

    def __len__(self):
        return self.n_data

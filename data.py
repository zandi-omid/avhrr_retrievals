"""
Provides the SPRSpatial dataset class to load training samples from the IPWGML
 SPR dataset.
"""
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def apply(tensors: Any, transform: torch.Tensor) -> torch.Tensor:
    """
    Apply transformation to any container containing torch.Tensors.

    Args:
        tensors: An arbitrarily nested list, dict, or tuple containing
            torch.Tensors.
        transform:

    Return:
        The same containiner but with the given transformation function applied to
        all tensors.
    """
    if isinstance(tensors, tuple):
        return tuple([apply(tensor, transform) for tensor in tensors])
    if isinstance(tensors, list):
        return [apply(tensor, transform) for tensors in tensors]
    if isinstance(tensors, dict):
        return {key: apply(tensor, transform) for key, tensor in tensors.items()}
    if isinstance(tensors, torch.Tensor):
        return transform(tensors)
    raise ValueError("Encountered an unsupported type %s in apply.", type(tensors))


class AVHRRDataset(Dataset):
    def __init__(self, config):
        """
        Dataset class for AVHRR input data.

        Args:
            config (dict): A configuration dictionary containing 'input_path'.
        """
        path = config["input_path"]
        arr = np.load(path)

        # Split input and target
        target_images = arr[:, -1, :, :] # Last channel is target
        input_images = arr[:, :-1, :, :] # Remaining are input features

        # Add channel dimension to targets: (N, 1, H, W)
        target_images = target_images[:, None, :, :]

        self.input_images = input_images
        self.target_images = target_images

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.input_images[idx]).float()
        y = torch.from_numpy(self.target_images[idx]).float()
        return x, y

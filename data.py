"""
Provides the SPRSpatial dataset class to load training samples from the IPWGML
 SPR dataset.
"""
from datetime import datetime
from functools import partial
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import ipwgml
from ipwgml import config
from ipwgml.data import download_missing
from ipwgml.input import GMI
from ipwgml.target import TargetConfig
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr


def get_median_time(filename_or_path: Path | str) -> datetime:
    """
    Get median time from the filanem of a IPWGML SPR training scene.

    Args:
        filename: The filename or path pointing to any spatial training data file.

    Return:
        A datetime object representing the median time of the training scene.
    """
    if isinstance(filename_or_path, Path):
        filename = filename_or_path.name
    else:
        filename = filename_or_path
    date_str = filename.split("_")[-1][:-3]
    median_time = datetime.strptime(date_str, "%Y%m%d%H%M%S")
    return median_time


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


class SPRSpatial(Dataset):
    """
    Dataset class providing access to the spatial variant of the satellite precipitation retrieval
    benchmark dataset.
    """
    def __init__(
            self,
            split: str
    ):
        super().__init__()

        ipwgml_path = config.get_data_path()
        self.reference_sensor = "gmi"
        self.geometry = "gridded"
        if not split.lower() in ["training", "validation", "testing"]:
            raise ValueError(
                "Split must be one of ['training', 'validation', 'testing']"
            )
        self.split = split

        self.retrieval_input = GMI()
        self.target_config = TargetConfig()

        self.augment = split == "training"

        # Download GMI input files
        dataset = f"spr/{self.reference_sensor}/{self.split}/{self.geometry}/spatial/gmi"
        download_missing(dataset, ipwgml_path, progress_bar=True)
        files = sorted(list((ipwgml_path / dataset).glob("*.nc")))
        self.gmi_files = np.array(files)

        dataset = f"spr/{self.reference_sensor}/{self.split}/{self.geometry}/spatial/target"
        download_missing(dataset, ipwgml_path, progress_bar=True)
        files = sorted(list((ipwgml_path / dataset).glob("*.nc")))
        self.target = np.array(files)

        self.check_consistency()
        self.worker_init_fn(0)

    def worker_init_fn(self, w_id: int) -> None:
        """
        Seeds the dataset loader's random number generator.
        """
        seed = int.from_bytes(os.urandom(4), "big") + w_id
        self.rng = np.random.default_rng(seed)


    def check_consistency(self):
        """
        Check consistency of training files.

        Raises:
            RuntimeError when the training scenes for any of the inputs is inconsistent with those
            available for the target.
        """
        target_times = set(map(get_median_time, self.target))
        inpt_times = set(map(get_median_time, self.gmi_files))
        if target_times != inpt_times:
            raise RuntimeError(
                f"Available target times are inconsistent with input files for input {inpt}."
            )

    def __len__(self) -> int:
        """
        The number of samples in the dataset.
        """
        return len(self.target)

    def __getitem__(self, ind: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Load sample from dataset.
        """
        with xr.open_dataset(self.target[ind]) as data:
            target_time = data.time
            target = self.target_config.load_reference_precip(data)
            target = torch.tensor(target.astype(np.float32))

        input_data = {}
        files = self.gmi_files
        data = self.retrieval_input.load_data(
            files[ind],
            target_time=target_time,
        )
        for name, arr in data.items():
            input_data[name] = torch.tensor(arr.astype(np.float32))

        if self.augment:
            flip_h = self.rng.random() > 0.5
            flip_v = self.rng.random() > 0.5
            dims = tuple()
            if flip_h:
                dims = dims + (-2,)
            if flip_v:
                dims = dims + (-1,)

            input_data = apply(input_data, partial(torch.flip, dims=dims))
            target = apply(target, partial(torch.flip, dims=dims))

        return input_data, target

# Convolutional AVHRR retrievals

This repository provides a basic setup of a precipitation retrieval using an EfficientNet-V2 encoder-decoder architecture.

## Installation

Two external packages are required to run the training. The ``pytorch_retrieve`` package contains the implementation of the EfficientNet-V2 model and takes care of the training. The ``iwpgml`` package provides access to the IPWGML benchmark dataset, which is used to demonstrate how to train the EfficientNet-V2 model.

### pytorch_retrieve

To install PyTorch retrieve, it is recommended to clone the repository and install it in editable mode. This can be achieved by executing the following two commands in a terminal window:

``` shellsession
git clone https://github.com/simonpf/pytorch_retrieve
pip install -e pytorch_retrieve
```

### ipwgml

The ``ipwgml`` package can be installed directly from GitHub using the following command:


``` shellsession
pip install git+https://github.com/ipwgml/ipwgml
```

## Overview

In addition to this README, the repository contains the following files:

- ``data.py``: This Python module defines the Dataset class responsible for loading the training samples from the IPWGML SPR dataset
- ``model.toml``: This configuration file defines the model used for the retrieval.
- ``training.toml``: This configuration file defines the training schedule for the retrieval model.
- ``compute.toml``: This configuration file defines the compute environment to use for the training.

## Training

The training process involves two main steps:

### Step 1: EDA

Perform EDA to compute training data statistics required for normalizing input data:

``` shellsession
pytorch_retrieve eda
```

### Step 2: Training

After the EDA, the training can be run using:

``` shellsession
pytorch_retrieve train
```

## Adapting the retrieval

To adapt the retrieval to AVHRR data, you will need to adapt of write a new dataset class similar to the ``SPRSpatial`` class in ``data.py``.  Then you will have to update the name of the input (currently ``obs_gmi``) in the ``model.toml`` and the name of the dataset class in the ``training.toml``.

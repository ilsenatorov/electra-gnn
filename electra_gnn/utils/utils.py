from math import ceil
import numpy as np
import torch
from copy import deepcopy


def mask_molecules(orig_data, mask_ratio=0.2):
    data = deepcopy(orig_data)
    num_nodes = data.num_nodes
    num_masked_nodes = ceil(num_nodes * mask_ratio)
    idx = np.random.choice(range(num_nodes), size=num_masked_nodes, replace=False)
    data.x[idx] = torch.zeros_like(data.x[idx])
    data.masked_idx = idx
    return data


def corrupt_molecules(orig_data, new_features, masked_idx):
    data = deepcopy(orig_data)
    data.x[masked_idx] = new_features
    data.corrupt_idx = masked_idx
    return data

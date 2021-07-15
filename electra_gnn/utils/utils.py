from math import ceil
import numpy as np
import torch
from copy import deepcopy
import torch_geometric


def mask_molecules(orig_data: torch_geometric.data.Data,
                   mask_ratio: float = 0.2  ):
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
    data.y = (orig_data.x != data.x).long().unsqueeze(1)
    return data

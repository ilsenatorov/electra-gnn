from electra_gnn.utils.feat import featurize
from os import path as osp

import numpy as np
import pandas as pd
import torch
from deepchem.feat import MolGraphConvFeaturizer
from torch.utils import data
from torch_geometric.data import InMemoryDataset
from .feat import featurize, featurize_popular
from torch.utils.data import random_split
from math import ceil


class MoleculeDataset(InMemoryDataset):
    def __init__(self, filename, only_popular=True, transform=None, pre_transform=None):
        basefilename = osp.basename(filename)
        basefilename = osp.splitext(basefilename)[0]
        root = osp.join('data', filename + '_only_popular' if only_popular else '_all')
        self.only_popular = only_popular
        self.filename = filename
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        df = pd.read_csv(self.filename)
        assert('smiles' in df.columns)
        featurization_func = featurize_popular if self.only_popular else featurize
        data_list = [featurization_func(molecule) for molecule in df['smiles'].to_list()]
        if 'y' in df.columns:  # add labels if available
            for idx, y in enumerate(df['y']):
                data_list[idx].y = y
        data_list = [molecule for molecule in data_list if molecule is not None]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def split_dataset(dataset: MoleculeDataset, train_frac=0.7, val_frac=0.2):
    total_size = len(dataset)
    train_size = ceil(total_size * train_frac)
    val_size = ceil(total_size * val_frac)
    test_size = total_size - train_size - val_size
    return random_split(dataset, [train_size, val_size, test_size])

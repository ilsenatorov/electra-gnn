from os import path as osp

import numpy as np
import pandas as pd
import torch
from deepchem.feat import MolGraphConvFeaturizer
from torch.utils import data
from torch_geometric.data import InMemoryDataset


class MoleculeDataset(InMemoryDataset):
    def __init__(self, filename, transform=None, pre_transform=None):
        basefilename = osp.basename(filename)
        basefilename = osp.splitext(basefilename)[0]
        root = osp.join('data', filename)
        self.filename = filename
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        df = pd.read_csv(self.filename)
        assert('smiles' in df.columns)
        featuriser = MolGraphConvFeaturizer()
        data_list = featuriser.featurize(df['smiles'])
        data_list = [x.to_pyg_graph() for x in data_list if not isinstance(x, np.ndarray)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

from pytorch_lightning import LightningModule
from torch_geometric.nn import GCNConv
from torch.nn import Linear, Embedding
import torch.nn.functional as F


class Generator(LightningModule):
    def __init__(self,
                 num_atom_types=60,
                 atom_embedding_dim=30,
                 hidden_dim=16):
        super().__init__()
        self.embedding = Embedding(num_atom_types, atom_embedding_dim, padding_idx=0)
        self.gcn1 = GCNConv(atom_embedding_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = Linear(hidden_dim, num_atom_types)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.embedding(x)
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        x = x[data.masked_idx]
        x = self.lin(x)
        return x

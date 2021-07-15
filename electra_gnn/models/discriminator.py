from pytorch_lightning import LightningModule
from torch_geometric.nn import GCNConv
from torch.nn import Linear, Embedding
import torch.nn.functional as F


class Discriminator(LightningModule):
    def __init__(self,
                 num_atom_types=120,
                 atom_embedding_dim=30,
                 hidden_dim=32):
        super().__init__()
        self.embedding = Embedding(num_atom_types, atom_embedding_dim, padding_idx=0)
        self.gcn1 = GCNConv(atom_embedding_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.embedding(x)
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        x = self.lin(x)
        return x

    def embed(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.embedding(x)
        x = F.relu(self.gcn1(x, edge_index))
        x = self.gcn2(x, edge_index)
        return x

from pytorch_lightning import LightningModule
from torch_geometric.nn import GCNConv
from torch.nn import Linear


class Generator(LightningModule):
    def __init__(self, feat_dim=30,
                 hidden_dim=32):
        super().__init__()
        self.gcn1 = GCNConv(feat_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = Linear(hidden_dim, feat_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gcn1(x, edge_index)
        x = self.gcn2(x, edge_index)
        x = x[data.masked_idx]
        x = self.lin(x)
        return x

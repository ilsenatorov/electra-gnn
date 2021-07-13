from pytorch_lightning import Trainer
from torch_geometric.data import DataLoader

from electra_gnn.models.pretraining_model import PretrainingModel
from electra_gnn.utils.data import MoleculeDataset

dataset = MoleculeDataset('raw_data/chembl_smiles.csv')
model = PretrainingModel()
trainer = Trainer(gpus=1)
dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=8)

trainer.fit(model, dataloader)

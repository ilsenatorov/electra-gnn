from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch_geometric.data import DataLoader

from electra_gnn.models.pretraining_model import PretrainingModel
from electra_gnn.utils.data import MoleculeDataset

logger = TensorBoardLogger('tb_logs',
                           name='electra',
                           default_hp_metric=False)
callbacks = [
    ModelCheckpoint(monitor='loss',
                    save_top_k=3,
                    mode='min'),
    EarlyStopping(monitor='loss',
                  patience=10,
                  mode='min')
]
trainer = Trainer(gpus=1,
                  callbacks=callbacks,
                  logger=logger,
                  )


dataset = MoleculeDataset('raw_data/chembl_smiles.csv')
model = PretrainingModel()
trainer = Trainer(gpus=1)
dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=8)

trainer.fit(model, dataloader)

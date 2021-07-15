from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch_geometric.data import DataLoader

from electra_gnn.models.pretraining_model import PretrainingModel
from electra_gnn.utils.data import MoleculeDataset

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('data', type=str)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--gen_lr', type=float, default=0.001)
parser.add_argument('--disc_lr', type=float, default=0.001)
parser.add_argument('--gen_hidden_dim', type=int, default=16)
parser.add_argument('--disc_hidden_dim', type=int, default=64)
parser.add_argument('--atom_embedding_dim', type=int, default=16)
parser.add_argument('--mask_ratio', type=float, default=0.2)
args = vars(parser.parse_args())


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


dataset = MoleculeDataset(args['data'])
model = PretrainingModel(**args)
print(model)
dataloader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'])

trainer.fit(model, dataloader)

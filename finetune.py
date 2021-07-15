from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch_geometric.data import DataLoader

from electra_gnn.models.finetuning_model import FinetuningClassificationModel, FinetuningRegressionModel
from electra_gnn.utils.data import MoleculeDataset, split_dataset

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('data', type=str)
parser.add_argument('--pretrained_model_file', type=str, default=None,
                    help='Checkpoint file of the pretrained model. If not specified, model is trained from scratch')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--task', type=str, default='class', help="'class' or 'reg' for classification and regression")
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--hidden_dim', type=int, default=16)
parser.add_argument('--atom_embedding_dim', type=int, default=16)
args = vars(parser.parse_args())


logger = TensorBoardLogger('tb_logs',
                           name='finetuning',
                           default_hp_metric=False)
callbacks = [
    ModelCheckpoint(monitor='val_loss',
                    save_top_k=3,
                    mode='min'),
    EarlyStopping(monitor='val_loss',
                  patience=10,
                  mode='min')
]
trainer = Trainer(gpus=1,
                  callbacks=callbacks,
                  logger=logger,
                  )


dataset = MoleculeDataset(args['data'])
train, val, test = split_dataset(dataset)
train_dataloader = DataLoader(train, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'])
val_dataloader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_workers'])
test_dataloader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_workers'])
model = {'class': FinetuningClassificationModel, 'reg': FinetuningRegressionModel}[args['task']]
model = model(**args)

trainer.fit(model, train_dataloader, val_dataloader)
train.test(model, test_dataloader=test_dataloader)

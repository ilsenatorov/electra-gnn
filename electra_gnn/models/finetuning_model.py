from numpy.core.fromnumeric import mean
from pytorch_lightning import LightningModule
from torch.nn.modules.linear import Linear
from .generator import Generator
from .discriminator import Discriminator
import torch
import torch.nn.functional as F
from torch.optim import Adam
from .pretraining_model import PretrainingModel
from torch_geometric.nn import global_add_pool
from torchmetrics.functional import accuracy, auroc, explained_variance, mean_absolute_error


class FinetuningClassificationModel(LightningModule):
    def __init__(self,
                 num_atom_types=30,
                 atom_embedding_dim=16,
                 hidden_dim=32,
                 pretrained_model_file=None,
                 lr=0.001,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        if pretrained_model_file:
            pretrained = PretrainingModel.load_from_checkpoint(pretrained_model_file)
            self.disc = pretrained.discriminator
            self.lin = Linear(pretrained.hparams.disc_hidden_dim, 1)
        else:
            self.disc = Discriminator(num_atom_types, atom_embedding_dim, hidden_dim)
            self.lin = Linear(hidden_dim, 1)

    def forward(self, batch):
        x = self.disc.embed(batch)
        x = global_add_pool(x, batch.batch)
        x = self.lin(x)
        return x

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)

    def shared_step(self, batch):
        out = self.forward(batch)
        targ = batch.y.unsqueeze(1)
        loss = F.binary_cross_entropy_with_logits(out, targ.float())
        acc = accuracy(out, targ)
        # auc = auroc(out, targ)
        return {
            'loss': loss,
            'acc': acc,
            # 'auc': auc
        }

    def training_step(self, data, batch_idx):
        ss = self.shared_step(data)
        return ss

    def validation_step(self, data, data_idx):
        ss = self.shared_step(data)
        # val_loss has to be logged for early stopping and reduce_lr
        for key, value in ss.items():
            self.log('val_' + key, value)
        return ss

    def test_step(self, data, data_idx):
        ss = self.shared_step(data)
        return ss

    def training_epoch_end(self, outputs):
        entries = outputs[0].keys()
        for i in entries:
            val = torch.stack([x[i] for x in outputs]).mean()
            self.logger.experiment.add_scalar('train_epoch_' + i, val, self.current_epoch)

    def validation_epoch_end(self, outputs):
        entries = outputs[0].keys()
        for i in entries:
            val = torch.stack([x[i] for x in outputs]).mean()
            self.logger.experiment.add_scalar('val_epoch_' + i, val, self.current_epoch)

    def test_epoch_end(self, outputs):
        entries = outputs[0].keys()
        metrics = {}
        for i in entries:
            val = torch.stack([x[i] for x in outputs]).mean()
            metrics['test_' + i] = val
        self.logger.log_hyperparams(self.hparams, metrics)


class FinetuningRegressionModel(FinetuningClassificationModel):
    def shared_step(self, batch):
        out = self.forward(batch)
        loss = F.mse_loss(out, batch.y)
        expvar = explained_variance(out, batch.y)
        mae = mean_absolute_error(out, batch.y)
        return {'loss': loss,
                'acc': expvar,
                'auc': mae}

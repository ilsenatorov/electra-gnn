from pytorch_lightning import LightningModule
from .generator import Generator
from .discriminator import Discriminator
from ..utils import mask_molecules, corrupt_molecules
import torch
import torch.nn.functional as F
from torch.optim import Adam
from .pretraining_model import PretrainingModel


class FinetuningModel(LightningModule):
    def __init__(self,
                 feat_dim=30,
                 hidden_dim=32,
                 mask_ratio=0.2,
                 pretrained_model_file=None):
        super().__init__()
        self.save_hyperparameters()
        pretrained_disc = PretrainingModel.load_from_checkpoint(pretrained_model_file).discriminator

    def forward(self, batch):
        raise NotImplementedError()

from pytorch_lightning import LightningModule
from .generator import Generator
from .discriminator import Discriminator
from ..utils import mask_molecules, corrupt_molecules
import torch
import torch.nn.functional as F
from torch.optim import Adam


class PretrainingModel(LightningModule):
    def __init__(self, feat_dim=30,
                 hidden_dim=32,
                 mask_ratio=0.2):
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(feat_dim, hidden_dim)
        self.discriminator = Discriminator(feat_dim)
        self.gen_loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.disc_loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.automatic_optimization = False

    def generate(self, batch):
        masked_batch = mask_molecules(batch, mask_ratio=self.hparams.mask_ratio)
        generated_features = self.generator(masked_batch)  # Try to predict masked features from the original data
        return generated_features

    def training_step(self, batch, batch_idx):
        genoptim, discoptim = self.optimizers()
        masked_batch = mask_molecules(batch, mask_ratio=self.hparams.mask_ratio)
        masked_idx = masked_batch.masked_idx
        generated_features = self.generator(masked_batch)  # Try to predict masked features from the original data
        gen_loss = self.gen_loss(batch.x[masked_idx], generated_features)  # calculate loss
        genoptim.zero_grad()
        self.manual_backward(gen_loss)
        genoptim.step()

        sigmoid_generated_features = torch.sigmoid(generated_features).detach().round()
        corrupt_batch = corrupt_molecules(batch, sigmoid_generated_features, masked_idx)
        disc_prediction = self.discriminator(corrupt_batch)
        corruption_labels = torch.zeros((batch.num_nodes, 1), dtype=torch.float32, device=self.device)
        corruption_labels[masked_idx] = 1.
        disc_loss = self.disc_loss(corruption_labels, disc_prediction)
        discoptim.zero_grad()
        self.manual_backward(disc_loss)
        discoptim.step()
        self.log_dict({'gen_loss': gen_loss, 'disc_loss': disc_loss, 'loss': gen_loss+disc_loss})

    def configure_optimizers(self):
        genoptim = Adam(self.generator.parameters())
        discoptim = Adam(self.discriminator.parameters())
        return genoptim, discoptim

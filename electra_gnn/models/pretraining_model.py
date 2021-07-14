from pytorch_lightning import LightningModule
from .generator import Generator
from .discriminator import Discriminator
from ..utils import mask_molecules, corrupt_molecules
import torch
import torch.nn.functional as F
from torch.optim import Adam


class PretrainingModel(LightningModule):
    def __init__(self,
                 num_atom_types=60,
                 atom_embedding_dim=30,
                 gen_hidden_dim=16,
                 disc_hidden_dim=32,
                 mask_ratio=0.2,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(num_atom_types, atom_embedding_dim, hidden_dim=gen_hidden_dim)
        self.discriminator = Discriminator(num_atom_types, atom_embedding_dim, hidden_dim=disc_hidden_dim)
        self.gen_loss = torch.nn.CrossEntropyLoss()
        self.disc_loss = torch.nn.BCEWithLogitsLoss()
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
        gen_loss = self.gen_loss(generated_features, batch.x[masked_idx])  # calculate loss
        genoptim.zero_grad()
        self.manual_backward(gen_loss)
        genoptim.step()

        corrupt_batch = corrupt_molecules(batch, generated_features.argmax(axis=1), masked_idx)
        disc_prediction = self.discriminator(corrupt_batch)
        corruption_labels = torch.zeros((batch.num_nodes, 1), dtype=torch.float32, device=self.device)
        corruption_labels[masked_idx] = 1.
        disc_loss = self.disc_loss(corruption_labels, disc_prediction)
        discoptim.zero_grad()
        self.manual_backward(disc_loss)
        discoptim.step()
        self.log_dict({'gen_loss': gen_loss, 'disc_loss': disc_loss, 'loss': gen_loss+disc_loss})

    def configure_optimizers(self):
        genoptim = Adam(self.generator.parameters(), lr=self.hparams.gen_lr)
        discoptim = Adam(self.discriminator.parameters(), lr=self.hparams.disc_lr)
        return genoptim, discoptim

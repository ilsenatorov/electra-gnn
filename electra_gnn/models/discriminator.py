from pytorch_lightning import LightningModule

class Discriminator(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

    
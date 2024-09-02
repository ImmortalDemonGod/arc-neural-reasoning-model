import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch import nn
from transformers import AdamW

class ARCTrainer(pl.LightningModule):
    def __init__(self, model, train_dataset, val_dataset, batch_size=32, lr=1e-4):
        super(ARCTrainer, self).__init__()
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.lr = lr

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask)
        loss = self.compute_loss(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask)
        loss = self.compute_loss(outputs, labels)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def compute_loss(self, outputs, labels):
        # Implement the loss computation
        return nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), labels.view(-1))

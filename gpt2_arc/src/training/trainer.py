# gpt2_arc/src/training/trainer.py
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

class ARCTrainer(pl.LightningModule):
    def __init__(self, model, train_dataset, val_dataset, batch_size=32, lr=1e-4):
        super(ARCTrainer, self).__init__()
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.lr = lr
        self.logged_metrics = {}  # Initialize logged_metrics to store metrics

    def training_step(self, batch, batch_idx):
        if isinstance(batch, tuple):
            input_ids, attention_mask, labels = batch
        elif isinstance(batch, dict):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
        else:
            raise ValueError("Batch must be either a tuple or a dictionary")
        
        # Ensure tensors
        input_ids = input_ids.long() if not isinstance(input_ids, torch.LongTensor) else input_ids
        attention_mask = attention_mask.float() if not isinstance(attention_mask, torch.FloatTensor) else attention_mask
        labels = labels.long() if not isinstance(labels, torch.LongTensor) else labels
        outputs = self(input_ids, attention_mask)
        loss = self.compute_loss(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask)
        loss = self.compute_loss(outputs, labels)
        self.log("val_loss", loss)
        self.logged_metrics['val_loss'] = loss.item()  # Manually add to logged_metrics

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def compute_loss(self, outputs, labels):
        return nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), labels.view(-1))

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask)


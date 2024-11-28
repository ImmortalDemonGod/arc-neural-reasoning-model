import logging
from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)

class BestEpochTrackerCallback(Callback):
    def __init__(self):
        super().__init__()
        self.best_epoch = 0

    def on_validation_end(self, trainer, pl_module):
        current_val_loss = trainer.callback_metrics.get("val_loss")
        if current_val_loss is not None:
            if not hasattr(self, 'best_val_loss') or current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                self.best_epoch = trainer.current_epoch
                logger.debug(f"New best_val_loss: {self.best_val_loss} at epoch {self.best_epoch}")
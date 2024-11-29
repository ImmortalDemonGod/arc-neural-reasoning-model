import optuna
import pytorch_lightning as pl
import logging
class CustomPruningCallback(pl.Callback):
    def __init__(self, trial: optuna.trial.Trial, monitor: str = "val_loss") -> None:
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self.logger = logging.getLogger(__name__)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch
        current_score = trainer.callback_metrics.get(self.monitor)
        
        self.logger.debug(f"Validation end - Epoch: {epoch}")
        self.logger.debug(f"Current score ({self.monitor}): {current_score}")
        self.logger.debug(f"All metrics: {trainer.callback_metrics}")
        
        if current_score is None:
            self.logger.warning(f"No value found for {self.monitor} metric")
            return
            
        self.trial.report(current_score, step=epoch)
        if self.trial.should_prune():
            self.logger.info(f"Trial {self.trial.number} pruned at epoch {epoch} with score {current_score}")
            raise optuna.TrialPruned()
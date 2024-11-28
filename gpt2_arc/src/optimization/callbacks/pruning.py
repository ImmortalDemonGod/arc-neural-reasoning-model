import optuna
import pytorch_lightning as pl

class CustomPruningCallback(pl.Callback):
    def __init__(self, trial: optuna.trial.Trial, monitor: str = "val_loss") -> None:
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return
        self.trial.report(current_score, step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()
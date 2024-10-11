from pytorch_lightning.callbacks import Callback

class ModelConfigSaver(Callback):
    def __init__(self, config):
        """
        Initialize the ModelConfigSaver callback with the current configuration.

        Args:
            config (Config): The configuration object containing model parameters.
        """
        super().__init__()
        self.config = config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """
        Override the checkpoint saving to include the model configuration.

        Args:
            trainer (pl.Trainer): The Trainer instance.
            pl_module (pl.LightningModule): The LightningModule being trained.
            checkpoint (dict): The checkpoint dictionary to be modified.
        """
        checkpoint['model_config'] = self.config.model.__dict__

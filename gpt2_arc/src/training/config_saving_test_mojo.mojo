struct ModelConfigSaver(Callback):
    fn __init__(self, config: Config) -> None:
        """
        Initialize the ModelConfigSaver callback with the current configuration.

        Args:
            config (Config): The configuration object containing model parameters.
        """
        super().__init__()
        self.config = config

    fn on_save_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict) -> None:
        """
        Override the checkpoint saving to include the model configuration.

        Args:
            trainer (pl.Trainer): The Trainer instance.
            pl_module (pl.LightningModule): The LightningModule being trained.
            checkpoint (dict): The checkpoint dictionary to be modified.
        """
        checkpoint['model_config'] = self.config.model.__dict__


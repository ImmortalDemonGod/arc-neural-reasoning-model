
from pytorch_lightning.callbacks import Callback
from typing import Optional, Dict
import torch.nn as nn
import logging
from .grokfast import gradfilter_ema, gradfilter_ma

logger = logging.getLogger(__name__)


class GrokfastCallback(Callback):
    def __init__(
        self,
        filter_type: str = 'ema',  # 'ema' or 'ma'
        alpha: float = 0.98,
        lamb: float = 2.0,
        window_size: int = 100,
        warmup: bool = True,
        trigger: bool = False,  # For ablation study.
    ):
        """
        Initializes the Grokfast callback.

        Args:
            filter_type (str): Type of Grokfast filter ('ema' or 'ma').
            alpha (float): Momentum parameter for EMA.
            lamb (float): Amplifying factor.
            window_size (int): Window size for MA.
            warmup (bool): Whether to use warmup for MA.
            trigger (bool): For ablation studies.
        """
        super().__init__()
        self.filter_type = filter_type
        self.alpha = alpha
        self.lamb = lamb
        self.window_size = window_size
        self.warmup = warmup
        self.trigger = trigger
        self.grads = None  # Will hold the state across batches

    def on_after_backward(self, trainer, pl_module):
        """
        Called after the backward pass and before the optimizer step.

        Args:
            trainer: The trainer instance.
            pl_module: The LightningModule instance.
        """
        model = pl_module.model  # Adjust if your model is nested differently

        if self.filter_type == 'ema':
            self.grads = gradfilter_ema(
                m=model,  # Pass the actual model
                grads=self.grads,
                alpha=self.alpha,
                lamb=self.lamb
            )
            logger.debug("Applied Grokfast-EMA filter.")
        elif self.filter_type == 'ma':
            self.grads = gradfilter_ma(
                m=model,  # Pass the actual model
                grads=self.grads,
                window_size=self.window_size,
                lamb=self.lamb,
                filter_type='mean',  # or 'sum' based on preference
                warmup=self.warmup,
                trigger=self.trigger
            )
            logger.debug("Applied Grokfast-MA filter.")
        else:
            logger.warning(f"Unknown Grokfast filter type: {self.filter_type}. Skipping gradient filtering.")

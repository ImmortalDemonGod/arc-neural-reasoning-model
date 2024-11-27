# gpt2_arc/src/training/training_config_manager.py
import argparse
import logging
from typing import Dict, Optional, Any, Tuple, Union, List
from pytorch_lightning.callbacks import Callback
import torch
import optuna
from dataclasses import asdict
import numpy as np
from lightning.pytorch.profilers import PyTorchProfiler
from pytorch_lightning.callbacks import ModelCheckpoint
from gpt2_arc.src.utils import GrokfastCallback

from torch.profiler import ProfilerActivity
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

from gpt2_arc.src.config import Config, ModelConfig, TrainingConfig

logger = logging.getLogger(__name__)

class ModelConfigSaver(Callback):
    def __init__(self, config: Config) -> None:
        """
        Initialize the ModelConfigSaver callback with the current configuration.

        Args:
            config (Config): The configuration object containing model parameters.
        """
        super().__init__()
        self.config = config

    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict) -> None:
        """
        Override the checkpoint saving to include the model configuration.

        Args:
            trainer (pl.Trainer): The Trainer instance.
            pl_module (pl.LightningModule): The LightningModule being trained.
            checkpoint (dict): The checkpoint dictionary to be modified.
        """
        checkpoint['model_config'] = self.config.model.__dict__

        
class ConfigurationManager:
    """Manages all configuration aspects of the training process."""
    
    def __init__(self, args: argparse.Namespace):
        """
        Initialize the configuration manager.
        
        Args:
            args: Command line arguments
        """
        self.args = args
        self.validate_initial_args()
    
    def validate_initial_args(self) -> None:
        """
        Validate initial command line arguments.
        
        Raises:
            ValueError: If any validation check fails
        """
        # Validate synthetic data configuration
        if self.args.use_synthetic_data and not self.args.synthetic_data_path:
            raise ValueError("--synthetic_data_path must be provided when using synthetic data.")

        # Validate data splits
        total_split = self.args.train_split + self.args.val_split + self.args.test_split
        if not abs(total_split - 1.0) < 1e-6:
            raise ValueError("Train, validation, and test splits must sum to 1.0")
        
        # Validate mamba configuration
        mamba_ratio_min = 0.0  # Define constant at class level
        if self.args.mamba_ratio < mamba_ratio_min:
            raise ValueError(f"mamba_ratio must be >= {mamba_ratio_min}")
            
        # Validate training parameters
        if self.args.val_check_interval <= 0:
            raise ValueError("The --val_check_interval must be a positive number.")
            
        # Additional validations
        if self.args.max_epochs <= 0:
            raise ValueError("max_epochs must be greater than 0")
            
        if self.args.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
            
        if self.args.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")
            
        if self.args.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
            
        if self.args.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be greater than 0")

    def _create_model_config(self, params: Optional[Dict[str, Any]] = None) -> ModelConfig:
        """Create ModelConfig from args or Optuna parameters."""
        if params:  # Using Optuna parameters
            n_head = 2 ** params['n_head_exp']
            n_embd = n_head * params['n_embd_multiplier']
            n_embd = 2 ** int(np.log2(n_embd))
            
            return ModelConfig(
                n_embd=n_embd,
                n_head=n_head,
                n_layer=params['n_layer'],
                dropout=params['dropout'],
                num_workers=self.args.num_workers,
                prefetch_factor=self.args.prefetch_factor,
                persistent_workers=not self.args.no_persistent_workers,
                pin_memory=not self.args.no_pin_memory,
            )
        else:  # Using command line arguments
            return ModelConfig(
                n_embd=self.args.n_embd,
                n_head=self.args.n_head,
                n_layer=self.args.n_layer,
                dropout=self.args.dropout,
                mamba_ratio=self.args.mamba_ratio,
                d_state=self.args.d_state,
                d_conv=self.args.d_conv,
                mamba_depth=self.args.mamba_depth,
                mamba_expand=self.args.mamba_expand,
            )

    def _create_training_config(self, params: Optional[Dict[str, Any]] = None) -> TrainingConfig:
        """Create TrainingConfig from args or Optuna parameters."""
        base_config = {
            'max_epochs': self.args.max_epochs,
            'use_gpu': self.args.use_gpu,
            'log_level': self.args.log_level,
            'use_synthetic_data': self.args.use_synthetic_data,
            'synthetic_data_path': self.args.synthetic_data_path,
            'use_grokfast': self.args.use_grokfast,
            'grokfast_type': self.args.grokfast_type,
            'grokfast_alpha': self.args.grokfast_alpha,
            'grokfast_lamb': self.args.grokfast_lamb,
            'grokfast_window_size': self.args.grokfast_window_size,
            'include_pad_in_loss': self.args.include_pad_in_loss,
            'include_pad_in_accuracy': self.args.include_pad_in_accuracy,
            'num_workers': self.args.num_workers,
            'prefetch_factor': self.args.prefetch_factor,
            'persistent_workers': not self.args.no_persistent_workers,
            'pin_memory': self.args.pin_memory,
            'symbol_freq': {},  # Initialize as empty dict instead of None
            'balance_symbols': False,  # Default to False if no frequencies
            'balancing_method': "none",  # Default to no balancing
        }
        
        if params:  # Using Optuna parameters
            base_config.update({
                'batch_size': params['batch_size'],
                'learning_rate': params['learning_rate'],
            })
        else:  # Using command line arguments
            base_config.update({
                'batch_size': self.args.batch_size,
                'learning_rate': self.args.learning_rate,
            })
            
        return TrainingConfig(**base_config)

    def create_initial_config(self) -> Config:
        """Create initial configuration from command line arguments."""
        torch.set_float32_matmul_precision(self.args.matmul_precision)
        logger.info(f"Set float32 matmul precision to: {self.args.matmul_precision}")
        
        if self.args.use_optuna:
            params = self._load_optuna_parameters()
            model_config = self._create_model_config(params)
            training_config = self._create_training_config(params)
        else:
            model_config = self._create_model_config()
            training_config = self._create_training_config()
            
        return Config(model=model_config, training=training_config)

    def _load_optuna_parameters(self) -> Dict[str, Any]:
        """Load best parameters from Optuna study."""
        logger.info("Loading best hyperparameters from Optuna study")
        study_name = self._get_study_name()
        
        try:
            study = optuna.load_study(study_name=study_name, storage=self.args.optuna_storage)
        except KeyError:
            study = optuna.create_study(study_name=study_name, storage=self.args.optuna_storage)
            
        logger.debug(f"Loaded best parameters: {study.best_params}")
        return study.best_params

    def _get_study_name(self) -> str:
        """Get the Optuna study name, handling the case of automatic selection."""
        if self.args.optuna_study_name:
            return self.args.optuna_study_name
            
        study_summaries = optuna.get_all_study_summaries(storage=self.args.optuna_storage)
        study_names = [summary.study_name for summary in study_summaries]
        
        if len(study_names) == 1:
            study_name = study_names[0]
            logger.info(f"Automatically selected the only available study: {study_name}")
            return study_name
        elif len(study_names) == 0:
            raise ValueError("No studies found in the specified Optuna storage.")
        else:
            raise ValueError("Multiple studies found. Please specify --optuna-study-name.")

    def update_config_with_symbol_freq(self, config: Config, symbol_freq_dict: Dict[int, float]) -> Config:
        """Update configuration with symbol frequencies."""
        training_dict = asdict(config.training)
        training_dict.update({
            'symbol_freq': symbol_freq_dict,
            'balance_symbols': bool(symbol_freq_dict),
            'balancing_method': "weighting" if symbol_freq_dict else "none"
        })
        
        return Config(
            model=config.model,
            training=TrainingConfig(**training_dict)
        )

    def load_checkpoint_config(self, checkpoint_path: str) -> Tuple[Config, Dict]:
        """
        Load configuration and state dict from a checkpoint file.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Tuple[Config, Dict]: Configuration object and state dictionary
        """
        logger.info(f"Loading configuration from checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        if 'model_config' not in checkpoint or 'training_config' not in checkpoint:
            raise KeyError("Checkpoint missing 'model_config' or 'training_config'.")
            
        config = Config(
            model=ModelConfig(**checkpoint['model_config']),
            training=TrainingConfig(**checkpoint['training_config'])
        )
        
        return config, checkpoint['state_dict']

    def validate_configuration(self, config: Config) -> None:
        """Validate the complete configuration."""
        # Validate Mamba layer configuration
        mamba_layers = int(config.model.n_layer * config.model.mamba_ratio)
        transformer_layers = config.model.n_layer - mamba_layers
        
        if mamba_layers < 0 or transformer_layers < 0:
            raise ValueError("Invalid layer count: mamba_layers and transformer_layers must be non-negative.")
            
        if mamba_layers + transformer_layers != config.model.n_layer:
            raise ValueError("Inconsistency in layer count: verify that mamba_ratio is set correctly.")
            
        logger.info(f"Layer configuration - Mamba: {mamba_layers}, Transformer: {transformer_layers}")
    
    def setup_logging(self) -> None:
        """
        Configure logging level based on arguments.
        """
        log_level = getattr(logging, self.args.log_level.upper() if hasattr(self.args, 'log_level') else 'DEBUG', logging.DEBUG)
        logger.setLevel(log_level)
        logger.debug(f"Logging level set to: {log_level}")

    def get_accelerator_config(self) -> Dict[str, Any]:
        """
        Get accelerator configuration based on arguments.
        
        Returns:
            Dict containing accelerator, devices, and strategy settings
        """
        if self.args.accelerator == "tpu":
            return {
                'accelerator': 'tpu',
                'devices': 'xla:1',  # Use 'xla:8' for TPU v3-8 pods
                'strategy': 'tpu_spawn'  # Recommended strategy for TPU
            }
        elif self.args.accelerator == "gpu":
            if torch.cuda.is_available():
                return {
                    'accelerator': 'gpu',
                    'devices': 1,
                    'strategy': 'auto'
                }
            else:
                return {
                    'accelerator': 'cpu',
                    'devices': 1,
                    'strategy': 'auto'
                }
        else:
            return {
                'accelerator': 'cpu',
                'devices': 1,
                'strategy': 'auto'
            }

    def get_profiler_config(self) -> Optional[PyTorchProfiler]:
        """
        Get profiler configuration based on arguments.
        
        Returns:
            Optional[PyTorchProfiler]: Configured profiler or None if profiling is disabled
        """
        if not self.args.use_profiler:
            return None
            
        return PyTorchProfiler(
            dirpath=self.args.profiler_dirpath,
            filename=self.args.profiler_filename,
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True
        )

    def get_tensorboard_logger(self, experiment_id: str) -> Union[TensorBoardLogger, bool]:
        """
        Get TensorBoard logger configuration.
        
        Args:
            experiment_id: Current experiment identifier
            
        Returns:
            Union[TensorBoardLogger, bool]: Configured logger or False if logging is disabled
        """
        if self.args.no_logging:
            logger.debug("Logging is disabled")
            return False
            
        tb_logger = TensorBoardLogger(
            save_dir="runs",
            name=f"experiment_{experiment_id}"
        )
        logger.debug(f"TensorBoard logger initialized. Log dir: {tb_logger.log_dir}")
        return tb_logger

    def get_callbacks(self, config: Config, model_checkpoint: Optional[str] = None) -> List[Callback]:
        """
        Get all training callbacks based on configuration.
        
        Args:
            config: Current configuration
            model_checkpoint: Optional path to model checkpoint
            
        Returns:
            List[Callback]: List of configured callbacks
        """
        callbacks = []
        
        # GrokFast callback
        if config.training.use_grokfast:
            grokfast_callback = GrokfastCallback(
                filter_type=config.training.grokfast_type,
                alpha=config.training.grokfast_alpha,
                lamb=config.training.grokfast_lamb,
                window_size=config.training.grokfast_window_size if config.training.grokfast_type == 'ma' else 100,
                warmup=True,
                trigger=False
            )
            callbacks.append(grokfast_callback)
            logger.info("GrokfastCallback added to the training callbacks.")
        
        # Checkpointing callbacks
        if not self.args.no_checkpointing:
            checkpoint_filename = f"{'resume-' if model_checkpoint else ''}checkpoint-step_{{step}}-val_loss_{{val_loss:.4f}}"
            checkpoint_callback = ModelCheckpoint(
                dirpath="checkpoints",
                filename=checkpoint_filename,
                save_top_k=3,
                monitor="val_loss",
                mode="min",
            )
            callbacks.append(checkpoint_callback)
            
            # ModelConfigSaver callback
            model_config_saver = ModelConfigSaver(config)
            callbacks.append(model_config_saver)
            logger.info("ModelConfigSaver callback added to the training callbacks.")
        
        return callbacks
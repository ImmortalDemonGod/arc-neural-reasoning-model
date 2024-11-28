# gpt2_arc/src/optimization/utils/model_utils.py
import logging
import typing
from typing import Optional
import torch
from gpt2_arc.src.models.gpt2 import GPT2ARC
from gpt2_arc.src.config import ModelConfig, TrainingConfig
import pytorch_lightning
from pytorch_lightning.utilities.model_summary import ModelSummary

logger = logging.getLogger(__name__)

def create_model(args, config, symbol_freq_dict, hyperparameters):
    """
    Create and initialize a GPT2ARC model with the given configuration and hyperparameters.
    
    Args:
        args: Command line arguments
        config: Model configuration object
        symbol_freq_dict: Dictionary of symbol frequencies
        hyperparameters: Dictionary of model hyperparameters
        
    Returns:
        GPT2ARC: Initialized model
    """
    # Update config with hyperparameters
    model_config = ModelConfig(
        n_embd=hyperparameters['n_embd'],
        n_head=hyperparameters['n_head'],
        n_layer=hyperparameters['n_layer'],
        dropout=hyperparameters['dropout'],
        mamba_ratio=hyperparameters['mamba_ratio'],
        d_state=hyperparameters['d_state'],
        d_conv=hyperparameters['d_conv'],
        mamba_depth=hyperparameters['mamba_depth'],
        mamba_expand=hyperparameters['mamba_expand']
    )
    
    training_config = TrainingConfig(
        batch_size=hyperparameters['batch_size'],
        learning_rate=hyperparameters['learning_rate'],
        max_epochs=hyperparameters['max_epochs'],
        use_grokfast=hyperparameters['use_grokfast'],
        grokfast_type=hyperparameters['grokfast_type'],
        grokfast_alpha=hyperparameters['grokfast_alpha'],
        grokfast_lamb=hyperparameters['grokfast_lamb'],
        grokfast_window_size=hyperparameters['grokfast_window_size'],
        include_pad_in_loss=args.include_pad_in_loss,
        symbol_freq=symbol_freq_dict,
        balance_symbols=bool(symbol_freq_dict),
        balancing_method="weighting" if symbol_freq_dict else "none",
        num_workers=args.num_workers if hasattr(args, 'num_workers') else None,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=not args.no_persistent_workers,
        pin_memory=not args.no_pin_memory
    )
    
    config.model = model_config
    config.training = training_config
    
    num_classes = config.training.num_classes
    
    if args.model_checkpoint:
        model = load_model_from_checkpoint(args, config, symbol_freq_dict)
    else:
        model = GPT2ARC(config=config, num_classes=num_classes, symbol_freq=symbol_freq_dict)
        
    return model

def load_model_from_checkpoint(args, config, symbol_freq_dict):
    """
    Load a model from a checkpoint file.
    
    Args:
        args: Command line arguments containing checkpoint path
        config: Model configuration
        symbol_freq_dict: Dictionary of symbol frequencies
        
    Returns:
        GPT2ARC: Loaded model
    """
    logger.info(f"Loading model checkpoint: {args.model_checkpoint}")
    try:
        checkpoint = torch.load(args.model_checkpoint, map_location="cpu")
        
        if 'model_config' in checkpoint:
            model_config_dict = checkpoint['model_config']
            config.model = ModelConfig(**model_config_dict)
            logger.debug("Extracted model configuration from checkpoint.")
        else:
            logger.error("No model configuration found in checkpoint.")
            raise ValueError("No model configuration found in checkpoint.")
        
        # Initialize model with extracted configuration
        model = GPT2ARC(
            config=config,
            num_classes=config.model.num_classes,
            symbol_freq=symbol_freq_dict,
            pad_symbol_idx=config.training.pad_symbol_idx
        )
        
        # Load state dict
        state_dict = checkpoint.get('state_dict', checkpoint)
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys when loading state_dict: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading state_dict: {unexpected_keys}")
            
        return model
    
    except Exception as e:
        logger.error(f"Error loading model from checkpoint: {str(e)}")
        raise

def generate_model_summary(model, trial):
    """
    Generate and save a model summary to trial attributes.
    
    Args:
        model: The model to summarize
        trial: Optuna trial object to store the summary
    """
    logger.debug("Generating model summary.")
    try:
        model_summary = str(ModelSummary(model, max_depth=-1))
        trial.set_user_attr("model_summary", model_summary)
        logger.debug("Model summary generated and saved to trial user attributes.")
    except Exception as e:
        logger.error(f"Error generating model summary: {e}")
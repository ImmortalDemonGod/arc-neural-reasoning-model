import logging

logger = logging.getLogger(__name__)

def suggest_hyperparameters(trial, args, config):
    hyperparameters = {}
    if not args.model_checkpoint:
        hyperparameters = suggest_model_hyperparameters(trial, args)
    else:
        hyperparameters = load_hyperparameters_from_checkpoint(args, config)
    hyperparameters.update(suggest_training_hyperparameters(trial, args))
    return hyperparameters

def suggest_model_hyperparameters(trial, args):
    hyperparameters = {}
    n_head_exp = trial.suggest_int("n_head_exp", args.n_head_exp_min, args.n_head_exp_max)
    n_head = 2 ** n_head_exp
    hyperparameters['n_head'] = n_head

    n_embd_multiplier = trial.suggest_int(
        "n_embd_multiplier", args.n_embd_multiplier_min, args.n_embd_multiplier_max
    )
    n_embd = n_head * n_embd_multiplier
    hyperparameters['n_embd'] = n_embd

    n_layer = trial.suggest_int("n_layer", args.n_layer_min, args.n_layer_max)
    hyperparameters['n_layer'] = n_layer

    # Mamba-specific hyperparameters
    hyperparameters['mamba_ratio'] = trial.suggest_float(
        "mamba_ratio", args.mamba_ratio_min, args.mamba_ratio_max, step=args.mamba_ratio_step
    )
    hyperparameters['d_state'] = trial.suggest_int("d_state", args.d_state_min, args.d_state_max)
    hyperparameters['d_conv'] = trial.suggest_int("d_conv_min", args.d_conv_min, args.d_conv_max)
    hyperparameters['dropout'] = trial.suggest_float(
        "dropout", args.dropout_min, args.dropout_max, step=args.dropout_step
    )
    hyperparameters['mamba_depth'] = trial.suggest_int("mamba_depth", args.mamba_depth_min, args.mamba_depth_max)
    hyperparameters['mamba_expand'] = trial.suggest_int("mamba_expand", args.mamba_expand_min, args.mamba_expand_max)

    logger.debug(f"Suggested hyperparameters: {hyperparameters}")
    return hyperparameters

def load_hyperparameters_from_checkpoint(args, config):
    # Load hyperparameters from the model checkpoint
    # Placeholder for actual implementation
    hyperparameters = {
        'n_head': config.model.n_head,
        'n_embd': config.model.n_embd,
        'n_layer': config.model.n_layer,
        'mamba_ratio': config.model.mamba_ratio,
        'd_state': config.model.d_state,
        'd_conv': config.model.d_conv,
        'dropout': config.model.dropout,
        'mamba_depth': config.model.mamba_depth,
        'mamba_expand': config.model.mamba_expand
    }
    logger.debug("Loaded hyperparameters from checkpoint.")
    return hyperparameters

def suggest_training_hyperparameters(trial, args):
    hyperparameters = {}
    hyperparameters['batch_size'] = trial.suggest_int("batch_size", args.batch_size_min, args.batch_size_max)
    hyperparameters['learning_rate'] = trial.suggest_float(
        "learning_rate", args.learning_rate_min, args.learning_rate_max, log=True
    )
    hyperparameters['max_epochs'] = trial.suggest_int("max_epochs", args.max_epochs_min, args.max_epochs_max)
    # Grokfast hyperparameters
    hyperparameters['use_grokfast'] = args.use_grokfast
    if args.use_grokfast:
        hyperparameters['grokfast_type'] = trial.suggest_categorical("grokfast_type", args.grokfast_type_choices)
        hyperparameters['grokfast_alpha'] = trial.suggest_float(
            "grokfast_alpha", args.grokfast_alpha_min, args.grokfast_alpha_max
        )
        hyperparameters['grokfast_lamb'] = trial.suggest_float(
            "grokfast_lamb", args.grokfast_lamb_min, args.grokfast_lamb_max
        )
        if hyperparameters['grokfast_type'] == "ma":
            hyperparameters['grokfast_window_size'] = trial.suggest_int(
                "grokfast_window_size", args.grokfast_window_size_min, args.grokfast_window_size_max
            )
    else:
        hyperparameters['grokfast_type'] = None
        hyperparameters['grokfast_alpha'] = None
        hyperparameters['grokfast_lamb'] = None
        hyperparameters['grokfast_window_size'] = None
    logger.debug(f"Suggested training hyperparameters: {hyperparameters}")
    return hyperparameters

def validate_hyperparameters(
    n_embd: int,
    n_head: int,
    n_layer: int,
    mamba_ratio: float,
    d_state: int,
    d_conv: int,
    dropout: float
) -> bool:
    """Validate that hyperparameters meet necessary constraints."""
    logger.debug(f"Validating hyperparameters: n_embd={n_embd}, n_head={n_head}, n_layer={n_layer}, "
                 f"mamba_ratio={mamba_ratio}, d_state={d_state}, d_conv={d_conv}, dropout={dropout}")
    assert n_embd % n_head == 0, f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
    assert n_embd >= n_head, f"n_embd ({n_embd}) must be greater than or equal to n_head ({n_head})"
    assert n_layer > 0, f"n_layer ({n_layer}) must be positive"
    assert d_state > 0, f"d_state ({d_state}) must be positive"
    assert d_conv > 0, f"d_conv ({d_conv}) must be positive"
    assert 0.0 <= dropout <= 1.0, f"dropout ({dropout}) must be between 0.0 and 1.0"
    logger.debug("Hyperparameters validated successfully")
    return True

# gpt2_arc/src/utils/logging_config.py
import logging
import sys

def configure_logging(log_level: str = "INFO") -> None:
    """Configure logging for all modules with consistent formatting."""
    # Remove any existing handlers from root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set numeric level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create and configure handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger.addHandler(console_handler)
    root_logger.setLevel(numeric_level)

    # List of all module loggers we want to explicitly configure
    module_loggers = [
        'gpt2_arc.src.optimization.trial_manager',
        'gpt2_arc.src.optimization.optimizer',
        'gpt2_arc.src.training.trainer',
        'gpt2_arc.src.utils.experiment_tracker',
        'gpt2_arc.src.data.arc_dataset',
        '__main__'
    ]

    # Configure each module logger
    for module in module_loggers:
        logger = logging.getLogger(module)
        logger.setLevel(numeric_level)
        logger.propagate = True  # Ensure messages propagate to root logger
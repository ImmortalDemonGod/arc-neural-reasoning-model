# gpt2_arc/src/training/training_manager.py
import logging
import os
from typing import Dict, Optional, Tuple, Any
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from gpt2_arc.src.models.gpt2 import GPT2ARC
from gpt2_arc.src.config import Config
from gpt2_arc.src.training.trainer import ARCTrainer
from gpt2_arc.src.data.arc_dataset import ARCDataset
from gpt2_arc.src.utils.results_collector import ResultsCollector
from gpt2_arc.src.utils.experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)

class TrainingManager:
    """Manages the training process including model initialization, training, testing, and results handling."""
    
    def __init__(self, config: Config, args: Any):
        """
        Initialize the training manager.
        
        Args:
            config: Configuration object containing model and training settings
            args: Command line arguments
        """
        self.config = config
        self.args = args
        self.trainer = None
        self.pl_trainer = None
        # Initialize results collector and experiment tracker in __init__ instead of setup_training
        self.results_collector = ResultsCollector(self.config)
        self.experiment_tracker = ExperimentTracker(self.config, project=self.args.project)
        
    def setup_training(self, model: GPT2ARC, train_data: ARCDataset, 
                      val_data: ARCDataset, test_data: ARCDataset, 
                      pl_trainer_config: Dict) -> None:
        """
        Set up training components including trainer and callbacks.
        
        Args:
            model: Initialized GPT2ARC model
            train_data: Training dataset
            val_data: Validation dataset
            test_data: Test dataset
            pl_trainer_config: PyTorch Lightning trainer configuration
        """
        # Initialize ARC trainer using the already initialized results_collector
        self.trainer = ARCTrainer(
            model=model,
            train_dataset=train_data,
            val_dataset=val_data,
            config=self.config,
            args=self.args,
            results_collector=self.results_collector,
            test_dataset=test_data
        )
        self.trainer.log_hyperparameters()
        
        # Initialize PyTorch Lightning trainer
        self.pl_trainer = pl.Trainer(**pl_trainer_config)
        
        if pl_trainer_config.get('logger'):
            log_path = pl_trainer_config['logger'].log_dir
            self.results_collector.set_tensorboard_log_path(log_path)
            logger.debug(f"TensorBoard log path set: {log_path}")

    def initialize_model(self) -> GPT2ARC:
        """Initialize the GPT2ARC model with configuration settings."""
        logger.info("Initializing model")
        model = GPT2ARC(
            config=self.config,
            num_classes=self.config.training.num_classes,
            pad_symbol_idx=self.config.training.pad_symbol_idx
        )
        logger.debug(f"Model initialized with config: {self.config}")
        return model
        
    def load_checkpoint(self, model: GPT2ARC) -> GPT2ARC:
        """Load model from checkpoint if specified."""
        if not self.args.model_checkpoint:
            return model
            
        logger.info(f"Loading model from checkpoint: {self.args.model_checkpoint}")
        checkpoint = torch.load(self.args.model_checkpoint)
        
        if 'model_config' not in checkpoint or 'state_dict' not in checkpoint:
            raise KeyError("Invalid checkpoint: missing model_config or state_dict")
            
        model.load_state_dict(checkpoint['state_dict'])
        logger.debug(f"Loaded model configuration with num_classes={self.config.training.num_classes}")
        return model
        
      
    def train_model(self) -> None:
        """Execute model training."""
        if not self.pl_trainer or not self.trainer:
            raise RuntimeError("Training components not initialized. Call setup_training first.")
            
        logger.info("Starting model training")
        self.pl_trainer.fit(self.trainer)
        
    def test_model(self, test_loader) -> Dict[str, float]:
        """
        Test the trained model and collect results.
        
        Args:
            test_loader: DataLoader for test dataset
            
        Returns:
            Dictionary containing test metrics
        """
        logger.info("Starting model evaluation")
        test_results = self.pl_trainer.test(model=self.trainer, dataloaders=test_loader)
        
        if not test_results:
            raise RuntimeError("No test results returned from trainer")
            
        metrics = self._process_test_results(test_results)
        self.results_collector.set_test_results(metrics)
        
        logger.info(f"Test results - Loss: {metrics['avg_test_loss']:.4f}, "
                   f"Accuracy: {metrics['avg_test_accuracy']:.4f}, "
                   f"Diff Accuracy: {metrics['avg_test_diff_accuracy']:.4f}")
                   
        return metrics
        
    def _process_test_results(self, test_results: list) -> Dict[str, float]:
        """Process raw test results into metrics dictionary."""
        metrics = {
            'avg_test_loss': sum(r['avg_test_loss'] for r in test_results) / len(test_results),
            'avg_test_accuracy': sum(r['avg_test_accuracy'] for r in test_results) / len(test_results),
            'avg_test_diff_accuracy': sum(r['avg_test_diff_accuracy'] for r in test_results) / len(test_results)
        }
        
        # Add task-specific results
        for result in test_results:
            for key, value in result.items():
                if key.endswith('_test_accuracy') or key.endswith('_test_diff_accuracy'):
                    metrics[key] = value
                    
        return metrics
        
    def save_model(self) -> str:
        """
        Save the final model with configuration.
        
        Returns:
            Path to saved model checkpoint
        """
        model_path = f"checkpoints/final_model_{self.results_collector.experiment_id}.pth"
        os.makedirs("checkpoints", exist_ok=True)
        
        checkpoint = {
            'state_dict': self.trainer.model.state_dict(),
            'model_config': self.config.model.__dict__,
            'training_config': self.config.training.__dict__,
            'pad_symbol_idx': self.config.training.pad_symbol_idx,
            'symbol_freq': self.config.training.symbol_freq
        }
        
        torch.save(checkpoint, model_path)
        self.results_collector.set_checkpoint_path(model_path)
        logger.debug(f"Model saved to: {model_path}")
        
        return model_path
        
    def save_results(self) -> str:
        """
        Save experiment results to JSON.
        
        Returns:
            Path to saved results file
        """
        os.makedirs("results", exist_ok=True)
        results_path = f"results/experiment_{self.results_collector.experiment_id}.json"
        self.results_collector.save_to_json(results_path)
        logger.debug(f"Results saved to: {results_path}")
        return results_path
        
    def cleanup(self) -> None:
        """Clean up resources and finish tracking."""
        if self.experiment_tracker:
            self.experiment_tracker.finish()
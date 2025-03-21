# gpt2_arc/tests/test_integration_experiment.py
import sys
import os

# Add the root directory of the project to the PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

import pytest
import torch
from gpt2_arc.src.data.arc_dataset import ARCDataset
from gpt2_arc.src.models.gpt2 import GPT2ARC
from gpt2_arc.src.training.trainer import ARCTrainer
from gpt2_arc.src.config import Config, ModelConfig, TrainingConfig
from gpt2_arc.src.utils.results_collector import ResultsCollector
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import arckit

@pytest.fixture
def setup_experiment():
    # Load a sample task from arckit
    task_id = "007bbfb7"  # Example task ID
    task_data = arckit.load_single(task_id)
    train_data = task_data.train
    val_data = task_data.test
    print(f"DEBUG: task_data type: {type(task_data)}")
    print(f"DEBUG: task_data attributes: {dir(task_data)}")
    print(f"DEBUG: train_data type: {type(train_data)}")
    print(f"DEBUG: train_data content: {train_data}")
    print(f"DEBUG: val_data type: {type(val_data)}")
    print(f"DEBUG: val_data content: {val_data}")
    train_dataset = ARCDataset([{"train": train_data, "test": val_data}])
    val_dataset = ARCDataset([{"train": train_data, "test": val_data}])

    # Model and config setup
    model_config = ModelConfig(n_embd=64, n_head=2, n_layer=1)
    training_config = TrainingConfig(batch_size=1, learning_rate=1e-4, max_epochs=1)
    config = Config(model=model_config, training=training_config)
    model = GPT2ARC(config=model_config)

    # Trainer setup
    trainer = ARCTrainer(model=model, train_dataset=train_dataset, val_dataset=val_dataset, config=config)
    return trainer, config

def test_full_experiment_run(setup_experiment):
    trainer, config = setup_experiment

    # PyTorch Lightning Trainer
    pl_trainer = Trainer(
        max_epochs=config.training.max_epochs,
        logger=TensorBoardLogger("tb_logs", name="arc_model_test"),
        callbacks=[ModelCheckpoint(dirpath="checkpoints", save_top_k=1, monitor="val_loss")],
        enable_checkpointing=True,
        enable_progress_bar=False,
        fast_dev_run=True
    )

    # Run training
    pl_trainer.fit(trainer)

    # Verify results
    results_summary = trainer.results_collector.get_summary()
    assert results_summary["experiment_id"] is not None
    assert "final_train_loss" in results_summary
    assert "final_val_loss" in results_summary

@pytest.mark.parametrize("invalid_data", [
    ({"input": [[0] * 30 for _ in range(30)]}),  # Missing output
    ({"output": [[0] * 30 for _ in range(30)]}),  # Missing input
])
def test_invalid_data_handling(invalid_data):
    with pytest.raises(ValueError):
        ARCDataset([invalid_data])

def test_model_convergence_issue(setup_experiment):
    trainer, config = setup_experiment
    trainer.config.training.learning_rate = 1e-10  # Set an inappropriate learning rate

    # PyTorch Lightning Trainer
    pl_trainer = Trainer(
        max_epochs=config.training.max_epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        fast_dev_run=True
    )

    # Run training
    pl_trainer.fit(trainer)

    # Verify that the model did not converge
    results_summary = trainer.results_collector.get_summary()
    assert results_summary["final_train_loss"] is not None
    assert results_summary["final_train_loss"] > 1.0  # Assuming a high loss indicates non-convergence

import logging
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

logger = logging.getLogger(__name__)

def train_and_evaluate(trainer, arc_trainer, test_data, config, trial):
    # Training
    arc_trainer.model.train()
    logger.debug("Starting training.")
    trainer.fit(arc_trainer)

    # Validation Loss
    best_val_loss = retrieve_best_val_loss(trainer, trial)

    # Testing
    test_loader = DataLoader(
        test_data,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    logger.info("Evaluating model on test dataset.")
    test_results = trainer.test(model=arc_trainer, dataloaders=test_loader)
    process_test_results(test_results, arc_trainer, best_val_loss, trainer)

    return best_val_loss

def retrieve_best_val_loss(trainer, trial):
    checkpoint_callback = next(cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint))
    if checkpoint_callback.best_model_score is not None:
        best_val_loss = checkpoint_callback.best_model_score.item()
        logger.info(f"Trial {trial.number}: Best validation loss: {best_val_loss}")
    else:
        logger.warning(f"Trial {trial.number}: No checkpoints were saved. Assigning a high validation loss.")
        best_val_loss = float('inf')
    return best_val_loss

def process_test_results(test_results, arc_trainer, best_val_loss, trainer):
    if test_results:
        avg_test_loss = sum(result['avg_test_loss'] for result in test_results) / len(test_results)
        avg_test_accuracy = sum(result['avg_test_accuracy'] for result in test_results) / len(test_results)
        avg_test_diff_accuracy = sum(result['avg_test_diff_accuracy'] for result in test_results) / len(test_results)

        logger.info(f"Test results - Loss: {avg_test_loss:.4f}, Accuracy: {avg_test_accuracy:.4f}, Diff Accuracy: {avg_test_diff_accuracy:.4f}")

        # Update final metrics with actual test results
        arc_trainer.results_collector.set_final_metrics({
            "best_val_loss": best_val_loss,
            "best_epoch": trainer.current_epoch,
            "final_test_loss": avg_test_loss,
            "final_test_accuracy": avg_test_accuracy,
            "final_test_diff_accuracy": avg_test_diff_accuracy
        })
import pytest
import torch
from src.models.gpt2 import GPT2ARC
from src.config import Config
from torch.utils.data import DataLoader
from src.utils.helpers import differential_pixel_accuracy
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture
def model():
    config = Config().model
    model = GPT2ARC(config)
    logger.debug(f"Model config: {config}")
    return model

@pytest.fixture
def inputs():
    inputs = torch.randn(2, 1, 32, 32)
    logger.debug(f"Input shape: {inputs.shape}, dtype: {inputs.dtype}")
    return inputs

@pytest.fixture
def targets():
    targets = torch.randint(0, 2, (2, 32, 32))
    logger.debug(f"Targets shape: {targets.shape}, dtype: {targets.dtype}")
    return targets

@pytest.fixture
def attention_mask():
    mask = torch.ones(2, 32*32)
    logger.debug(f"Attention mask shape: {mask.shape}, dtype: {mask.dtype}")
    return mask

@pytest.fixture
def dataloader(inputs, targets, attention_mask):
    dataset = list(zip(inputs, targets, attention_mask))
    loader = DataLoader(dataset, batch_size=1)
    logger.debug(f"Dataloader created with {len(loader)} batches")
    return loader

def test_no_grad_calculation(model, inputs, attention_mask):
    logger.debug("Starting test_no_grad_calculation")
    with torch.no_grad():
        outputs = model(inputs, attention_mask=attention_mask)
        logger.debug(f"Output shape: {outputs.shape}, requires_grad: {outputs.requires_grad}")
        assert not outputs.requires_grad, "Gradients should not be tracked in evaluation mode."

def test_data_loop_for_evaluation(model, dataloader):
    logger.debug("Starting test_data_loop_for_evaluation")
    model.eval()
    for batch_idx, (inputs, targets, attention_mask) in enumerate(dataloader):
        outputs = model(inputs, attention_mask=attention_mask)
        logger.debug(f"Batch {batch_idx}: Input shape: {inputs.shape}, Output shape: {outputs.shape}")
        assert outputs is not None, f"Model returned None for batch {batch_idx}"

def test_model_predictions(model, inputs, attention_mask):
    logger.debug("Starting test_model_predictions")
    outputs = model(inputs, attention_mask=attention_mask)
    logger.debug(f"Model output shape: {outputs.shape}, dtype: {outputs.dtype}")
    assert outputs.dim() == 3, f"Expected 3D output, got {outputs.dim()}D"
    assert outputs.size(0) == inputs.size(0), f"Batch size mismatch: {outputs.size(0)} vs {inputs.size(0)}"

def test_standard_pixel_accuracy(model, inputs, targets):
    logger.debug("Starting test_standard_pixel_accuracy")
    outputs = model(inputs)
    logger.debug(f"Outputs shape: {outputs.shape}, Targets shape: {targets.shape}")
    outputs = outputs.view(targets.shape[0], -1, targets.shape[1], targets.shape[2])
    predicted = outputs.argmax(dim=1)
    accuracy = (predicted == targets).float().mean().item()
    logger.debug(f"Calculated accuracy: {accuracy}")
    assert 0.0 <= accuracy <= 1.0, f"Accuracy should be between 0 and 1, got {accuracy}"

def test_differential_pixel_accuracy(model, inputs, targets):
    logger.debug("Starting test_differential_pixel_accuracy")
    outputs = model(inputs)
    logger.debug(f"Outputs shape: {outputs.shape}, Targets shape: {targets.shape}")
    outputs = outputs.view(targets.shape[0], -1, targets.shape[1], targets.shape[2])
    predicted = outputs.argmax(dim=1)
    diff_accuracy, _, _ = differential_pixel_accuracy(inputs, targets, predicted)
    logger.debug(f"Calculated differential accuracy: {diff_accuracy}")
    assert 0.0 <= diff_accuracy <= 1.0, f"Differential pixel accuracy should be between 0 and 1, got {diff_accuracy}"

def test_task_accuracies_tracking(model, dataloader, is_training=False):
    logger.debug("Starting test_task_accuracies_tracking")
    task_accuracies = {}
    model.eval()
    for batch_idx, (inputs, targets, attention_mask) in enumerate(dataloader):
        outputs = model(inputs, attention_mask=attention_mask)
        logger.debug(f"Batch {batch_idx}: Outputs shape: {outputs.shape}, Targets shape: {targets.shape}")
        outputs = outputs.view(targets.shape[0], -1, targets.shape[1], targets.shape[2])
        accuracy = (outputs.argmax(dim=1) == targets).float().mean().item()
        task_id = getattr(dataloader, 'task_id', 'default_task')
        if task_id not in task_accuracies:
            task_accuracies[task_id] = []
        task_accuracies[task_id].append(accuracy)
        logger.debug(f"Task accuracies after batch {batch_idx}: {task_accuracies}")
    assert task_accuracies, "Task accuracies dictionary should not be empty"
    assert 'default_task' in task_accuracies, "Default task should be logged in task accuracies"

def test_final_metric_calculation(model, dataloader, attention_mask):
    logger.debug("Starting test_final_metric_calculation")
    model.eval()
    total_loss, total_accuracy = 0, 0
    num_batches = 0
    for batch_idx, (inputs, targets, attention_mask) in enumerate(dataloader):
        outputs = model(inputs, attention_mask=attention_mask)
        logger.debug(f"Batch {batch_idx}: Outputs shape: {outputs.shape}, Targets shape: {targets.shape}")
        outputs = outputs.view(targets.shape[0], -1, targets.shape[1], targets.shape[2])
        loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(1)), targets.view(-1))
        total_loss += loss.item()
        accuracy = (outputs.argmax(dim=1) == targets).float().mean().item()
        total_accuracy += accuracy
        num_batches += 1
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    logger.debug(f"Final metrics - Average loss: {avg_loss}, Average accuracy: {avg_accuracy}")
    assert avg_loss >= 0, f"Average loss should be non-negative, got {avg_loss}"
    assert 0.0 <= avg_accuracy <= 1.0, f"Average accuracy should be between 0 and 1, got {avg_accuracy}"

@pytest.mark.skip(reason="Model does not have an 'evaluate' method.")
def test_return_of_evaluation_results(model, dataloader):
    logger.debug("Starting test_return_of_evaluation_results")
    if not hasattr(model, 'evaluate'):
        logger.warning("Model does not have an 'evaluate' method. Skipping test.")
        pytest.skip("Model does not have an 'evaluate' method.")
    results = model.evaluate(dataloader)
    logger.debug(f"Evaluation results: {results}")
    assert "loss" in results and "accuracy" in results, "Evaluation results should return loss and accuracy."
    assert isinstance(results["loss"], float), f"Loss should be a float, got {type(results['loss'])}"
    assert 0.0 <= results["accuracy"] <= 1.0, f"Accuracy should be between 0 and 1, got {results['accuracy']}"

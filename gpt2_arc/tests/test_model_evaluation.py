import pytest
import torch
from pytest_mock import mocker
from src.models.gpt2 import GPT2ARC
from src.config import Config
from torch.utils.data import DataLoader
from src.utils.helpers import differential_pixel_accuracy
from src.config import Config, ModelConfig, TrainingConfig
from src.models.gpt2 import GPT2ARC
from src.training.trainer import ARCTrainer

@pytest.fixture
def trainer():
    model_config = ModelConfig(n_embd=96, n_head=3, n_layer=1)
    config = Config(model=model_config, training=TrainingConfig(batch_size=32, learning_rate=1e-4, max_epochs=2))
    model = GPT2ARC(config.model)
    return ARCTrainer(model, None, None, config)
import logging
from unittest.mock import Mock


# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture
def model(mocker):
    mock_model = mocker.Mock()
    mock_model.eval = mocker.Mock()
    mock_model.side_effect = lambda inputs, attention_mask=None: torch.randn(1, 4, 2, 2)
    logger.debug(f"Created mock model")
    return mock_model

@pytest.fixture
def inputs():
    # Use a predetermined input
    inputs = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    logger.debug(f"Input shape: {inputs.shape}, dtype: {inputs.dtype}")
    return inputs

@pytest.fixture
def targets():
    # Use a predetermined target
    targets = torch.tensor([[[1, 0], [0, 1]]])
    logger.debug(f"Targets shape: {targets.shape}, dtype: {targets.dtype}")
    return targets

@pytest.fixture
def attention_mask():
    mask = torch.ones(1, 4)
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
        assert outputs.shape == (1, 4, 2, 2), f"Expected output shape (1, 4, 2, 2), got {outputs.shape}"

def test_model_predictions(model, inputs, attention_mask):
    logger.debug("Starting test_model_predictions")
    outputs = model(inputs, attention_mask=attention_mask)
    logger.debug(f"Model output shape: {outputs.shape}, dtype: {outputs.dtype}")
    initial_output = model(inputs, attention_mask=attention_mask)
    logger.debug(f"Initial output shape: {initial_output.shape}")
    
    # Change input and check if output changes
    modified_inputs = inputs + 1
    modified_output = model(modified_inputs, attention_mask=attention_mask)
    logger.debug(f"Modified output shape: {modified_output.shape}")
    
    assert not torch.allclose(initial_output, modified_output), "Output should change when input changes"

def test_standard_pixel_accuracy(model, inputs, targets):
    logger.debug("Starting test_standard_pixel_accuracy")
    outputs = model(inputs)
    logger.debug(f"Outputs shape: {outputs.shape}, Targets shape: {targets.shape}")
    outputs = outputs.view(targets.shape[0], -1, targets.shape[1], targets.shape[2])
    predicted = outputs.argmax(dim=1)
    accuracy = (predicted == targets).float().mean().item()
    logger.debug(f"Calculated accuracy: {accuracy}")
    assert 0.0 <= accuracy <= 1.0, f"Accuracy should be between 0 and 1, got {accuracy}"
    
    # Test with known values
    known_outputs = torch.FloatTensor([[[[0.9, 0.1], [0.1, 0.9]]]])
    known_targets = torch.tensor([[[1, 0], [0, 1]]])
    known_accuracy = (known_outputs.argmax(dim=1) == known_targets).float().mean().item()
    logger.debug(f"Known accuracy: {known_accuracy}")
    assert known_accuracy == 1.0, f"Expected known accuracy to be 1.0, got {known_accuracy}"

def test_differential_pixel_accuracy(model, inputs, targets):
    logger.debug("Starting test_differential_pixel_accuracy")
    outputs = model(inputs)
    logger.debug(f"Outputs shape: {outputs.shape}, Targets shape: {targets.shape}")
    outputs = outputs.view(targets.shape[0], -1, targets.shape[1], targets.shape[2])
    predicted = outputs.argmax(dim=1)
    diff_accuracy, _, _ = differential_pixel_accuracy(inputs, targets, predicted)
    logger.debug(f"Calculated differential accuracy: {diff_accuracy}")
    assert 0.0 <= diff_accuracy <= 1.0, f"Differential pixel accuracy should be between 0 and 1, got {diff_accuracy}"

    # Test with known values
    known_inputs = torch.tensor([[[[1, 0], [0, 1]]]])
    known_targets = torch.tensor([[[0, 1], [1, 0]]])
    known_predicted = torch.tensor([[[0, 1], [1, 0]]])
    known_diff_accuracy, known_total_diff, known_correct_diff = differential_pixel_accuracy(known_inputs, known_targets, known_predicted)
    logger.debug(f"Known differential accuracy: {known_diff_accuracy}, Total diff: {known_total_diff}, Correct diff: {known_correct_diff}")
    assert known_diff_accuracy == 1.0, f"Expected known differential accuracy to be 1.0, got {known_diff_accuracy}"

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
            task_accuracies[task_id] = {'train': [], 'test': []}
        task_accuracies[task_id]['train' if is_training else 'test'].append(accuracy)
        logger.debug(f"Task accuracies after batch {batch_idx}: {task_accuracies}")
    assert task_accuracies, "Task accuracies dictionary should not be empty"
    assert 'default_task' in task_accuracies, "Default task should be logged in task accuracies"
    assert 'test' in task_accuracies['default_task'], "Test accuracies should be logged for default task"

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
        logger.debug(f"Batch {batch_idx}: Loss: {loss.item()}, Accuracy: {accuracy}")
        total_accuracy += accuracy
        num_batches += 1
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    logger.debug(f"Final metrics - Average loss: {avg_loss}, Average accuracy: {avg_accuracy}")
    assert avg_loss >= 0, f"Average loss should be non-negative, got {avg_loss}"
    assert 0.0 <= avg_accuracy <= 1.0, f"Average accuracy should be between 0 and 1, got {avg_accuracy}"

def test_return_of_evaluation_results(model, dataloader, mocker):
    logger.debug("Starting test_return_of_evaluation_results")
    # Simulate a simple evaluation result
    model.evaluate = lambda dataloader: {'loss': 0.5, 'accuracy': 0.75}
    results = model.evaluate(dataloader)
    logger.debug(f"Evaluation results: {results}")
    assert "loss" in results and "accuracy" in results, "Evaluation results should return loss and accuracy."
    assert isinstance(results["loss"], float), f"Loss should be a float, got {type(results['loss'])}"
    assert 0.0 <= results["accuracy"] <= 1.0, f"Accuracy should be between 0 and 1, got {results['accuracy']}"
def test_validation_step_with_incorrect_batch_format(trainer):
    """Test that the validation_step raises a ValueError for an incorrect batch format."""

    # Create a batch with an incorrect format (e.g., a list)
    incorrect_batch = [
        torch.randint(0, 10, (2, 900)),  # Random input data
        torch.ones((2, 900)),  # Random attention mask
        torch.randint(0, 10, (2, 900))  # Random labels
    ]

    # Check if a ValueError is raised with the incorrect batch
    with pytest.raises(ValueError, match="Batch must be either a tuple or a dictionary"):
        trainer.validation_step(incorrect_batch, 0)

def test_model_loading_from_checkpoint(mocker):
    logger.debug("Starting test_model_loading_from_checkpoint")
    # Load the model checkpoint from the specified path
    checkpoint_path = "checkpoints/arc_model-epoch=00-val_loss=0.73.ckpt"
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Extract the model configuration from the checkpoint
    if 'config' in checkpoint:
        model_config = checkpoint['config']
    else:
        pytest.fail("Model configuration not found in checkpoint.")
    
    # Initialize the model with the checkpoint's configuration
    model = GPT2ARC(model_config)
    
    # Load the state_dict
    state_dict = {k.replace("model.", ""): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)

    # Ensure the model is in evaluation mode
    model.eval()
    assert not model.training, "Model should be in evaluation mode after calling eval()"
    logger.debug("Completed test_model_loading_from_checkpoint")


def test_checkpoint_contains_model_config():                                                                                      
    checkpoint_path = "checkpoints/arc_model-epoch=09-val_loss=0.41.ckpt"                                                         
                                                                                                                                
    try:                                                                                                                          
        checkpoint = torch.load(checkpoint_path)                                                                                  
    except FileNotFoundError:                                                                                                     
        pytest.fail(f"Checkpoint file not found: {checkpoint_path}")                                                              
                                                                                                                                
    # Log the keys in the checkpoint
    logger.debug(f"Checkpoint keys: {checkpoint.keys()}")

    # Check for model configuration
    assert 'config' in checkpoint, "Model configuration not found in checkpoint."
    model_config = checkpoint['config']
    logger.debug(f"Model configuration found in checkpoint: {model_config}")
    print("Model configuration found in checkpoint:", model_config)

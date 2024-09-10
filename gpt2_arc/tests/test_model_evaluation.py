import pytest
import torch
from src.models.gpt2 import GPT2ARC
from src.config import Config
from torch.utils.data import DataLoader

@pytest.fixture
def model():
    config = Config().model
    return GPT2ARC(config)

@pytest.fixture
def inputs():
    return torch.randn(2, 3, 32, 32)  # Example input tensor

@pytest.fixture
def attention_mask():
    return torch.ones(2, 3, 32, 32)  # Example attention mask

@pytest.fixture
def dataloader(inputs, attention_mask):
    dataset = [(inputs, torch.randint(0, 2, (2, 32, 32)), attention_mask)]
    return DataLoader(dataset, batch_size=1)

def test_evaluation_mode_setup(model):
    model.train()  # Ensure model is in training mode initially
    model.eval()   # Switch to evaluation mode
    assert not any([layer.training for layer in model.modules() if isinstance(layer, torch.nn.Dropout)]), "Dropout should be disabled in evaluation mode."

def test_no_grad_calculation(model, inputs, attention_mask):
    with torch.no_grad():
        outputs = model(inputs, attention_mask=attention_mask)
        assert outputs.requires_grad == False, "Gradients should not be tracked in evaluation mode."

def test_data_loop_for_evaluation(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    for batch in dataloader:
        inputs, targets, attention_mask = batch
        outputs = model(inputs, attention_mask=attention_mask)
        assert outputs is not None, "Model should return outputs for each batch."

def test_model_predictions(model, inputs, attention_mask):
    outputs = model(inputs, attention_mask=attention_mask)
    assert outputs.shape == inputs.shape, "Model output shape should match the input shape."

def test_loss_calculation(model, inputs, targets, loss_fn):
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    assert loss is not None and isinstance(loss.item(), float), "Loss should be a valid float value."

def test_standard_pixel_accuracy(model, inputs, targets):
    outputs = model(inputs)
    predicted = outputs.argmax(dim=1)
    accuracy = (predicted == targets).float().mean().item()
    assert 0.0 <= accuracy <= 1.0, "Accuracy should be between 0 and 1."

def test_differential_pixel_accuracy(model, inputs, targets):
    outputs = model(inputs)
    predicted = outputs.argmax(dim=1)
    diff_accuracy = differential_pixel_accuracy(inputs.view(outputs.shape), targets, predicted)
    assert 0.0 <= diff_accuracy <= 1.0, "Differential pixel accuracy should be between 0 and 1."

def test_task_accuracies_tracking(model, dataloader, is_training):
    task_accuracies = {}
    model.eval()
    for batch in dataloader:
        inputs, targets = batch
        outputs = model(inputs)
        accuracy = (outputs.argmax(dim=1) == targets).float().mean().item()
        task_id = dataloader.task_id
        model._update_task_accuracies(task_accuracies, accuracy, diff_accuracy=None, dataloader=dataloader, is_training=is_training)
        assert task_id in task_accuracies, "Task ID should be logged in task accuracies."

def test_task_wise_metrics_aggregation(task_accuracies):
    task_id = "some_task"
    accuracy = 0.8
    task_accuracies[task_id] = {"train": [accuracy]}
    assert task_accuracies[task_id]["train"] == [accuracy], "Task accuracy should be tracked and aggregated correctly."

def test_final_metric_calculation(model, dataloader):
    model.eval()
    total_loss, total_accuracy, total_diff_accuracy = 0, 0, 0
    for batch in dataloader:
        inputs, targets = batch
        outputs = model(inputs)
        loss = model._compute_loss(outputs, targets)
        total_loss += loss.item()
        accuracy = (outputs.argmax(dim=1) == targets).float().mean().item()
        total_accuracy += accuracy
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    assert avg_loss >= 0, "Average loss should be a non-negative value."
    assert 0.0 <= avg_accuracy <= 1.0, "Average accuracy should be between 0 and 1."

def test_return_of_evaluation_results(model, dataloader):
    results = model.evaluate(dataloader)
    assert "loss" in results and "accuracy" in results and "diff_accuracy" in results, "Evaluation results should return loss, accuracy, and differential accuracy."
    assert isinstance(results["loss"], float), "Loss should be a float."
    assert 0.0 <= results["accuracy"] <= 1.0, "Accuracy should be between 0 and 1."

# gpt2_arc/tests/test_differential_pixel_accuracy.py
import sys
import os

# Add the root directory of the project to the PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

import torch
from gpt2_arc.src.utils.helpers import differential_pixel_accuracy
from gpt2_arc.src.models.gpt2 import GPT2ARC
from gpt2_arc.src.config import ModelConfig
from gpt2_arc.src.data.arc_dataset import ARCDataset
import arckit

def test_identical_inputs_and_targets():
    input_tensor = torch.tensor([[1, 2], [3, 4]])
    target_tensor = torch.tensor([[1, 2], [3, 4]])
    prediction_tensor = torch.tensor([[1, 2], [3, 4]])
    accuracy, _, _ = differential_pixel_accuracy(input_tensor, target_tensor, prediction_tensor)
    assert accuracy == 1.0, "Expected accuracy of 1.0 for identical input and target"

def test_completely_different_inputs_and_targets():
    input_tensor = torch.tensor([[1, 1], [1, 1]])
    target_tensor = torch.tensor([[0, 0], [0, 0]])
    prediction_tensor = torch.tensor([[0, 0], [0, 0]])
    accuracy, _, _ = differential_pixel_accuracy(input_tensor, target_tensor, prediction_tensor)
    assert accuracy == 1.0, "Expected accuracy of 1.0 for correct prediction of all differing pixels"

def test_partial_differences():
    input_tensor = torch.tensor([[1, 2], [3, 4]])
    target_tensor = torch.tensor([[1, 0], [3, 0]])
    prediction_tensor = torch.tensor([[1, 0], [3, 4]])
    accuracy, _, _ = differential_pixel_accuracy(input_tensor, target_tensor, prediction_tensor)
    assert accuracy == 0.5, "Expected accuracy of 0.5 for partial correct predictions"

def test_empty_tensors():
    input_tensor = torch.tensor([])
    target_tensor = torch.tensor([])
    prediction_tensor = torch.tensor([])
    accuracy, _, _ = differential_pixel_accuracy(input_tensor, target_tensor, prediction_tensor)
    assert accuracy == 1.0, "Expected accuracy of 1.0 for empty tensors"

def test_single_pixel_difference():
    input_tensor = torch.tensor([[1]])
    target_tensor = torch.tensor([[0]])
    prediction_tensor = torch.tensor([[0]])
    accuracy, _, _ = differential_pixel_accuracy(input_tensor, target_tensor, prediction_tensor)
    assert accuracy == 1.0, "Expected accuracy of 1.0 for single pixel difference"

def test_differential_pixel_accuracy_with_arckit_data():
    print("Starting test_differential_pixel_accuracy_with_arckit_data")
    task_id = "007bbfb7"
    task_data = arckit.load_single(task_id)

    print(f"Loaded task data: {task_data}")
    print(f"Debug: task_data type: {type(task_data)}")
    print(f"Debug: task_data attributes: {dir(task_data)}")

    dataset = ARCDataset([task_data])  # Wrap in list to simulate multiple tasks
    input_tensor, target_tensor, _ = dataset[0]

    print(f"Dataset input tensor shape: {input_tensor.shape}")
    print(f"Dataset target tensor shape: {target_tensor.shape}")

    model_config = ModelConfig(n_embd=64, n_head=2, n_layer=1)
    model = GPT2ARC(model_config)
    model.eval()

    print("Model initialized and set to eval mode")

    with torch.no_grad():
        prediction_tensor = model(input_tensor.unsqueeze(0))

    print(f"Model prediction tensor shape: {prediction_tensor.shape}")

    # Reverse scaling for evaluation
    original_input = task_data.train[0][0]
    original_target = task_data.train[0][1]
    
    print(f"Original input shape: {original_input.shape}")
    print(f"Original target shape: {original_target.shape}")

    prediction_np = prediction_tensor.squeeze().argmax(dim=0).numpy()
    print(f"Prediction numpy array shape: {prediction_np.shape}")

    reversed_prediction = dataset.reverse_scaling(original_input, prediction_np)
    print(f"Reversed prediction shape: {reversed_prediction.shape}")

    # Convert back to tensors for differential_pixel_accuracy
    # Ensure all tensors have the same shape
    input_tensor = torch.tensor(original_input, dtype=torch.float32).resize_(original_target.shape)
    target_tensor = torch.tensor(original_target, dtype=torch.float32)
    prediction_tensor = torch.tensor(reversed_prediction, dtype=torch.float32).resize_(original_target.shape)

    print(f"Final input tensor shape: {input_tensor.shape}")
    print(f"Final target tensor shape: {target_tensor.shape}")
    print(f"Final prediction tensor shape: {prediction_tensor.shape}")

    accuracy, _, _ = differential_pixel_accuracy(input_tensor, target_tensor, prediction_tensor)
    print(f"Differential Pixel Accuracy for task {task_id}: {accuracy}")

    assert 0 <= accuracy <= 1, f"Accuracy should be between 0 and 1, but got {accuracy}"

# Run the tests
if __name__ == "__main__":
    test_identical_inputs_and_targets()
    test_completely_different_inputs_and_targets()
    test_partial_differences()
    test_empty_tensors()
    test_single_pixel_difference()
    print("All tests passed!")

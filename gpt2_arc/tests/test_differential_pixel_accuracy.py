import torch
from gpt2_arc.src.utils.helpers import differential_pixel_accuracy

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

# Run the tests
if __name__ == "__main__":
    test_identical_inputs_and_targets()
    test_completely_different_inputs_and_targets()
    test_partial_differences()
    test_empty_tensors()
    test_single_pixel_difference()
    print("All tests passed!")

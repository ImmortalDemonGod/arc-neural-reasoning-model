# gpt2_arc/src/utils/helpers.py
import torch

def differential_pixel_accuracy(input, target, prediction):
    print(f"Differential pixel accuracy - Input shape: {input.shape}, Target shape: {target.shape}, Prediction shape: {prediction.shape}")
    
    assert isinstance(input, torch.Tensor) and isinstance(target, torch.Tensor) and isinstance(prediction, torch.Tensor), "All inputs must be torch.Tensor"
    assert input.numel() == target.numel() == prediction.numel(), "Input, target, and prediction must have the same number of elements"

    input = input.view_as(target)
    prediction = prediction.view_as(target)
    
    print(f"Reshaped - Input: {input.shape}, Target: {target.shape}, Prediction: {prediction.shape}")

    input_target_diff = input != target
    correct_diff_predictions = (prediction == target) & input_target_diff

    total_diff_pixels = input_target_diff.sum().item()
    correct_diff_pixels = correct_diff_predictions.sum().item()

    print(f"Total different pixels: {total_diff_pixels}")
    print(f"Correctly predicted different pixels: {correct_diff_pixels}")

    if total_diff_pixels > 0:
        accuracy = correct_diff_pixels / total_diff_pixels
    else:
        accuracy = 1.0  # If no pixels differ, consider it 100% accurate

    print(f"Calculated accuracy: {accuracy}")
    return accuracy, input_target_diff, correct_diff_predictions

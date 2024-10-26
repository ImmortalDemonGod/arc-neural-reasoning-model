# gpt2_arc/src/utils/helpers.py
import time
import torch
import logging

logger = logging.getLogger(__name__)

from typing import Tuple, Dict

def calculate_mamba_efficiency(model: torch.nn.Module, input_data: torch.Tensor) -> Dict[str, float]:
    """
    Calculates performance metrics specific to Mamba layers in the model.

    Args:
        model: The GPT2ARC model instance.
        input_data: A sample input tensor.

    Returns:
        A dictionary containing Mamba-specific performance metrics.
    """
    metrics = {}
    model.eval()
    device = next(model.parameters()).device
    input_data = input_data.to(device)
    with torch.no_grad():
        start_time = time.time()
        _ = model(input_data)
        total_time = time.time() - start_time
    metrics['mamba_forward_pass_time'] = total_time
    mamba_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if 'mamba_block' in name:
            mamba_params += param_count
    metrics['mamba_params'] = mamba_params
    metrics['total_params'] = total_params
    metrics['mamba_params_ratio'] = mamba_params / total_params if total_params > 0 else 0
    return metrics


def differential_pixel_accuracy(input: torch.Tensor, target: torch.Tensor, prediction: torch.Tensor, pad_symbol_idx: int = 10) -> Tuple[float, torch.Tensor, torch.Tensor]:
    logger.debug(f"Differential pixel accuracy - Input shape: {input.shape}, Target shape: {target.shape}, Prediction shape: {prediction.shape}")
    
    assert isinstance(input, torch.Tensor) and isinstance(target, torch.Tensor) and isinstance(prediction, torch.Tensor), "All inputs must be torch.Tensor"
    assert input.numel() == target.numel() == prediction.numel(), "Input, target, and prediction must have the same number of elements"

    """
    Compute differential pixel accuracy, excluding padding tokens.

    Args:
        input (torch.Tensor): Input tensor for the model.
        target (torch.Tensor): Ground truth labels.
        prediction (torch.Tensor): Model predictions.
        pad_symbol_idx (int): Index of the padding token to exclude from calculations.

    Returns:
        tuple: A tuple containing:
            - accuracy (float): Differential pixel accuracy.
            - input_target_diff (torch.Tensor): Tensor indicating where input differs from target.
            - correct_diff_predictions (torch.Tensor): Tensor indicating correct predictions of differing pixels.
    """
    device = input.device
    target = target.to(device)
    prediction = prediction.to(device)
    prediction = prediction.view_as(target)
    
    logger.debug(f"Reshaped - Input: {input.shape}, Target: {target.shape}, Prediction: {prediction.shape}")

    # Exclude padding tokens by creating a valid mask
    valid_mask = target != pad_symbol_idx
    input_target_diff = (input != target) & valid_mask
    correct_diff_predictions = (prediction == target) & input_target_diff

    total_diff_pixels = input_target_diff.sum().item()
    correct_diff_pixels = correct_diff_predictions.sum().item()

    logger.debug(f"Total different pixels: {total_diff_pixels}")
    logger.debug(f"Correctly predicted different pixels: {correct_diff_pixels}")

    if total_diff_pixels > 0:
        accuracy = correct_diff_pixels / total_diff_pixels
    else:
        accuracy = 1.0  # If no pixels differ, consider it 100% accurate

    logger.debug(f"Calculated accuracy: {accuracy}")
    return accuracy, input_target_diff, correct_diff_predictions

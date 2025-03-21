import torch
import time

from typing import Dict

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
    model.eval()  # Set model to evaluation mode

    # Ensure model and input data are on the same device
    device = next(model.parameters()).device
    input_data = input_data.to(device)

    # Measure the forward pass time
    with torch.no_grad():
        start_time = time.time()
        _ = model(input_data)
        total_time = time.time() - start_time

    metrics['mamba_forward_pass_time'] = total_time

    # Count the number of parameters in Mamba layers
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

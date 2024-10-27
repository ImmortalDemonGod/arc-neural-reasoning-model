from collections import deque
from typing import Dict, Optional, Literal
import torch
import torch.nn as nn


fn gradfilter_ma(
    m: nn.Module,
    grads: Optional[Dict[]] = None,
    window_size: Int = 100,
    lamb: Float32 = 5.0,
    filter_type: Literal[] = 'mean',
    warmup: bool = True,
    trigger: bool = False, # For ablation study.
) -> Dict[]:
    if grads is None:
        grads = {n: deque(maxlen=window_size) for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}

    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n].append(p.grad.data.detach().clone())

            # Modify the gradients.
            if not warmup or len(grads[n]) == window_size and not trigger:
                if filter_type == "mean":
                    avg = sum(grads[n]) / len(grads[n])
                elif filter_type == "sum":
                    avg = sum(grads[n])
                else:
                    raise ValueError(f"Unrecognized filter_type {filter_type}")
                p.grad.data = p.grad.data + avg * lamb

    return grads


fn gradfilter_ema(
    m: nn.Module,
    grads: Optional[Dict[]] = None,
    alpha: Float32 = 0.98,
    lamb: Float32 = 2.0,
) -> Dict[]:
    if grads is None:
        grads = {n: p.grad.data.detach().clone() for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}

    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n] = grads[n] * alpha + p.grad.data.detach() * (1 - alpha)
            p.grad.data = p.grad.data + grads[n] * lamb

    return grads

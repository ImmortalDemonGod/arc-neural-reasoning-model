# gpt2_arc/src/data/utils/grid_ops.py
import torch
import torch.nn.functional as F
import numpy as np
from typing import Union
import logging

logger = logging.getLogger(__name__)

class GridOperations:
    """Handles grid preprocessing and transformation operations"""
    
    @staticmethod
    def preprocess_grid(grid: Union[list, np.ndarray, torch.Tensor], 
                       pad_symbol_idx: int,
                       target_height: int = 30,
                       target_width: int = 30) -> torch.Tensor:
        """Preprocesses a grid to the target dimensions"""
        try:
            # Convert to tensor if needed
            if isinstance(grid, list):
                grid_tensor = torch.as_tensor(grid, dtype=torch.float32)
            elif isinstance(grid, np.ndarray):
                grid_tensor = torch.as_tensor(grid, dtype=torch.float32)
            elif isinstance(grid, torch.Tensor):
                grid_tensor = grid.float()
            elif isinstance(grid, int):
                grid_tensor = torch.as_tensor([[grid]], dtype=torch.float32)
            else:
                raise ValueError(f"Unexpected grid type: {type(grid)}")

            # Ensure 3D shape [C, H, W]
            if grid_tensor.ndim == 2:
                grid_tensor = grid_tensor.unsqueeze(0)
            elif grid_tensor.ndim != 3:
                raise ValueError(f"Unexpected dimensions: {grid_tensor.ndim}")

            # Apply padding
            return GridOperations.pad_grid(
                grid_tensor, 
                target_height, 
                target_width, 
                pad_symbol_idx
            )

        except Exception as e:
            logger.error(f"Grid preprocessing failed: {str(e)}")
            raise

    @staticmethod
    def pad_grid(grid: torch.Tensor, 
                 height: int, 
                 width: int, 
                 pad_value: int) -> torch.Tensor:
        """Pads a grid to specified dimensions"""
        _, h, w = grid.shape
        pad_h = max((height - h) // 2, 0)
        pad_w = max((width - w) // 2, 0)
        
        padding = (
            pad_w,               # left
            width - w - pad_w,   # right
            pad_h,               # top
            height - h - pad_h   # bottom
        )
        
        return F.pad(grid, padding, mode='constant', value=pad_value)


    @staticmethod
    def kronecker_scale(X: np.ndarray, target_height: int = 30, target_width: int = 30) -> np.ndarray:
        """Scales a grid using Kronecker product"""
        h, w = X.shape
        scale_h = target_height / h
        scale_w = target_width / w
        d = int(np.floor(min(scale_h, scale_w)))
        return np.kron(X, np.ones((d, d)))

    @staticmethod
    def reverse_scaling(X_orig: np.ndarray, X_pred: np.ndarray) -> np.ndarray:
        """Reverses the scaling of a grid to match original dimensions"""
        h, w = X_orig.shape
        if X_pred.ndim == 1:
            X_pred = X_pred.reshape((int(np.sqrt(X_pred.size)), -1))
        
        X_pred_cropped = X_pred[:h, :w]
        
        if h == X_pred.shape[0] and w == X_pred.shape[1]:
            return X_pred_cropped
        
        d_h = X_pred_cropped.shape[0] // h
        d_w = X_pred_cropped.shape[1] // w
        
        if d_h > 0 and d_w > 0:
            X_rev = X_pred_cropped.reshape(h, d_h, w, d_w).mean(axis=(1, 3))
        else:
            raise ValueError("Invalid dimensions for reverse scaling")
            
        return np.resize(X_rev.round().astype(int), X_orig.shape)

    @staticmethod
    def scale_grid(grid: np.ndarray, height: int, width: int) -> np.ndarray:
        """Scales a grid to specified dimensions (currently preserves original size)"""
        return grid  # No scaling, preserve original size
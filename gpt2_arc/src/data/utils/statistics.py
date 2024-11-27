# gp2_arc/src/data/utils/statistics.py
from typing import Dict, Any, List
import numpy as np
import logging
from .custom_exceptions import ValidationError, DataLoadingError

logger = logging.getLogger(__name__)

class DatasetStatistics:
    """Handles computation and management of dataset statistics"""
    
    def __init__(self, num_symbols: int):
        # In DatasetStatistics.__init__, add:
        """
        Initialize statistics computer.

        Args:
            num_symbols (int): Number of unique symbols in the dataset
        """

        self.num_symbols = num_symbols
        self.statistics = {}
    
    def compute_all_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Computes all dataset statistics.

        Args:
            data (List[Dict[str, Any]]): List of dataset samples containing 'input' and 'output' tensors

        Returns:
            Dict[str, Any]: Dictionary containing:
                - grid_size_stats: Dictionary with max_height and max_width
                - symbol_frequencies: Array of symbol frequencies

        Raises:
            DataLoadingError: If data processing fails
            ValidationError: If data validation fails
        """
        try:
            grid_size_stats = self.compute_grid_size_stats(data)
            symbol_frequencies = self.compute_symbol_frequencies(data)
            
            self.statistics = {
                "grid_size_stats": grid_size_stats,
                "symbol_frequencies": symbol_frequencies
            }
            
            return self.statistics
        except ValidationError as e:
            logger.error(f"Validation error in statistics computation: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to compute statistics: {e}")
            raise DataLoadingError(f"Statistics computation failed: {str(e)}")

    def compute_grid_size_stats(self, data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Computes grid size statistics from dataset samples"""
        max_height, max_width = 0, 0
        for sample in data:
            max_height = max(max_height, sample["input"].shape[1], sample["output"].shape[1])
            max_width = max(max_width, sample["input"].shape[2], sample["output"].shape[2])
        
        return {"max_height": max_height, "max_width": max_width}

    def compute_symbol_frequencies(self, data: List[Dict[str, Any]]) -> np.ndarray:
        """
        Computes symbol frequency distribution.
        
        Args:
            data: List of dataset samples containing input and output tensors
            
        Returns:
            ndarray: Normalized frequency distribution of symbols
            
        Raises:
            ValidationError: If symbols exceed allowed range
            DataLoadingError: If computation fails
        """
        try:
            symbol_counts = np.zeros(self.num_symbols, dtype=int)
            max_symbol_in_data = 0
            
            for sample in data:
                input_symbols = sample["input"].flatten().numpy().astype(int)
                output_symbols = sample["output"].flatten().numpy().astype(int)
                
                if input_symbols.size > 0:
                    max_symbol_in_data = max(max_symbol_in_data, input_symbols.max())
                if output_symbols.size > 0:
                    max_symbol_in_data = max(max_symbol_in_data, output_symbols.max())
                    
                symbol_counts += np.bincount(input_symbols, minlength=self.num_symbols)
                symbol_counts += np.bincount(output_symbols, minlength=self.num_symbols)

            if max_symbol_in_data >= self.num_symbols:
                raise ValidationError(f"Symbol index {max_symbol_in_data} exceeds allowed range {self.num_symbols-1}")

            total_symbols = symbol_counts.sum()
            if total_symbols == 0:
                return np.zeros_like(symbol_counts, dtype=float)
                
            return symbol_counts / total_symbols
        except Exception as e:
            raise DataLoadingError(f"Failed to compute symbol frequencies: {str(e)}")

    def get_statistics(self) -> Dict[str, Any]:
        """Returns computed statistics"""
        return self.statistics
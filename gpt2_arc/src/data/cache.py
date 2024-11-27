# gpt2_arc/src/data/cache.py
import os
import json
import pickle
import hashlib
import logging
from typing import Optional, Tuple, List, Dict, Any, Union, TYPE_CHECKING
from .utils.custom_exceptions import ARCDatasetError
from .config.arc_dataset_config import ARCDatasetConfig

if TYPE_CHECKING:
    from arckit.data import TaskSet

logger = logging.getLogger(__name__)

class ARCCache:
    """Handles caching of ARC dataset and its statistics."""

    def __init__(self, cache_dir: str):
        """Initialize cache manager.
        
        Args:
            cache_dir (str): Directory for storing cache files
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.version = "v1"  # Cache version for compatibility tracking

    def generate_path(self, config: ARCDatasetConfig) -> str:
        """Generate cache file path based on dataset configuration.
        
        Args:
            config: Dataset configuration object
            
        Returns:
            str: Path to cache file
        """
        # Create stable string representation of data source
        if isinstance(config.data_source, str):
            data_source_str = os.path.abspath(config.data_source)
        elif hasattr(config.data_source, 'tasks'):  # TaskSet
            data_source_str = f"TaskSet:{len(config.data_source.tasks)}"
        elif isinstance(config.data_source, list):
            data_source_str = f"List:{len(config.data_source)}"
        else:
            data_source_str = str(config.data_source)

        # Create deterministic hash input
        hash_input = json.dumps({
            'version': self.version,
            'data_source': data_source_str,
            'num_symbols': config.num_symbols,
            'is_test': config.is_test,
            'test_split': config.test_split,
            'pad_symbol_idx': config.pad_symbol_idx
        }, sort_keys=True).encode('utf-8')

        # Generate filename from hash
        hash_digest = hashlib.md5(hash_input).hexdigest()
        cache_filename = f"arc_dataset_cache_{hash_digest}.pkl"
        
        return os.path.join(self.cache_dir, cache_filename)

    def save(self, config: ARCDatasetConfig, data: List[Dict], statistics: Dict) -> None:
        """Save dataset and statistics to cache.
        
        Args:
            config: Dataset configuration object
            data: List of dataset samples
            statistics: Dictionary of computed statistics
            
        Raises:
            ARCDatasetError: If saving fails
        """
        try:
            cache_data = {
                "version": self.version,
                "data": data,
                "statistics": statistics
            }
            
            temp_path = os.path.join(self.cache_dir, "temp_cache.pkl")
            final_path = self.generate_path(config)
            
            # Save to temporary file first
            with open(temp_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            # Validate saved data
            self._validate_cache(temp_path)
            
            # Move to final location
            os.replace(temp_path, final_path)
            
            logger.info(f"Successfully saved cache with {len(data)} samples")
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise ARCDatasetError(f"Failed to save cache: {str(e)}")

    def load(self, path: str) -> Optional[Tuple[List[Dict], Dict]]:
        """Load dataset and statistics from cache.
        
        Args:
            path: Path to cache file
            
        Returns:
            Optional[Tuple[List[Dict], Dict]]: (data, statistics) if successful, None otherwise
        """
        if not os.path.exists(path):
            return None
            
        try:
            with open(path, 'rb') as f:
                cache_data = pickle.load(f)
                
            # Version check
            if cache_data.get("version", "v0") != self.version:
                logger.warning(f"Cache version mismatch. Expected {self.version}, got {cache_data.get('version')}")
                return None
                
            # Validate content
            if not self._validate_cache_content(cache_data):
                logger.warning("Cache validation failed")
                return None
                
            return cache_data["data"], cache_data["statistics"]
            
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return None

    def _validate_cache(self, path: str) -> bool:
        """Validate a cache file.
        
        Args:
            path: Path to cache file
            
        Returns:
            bool: True if cache is valid
        """
        try:
            with open(path, 'rb') as f:
                cache_data = pickle.load(f)
            return self._validate_cache_content(cache_data)
        except Exception:
            return False

    def _validate_cache_content(self, cache_data: Dict) -> bool:
        """Validate cache content structure and data types.
        
        Args:
            cache_data: Loaded cache data
            
        Returns:
            bool: True if content is valid
        """
        if not isinstance(cache_data, dict):
            return False
            
        required_keys = {"version", "data", "statistics"}
        if not all(key in cache_data for key in required_keys):
            return False
            
        if not isinstance(cache_data["data"], list):
            return False
            
        if not isinstance(cache_data["statistics"], dict):
            return False
            
        return True
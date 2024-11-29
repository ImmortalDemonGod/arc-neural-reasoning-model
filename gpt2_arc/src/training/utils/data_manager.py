# gpt2_arc/src/training/utils/data_manager.py
import logging
from typing import Dict, List, Optional, Any, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm
import arckit
from gpt2_arc.src.data.arc_dataset import ARCDataset
from gpt2_arc.src.config import Config

logger = logging.getLogger(__name__)

class DataManager:
    """Manages all data-related operations for the ARC training process."""
    
    def __init__(self, config: Config, args: Any):
        """
        Initialize the DataManager.
        
        Args:
            config: Configuration object containing model and training settings
            args: Parsed command line arguments
        """
        self.config = config
        self.args = args
        self._validate_data_config()
        
    def _validate_data_config(self) -> None:
        """Validate data-related configuration settings."""
        if self.args.use_synthetic_data and not self.args.synthetic_data_path:
            raise ValueError("synthetic_data_path must be provided when use_synthetic_data is True")
            
        if self.args.max_train_samples is not None and self.args.max_train_samples <= 0:
            raise ValueError("max_train_samples must be greater than 0")
            
        total_split = self.args.train_split + self.args.val_split + self.args.test_split
        if not abs(total_split - 1.0) < 1e-6:
            raise ValueError("Train, validation, and test splits must sum to 1.0")

    def prepare_val_or_test_data(self, eval_set: Any, is_validation: bool = True) -> List[Dict[str, Any]]:
        """
        Prepare validation or test data from the arckit evaluation set.
        
        Args:
            eval_set: The evaluation TaskSet from arckit.load_data()
            is_validation: Boolean indicating whether it's validation data
            
        Returns:
            List of dictionaries with keys 'input', 'output', and 'task_id'
        """
        logger.debug(f"Preparing {'validation' if is_validation else 'test'} data")
        samples = []
        
        for task in tqdm(eval_set.tasks, 
                        desc=f"Processing tasks for {'validation' if is_validation else 'test'} dataset"):
            for ex in task.train if is_validation else task.test:
                sample = {
                    'input': ex[0],
                    'output': ex[1],
                    'task_id': task.id
                }
                samples.append(sample)
                
        logger.debug(f"Prepared {len(samples)} samples for {'validation' if is_validation else 'test'} dataset")
        return samples

    def load_dataset(self, dataset_type: str = 'train',
                    all_synthetic_data: Optional[Dict[str, ARCDataset]] = None) -> ARCDataset:
        """
        Load and prepare a dataset of the specified type.
        
        Args:
            dataset_type: Type of dataset to load ('train', 'val', or 'test')
            all_synthetic_data: Optional dictionary containing synthetic datasets
            
        Returns:
            ARCDataset object
        """
        logger.debug(f"Loading {dataset_type} dataset")
        
        if dataset_type.lower() == 'train' and self.args.use_synthetic_data:
            if all_synthetic_data is None:
                raise ValueError("all_synthetic_data must be provided when using synthetic data")
            dataset = all_synthetic_data['train_dataset']
        else:
            if dataset_type.lower() == 'train':
                data_source = arckit.load_data()[0]
                is_test = False
            else:
                _, eval_set = arckit.load_data()
                data_source = self.prepare_val_or_test_data(
                    eval_set,
                    is_validation=(dataset_type.lower() == 'val')
                )
                is_test = (dataset_type.lower() == 'test')
                
            dataset = ARCDataset(
                data_source=data_source,
                is_test=is_test,
                max_samples=self.args.max_train_samples if dataset_type.lower() == 'train' else None,
                num_symbols=self.config.training.num_symbols,
                pad_symbol_idx=self.config.training.pad_symbol_idx,
                symbol_freq=self.config.training.symbol_freq if self.args.enable_symbol_freq else None
            )
            
        if len(dataset) == 0:
            raise ValueError(f"No samples loaded for {dataset_type} dataset")
            
        logger.debug(f"Loaded {dataset_type} dataset with {len(dataset)} samples")
        return dataset

    def load_and_split_synthetic_data(self) -> Dict[str, ARCDataset]:
        """
        Load synthetic data using ARCDataset.
        
        Returns:
            Dictionary containing only 'train_dataset'
        """
        logger.debug("Loading synthetic data")
        
        synthetic_dataset = ARCDataset(
            data_source=self.args.synthetic_data_path,
            is_test=False,
            max_samples=self.args.max_train_samples,
            num_symbols=self.config.training.num_symbols,
            pad_symbol_idx=self.config.training.pad_symbol_idx,
            symbol_freq=self.config.training.symbol_freq if self.args.enable_symbol_freq else None
        )
        
        total_samples = len(synthetic_dataset)
        logger.debug(f"Loaded {total_samples} synthetic samples")
        
        if total_samples == 0:
            raise ValueError("No synthetic samples were loaded")
            
        return {'train_dataset': synthetic_dataset}

    def create_data_loaders(self, train_data: ARCDataset, val_data: ARCDataset,
                          test_data: ARCDataset) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create DataLoader objects for training, validation, and test datasets.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            test_data: Test dataset
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        logger.debug("Creating data loaders")
        
        loader_kwargs = {
            'batch_size': self.config.training.batch_size,
            'num_workers': self.config.training.num_workers,
            'pin_memory': self.config.training.pin_memory
        }
        
        if self.config.training.num_workers > 0:
            loader_kwargs.update({
                'persistent_workers': self.config.training.persistent_workers,
                'prefetch_factor': self.args.prefetch_factor
            })
            
        train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test_data, shuffle=False, **loader_kwargs)
        
        logger.debug("Data loaders created successfully")
        return train_loader, val_loader, test_loader

    def load_all_datasets(self) -> Tuple[ARCDataset, ARCDataset, ARCDataset]:
        """
        Load all datasets (train, validation, test) based on configuration.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        logger.info("Loading all datasets")
        
        # Load synthetic data if specified
        all_synthetic_data = None
        if self.args.use_synthetic_data:
            all_synthetic_data = self.load_and_split_synthetic_data()
            
        # Load datasets sequentially
        train_data = self.load_dataset('train', all_synthetic_data)
        val_data = self.load_dataset('val')
        test_data = self.load_dataset('test')
        
        # Log dataset information
        logger.info(f"Training dataset source: {'synthetic data' if self.args.use_synthetic_data else 'official ARC data'}")
        logger.info("Validation dataset source: official ARC data")
        logger.info("Test dataset source: official ARC data")
        
        logger.debug(f"Dataset sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data

    def calculate_symbol_frequencies(self, train_data: ARCDataset) -> Dict[int, float]:
        """
        Calculate symbol frequencies from training data if enabled.
        
        Args:
            train_data: Training dataset
            
        Returns:
            Dictionary mapping symbol indices to their frequencies
        """
        if not self.args.enable_symbol_freq:
            return {}
            
        logger.debug("Calculating symbol frequencies")
        symbol_freq = train_data.get_symbol_frequencies()
        return {i: float(freq) for i, freq in enumerate(symbol_freq)}